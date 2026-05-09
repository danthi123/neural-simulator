#!/usr/bin/env bash
# Auto-launch v400 interference confirmation when Phase 1.5 multi-seed completes.
#
# Background: 2026-05-09 Phase 1.5 multi-seed at scaled arch hit a partial
# pattern — 3/4 benchmarks pass cleanly but `interference` scores 0.34
# (below 0.5 threshold) at events_per_word=200. Validated Tier 2.1 used
# 400 events/word. Hypothesis: interference is under-trained.
#
# This chain waits for the current Phase 1.5 batch to finish (whether via
# the multiseed_phase_1_5.sh wrapper OR a manual launch), then auto-fires
# the phase_1_5_interference_only_v400 preset on a single seed for fast
# (~70 min) hypothesis confirmation. Result: if score lifts to >=0.5, we
# know the events/word lever works; if it stays below, we know it's an
# architectural issue and need a different fix.
#
# Defaults:
#   - seed: 42 (matches the failing seed for cleanest before/after)
#   - webapp: http://127.0.0.1:8765
#   - poll interval: 60s
#
# Usage:
#   bash scripts/chain_phase_1_5_to_v400_interference.sh [seed]
#   bash scripts/chain_phase_1_5_to_v400_interference.sh 42
#
# Outputs:
#   - per-seed JSON: research/findings/raw/g11_bg/g11_seed*_phase_1_5_interference_only_v400_*.json
#   - findings markdown: research/findings/YYYY-MM-DD-phase-1-5-v400-interference-confirmation.md

set -e

SEED="${1:-42}"
WEBAPP="${WEBAPP:-http://127.0.0.1:8765}"
TODAY="$(date +%Y-%m-%d)"
FINDINGS_MD="research/findings/${TODAY}-phase-1-5-v400-interference-confirmation.md"

# Verify webapp reachable
if ! curl -sf "${WEBAPP}/api/info" > /dev/null; then
  echo "ERROR: webapp not reachable at ${WEBAPP}"
  exit 1
fi

echo "=== chain_phase_1_5_to_v400_interference ==="
echo "  seed:   ${SEED}"
echo "  webapp: ${WEBAPP}"
echo "  Will wait for any active /api/runs/launch run to complete,"
echo "  then auto-fire phase_1_5_interference_only_v400 at seed ${SEED}."
echo

# Step 1: poll until no runs are running. Use /api/runs/launch which
# tracks webapp-launched runs. Don't filter by run_id — we wait for ALL
# active runs to drain (covers the multi-seed wrapper + any manual
# launches the user kicked off).
echo "[wait] polling /api/runs/launch every 60s for active runs..."
while true; do
  RUNNING=$(curl -s "${WEBAPP}/api/runs/launch" \
    | python -c "
import json,sys
d = json.loads(sys.stdin.read())
running = [r for r in d.get('runs', []) if r.get('running')]
print(len(running))
")
  if [ "$RUNNING" = "0" ]; then
    echo "[wait] no active runs — proceeding to launch."
    break
  fi
  echo "[wait] $(date +%H:%M:%S)  ${RUNNING} runs still active, sleeping 60s..."
  sleep 60
done

# Step 2: launch the v400 interference confirmation
echo
echo "[launch] phase_1_5_interference_only_v400 seed ${SEED}"
RESP=$(curl -s -X POST "${WEBAPP}/api/runs/launch" \
  -H "Content-Type: application/json" \
  -d "{\"preset\":\"phase_1_5_interference_only_v400\",\"seed\":${SEED},\"extra_args\":[]}")
RUN_ID=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('run_id',''))")
OUT_PATH=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('out_path',''))")

if [ -z "$RUN_ID" ]; then
  echo "FAIL: launch returned no run_id"
  echo "Response: $RESP"
  exit 1
fi

echo "[launch] run_id: ${RUN_ID}"
echo "[launch] out:    ${OUT_PATH}"

# Step 3: poll until done
echo "[run] polling for completion every 60s..."
while true; do
  sleep 60
  STATUS=$(curl -s "${WEBAPP}/api/runs/launch/${RUN_ID}" \
    | python -c "
import json,sys
d = json.loads(sys.stdin.read())
print(d.get('running'), d.get('returncode'), int(d.get('elapsed_sec',0)/60))
")
  RUNNING=$(echo "$STATUS" | awk '{print $1}')
  RC=$(echo "$STATUS" | awk '{print $2}')
  MIN=$(echo "$STATUS" | awk '{print $3}')
  if [ "$RUNNING" = "False" ]; then
    echo "[run] done (rc=${RC}, elapsed ~${MIN} min)"
    break
  fi
  echo "[run] $(date +%H:%M:%S)  elapsed ~${MIN} min, still running..."
done

# Step 4: extract interference score from result JSON + write findings
echo
if [ ! -f "$OUT_PATH" ]; then
  echo "WARN: result JSON not found at $OUT_PATH"
  exit 1
fi

python -c "
import json, sys
data = json.load(open(r'${OUT_PATH}'))
benchmarks = data.get('benchmarks', [])
inter = next((b for b in benchmarks if b.get('name') == 'interference'), None)
if inter is None:
    print('FAIL: no interference benchmark in result')
    sys.exit(1)
print(f'\n=== INTERFERENCE @ 400 events/word, seed ${SEED} ===')
print(f'  score:    {inter[\"score\"]:.3f}')
print(f'  pass:     {inter[\"pass\"]}')
print(f'  details:  {inter.get(\"details\", {})}')
print()
# Decide verdict for the hypothesis
prev = 0.34  # seed 42 at events_per_word=200
new = inter['score']
delta = new - prev
verdict = 'CONFIRMED' if new >= 0.5 else 'REFUTED'
print(f'=== HYPOTHESIS {verdict} ===')
print(f'  prev (200 events): {prev:.3f}  new (400 events): {new:.3f}')
print(f'  delta: {delta:+.3f} pp')
print(f'  threshold (>=0.5): {\"PASS\" if new >= 0.5 else \"FAIL\"}')

# Write findings doc
md = '''# ${TODAY} — Phase 1.5 v400 interference confirmation (single-seed)

Auto-fired by scripts/chain_phase_1_5_to_v400_interference.sh after the
prior Phase 1.5 batch completed. Tests whether raising
events_per_word from 200 to 400 lifts the interference benchmark above
the 0.5 pass threshold (per the 2026-05-09 under-training hypothesis).

## Headline

| seed | events/word | score | pass | delta vs 200ev |
|---|---|---|---|---|
| ${SEED} | 400 | ''' + f'{new:.3f}' + ''' | ''' + str(inter['pass']) + ''' | ''' + f'{delta:+.3f}' + ''' |

## Verdict

Hypothesis: ''' + verdict + '''

''' + ('Raising events_per_word from 200 to 400 lifts interference '
       'above the 0.5 threshold. The under-training hypothesis is '
       'confirmed at single seed. Next: re-run the full Phase 1.5 '
       'multi-seed at events_per_word=400 (preset '
       'phase_1_5_unified_scaled_v400) to confirm at multi-seed.'
       if new >= 0.5 else
       'Raising events_per_word from 200 to 400 does NOT lift '
       'interference above the 0.5 threshold. The under-training '
       'hypothesis is refuted; interference failure is structural '
       '(architecture or rule-level), not a tuning issue. Next: '
       'investigate architectural causes — n_motor_per_action capacity, '
       'embodied-Hebbian rule limits on interleaved binding, or '
       'whether interleaved training fundamentally needs a different '
       'curriculum.') + '''

## Per-word accuracy

''' + ' '.join(f'{w}={a:.2f}' for w, a in (inter.get('details', {}).get('per_word_acc', {})).items()) + '''

## Related

- 2026-05-09-Phase-1.5-interference-undertraining-hypothesis.md (the predicted experiment)
- 2026-05-09-Phase1.3-Tier2.1-12word-scaled-3seed-CONFIRMED.md (capacity rule)
- 2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md (validated 400 events/word config)
'''
import os
out_path = os.path.expanduser(r'${FINDINGS_MD}')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(md)
print(f'\nFindings doc: {out_path}')
"

echo
echo "=== chain complete ==="
echo "  result JSON:   ${OUT_PATH}"
echo "  findings MD:   ${FINDINGS_MD}"
