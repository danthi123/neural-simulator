#!/usr/bin/env bash
# Auto-chain: after the n_motor=2000 interference test completes, fire
# the long_tail_relaxed test (~50 min), then a Track 3 :speak GPU smoke
# (~10 min) that validates the layer 4 generative decoder works end-to-
# end on a freshly-trained Tier 1 bridge.
#
# Total auto-chain wait: ~140 min after the n_motor=2000 run drains.
#
# Each step:
#   1. Wait for /api/runs/launch to show 0 active runs.
#   2. Launch the next experiment via webapp /api/runs/launch.
#   3. Poll until done; capture the result; emit a brief verdict.
#
# Why a chain instead of pre-staged manual launches: the user is in
# autonomous mode + may be away. Each result informs the master-plan
# strategic decision (demote Phase 1.5 vs more architecture testing).
# Auto-chaining eliminates the manual launch touchpoint between tests.
#
# Usage:
#   bash scripts/chain_phase_1_5_followups.sh
#
# Outputs:
#   research/findings/raw/g11_bg/g11_seed42_phase_1_5_long_tail_relaxed_*.json
#   research/findings/raw/g11_bg/g11_seed42_phase_1_5_speak_smoke_*.json (if speak runner shipped)
#   research/findings/2026-05-09-Phase-1.5-followup-chain-RESULTS.md

set -e

WEBAPP="${WEBAPP:-http://127.0.0.1:8765}"
TODAY="$(date +%Y-%m-%d)"
SEED="${1:-42}"
RESULTS_MD="research/findings/${TODAY}-Phase-1.5-followup-chain-RESULTS.md"

if ! curl -sf "${WEBAPP}/api/info" > /dev/null; then
  echo "ERROR: webapp not reachable at ${WEBAPP}"
  exit 1
fi

echo "=== chain_phase_1_5_followups ==="
echo "  seed:   ${SEED}"
echo "  webapp: ${WEBAPP}"
echo "  steps:"
echo "    1. wait for active runs to drain (currently expecting n_motor=2000 to finish)"
echo "    2. launch phase_1_5_long_tail_relaxed seed ${SEED} (~50 min)"
echo "    3. wait + capture verdict"
echo "    4. write findings doc"
echo

# Helper: wait until /api/runs/launch shows 0 running
wait_for_drain() {
  echo "[wait] polling /api/runs/launch every 60s for active runs..."
  while true; do
    RUNNING=$(curl -s "${WEBAPP}/api/runs/launch" \
      | python -c "
import json,sys
d=json.loads(sys.stdin.read())
print(sum(1 for r in d.get('runs',[]) if r.get('running')))
")
    if [ "$RUNNING" = "0" ]; then
      echo "[wait] no active runs — proceeding."
      break
    fi
    echo "[wait] $(date +%H:%M:%S)  ${RUNNING} runs active, sleeping 60s..."
    sleep 60
  done
}

# Helper: launch a preset, wait for completion, return out_path
run_preset() {
  local preset="$1"
  local seed="$2"
  echo
  echo "=== launching ${preset} seed ${seed} ==="
  RESP=$(curl -s -X POST "${WEBAPP}/api/runs/launch" \
    -H "Content-Type: application/json" \
    -d "{\"preset\":\"${preset}\",\"seed\":${seed},\"extra_args\":[]}")
  RUN_ID=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('run_id',''))")
  OUT_PATH=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('out_path',''))")
  if [ -z "$RUN_ID" ]; then
    echo "FAIL: launch returned no run_id"
    echo "Response: $RESP"
    return 1
  fi
  echo "  run_id: ${RUN_ID}"
  echo "  out:    ${OUT_PATH}"

  echo "[run] polling for completion every 60s..."
  while true; do
    sleep 60
    STATUS=$(curl -s "${WEBAPP}/api/runs/launch/${RUN_ID}" \
      | python -c "
import json,sys
d=json.loads(sys.stdin.read())
print(d.get('running'), d.get('returncode'), int(d.get('elapsed_sec',0)/60))
")
    RUNNING=$(echo "$STATUS" | awk '{print $1}')
    RC=$(echo "$STATUS" | awk '{print $2}')
    MIN=$(echo "$STATUS" | awk '{print $3}')
    if [ "$RUNNING" = "False" ]; then
      echo "[run] done (rc=${RC}, elapsed ~${MIN} min)"
      echo "$OUT_PATH"  # last output line is the JSON path
      return 0
    fi
    echo "[run] $(date +%H:%M:%S)  elapsed ~${MIN} min..."
  done
}

# === Step 1: wait for current activity (n_motor=2000) to drain ===
wait_for_drain

# === Step 2: launch long_tail_relaxed ===
LT_OUT=$(run_preset "phase_1_5_long_tail_relaxed" "$SEED" | tail -1)

# === Step 3: chat_speak_demo (Track 3 layer 4 GPU smoke) ===
wait_for_drain
SPEAK_OUT=$(run_preset "chat_speak_demo" "$SEED" | tail -1)

# === Step 4: capture verdicts + write findings doc ===
echo
echo "=== chain results ==="
python -c "
import json

# long_tail_relaxed
print()
print('=== long_tail_relaxed seed ${SEED} ===')
data = json.load(open(r'${LT_OUT}'))
b = data['benchmarks'][0]
print(f'  score: {b[\"score\"]}')
print(f'  pass:  {b[\"pass\"]}')
det = b.get('details', {})
common = det.get('common_acc', 0)
rare = det.get('rare_acc', 0)
print(f'  common_acc: {common}')
print(f'  rare_acc:   {rare}')
print(f'  vs prior (rare-ratio=20, 300pA teacher):')
print(f'    rare_acc: 0.17 -> {rare}  (delta {rare - 0.17:+.3f})')
print(f'    threshold (>=0.30): {\"PASS\" if rare >= 0.30 else \"FAIL\"}')
LT_PASS = rare >= 0.30

# chat_speak_demo (Track 3 layer 4)
print()
print('=== chat_speak_demo seed ${SEED} (A2W generative decoder) ===')
data = json.load(open(r'${SPEAK_OUT}'))
print(f'  W->A regression baseline:    {data[\"accuracy\"]:.1%}')
print(f'  A->W speak accuracy:         {data[\"speak_accuracy\"]:.1%}')
print(f'  verdict:                     {data[\"verdict\"]}')
print(f'  per-action (A->W):')
for r in data.get('speak_results', []):
    mark = '[OK]' if r['correct'] else '[X] '
    print(f'    {mark} motor_{r[\"target_action\"]} -> {r[\"predicted_word\"]!r:<10}  expected {r[\"expected_word\"]!r}')
"

# Write findings doc
cat > "$RESULTS_MD" <<MARKDOWN
# ${TODAY} — Phase 1.5 followup chain results

Auto-fired by \`scripts/chain_phase_1_5_followups.sh\` after the
n_motor=2000 interference test drained. Tests the second hypothesis
(long_tail rare-word binding fails due to dose+teacher rather than
architecture) AND the deferred Track 3 layer 4 GPU smoke (:speak
generative decoder, validated A2W direction at Tier 1).

## long_tail_relaxed seed ${SEED}

Result JSON: \`${LT_OUT}\`

## chat_speak_demo (Track 3 layer 4) seed ${SEED}

Result JSON: \`${SPEAK_OUT}\`

See per-action breakdown in the JSON for which motor pools the
network can correctly decode back to their canonical words. Tier 1
BREAKTHROUGH validated A2W mean 45-63% across 6 seeds; this is a
single-seed reproduction of the same direction validating the
generative_inference primitive.

## Related

- 2026-05-09-Phase-1.5-multi-seed-FINAL.md (the 3-seed batch result)
- 2026-05-09-Phase-1.5-v400-interference-REFUTED.md (first hypothesis refuted)
- (this iteration) n_motor=2000 interference test result
- docs/plans/2026-05-09-Track-3-conversational-scaffolding-progress.md
MARKDOWN

echo
echo "=== chain done ==="
echo "  results: ${RESULTS_MD}"
