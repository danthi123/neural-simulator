#!/usr/bin/env bash
# Path A overnight chain — autonomous arc continuation post 2026-05-09.
#
# Sequence:
#   1. Wait for current chat_speak_synonym_demo seed 42 smoke to complete.
#   2. If smoke GO (any-synonym A2W >= 50%): launch 6-seed multi-seed.
#      If NO-GO: skip multi-seed, log decision, proceed to step 3.
#   3. Launch 16-word smoke (capacity rule extension probe, ~35 min/seed).
#   4. Aggregate + document each step.
#
# Total ETA: ~2-3 hrs (multi-seed ~80 min + 16-word smoke ~35 min +
# overhead) if everything PASSES. Less if the chat_speak_synonym smoke
# is NO-GO (we skip multi-seed).
#
# Usage:
#   nohup bash scripts/chain_path_a_overnight.sh > chain_path_a.log 2>&1 &
#
# Resume-safe: if you kill+rerun, it'll wait for the in-flight run to
# complete and then proceed from where it was.

set -e
WEBAPP="${WEBAPP:-http://127.0.0.1:8765}"
TODAY="$(date +%Y-%m-%d)"

if ! curl -sf "${WEBAPP}/api/info" > /dev/null; then
  echo "ERROR: webapp not reachable at ${WEBAPP}"
  exit 1
fi

echo "=== chain_path_a_overnight start $(date +%Y-%m-%dT%H:%M:%S) ==="

# ─── Step 1: wait for any in-flight runs ─────────────────────────────
echo "[step 1] waiting for in-flight runs to complete..."
while true; do
  RUNNING=$(curl -s "${WEBAPP}/api/runs/launch" \
    | python -c "
import json,sys
d=json.loads(sys.stdin.read())
print(sum(1 for r in d.get('runs',[]) if r.get('running')))
")
  if [ "$RUNNING" = "0" ]; then break; fi
  echo "[wait] $(date +%H:%M:%S)  ${RUNNING} active, sleeping 60s..."
  sleep 60
done
echo "[step 1] GPU idle, all runs complete."

# Look up most recent chat_speak_synonym_demo seed 42 result
SMOKE_JSON=$(ls -t research/findings/raw/g11_bg/g11_seed42_chat_speak_synonym_demo_*.json 2>/dev/null \
  | grep -v cmd.json | head -1)
if [ -z "$SMOKE_JSON" ]; then
  echo "ERROR: no chat_speak_synonym_demo seed 42 result found; abort."
  exit 1
fi

# ─── Step 2: smoke verdict + maybe multi-seed ────────────────────────
echo "[step 2] reading smoke verdict from $SMOKE_JSON"
SMOKE_GO=$(python -c "
import json
d=json.load(open(r'$SMOKE_JSON'))
go = d.get('go', False)
a2w = d.get('speak_accuracy', 0.0)
print('GO' if (go or a2w >= 0.50) else 'NO-GO')
")
A2W_PCT=$(python -c "
import json
d=json.load(open(r'$SMOKE_JSON'))
print(f\"{d.get('speak_accuracy', 0.0):.1%}\")
")
echo "[step 2] smoke: $SMOKE_GO (any-synonym A2W $A2W_PCT)"

if [ "$SMOKE_GO" = "GO" ]; then
  echo "[step 2] smoke GO -> launching 6-seed multi-seed..."
  bash scripts/multiseed_chat_speak_synonym_demo.sh
  echo "[step 2] multi-seed complete."
else
  echo "[step 2] smoke NO-GO -> skipping multi-seed (would just confirm failure)."
  echo "[step 2] documented in chain log; will investigate separately."
fi

# ─── Step 3: 16-word smoke (capacity rule extension) ─────────────────
echo "[step 3] launching 16-word smoke..."
RESP=$(curl -s -X POST "${WEBAPP}/api/runs/launch" \
  -H "Content-Type: application/json" \
  -d '{"preset":"consolidation_synonym_16word_scaled_smoke","seed":42,"extra_args":[]}')
RUN_ID=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('run_id',''))")
OUT=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('out_path',''))")
if [ -z "$RUN_ID" ]; then echo "FAIL launch: $RESP"; exit 1; fi
echo "[step 3] rid=$RUN_ID  out=...$(basename $OUT)"
while true; do
  sleep 60
  RUNNING=$(curl -s "${WEBAPP}/api/runs/launch/${RUN_ID}" \
    | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('running'))")
  [ "$RUNNING" = "False" ] && break
done
echo "[step 3] 16-word smoke complete."

# Read the 16-word smoke result
RES_16W=$(python -c "
import json
d=json.load(open(r'$OUT'))
ret = d.get('retention', {})
verdict = d.get('verdict', '?')
prim = ret.get('primary', 0.0) if isinstance(ret, dict) else 0.0
syn  = ret.get('synonym', 0.0) if isinstance(ret, dict) else 0.0
print(f'verdict={verdict}  primary={prim:.0%}  synonym={syn:.0%}')
")
echo "[step 3] 16-word smoke result: $RES_16W"

echo "=== chain_path_a_overnight done $(date +%Y-%m-%dT%H:%M:%S) ==="
echo ""
echo "Next manual decision:"
echo "  - If 16-word smoke primary >= 50%: launch consolidation_synonym_16word_scaled_medium"
echo "  - If 16-word smoke primary < 50%: capacity boundary at 4 sub-pops/motor_X,"
echo "    next experiment is scale further (n_motor=3000) or pivot to Tier 2.3 phrases"
