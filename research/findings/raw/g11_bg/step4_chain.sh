#!/usr/bin/env bash
# Step 4: A+E multi-goal deterministic (n=6 baseline + n=6 A+E = 12 runs).
# Fires after Step 3 (6 outputs) completes.
# All launches via webapp API → visible in Live mode.
set -e
# Use ABSOLUTE path to avoid the cd-bug in steps_3_4_chain.sh.
cd /e/Documents/Projects/sim

API="http://localhost:8765/api/runs/launch"
post_run() {
  local seed="$1" extras_json="$2" outname="$3"
  local body
  body=$(python -c "
import json, sys
print(json.dumps({
  'preset': 'baseline',
  'seed': int(sys.argv[1]),
  'extra_args': json.loads(sys.argv[2]),
  'out_filename': sys.argv[3],
}))
" "$seed" "$extras_json" "$outname")
  curl -s -X POST "$API" -H "Content-Type: application/json" -d "$body" \
    | python -c "import json, sys; print(json.load(sys.stdin).get('run_id', '?'))"
}

echo "[step4] waiting for Step 3 (6 outputs) to finish..."
while true; do
  done_count=0
  for SEED in 42 43 44 100 101 102; do
    [ -f "/e/Documents/Projects/sim/research/findings/raw/g11_bg/g11_seed${SEED}_step3_AE_sensed.json" ] && done_count=$((done_count + 1))
  done
  if [ "$done_count" -ge 6 ]; then break; fi
  sleep 60
done
echo "[step4] Step 3 done at $(date +%H:%M:%S); kicking off Step 4"

MULTI_BASE='["--bg-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-fsis","--goal-schedule","multi","--deterministic"]'
MULTI_AE='["--bg-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-fsis","--enable-cluster-a-closed-loop","--enable-cluster-e-topography","--goal-schedule","multi","--deterministic"]'
for SEED in 42 43 44 100 101 102; do
  for LABEL in baseline AE; do
    if [ "$LABEL" = "AE" ]; then EXTRAS=$MULTI_AE; else EXTRAS=$MULTI_BASE; fi
    OUT="g11_seed${SEED}_step4_multidet_${LABEL}.json"
    RID=$(post_run "$SEED" "$EXTRAS" "$OUT")
    echo "  step4 seed=$SEED $LABEL → run_id=$RID"
    sleep 1
  done
done
echo "[step4] all 12 launches submitted."

# Wait for Step 4 to finish (12 outputs)
echo "[step4] waiting for 12 outputs..."
while true; do
  done_count=0
  for SEED in 42 43 44 100 101 102; do
    for LABEL in baseline AE; do
      [ -f "/e/Documents/Projects/sim/research/findings/raw/g11_bg/g11_seed${SEED}_step4_multidet_${LABEL}.json" ] && done_count=$((done_count + 1))
    done
  done
  if [ "$done_count" -ge 12 ]; then break; fi
  sleep 60
done
echo "[step4] ALL DONE at $(date +%H:%M:%S)"
