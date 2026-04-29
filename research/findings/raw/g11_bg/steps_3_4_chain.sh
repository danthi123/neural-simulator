#!/usr/bin/env bash
# Chains Step 3 (A+E + sensed-reward, n=6) and Step 4 (A+E multi-goal
# deterministic, n=6 baseline + n=6 A+E = 12 runs) after tier-4 finishes.
# All launches via the webapp API so they appear in the Live picker.
set -e
cd "$(dirname "$0")/../../.."

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

# Wait for tier-4 to complete (all 12 expected JSON files exist).
echo "[chain] waiting for tier-4 (12 outputs) to finish..."
while true; do
  done_count=0
  for SEED in 200 201 202 300 301 302; do
    for LABEL in baseline AE; do
      [ -f "research/findings/raw/g11_bg/g11_seed${SEED}_tier4_det_${LABEL}.json" ] && done_count=$((done_count + 1))
    done
  done
  if [ "$done_count" -ge 12 ]; then break; fi
  sleep 60
done
echo "[chain] tier-4 done at $(date +%H:%M:%S); kicking off Step 3 (A+E + sensed)"

# Step 3: A+E + --sensed-reward, n=6 (seeds 42-44, 100-102)
SENSED_EXTRAS='["--bg-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-fsis","--enable-cluster-a-closed-loop","--enable-cluster-e-topography","--sensed-reward","--deterministic"]'
for SEED in 42 43 44 100 101 102; do
  echo "  [step3 seed $SEED A+E+sensed] launching..."
  RID=$(post_run "$SEED" "$SENSED_EXTRAS" "g11_seed${SEED}_step3_AE_sensed.json")
  echo "    run_id=$RID"
  sleep 1
done

# Wait for Step 3 to finish (6 outputs).
echo "[chain] waiting for Step 3 (6 outputs)..."
while true; do
  done_count=0
  for SEED in 42 43 44 100 101 102; do
    [ -f "research/findings/raw/g11_bg/g11_seed${SEED}_step3_AE_sensed.json" ] && done_count=$((done_count + 1))
  done
  if [ "$done_count" -ge 6 ]; then break; fi
  sleep 60
done
echo "[chain] Step 3 done at $(date +%H:%M:%S); kicking off Step 4 (A+E multi-goal det)"

# Step 4: multi-goal deterministic, n=6 baseline + n=6 A+E
MULTI_BASE_EXTRAS='["--bg-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-fsis","--goal-schedule","multi","--deterministic"]'
MULTI_AE_EXTRAS='["--bg-lateral-inhibition","--enable-d1-d2-asymmetry","--enable-striatal-fsis","--enable-cluster-a-closed-loop","--enable-cluster-e-topography","--goal-schedule","multi","--deterministic"]'
for SEED in 42 43 44 100 101 102; do
  for LABEL in baseline AE; do
    if [ "$LABEL" = "AE" ]; then EXTRAS=$MULTI_AE_EXTRAS; else EXTRAS=$MULTI_BASE_EXTRAS; fi
    OUT="g11_seed${SEED}_step4_multidet_${LABEL}.json"
    echo "  [step4 seed $SEED $LABEL] launching..."
    RID=$(post_run "$SEED" "$EXTRAS" "$OUT")
    echo "    run_id=$RID"
    sleep 1
  done
done

# Wait for Step 4 to finish (12 outputs).
echo "[chain] waiting for Step 4 (12 outputs)..."
while true; do
  done_count=0
  for SEED in 42 43 44 100 101 102; do
    for LABEL in baseline AE; do
      [ -f "research/findings/raw/g11_bg/g11_seed${SEED}_step4_multidet_${LABEL}.json" ] && done_count=$((done_count + 1))
    done
  done
  if [ "$done_count" -ge 12 ]; then break; fi
  sleep 60
done
echo "[chain] ALL DONE at $(date +%H:%M:%S)"
