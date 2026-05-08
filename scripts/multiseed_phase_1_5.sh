#!/usr/bin/env bash
# Multi-seed Phase 1.5 unified continual-learning eval suite via the
# webapp launcher API.
#
# Sequentially launches N seeds via /api/runs/launch (so they don't
# contend on GPU), waits for each to finish, then aggregates with
# phase_1_5_aggregate.
#
# Defaults:
#   - preset: phase_1_5_unified_scaled (Tier 2.1 v4 arch for 8-word
#     benchmarks; default arch fails interference + long_tail per
#     2026-05-07 smoke result)
#   - seeds: 42 43 44 100 101 102 (Phase 1.4 6-seed protocol)
#   - webapp: http://127.0.0.1:8765
#
# Usage:
#   bash scripts/multiseed_phase_1_5.sh [preset] [seed1 seed2 ...]
#
# Examples:
#   bash scripts/multiseed_phase_1_5.sh                              # scaled, 6 default seeds
#   bash scripts/multiseed_phase_1_5.sh phase_1_5_unified            # default arch
#   bash scripts/multiseed_phase_1_5.sh phase_1_5_unified_scaled 42 43 44  # 3 seeds
#
# Per-seed wall clock at scaled arch: ~150-180 min (4 benchmarks
# sequentially). 6-seed total: ~16 hrs. 3-seed total: ~8 hrs.
#
# Outputs: per-seed JSON in research/findings/raw/g11_bg/g11_seed*_phase_1_5*.json
# Aggregate JSON + findings markdown if all seeds completed.

set -e

PRESET="${1:-phase_1_5_unified_scaled}"
shift || true
if [ "$#" -eq 0 ]; then
  SEEDS=(42 43 44 100 101 102)
else
  SEEDS=("$@")
fi

WEBAPP="${WEBAPP:-http://127.0.0.1:8765}"
TODAY="$(date +%Y-%m-%d)"
AGG_OUT="research/findings/raw/g11_bg/${PRESET}_aggregate_${TODAY}.json"
FINDINGS_MD="research/findings/${TODAY}-${PRESET}-multi-seed.md"

# Verify webapp is up
if ! curl -sf "${WEBAPP}/api/info" > /dev/null; then
  echo "ERROR: webapp not reachable at ${WEBAPP}"
  echo "Start it: python -m uvicorn webapp.server:app --host 0.0.0.0 --port 8765 --reload"
  exit 1
fi

echo "=== Phase 1.5 multi-seed launcher ==="
echo "  preset: ${PRESET}"
echo "  seeds:  ${SEEDS[@]}"
echo "  webapp: ${WEBAPP}"
echo

JSON_PATHS=()
for SEED in "${SEEDS[@]}"; do
  echo "--- seed ${SEED} ---"

  RESP=$(curl -s -X POST "${WEBAPP}/api/runs/launch" \
    -H "Content-Type: application/json" \
    -d "{\"preset\":\"${PRESET}\",\"seed\":${SEED},\"extra_args\":[]}")
  RUN_ID=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('run_id',''))")
  OUT_PATH=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('out_path',''))")

  if [ -z "$RUN_ID" ]; then
    echo "  FAIL: launch returned no run_id"
    echo "  Response: $RESP"
    exit 1
  fi

  echo "  run_id: ${RUN_ID}"
  echo "  out:    ${OUT_PATH}"

  # Poll until finished (Phase 1.5 takes longer than chat demos)
  while true; do
    sleep 30
    STATUS=$(curl -s "${WEBAPP}/api/runs/launch/${RUN_ID}" \
      | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('running'), d.get('returncode'), int(d.get('elapsed_sec',0)/60))")
    RUNNING=$(echo "$STATUS" | awk '{print $1}')
    RC=$(echo "$STATUS" | awk '{print $2}')
    MIN=$(echo "$STATUS" | awk '{print $3}')
    if [ "$RUNNING" = "False" ]; then
      echo "  done (rc=${RC}, elapsed ~${MIN} min)"
      if [ "$RC" != "0" ]; then
        echo "  WARN: non-zero exit, skipping aggregate for this seed"
      else
        JSON_PATHS+=("$OUT_PATH")
      fi
      break
    fi
  done
done

echo
echo "=== aggregating ${#JSON_PATHS[@]} seed JSONs ==="
if [ "${#JSON_PATHS[@]}" -eq 0 ]; then
  echo "FAIL: no seeds completed successfully, skipping aggregate"
  exit 1
fi

python -m research.runners.phase_1_5_aggregate "${JSON_PATHS[@]}" \
  --out "$AGG_OUT" \
  --findings-md "$FINDINGS_MD" \
  --label "${PRESET} multi-seed (${#JSON_PATHS[@]} seeds)"

echo
echo "=== done ==="
echo "  aggregate: ${AGG_OUT}"
echo "  findings:  ${FINDINGS_MD}"
