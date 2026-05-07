#!/usr/bin/env bash
# Multi-seed chat demo runner using the webapp launcher API.
#
# Sequentially launches N seeds via /api/runs/launch (so they don't
# contend on GPU), waits for each to finish, then aggregates with
# chat_demo_aggregate.
#
# Defaults:
#   - preset: chat_demo (Tier 1 4-word, ~6 min/seed)
#   - seeds: 42 43 44 100 101 102 (matches Phase 1.4 6-seed protocol)
#   - webapp: http://127.0.0.1:8765
#
# Usage:
#   bash scripts/multiseed_chat_demo.sh [preset] [seed1 seed2 ...]
#
# Examples:
#   bash scripts/multiseed_chat_demo.sh                       # chat_demo, 6 default seeds
#   bash scripts/multiseed_chat_demo.sh chat_synonym_demo     # synonym demo, default seeds
#   bash scripts/multiseed_chat_demo.sh chat_demo 42 43 44    # tier 1, 3 seeds
#
# Outputs: per-seed JSON in research/findings/raw/g11_bg/g11_seed*.json,
# plus aggregate JSON + findings markdown if all seeds completed.

set -e

PRESET="${1:-chat_demo}"
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

echo "=== multi-seed chat demo launcher ==="
echo "  preset: ${PRESET}"
echo "  seeds:  ${SEEDS[@]}"
echo "  webapp: ${WEBAPP}"
echo

JSON_PATHS=()
for SEED in "${SEEDS[@]}"; do
  echo "--- seed ${SEED} ---"

  # Launch
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

  # Poll until finished
  while true; do
    sleep 15
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

python -m research.runners.chat_demo_aggregate "${JSON_PATHS[@]}" \
  --out "$AGG_OUT" \
  --findings-md "$FINDINGS_MD" \
  --label "${PRESET} multi-seed (${#JSON_PATHS[@]} seeds)"

echo
echo "=== done ==="
echo "  aggregate: ${AGG_OUT}"
echo "  findings:  ${FINDINGS_MD}"
