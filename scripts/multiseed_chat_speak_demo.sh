#!/usr/bin/env bash
# Track 3 v2 — multi-seed chat_speak_demo (validates 75% A2W reproduces)
# Usage: bash scripts/multiseed_chat_speak_demo.sh [seed1 seed2 ...]
# Default 6-seed protocol: 42 43 44 100 101 102

set -e
SEEDS="${@:-42 43 44 100 101 102}"
WEBAPP="${WEBAPP:-http://127.0.0.1:8765}"
TODAY="$(date +%Y-%m-%d)"

if ! curl -sf "${WEBAPP}/api/info" > /dev/null; then
  echo "ERROR: webapp not reachable at ${WEBAPP}"
  exit 1
fi

echo "=== multiseed_chat_speak_demo ==="
echo "  seeds: $SEEDS"

# Wait for any active GPU runs first (e.g. Phase 2.2b)
echo "[wait] for GPU to free up..."
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

JSON_PATHS=()
for SEED in $SEEDS; do
  echo "--- seed $SEED ---"
  RESP=$(curl -s -X POST "${WEBAPP}/api/runs/launch" \
    -H "Content-Type: application/json" \
    -d "{\"preset\":\"chat_speak_demo\",\"seed\":$SEED,\"extra_args\":[]}")
  RUN_ID=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('run_id',''))")
  OUT=$(echo "$RESP" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('out_path',''))")
  if [ -z "$RUN_ID" ]; then echo "FAIL launch: $RESP"; exit 1; fi
  echo "  rid=$RUN_ID  out=...$(basename $OUT)"
  while true; do
    sleep 30
    RUNNING=$(curl -s "${WEBAPP}/api/runs/launch/${RUN_ID}" \
      | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('running'))")
    [ "$RUNNING" = "False" ] && break
  done
  JSON_PATHS+=("$OUT")
done

# Aggregate via chat_demo_aggregate (handles chat_speak_demo schema)
AGG_OUT="research/findings/raw/g11_bg/chat_speak_demo_multiseed_${TODAY}.json"
FINDINGS_MD="research/findings/${TODAY}-chat_speak_demo-multiseed.md"
python -m research.runners.chat_demo_aggregate "${JSON_PATHS[@]}" \
  --out "$AGG_OUT" \
  --findings-md "$FINDINGS_MD" \
  --label "chat_speak_demo Track 3 v2 multi-seed (Track 3 layer 4 robustness check)"

echo "=== done ==="
echo "  aggregate: $AGG_OUT"
echo "  findings:  $FINDINGS_MD"
