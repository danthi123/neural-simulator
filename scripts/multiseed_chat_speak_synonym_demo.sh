#!/usr/bin/env bash
# Track 3 layer 4 :speak Tier 2.1 8-word — multi-seed chat_speak_synonym_demo
# Validates the production-side analog of Tier 2.1 (Tier 2.1 v4 W→A was
# 5/6 aligned at 63.7% — does the same architecture reproduce A→W on
# the 8-word synonym vocab?).
#
# Usage: bash scripts/multiseed_chat_speak_synonym_demo.sh [seed1 seed2 ...]
# Default 6-seed protocol: 42 43 44 100 101 102
# Wall clock: ~10–13 min/seed × 6 = ~60–80 min total

set -e
SEEDS="${@:-42 43 44 100 101 102}"
WEBAPP="${WEBAPP:-http://127.0.0.1:8765}"
TODAY="$(date +%Y-%m-%d)"

if ! curl -sf "${WEBAPP}/api/info" > /dev/null; then
  echo "ERROR: webapp not reachable at ${WEBAPP}"
  exit 1
fi

# Verify preset is wired up (in case webapp wasn't restarted)
HAS_PRESET=$(curl -s "${WEBAPP}/api/info" \
  | python -c "import json,sys; d=json.load(sys.stdin); print('chat_speak_synonym_demo' in d.get('presets',[]))")
if [ "$HAS_PRESET" != "True" ]; then
  echo "ERROR: chat_speak_synonym_demo not in webapp presets — restart uvicorn worker"
  exit 1
fi

echo "=== multiseed_chat_speak_synonym_demo ==="
echo "  seeds: $SEEDS"

# Wait for any active GPU runs first
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
    -d "{\"preset\":\"chat_speak_synonym_demo\",\"seed\":$SEED,\"extra_args\":[]}")
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

# Aggregate via chat_demo_aggregate (chat_speak branch handles the
# chat_speak_synonym_demo schema — it shares all the speak_* fields
# with chat_speak_demo since the synonym variant just extends the
# vocab. The any-synonym A→W is reported as speak_accuracy.).
AGG_OUT="research/findings/raw/multi_seed/chat_speak_synonym_demo_6seed_${TODAY}.json"
mkdir -p "$(dirname $AGG_OUT)"
FINDINGS_MD="research/findings/${TODAY}-chat_speak_synonym_demo-Tier2.1-8word-MULTI-SEED.md"
python -m research.runners.chat_demo_aggregate "${JSON_PATHS[@]}" \
  --out "$AGG_OUT" \
  --findings-md "$FINDINGS_MD" \
  --label "chat_speak_synonym_demo (Tier 2.1 8-word :speak, 6-seed multi-seed)"

echo "=== done ==="
echo "  aggregate: $AGG_OUT"
echo "  findings:  $FINDINGS_MD"
