#!/usr/bin/env bash
# One arm of the three-session check, meant to run under the brain lock:
#   flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock docs/drafts/shared-ltm/run_arm.sh <arm> <cap_gb> [seed] [turnset]
# arm: off    = BRAIN_SHARED_LTM=0 BRAIN_LTM_RNG_ISOLATE=0 (a private LTM copy per conversation; today's default)
#      offiso = BRAIN_SHARED_LTM=0 BRAIN_LTM_RNG_ISOLATE=1 (private copies, loads/builds kept off the global RNG)
#      on     = BRAIN_SHARED_LTM=1 (one shared store; RNG isolation forced on)
# Output: docs/drafts/shared-ltm/runs/<arm>[_s<seed>_<turnset>].json (the first run: no suffix, seed 7, v1).
set -uo pipefail
ARM="${1:?off|offiso|on}"
CAP="${2:?cap_gb}"
SEED="${3:-7}"
TURNSET="${4:-v1}"
W="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$W" || exit 2
case "$ARM" in
  off) FLAGS=(BRAIN_SHARED_LTM=0 BRAIN_LTM_RNG_ISOLATE=0) ;;
  offiso) FLAGS=(BRAIN_SHARED_LTM=0 BRAIN_LTM_RNG_ISOLATE=1) ;;
  on) FLAGS=(BRAIN_SHARED_LTM=1) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
NAME="$ARM"
[ "$#" -ge 3 ] && NAME="${ARM}_s${SEED}_${TURNSET}"
bash tools/mem_ok.sh 14 || exit 3
bash tools/memcap.sh "$CAP" -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 BRAIN_CHAT_SEED="$SEED" \
  "${FLAGS[@]}" /home/dant123/Projects/sim/.venv/bin/python -u docs/drafts/shared-ltm/three_sessions.py \
  "docs/drafts/shared-ltm/runs/$NAME.json" "$TURNSET" 2>&1 | grep -v --line-buffered "SIM_BRIDGE:" \
  > "docs/drafts/shared-ltm/runs/$NAME.log"
echo "ARM=$NAME RC=${PIPESTATUS[0]}"
