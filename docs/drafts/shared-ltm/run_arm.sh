#!/usr/bin/env bash
# One arm of the three-session check, meant to run under the brain lock:
#   flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock docs/drafts/shared-ltm/run_arm.sh <off|on> <cap_gb>
# off = BRAIN_SHARED_LTM=0 (a private LTM copy per conversation); on = the default (one shared store).
set -uo pipefail
ARM="${1:?off|on}"
CAP="${2:?cap_gb}"
W="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$W" || exit 2
SHARED=()
[ "$ARM" = off ] && SHARED=(BRAIN_SHARED_LTM=0)
bash tools/mem_ok.sh "$CAP" || exit 3
bash tools/memcap.sh "$CAP" -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "${SHARED[@]}" \
  /home/dant123/Projects/sim/.venv/bin/python -u docs/drafts/shared-ltm/three_sessions.py \
  "docs/drafts/shared-ltm/runs/$ARM.json" > "docs/drafts/shared-ltm/runs/$ARM.log" 2>&1
echo "ARM=$ARM RC=$?"
