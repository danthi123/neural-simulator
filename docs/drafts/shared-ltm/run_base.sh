#!/usr/bin/env bash
# The base-6a3bfe247 OFF hash (review item 4): run this branch's driver against a code-only export of the base commit
# (git archive 6a3bfe247, minus research/findings, which is symlinked back; data symlinked), under the brain lock:
#   flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock docs/drafts/shared-ltm/run_base.sh <base_dir> <seed> <turnset>
# Output: docs/drafts/shared-ltm/runs/base6a3bfe247_s<seed>_<turnset>.json. Compare with off_s<seed>_<turnset>.json.
set -uo pipefail
BASE="${1:?base_dir}"
SEED="${2:-7}"
TURNSET="${3:-v2}"
W="$(cd "$(dirname "$0")/../../.." && pwd)"
NAME="base6a3bfe247_s${SEED}_${TURNSET}"
bash "$W/tools/mem_ok.sh" 14 || exit 3
cd "$BASE" || exit 2
bash "$W/tools/memcap.sh" 20 -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 BRAIN_CHAT_SEED="$SEED" \
  /home/dant123/Projects/sim/.venv/bin/python -u "$W/docs/drafts/shared-ltm/three_sessions.py" \
  "$W/docs/drafts/shared-ltm/runs/$NAME.json" "$TURNSET" 2>&1 | grep -v --line-buffered "SIM_BRIDGE:" \
  > "$W/docs/drafts/shared-ltm/runs/$NAME.log"
echo "BASE=$NAME RC=${PIPESTATUS[0]}"
