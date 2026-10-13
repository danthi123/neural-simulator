#!/usr/bin/env bash
# One arm of the M2 reply-hash check, under the brain lock (the lock covers ANY process that builds a brain):
#   flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock docs/drafts/one-brain-m2/run_hashes.sh \
#       <name> <checkout_dir> <seed> <single|pair> [ENV=VAL ...]
# <checkout_dir>: the base export (git archive afeb1bcc9) or this worktree; its `webapp` is the one imported.
# Output: docs/drafts/one-brain-m2/runs/<name>_s<seed>_<script>.json (+ .log).
set -uo pipefail
NAME="${1:?name}"
DIR="${2:?checkout_dir}"
SEED="${3:?seed}"
SCRIPT="${4:?single|pair}"
shift 4
W="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$W/docs/drafts/one-brain-m2/runs/${NAME}_s${SEED}_${SCRIPT}"
mkdir -p "$W/docs/drafts/one-brain-m2/runs"
bash "$W/tools/mem_ok.sh" 12 || exit 3
cd "$DIR" || exit 2
bash "$W/tools/memcap.sh" 12 -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 BRAIN_CHAT_SEED="$SEED" \
  "$@" /home/dant123/Projects/sim/.venv/bin/python -u "$W/docs/drafts/one-brain-m2/conversation_hashes.py" \
  "$OUT.json" "$SCRIPT" 2>&1 | grep -v --line-buffered "SIM_BRIDGE:" > "$OUT.log"
echo "RUN=${NAME}_s${SEED}_${SCRIPT} RC=${PIPESTATUS[0]}"
