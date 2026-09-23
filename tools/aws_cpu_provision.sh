#!/usr/bin/env bash
# aws_cpu_provision.sh — provision a running AWS CPU instance (from tools/aws_cpu_launch.sh) for RAM-heavy
# integrated brain-chat verifies: rsync the CODE ONLY + build a NUMPY venv (NO cupy — CPU instance). Then verify
# the numpy CPU path imports (scipy present, so sim/bridge.py does NOT silently fall back to cupy) + the battery imports.
#
# WHY CODE-ONLY (the 2026-09-16 lesson, baked in): the first CPU provision rsync'd the whole tree and started
# uploading bridges/ (25GB of model checkpoints) + data/ (8GB cifar/corpus), which FILLED the 60GB instance disk
# ("No space left on device") before any venv built. The integrated battery builds a small tiny-demo brain in-code
# and needs NONE of bridges/data/deploy/references — so this excludes them. Result: ~1GB upload, fits easily.
#
#   bash tools/aws_cpu_launch.sh                  # launch r7i.4xlarge (records instance in research/queue/.aws_gpu)
#   bash tools/aws_cpu_provision.sh               # THIS: rsync code + numpy venv + verify imports
#   # then dispatch, e.g.:  <ssh> 'cd ~/sim && SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES="" .venv/bin/python -m research.runners.onebrain_regression_battery --flag <FLAG>'
#   bash tools/aws_gpu.sh terminate              # + delete the SG (see .aws_gpu) when done — no billing leak
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
STATE="$ROOT/research/queue/.aws_gpu"
IID=$(awk -F= '/^instance=/{print $2}' "$STATE" 2>/dev/null)
KEY=$(awk -F= '/^key=/{print $2}' "$STATE" 2>/dev/null)
REGION=$(awk -F= '/^region=/{print $2}' "$STATE" 2>/dev/null); REGION=${REGION:-us-east-1}
[ -n "$IID" ] || { echo "no instance recorded in $STATE — run tools/aws_cpu_launch.sh first"; exit 1; }
IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
[ -n "$IP" ] && [ "$IP" != None ] || { echo "instance $IID has no public IP (not running?)"; exit 1; }
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 ubuntu@$IP"
echo "[aws-cpu] instance=$IID IP=$IP"

$SSH "sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv rsync >/dev/null 2>&1; mkdir -p ~/sim" || exit 1
echo "[aws-cpu] rsync CODE ONLY (excludes bridges/data/deploy/references + raw + venvs — the disk-full lesson)…"
# .claude/ (2026-09-23 addition): NEVER shipped anywhere -- it is where every worktree checkout + session state
# lives (hundreds of GB locally), found while sizing the GPU-lane fix (tools/aws_provision.sh) for the SAME
# board note this script's own header already cites. This script already excluded bridges/data/deploy/references
# correctly; .claude/ was the one gap it shared with the GPU lane.
rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  --exclude '.git' --exclude '.claude' --exclude 'research/findings/raw' --exclude 'research/queue/*.log' \
  --exclude '.venv' --exclude '.venv-rag' \
  --exclude 'data' --exclude 'bridges' --exclude 'deploy' --exclude 'references' \
  --exclude '*.pyc' --exclude '__pycache__' --exclude '*.h5' --exclude '*.pkl' \
  ./ ubuntu@"$IP":~/sim/ || exit 1
# LTM knowledge bundles (2026-09-23, ~105MB): see tools/aws_provision.sh's identical block / tools/
# pool_sync_assets.sh for the full writeup -- _default_ltm_bundle_dir() (webapp/server.py) looks for
# sim-data/knowledge_bundles/* at $HOME/Projects/sim-data, outside this repo, which the rsync above never
# reaches. Best-effort: a sync failure degrades KNOWLEDGE only (still structurally non-degenerate), never
# blocks provisioning.
SIM_DATA_ROOT="${SIM_DATA_ROOT:-$HOME/Projects/sim-data}/knowledge_bundles"
if [ -d "$SIM_DATA_ROOT" ]; then
  echo "[aws-cpu] syncing LTM knowledge bundles…"
  for b in wikidata_100k wikidata_core_15k; do
    [ -d "$SIM_DATA_ROOT/$b" ] || continue
    $SSH "mkdir -p ~/Projects/sim-data/knowledge_bundles/$b"
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      "$SIM_DATA_ROOT/$b/" "ubuntu@$IP:~/Projects/sim-data/knowledge_bundles/$b/" || \
      echo "[aws-cpu] (warning: LTM bundle $b sync failed -- remote brain will build with no LTM)" >&2
  done
else
  echo "[aws-cpu] (no local $SIM_DATA_ROOT -- skipping LTM sync; remote brain will build with no LTM)"
fi
echo "[aws-cpu] venv + numpy/scipy/fastapi/pydantic (CPU set, NO cupy)…"
$SSH "cd ~/sim && python3 -m venv .venv && .venv/bin/pip -q install --upgrade pip >/dev/null 2>&1 && \
      .venv/bin/pip -q install numpy scipy 'fastapi>=0.115' 'uvicorn[standard]>=0.34' 'pydantic>=2.0' 'pyyaml>=6.0' psutil h5py hdf5plugin 2>&1 | tail -3"
echo "[aws-cpu] VERIFY the numpy CPU path + battery import (scipy present => no silent cupy fallback):"
$SSH "cd ~/sim && SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' .venv/bin/python -c \"
import scipy.sparse, numpy
import research.runners.onebrain_regression_battery as b
print('  numpy', numpy.__version__, '| scipy ok | battery import ok | run_regression_battery:', hasattr(b,'run_regression_battery'))
\"" || { echo '[aws-cpu] ⛔ verify failed — see above'; exit 1; }
# The line above imports the MODULE, not `webapp.server` -- onebrain_regression_battery's
# `from webapp.server import brain_chat, BrainChatRequest` is INSIDE `_collect_worker()`, so a module-level
# import verify never exercises it (a broken webapp import would still pass the check above, then crash every
# worker subprocess at battery-run time -- board note 2026-09-22, "exercised=0"). Test the REAL import directly.
echo "[aws-cpu] VERIFY webapp.server (the battery's ACTUAL runtime import, not just the module-level one):"
$SSH "cd ~/sim && SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' .venv/bin/python -c \"
from webapp.server import brain_chat, BrainChatRequest
print('  webapp.server imports ok')
\"" || { echo '[aws-cpu] ⛔ webapp.server import FAILED -- every faculty will read not-exercised; do NOT dispatch the battery here'; exit 1; }
bash "$ROOT/tools/aws_brain_sanity_check.sh" "$SSH" numpy || exit 1
echo "[aws-cpu] READY + brain-size-verified. Dispatch integrated verifies, then: tools/aws_gpu.sh terminate (+ delete the SG from $STATE)."
