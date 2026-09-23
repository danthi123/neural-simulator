#!/usr/bin/env bash
# Provision the AWS GPU lane: repo + venv + cupy, then verify the GPU is actually reachable FROM PYTHON, AND
# that the brain the battery actually builds comes out the right SIZE.
#
# The last check that matters is the one nobody runs: `nvidia-smi` proving a GPU exists says nothing about
# whether cupy can USE it. A 4-arm sweep once ran ~50 min on CPU while every monitor read healthy, because
# SIM_BACKEND=numpy silently won. So this asserts cp.cuda.runtime.getDeviceCount() > 0 and refuses otherwise.
#
# 2026-09-23 FIX (board note 2026-09-22: "AWS Phase 2 ABANDONED -- AWS env built a DEGENERATE 2-neuron/0-synapse
# brain (even with h5py/hdf5plugin) -> exercised=0"). ROOT-CAUSED, not guessed: the regression battery / every
# faculty verify imports `from webapp.server import brain_chat, BrainChatRequest`
# (research/runners/onebrain_regression_battery.py:384) -- and this script's rsync used to `--exclude 'webapp'`
# outright, plus never installed fastapi/pydantic (webapp/server.py's own top-level imports). Either alone
# crashes that import; together it is certain. A crashed worker subprocess returns no responses, and
# `compare()`'s `(on_responses or {}).get(turn_label) or {}` then reads EVERY faculty as "not-exercised" --
# exactly the observed "exercised=0", independent of the exact traceback. `tools/pool_provision.sh` (the mini-PC
# lane) already ships webapp/ + fastapi/pydantic and even runs this SAME import as its own post-provision check
# -- this script is now brought in line with that proven pattern, plus the NEW brain_build_sanity comparison
# below (root-causing the import gap does not by itself prove no OTHER silent-degenerate path exists).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
STATE="$ROOT/research/queue/.aws_gpu"
IID=$(awk -F= '/^instance=/{print $2}' "$STATE"); KEY=$(awk -F= '/^key=/{print $2}' "$STATE")
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[].Instances[].PublicIpAddress' --output text)
[ -z "$IP" ] && { echo "no public IP for $IID"; exit 1; }
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 ubuntu@$IP"
echo "[aws] provisioning $IID at $IP"
$SSH "sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv rsync >/dev/null 2>&1; mkdir -p ~/sim" || exit 1
# webapp/ (1.7M) is now SHIPPED -- it was the root cause (see header). The multi-GB payloads stay excluded:
# data/ (11G, mostly data/corpus/), bridges/lmtrain/ (24G of Qwen mouth checkpoints), deploy/ (2.1G, almost
# entirely one Ubuntu ISO installer for the mini-PC pool -- irrelevant to an already-running EC2 instance),
# references/ (186M of textbook PDFs, used only by the LOCAL RAG index) and research/queue/*.log (queue-dispatch
# logs that grow into the hundreds of MB and carry no code). NEWLY FOUND while sizing this fix (a live rsync
# --dry-run against the un-excluded tree, per the "don't just fix the reported bug, verify the whole payload"
# discipline): `.claude/` was NEVER excluded and is where every worktree checkout + session state lives --
# hundreds of GB locally, so its absence from any exclude list was a second, independent multi-GB-payload risk
# beyond the one the board note named. The rest of bridges/ (developed/ 34M, lineage/ 587M, the *_aw/ lineages
# ~25M each) ships: developed_brain_io's is_developed_brain_bundle()/load_developed_brain() path reads
# brain.json/grounded_codes.npz/facts.json from there, and none of it approaches "multi-GB checkpoint" scale
# once lmtrain/ is excluded.
rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  --exclude '.git' --exclude '.claude' --exclude 'research/findings/raw' \
  --exclude 'research/queue/*.log' --exclude '.venv' --exclude '.venv-rag' \
  --exclude 'data' --exclude 'bridges/lmtrain' --exclude 'deploy' --exclude 'references' --exclude '*.pt' \
  "$ROOT/" "ubuntu@$IP:~/sim/" || exit 1
# LTM knowledge bundles (2026-09-23, ~105MB): _default_ltm_bundle_dir() (webapp/server.py) looks for
# sim-data/knowledge_bundles/{wikidata_100k,wikidata_core_15k} at $HOME/Projects/sim-data -- a directory
# OUTSIDE this repo entirely, which the rsync above never reaches. Without it the brain still builds to the
# correct STRUCTURAL size (brain_build_sanity below still passes -- LTM attaches post-construction, see
# webapp/server.py::_build_chat_brain's tiny-demo branch) but silently carries NO cortical long-term memory.
# Best-effort: a sync failure degrades KNOWLEDGE only, never blocks provisioning. See tools/pool_sync_assets.sh.
SIM_DATA_ROOT="${SIM_DATA_ROOT:-$HOME/Projects/sim-data}/knowledge_bundles"
if [ -d "$SIM_DATA_ROOT" ]; then
  echo "[aws] syncing LTM knowledge bundles…"
  for b in wikidata_100k wikidata_core_15k; do
    [ -d "$SIM_DATA_ROOT/$b" ] || continue
    $SSH "mkdir -p ~/Projects/sim-data/knowledge_bundles/$b"
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      "$SIM_DATA_ROOT/$b/" "ubuntu@$IP:~/Projects/sim-data/knowledge_bundles/$b/" || \
      echo "[aws] (warning: LTM bundle $b sync failed -- remote brain will build with no LTM)" >&2
  done
else
  echo "[aws] (no local $SIM_DATA_ROOT -- skipping LTM sync; remote brain will build with no LTM)"
fi
$SSH "cd ~/sim && python3 -m venv .venv && .venv/bin/pip -q install --upgrade pip && \
      .venv/bin/pip -q install numpy scipy 'cupy-cuda12x' h5py hdf5plugin pyyaml \
        'fastapi>=0.115' 'uvicorn[standard]>=0.34' 'pydantic>=2.0' psutil 2>&1 | tail -2"
echo "[aws] VERIFYING cupy sees the GPU (nvidia-smi is not evidence that PYTHON can use it):"
$SSH "cd ~/sim && .venv/bin/python -c \"
import cupy as cp
n = cp.cuda.runtime.getDeviceCount()
assert n > 0, 'cupy sees NO device -- the lane would silently run on CPU'
print('  cupy devices:', n, '|', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
x = cp.arange(1000); print('  gpu sum ok:', int(x.sum()))
\"" || { echo "[aws] ⛔ cupy cannot use the GPU -- do NOT dispatch GPU work here"; exit 1; }
echo "[aws] VERIFYING webapp.server (the battery's REAL handler) actually imports remotely:"
$SSH "cd ~/sim && SIM_BACKEND=numpy .venv/bin/python -c \"
from webapp.server import brain_chat, BrainChatRequest
print('  webapp.server imports ok')
\"" || { echo "[aws] ⛔ webapp.server import FAILED remotely -- every faculty will read not-exercised (the exact 2026-09-22 symptom); do NOT dispatch the battery here"; exit 1; }
bash "$ROOT/tools/aws_brain_sanity_check.sh" "$SSH" cupy || exit 1
echo "[aws] provisioned + brain-size-verified. Dispatch with: $SSH 'cd ~/sim && SIM_BACKEND=cupy .venv/bin/python -m research.runners.<X>'"
