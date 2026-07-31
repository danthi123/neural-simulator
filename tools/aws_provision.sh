#!/usr/bin/env bash
# Provision the AWS GPU lane: repo + venv + cupy, then verify the GPU is actually reachable FROM PYTHON.
#
# The last check that matters is the one nobody runs: `nvidia-smi` proving a GPU exists says nothing about
# whether cupy can USE it. A 4-arm sweep once ran ~50 min on CPU while every monitor read healthy, because
# SIM_BACKEND=numpy silently won. So this asserts cp.cuda.runtime.getDeviceCount() > 0 and refuses otherwise.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
STATE="$ROOT/research/queue/.aws_gpu"
IID=$(awk -F= '/^instance=/{print $2}' "$STATE"); KEY=$(awk -F= '/^key=/{print $2}' "$STATE")
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[].Instances[].PublicIpAddress' --output text)
[ -z "$IP" ] && { echo "no public IP for $IID"; exit 1; }
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 ubuntu@$IP"
echo "[aws] provisioning $IID at $IP"
$SSH "sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv rsync >/dev/null 2>&1; mkdir -p ~/sim" || exit 1
rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  --exclude '.git' --exclude 'research/findings/raw' --exclude '.venv' --exclude '.venv-rag' \
  --exclude 'data' --exclude 'webapp' --exclude '*.pt' \
  "$ROOT/" "ubuntu@$IP:~/sim/" || exit 1
$SSH "cd ~/sim && python3 -m venv .venv && .venv/bin/pip -q install --upgrade pip && \
      .venv/bin/pip -q install numpy scipy 'cupy-cuda12x' 2>&1 | tail -2"
echo "[aws] VERIFYING cupy sees the GPU (nvidia-smi is not evidence that PYTHON can use it):"
$SSH "cd ~/sim && .venv/bin/python -c \"
import cupy as cp
n = cp.cuda.runtime.getDeviceCount()
assert n > 0, 'cupy sees NO device -- the lane would silently run on CPU'
print('  cupy devices:', n, '|', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
x = cp.arange(1000); print('  gpu sum ok:', int(x.sum()))
\"" || { echo "[aws] ⛔ cupy cannot use the GPU -- do NOT dispatch GPU work here"; exit 1; }
echo "[aws] provisioned. Dispatch with: $SSH 'cd ~/sim && SIM_BACKEND=cupy .venv/bin/python -m research.runners.<X>'"
