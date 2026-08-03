#!/usr/bin/env bash
# Provision the mini-PC pool (pool40/41/42) for numpy-backend sim runs, from the LOCAL box over the LAN.
# The nodes have python3 + internet but NO pip/numpy/repo (reimaged). This is IDEMPOTENT — safe to re-run.
#   1. rsync the code (sim/ + research/ + experiment/ + tests support) over ssh (repos are private → no clone).
#   2. create a venv (python3 -m venv) and pip-install numpy + scipy (scipy REQUIRED for SIM_BACKEND=numpy sparse).
# Usage:  bash tools/pool_provision.sh [pool40 pool41 pool42]
set -uo pipefail
cd "$(dirname "$0")/.."
NODES=("${@:-pool40 pool41 pool42}"); NODES=(${NODES[@]})
REMOTE_ROOT="derisk-pool/sim"   # matches the prior gaming-window manifest path
for h in "${NODES[@]}"; do
  echo "=== provisioning $h ==="
  ssh -o ConnectTimeout=10 "$h" "mkdir -p ~/$REMOTE_ROOT" || { echo "  SSH FAIL $h"; continue; }
  # 1. code (exclude heavy/irrelevant: git, caches, checkpoints, recordings, raw data, venvs, node_modules)
  rsync -az --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv*' \
    --exclude='bridges/' --exclude='simulation_recordings_h5/' --exclude='simulation_checkpoints_h5/' \
    --exclude='research/findings/raw/' --exclude='webapp/' --exclude='node_modules/' --exclude='.venv-rag/' \
    sim/ "$h:~/$REMOTE_ROOT/sim/"
  rsync -az --exclude='__pycache__' --exclude='*.pyc' --exclude='findings/raw/' \
    research/runners/ "$h:~/$REMOTE_ROOT/research/runners/"
  ssh "$h" "touch ~/$REMOTE_ROOT/research/__init__.py ~/$REMOTE_ROOT/research/runners/__init__.py 2>/dev/null; \
            mkdir -p ~/$REMOTE_ROOT/research/findings/raw"
  rsync -az --exclude='__pycache__' experiment/ "$h:~/$REMOTE_ROOT/experiment/" 2>/dev/null
  # 2. ensurepip/venv are missing on these Ubuntu 22.04 nodes -> install via passwordless sudo (verified available)
  ssh "$h" "python3 -c 'import ensurepip' 2>/dev/null || { echo '  installing python3.10-venv+pip'; \
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv python3-pip >/dev/null 2>&1 || \
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1 && \
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv python3-pip >/dev/null 2>&1; }"
  # 3. venv + numpy + scipy (idempotent: recreate if the prior broken attempt left a pip-less venv)
  ssh "$h" "cd ~/$REMOTE_ROOT && \
    { test -x .venv/bin/python && .venv/bin/python -m pip --version >/dev/null 2>&1 || \
      { rm -rf .venv; python3 -m venv .venv; }; } && \
    .venv/bin/python -m pip -q install --upgrade pip >/dev/null 2>&1; \
    .venv/bin/python -m pip -q install numpy scipy h5py pyyaml 2>&1 | tail -1; \
    echo -n '  numpy/scipy=' ; .venv/bin/python -c 'import numpy,scipy; print(numpy.__version__, scipy.__version__)' 2>&1 | tail -1; \
    echo -n '  sim imports=' ; SIM_BACKEND=numpy .venv/bin/python -c 'import sys; sys.path.insert(0,\".\"); from sim.backend import get_backend; print(get_backend()[1])' 2>&1 | tail -1"
  echo "  done $h"
done
echo "ALL PROVISION DONE"
