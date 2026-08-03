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
SOURCE_SHA=$(git rev-parse HEAD 2>/dev/null || printf unknown)
STAGE=$(mktemp -d)
MANIFEST=$(mktemp)
REVISION=$(mktemp)
FAILED_NODES=()
trap 'rm -rf "$STAGE"; rm -f "$MANIFEST" "$REVISION"' EXIT
git archive HEAD sim research/__init__.py research/runners experiment tools requirements.txt \
  | tar -x -C "$STAGE"
(cd "$STAGE" && { \
  find sim research/runners experiment tools -type f \
    \( -name '*.py' -o -name '*.sh' \) -print0; \
  printf 'research/__init__.py\0'; \
} | sort -z | xargs -0 sha256sum) > "$MANIFEST"
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
EXCLUDED_DIRTY=$(git status --porcelain -- sim research/runners experiment tools 2>/dev/null | wc -l)
printf 'git_sha=%s\nsource_kind=git_archive\nsource_manifest_sha256=%s\nexcluded_worktree_paths=%s\ncreated_utc=%s\n' \
  "$SOURCE_SHA" "$MANIFEST_SHA" "$EXCLUDED_DIRTY" "$(date -u +%FT%TZ)" > "$REVISION"
for h in "${NODES[@]}"; do
  echo "=== provisioning $h ==="
  ssh -o ConnectTimeout=10 "$h" "mkdir -p ~/$REMOTE_ROOT" || {
    echo "  SSH FAIL $h"
    FAILED_NODES+=("$h:ssh")
    continue
  }
  # 1. code (exclude heavy/irrelevant: git, caches, checkpoints, recordings, raw data, venvs, node_modules)
  rsync -az --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv*' \
    --exclude='bridges/' --exclude='simulation_recordings_h5/' --exclude='simulation_checkpoints_h5/' \
    --exclude='research/findings/raw/' --exclude='webapp/' --exclude='node_modules/' --exclude='.venv-rag/' \
    "$STAGE/sim/" "$h:~/$REMOTE_ROOT/sim/"
  rsync -az --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='findings/raw/' \
    "$STAGE/research/runners/" "$h:~/$REMOTE_ROOT/research/runners/"
  rsync -az "$STAGE/research/__init__.py" "$h:~/$REMOTE_ROOT/research/__init__.py"
  ssh "$h" "mkdir -p ~/$REMOTE_ROOT/research/findings/raw"
  rsync -az --delete --exclude='__pycache__' "$STAGE/experiment/" "$h:~/$REMOTE_ROOT/experiment/" 2>/dev/null
  rsync -az --delete --exclude='__pycache__' "$STAGE/tools/" "$h:~/$REMOTE_ROOT/tools/" 2>/dev/null
  rsync -az "$STAGE/requirements.txt" "$h:~/$REMOTE_ROOT/requirements.txt" 2>/dev/null
  rsync -az "$MANIFEST" "$h:~/$REMOTE_ROOT/.source_manifest.sha256"
  rsync -az "$REVISION" "$h:~/$REMOTE_ROOT/.source_revision"
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
  ssh "$h" "cd ~/$REMOTE_ROOT && .venv/bin/python -c 'import json,numpy,scipy,h5py,yaml; json.dump({\"numpy\":numpy.__version__,\"scipy\":scipy.__version__,\"h5py\":h5py.__version__,\"pyyaml\":yaml.__version__},open(\".pool_environment.json\",\"w\"),sort_keys=True,indent=2)'"
  REMOTE_MANIFEST=$(ssh "$h" "cd ~/$REMOTE_ROOT && sha256sum .source_manifest.sha256 | awk '{print \$1}'")
  if [ "$REMOTE_MANIFEST" != "$MANIFEST_SHA" ]; then
    echo "  MANIFEST FAIL local=$MANIFEST_SHA remote=$REMOTE_MANIFEST" >&2
    FAILED_NODES+=("$h:manifest")
    continue
  fi
  ssh "$h" "cd ~/$REMOTE_ROOT && sha256sum -c .source_manifest.sha256 >/dev/null" || {
    echo "  SOURCE FILE VERIFY FAIL" >&2
    FAILED_NODES+=("$h:source-verify")
    continue
  }
  ssh "$h" "cd ~/$REMOTE_ROOT && sed 's/^[0-9a-f]\\{64\\}  //' .source_manifest.sha256 | while IFS= read -r path; do chmod a-w -- \"\$path\" || exit 1; done && chmod a-w .source_manifest.sha256 .source_revision" || {
    echo "  SOURCE READ-ONLY FAIL" >&2
    FAILED_NODES+=("$h:read-only")
    continue
  }
  echo "  source git=$SOURCE_SHA manifest=$MANIFEST_SHA excluded_worktree_paths=$EXCLUDED_DIRTY"
  echo "  done $h"
done
if ((${#FAILED_NODES[@]})); then
  printf 'PROVISION FAILED: %s\n' "${FAILED_NODES[*]}" >&2
  exit 1
fi
echo "ALL PROVISION DONE"
