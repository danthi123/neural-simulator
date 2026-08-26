#!/usr/bin/env bash
# Provision the mini-PC pool (pool40/41/42) for numpy-backend sim runs, from the LOCAL box over the LAN.
# The nodes have python3 + internet but NO pip/numpy/repo (reimaged). This is IDEMPOTENT — safe to re-run.
#   1. rsync the code (sim/ + research/ + experiment/ + tests support) over ssh (repos are private → no clone).
#   2. create a venv (python3 -m venv) and pip-install numpy + scipy (scipy REQUIRED for SIM_BACKEND=numpy sparse).
# Usage:  bash tools/pool_provision.sh [--revision <commit>] [--isolated] [pool40 pool41 pool42]
set -euo pipefail
cd "$(dirname "$0")/.."
REVISION_REF=HEAD
ISOLATED=0
while (( $# )); do
  case "$1" in
    --revision)
      if [[ -z "${2:-}" ]]; then
        echo "usage: bash tools/pool_provision.sh [--revision <commit>] [--isolated] [pool40 pool41 pool42]" >&2
        exit 2
      fi
      REVISION_REF=$2
      shift 2
      ;;
    --isolated)
      ISOLATED=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done
NODES=("${@:-pool40 pool41 pool42}"); NODES=(${NODES[@]})
SOURCE_SHA=$(git rev-parse --verify "${REVISION_REF}^{commit}" 2>/dev/null) || {
  echo "invalid source revision: $REVISION_REF" >&2
  exit 2
}
# STALE-SOURCE GUARD (2026-08-26). This script rsyncs --delete over whatever the pool nodes already have, so
# provisioning from a checkout that is BEHIND origin/main silently regresses already-fixed code across every
# node in one shot. Measured: a provisioning run from a ~2-week-stale worktree (missing the RFPhasorComposer
# RUNTIME GROWTH fix, commit 5b2d1d7c3e, 2026-08-12) overwrote pool40/41/42's own up-to-date git checkouts and
# crash-looped the GNW coincidence-integrator derisk with `KeyError: 'confirm'` 25+ times over 3+ hours before
# anyone traced it to the PROVISIONER rather than the runner. Neither the source manifest nor the ancestry
# attestation below catches this: both faithfully certify whatever was archived, stale or not. Refuse unless the
# revision being provisioned already contains origin/main.
if timeout 15 git fetch origin main --quiet 2>/dev/null && git rev-parse --verify origin/main^{commit} >/dev/null 2>&1; then
  MAIN_SHA=$(git rev-parse origin/main)
  if ! git merge-base --is-ancestor "$MAIN_SHA" "$SOURCE_SHA" 2>/dev/null; then
    BEHIND=$(git rev-list --count "$SOURCE_SHA..$MAIN_SHA" 2>/dev/null || echo "?")
    echo "⛔ REFUSED: revision $SOURCE_SHA ($REVISION_REF) does not contain origin/main ($MAIN_SHA) -- it is" >&2
    echo "   $BEHIND commit(s) behind (or diverged from) main and would regress the pool nodes' code." >&2
    echo "   Rebase/merge origin/main into $REVISION_REF first, or set POOL_PROVISION_ALLOW_STALE=1 to" >&2
    echo "   provision anyway (deliberate/isolated reproduction of an old state only)." >&2
    [ "${POOL_PROVISION_ALLOW_STALE:-0}" = "1" ] || exit 2
  fi
else
  echo "  (warning: could not resolve origin/main -- skipping the stale-source ancestry guard)" >&2
fi
if (( ISOLATED )); then
  REMOTE_ROOT="derisk-pool/revisions/$SOURCE_SHA"
else
  REMOTE_ROOT="derisk-pool/sim"   # compatibility path for existing pool jobs
fi
STAGE=$(mktemp -d)
MANIFEST=$(mktemp)
REVISION=$(mktemp)
FAILED_NODES=()
trap 'rm -rf "$STAGE"; rm -f "$MANIFEST" "$REVISION"' EXIT
git archive "$SOURCE_SHA" sim research/__init__.py research/runners research/specs research/fixtures \
  research/findings ':(exclude)research/findings/raw' experiment tools tests \
  docs CLAUDE.md GAP_CLOSURE_MISSION.md README.md ROADMAP.md requirements.txt requirements-dev.txt \
  | tar -x -C "$STAGE"
# Execute the generator extracted from HEAD. A dirty worktree copy must not mint
# the trust record for a different archived source revision.
python3 "$STAGE/tools/pool/provisioning/ancestry_attestation.py" create \
  --repo . --revision "$SOURCE_SHA" --output "$STAGE/.source_ancestry.json" >/dev/null || {
    echo "SOURCE ANCESTRY ATTESTATION FAIL" >&2
    exit 1
  }
ANCESTRY_SHA=$(sha256sum "$STAGE/.source_ancestry.json" | awk '{print $1}')
rm -f "$MANIFEST"
python3 "$STAGE/tools/pool/provisioning/source_manifest.py" create \
  --root "$STAGE" --output "$MANIFEST" >/dev/null
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
EXCLUDED_DIRTY=$(git status --porcelain -- sim research/runners experiment tools 2>/dev/null | wc -l)
printf 'git_sha=%s\nsource_kind=git_archive\nsource_manifest_sha256=%s\nsource_ancestry_sha256=%s\nexcluded_worktree_paths=%s\ncreated_utc=%s\n' \
  "$SOURCE_SHA" "$MANIFEST_SHA" "$ANCESTRY_SHA" "$EXCLUDED_DIRTY" "$(date -u +%FT%TZ)" > "$REVISION"
for h in "${NODES[@]}"; do
  echo "=== provisioning $h:$REMOTE_ROOT ==="
  ssh -o ConnectTimeout=10 "$h" "mkdir -p \
    ~/$REMOTE_ROOT/sim \
    ~/$REMOTE_ROOT/research/runners \
    ~/$REMOTE_ROOT/research/specs \
    ~/$REMOTE_ROOT/research/fixtures \
    ~/$REMOTE_ROOT/research/findings/raw \
    ~/$REMOTE_ROOT/experiment \
    ~/$REMOTE_ROOT/tools \
    ~/$REMOTE_ROOT/tests \
    ~/$REMOTE_ROOT/docs" || {
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
  rsync -az --delete "$STAGE/research/specs/" "$h:~/$REMOTE_ROOT/research/specs/"
  rsync -az --delete "$STAGE/research/fixtures/" "$h:~/$REMOTE_ROOT/research/fixtures/"
  rsync -az --delete --exclude='raw/' \
    "$STAGE/research/findings/" "$h:~/$REMOTE_ROOT/research/findings/"
  rsync -az "$STAGE/research/__init__.py" "$h:~/$REMOTE_ROOT/research/__init__.py"
  ssh "$h" "mkdir -p ~/$REMOTE_ROOT/research/findings/raw"
  rsync -az --delete --exclude='__pycache__' "$STAGE/experiment/" "$h:~/$REMOTE_ROOT/experiment/" 2>/dev/null
  rsync -az --delete --exclude='__pycache__' "$STAGE/tools/" "$h:~/$REMOTE_ROOT/tools/" 2>/dev/null
  rsync -az --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$STAGE/tests/" "$h:~/$REMOTE_ROOT/tests/"
  rsync -az --delete "$STAGE/docs/" "$h:~/$REMOTE_ROOT/docs/"
  rsync -az "$STAGE/CLAUDE.md" "$STAGE/GAP_CLOSURE_MISSION.md" "$STAGE/README.md" \
    "$STAGE/ROADMAP.md" "$h:~/$REMOTE_ROOT/"
  rsync -az "$STAGE/requirements.txt" "$h:~/$REMOTE_ROOT/requirements.txt" 2>/dev/null
  rsync -az "$STAGE/requirements-dev.txt" "$h:~/$REMOTE_ROOT/requirements-dev.txt" 2>/dev/null
  rsync -az "$MANIFEST" "$h:~/$REMOTE_ROOT/.source_manifest.sha256"
  rsync -az "$REVISION" "$h:~/$REMOTE_ROOT/.source_revision"
  rsync -az "$STAGE/.source_ancestry.json" "$h:~/$REMOTE_ROOT/.source_ancestry.json"
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
    .venv/bin/python -m pip -q install \
      numpy==2.2.6 scipy==1.15.3 h5py==3.16.0 pillow==12.0.0 pyyaml==6.0.3 pytest==8.4.1 2>&1 | tail -1; \
    echo -n '  numpy/scipy=' ; .venv/bin/python -c 'import numpy,scipy; print(numpy.__version__, scipy.__version__)' 2>&1 | tail -1; \
    echo -n '  sim imports=' ; SIM_BACKEND=numpy .venv/bin/python -c 'import sys; sys.path.insert(0,\".\"); from sim.backend import get_backend; print(get_backend()[1])' 2>&1 | tail -1"
  ssh "$h" "cd ~/$REMOTE_ROOT && .venv/bin/python -c 'import json,sys,numpy,scipy,h5py,PIL,yaml; json.dump({\"python_major_minor\":\"%s.%s\" % sys.version_info[:2],\"numpy\":numpy.__version__,\"scipy\":scipy.__version__,\"h5py\":h5py.__version__,\"pillow\":PIL.__version__,\"pyyaml\":yaml.__version__},open(\".pool_environment.json\",\"w\"),sort_keys=True,separators=(\",\",\":\"))'"
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
  ssh "$h" "cd ~/$REMOTE_ROOT && .venv/bin/python tools/pool/provisioning/source_manifest.py verify --root . --manifest .source_manifest.sha256 --expected-sha256 '$MANIFEST_SHA' >/dev/null" || {
    echo "  COMPLETE SOURCE FILE SET VERIFY FAIL" >&2
    FAILED_NODES+=("$h:complete-source-verify")
    continue
  }
  ssh "$h" "cd ~/$REMOTE_ROOT && sed 's/^[0-9a-f]\\{64\\}  //' .source_manifest.sha256 | while IFS= read -r path; do chmod a-w -- \"\$path\" || exit 1; done && chmod a-w .source_manifest.sha256 .source_revision .source_ancestry.json" || {
    echo "  SOURCE READ-ONLY FAIL" >&2
    FAILED_NODES+=("$h:read-only")
    continue
  }
  echo "  source git=$SOURCE_SHA manifest=$MANIFEST_SHA ancestry=$ANCESTRY_SHA excluded_worktree_paths=$EXCLUDED_DIRTY"
  echo "  done $h"
done
if ((${#FAILED_NODES[@]})); then
  printf 'PROVISION FAILED: %s\n' "${FAILED_NODES[*]}" >&2
  exit 1
fi
echo "ALL PROVISION DONE"
