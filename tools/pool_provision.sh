#!/usr/bin/env bash
# Provision the mini-PC pool (pool40/41/42) for numpy-backend sim runs, from the LOCAL box over the LAN.
# The nodes have python3 + internet but NO pip/numpy/repo (reimaged). This is IDEMPOTENT — safe to re-run.
#   1. rsync the code (sim/ + webapp/ + research/ + experiment/ + tests support) over ssh (repos are private → no clone).
#   2. create a venv (python3 -m venv) and pip-install numpy + scipy (scipy REQUIRED for SIM_BACKEND=numpy sparse)
#      + fastapi/pydantic (REQUIRED by any runner that imports webapp.server, e.g. onebrain_regression_battery.py).
# Usage:  bash tools/pool_provision.sh [--revision <commit>] [--isolated] [pool40 pool41 pool42]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"   # used by the local reference-brain sanity build below (was unbound under set -u, 2026-09-23)
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
# AWS-AS-EXTRA-POOL-NODE (2026-09-23) -- same repo-local, gitignored ssh config as pool_autodispatch.sh /
# pool_sync.sh / pool_queue.sh (see pool_autodispatch.sh's header comment for the full rationale). ABSENT by
# default, so every ssh/rsync call below is unchanged for anyone who hasn't run `aws_pool_node.sh up` (a bare
# `ssh poolNN` / `rsync -e ssh` still resolves poolNN exactly as before). A caller targeting an AWS node passes
# its ssh-config alias (e.g. `pool_provision.sh pool1`) -- resolution of that alias comes from the Host entry
# `aws_pool_node.sh up` wrote into this same file.
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_CMD="ssh"; [ -f "$POOL_SSH_CONFIG" ] && SSH_CMD="ssh -F $POOL_SSH_CONFIG"
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
git archive "$SOURCE_SHA" sim webapp research/__init__.py research/runners research/specs research/fixtures \
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
EXCLUDED_DIRTY=$(git status --porcelain -- sim webapp research/runners experiment tools 2>/dev/null | wc -l)
printf 'git_sha=%s\nsource_kind=git_archive\nsource_manifest_sha256=%s\nsource_ancestry_sha256=%s\nexcluded_worktree_paths=%s\ncreated_utc=%s\n' \
  "$SOURCE_SHA" "$MANIFEST_SHA" "$ANCESTRY_SHA" "$EXCLUDED_DIRTY" "$(date -u +%FT%TZ)" > "$REVISION"
# LOCAL REFERENCE BUILD (2026-09-23, closing the sibling gap to tools/aws_brain_sanity_check.sh -- the mini-PC
# pool had NO non-degenerate build check at all). Build the SAME tiny-demo brain locally ONCE via
# tools/brain_build_sanity.py (sums every SimulationBridge the process constructs; composer_kind='rf', the
# fast numpy path -- this is a STRUCTURAL check, not a science run), then diff each node's own build against
# it below. Guarded by mem_ok/memcap (shared box); a failed/unavailable local reference SKIPS the per-node
# comparison rather than failing the whole provision (the code/asset sync above is still valuable on its own).
LOCAL_ENGINE_PYTHON="${SIM_ENGINE_PYTHON:-$ROOT/.venv/bin/python}"
LOCAL_SANITY_JSON=""
if [ -x "$LOCAL_ENGINE_PYTHON" ] && bash "$ROOT/tools/mem_ok.sh" 8 >&2; then
  LOCAL_SANITY_JSON=$(cd "$STAGE" && bash "$ROOT/tools/memcap.sh" 8 -- \
    env SIM_BACKEND=numpy "$LOCAL_ENGINE_PYTHON" -m tools.brain_build_sanity 2>/dev/null | tail -1) || true
fi
if [ -n "$LOCAL_SANITY_JSON" ]; then
  echo "  local reference brain: $LOCAL_SANITY_JSON"
else
  echo "  (no local reference brain build -- per-node non-degenerate comparison will be skipped below)" >&2
fi

for h in "${NODES[@]}"; do
  echo "=== provisioning $h:$REMOTE_ROOT ==="
  $SSH_CMD -o ConnectTimeout=10 "$h" "mkdir -p \
    ~/$REMOTE_ROOT/sim \
    ~/$REMOTE_ROOT/webapp \
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
  rsync -az -e "$SSH_CMD" --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv*' \
    --exclude='bridges/' --exclude='simulation_recordings_h5/' --exclude='simulation_checkpoints_h5/' \
    --exclude='research/findings/raw/' --exclude='node_modules/' --exclude='.venv-rag/' \
    "$STAGE/sim/" "$h:~/$REMOTE_ROOT/sim/"
  # webapp/ (2026-09-09 fix): many research/runners/* import `webapp.server` (brain_chat / BrainChatRequest) --
  # the cross-faculty regression battery (onebrain_regression_battery.py) among them. It was previously excluded
  # from the archive entirely, so any runner that imports it could not be pool-routed and was forced to run
  # locally, competing with GPU work for RAM. webapp/ is small (~1.7MB incl. static/) so this does not meaningfully
  # widen the payload; static/ must ship too because `app.mount(..., StaticFiles(directory=STATIC_DIR))` at
  # webapp/server.py module-import time raises RuntimeError if that directory does not exist on disk.
  rsync -az -e "$SSH_CMD" --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$STAGE/webapp/" "$h:~/$REMOTE_ROOT/webapp/"
  rsync -az -e "$SSH_CMD" --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='findings/raw/' \
    "$STAGE/research/runners/" "$h:~/$REMOTE_ROOT/research/runners/"
  rsync -az -e "$SSH_CMD" --delete "$STAGE/research/specs/" "$h:~/$REMOTE_ROOT/research/specs/"
  rsync -az -e "$SSH_CMD" --delete "$STAGE/research/fixtures/" "$h:~/$REMOTE_ROOT/research/fixtures/"
  rsync -az -e "$SSH_CMD" --delete --exclude='raw/' \
    "$STAGE/research/findings/" "$h:~/$REMOTE_ROOT/research/findings/"
  rsync -az -e "$SSH_CMD" "$STAGE/research/__init__.py" "$h:~/$REMOTE_ROOT/research/__init__.py"
  $SSH_CMD "$h" "mkdir -p ~/$REMOTE_ROOT/research/findings/raw"
  # CORPUS (2026-09-23): the small corpus files corpus-LEARNED organs read (see load_bearing_fraction CORPUS GUARD).
  $SSH_CMD "$h" "mkdir -p ~/$REMOTE_ROOT/data/corpus"
  # data/corpus is git-excluded, so a git WORKTREE (every isolated agent) has none: fall back to the PRIMARY
  # checkout's copy (the parent of the shared git dir). Sync only the files that exist (a missing optional file
  # used to fail the whole rsync), and warn loudly when the core file is absent. (2026-09-23, D3 fix round.)
  CORPUS_DIR="$ROOT/data/corpus"
  if [ ! -f "$CORPUS_DIR/tinystories.txt" ]; then
    _common=$(cd "$ROOT" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
    [ -n "$_common" ] && [ -f "$(dirname "$_common")/data/corpus/tinystories.txt" ] && CORPUS_DIR="$(dirname "$_common")/data/corpus"
  fi
  _cfiles=()
  for _f in tinystories.txt wikitext.txt simplewiki.txt websters1913.json run3_ra_grounded_frames.txt; do
    [ -e "$CORPUS_DIR/$_f" ] && _cfiles+=("$_f")
  done
  [ -e "$CORPUS_DIR/tinystories.txt" ] || echo "  ⛔ WARNING: no data/corpus/tinystories.txt in $ROOT or the primary checkout -- corpus-learned organs will fail on $h" >&2
  if [ ${#_cfiles[@]} -gt 0 ]; then
    ( cd "$CORPUS_DIR" && rsync -aL -e "$SSH_CMD" "${_cfiles[@]}" "$h:$REMOTE_ROOT/data/corpus/" ) || echo "  (warning: corpus sync to $h failed)" >&2
  fi
  rsync -az -e "$SSH_CMD" --delete --exclude='__pycache__' "$STAGE/experiment/" "$h:~/$REMOTE_ROOT/experiment/" 2>/dev/null
  rsync -az -e "$SSH_CMD" --delete --exclude='__pycache__' "$STAGE/tools/" "$h:~/$REMOTE_ROOT/tools/" 2>/dev/null
  rsync -az -e "$SSH_CMD" --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$STAGE/tests/" "$h:~/$REMOTE_ROOT/tests/"
  rsync -az -e "$SSH_CMD" --delete "$STAGE/docs/" "$h:~/$REMOTE_ROOT/docs/"
  rsync -az -e "$SSH_CMD" "$STAGE/CLAUDE.md" "$STAGE/GAP_CLOSURE_MISSION.md" "$STAGE/README.md" \
    "$STAGE/ROADMAP.md" "$h:~/$REMOTE_ROOT/"
  rsync -az -e "$SSH_CMD" "$STAGE/requirements.txt" "$h:~/$REMOTE_ROOT/requirements.txt" 2>/dev/null
  rsync -az -e "$SSH_CMD" "$STAGE/requirements-dev.txt" "$h:~/$REMOTE_ROOT/requirements-dev.txt" 2>/dev/null
  rsync -az -e "$SSH_CMD" "$MANIFEST" "$h:~/$REMOTE_ROOT/.source_manifest.sha256"
  rsync -az -e "$SSH_CMD" "$REVISION" "$h:~/$REMOTE_ROOT/.source_revision"
  rsync -az -e "$SSH_CMD" "$STAGE/.source_ancestry.json" "$h:~/$REMOTE_ROOT/.source_ancestry.json"
  # LTM knowledge bundles (2026-09-23, ~105MB, NOT a multi-GB haul): _default_ltm_bundle_dir()
  # (webapp/server.py) looks for sim-data/knowledge_bundles/{wikidata_100k,wikidata_core_15k} at
  # $HOME/Projects/sim-data on whatever box is running -- a directory OUTSIDE this repo entirely, which no
  # provisioner shipped before. Without it, every remote brain silently ships with 5 hardcoded facts and NO
  # cortical long-term memory (source stays "tiny-demo", never "tiny-demo +LTM") -- see
  # tools/pool_sync_assets.sh for the full writeup. Best-effort: a sync failure degrades the remote brain's
  # KNOWLEDGE (still non-degenerate structurally), never crashes provisioning.
  bash "$ROOT/tools/pool_sync_assets.sh" "$h" >&2 || echo "  (warning: LTM asset sync failed for $h -- remote brain will build with no LTM)" >&2
  # research/ is synced PER-SUBDIR (above), so --delete never reaches research/ ROOT files or research/<subdir>s
  # this script does not sync. A past full-tree sync left such files on pool41 (FAILURE_LOG.md, research/biology/*,
  # ...) and the strict complete-source verify below then failed EVERY re-provision on "extra files" — which kept
  # the node unusable for days (2026-09-23). Prune every research/ file the manifest does not carry, keeping run
  # OUTPUTS exactly as source_manifest.py's verify ignores them (findings/raw/, experiment-runtime/, *.log, *.out).
  $SSH_CMD "$h" "cd ~/$REMOTE_ROOT && sed 's/^[0-9a-f]\\{64\\}  //' .source_manifest.sha256 | sort > /tmp/.prov_keep.\$\$ && find research -type f ! -path 'research/findings/raw/*' ! -path 'research/experiment-runtime/*' ! -path '*/__pycache__/*' ! -name '*.log' ! -name '*.out' | sort | comm -23 - /tmp/.prov_keep.\$\$ | while IFS= read -r p; do chmod u+w -- \"\$p\" 2>/dev/null; rm -f -- \"\$p\"; done; rm -f /tmp/.prov_keep.\$\$"
  # 2. ensurepip/venv are missing on these Ubuntu 22.04 nodes -> install via passwordless sudo (verified available)
  $SSH_CMD "$h" "python3 -c 'import ensurepip' 2>/dev/null || { echo '  installing python3.10-venv+pip'; \
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv python3-pip >/dev/null 2>&1 || \
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1 && \
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv python3-pip >/dev/null 2>&1; }"
  # 3. venv + numpy + scipy (idempotent: recreate if the prior broken attempt left a pip-less venv)
  $SSH_CMD "$h" "cd ~/$REMOTE_ROOT && \
    { test -x .venv/bin/python && .venv/bin/python -m pip --version >/dev/null 2>&1 || \
      { rm -rf .venv; python3 -m venv .venv; }; } && \
    .venv/bin/python -m pip -q install --upgrade pip >/dev/null 2>&1; \
    .venv/bin/python -m pip -q install \
      numpy==2.2.6 scipy==1.15.3 h5py==3.16.0 pillow==12.0.0 pyyaml==6.0.3 pytest==8.4.1 \
      fastapi==0.139.1 pydantic==2.13.4 2>&1 | tail -1; \
    echo -n '  numpy/scipy=' ; .venv/bin/python -c 'import numpy,scipy; print(numpy.__version__, scipy.__version__)' 2>&1 | tail -1; \
    echo -n '  sim imports=' ; SIM_BACKEND=numpy .venv/bin/python -c 'import sys; sys.path.insert(0,\".\"); from sim.backend import get_backend; print(get_backend()[1])' 2>&1 | tail -1; \
    echo -n '  webapp imports=' ; SIM_BACKEND=numpy .venv/bin/python -c 'import sys; sys.path.insert(0,\".\"); from webapp.server import brain_chat, BrainChatRequest; print(\"ok\")' 2>&1 | tail -1"
  $SSH_CMD "$h" "cd ~/$REMOTE_ROOT && .venv/bin/python -c 'import json,sys,numpy,scipy,h5py,PIL,yaml,fastapi,pydantic; json.dump({\"python_major_minor\":\"%s.%s\" % sys.version_info[:2],\"numpy\":numpy.__version__,\"scipy\":scipy.__version__,\"h5py\":h5py.__version__,\"pillow\":PIL.__version__,\"pyyaml\":yaml.__version__,\"fastapi\":fastapi.__version__,\"pydantic\":pydantic.VERSION},open(\".pool_environment.json\",\"w\"),sort_keys=True,separators=(\",\",\":\"))'"
  REMOTE_MANIFEST=$($SSH_CMD "$h" "cd ~/$REMOTE_ROOT && sha256sum .source_manifest.sha256 | awk '{print \$1}'")
  if [ "$REMOTE_MANIFEST" != "$MANIFEST_SHA" ]; then
    echo "  MANIFEST FAIL local=$MANIFEST_SHA remote=$REMOTE_MANIFEST" >&2
    FAILED_NODES+=("$h:manifest")
    continue
  fi
  $SSH_CMD "$h" "cd ~/$REMOTE_ROOT && sha256sum -c .source_manifest.sha256 >/dev/null" || {
    echo "  SOURCE FILE VERIFY FAIL" >&2
    FAILED_NODES+=("$h:source-verify")
    continue
  }
  $SSH_CMD "$h" "cd ~/$REMOTE_ROOT && .venv/bin/python tools/pool/provisioning/source_manifest.py verify --root . --manifest .source_manifest.sha256 --expected-sha256 '$MANIFEST_SHA' >/dev/null" || {
    echo "  COMPLETE SOURCE FILE SET VERIFY FAIL" >&2
    FAILED_NODES+=("$h:complete-source-verify")
    continue
  }
  $SSH_CMD "$h" "cd ~/$REMOTE_ROOT && sed 's/^[0-9a-f]\\{64\\}  //' .source_manifest.sha256 | while IFS= read -r path; do chmod a-w -- \"\$path\" || exit 1; done && chmod a-w .source_manifest.sha256 .source_revision .source_ancestry.json" || {
    echo "  SOURCE READ-ONLY FAIL" >&2
    FAILED_NODES+=("$h:read-only")
    continue
  }
  echo "  source git=$SOURCE_SHA manifest=$MANIFEST_SHA ancestry=$ANCESTRY_SHA excluded_worktree_paths=$EXCLUDED_DIRTY"
  # NON-DEGENERATE SANITY CHECK (2026-09-23): "does it import" (the checks above) cannot catch a degenerate
  # build -- board note 2026-09-22's AWS "2-neuron/0-synapse" brain imported everything fine. Build the SAME
  # tiny-demo brain on THIS node and compare its neuron/synapse totals to the local reference computed above.
  # Advisory-only when no local reference exists (never blocks provisioning on a box that could not itself
  # build one); a FAILED or MISMATCHED remote build marks the node failed.
  REMOTE_SANITY_JSON=$($SSH_CMD "$h" "cd ~/$REMOTE_ROOT && SIM_BACKEND=numpy .venv/bin/python -m tools.brain_build_sanity" 2>/dev/null | tail -1) || true
  if [ -z "$REMOTE_SANITY_JSON" ]; then
    echo "  ⛔ SANITY CHECK FAILED (no output / brain build crashed) on $h" >&2
    FAILED_NODES+=("$h:sanity-crash")
    continue
  fi
  echo "  remote brain: $REMOTE_SANITY_JSON"
  if ! echo "$REMOTE_SANITY_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("ok") else 1)' 2>/dev/null; then
    echo "  ⛔ DEGENERATE/FAILED remote brain build on $h: $REMOTE_SANITY_JSON" >&2
    FAILED_NODES+=("$h:sanity-degenerate")
    continue
  fi
  if [ -n "$LOCAL_SANITY_JSON" ]; then
    MATCH=$(python3 -c '
import json, sys
local = json.loads(sys.argv[1]); remote = json.loads(sys.argv[2])
print("1" if (local.get("n_neurons") == remote.get("n_neurons") and local.get("n_synapses") == remote.get("n_synapses")) else "0")
' "$LOCAL_SANITY_JSON" "$REMOTE_SANITY_JSON" 2>/dev/null) || MATCH="0"
    if [ "$MATCH" != "1" ]; then
      echo "  ⛔ MISMATCH vs local reference on $h: local=$LOCAL_SANITY_JSON remote=$REMOTE_SANITY_JSON" >&2
      FAILED_NODES+=("$h:sanity-mismatch")
      continue
    fi
  fi
  echo "  done $h"
done
if ((${#FAILED_NODES[@]})); then
  printf 'PROVISION FAILED: %s\n' "${FAILED_NODES[*]}" >&2
  exit 1
fi
echo "ALL PROVISION DONE"
