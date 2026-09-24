#!/usr/bin/env bash
# tools/pool_backfill_provisioned_markers.sh — write the missing `.provisioned_ok` completion marker (fix round
# #2) onto pool revision directories that were provisioned BEFORE that marker existed.
#
# THE DEFECT (2026-09-23 fix round #3, re-review HIGH): pool_autodispatch.sh's revision_available() started
# requiring ~/derisk-pool/revisions/<sha>/.provisioned_ok as the LAST step of a --isolated pool_provision.sh run
# (fix round #2, closing a HALF-PROVISIONED-DIR hazard). Every revision already on pool40/41/42 BEFORE that
# change is a full, genuinely-provisioned directory with no marker at all -- so once revision_available()'s new
# check merges to main (where the live dispatcher runs), every job pinned to one of those revisions is skipped on
# EVERY node, forever (until the 12h staleness cutoff quietly drops it), even though the node can run it fine.
# Read-only probe 2026-09-23: pool41 has 464d970/01f7a5a/d4484ac/7f90034 as full directories with NO marker;
# pool42 the same for 464d970/01f7a5a/7f90034. All 20 jobs in the live queue were pinned to those four revisions.
#
# THE FIX: verify each markerless revision directory with the SAME checks pool_provision.sh ends a successful
# --isolated provision with -- the venv's python imports numpy/scipy/sim/webapp, and the source manifest verifies
# (every tracked file's hash matches AND no untracked file exists under the recorded set) -- and ONLY THEN write
# the marker. revision_available() (tools/pool_autodispatch.sh) and pool_queue.sh's `add` gate both call the ONE
# shared predicate in tools/pool_revision_marker.sh; this script is the migration that makes existing directories
# satisfy it. It never touches the caller's local working tree or HEAD -- a legacy revision may predate it by
# months, and this only asks "does this directory, as it stands, match what it already claims about itself".
#
# ORDERING (load-bearing, see docs/BUILD_LANE_CHECKLIST.md / this branch's merge instructions): run this BEFORE
# the dispatcher (or its systemd unit) is restarted onto a revision_available() that requires the marker. Until
# then a job pinned to a legacy revision stays safely QUEUED either way (revision_available fails closed and
# pop_job leaves it in the queue rather than popping-and-losing it -- see pool_autodispatch.sh's
# revision_available_cached) -- running this first just avoids waiting out the 12h staleness cutoff for no
# reason on capacity that was already there.
#
# Usage: bash tools/pool_backfill_provisioned_markers.sh [--dry-run] [pool40 pool41 pool42 ...]
#   --dry-run   print what WOULD be marked/skipped without writing anything.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/pool_revision_marker.sh
source "$ROOT/tools/pool_revision_marker.sh"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi
NODES=("${@:-pool40 pool41 pool42}"); NODES=(${NODES[@]})
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_F=(); [ -f "$POOL_SSH_CONFIG" ] && SSH_F=(-F "$POOL_SSH_CONFIG")

# THE REMOTE PAYLOAD (runs entirely ON the node -- no revision's file contents ever cross the wire). Mirrors
# pool_provision.sh's own venv-import block, then its two manifest-verify calls, in the SAME order it runs them
# before writing the marker itself. `--expected-sha256` is the manifest FILE's OWN hash (source_manifest.py's
# self-verify mode: "does this directory currently match what its own recorded manifest claims"), never a
# caller-supplied local revision -- this script deliberately never compares against the CALLER's working tree.
read -r -d '' REMOTE_BACKFILL_SCRIPT <<REMOTE_EOF || true
DRY_RUN="\$1"
MARKER="$POOL_REVISION_MARKER_FILE"
shopt -s nullglob
for d in "\$HOME"/derisk-pool/revisions/*/; do
  d="\${d%/}"
  [ -f "\$d/\$MARKER" ] && continue
  name=\$(basename "\$d")
  if [ ! -x "\$d/.venv/bin/python" ]; then
    echo "SKIP \$name: no .venv/bin/python"; continue
  fi
  if ! ( cd "\$d" && SIM_BACKEND=numpy SIM_NO_PROVENANCE=1 .venv/bin/python -c '
import sys
sys.path.insert(0, ".")
import numpy, scipy
from sim.backend import get_backend
get_backend()
from webapp.server import brain_chat, BrainChatRequest
' >/dev/null 2>&1 ); then
    echo "SKIP \$name: numpy/scipy/sim/webapp import check failed"; continue
  fi
  if [ ! -f "\$d/.source_manifest.sha256" ]; then
    echo "SKIP \$name: no .source_manifest.sha256 -- not a --isolated provision, or predates it"; continue
  fi
  if ! ( cd "\$d" && sha256sum -c .source_manifest.sha256 >/dev/null 2>&1 ); then
    echo "SKIP \$name: source file verify (sha256sum -c .source_manifest.sha256) failed"; continue
  fi
  self_sha=\$(sha256sum "\$d/.source_manifest.sha256" | awk '{print \$1}')
  if ! ( cd "\$d" && .venv/bin/python tools/pool/provisioning/source_manifest.py verify --root . \\
        --manifest .source_manifest.sha256 --expected-sha256 "\$self_sha" >/dev/null 2>&1 ); then
    echo "SKIP \$name: complete source file set verify failed (untracked file present, or manifest tampered)"; continue
  fi
  if [ "\$DRY_RUN" = "1" ]; then
    echo "WOULD-MARK \$name"
  else
    touch "\$d/\$MARKER" && echo "MARKED \$name" || echo "SKIP \$name: could not write marker"
  fi
done
REMOTE_EOF

STATUS=0
for h in "${NODES[@]}"; do
  echo "=== backfilling provisioned markers on $h ==="
  OUT=$(timeout 120 ssh "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=10 "$h" \
        "bash -s -- '$DRY_RUN'" <<<"$REMOTE_BACKFILL_SCRIPT" 2>&1) || {
    echo "  ⛔ SSH FAIL on $h (or the remote script itself failed) -- see below:" >&2
    STATUS=1
  }
  if [ -z "$OUT" ]; then
    echo "  (no legacy revisions needing a marker on $h)"
  else
    printf '%s\n' "$OUT" | sed "s/^/  /"   # SKIP lines are informational (e.g. a verify failure), never fail the run
  fi
done
exit $STATUS
