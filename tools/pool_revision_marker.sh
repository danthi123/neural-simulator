#!/usr/bin/env bash
# tools/pool_revision_marker.sh -- the ONE shared "has this pool revision directory COMPLETED provisioning"
# predicate (2026-09-23 fix round #3, re-review MEDIUM).
#
# THE DEFECT: pool_autodispatch.sh's revision_available() and pool_queue.sh's `add` remote-validity probe asked
# TWO DIFFERENT questions of the exact same revision directory. Fix round #2 made revision_available() require
# the `.provisioned_ok` completion marker (a bare `[ -d ... ]` passed for a half-provisioned dir, since
# pool_provision.sh's remote `mkdir -p` creates the directory FIRST, before rsync/venv/manifest-verify/sanity
# even run) -- but `pool_queue.sh add`'s own per-node probe was never updated and still asked only `[ -d ... ]`.
# Result: `add` could report NODE_OK and stage a revision-pinned job onto a node the dispatcher will then skip
# forever (the job silently stranded, distinct from -- but caused by the same root issue as -- the "legacy
# revision, no marker at all" case tools/pool_backfill_provisioned_markers.sh migrates).
#
# Source this file and call revision_marker_probe_cmd to get the exact remote shell predicate; both scripts run
# the SAME string over ssh so they can never drift into asking different questions again.
POOL_REVISION_MARKER_FILE=".provisioned_ok"

revision_marker_probe_cmd() {
  # revision_marker_probe_cmd <remote-dir-relative-to-\$HOME> -- prints the exact `[ -f ... ]` predicate string
  # to run over ssh on the node. e.g. revision_marker_probe_cmd "derisk-pool/revisions/$sha"
  printf '[ -f ~/%s/%s ]' "$1" "${POOL_REVISION_MARKER_FILE:-.provisioned_ok}"
}
