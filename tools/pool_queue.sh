#!/usr/bin/env bash
# pool_queue.sh — stage work for the pool AHEAD of it going idle.
#   bash tools/pool_queue.sh add '<remote command>'   # enqueue (run from ~/derisk-pool/sim on the node)
#   bash tools/pool_queue.sh list                     # show depth + contents
#   bash tools/pool_queue.sh depth                    # just the number (used by workflow_check)
# The point is anticipation: workflow_check FAILS when this queue is EMPTY, because "nothing staged" is the
# actual defect. "Pool idle" is only its symptom, and alarming on the symptom caps utilisation at my reaction time.
set -uo pipefail
Q=/home/dant123/Projects/sim/research/queue/pool.queue
mkdir -p "$(dirname "$Q")"; touch "$Q"
case "${1:-list}" in
  add)   [ -n "${2:-}" ] || { echo "usage: pool_queue.sh add '<command>'" >&2; exit 2; }
         # TIMESTAMP every entry (2026-07-31). The first run of this queue reused a path that already held 69
         # STALE jobs from an opsweep stopped days earlier as live-but-stalled, and the dispatcher cheerfully
         # launched three of them on real nodes. An un-timestamped queue cannot tell staged-work from debris.
         printf '%s\t%s\n' "$(date +%s)" "$2" >> "$Q"
         echo "queued (depth now $(grep -cvE '^\s*(#|$)' "$Q"))" ;;
  depth) grep -cvE '^\s*(#|$)' "$Q" ;;
  list)  echo "depth: $(grep -cvE '^\s*(#|$)' "$Q")"
         awk -F'\t' -v now="$(date +%s)" 'NF>1 {printf "  %4.1fh old  %s\n", (now-$1)/3600, substr($2,1,110)}' "$Q" ;;
  *)     echo "usage: pool_queue.sh {add|list|depth}" >&2; exit 2 ;;
esac
