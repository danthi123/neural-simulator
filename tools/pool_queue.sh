#!/usr/bin/env bash
# pool_queue.sh — stage work for the pool AHEAD of it going idle.
#   bash tools/pool_queue.sh add '<remote command>'   # enqueue (run from ~/derisk-pool/sim on the node)
#   bash tools/pool_queue.sh list                     # show depth + contents
#   bash tools/pool_queue.sh depth                    # just the number (used by workflow_check)
# The point is anticipation: workflow_check FAILS when this queue is EMPTY, because "nothing staged" is the
# actual defect. "Pool idle" is only its symptom, and alarming on the symptom caps utilisation at my reaction time.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
Q=/home/dant123/Projects/sim/research/queue/pool.queue
mkdir -p "$(dirname "$Q")"; touch "$Q"
case "${1:-list}" in
  add)   [ -n "${2:-}" ] || { echo "usage: pool_queue.sh add '<command>' --checked '<what the record says>'" >&2; exit 2; }
         # THE RECORD-CHECK GATE. --checked forces a sentence about what the existing record says BEFORE compute
         # is spent. It is not a formality: ~94 GPU-hours went on re-deriving a NO-GO banked a week earlier, and
         # `before_you_build.sh` -- which exists to catch exactly that -- was simply not run, because running it
         # was a thing to remember. The dispatcher refuses any line lacking the resulting "#checked:" token, so
         # this cannot be skipped by queueing directly.
         CHECKED=""
         if [ "${3:-}" = "--checked" ] && [ -n "${4:-}" ]; then CHECKED="$4"; fi
         if [ -z "$CHECKED" ]; then
           echo "⛔ REFUSED: --checked '<what the record already says about this>' is required." >&2
           echo "   Run first:  bash tools/before_you_build.sh \"<the defect/question>\"" >&2
           echo "   Then:       bash tools/pool_queue.sh add '<cmd>' --checked 'corpus: nothing covers laps x dwell at w_max>W0'" >&2
           exit 2
         fi
         # TIMESTAMP every entry (2026-07-31). The first run of this queue reused a path that already held 69
         # STALE jobs from an opsweep stopped days earlier as live-but-stalled, and the dispatcher cheerfully
         # launched three of them on real nodes. An un-timestamped queue cannot tell staged-work from debris.
         # COMMAND VALIDITY GATE (2026-07-31). Nine drive-axis jobs were queued, dispatched, and died instantly
         # on `error: unrecognized arguments: --drive 8000` -- the knob existed in run() but never on the CLI.
         # Nothing noticed for an hour, because the dispatcher reports LAUNCHING a job, not its exit status, and
         # I was checking whether RESULTS landed rather than whether jobs SUCCEEDED. The --checked gate had made
         # me state what the record says; it never asked whether the command could run at all.
         # Cheap and total: ask the runner's own argparse. --help exits 0 iff the module imports and parses.
         MOD=$(printf '%s' "$2" | grep -oE '\-m +research\.runners\.[A-Za-z0-9_]+' | awk '{print $2}' | head -1)
         # INTERPRETER GUARD (2026-08-01). The pool nodes have NO bare `python` -- only `.venv/bin/python`. A
         # command running `-m research...` via bare `python` passes EVERY check below (they all shell out to
         # `.venv/bin/python` themselves) yet dispatches and produces NOTHING on the node. Measured: the 42-job
         # brain-quench sweep AND the first affect sweep both staged with bare `python` and silently produced
         # zero output -- validated, dispatched, dead. The check that ran the module and the command that ran on
         # the node used different interpreters; this closes that seam.
         if [ -n "$MOD" ]; then case "$2" in
             *".venv/bin/python"*) ;;                                     # sanctioned interpreter -- ok
             *) echo "⛔ REFUSED: '$MOD' would run via a BARE 'python' -- pool nodes have none (silent no-output)." >&2
                echo "   Use: SIM_BACKEND=numpy .venv/bin/python -u -m $MOD ..." >&2; exit 2 ;;
           esac; fi
         if [ -n "$MOD" ]; then
           FLAGS=$(printf '%s' "$2" | grep -oE '[-][-][a-z][a-z0-9-]*' | sort -u)
           HELP=$(cd "$ROOT" && timeout 90 .venv/bin/python -m "$MOD" --help 2>&1)
           if [ $? -ne 0 ] && ! printf '%s' "$HELP" | grep -q "usage:"; then
             echo "⛔ REFUSED: $MOD does not even import/parse. Fix it before queueing." >&2
             printf '%s\n' "$HELP" | tail -5 >&2; exit 2
           fi
           BAD=""
           for f in $FLAGS; do printf '%s' "$HELP" | grep -q -- "$f" || BAD="$BAD $f"; done
           if [ -n "$BAD" ]; then
             echo "⛔ REFUSED: $MOD does not accept:$BAD" >&2
             echo "   The job would be dispatched, die on argparse, and free the node silently." >&2
             echo "   Accepted flags:" >&2
             printf '%s\n' "$HELP" | grep -oE '[-][-][a-z][a-z0-9-]*' | sort -u | tr '\n' ' ' | sed 's/^/     /' >&2
             echo >&2; exit 2
           fi
         fi
         # REMOTE VALIDITY (2026-07-31). The argparse gate above validates against the LOCAL repo, but the
         # job RUNS ON A NODE — and the nodes hold an rsync'd COPY, not a git checkout, so a brand-new
         # runner passes every local check and dies instantly with "No module named ...". That is exactly
         # the failure the argparse gate was built for, one layer up: an INTEGRATION SEAM between the
         # validator's world and the executor's world. Measured: _affect_eviction_derisk was queued, gated,
         # dispatched, and exited rc=1 on all three nodes.
         if [ -n "$MOD" ]; then
           NODE_OK=""
           for n in pool40 pool41 pool42; do
             if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 "$n" \
                  "cd ~/derisk-pool/sim && SIM_BACKEND=numpy .venv/bin/python -m $MOD --help" \
                  >/dev/null 2>&1; then NODE_OK="$n"; break; fi
           done
           if [ -z "$NODE_OK" ]; then
             echo "⛔ REFUSED: $MOD imports LOCALLY but on NO reachable pool node." >&2
             echo "   The nodes carry an rsync'd copy, not a checkout — a new runner must be shipped first:" >&2
             echo "   for n in pool40 pool41 pool42; do rsync -az --include='*/' --include='*.py' --exclude='*' \\" >&2
             echo "       research/runners/ \$n:~/derisk-pool/sim/research/runners/; done" >&2
             exit 2
           fi
         fi
         # DUPLICATE GUARD (2026-07-31). Identical commands were dispatched repeatedly while other lanes sat
         # unserved: _b1_v1_selforg_onbridge_derisk --seeds 42 43 44 ran on pool41 AND pool42 concurrently,
         # _emerge72_construction_registry_derisk went out 4x and _self_schema_region_derisk 3x inside 90
         # minutes. A full queue and a busy pool looked like good utilisation and were re-deriving one result.
         # Compare the COMMAND only (field 2, minus the #checked: annotation) against both the queue and the
         # running set, so a re-run must be made deliberate rather than happening by accident.
         NEW_CMD=$(printf '%s' "$2" | tr -s ' ')
         DUP=""
         for f in "$Q" "${Q%.queue}.running"; do
           [ -f "$f" ] || continue
           while IFS= read -r line; do
             existing=$(printf '%s' "$line" | cut -f2- | sed 's/  #checked:.*//' | tr -s ' ')
             [ "$existing" = "$NEW_CMD" ] && DUP="$f"
           done < "$f"
         done
         if [ -n "$DUP" ] && [ "${FORCE_DUP:-0}" != "1" ]; then
           echo "⛔ REFUSED: this exact command is already queued or running (in $(basename "$DUP"))." >&2
           echo "   Re-running an identical job produces an identical result and starves another lane." >&2
           echo "   If the repeat is deliberate (a genuine replication), re-run with FORCE_DUP=1." >&2
           exit 2
         fi
         printf '%s\t%s  #checked:%s\n' "$(date +%s)" "$2" "$CHECKED" >> "$Q"
         echo "queued (depth now $(grep -cvE '^\s*(#|$)' "$Q"))" ;;
  depth) grep -cvE '^\s*(#|$)' "$Q" ;;
  list)  echo "depth: $(grep -cvE '^\s*(#|$)' "$Q")"
         awk -F'\t' -v now="$(date +%s)" 'NF>1 {printf "  %4.1fh old  %s\n", (now-$1)/3600, substr($2,1,110)}' "$Q" ;;
  *)     echo "usage: pool_queue.sh {add|list|depth}" >&2; exit 2 ;;
esac
