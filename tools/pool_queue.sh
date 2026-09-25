#!/usr/bin/env bash
# pool_queue.sh — stage work for the pool AHEAD of it going idle.
#   bash tools/pool_queue.sh add '<remote command>'   # enqueue (run from ~/derisk-pool/sim on the node)
#   bash tools/pool_queue.sh list                     # show depth + contents
#   bash tools/pool_queue.sh depth                    # just the number (used by workflow_check)
# The point is anticipation: workflow_check fails when this queue is EMPTY while a CPU-compatible lane is ready,
# because "nothing staged" is the actual defect in that state. When all such lanes are banked, the bounded
# no-ready-work waiver records why replaying old commands would be worse than leaving the queue empty.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=tools/pool_revision_marker.sh
source "$ROOT/tools/pool_revision_marker.sh"
Q="${POOL_QUEUE_PATH:-/home/dant123/Projects/sim/research/queue/pool.queue}"
mkdir -p "$(dirname "$Q")"; touch "$Q"
# AWS-AS-EXTRA-POOL-NODE (2026-09-23) -- same repo-local, gitignored ssh config as pool_autodispatch.sh /
# pool_provision.sh / pool_sync.sh (see pool_autodispatch.sh's header comment for the full rationale). ABSENT
# by default, so the reachability/argparse probe below is unchanged for anyone who hasn't run
# `aws_pool_node.sh up`. The probe's own node list also grows with .pool_extra_nodes, read fresh each call.
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_F=(); [ -f "$POOL_SSH_CONFIG" ] && SSH_F=(-F "$POOL_SSH_CONFIG")
EXTRA_NODES_FILE="${POOL_EXTRA_NODES_FILE:-$ROOT/research/queue/.pool_extra_nodes}"
probe_nodes() {
  local extra=""
  [ -f "$EXTRA_NODES_FILE" ] && extra=$(grep -vE '^[[:space:]]*(#|$)' "$EXTRA_NODES_FILE" 2>/dev/null | tr -s '[:space:]' ' ')
  printf '%s %s' "${POOL_NODES:-pool40 pool41 pool42}" "$extra"
}

if [ "${1:-}" = "--probe-node" ]; then
  # TEST SEAM (2026-09-23): exercises the EXACT reachability + --help ssh calls `add`'s remote-validity gate
  # makes (same flags, same use of SSH_F), against one node/module pair, without staging a real queue entry --
  # so a stubbed `ssh` on PATH can assert -F is/isn't present without a real pool node or a real runner module.
  [ "$#" -eq 3 ] || { echo "usage: $0 --probe-node <node> <module>" >&2; exit 2; }
  n="$2"; MOD="$3"
  if ! timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$n" true >/dev/null 2>&1; then
    echo "UNREACHABLE"; exit 1
  fi
  if timeout 25 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=8 "$n" \
       "cd ~/derisk-pool/sim && SIM_NO_PROVENANCE=1 SIM_BACKEND=numpy .venv/bin/python -m $MOD --help" \
       >/dev/null 2>&1; then echo OK; else echo BAD; fi
  exit 0
fi

valid_depth() {
  awk -F'\t' '$1 ~ /^[0-9]+$/ && NF > 1 {n++} END {print n+0}' "$Q"
}

malformed_depth() {
  awk -F'\t' '
    $0 !~ /^[[:space:]]*(#|$)/ && !($1 ~ /^[0-9]+$/ && NF > 1) {n++}
    END {print n+0}
  ' "$Q"
}

strip_checked_reason_prefix() {
  # strip_checked_reason_prefix <pool.running JOB field> -- pool_autodispatch.sh's pop_job() prepends
  # "POOL_CHECKED_REASON=$(printf '%q' "$checked_reason") " to every job before it is either recorded
  # in pool.running or dispatched (see pop_job, tools/pool_autodispatch.sh). We only need the BOUNDARY
  # of that token to recover the underlying command -- not its decoded content -- so no unescaping is
  # attempted; this just skips past it and prints what remains. Handles both forms bash's %q emits: a
  # backslash-escaped token (the common case -- %q backslash-escapes space/$/'/"/(/)/;/etc. but leaves
  # comma/colon bare, e.g. a reason like "D6 N=2000 OOM root-cause found: ... (2026-09-23 11:33, ...)")
  # and a $'...' ANSI-C-quoted token (%q's fallback when the value holds control characters). A field
  # with no such prefix at all is echoed back unchanged.
  local s="$1" rest
  case "$s" in
    "POOL_CHECKED_REASON="*) s="${s#POOL_CHECKED_REASON=}" ;;
    *) printf '%s' "$s"; return 0 ;;
  esac
  case "$s" in
    \$\'*)
      rest="${s#\$\'}"
      while [ -n "$rest" ]; do
        case "$rest" in
          \\*) rest="${rest:2}" ;;             # an escaped pair inside the $'...' body -- keep both, move on
          \'*) rest="${rest:1}"; break ;;      # the unescaped closing quote -- token ends here
          *) rest="${rest:1}" ;;
        esac
      done
      s="$rest"
      ;;
    *)
      rest="$s"
      while [ -n "$rest" ]; do
        case "$rest" in
          \\*) rest="${rest:2}" ;;             # backslash-escaped char (incl. an escaped space) -- part of the token
          ' '*) break ;;                       # first UNescaped space -- the token/job separator
          *) rest="${rest:1}" ;;
        esac
      done
      s="$rest"
      ;;
  esac
  while :; do   # drop the separating space(s) left before the job command
    case "$s" in
      ' '*) s="${s:1}" ;;
      *) break ;;
    esac
  done
  printf '%s' "$s"
}

job_liveness_on_node() {
  # job_liveness_on_node <node> <exact pool.running JOB field text> -- echoes ALIVE, DEAD or UNREACH.
  # The text is exactly what pool_autodispatch.sh:remote_launch_command base64-encoded into JOB_B64 for
  # that dispatch (pop_job's output, byte-for-byte -- fill_node passes the identical string to both the
  # CLAIMED/pool.running record and remote_launch_command), so a still-running process of that exact
  # claim carries `JOB_B64=<same b64>` in its /proc/<pid>/environ for its whole lifetime (inherited
  # through remote_launch_command's `setsid bash -c`). Same ssh/-F/timeout conventions as --probe-node.
  local node="$1" job_text="$2" b64
  b64=$(printf '%s' "$job_text" | base64 -w0)
  if ! timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$node" true >/dev/null 2>&1; then
    echo UNREACH; return 0
  fi
  if timeout 20 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=8 "$node" \
       "for e in /proc/[0-9]*/environ; do tr '\\0' '\\n' < \$e 2>/dev/null; done | grep -qxF 'JOB_B64=$b64'" \
       >/dev/null 2>&1; then
    echo ALIVE
  else
    echo DEAD
  fi
}

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
         # PINNED-REVISION FLAG CHECK (2026-09-24). A job pinned to `cd ~/derisk-pool/revisions/<sha>` runs THAT revision's
         # runner, which main's checkout may not have yet (an unmerged branch's 6-seed run): checking its flags against
         # main refused every such job ("does not even import/parse"). For pinned jobs the flag check runs below,
         # against the pinned revision's own --help on the node; unpinned jobs keep this local check.
         PINNED_REV=$(printf '%s' "$2" | grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' | head -1)
         FLAGS=$(printf '%s' "$2" | grep -oE '[-][-][a-z][a-z0-9-]*' | sort -u)
         if [ -n "$MOD" ] && [ -z "$PINNED_REV" ]; then
           HELP=$(cd "$ROOT" && SIM_NO_PROVENANCE=1 timeout 90 .venv/bin/python -m "$MOD" --help 2>&1)
           # Here-strings, not `printf | grep -q` (2026-09-24): under `set -o pipefail`, grep -q exits on the first
           # match, printf takes SIGPIPE on a help text larger than the 64 KB pipe buffer, and the pipeline FAILS --
           # a 72 KB --help reported present flags as missing, a different random subset on every call.
           if [ $? -ne 0 ] && ! grep -q "usage:" <<<"$HELP"; then
             echo "⛔ REFUSED: $MOD does not even import/parse. Fix it before queueing." >&2
             printf '%s\n' "$HELP" | tail -5 >&2; exit 2
           fi
           BAD=""
           for f in $FLAGS; do grep -q -- "$f" <<<"$HELP" || BAD="$BAD $f"; done
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
         # SINGLE-NODE RESILIENCE (2026-09-03). The old form required the runner to --help on ALL of
         # pool40/41/42 and refused otherwise -- so ONE unreachable node stranded the WHOLE pool (pool40
         # offline "No route to host" refused every add while pool41/42 sat idle with 24 free cores). But
         # the dispatcher (pool_autodispatch.sh: node_is_idle) already ssh-health-checks each node every
         # cycle and SKIPS unreachable ones, so an unreachable node is NEVER a dispatch target -- gating
         # staging on it is redundant AND fragile. Fix: probe reachability first, SKIP unreachable nodes
         # (dispatcher skips them too), and only refuse when a REACHABLE node lacks the runner (the real
         # integration-seam check, preserved) or when NO node is reachable at all.
         if [ -n "$MOD" ]; then
           NODE_BAD=""; NODE_OK=""; NODE_UNREACH=""; NODE_SKIP=""
           # ISOLATED-REVISION SEAM (2026-09-23). A job pinned to `cd ~/derisk-pool/revisions/<sha> && ...` (the
           # `pool_provision.sh --isolated` layout) RUNS in that revision dir, but this check probed the shared
           # ~/derisk-pool/sim copy -- so a NEW runner that exists only in the isolated revision was refused, and a
           # runner that exists in the shared copy but NOT in the pinned revision was wrongly accepted. Probe the
           # directory the job will actually run in.
           REMOTE_DIR=$(printf '%s' "$2" | grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' | head -1)
           IS_REVISION=0; [ -n "$REMOTE_DIR" ] && IS_REVISION=1
           REMOTE_DIR="${REMOTE_DIR:-derisk-pool/sim}"
           for n in $(probe_nodes); do
             if ! timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$n" true >/dev/null 2>&1; then
               NODE_UNREACH="$NODE_UNREACH $n"; continue
             fi
             # MISSING-REVISION-DIR IS "SKIP", NOT "BAD" (2026-09-23 fix round). A reachable node that simply has
             # not been provisioned with THIS revision yet (e.g. a freshly-`up`'d AWS pool node, before any
             # `--isolated --revision <sha>` provision has targeted it) is not a broken node -- it is a node this
             # PARTICULAR job cannot use yet. Counting it as NODE_BAD wrongly REFUSED staging revision-pinned work
             # for every OTHER (perfectly capable) node too, because the refusal fires on "any reachable+bad node"
             # regardless of whether other reachable nodes are fine. Reproduced: registering one AWS node with only
             # ~/derisk-pool/sim provisioned made `add` refuse ALL revision-pinned adds, including ones pool40/41/42
             # could already run.
             #
             # SHARED PREDICATE (2026-09-23 fix round #3, re-review MEDIUM): this used to ask `[ -d ~/$REMOTE_DIR ]`
             # -- bare directory existence -- while pool_autodispatch.sh's revision_available() (the check that
             # actually decides whether the dispatcher will EVER hand this job to this node) requires the
             # `.provisioned_ok` completion marker. A half-provisioned dir (pool_provision.sh's remote `mkdir -p`
             # creates it FIRST, before rsync/venv/manifest-verify/sanity even run) or a LEGACY dir predating that
             # marker therefore passed `add` and got the job staged onto a node the dispatcher would then skip
             # forever -- the job silently stranded. Both scripts now call the SAME
             # tools/pool_revision_marker.sh:revision_marker_probe_cmd so they can never ask two different
             # questions of the same directory again.
             if [ "$IS_REVISION" = 1 ] && ! timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$n" \
                  "$(revision_marker_probe_cmd "$REMOTE_DIR")" >/dev/null 2>&1; then
               NODE_SKIP="$NODE_SKIP $n"; continue
             fi
             if RHELP=$(timeout 60 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=8 "$n" \
                  "cd ~/$REMOTE_DIR && SIM_NO_PROVENANCE=1 SIM_BACKEND=numpy .venv/bin/python -m $MOD --help" 2>/dev/null); then
               NODE_OK="$NODE_OK $n"; [ -z "${REMOTE_HELP:-}" ] && REMOTE_HELP="$RHELP"
             else NODE_BAD="$NODE_BAD $n"; fi
           done
           if [ "$IS_REVISION" = 1 ] && [ -n "${REMOTE_HELP:-}" ]; then
             BAD=""
             for f in $FLAGS; do grep -q -- "$f" <<<"$REMOTE_HELP" || BAD="$BAD $f"; done
             if [ -n "$BAD" ]; then
               echo "⛔ REFUSED: $MOD at the pinned revision does not accept:$BAD" >&2
               echo "   The job would be dispatched, die on argparse, and free the node silently." >&2
               exit 2
             fi
           fi
           [ -n "$NODE_UNREACH" ] && echo "ℹ️  skipping unreachable node(s):$NODE_UNREACH (dispatcher health-checks + skips them too)" >&2
           [ -n "$NODE_SKIP" ] && echo "ℹ️  skipping node(s) not yet provisioned with this revision:$NODE_SKIP (the dispatcher skips them for this job too, until provisioned)" >&2
           if [ -n "$NODE_BAD" ]; then
             echo "⛔ REFUSED: $MOD is not runnable on REACHABLE dispatch target(s):$NODE_BAD" >&2
             echo "   Those nodes are UP but the runner fails there (stale rsync?); synchronize them:" >&2
             echo "   bash tools/pool_provision.sh$NODE_BAD" >&2
             exit 2
           fi
           if [ -z "$NODE_OK" ]; then
             echo "⛔ REFUSED: no reachable pool node can run $MOD (unreachable:$NODE_UNREACH)." >&2
             echo "   Restore a node before queueing (pool health: .venv/bin/python tools/pool_health.py)." >&2
             exit 2
           fi
         fi
         # DUPLICATE GUARD (2026-07-31). Identical commands were dispatched repeatedly while other lanes sat
         # unserved: _b1_v1_selforg_onbridge_derisk --seeds 42 43 44 ran on pool41 AND pool42 concurrently,
         # _emerge72_construction_registry_derisk went out 4x and _self_schema_region_derisk 3x inside 90
         # minutes. A full queue and a busy pool looked like good utilisation and were re-deriving one result.
         # Compare the COMMAND only (minus queue metadata) against both the queue and the running set, so a
         # re-run must be made deliberate rather than happening by accident.
         #
         # THE RUNNING-SET HALF NEVER MATCHED (bug measured 2026-09-25 07:20). A pool.running record is
         # `date<TAB>node<TAB>job` (fill_node/CLAIMED, tools/pool_autodispatch.sh), so `cut -f2-` on it
         # yields "<node>\t<job>", never a bare command -- and the job field itself carries a
         # `POOL_CHECKED_REASON=<%q token> ` prefix the dispatcher's pop_job() adds, which the queue-only
         # "#checked:" stripping never touches. The comparison was therefore comparing NEW_CMD against
         # "<node>\t...POOL_CHECKED_REASON=...", which cannot ever be equal. Consequence: all 18 D6 N=2000
         # cells were re-queued at 23:41-00:05 while the first dispatch (~05:00, same revision) was still
         # alive on pool41/pool42; 7 duplicates ran 7-26h on 15 GB mini-PCs before being stopped by hand.
         #
         # FIX: parse the record correctly (strip_checked_reason_prefix, above) AND never trust a text match
         # alone -- a stopped/crashed job leaves its pool.running line behind (nothing here retires it), so a
         # text match is only a CANDIDATE; job_liveness_on_node (above) verifies against the claimed node's
         # own /proc/*/environ before refusing. The queue-side comparison carries no such staleness problem
         # (a queued line is retired the moment it is popped) and is left exactly as it was.
         NEW_CMD=$(printf '%s' "$2" | tr -s ' ')
         DUP=""
         if [ -f "$Q" ]; then
           while IFS= read -r line; do
             existing=$(printf '%s' "$line" | cut -f2- | sed 's/  #checked:.*//' | tr -s ' ')
             [ "$existing" = "$NEW_CMD" ] && DUP="$Q"
           done < "$Q"
         fi
         if [ -n "$DUP" ] && [ "${FORCE_DUP:-0}" != "1" ]; then
           echo "⛔ REFUSED: this exact command is already queued (in $(basename "$DUP"))." >&2
           echo "   Re-running an identical job produces an identical result and starves another lane." >&2
           echo "   If the repeat is deliberate (a genuine replication), re-run with FORCE_DUP=1." >&2
           exit 2
         fi
         RUNNING_FILE="${Q%.queue}.running"
         RUN_ALIVE_NODE=""; RUN_ALIVE_DATE=""; RUN_UNREACH_NODE=""; RUN_DEAD_NODE=""; RUN_DEAD_DATE=""
         if [ -f "$RUNNING_FILE" ]; then
           declare -A _RQ_LIVENESS=()   # node -> ALIVE|DEAD|UNREACH; one ssh per distinct node even with several matches
           while IFS=$'\t' read -r rdate rnode rjob; do
             [ -n "$rnode" ] && [ -n "$rjob" ] || continue
             norm=$(strip_checked_reason_prefix "$rjob" | tr -s ' ')
             [ "$norm" = "$NEW_CMD" ] || continue
             if [ -z "${_RQ_LIVENESS[$rnode]+x}" ]; then
               _RQ_LIVENESS[$rnode]=$(job_liveness_on_node "$rnode" "$rjob")
             fi
             case "${_RQ_LIVENESS[$rnode]}" in
               ALIVE) RUN_ALIVE_NODE="$rnode"; RUN_ALIVE_DATE="$rdate" ;;
               UNREACH) [ -n "$RUN_UNREACH_NODE" ] || RUN_UNREACH_NODE="$rnode" ;;
               DEAD)    [ -n "$RUN_DEAD_NODE" ] || { RUN_DEAD_NODE="$rnode"; RUN_DEAD_DATE="$rdate"; } ;;
             esac
           done < "$RUNNING_FILE"
         fi
         if [ -n "$RUN_ALIVE_NODE" ]; then
           echo "⛔ REFUSED: this exact command is already RUNNING on $RUN_ALIVE_NODE since $RUN_ALIVE_DATE; FORCE_DUP=1 only for a deliberate replication." >&2
           [ "${FORCE_DUP:-0}" = "1" ] || exit 2
         elif [ -n "$RUN_UNREACH_NODE" ]; then
           echo "⛔ REFUSED: a matching claim exists on $RUN_UNREACH_NODE but it could not be reached to verify liveness; failing closed. FORCE_DUP=1 to override." >&2
           [ "${FORCE_DUP:-0}" = "1" ] || exit 2
         elif [ -n "$RUN_DEAD_NODE" ]; then
           echo "ℹ️  a previous identical claim exists ($RUN_DEAD_NODE, $RUN_DEAD_DATE) but is no longer alive -- queueing." >&2
         fi
         # APPEND UNDER THE DISPATCHER'S LOCK (2026-09-24). pop_job rewrites the queue (awk > tmp; mv) under
         # "$Q.lock"; an unlocked append landing between its read and its mv went to the replaced inode and was lost.
         # FRONT=1 (2026-09-24): put the line at the HEAD of the queue (short, latency-critical checks ahead of a large
         # battery). Same lock, same line format; the dispatcher's first-fit scan still applies.
         ( flock -w 120 9 || { echo "⛔ could not take $Q.lock in 120 s" >&2; exit 1; }
           if [ "${FRONT:-0}" = "1" ]; then
             { printf '%s\t%s  #checked:%s\n' "$(date +%s)" "$2" "$CHECKED"; cat "$Q" 2>/dev/null; } > "$Q.front.tmp" && mv "$Q.front.tmp" "$Q"
           else
             printf '%s\t%s  #checked:%s\n' "$(date +%s)" "$2" "$CHECKED" >> "$Q"
           fi ) 9>"$Q.lock" || exit 1
         echo "queued (depth now $(valid_depth))" ;;
  depth) valid_depth ;;
  malformed-depth) malformed_depth ;;
  list)  echo "depth: $(valid_depth)"
         BAD=$(malformed_depth)
         [ "$BAD" -eq 0 ] || echo "malformed: $BAD (will be quarantined by dispatcher)"
         awk -F'\t' -v now="$(date +%s)" 'NF>1 {printf "  %4.1fh old  %s\n", (now-$1)/3600, substr($2,1,110)}' "$Q" ;;
  *)     echo "usage: pool_queue.sh {add|list|depth|malformed-depth}" >&2; exit 2 ;;
esac
