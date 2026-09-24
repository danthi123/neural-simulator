#!/usr/bin/env bash
# pool_autodispatch.sh — keep the mini-PC pool fed WITHOUT waiting for me to notice it went idle.
#
# WHY (2026-07-31, owner-flagged after the same lapse recurred seven times in one session). The cluster went from
# idle-all-night to working only after the owner pointed it out, and the check that was meant to catch it was
# first structurally blind to the cluster and then unable to fire at all. Once BOTH were fixed the pattern became:
# heartbeat fires -> I dispatch -> the job finishes in ~12 minutes -> the pool idles -> heartbeat fires again.
# Detection was solved; ANTICIPATION was not, so utilisation was capped by my response latency rather than by the
# hardware. Seven idle-pool alarms in one session, every one of them work that could have been queued in advance.
#
# THE FIX IS A QUEUE, not a faster reaction. Jobs are staged ahead of time; this loop hands the next one to
# whichever node is free. The mechanical part is in tools/workflow_check.sh, which now fails when the QUEUE IS
# EMPTY -- i.e. it alarms on "nothing staged", the actual defect, rather than on "pool idle", the symptom.
#
#   bash tools/pool_queue.sh add '<remote shell command>'     # stage work
#   nohup bash tools/pool_autodispatch.sh > /tmp/pool_dispatch.log 2>&1 &
#
# A job line is a command run on the node, from ~/derisk-pool/sim. Lines starting with # are ignored.
set -uo pipefail
ROOT=/home/dant123/Projects/sim
QUEUE="${POOL_QUEUE_PATH:-$ROOT/research/queue/pool.queue}"
CLAIMED="${POOL_RUNNING_PATH:-${QUEUE%.queue}.running}"
POLL="${POOL_DISPATCH_POLL:-60}"
# MEMORY RESERVATIONS (2026-09-23). The RSS snapshot node_is_idle reads is taken seconds after the previous launch,
# before that job has grown -- so one fill cycle handed pool42 SIX D6 workers (each grows to 4-5 GB on a 15 GB node)
# in 30 s, and an earlier cycle left pool41 with 167 MB free. Every dispatch now RESERVES its expected size on that
# node for POOL_GROWTH_WINDOW_S; a job declares its size with `mem_gb=N` anywhere in its line (e.g. in the
# --checked reason), else POOL_JOB_EST_GB. The reservation double-counts a job whose RSS has already caught up
# (MemAvailable fell AND it is still reserved) -- deliberately conservative: an under-filled node costs minutes, an
# OOM costs every job on the node plus a silent rc=0 from runners that swallow worker deaths.
RESV="${POOL_RESERVATIONS_PATH:-$ROOT/research/queue/.pool_reservations}"
GROWTH_WINDOW_S="${POOL_GROWTH_WINDOW_S:-600}"   # D6/LB workers reach full RSS in ~4 min (measured 2026-09-23)
# AWS-AS-EXTRA-POOL-NODE (2026-09-23, tools/aws_pool_node.sh). Two gitignored, machine-local files, never
# ~/.ssh/config (which this tooling must never edit):
#   .pool_ssh_config  -- `Include`s the user's own ~/.ssh/config, then adds Host entries for AWS pool nodes
#                         (HostName/User ubuntu/IdentityFile/StrictHostKeyChecking accept-new). ABSENT by
#                         default, so every ssh/rsync call below is BYTE-IDENTICAL to before this feature for
#                         anyone who has not run `aws_pool_node.sh up` -- existing pool40/41/42 behaviour is
#                         unchanged (`ssh -F <this file> poolNN` still resolves poolNN via the Included config).
#   .pool_extra_nodes -- one AWS node name per line (e.g. "pool1"), re-read EACH CYCLE below (not once at
#                         startup) so `aws_pool_node.sh up`/`down` can add/remove a node with no dispatcher
#                         restart -- the whole point of a systemd-managed singleton dispatcher.
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_F=(); [ -f "$POOL_SSH_CONFIG" ] && SSH_F=(-F "$POOL_SSH_CONFIG")
EXTRA_NODES_FILE="${POOL_EXTRA_NODES_FILE:-$ROOT/research/queue/.pool_extra_nodes}"

refresh_ssh_f() {
  # Re-evaluate SSH_F EVERY cycle, not once at process start (2026-09-23 fix round). This dispatcher runs as a
  # long-lived systemd singleton; the whole POINT of re-reading .pool_extra_nodes every cycle (below) is that
  # `aws_pool_node.sh up` can add a node with no dispatcher restart -- but `up` also CREATES .pool_ssh_config
  # for the first AWS node, and a SSH_F computed once at startup never noticed. Measured: the live dispatcher
  # (started before any AWS node existed) kept calling bare `ssh pool1` after `up` wired pool1 in -- pool1 has
  # no entry in the user's own ~/.ssh/config, so node_is_idle always failed and pool1 was never used while it
  # billed until aws_idle_stop stopped it.
  SSH_F=(); [ -f "$POOL_SSH_CONFIG" ] && SSH_F=(-F "$POOL_SSH_CONFIG")
}

extra_nodes() {
  [ -f "$EXTRA_NODES_FILE" ] && grep -vE '^[[:space:]]*(#|$)' "$EXTRA_NODES_FILE" 2>/dev/null | tr -s '[:space:]' ' '
}

revision_available() {
  # revision_available <node> <sha> -- does ~/derisk-pool/revisions/<sha> already exist on <node>? Used so a
  # revision-pinned job (`cd ~/derisk-pool/revisions/<sha> && ...`, from `pool_provision.sh --isolated`) is
  # never handed to a node that was never provisioned with that revision (2026-09-23 fix round: reproduced --
  # AWS node provisioned only at ~/derisk-pool/sim from HEAD, every queued job pinned to an isolated revision,
  # `cd` failed, the job was already popped from the queue and its result was never pulled -- silently lost).
  # Fails closed: unreachable/timeout/missing dir all return non-zero (job stays queued for another node/cycle).
  local node="$1" sha="$2"
  timeout 10 ssh "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$node" \
    "[ -d ~/derisk-pool/revisions/$sha ]" 2>/dev/null
}

job_est_gb() {
  local h
  h=$(printf '%s' "$1" | grep -oE 'mem_gb=[0-9]+' | head -1 | cut -d= -f2)
  # No hint but a memcap wrapper: its cap is the job's own declared ceiling (swap-probe LB lines, 2026-09-23).
  [ -z "$h" ] && h=$(printf '%s' "$1" | grep -oE 'memcap\.sh [0-9]+' | head -1 | awk '{print $2}')
  # ...else the runner's measured peak from tools/pool_runner_mem.tsv (an agent forgot the hint on 12 LB lines).
  if [ -z "$h" ]; then
    local mod; mod=$(printf '%s' "$1" | grep -oE -- '-m research\.runners\.[A-Za-z0-9_]+' | head -1 | sed 's/.*\.//')
    [ -n "$mod" ] && h=$(awk -F'\t' -v m="$mod" '$1==m {print $2; exit}' "${POOL_RUNNER_MEM_PATH:-$ROOT/tools/pool_runner_mem.tsv}" 2>/dev/null)
  fi
  echo "${h:-${POOL_JOB_EST_GB:-1}}"
}

committed_from_environ() {
  # stdin: unique "POOL_JOB_ID=<id> POOL_JOB_MEM_GB=<n>" lines (one per running job) -> total declared GB
  awk '{for(i=1;i<=NF;i++) if ($i ~ /^POOL_JOB_MEM_GB=/) {split($i,a,"="); s+=a[2]}} END{print s+0}'
}

reserved_gb() {
  awk -v n="$1" -v now="$(date +%s)" -v w="$GROWTH_WINDOW_S" \
      '$2==n && now-$1 < w {s+=$3} END{print s+0}' "$RESV" 2>/dev/null || echo 0
}

peek_est_gb() {
  # The size of the job pop_job would hand out next (same first-fresh-line rule), so a big job is not sent to a
  # node that only has room for a default-sized one.
  local cutoff job
  cutoff=$(( $(date +%s) - ${POOL_JOB_MAX_AGE:-43200} ))
  job=$(awk -F'\t' -v c="$cutoff" 'NF>1 && $1+0 >= c {print $2; exit}' "$QUEUE" 2>/dev/null)
  job_est_gb "$job"
}
NODES="${POOL_NODES:-pool40 pool41 pool42}"
NODE_BUDGET=0

mkdir -p "$(dirname "$QUEUE")"; touch "$QUEUE" "$CLAIMED"

node_is_idle() {
  # Has capacity = fewer than (cores - headroom) research runners AND load below its core count. The pool jobs are
  # single-threaded numpy (OPENBLAS_NUM_THREADS=1), so a 12-core node can run ~11 at once — the old "procs==0 AND
  # load<cores/4" gate treated each node as a SINGLE-job worker and wasted 11/12 cores (owner 2026-09-02: fill the
  # pool). Cap is overridable via POOL_JOBS_PER_NODE. Bracket the pgrep pattern: an un-bracketed one matches the
  # ssh command carrying it, the self-match that made an earlier check unable to ever fire.
  local out node="$1"   # `set -- $out` below overwrites $1 -- the first reservation check read the core count as the node
  # Line 1: metrics (+ MemTotal). Lines 2..: one per RUNNING dispatched job, from the POOL_JOB_ID/POOL_JOB_MEM_GB its
  # processes inherit (see remote_launch_command), so its DECLARED size counts for its whole lifetime (COMMITTED below).
  # (A ps-args scan was tried first and found nothing: bash execs the job's last command, so no wrapper stays visible.)
  # -F "${SSH_F[@]}" carries the AWS-pool-node repo-local ssh config when one exists (merge of the two fix rounds:
  # main's committed/lifetime-budget tracking + this branch's SSH_F routing), else it is empty and unchanged.
  local raw
  raw=$(timeout 12 ssh "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$node" \
        "echo \$(nproc) \$(cut -d' ' -f1 /proc/loadavg) \$(pgrep -c -f '^[^ ]*/?python[0-9.]* .*-m [r]esearch\.runners' 2>/dev/null | head -1) \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) \$(ps -eo rss,args | awk '\$2 ~ /python/ && /-m [r]esearch\\.runners/ {if (\$1>m) m=\$1} END{print int((m+1048575)/1048576)}') \$(awk '/MemTotal/{print int(\$2/1048576)}' /proc/meminfo); for e in /proc/[0-9]*/environ; do tr '\\0' '\\n' < \$e 2>/dev/null | grep -E '^POOL_JOB_(ID|MEM_GB)=' | sort | paste -sd' '; done | grep POOL_JOB_ID | sort -u || true" 2>/dev/null) || return 1
  out=$(printf '%s\n' "$raw" | head -1)
  local committed
  committed=$(printf '%s\n' "$raw" | tail -n +2 | committed_from_environ)
  set -- $out
  local cores="${1:-0}" load="${2:-99}" procs="${3:-99}" avail_gb="${4:-0}" max_job_gb="${5:-0}" total_gb="${6:-0}"
  # COUNT ONLY PYTHON RUNNERS (2026-09-23). The old `pgrep -fc research.runners` also counted every `flock`
  # waiter, `bash -c` wrapper and ssh line carrying the job text, so a node with 2 working jobs + 4 jobs queued
  # behind a lab-chosen flock read 11/11 "full" at load ~5.6 on 12 cores while 13 jobs sat queued. Load stays
  # the CPU gate; a MemAvailable floor (POOL_MIN_AVAIL_GB, default 3) now guards RAM (15 GB nodes).
  local cap="${POOL_JOBS_PER_NODE:-$(( cores > 2 ? cores - 1 : 1 ))}"   # single-threaded jobs: fill to cores-1
  [ "${procs:-99}" -lt "$cap" ] || return 1
  [ "${avail_gb:-0}" -ge "${POOL_MIN_AVAIL_GB:-3}" ] || return 1
  # Jobs GROW after dispatch (2026-09-23: three D6 workers reached 4-5 GB each and left pool41 with 167 MB free,
  # load 29/12) -- so also require headroom for one more job the size of the largest one already running there.
  [ "${avail_gb:-0}" -ge "${max_job_gb:-0}" ] || return 1
  # ...and room for the NEXT job on top of everything dispatched here within the growth window (see RESV above).
  local resv
  resv=$(reserved_gb "$node")
  NODE_BUDGET=$(( ${avail_gb:-0} - ${resv:-0} - ${POOL_MIN_AVAIL_GB:-3} ))   # GB pop_job may hand this node
  # COMMITTED (2026-09-23 20:45): a snapshot between an LB job's worker phases read 9 GB free on pool42 while two
  # swap-probe jobs (each peaking ~6 GB) were running; the growth-window reservations had expired, a third was sent,
  # and BOTH nodes thrashed until ssh timed out. So the budget is also capped by MemTotal minus an OS reserve minus
  # the declared size of every job still running there, for as long as it runs.
  local lifetime_budget=$(( ${total_gb:-0} - ${POOL_OS_RESERVE_GB:-2} - committed ))
  [ "${total_gb:-0}" -gt 0 ] && [ "$lifetime_budget" -lt "$NODE_BUDGET" ] && NODE_BUDGET=$lifetime_budget
  [ "$NODE_BUDGET" -ge 1 ] || return 1
  awk -v l="$load" -v c="$cores" 'BEGIN{exit !(l < c - 0.5)}'
}

remote_launch_command() {
  # Encode both the job and the fixed wrapper so shell metacharacters, tabs, and
  # newlines cannot corrupt the remote command or its one-line status record.
  local job_b64 wrapper wrapper_b64
  job_b64=$(printf '%s' "$1" | base64 -w0) || return 1
  wrapper='job=$(printf "%s" "$JOB_B64" | base64 -d) || exit 125
bash -c "$job" > autodispatch.out 2>&1
rc=$?
printf "v2\t%s\t%s\t%s\n" "$(date +%s)" "$rc" "$JOB_B64" >> job_status.log'
  wrapper_b64=$(printf '%s' "$wrapper" | base64 -w0) || return 1
  # POOL_JOB_ID / POOL_JOB_MEM_GB are INHERITED by every process of the job (they survive bash's exec of the last
  # command and memcap's scope), so node_is_idle can sum the declared sizes of the jobs still running there.
  local est jid
  est=$(job_est_gb "$1"); jid="$(date +%s%N)-$RANDOM"
  printf "cd ~/derisk-pool/sim && POOL_JOB_ID='%s' POOL_JOB_MEM_GB='%s' JOB_B64='%s' WRAPPER_B64='%s' setsid bash -c 'printf \"%%s\" \"\$WRAPPER_B64\" | base64 -d | bash' </dev/null >/dev/null 2>&1 & exit 0" \
    "$jid" "$est" "$job_b64" "$wrapper_b64"
}

pop_job() {
  # pop_job <max_gb> [node] -- atomically take the first non-comment line that fits <max_gb> AND (if [node] is
  # given and the candidate is revision-pinned) whose revision dir already exists on [node]. flock keeps two
  # dispatcher instances from claiming the same job. [node] is OPTIONAL and omitted by test seams that only
  # care about size-based selection (--pop-once) -- when absent, the revision check is skipped entirely
  # (unchanged pre-fix behaviour), matching every caller that never dispatches to a real node.
  local job="" node="${2:-}"
  exec 9>"$QUEUE.lock"
  flock 9 || return 1
  # A generic queue producer once wrote GPU-style command-only lines into this
  # timestamped pool queue. Monitoring counted them, while this consumer could
  # never select them. Preserve such records for diagnosis and remove them from
  # the live queue so malformed work cannot masquerade as work in transit.
  local malformed_count
  malformed_count=$(awk -F'\t' '
    $0 !~ /^[[:space:]]*(#|$)/ && !($1 ~ /^[0-9]+$/ && NF > 1) {n++}
    END {print n+0}
  ' "$QUEUE")
  if [ "${malformed_count:-0}" -gt 0 ]; then
    awk -F'\t' '
      $0 !~ /^[[:space:]]*(#|$)/ && !($1 ~ /^[0-9]+$/ && NF > 1)
    ' "$QUEUE" | while IFS= read -r line; do
      printf '%s\t%s\n' "$(date +%s)" "$line"
    done >> "$QUEUE.malformed"
    awk -F'\t' '
      $0 ~ /^[[:space:]]*(#|$)/ || ($1 ~ /^[0-9]+$/ && NF > 1)
    ' "$QUEUE" > "$QUEUE.tmp"
    mv "$QUEUE.tmp" "$QUEUE"
    echo "[pool-dispatch] BLOCKED + quarantined $malformed_count malformed queue record(s); use tools/pool_queue.sh" >&2
  fi
  # STALENESS GUARD: an entry older than MAX_AGE is debris, not staged work. Learned immediately -- the first
  # dispatcher run found 69 jobs from an opsweep abandoned days earlier and launched three of them.
  local now cutoff
  now=$(date +%s); cutoff=$(( now - ${POOL_JOB_MAX_AGE:-43200} ))
  # FIRST FIT, not strict head (2026-09-23): a node with 4 GB free sat idle behind a 5 GB D6 arm while five 0.65 GB
  # vision jobs queued behind it. Take the first fresh line whose declared size fits the node's budget ($1, GB).
  local max_gb="${1:-999}" cand sha
  while IFS= read -r cand; do
    [ "$(job_est_gb "$cand")" -le "$max_gb" ] || continue
    # REVISION-DIR SEAM (2026-09-23 fix round). A job pinned to `cd ~/derisk-pool/revisions/<sha>` must never be
    # popped for a node that does not have that revision -- that would remove it from the queue (below) with no
    # node able to run it, i.e. lose it. `continue` past it (leaving it in the queue) and keep scanning for a
    # candidate this node CAN run; if none exists this cycle, the outer `if [ -z "$job" ]` returns empty as usual.
    sha=$(printf '%s' "$cand" | grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' | head -1 | sed 's#.*/##')
    if [ -n "$sha" ] && [ -n "$node" ] && ! revision_available "$node" "$sha"; then continue; fi
    job="$cand"; break
  done < <(awk -F'\t' -v c="$cutoff" 'NF>1 && $1+0 >= c {print $2}' "$QUEUE")
  # THE RECORD-CHECK GATE (2026-07-31), copied from tools/lane_dispatch.sh:47 where it is already proven.
  # A job may only run if it carries "#checked:", which tools/pool_queue.sh only attaches when a reason is
  # given. This sits ON THE EXECUTION PATH deliberately: before_you_build.sh existed and was skipped, and that
  # skip cost ~94 GPU-hours re-deriving a NO-GO banked a week earlier. The failure-taxonomy pass found that the
  # ONLY two mechanisms which ever stopped a mistake uninvited are the two on paths you cannot avoid -- the
  # pre-commit hook and this gate. Unchecked lines are SET ASIDE, never silently dropped.
  case "$job" in
    ""|*"#checked:"*) ;;
    *) echo "[pool-dispatch] BLOCKED unchecked job -- requeue via: bash tools/pool_queue.sh add '<cmd>' --checked '<what the record says>'" >&2
       echo "                $(echo "$job" | cut -c1-96)" >&2
       grep -vF "	$job" "$QUEUE" > "$QUEUE.tmp" 2>/dev/null || true
       mv "$QUEUE.tmp" "$QUEUE"
       printf '%s\t%s\n' "$(date +%s)" "$job" >> "$QUEUE.unchecked"
       flock -u 9; printf ''; return 0 ;;
  esac
  local stale
  stale=$(awk -F'\t' -v c="$cutoff" 'NF>1 && $1+0 < c' "$QUEUE" | wc -l)
  [ "${stale:-0}" -gt 0 ] && echo "[pool-dispatch] SKIPPING $stale stale entr(ies) older than $(( ${POOL_JOB_MAX_AGE:-43200} / 3600 ))h" >&2
  if [ -z "$job" ]; then flock -u 9; printf ''; return 0; fi
  # Keep the full timestamped record, including the checked reason, before the
  # execution copy strips queue metadata. Artifact collection uses this claim
  # to reconstruct the exact command and rationale.
  printf '%s\t%s\n' "$(date +%s)" "$job" >> "$QUEUE.claims"
  if true; then
    grep -vF "	$job" "$QUEUE" > "$QUEUE.tmp" 2>/dev/null || true
    mv "$QUEUE.tmp" "$QUEUE"          # unconditional: grep -v exits 1 when it filters everything, and a
                                       # `&& mv` there once made a single-line queue never clear, relaunching
                                       # the same job nine times.
  fi
  flock -u 9
  # STRIP the trailing "#checked:<reason>" before the job is executed. The token is queue METADATA, not part of
  # the command. It survived the old wrapper because nothing followed $JOB on the line -- but the exit-status
  # wrapper puts the job inside a brace group, `{ $JOB; } > out`, and a `#` comments out the closing `; }`.
  # Result: an unterminated brace group, a syntax error, NO job run, and NO status line either (the printf lives
  # in the same bash -c). Six w0 jobs were dispatched into that and produced nothing; the dispatch log said
  # "dispatched" six times. Caught because the results never appeared AND job_status.log stayed empty -- the
  # exit-status capture failing was itself the clue that the wrapper, not the job, was broken.
  local checked_reason="${job#*#checked:}"
  job="${job%%#checked:*}"
  job=$(printf '%s' "$job" | sed 's/[[:space:]]*$//')
  job="POOL_CHECKED_REASON=$(printf '%q' "$checked_reason") $job"
  printf '%s' "$job"
}

if [ "${1:-}" = "--pop-once" ]; then
  pop_job "${2:-999}" "${3:-}"
  exit $?
fi
if [ "${1:-}" = "--reserved-gb" ]; then reserved_gb "$2"; exit 0; fi
if [ "${1:-}" = "--peek-est-gb" ]; then peek_est_gb; exit 0; fi
if [ "${1:-}" = "--node-budget" ]; then   # live diagnostic: would this node take work, and how many GB?
  if node_is_idle "$2"; then echo "idle budget=${NODE_BUDGET}GB"; else echo "busy/unreachable (budget=${NODE_BUDGET}GB)"; fi; exit 0
fi
if [ "${1:-}" = "--committed-gb" ]; then   # stdin: unique POOL_JOB_ID/POOL_JOB_MEM_GB lines from a node
  committed_from_environ; exit 0
fi
if [ "${1:-}" = "--render-remote-command" ]; then
  [ "$#" -eq 2 ] || { echo "usage: $0 --render-remote-command '<job>'" >&2; exit 2; }
  remote_launch_command "$2"
  exit $?
fi
if [ "${1:-}" = "--print-ssh-f-loop" ]; then
  # TEST SEAM (2026-09-23): proves refresh_ssh_f is called EVERY cycle of the real dispatch loop shape (not just
  # once at process start) without running the full dispatcher (no queue popping, no real ssh calls to a node).
  # Prints one line per cycle -- "F" if SSH_F currently carries -F<config>, else "NOF" -- so a test can create
  # POOL_SSH_CONFIG's file BETWEEN cycles and see the NEXT line flip, inside one long-lived process.
  [ "$#" -eq 3 ] || { echo "usage: $0 --print-ssh-f-loop <n-cycles> <sleep-s>" >&2; exit 2; }
  for _i in $(seq 1 "$2"); do
    refresh_ssh_f
    if [ "${#SSH_F[@]}" -gt 0 ]; then echo "F"; else echo "NOF"; fi
    sleep "$3"
  done
  exit 0
fi
if [ "${1:-}" = "--revision-available" ]; then
  # TEST SEAM (2026-09-23): exercises the REAL revision_available ssh call (same argv construction, including
  # SSH_F) against one node/sha pair, without a real node or a real revision -- so a stubbed `ssh` on PATH can
  # assert the exact `[ -d ~/derisk-pool/revisions/<sha> ]` probe it makes.
  [ "$#" -eq 3 ] || { echo "usage: $0 --revision-available <node> <sha>" >&2; exit 2; }
  revision_available "$2" "$3"; exit $?
fi
if [ "${1:-}" = "--node-idle" ]; then
  # TEST SEAM (2026-09-23): exercises the REAL node_is_idle ssh call (same argv construction, including
  # SSH_F) without running the dispatch loop, so a stubbed `ssh` on PATH can assert -F is/isn't present.
  [ "$#" -eq 2 ] || { echo "usage: $0 --node-idle <node>" >&2; exit 2; }
  node_is_idle "$2"; exit $?
fi
if [ "${1:-}" = "--nodes-this-cycle" ]; then
  # TEST SEAM: prints the node list ONE dispatch cycle would use -- POOL_NODES plus whatever
  # .pool_extra_nodes (or $POOL_EXTRA_NODES_FILE) currently names, re-read fresh on every call.
  printf '%s %s\n' "$NODES" "$(extra_nodes)" | tr -s ' ' | sed 's/^ *//; s/ *$//'
  exit 0
fi

# SINGLETON GUARD (2026-07-31): repeated restarts during testing left THREE dispatchers polling at once. flock
# in pop_job stops two of them claiming the same job, so correctness was safe -- but three pollers triple the ssh
# load on every node each cycle, and make "is the dispatcher up?" ambiguous, which now matters because
# workflow_check gates on exactly that. Refuse to start when one is already running.
# Exempt the systemd-managed run: systemd ALREADY guarantees one instance per service, so applying the guard
# there makes the two fight -- systemd restarts the unit, the fresh instance sees the outgoing one, exits 0, and
# Restart=always loops it forever (the unit sits in "activating" and never reaches active). INVOCATION_ID is set
# by systemd only. A MANUAL start still refuses, which is the case the guard was actually written for.
# Ask systemd, do not sniff the process table. Two earlier attempts got this wrong:
#   (1) a bare pgrep guard fought systemd -- the unit restarts, the fresh instance saw the outgoing one, exited 0,
#       and Restart=always looped it forever, leaving the unit stuck "activating";
#   (2) exempting systemd via INVOCATION_ID silently disabled the guard for EVERY manual start, because that
#       variable is set for any process under a systemd unit and is INHERITED by children -- Claude Code itself
#       runs under one, so the exemption always fired and a manual start happily launched a second dispatcher.
# MainPID is unambiguous: refuse only when the unit is active and this process is not the unit.
_MAIN=$(systemctl --user show -p MainPID --value pool-dispatch.service 2>/dev/null || echo 0)
if systemctl --user is-active --quiet pool-dispatch.service 2>/dev/null && [ "${_MAIN:-0}" != "$$" ]; then
  echo "[pool-dispatch] REFUSING to start: pool-dispatch.service is active (MainPID $_MAIN). Use: systemctl --user restart pool-dispatch.service"
  exit 0
fi

echo "[pool-dispatch] started $(date '+%H:%M:%S') | queue=$QUEUE | poll=${POLL}s | nodes=$NODES (+ any in $EXTRA_NODES_FILE, re-read each cycle)"
while true; do
  refresh_ssh_f
  CYCLE_NODES="$NODES $(extra_nodes)"
  for NODE in $CYCLE_NODES; do
    # FILL the node to capacity within this cycle (while, not if) — with the per-node cap raised for
    # single-threaded numpy jobs (owner 2026-09-02), a single if-per-cycle would need ~cap cycles to fill.
    while node_is_idle "$NODE"; do
      JOB=$(pop_job "$NODE_BUDGET" "$NODE")
      [ -z "$JOB" ] && break
      echo "[pool-dispatch] $(date '+%H:%M:%S') $NODE <- $JOB"
      printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$NODE" "$JOB" >> "$CLAIMED"
      # CAPTURE THE EXIT STATUS (2026-07-31). Previously this logged that a job was LAUNCHED and nothing more,
      # so a job that died was indistinguishable from one that succeeded. Nine jobs died instantly on an argparse
      # error and went unnoticed for an hour, because the only evidence of failure sat in autodispatch.out on a
      # node nobody reads. The wrapper appends a timestamped v2 record with a numeric rc and base64-encoded job;
      # the encoding keeps multiline pytest expressions from becoming fake status rows.
      REMOTE_COMMAND=$(remote_launch_command "$JOB") || {
        echo "[pool-dispatch] failed to encode job for $NODE" >&2
        break
      }
      ssh -f -n "${SSH_F[@]}" -o BatchMode=yes "$NODE" "$REMOTE_COMMAND" 2>/dev/null
      printf '%s %s %s\n' "$(date +%s)" "$NODE" "$(job_est_gb "$JOB")" >> "$RESV"
      sleep 5     # let the launch register before this node's next capacity check
    done
  done
  sleep "$POLL"
done
