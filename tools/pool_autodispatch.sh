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
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's OWN tools/ dir (unlike $ROOT above,
# this follows whichever checkout/worktree is actually running -- used below so the stale-HostName refresh
# invokes the SAME checkout's aws_pool_node.sh, not always the one at the hardcoded $ROOT.
# shellcheck source=tools/pool_revision_marker.sh
source "$SELF_DIR/pool_revision_marker.sh"
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
# STALE-HOSTNAME AUTO-REFRESH, DISPATCHER SIDE (2026-09-25 review, LOW: "spec gap"). tools/pool_sync.sh already
# self-heals a stale ip (research/queue/.aws_<node> exists -> one `aws_pool_node.sh refresh <node>` + retry, see
# its own header), but THIS dispatcher's own node_is_idle probe never did, even though the spec says every entry
# point that finds a stale HostName should -- an AWS pool node whose ip changed (every stop/start, no Elastic IP
# in this feature) between routine syncs stayed unreachable-to-dispatch until the NEXT pool_sync cadence noticed.
# Overridable so tests never touch the shared production dir/marks.
AWS_STATE_DIR_FOR_REFRESH="${POOL_AWS_STATE_DIR:-$ROOT/research/queue}"
STALE_REFRESH_MARK_DIR="${POOL_STALE_REFRESH_MARK_DIR:-$ROOT/research/queue/.pool_stale_refresh}"
STALE_REFRESH_RATE_S="${POOL_STALE_REFRESH_RATE_S:-300}"   # at most one refresh attempt per node per 5 min --
# node_is_idle runs on every fill_node poll (POOL_DISPATCH_POLL, default 60s) for every configured node, so an
# UN-rate-limited refresh would shell out to `aws describe-instances`/`aws ec2` on every single cycle for any
# node that stays unreachable for a mundane reason (genuinely stopped, network blip) -- rate-limiting keeps this
# self-heal cheap while still resolving a stale ip well within one routine pool_sync cadence (15 min).

_maybe_refresh_stale_aws_node() {   # _maybe_refresh_stale_aws_node <node> -- called when node_is_idle's ssh
  # probe fails for a node tools/aws_pool_node.sh manages (a research/queue/.aws_<node> state file exists). Best-
  # effort, silent on failure (this is a self-heal, not a correctness gate -- node_is_idle already returns 1
  # either way, so a failed refresh attempt changes nothing about THIS cycle's dispatch decision).
  local node="$1" state="$AWS_STATE_DIR_FOR_REFRESH/.aws_$node" mark="$STALE_REFRESH_MARK_DIR/$node"
  [ -f "$state" ] || return 0                                  # not an AWS-managed node -- nothing to refresh
  grep -q '^# TORN DOWN' "$state" 2>/dev/null && return 0      # torn down -- refreshing a gone node is pointless
  mkdir -p "$STALE_REFRESH_MARK_DIR" 2>/dev/null
  if [ -f "$mark" ]; then
    local last age
    last=$(stat -c %Y "$mark" 2>/dev/null || stat -f %m "$mark" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - last ))
    [ "$age" -lt "$STALE_REFRESH_RATE_S" ] && return 0         # rate-limited -- refreshed too recently
  fi
  touch "$mark" 2>/dev/null
  # timeout 30 (2026-09-25 review, LOW/INFO): this runs INSIDE node_is_idle, on the MAIN dispatch loop, once per
  # unreachable AWS-managed node per cycle -- a hung AWS API call inside `refresh` (describe-instances) had no
  # bound at all and could stall dispatch for every OTHER node indefinitely. 30s comfortably covers `refresh`'s
  # own normal (sub-few-second) runtime; a timed-out refresh changes nothing about THIS cycle's dispatch
  # decision either way (best-effort/silent, exactly like any other refresh failure here).
  AWS_POOL_NODE_STATE_FILE="$state" POOL_SSH_CONFIG="$POOL_SSH_CONFIG" \
    timeout 30 bash "$SELF_DIR/aws_pool_node.sh" refresh "$node" >>"${POOL_STALE_REFRESH_LOG:-/dev/null}" 2>&1
}

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

cycle_setup() {
  # ONE-FUNCTION FIX (2026-09-23 fix round #2, re-review MEDIUM): the OLD `--print-ssh-f-loop` test seam called
  # `refresh_ssh_f` directly, inline -- a COPY of the production loop's shape, not the production loop itself.
  # Mutation check confirmed the gap: deleting the real `while true` loop's `refresh_ssh_f` call left every test
  # passing except the pre-existing unrelated failure. `cycle_setup` is now the ONE place either the production
  # loop or a test calls -- there is no second copy left to drift from it, so removing this call from here
  # breaks BOTH the live dispatcher AND the test seam identically. Sets $NODES (defined below) and
  # $EXTRA_NODES_FILE (defined above) as globals -- both must already be set before this is first called, which
  # holds for both callers (the production loop runs after NODES= below; the test seam sets POOL_QUEUE_PATH/
  # POOL_EXTRA_NODES_FILE via env before this script even starts).
  refresh_ssh_f
  CYCLE_NODES="$NODES $(extra_nodes)"
  # Reset the per-cycle revision-availability cache (see its own comment, right before revision_available_cached())
  # so each cycle re-probes fresh (a node CAN gain a revision between cycles, e.g. aws_pool_node.sh's
  # pre-provision loop finishing after this cycle started, or tools/pool_backfill_provisioned_markers.sh running
  # between cycles). A FRESH FILE per cycle, never reused -- see revision_available_cached's own comment for why
  # this is a file and not a plain bash associative array.
  [ -n "${REV_CACHE_FILE:-}" ] && rm -f "$REV_CACHE_FILE" 2>/dev/null
  REV_CACHE_FILE=$(mktemp "${TMPDIR:-/tmp}/pool_revcache.XXXXXX" 2>/dev/null) || REV_CACHE_FILE=""
}

revision_available() {
  # revision_available <node> <sha> -- has ~/derisk-pool/revisions/<sha> COMPLETED provisioning on <node>? Used
  # so a revision-pinned job (`cd ~/derisk-pool/revisions/<sha> && ...`, from `pool_provision.sh --isolated`) is
  # never handed to a node that was never provisioned with that revision (2026-09-23 fix round: reproduced --
  # AWS node provisioned only at ~/derisk-pool/sim from HEAD, every queued job pinned to an isolated revision,
  # `cd` failed, the job was already popped from the queue and its result was never pulled -- silently lost).
  #
  # BUGFIX (fix round #2): checking `[ -d <revdir> ]` alone was WRONG -- pool_provision.sh's remote `mkdir -p`
  # creates that directory FIRST, before rsync/venv/manifest-verify/sanity-check even run, and a later failure
  # there just `continue`s past the node WITHOUT removing it. A node whose provision partly failed (missing
  # .venv, unverified/tampered source, a degenerate sanity build) still passed this check, got handed the job,
  # and the crash landed only in that node's own job_status.log -- lost the same way, just one layer down. The
  # probe now requires `.provisioned_ok`, a marker pool_provision.sh writes as the LAST step of a fully
  # successful --isolated run for that node (see pool_provision.sh's per-node loop) -- a half- or badly-
  # provisioned revision dir never has it. tools/pool_revision_marker.sh's revision_marker_probe_cmd is the ONE
  # place this predicate string is spelled out -- pool_queue.sh's `add` gate calls the SAME function (2026-09-23
  # fix round #3, re-review MEDIUM: they used to ask two different questions of the same directory).
  #
  # Fails closed: unreachable/timeout/missing marker all return non-zero (job stays queued for another node/cycle).
  #
  # `-n` IS LOAD-BEARING (2026-09-25 incident: pool1+pool2 starved 07:35-09:59 EDT with 74 already-runnable
  # mem_gb=8 B2b jobs queued behind ONE job pinned to a not-yet-provisioned revision). This function is called
  # from INSIDE pop_job's `while IFS= read -r cand; do ... done < <(awk ...)` loop (the per-candidate revision
  # check). Without `-n`, ssh -- even run non-interactively, even with BatchMode=yes -- still opens and forwards
  # its OWN stdin to the remote command, and that stdin is the SAME fd 0 the enclosing `while read` loop is
  # consuming from the process substitution. The very first time a revision-pinned candidate needs a real probe,
  # this ssh call drains the rest of that pipe before the remote `test -f .../.provisioned_ok` even returns --
  # so pop_job's scan is silently truncated to that ONE candidate and returns empty, discarding every OTHER
  # admissible job (different revision, no revision pin, smaller) behind it in the SAME call. Because the
  # unavailable-revision job is left queued (never popped), it is the first candidate again on the NEXT cycle
  # too, so the starvation repeats indefinitely until that one revision happens to become available -- exactly
  # what happened to pool1 (idle-stopped mid-starvation, looking exactly like an AWS-lane defect) and pool2
  # (never stopped, never touched by AWS tooling at all, starved identically) on 2026-09-25. Reproduced in
  # isolation and pinned by tests/test_pool_autodispatch_workflow.py::test_pop_job_does_not_let_an_unavailable_revision_probe_swallow_later_queued_candidates
  # (existing revision-check tests never caught this: their stub `ssh` binaries are plain `echo "$*" >> log;
  # exit N` -- they never read stdin at all, so they cannot model the real ssh behaviour this bug depended on).
  # `-f -n` is already the established idiom in THIS file for the same reason (see fill_node's dispatch ssh
  # call) -- this brings the read-only probe in line with it.
  local node="$1" sha="$2"
  timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$node" \
    "$(revision_marker_probe_cmd "derisk-pool/revisions/$sha")" 2>/dev/null
}

revision_available_cached() {
  # PER-CYCLE REVISION-AVAILABILITY CACHE (2026-09-23 fix round #2, LOW; made durable in fix round #3). pop_job's
  # revision check used to make ONE ssh round trip per revision-pinned CANDIDATE it looked at, per pop, WHILE
  # HOLDING the queue flock -- a node missing several pinned revisions (or a hung node) could hold that lock for
  # multiple x10s timeouts on a single pop.
  #
  # BUGFIX (fix round #3, re-review HIGH -- "claimed fix ineffective in production"): a plain bash associative
  # array does NOT survive this function being called from inside `JOB=$(pop_job ...)` -- that `$(...)` is a
  # command-substitution SUBSHELL (bash forks a child process to run pop_job and capture its stdout), so any
  # `REV_CACHE[$key]=1` written in there is a write to the CHILD's own copy of the array and is discarded the
  # instant that subshell exits. The production `while node_is_idle "$NODE"; do JOB=$(pop_job ...); ...; done`
  # loop calls pop_job exactly this way, so the "cache" re-probed by ssh on EVERY single pop -- the fix round #2
  # test that exercised this function only proved it was self-consistent WITHIN one process (never through the
  # actual `$(...)` call path the live loop uses), so it stayed green while the live dispatcher never benefited.
  # A plain FILE survives past the subshell's exit (the write is a real write to disk, not to a process-local
  # variable) -- one file per cycle (created/rotated in cycle_setup, above), read and appended by every pop_job
  # subshell this cycle, so the SAME probe answer is reused across separate `$(pop_job ...)` calls, not just
  # across candidates scanned within one call. See --fill-node below for the test that goes through the real
  # command-substitution path (fix round #2's test is retained too, since a plain-array direct call is still a
  # valid thing to pin, just not sufficient on its own).
  local node="$1" sha="$2" key="$1:$2" cache="${REV_CACHE_FILE:-}"
  if [ -n "$cache" ] && [ -f "$cache" ]; then
    local hit
    hit=$(awk -F'\t' -v k="$key" '$1==k {print $2; exit}' "$cache" 2>/dev/null)
    if [ -n "$hit" ]; then
      [ "$hit" = "1" ]
      return
    fi
  fi
  local ok
  if revision_available "$node" "$sha"; then ok=1; else ok=0; fi
  [ -n "$cache" ] && printf '%s\t%s\n' "$key" "$ok" >> "$cache"
  if [ "$ok" = "0" ]; then
    # Rate-limited to once per (node, sha) per cycle: this branch only runs the FIRST time this (node, sha) pair
    # is probed this cycle (a cache hit above returns early without logging again), whether that first probe
    # happens in this subshell or a prior one this cycle -- the file itself is the rate-limit, not a separate
    # REV_MISSING_LOGGED array (which had the exact same cross-subshell survival problem this whole fix is about).
    echo "[pool-dispatch] revision $sha not provisioned on $node -- job(s) pinned to it stay queued this cycle" >&2
  fi
  [ "$ok" = "1" ]
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
        "echo \$(nproc) \$(cut -d' ' -f1 /proc/loadavg) \$(pgrep -c -f '^[^ ]*/?python[0-9.]* .*-m [r]esearch\.runners' 2>/dev/null | head -1) \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) \$(ps -eo rss,args | awk '\$2 ~ /python/ && /-m [r]esearch\\.runners/ {if (\$1>m) m=\$1} END{print int((m+1048575)/1048576)}') \$(awk '/MemTotal/{print int(\$2/1048576)}' /proc/meminfo); for e in /proc/[0-9]*/environ; do tr '\\0' '\\n' < \$e 2>/dev/null | grep -E '^POOL_JOB_(ID|MEM_GB)=' | sort | paste -sd' '; done | grep POOL_JOB_ID | sort -u || true" 2>/dev/null) || {
    _maybe_refresh_stale_aws_node "$node"   # 2026-09-25 review, LOW -- see its own comment above
    return 1
  }
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
  # THREADS (2026-09-24): node_is_idle fills a node to cores-1 jobs on the premise that each is single-threaded, but a
  # job that never set a thread count ran numpy/BLAS on EVERY core (23 threads per process on 12-core pool41, load 61),
  # and the load gate then refused all dispatch for 50 min with 152 jobs queued. So every job now inherits a default
  # of POOL_JOB_THREADS (1); a job that sets its own OMP_NUM_THREADS=... in its command line still wins.
  local est jid th="${POOL_JOB_THREADS:-1}"
  est=$(job_est_gb "$1"); jid="$(date +%s%N)-$RANDOM"
  printf "cd ~/derisk-pool/sim && POOL_JOB_ID='%s' POOL_JOB_MEM_GB='%s' OMP_NUM_THREADS='%s' OPENBLAS_NUM_THREADS='%s' MKL_NUM_THREADS='%s' NUMEXPR_NUM_THREADS='%s' JOB_B64='%s' WRAPPER_B64='%s' setsid bash -c 'printf \"%%s\" \"\$WRAPPER_B64\" | base64 -d | bash' </dev/null >/dev/null 2>&1 & exit 0" \
    "$jid" "$est" "$th" "$th" "$th" "$th" "$job_b64" "$wrapper_b64"
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
    if [ -n "$sha" ] && [ -n "$node" ] && ! revision_available_cached "$node" "$sha"; then continue; fi
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

fill_node() {
  # fill_node <node> -- FILL the node to capacity within this cycle (while, not if) — with the per-node cap
  # raised for single-threaded numpy jobs (owner 2026-09-02), a single if-per-cycle would need ~cap cycles to
  # fill. Extracted into its own function (2026-09-23 fix round #3, re-review HIGH) so the --fill-node test seam
  # below calls the EXACT production code path -- including the real `JOB=$(pop_job ...)` command substitution
  # -- rather than a re-typed copy that could silently drift from what the live `while true` loop (at the bottom
  # of this file) runs.
  local NODE="$1"
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
    sleep "${POOL_DISPATCH_LAUNCH_SLEEP:-5}"     # let the launch register before this node's next capacity check
  done
}

if [ "${1:-}" = "--pop-once" ]; then
  pop_job "${2:-999}" "${3:-}"
  exit $?
fi
if [ "${1:-}" = "--reserved-gb" ]; then reserved_gb "$2"; exit 0; fi
if [ "${1:-}" = "--peek-est-gb" ]; then peek_est_gb; exit 0; fi
if [ "${1:-}" = "--min-queued-est-gb" ]; then   # smallest declared size among fresh queued jobs; empty if none
  cutoff=$(( $(date +%s) - ${POOL_JOB_MAX_AGE:-43200} )); m=""
  while IFS= read -r cand; do e=$(job_est_gb "$cand"); { [ -z "$m" ] || [ "$e" -lt "$m" ]; } && m="$e"; done \
    < <(awk -F'\t' -v c="$cutoff" 'NF>1 && $1+0 >= c && $0 ~ /#checked:/ {print $2}' "$QUEUE" 2>/dev/null)
  echo "$m"; exit 0
fi
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
  # TEST SEAM (2026-09-23, hardened fix round #2): proves refresh_ssh_f is called EVERY cycle of the REAL
  # dispatch loop (not just once at process start) without running the full dispatcher (no queue popping, no
  # real ssh calls to a node). Calls `cycle_setup` -- the SAME function the production `while true` loop below
  # calls, not a separate copy of its shape -- so a mutation that removes `refresh_ssh_f` from `cycle_setup`
  # breaks this test too (the old version called `refresh_ssh_f` inline here, a copy the mutation could dodge).
  # Prints one line per cycle -- "F" if SSH_F currently carries -F<config>, else "NOF" -- so a test can create
  # POOL_SSH_CONFIG's file BETWEEN cycles and see the NEXT line flip, inside one long-lived process.
  [ "$#" -eq 3 ] || { echo "usage: $0 --print-ssh-f-loop <n-cycles> <sleep-s>" >&2; exit 2; }
  for _i in $(seq 1 "$2"); do
    cycle_setup
    if [ "${#SSH_F[@]}" -gt 0 ]; then echo "F"; else echo "NOF"; fi
    sleep "$3"
  done
  exit 0
fi
if [ "${1:-}" = "--revision-available" ]; then
  # TEST SEAM (2026-09-23): exercises the REAL revision_available ssh call (same argv construction, including
  # SSH_F) against one node/sha pair, without a real node or a real revision -- so a stubbed `ssh` on PATH can
  # assert the exact `[ -f ~/derisk-pool/revisions/<sha>/.provisioned_ok ]` probe it makes (fix round #2: a
  # completion MARKER, not bare directory existence).
  [ "$#" -eq 3 ] || { echo "usage: $0 --revision-available <node> <sha>" >&2; exit 2; }
  revision_available "$2" "$3"; exit $?
fi
if [ "${1:-}" = "--fill-node" ]; then
  # TEST SEAM (2026-09-23 fix round #3): calls cycle_setup then fill_node -- the SAME function the production
  # `while true` loop (below) calls once per node per cycle, via the real `JOB=$(pop_job ...)` command
  # substitution -- so a mutation to the per-cycle revision cache (or to fill_node itself) breaks this test too.
  # Unlike --pop-once (a fresh process per call, so its own cache always starts empty by construction and can
  # never demonstrate cross-call persistence), this drives MULTIPLE pop_job calls inside ONE process, exactly
  # like the live dispatcher does when filling one node to capacity within a cycle.
  [ "$#" -eq 2 ] || { echo "usage: $0 --fill-node <node>" >&2; exit 2; }
  cycle_setup
  fill_node "$2"
  exit 0
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
  cycle_setup
  for NODE in $CYCLE_NODES; do
    fill_node "$NODE"
  done
  sleep "$POLL"
done
