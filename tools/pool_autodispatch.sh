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
# Shared repo root = the parent of the ONE git common dir (the resolution tools/gpu_queue.sh uses), so every
# worktree reads the same queue on whichever host holds the checkout (was the workstation path until the
# 2026-09-30 move of the project home to the server, docs/plans/2026-09-30-home-to-server-MIGRATION.md).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# RUNTIME MODE (2026-09-30, trust boundary of the server move; docs/plans/2026-09-30-home-to-server-MIGRATION.md
# section 10). On srv-host-0 this script runs as the unprivileged user `sim` from a reviewed checkout
# (/home/sim/runtime) that the agent cannot write, and the agent-writable workspace is DATA only: SIM_WORKSPACE names
# it, and only the queue files below are read or written there. In this mode no git command runs at all (a git
# command inside the workspace would execute its agent-controlled .git/config), and everything that can make this
# process run a program or pick an ssh target stays in the code root ($ROOT): the repo-local ssh config (its
# ProxyCommand runs a shell), the extra-node list, the AWS state files and the pause markers. On the server those
# files do not exist in the runtime, so the AWS lane is off there. Without SIM_WORKSPACE nothing changes.
if [ -n "${SIM_WORKSPACE:-}" ]; then
  DATA_ROOT="$SIM_WORKSPACE"
else
  _GC=$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) && ROOT=$(dirname "$_GC")
  DATA_ROOT="$ROOT"
fi
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's OWN tools/ dir (unlike $ROOT above,
# this follows whichever checkout/worktree is actually running -- used below so the stale-HostName refresh
# invokes the SAME checkout's aws_pool_node.sh, not always the one at the hardcoded $ROOT.
# shellcheck source=tools/pool_revision_marker.sh
source "$SELF_DIR/pool_revision_marker.sh"
# shellcheck source=tools/pool_nodes.sh
source "$SELF_DIR/pool_nodes.sh"
QUEUE="${POOL_QUEUE_PATH:-$DATA_ROOT/research/queue/pool.queue}"
CLAIMED="${POOL_RUNNING_PATH:-${QUEUE%.queue}.running}"
POLL="${POOL_DISPATCH_POLL:-60}"
# MEMORY RESERVATIONS (2026-09-23). The RSS snapshot node_is_idle reads is taken seconds after the previous launch,
# before that job has grown -- so one fill cycle handed pool42 SIX D6 workers (each grows to 4-5 GB on a 15 GB node)
# in 30 s, and an earlier cycle left pool41 with 167 MB free. Every dispatch now RESERVES its expected size on that
# node for POOL_GROWTH_WINDOW_S; a job declares its size with `mem_gb=N` anywhere in its line (e.g. in the
# --checked reason), else POOL_JOB_EST_GB. The reservation double-counts a job whose RSS has already caught up
# (MemAvailable fell AND it is still reserved) -- deliberately conservative: an under-filled node costs minutes, an
# OOM costs every job on the node plus a silent rc=0 from runners that swallow worker deaths.
RESV="${POOL_RESERVATIONS_PATH:-$DATA_ROOT/research/queue/.pool_reservations}"
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
# reregister_stale_paused_nodes / PAUSE_MARK_DIR (2026-09-25 review, MEDIUM: "a cheap check in the dispatcher
# cycle" + LOW: "fill_node never re-reads registration -- a cycle already running keeps filling the node through
# the whole pause"). Sourced via $SELF_DIR (this checkout's own tools/, not the hardcoded $ROOT above -- same
# reasoning as pool_revision_marker.sh's own sourcing just above). LOG left unset (default /dev/stderr inside
# the library's own `${LOG:-...}` guards) -- this script has no log file of its own to reuse.
# shellcheck source=tools/aws_stop_safety_lib.sh
source "$SELF_DIR/aws_stop_safety_lib.sh"
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

valid_node_names() {
  # stdin: whitespace-separated node names -> stdout: the ones that are plain host aliases, one line. A host argument
  # that starts with '-' is parsed by ssh/rsync as an OPTION (-oProxyCommand=... runs a shell), and a name arrives
  # here from files (.pool_extra_nodes, pause markers), so every name is checked before any ssh sees it (2026-09-30
  # trust boundary). `read -a` splits without glob expansion.
  local n out="" names=()
  read -r -a names || true
  for n in "${names[@]}"; do
    if [[ $n =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then out="$out $n"
    else echo "[pool-dispatch] ⛔ ignoring node name $(printf '%q' "$n") (not a plain host alias)" >&2; fi
  done
  printf '%s\n' "${out# }"
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
  # REREGISTER ANY STALE PAUSE, EVERY CYCLE (2026-09-25 review, MEDIUM: "a cheap check in the dispatcher cycle").
  # Runs BEFORE CYCLE_NODES is computed below so a node this call just re-registered is picked up THIS cycle,
  # not the next one.
  reregister_stale_paused_nodes
  CYCLE_NODES=$(printf '%s %s\n' "$NODES" "$(extra_nodes)" | valid_node_names)
  SANDBOX_READY=()   # re-probe every node's sandbox readiness each cycle (bwrap can be installed between cycles)
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
  local job src
  src=$(queue_for_reading)
  job=$(awk -F'\t' -v now="$(date +%s)" -v d="${POOL_JOB_MAX_AGE:-43200}" -v cap="${POOL_JOB_MAX_AGE_CAP_H:-240}" \
        "$POOL_FRESH_AWK"' NF>1 && $1+0 >= now - maxage($0) {print $2; exit}' "$src" 2>/dev/null)
  queue_done_reading "$src"
  job_est_gb "$job"
}
# PER-LINE MAX AGE (2026-09-30). The staleness guard below treats a line older than POOL_JOB_MAX_AGE (12 h) as
# debris. That is right for an abandoned sweep and wrong for a batch that is simply waiting in a deep queue: the
# 2026-09-29 evening landing queued ~550 legitimate lines (three to five days of pool work), and every one would
# have been skipped from the next morning on. A line may declare `max_age_h=N` anywhere in its text (usually the
# --checked reason, beside mem_gb=N and pool_node=<name>) to set its own limit; it is capped at
# POOL_JOB_MAX_AGE_CAP_H (240 h), so an abandoned batch still expires. A line with no token keeps the default.
POOL_FRESH_AWK='function maxage(l,  m) { if (match(l, /max_age_h=[0-9]+/)) { m = substr(l, RSTART + 10, RLENGTH - 10) + 0; if (m > cap) m = cap; return m * 3600 } return d }'
NODES="${POOL_NODES:-$POOL_NODES_DEFAULT}"
# GATE-SEED CLASS (2026-11-02, RULING gates-front-1102). pool_queue.sh's GATE-SEED GUARD defines a gate line as one
# whose --seed/--seeds list holds 42/43/44/100/101/102; this is that same test, in awk, so pop_job can try gate lines
# first. On 11-01/02 gate lines sat behind hundreds of smaller dev-smoke lines and were withdrawn and re-added at the
# front by hand. Keep the seed list in step with GATE_SEEDS_LIST in tools/pool_queue.sh.
POOL_GATE_AWK='function isgate(l,  r, m, n, a, i) { r = l
  while (match(r, /--seeds?(=|[ \t]+)[0-9]+([, \t]+[0-9]+)*/)) { m = substr(r, RSTART, RLENGTH); r = substr(r, RSTART + RLENGTH)
    gsub(/[^0-9]+/, " ", m); n = split(m, a, " "); for (i = 1; i <= n; i++) if (a[i] ~ /^(42|43|44|100|101|102)$/) return 1 } return 0 }'
NODE_BUDGET=0

# PER-NODE LIMITS (2026-10-02, cloud session 5: the workstation joins the pool as node `wspool`, user `simpool`).
# Every capacity knob below is global and node_is_idle reads nproc, loadavg and /proc/meminfo host-wide, which is
# right for a dedicated mini-PC and wrong for a shared machine whose owner also games, trains and serves a model on
# it. POOL_NODE_LIMITS holds space-separated entries `<node>:jobs=N,mem_gb=M,min_avail_gb=K` (any non-empty subset of
# the three keys, each a positive integer). For a node with an entry:
#   jobs=N          the job cap is min(the usual cap, N);
#   mem_gb=M        NODE_BUDGET is also capped at M minus the POOL_JOB_MEM_GB of every live job on that node (the
#                   environ scan below; the pool account on such a node sees only its own processes, which are
#                   exactly the pool's jobs), so the pool never holds more than M GB there by declaration;
#   min_avail_gb=K  replaces POOL_MIN_AVAIL_GB for that node, in the floor check and in the budget.
# A node with no entry runs the identical global code path (pinned by tests/test_pool_autodispatch_workflow.py).
# Parsed and validated ONCE, here, before any seam or the loop: a malformed value stops the process (exit 2) rather
# than silently dispatching a shared machine at mini-PC limits.
declare -A NODE_LIM_SET=() NODE_LIM_JOBS=() NODE_LIM_MEM=() NODE_LIM_MINAVAIL=()
_node_limits_die() {
  echo "[pool-dispatch] ⛔ POOL_NODE_LIMITS: $1 -- expected space-separated <node>:jobs=N,mem_gb=M,min_avail_gb=K (keys optional, each N a positive integer); refusing to run" >&2
  exit 2
}
parse_node_limits() {
  local entry node kvs kv key val entries=() pairs=()
  read -r -a entries <<<"${POOL_NODE_LIMITS:-}" || true   # `read -a` splits without glob expansion
  for entry in "${entries[@]}"; do
    [[ $entry == *:* ]] || _node_limits_die "entry $(printf '%q' "$entry") has no ':'"
    node="${entry%%:*}" kvs="${entry#*:}"
    # The node-name rule valid_node_names applies (a name is later an ssh host argument).
    [[ $node =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || _node_limits_die "node name $(printf '%q' "$node") is not a plain host alias"
    [ -z "${NODE_LIM_SET[$node]:-}" ] || _node_limits_die "node $node has more than one entry"
    [[ -n $kvs && $kvs != ,* && $kvs != *, && $kvs != *,,* ]] || _node_limits_die "entry $(printf '%q' "$entry") has an empty limit list or an empty item"
    IFS=, read -r -a pairs <<<"$kvs"
    for kv in "${pairs[@]}"; do
      [[ $kv == *=* ]] || _node_limits_die "item $(printf '%q' "$kv") for $node is not key=N"
      key="${kv%%=*}" val="${kv#*=}"
      # Leading zeros are refused (bash arithmetic reads 08 as bad octal) and six digits is far above any node.
      [[ $val =~ ^[1-9][0-9]{0,5}$ ]] || _node_limits_die "$key=$(printf '%q' "$val") for $node is not a positive integer"
      case "$key" in
        jobs)         [ -z "${NODE_LIM_JOBS[$node]:-}" ]     || _node_limits_die "key jobs repeated for $node";     NODE_LIM_JOBS[$node]=$val ;;
        mem_gb)       [ -z "${NODE_LIM_MEM[$node]:-}" ]      || _node_limits_die "key mem_gb repeated for $node";   NODE_LIM_MEM[$node]=$val ;;
        min_avail_gb) [ -z "${NODE_LIM_MINAVAIL[$node]:-}" ] || _node_limits_die "key min_avail_gb repeated for $node"; NODE_LIM_MINAVAIL[$node]=$val ;;
        *) _node_limits_die "unknown key $(printf '%q' "$key") for $node (known: jobs, mem_gb, min_avail_gb)" ;;
      esac
    done
    NODE_LIM_SET[$node]="$kvs"
  done
}
parse_node_limits
# One log line per node per POOL_NODE_LIMIT_LOG_RATE_S (default 10 min) when a per-node limit is what refused a
# dispatch. Kept in memory, not in a mark file: node_is_idle and fill_node run in the loop's own shell (never a
# subshell), and on the server's runtime mode the code root is read-only, so a mark file there could never be
# written and the rate limit would silently become one line per poll.
NODE_LIMIT_LOG_RATE_S="${POOL_NODE_LIMIT_LOG_RATE_S:-600}"
[[ $NODE_LIMIT_LOG_RATE_S =~ ^[0-9]+$ ]] || NODE_LIMIT_LOG_RATE_S=600
declare -A NODE_LIMIT_LOGGED_AT=()
NODE_LIMIT_GLOBAL_BUDGET=0   # for a node with an entry: the budget the global settings alone would give (0 = refused)

log_node_limit_refusal() {   # log_node_limit_refusal <node> <why> -- rate-limited per node
  local node="$1" now last
  now=$(date +%s); last="${NODE_LIMIT_LOGGED_AT[$node]:-}"
  if [ -n "$last" ] && [ $(( now - last )) -lt "$NODE_LIMIT_LOG_RATE_S" ]; then return 0; fi
  NODE_LIMIT_LOGGED_AT[$node]=$now
  echo "[pool-dispatch] $(date '+%H:%M:%S') $node: per-node limit refused dispatch (POOL_NODE_LIMITS $node:${NODE_LIM_SET[$node]}): $2; the global settings alone would have dispatched (next such line for $node in >= ${NODE_LIMIT_LOG_RATE_S}s)" >&2
}

# POOL-JOB SANDBOX (2026-09-30 security review, HIGH #2: "pool-submit is a LAN foothold"). Every job used to run over
# ssh as the node's login user -- passwordless sudo, full network -- so an agent-authored line (the gateway's
# pool-submit, or the local-model driver's `pool_queue.sh add`) could reach the LAN or take the node. A sandboxed
# launch sends tools/pool_job_sandbox.sh as the remote wrapper instead of the legacy four lines: bubblewrap built from
# empty, no network, no-new-privs, the pinned revision read-only, only its research/findings/raw writable (an agent
# line: only its own _agent_jobs/<id>); see that file's header for the full recipe.
#   POOL_SANDBOX=auto  (default) sandbox on every node whose probe says POOL_SANDBOX_READY; on a node that is not
#                      ready, launch as before and log a loud UNSANDBOXED warning with every such dispatch.
#   POOL_SANDBOX=on    sandbox or nothing: a node that is not ready gets no work (jobs stay queued).
#   POOL_SANDBOX=off   the legacy launch everywhere (no probe).
# Whatever the mode, a line carrying AGENT_JOB_SRC=sim-agent (built by tools/agent_gateway_submit.py) is only ever
# handed to a node that is sandbox-ready (pop_job skips it otherwise, leaving it queued). Anything but on|auto|off
# fails closed to `on`. The live loop publishes its mode each cycle to $SANDBOX_STATE; tools/agent_gateway.sh accepts
# pool-submit only while that file says mode=on and is fresh.
POOL_SANDBOX_MODE="${POOL_SANDBOX:-auto}"
case "$POOL_SANDBOX_MODE" in
  on|auto|off) ;;
  *) echo "[pool-dispatch] ⛔ POOL_SANDBOX='$POOL_SANDBOX_MODE' is not on|auto|off -- treating it as on (fail closed)" >&2
     POOL_SANDBOX_MODE=on ;;
esac
SANDBOX_WRAPPER="${POOL_SANDBOX_WRAPPER:-$SELF_DIR/pool_job_sandbox.sh}"   # TEST-ONLY override
SANDBOX_STATE="${POOL_SANDBOX_STATE_PATH:-$DATA_ROOT/research/queue/.pool_sandbox_state}"
AGENT_JOB_MARKER="AGENT_JOB_SRC=sim-agent"   # the marker tools/agent_gateway_submit.py embeds in every line it builds
declare -A SANDBOX_READY=()   # node -> 1/0, reset every cycle in cycle_setup (fill_node runs in the main shell)
NODE_SANDBOX=0                # the decision for the node fill_node is filling right now (read by pop_job too)

sandbox_probe() {
  # sandbox_probe <node> -- runs tools/pool_job_sandbox.sh's `probe` action ON the node (the same file a sandboxed
  # launch runs, sent the same way), which starts the real base sandbox and checks no-new-privs inside it. Prints the
  # probe's one-line answer; returns 0 only for POOL_SANDBOX_READY. `-n`: see revision_available's comment.
  local node="$1" b64 out
  b64=$(base64 -w0 < "$SANDBOX_WRAPPER" 2>/dev/null) || { echo "POOL_SANDBOX_NOT_READY reason=wrapper-unreadable"; return 1; }
  out=$(timeout 20 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$node" \
        "POOL_SANDBOX_ACTION=probe SANDBOX_B64='$b64' bash -c 'printf \"%s\" \"\$SANDBOX_B64\" | base64 -d | bash'" \
        2>/dev/null | grep -m1 -E '^POOL_SANDBOX_(NOT_)?READY')
  printf '%s\n' "${out:-POOL_SANDBOX_NOT_READY reason=no-answer (unreachable, or the probe printed nothing)}"
  case "$out" in POOL_SANDBOX_READY*) return 0 ;; esac
  return 1
}

node_sandbox_decision() {
  # node_sandbox_decision <node> -- sets NODE_SANDBOX (1 = launch sandboxed) for the node about to be filled.
  # Returns 1 (dispatch NOTHING to this node this cycle) only in `on` mode on a node that is not sandbox-ready.
  local node="$1" ans
  NODE_SANDBOX=0
  [ "$POOL_SANDBOX_MODE" = off ] && return 0
  if [ -z "${SANDBOX_READY[$node]:-}" ]; then
    ans=$(sandbox_probe "$node")
    case "$ans" in
      POOL_SANDBOX_READY*) SANDBOX_READY[$node]=1 ;;
      *) SANDBOX_READY[$node]=0
         if [ "$POOL_SANDBOX_MODE" = on ]; then
           echo "[pool-dispatch] ⛔ POOL_SANDBOX=on: $node is not sandbox-ready ($ans) -- dispatching nothing to it this cycle" >&2
         else
           echo "[pool-dispatch] ⚠ $node is not sandbox-ready ($ans) -- POOL_SANDBOX=auto launches there UNSANDBOXED" >&2
         fi ;;
    esac
  fi
  if [ "${SANDBOX_READY[$node]}" = 1 ]; then NODE_SANDBOX=1; return 0; fi
  [ "$POOL_SANDBOX_MODE" = on ] && return 1
  return 0
}

write_sandbox_state() {
  # The live loop's effective mode, for tools/agent_gateway.sh (pool-submit only while mode=on and fresh).
  local tmp="$SANDBOX_STATE.tmp.$$"
  mkdir -p "$(dirname "$SANDBOX_STATE")" 2>/dev/null
  printf 'mode=%s ts=%s pid=%s\n' "$POOL_SANDBOX_MODE" "$(date +%s)" "$$" > "$tmp" 2>/dev/null && mv -f "$tmp" "$SANDBOX_STATE"
}

# Runtime mode creates nothing up front: a missing queue just means nothing to pop, and every write into the
# workspace happens inside the unit's confinement (see the guard before the main loop).
[ -n "${SIM_WORKSPACE:-}" ] || { mkdir -p "$(dirname "$QUEUE")"; touch "$QUEUE" "$CLAIMED"; }

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
  local cores load procs avail_gb max_job_gb total_gb rest
  read -r cores load procs avail_gb max_job_gb total_gb rest <<<"$out"
  cores="${cores:-0}" load="${load:-99}" procs="${procs:-99}" avail_gb="${avail_gb:-0}" max_job_gb="${max_job_gb:-0}"
  total_gb="${total_gb:-0}"
  # A NODE'S ANSWER IS UNTRUSTED (2026-09-30 trust boundary). Pool jobs are agent-authored and run on the node, so
  # its probe output must never reach bash arithmetic unchecked: $(( x )) evaluates a subscript such as
  # `a[$(cmd)]` inside x, i.e. runs cmd here, as the dispatcher's user. `cores` and `total_gb` reached $(( )) with
  # no check before this. Every field must be a plain number, else the node is treated as busy.
  if ! [[ $cores =~ ^[0-9]+$ && $procs =~ ^[0-9]+$ && $avail_gb =~ ^[0-9]+$ && $max_job_gb =~ ^[0-9]+$ \
          && $total_gb =~ ^[0-9]+$ && $committed =~ ^[0-9]+$ && $load =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[pool-dispatch] ⛔ $node: malformed capacity-probe answer, treated as busy: $(printf '%q' "$out" | cut -c1-160)" >&2
    return 1
  fi
  # COUNT ONLY PYTHON RUNNERS (2026-09-23). The old `pgrep -fc research.runners` also counted every `flock`
  # waiter, `bash -c` wrapper and ssh line carrying the job text, so a node with 2 working jobs + 4 jobs queued
  # behind a lab-chosen flock read 11/11 "full" at load ~5.6 on 12 cores while 13 jobs sat queued. Load stays
  # the CPU gate; a MemAvailable floor (POOL_MIN_AVAIL_GB, default 3) now guards RAM (15 GB nodes).
  local cap="${POOL_JOBS_PER_NODE:-$(( cores > 2 ? cores - 1 : 1 ))}"   # single-threaded jobs: fill to cores-1
  _NI_PROCS="$procs" _NI_AVAIL="$avail_gb" _NI_MAXJOB="$max_job_gb" _NI_TOTAL="$total_gb" _NI_COMMITTED="$committed"
  _NI_LOAD="$load" _NI_CORES="$cores" _NI_RESV=""   # reservations are read only once the cheap checks pass
  NODE_LIMIT_GLOBAL_BUDGET=0
  if [ -z "${NODE_LIM_SET[$node]:-}" ]; then
    capacity_verdict "$node" "$cap" "${POOL_MIN_AVAIL_GB:-3}" ""   # no entry: the global path, unchanged
    return
  fi
  # A node with a POOL_NODE_LIMITS entry: the verdict under its limits decides. When that refuses, the same probe is
  # judged under the global settings alone; if THOSE would dispatch, the limit is what refused, and that is logged.
  local lcap="$cap" lmin="${NODE_LIM_MINAVAIL[$node]:-${POOL_MIN_AVAIL_GB:-3}}" lmem="${NODE_LIM_MEM[$node]:-}"
  if [ -n "${NODE_LIM_JOBS[$node]:-}" ] && [ "${NODE_LIM_JOBS[$node]}" -lt "$cap" ] 2>/dev/null; then lcap="${NODE_LIM_JOBS[$node]}"; fi
  local lim_budget lim_rc why
  capacity_verdict "$node" "$lcap" "$lmin" "$lmem"; lim_rc=$?
  lim_budget="$NODE_BUDGET" why="$NI_REASON"
  if capacity_verdict "$node" "$cap" "${POOL_MIN_AVAIL_GB:-3}" ""; then NODE_LIMIT_GLOBAL_BUDGET="$NODE_BUDGET"; fi
  NODE_BUDGET="$lim_budget"
  if [ "$lim_rc" -ne 0 ] && [ "$NODE_LIMIT_GLOBAL_BUDGET" -ge 1 ]; then log_node_limit_refusal "$node" "$why"; fi
  return "$lim_rc"
}

capacity_verdict() {
  # capacity_verdict <node> <job-cap> <min-avail-gb> <mem-gb-or-empty> -- judge node_is_idle's probe (the _NI_*
  # values it just set) under these settings. Returns 0 = has capacity; sets NODE_BUDGET at the same point the
  # inline code always did (a refusal before that point leaves it as it was) and NI_REASON on a refusal.
  local node="$1" cap="$2" min_avail="$3" mem="$4"
  NI_REASON=""
  [ "${_NI_PROCS:-99}" -lt "$cap" ] || { NI_REASON="${_NI_PROCS} pool runner(s) running, job cap $cap"; return 1; }
  [ "${_NI_AVAIL:-0}" -ge "$min_avail" ] || { NI_REASON="MemAvailable ${_NI_AVAIL} GB is below the ${min_avail} GB floor"; return 1; }
  # Jobs GROW after dispatch (2026-09-23: three D6 workers reached 4-5 GB each and left pool41 with 167 MB free,
  # load 29/12) -- so also require headroom for one more job the size of the largest one already running there.
  [ "${_NI_AVAIL:-0}" -ge "${_NI_MAXJOB:-0}" ] || { NI_REASON="MemAvailable ${_NI_AVAIL} GB < largest running job ${_NI_MAXJOB} GB"; return 1; }
  # ...and room for the NEXT job on top of everything dispatched here within the growth window (see RESV above).
  [ -n "$_NI_RESV" ] || _NI_RESV=$(reserved_gb "$node")
  NODE_BUDGET=$(( ${_NI_AVAIL:-0} - ${_NI_RESV:-0} - min_avail ))   # GB pop_job may hand this node
  # COMMITTED (2026-09-23 20:45): a snapshot between an LB job's worker phases read 9 GB free on pool42 while two
  # swap-probe jobs (each peaking ~6 GB) were running; the growth-window reservations had expired, a third was sent,
  # and BOTH nodes thrashed until ssh timed out. So the budget is also capped by MemTotal minus an OS reserve minus
  # the declared size of every job still running there, for as long as it runs.
  local lifetime_budget=$(( ${_NI_TOTAL:-0} - ${POOL_OS_RESERVE_GB:-2} - _NI_COMMITTED ))
  [ "${_NI_TOTAL:-0}" -gt 0 ] && [ "$lifetime_budget" -lt "$NODE_BUDGET" ] && NODE_BUDGET=$lifetime_budget
  # PER-NODE mem_gb (POOL_NODE_LIMITS): the pool's own declared total on this node stays within mem_gb.
  if [ -n "$mem" ]; then
    local node_mem_budget=$(( mem - _NI_COMMITTED ))
    [ "$node_mem_budget" -lt "$NODE_BUDGET" ] && NODE_BUDGET=$node_mem_budget
  fi
  [ "$NODE_BUDGET" -ge 1 ] || { NI_REASON="memory budget ${NODE_BUDGET} GB (MemAvailable ${_NI_AVAIL}, reserved ${_NI_RESV:-0}, floor ${min_avail}, committed ${_NI_COMMITTED}${mem:+, mem_gb $mem}) is below 1 GB"; return 1; }
  awk -v l="$_NI_LOAD" -v c="$_NI_CORES" 'BEGIN{exit !(l < c - 0.5)}' || { NI_REASON="load ${_NI_LOAD} on ${_NI_CORES} cores"; return 1; }
}

remote_launch_command() {
  # remote_launch_command <job> [sandbox=0|1]
  # Encode both the job and the fixed wrapper so shell metacharacters, tabs, and
  # newlines cannot corrupt the remote command or its one-line status record.
  # sandbox=1 (2026-09-30): the wrapper is tools/pool_job_sandbox.sh (runs the job inside bubblewrap and writes the
  # SAME v2 status row); the rest of the command -- cd, POOL_JOB_ID/MEM_GB, thread defaults, JOB_B64, setsid -- is
  # identical, so node_is_idle's environ scan, pool_queue's JOB_B64 liveness scan and check_fast_fail are unchanged.
  local job_b64 wrapper wrapper_b64 action=""
  job_b64=$(printf '%s' "$1" | base64 -w0) || return 1
  if [ "${2:-0}" = 1 ]; then
    [ -r "$SANDBOX_WRAPPER" ] || return 1
    wrapper_b64=$(base64 -w0 < "$SANDBOX_WRAPPER") || return 1
    action="POOL_SANDBOX_ACTION='run' "
  else
  wrapper='job=$(printf "%s" "$JOB_B64" | base64 -d) || exit 125
bash -c "$job" > autodispatch.out 2>&1
rc=$?
printf "v2\t%s\t%s\t%s\n" "$(date +%s)" "$rc" "$JOB_B64" >> job_status.log'
  wrapper_b64=$(printf '%s' "$wrapper" | base64 -w0) || return 1
  fi
  # THE BRACE GROUP (2026-10-06): `cd X && ... setsid ... </dev/null >/dev/null 2>&1 & exit 0` backgrounded the whole
  # and-list as a subshell whose own stdout/stderr (ssh's channel) were NOT redirected -- only the setsid command's
  # were -- so the channel stayed open for the whole job. `ssh -f` hid it (it just kept one lingering local ssh process
  # per running job); the same shape in tools/pool_canary.sh made a plain-ssh launch hang until its timeout. The
  # redirections now bind to the whole group, so the channel closes at once (tests/test_pool_canary_guard.py).
  # POOL_JOB_ID / POOL_JOB_MEM_GB are INHERITED by every process of the job (they survive bash's exec of the last
  # command and memcap's scope), so node_is_idle can sum the declared sizes of the jobs still running there.
  # THREADS (2026-09-24): node_is_idle fills a node to cores-1 jobs on the premise that each is single-threaded, but a
  # job that never set a thread count ran numpy/BLAS on EVERY core (23 threads per process on 12-core pool41, load 61),
  # and the load gate then refused all dispatch for 50 min with 152 jobs queued. So every job now inherits a default
  # of POOL_JOB_THREADS (1); a job that sets its own OMP_NUM_THREADS=... in its command line still wins.
  local est jid th="${POOL_JOB_THREADS:-1}"
  est=$(job_est_gb "$1"); jid="$(date +%s%N)-$RANDOM"
  printf "{ cd ~/derisk-pool/sim && POOL_JOB_ID='%s' POOL_JOB_MEM_GB='%s' OMP_NUM_THREADS='%s' OPENBLAS_NUM_THREADS='%s' MKL_NUM_THREADS='%s' NUMEXPR_NUM_THREADS='%s' JOB_B64='%s' WRAPPER_B64='%s' %ssetsid bash -c 'printf \"%%s\" \"\$WRAPPER_B64\" | base64 -d | bash'; } </dev/null >/dev/null 2>&1 & exit 0" \
    "$jid" "$est" "$th" "$th" "$th" "$th" "$job_b64" "$wrapper_b64" "$action"
}

check_fast_fail() {
  # check_fast_fail <node> <job-as-dispatched> -- READ-ONLY. Tails the node's OWN job_status.log for the v2
  # record this exact dispatch produced and, when it died with rc=127 ("command not found") or rc=2 (a shell
  # syntax/usage error) -- the signature of a job that never actually ran -- logs a LOUD line HERE (never
  # remotely; this never writes anything to the node, and never touches remote_launch_command's wrapper).
  #
  # WHY (2026-09-25). Six SETTLE A2 pool lines and, independently, a bare `status` job in gpu_queue.log
  # (2026-08-31/09-01, three separate cycles) all died this way in well under a second, and NOTHING said so
  # louder than an ordinary v2 status row nobody was tailing -- the board and the claim record both said
  # "dispatched" while nothing ran. tools/queue_job_shape_check.sh now refuses that SHAPE at enqueue time; this
  # is the belt-and-suspenders net for whatever it cannot see from here (a module importable on this node's
  # SHARED checkout but not on the pinned revision this job actually runs in, e.g.) -- it is deliberately not
  # a substitute for the enqueue-time check, since a fast-fail already happened by the time this runs.
  #
  # Called from fill_node AFTER the existing POOL_DISPATCH_LAUNCH_SLEEP (default 5s, already there to "let the
  # launch register") -- every real rc=127/2 in the historical record completed in well under 1s, so 5s is
  # ample; a job that legitimately takes longer than that to even START is simply not yet in the log and this
  # silently finds nothing (best-effort, never a false alarm).
  #
  # THIS DISPATCH'S record only (2026-09-25 fix round, review MEDIUM). <t0> is the local `date +%s` fill_node
  # took just BEFORE its `ssh -f` launch. job_status.log is append-only and keeps every earlier attempt, so the
  # first version (`grep -F <b64> | tail -1`, no time check) reported a re-dispatch of an identical job whose
  # EARLIER attempt had died rc=127 as a new FAST-FAIL -- and its substring match could also hit a longer job
  # whose base64 merely starts with this one's. A record now counts only when its job field is EXACTLY this
  # job's base64 AND its timestamp is >= t0. The remote side stays a read-only grep; the exact-match and time
  # filter run HERE, on its output. The record's timestamp is the node's clock: a node running more than a
  # second or so behind this host makes a genuine fast-fail read as older than t0 and go unreported (a miss,
  # never a false alarm). With no numeric <t0> there is no way to tell this dispatch from an earlier one, so it
  # reports nothing.
  local node="$1" job="$2" t0="${3:-}" job_b64 remote_cmd raw rec rc ts
  [[ $t0 =~ ^[0-9]+$ ]] || return 0
  job_b64=$(printf '%s' "$job" | base64 -w0) || return 0
  remote_cmd=$(printf 'grep -F -- %q ~/derisk-pool/sim/job_status.log 2>/dev/null | tail -n 50' "$job_b64")
  raw=$(timeout 10 ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=6 "$node" "$remote_cmd" 2>/dev/null)
  [ -n "$raw" ] || return 0
  rec=$(printf '%s\n' "$raw" | awk -F'\t' -v b="$job_b64" -v t="$t0" \
        '$1 == "v2" && $4 == b && $2 ~ /^[0-9]+$/ && $2 + 0 >= t + 0 {r = $0} END {if (r != "") print r}')
  [ -n "$rec" ] || return 0
  ts=$(printf '%s' "$rec" | cut -f2)
  rc=$(printf '%s' "$rec" | cut -f3)
  case "$rc" in
    127|2)
      echo "[pool-dispatch] ⛔ FAST-FAIL: $node rc=$rc $(( ts - t0 ))s after dispatch (died on argv[0]/syntax, not a real run): $(printf '%s' "$job" | tr '\n\t' '  ' | cut -c1-160)" >&2 ;;
    126)
      # tools/pool_job_sandbox.sh refused to build the sandbox (its reason is in the node's autodispatch.out); the job
      # never ran. (bash's own "not executable" is also 126 -- equally a job that never ran.)
      echo "[pool-dispatch] ⛔ FAST-FAIL: $node rc=126 $(( ts - t0 ))s after dispatch (sandbox refused or not executable -- job NOT run; see the node's autodispatch.out): $(printf '%s' "$job" | tr '\n\t' '  ' | cut -c1-160)" >&2 ;;
  esac
}

queue_snapshot() {
  # queue_snapshot -- RUNTIME MODE ONLY (SIM_WORKSPACE set): copy the live queue into a PRIVATE file and print its
  # path. pop_job copies queue text into files the agent can read (.claims, .malformed, .unchecked, pool.running) and
  # to the nodes, and the agent owns every directory on the way to $QUEUE, so pool.queue could be a symlink to one of
  # this user's secrets (~/.ssh/sim_ed25519: its lines would be quarantined as "malformed", readable). The check runs
  # on the OPENED file, so no swap between a check and the read can beat it: it must be a regular file on the
  # workspace's own filesystem (the device of the workspace's parent, the dataset mount, which the agent cannot
  # replace), i.e. a file the agent could read anyway. Anything else: no pop this cycle.
  local snap fd want got kind
  want=$(stat -c %d -- "$(dirname "$SIM_WORKSPACE")" 2>/dev/null) || return 1
  { exec {fd}<"$QUEUE"; } 2>/dev/null || return 1   # no queue file yet: nothing to pop (quietly)
  got=$(stat -L -c %d /dev/stdin <&"$fd" 2>/dev/null); kind=$(stat -L -c %F /dev/stdin <&"$fd" 2>/dev/null)
  case "$kind" in "regular file"|"regular empty file") ;; *) got="" ;; esac
  if [ -z "$got" ] || [ "$got" != "$want" ]; then
    exec {fd}<&-
    echo "[pool-dispatch] ⛔ REFUSING the queue: $QUEUE is not a regular file on the workspace filesystem (device ${got:-?} != ${want}, type ${kind:-?}) -- nothing popped" >&2
    return 1
  fi
  snap=$(mktemp "${TMPDIR:-/tmp}/pool_queue_snap.XXXXXX") || { exec {fd}<&-; return 1; }
  cat <&"$fd" > "$snap"; exec {fd}<&-
  printf '%s\n' "$snap"
}

queue_for_reading() {
  # queue_for_reading -- the file a READ-ONLY seam (--peek-est-gb, --min-queued-est-gb) reads. Legacy mode: $QUEUE.
  # Runtime mode: a private snapshot taken through queue_snapshot's device check, or /dev/null when the queue is
  # missing or refused, so no reader of the workspace queue bypasses that check (2026-09-30 re-review, LOW: these two
  # seams read $QUEUE directly; their output is numeric-only, but a second reader path must not skip the guard).
  if [ -z "${SIM_WORKSPACE:-}" ]; then printf '%s\n' "$QUEUE"; return 0; fi
  queue_snapshot || printf '%s\n' /dev/null
}

queue_done_reading() {   # queue_done_reading <path from queue_for_reading> -- removes a private snapshot, nothing else
  case "$1" in "$QUEUE"|/dev/null|"") ;; *) rm -f -- "$1" ;; esac
}

queue_install() {
  # queue_install <new> <read-copy> -- make <new> the live queue. Legacy mode: <new> is $QUEUE.tmp, renamed over the
  # queue (unchanged). Runtime mode: <new> and <read-copy> are private; the content is WRITTEN into the workspace
  # (never read back from it) and renamed into place, and the private read copy follows it.
  if [ -n "${SIM_WORKSPACE:-}" ]; then
    cp -f "$1" "$2" && cat "$1" > "$QUEUE.tmp" && mv -f "$QUEUE.tmp" "$QUEUE"
    rm -f "$1"
  else
    mv "$1" "$QUEUE"
  fi
}

node_total_budget() {   # GB this node could hold with nothing else running (gate-line hold, see pop_job); 0 = unknown
  local t=$(( ${_NI_TOTAL:-0} - ${POOL_OS_RESERVE_GB:-2} ))
  [ "${_NI_TOTAL:-0}" -gt 0 ] || t=0
  [ -n "${NODE_LIM_MEM[$1]:-}" ] && [ "${NODE_LIM_MEM[$1]}" -lt "$t" ] && t="${NODE_LIM_MEM[$1]}"
  echo "$t"
}

pop_job() {
  # pop_job <max_gb> [node] [node_total_gb] -- atomically take the first non-comment line that fits <max_gb> AND (if [node] is
  # given and the candidate is revision-pinned) whose revision dir already exists on [node]. flock keeps two
  # dispatcher instances from claiming the same job. [node] is OPTIONAL and omitted by test seams that only
  # care about size-based selection (--pop-once) -- when absent, the revision check is skipped entirely
  # (unchanged pre-fix behaviour), matching every caller that never dispatches to a real node.
  local job="" node="${2:-}" qin="$QUEUE" qnew="$QUEUE.tmp"
  exec 9>"$QUEUE.lock"
  flock 9 || return 1
  # RUNTIME MODE: every read below is of a private snapshot (see queue_snapshot); legacy mode reads $QUEUE as before.
  if [ -n "${SIM_WORKSPACE:-}" ]; then
    qin=$(queue_snapshot) || { flock -u 9; return 1; }
    qnew="$qin.new"
    trap 'rm -f "$qin" "$qnew"; trap - RETURN' RETURN
  fi
  # A generic queue producer once wrote GPU-style command-only lines into this
  # timestamped pool queue. Monitoring counted them, while this consumer could
  # never select them. Preserve such records for diagnosis and remove them from
  # the live queue so malformed work cannot masquerade as work in transit.
  local malformed_count
  malformed_count=$(awk -F'\t' '
    $0 !~ /^[[:space:]]*(#|$)/ && !($1 ~ /^[0-9]+$/ && NF > 1) {n++}
    END {print n+0}
  ' "$qin")
  if [ "${malformed_count:-0}" -gt 0 ]; then
    awk -F'\t' '
      $0 !~ /^[[:space:]]*(#|$)/ && !($1 ~ /^[0-9]+$/ && NF > 1)
    ' "$qin" | while IFS= read -r line; do
      printf '%s\t%s\n' "$(date +%s)" "$line"
    done >> "$QUEUE.malformed"
    awk -F'\t' '
      $0 ~ /^[[:space:]]*(#|$)/ || ($1 ~ /^[0-9]+$/ && NF > 1)
    ' "$qin" > "$qnew"
    queue_install "$qnew" "$qin"
    echo "[pool-dispatch] BLOCKED + quarantined $malformed_count malformed queue record(s); use tools/pool_queue.sh" >&2
  fi
  # STALENESS GUARD: an entry older than MAX_AGE is debris, not staged work. Learned immediately -- the first
  # dispatcher run found 69 jobs from an opsweep abandoned days earlier and launched three of them.
  local now
  now=$(date +%s)
  # FIRST FIT, not strict head (2026-09-23): a node with 4 GB free sat idle behind a 5 GB D6 arm while five 0.65 GB
  # vision jobs queued behind it. Take the first fresh line whose declared size fits the node's budget ($1, GB).
  # GATE LINES FIRST (2026-11-02, RULING gates-front-1102): the scan order is gate-seed lines (queue order), then the
  # rest (queue order), each still first-fit by size. ANTI-STARVATION: a gate line that has waited >=
  # POOL_GATE_HOLD_MIN minutes (30), does not fit this node's budget now, but WOULD fit its total ($3 GB: MemTotal
  # minus the OS reserve, or the node's mem_gb limit; unknown/0 = never hold) puts a hold on this node: it takes no
  # non-gate line until the gate line fits, so freed memory accumulates instead of being re-spent on small lines.
  local max_gb="${1:-999}" total_gb="${3:-0}" cand sha want_node ts est hold=0
  local hold_s=$(( ${POOL_GATE_HOLD_MIN:-30} * 60 )) rank   # rank 0 = gate-seed line, 1 = any other line
  while IFS=$'\t' read -r rank ts cand; do
    est=$(job_est_gb "$cand")
    if [ "$rank" = 1 ] && [ "$hold" = 1 ]; then break; fi   # held for a waiting gate line: no non-gate backfill
    if [ "$est" -gt "$max_gb" ]; then
      if [ "$rank" = 0 ] && [ "$est" -le "$total_gb" ] && [ $(( now - ts )) -ge "$hold_s" ]; then
        want_node=$(printf '%s' "$cand" | grep -oE 'pool_node=[A-Za-z0-9_.-]+' | head -1 | cut -d= -f2)
        if [ -z "$want_node" ] || [ -z "$node" ] || [ "$want_node" = "$node" ]; then
          hold=1; echo "[pool-dispatch] ${node:-node}: holding for a waiting gate line (${est} GB > budget ${max_gb} GB, waited $(( (now - ts) / 60 )) min)" >&2
        fi
      fi
      continue
    fi
    # NODE CONSTRAINT (2026-09-25, B2b torn-cell re-run safety: research/coordination/b2b0924_reruns.tsv). A line
    # may declare `pool_node=<name>` anywhere in its text -- typically in the --checked reason, alongside
    # mem_gb=N, the same way that token is already read "anywhere in the line" (see job_est_gb above) -- to
    # require that ONLY that node may pop it. Use case: re-running a cell on a host DIFFERENT from the one whose
    # torn dispatcher-fragment first executed it (A1.4(b) of the B2b prereg's Amendment 2), without falling back
    # to a manual, unlogged ssh launch. OPT-IN: a line with no token is scanned exactly as before. `continue`
    # past a mismatch (leaving it queued for its named node), the same shape as the revision-dir skip below --
    # never popped-and-lost for a node it does not name.
    want_node=$(printf '%s' "$cand" | grep -oE 'pool_node=[A-Za-z0-9_.-]+' | head -1 | cut -d= -f2)
    if [ -n "$want_node" ] && [ -n "$node" ] && [ "$want_node" != "$node" ]; then continue; fi
    # AGENT LINES ONLY SANDBOXED (2026-09-30 security review, HIGH #2). A line tools/agent_gateway_submit.py built
    # carries AGENT_JOB_SRC=sim-agent; it is never popped for a launch that would not be sandboxed (NODE_SANDBOX is
    # set by fill_node's node_sandbox_decision just before this pop), whatever POOL_SANDBOX says -- it stays queued.
    case "$cand" in *"$AGENT_JOB_MARKER"*) [ "${NODE_SANDBOX:-0}" = 1 ] || continue ;; esac
    # REVISION-DIR SEAM (2026-09-23 fix round). A job pinned to `cd ~/derisk-pool/revisions/<sha>` must never be
    # popped for a node that does not have that revision -- that would remove it from the queue (below) with no
    # node able to run it, i.e. lose it. `continue` past it (leaving it in the queue) and keep scanning for a
    # candidate this node CAN run; if none exists this cycle, the outer `if [ -z "$job" ]` returns empty as usual.
    sha=$(printf '%s' "$cand" | grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' | head -1 | sed 's#.*/##')
    if [ -n "$sha" ] && [ -n "$node" ] && ! revision_available_cached "$node" "$sha"; then continue; fi
    job="$cand"; break
  done < <(awk -F'\t' -v now="$now" -v d="${POOL_JOB_MAX_AGE:-43200}" -v cap="${POOL_JOB_MAX_AGE_CAP_H:-240}" \
             "$POOL_FRESH_AWK$POOL_GATE_AWK"' NF>1 && $1+0 >= now - maxage($0) {print (isgate($2) ? 0 : 1) "\t" $1 "\t" $2}' "$qin" \
           | sort -s -t$'\t' -k1,1n 2>/dev/null)   # silence 'write failed: Broken pipe' when the consumer stops early
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
       grep -vF "	$job" "$qin" > "$qnew" 2>/dev/null || true
       queue_install "$qnew" "$qin"
       printf '%s\t%s\n' "$(date +%s)" "$job" >> "$QUEUE.unchecked"
       flock -u 9; printf ''; return 0 ;;
  esac
  local stale
  stale=$(awk -F'\t' -v now="$now" -v d="${POOL_JOB_MAX_AGE:-43200}" -v cap="${POOL_JOB_MAX_AGE_CAP_H:-240}" \
          "$POOL_FRESH_AWK"' NF>1 && $1+0 < now - maxage($0)' "$qin" | wc -l)
  [ "${stale:-0}" -gt 0 ] && echo "[pool-dispatch] SKIPPING $stale stale entr(ies) older than $(( ${POOL_JOB_MAX_AGE:-43200} / 3600 ))h" >&2
  if [ -z "$job" ]; then flock -u 9; printf ''; return 0; fi
  # Keep the full timestamped record, including the checked reason, before the
  # execution copy strips queue metadata. Artifact collection uses this claim
  # to reconstruct the exact command and rationale.
  printf '%s\t%s\n' "$(date +%s)" "$job" >> "$QUEUE.claims"
  if true; then
    grep -vF "	$job" "$qin" > "$qnew" 2>/dev/null || true
    queue_install "$qnew" "$qin"      # unconditional: grep -v exits 1 when it filters everything, and a
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

min_queued_est_gb() {   # smallest declared size among fresh #checked queued jobs; empty if none
  local m="" e cand src
  src=$(queue_for_reading)
  while IFS= read -r cand; do e=$(job_est_gb "$cand"); { [ -z "$m" ] || [ "$e" -lt "$m" ]; } && m="$e"; done \
    < <(awk -F'\t' -v now="$(date +%s)" -v d="${POOL_JOB_MAX_AGE:-43200}" -v cap="${POOL_JOB_MAX_AGE_CAP_H:-240}" \
          "$POOL_FRESH_AWK"' NF>1 && $1+0 >= now - maxage($0) && $0 ~ /#checked:/ {print $2}' "$src" 2>/dev/null)
  queue_done_reading "$src"
  echo "$m"
}

declare -A NODE_BUDGET_LOGGED_AT=()
note_budget_limit_refusal() {
  # note_budget_limit_refusal <node> -- pop_job found nothing for a node node_is_idle passed. Read-only, rate-limited
  # per node; the size test ignores pool_node/revision pins, so it can only say a job of that size was waiting.
  # (1) The node has a POOL_NODE_LIMITS entry whose limits shrank its budget, and the smallest queued job fits the
  #     budget the global settings alone would give but not the per-node one: the limit is what refused it.
  # (2) Otherwise, when even the smallest queued job is larger than this node's budget, say so too. Before
  #     2026-10-20 this case was SILENT for a node with no entry: every STN-sweep line reserved 12 GB, pool41/42 had
  #     less, and the two nodes sat idle about 30 h with 32 lines queued and no log line (RULING pool-stall-1020).
  local node="$1" m now last
  m=$(min_queued_est_gb)
  [[ $m =~ ^[0-9]+$ ]] || return 0
  [ "$m" -gt "$NODE_BUDGET" ] || return 0     # a job of a size that fits was waiting: pins, not size, held it
  if [ -n "${NODE_LIM_SET[$node]:-}" ] && [ "$NODE_LIMIT_GLOBAL_BUDGET" -gt "$NODE_BUDGET" ] \
     && [ "$m" -le "$NODE_LIMIT_GLOBAL_BUDGET" ]; then
    log_node_limit_refusal "$node" "the smallest queued job (${m} GB) exceeds the per-node memory budget (${NODE_BUDGET} GB; the global budget is ${NODE_LIMIT_GLOBAL_BUDGET} GB)"
    return 0
  fi
  now=$(date +%s); last="${NODE_BUDGET_LOGGED_AT[$node]:-}"
  if [ -n "$last" ] && [ $(( now - last )) -lt "$NODE_LIMIT_LOG_RATE_S" ]; then return 0; fi
  NODE_BUDGET_LOGGED_AT[$node]=$now
  echo "[pool-dispatch] $(date '+%H:%M:%S') $node: idle, but the smallest queued job (${m} GB) exceeds this node's memory budget (${NODE_BUDGET} GB), so nothing can dispatch here; size each line's mem_gb from a measured peak x1.3 (next such line for $node in >= ${NODE_LIMIT_LOG_RATE_S}s)" >&2
}

fill_node() {
  # fill_node <node> -- FILL the node to capacity within this cycle (while, not if) — with the per-node cap
  # raised for single-threaded numpy jobs (owner 2026-09-02), a single if-per-cycle would need ~cap cycles to
  # fill. Extracted into its own function (2026-09-23 fix round #3, re-review HIGH) so the --fill-node test seam
  # below calls the EXACT production code path -- including the real `JOB=$(pop_job ...)` command substitution
  # -- rather than a re-typed copy that could silently drift from what the live `while true` loop (at the bottom
  # of this file) runs.
  local NODE="$1" t0
  # PAUSE-MARKER CHECK, EVERY ITERATION (2026-09-25 review, LOW: "the pause does not close the race at its
  # source" -- CYCLE_NODES is built ONCE per cycle in cycle_setup, so a node paused (by
  # tools/aws_stop_safety_lib.sh's pause_dispatch_for_node, for a sync-before-stop window) mid-cycle used to keep
  # being filled for the REST of that cycle; the existing re-check-before-stop in aws_idle_stop.sh/aws_budget.sh
  # only DETECTS a job that landed there, it never prevented one. Checking the CHEAP marker file first (before
  # node_is_idle's ssh round trip) on every iteration -- not just once per cycle -- stops filling THIS node the
  # INSTANT it is paused, closing the race within the same cycle it started in, not just the next one.
  while [ ! -f "$PAUSE_MARK_DIR/$NODE" ] && node_is_idle "$NODE"; do
    node_sandbox_decision "$NODE" || break   # POOL_SANDBOX=on and this node cannot sandbox: nothing is popped
    JOB=$(pop_job "$NODE_BUDGET" "$NODE" "$(node_total_budget "$NODE")")
    if [ -z "$JOB" ]; then note_budget_limit_refusal "$NODE"; break; fi
    echo "[pool-dispatch] $(date '+%H:%M:%S') $NODE <- $JOB"
    if [ "$NODE_SANDBOX" = 1 ]; then
      echo "[pool-dispatch] $NODE: sandboxed launch (bubblewrap: no network, no-new-privs, revision read-only)" >&2
    else
      echo "[pool-dispatch] ⚠ UNSANDBOXED launch on $NODE (POOL_SANDBOX=$POOL_SANDBOX_MODE): full network + the node's sudo -- see tools/pool_job_sandbox.sh" >&2
    fi
    printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$NODE" "$JOB" >> "$CLAIMED"
    # CAPTURE THE EXIT STATUS (2026-07-31). Previously this logged that a job was LAUNCHED and nothing more,
    # so a job that died was indistinguishable from one that succeeded. Nine jobs died instantly on an argparse
    # error and went unnoticed for an hour, because the only evidence of failure sat in autodispatch.out on a
    # node nobody reads. The wrapper appends a timestamped v2 record with a numeric rc and base64-encoded job;
    # the encoding keeps multiline pytest expressions from becoming fake status rows.
    REMOTE_COMMAND=$(remote_launch_command "$JOB" "$NODE_SANDBOX") || {
      echo "[pool-dispatch] failed to encode job for $NODE" >&2
      break
    }
    t0=$(date +%s)   # taken BEFORE the launch: check_fast_fail accepts only a status record written at/after it
    ssh -f -n "${SSH_F[@]}" -o BatchMode=yes "$NODE" "$REMOTE_COMMAND" 2>/dev/null
    printf '%s %s %s\n' "$(date +%s)" "$NODE" "$(job_est_gb "$JOB")" >> "$RESV"
    sleep "${POOL_DISPATCH_LAUNCH_SLEEP:-5}"     # let the launch register before this node's next capacity check
    [ "${POOL_SKIP_FAST_FAIL_CHECK:-0}" = "1" ] || check_fast_fail "$NODE" "$JOB" "$t0"
  done
}

if [ "${1:-}" = "--pop-once" ]; then
  pop_job "${2:-999}" "${3:-}" "${4:-0}"
  exit $?
fi
if [ "${1:-}" = "--reserved-gb" ]; then reserved_gb "$2"; exit 0; fi
if [ "${1:-}" = "--peek-est-gb" ]; then peek_est_gb; exit 0; fi
if [ "${1:-}" = "--min-queued-est-gb" ]; then   # smallest declared size among fresh queued jobs; empty if none
  min_queued_est_gb; exit 0
fi
if [ "${1:-}" = "--node-idle-repeat" ]; then
  # TEST SEAM (2026-10-02): node_is_idle <n> times in ONE process (as the live loop does across polls), printing
  # its return code each time, so the per-node-limit log's in-memory rate limit can be observed.
  [ "$#" -eq 3 ] || { echo "usage: $0 --node-idle-repeat <node> <n>" >&2; exit 2; }
  for _i in $(seq 1 "$3"); do node_is_idle "$2"; echo "$?"; done
  exit 0
fi
if [ "${1:-}" = "--node-budget" ]; then   # live diagnostic: would this node take work, and how many GB?
  if node_is_idle "$2"; then echo "idle budget=${NODE_BUDGET}GB"; else echo "busy/unreachable (budget=${NODE_BUDGET}GB)"; fi; exit 0
fi
if [ "${1:-}" = "--committed-gb" ]; then   # stdin: unique POOL_JOB_ID/POOL_JOB_MEM_GB lines from a node
  committed_from_environ; exit 0
fi
if [ "${1:-}" = "--render-remote-command" ]; then
  # optional 3rd argument `sandbox` renders the sandboxed launch (tools/pool_job_sandbox.sh as the wrapper)
  { [ "$#" -eq 2 ] || { [ "$#" -eq 3 ] && [ "$3" = sandbox ]; }; } \
    || { echo "usage: $0 --render-remote-command '<job>' [sandbox]" >&2; exit 2; }
  if [ "$#" -eq 3 ]; then remote_launch_command "$2" 1; else remote_launch_command "$2"; fi
  exit $?
fi
if [ "${1:-}" = "--sandbox-probe" ]; then
  # TEST SEAM + live diagnostic (2026-09-30): the REAL readiness probe against one node; prints its one-line answer
  # and exits 0 only for POOL_SANDBOX_READY.
  [ "$#" -eq 2 ] || { echo "usage: $0 --sandbox-probe <node>" >&2; exit 2; }
  sandbox_probe "$2"; exit $?
fi
if [ "${1:-}" = "--sandbox-mode" ]; then echo "$POOL_SANDBOX_MODE"; exit 0; fi
if [ "${1:-}" = "--write-sandbox-state" ]; then   # TEST SEAM: the live loop's per-cycle state write, exactly
  write_sandbox_state; exit $?
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
if [ "${1:-}" = "--check-fast-fail" ]; then
  # TEST SEAM (2026-09-25): exercises the REAL check_fast_fail ssh call (same argv construction, including
  # SSH_F) against one node/job pair, without a real node -- so a stubbed `ssh` on PATH can assert both the
  # exact read-only grep probe it makes AND that a matching rc=127/2 record is logged loudly to stderr.
  # <t0> is the dispatch time fill_node would pass (records older than it are earlier attempts, not this one).
  { [ "$#" -eq 3 ] || [ "$#" -eq 4 ]; } || { echo "usage: $0 --check-fast-fail <node> <job> [<t0>]" >&2; exit 2; }
  check_fast_fail "$2" "$3" "${4:-}"; exit 0
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
  printf '%s %s\n' "$NODES" "$(extra_nodes)" | valid_node_names
  exit 0
fi

# RUNTIME MODE NEEDS ITS CONFINEMENT. The loop appends to workspace files (pool.running, .claims, the reservations),
# and a symlink the agent planted there is followed. sim-pool-dispatch.service makes this user's own code, keys and
# config read-only (ProtectHome=read-only), so such a write cannot land in them. Run by hand as sim, it could: a
# planted pool.running -> <runtime>/tools/pool_sync.sh would append queue text to code sim runs. So when the code
# root is writable, the loop does not start. The one-shot seams above do not append to the workspace.
if [ -n "${SIM_WORKSPACE:-}" ] && [ -w "$ROOT" ]; then
  echo "[pool-dispatch] ⛔ REFUSING to start: runtime mode (SIM_WORKSPACE) with a writable code root ($ROOT) means no confinement -- run it as sim-pool-dispatch.service" >&2
  exit 1
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

echo "[pool-dispatch] started $(date '+%H:%M:%S') | queue=$QUEUE | poll=${POLL}s | nodes=$NODES (+ any in $EXTRA_NODES_FILE, re-read each cycle) | POOL_SANDBOX=$POOL_SANDBOX_MODE | POOL_NODE_LIMITS=${POOL_NODE_LIMITS:-none}"
while true; do
  write_sandbox_state   # read by tools/agent_gateway.sh: pool-submit only while this says mode=on (and is fresh)
  cycle_setup
  for NODE in $CYCLE_NODES; do
    fill_node "$NODE"
  done
  sleep "$POLL"
done
