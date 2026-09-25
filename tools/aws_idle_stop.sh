#!/usr/bin/env bash
# aws_idle_stop.sh — stop any RUNNING project EC2 instance that has been idle for AWS_IDLE_MINUTES (default
# 20) AND has no research runner process on it. Part of the AWS budget guard (owner-approved 2026-09-23),
# meant to run on a schedule via tools/install_aws_guard_timer.sh (systemd --user, every 10 min).
#
# Non-destructive: this only ever `stop`s (never `terminate`s) — the EBS root volume persists and the
# instance can be restarted. `stop` was chosen deliberately: a wrongly-stopped instance costs a restart, a
# wrongly-NOT-stopped instance costs money, so being a little eager to stop is fine here (unlike terminate).
#
# Idle signal, in priority order (both the CPU signal AND the no-runner check must hold; either being
# INCONCLUSIVE means "keep" — see tools/aws_cost_lib.py::idle_decision for the reasoning):
#   1. CloudWatch CPUUtilization — basic monitoring gives free 5-min datapoints, no agent needed. Idle iff
#      there is at least one datapoint in the last AWS_IDLE_MINUTES and every one is below AWS_IDLE_CPU_PCT.
#   2. If CloudWatch has no datapoints yet (brand-new instance / API hiccup), fall back to SSH `uptime`
#      1-minute load average (only possible for the instance recorded in research/queue/.aws_gpu, since
#      that's the only place this repo's tooling keeps a per-instance SSH key today).
#   3. No-runner check, always required in addition to (1) or (2): SSH `pgrep -f '[r]esearch\.runners'` — the
#      bracket trick keeps the pgrep/ssh invocation itself from matching its own argv.
# All arithmetic (idle-from-samples, idle-from-loadavg, the final stop/keep call) is in tools/aws_cost_lib.py
# so it is unit-testable without a real AWS/SSH round-trip — see tests/test_aws_cost_lib.py.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-python3}"
REGION="${AWS_REGION:-us-east-1}"
IDLE_MINUTES="${AWS_IDLE_MINUTES:-20}"
IDLE_CPU_PCT="${AWS_IDLE_CPU_PCT:-10}"
# AWS_IDLE_STOP_LOG / AWS_GPU_STATE_FILE overrides exist for tests (tests/test_aws_budget_guard_workflow.py)
# so they never write into or read from the SHARED production queue dir (other sessions/agents touch
# research/queue/ concurrently; see the worktree environment note on this).
LOG="${AWS_IDLE_STOP_LOG:-$ROOT/research/queue/aws_idle_stop.log}"
GPU_STATE="${AWS_GPU_STATE_FILE:-$ROOT/research/queue/.aws_gpu}"
# POOL_SSH_CONFIG override exists for tests, same reasoning as the log/state overrides above -- it must never
# resolve to the SHARED production .pool_ssh_config (other sessions/agents write it concurrently).
POOL_SSH_CFG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"

# sync_node_before_stop / pause_dispatch_for_node / resume_dispatch_for_node (2026-09-25 review, fix round 3):
# EXTRACTED into tools/aws_stop_safety_lib.sh so tools/aws_budget.sh's hard-cap `enforce` can share them rather
# than diverge -- the review's own still-open item was that `enforce` never got a sync-before-stop treatment at
# all. Depends on ROOT/LOG/POOL_SSH_CFG (all set above) and sets/uses EXTRA_NODES_FILE for the pause/resume pair.
source "$ROOT/tools/aws_stop_safety_lib.sh"

# RE-REGISTER ANY STALE PAUSE, AT STARTUP (2026-09-25 review, MEDIUM: "the pause/resume pair has no durable
# restore"). If a PREVIOUS cycle's process was killed mid-sync (SIGKILL, OOM, a reboot, systemctl stop) after
# pause_dispatch_for_node removed a node's registration but before its own resume_dispatch_for_node could run,
# that node is permanently missing from dispatch -- every later cycle reads was_reg=0 for it and never notices.
# Cheap (a directory listing); safe to run unconditionally every cycle.
reregister_stale_paused_nodes

json=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=instance-state-name,Values=running,pending" --output json 2>/dev/null)
if [ -z "$json" ]; then
  echo "$(date -u '+%FT%TZ') [aws_idle_stop] could not reach AWS (describe-instances failed) — skipping this cycle" | tee -a "$LOG"
  exit 0
fi

ids=$("$PY" "$ROOT/tools/aws_cost_lib.py" project-ids <<<"$json")
if [ -z "$ids" ]; then
  exit 0   # nothing running -> nothing to do (the common case; not logged to avoid log spam every 10 min)
fi
# LAUNCH GRACE (2026-09-24): never judge an instance younger than the idle window. With no CloudWatch data yet the
# check fell back to an instant SSH load reading and stopped two freshly launched battery instances mid-provision.
young=$("$PY" "$ROOT/tools/aws_cost_lib.py" young-ids --minutes "$IDLE_MINUTES" <<<"$json")

# The only per-instance SSH key this repo's tooling durably records today (see CLAUDE.md "the AWS lane" and
# tools/aws_gpu.sh / tools/aws_cpu_launch.sh, which share this single state file — at most one instance is
# ever live via this repo's own launch scripts at a time).
state_iid=""; state_key=""; state_file="$GPU_STATE"
if [ -f "$GPU_STATE" ]; then
  state_iid=$(awk -F= '/^instance=/{print $2}' "$GPU_STATE" 2>/dev/null)
  state_key=$(awk -F= '/^key=/{print $2}' "$GPU_STATE" 2>/dev/null)
fi

# fd 3 FOR THE LOOP (2026-09-25 review, HIGH #2): `while read iid; do ... ssh ...; done <<<"$ids"` puts $ids on
# fd 0 for the ENTIRE loop body, including every ssh/rsync/pool_sync call inside it -- and ssh, even when its
# remote command never reads stdin itself, still drains whatever local stdin it inherits (the same class of bug
# 096dfdae0 fixed in the dispatcher's revision_available probe). With N running instances, the FIRST instance's
# ssh calls drained the rest of $ids off fd 0, so `read -r iid` hit EOF and the loop silently ended after one
# instance -- reproduced with a stdin-draining ssh stub, and matching the production aws_idle_stop.log (exactly
# one load1 line per cycle while pool1 and pool2 both ran; pool2 was never even reached while pool1 sorted
# first). Reading the id list from a SEPARATE fd (3, never touched by anything in the loop body) makes this
# structurally impossible regardless of any one remote call's own stdin habits -- kept in addition to (not
# instead of) `-n`/`</dev/null` on every remote call below, per the review's "AND" (belt and suspenders: a
# future remote call added inside this loop without `-n` still cannot swallow $ids, only its own local stdin).
while IFS= read -r iid <&3; do
  [ -z "$iid" ] && continue
  if printf '%s\n' "$young" | grep -qx "$iid"; then
    echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid launched < ${IDLE_MINUTES}m ago — within launch grace, keep" >> "$LOG"
    continue
  fi

  # We only have a durable per-instance SSH key for the one lane this repo's launch scripts record (see the
  # comment above) — resolve its IP once and reuse it for both the CPU load-average fallback and the
  # no-runner check below.
  have_ssh=0; ip=""
  # MULTI-INSTANCE (2026-09-23): parallel batteries record extra instances in research/queue/.aws_cpu2, .aws_cpu3, ...
  # With only .aws_gpu consulted, every other instance was unverifiable -> treated as busy -> NEVER stopped (an idle
  # r7i.4xlarge ran ~30 min past its last job). Find the state file that records THIS instance and use its key.
  if [ "$iid" != "$state_iid" ] && [ -z "${AWS_GPU_STATE_FILE:-}" ]; then
    for sf in "$ROOT"/research/queue/.aws_*; do
      [ -f "$sf" ] || continue
      if grep -q "^instance=$iid\$" "$sf" 2>/dev/null; then
        state_iid="$iid"; state_key=$(awk -F= '/^key=/{print $2}' "$sf" 2>/dev/null); state_file="$sf"; break
      fi
    done
  fi
  # NODE NAME (2026-09-25, for sync-before-stop below): every state file this repo's AWS tooling writes is
  # `research/queue/.aws_<node-name>` (aws_pool_node.sh's own convention -- `.aws_pool1`, `.aws_pool2`, ... --
  # and aws_cpu_launch.sh's single-instance default `.aws_gpu`), so its basename minus the `.aws_` prefix IS
  # the node's ssh-dispatch alias when one is registered, and a harmless best-effort label ("gpu") otherwise.
  node_name=$(basename "$state_file" 2>/dev/null); node_name="${node_name#.aws_}"
  if [ "$iid" = "$state_iid" ] && [ -n "$state_key" ] && [ -f "$state_key" ]; then
    ip=$(aws ec2 describe-instances --instance-ids "$iid" --region "$REGION" \
          --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
    [ -n "$ip" ] && [ "$ip" != "None" ] && have_ssh=1
  fi

  end=$(date -u +%Y-%m-%dT%H:%M:%S)
  start=$(date -u -d "-${IDLE_MINUTES} minutes" +%Y-%m-%dT%H:%M:%S 2>/dev/null \
          || date -u -v-"${IDLE_MINUTES}"M +%Y-%m-%dT%H:%M:%S 2>/dev/null)
  cw=$(aws cloudwatch get-metric-statistics --region "$REGION" --namespace AWS/EC2 --metric-name CPUUtilization \
        --dimensions Name=InstanceId,Value="$iid" --start-time "$start" --end-time "$end" --period 300 \
        --statistics Average --output json 2>/dev/null)

  cpu_idle=1
  # IDLE-SIGNAL STRENGTH (2026-09-25 review, MEDIUM): the two branches below are NOT equally strong evidence --
  # CloudWatch gives one datapoint per 5 min, so a CONCLUSIVE read spans the whole ${IDLE_MINUTES}-minute window;
  # the SSH fallback is a SINGLE 1-minute load-average sample taken at the instant this cycle happens to run.
  # Recorded here so the eventual STOPPING log line says which kind of evidence justified the stop, rather than
  # implying both are "idle >= ${IDLE_MINUTES}m" alike.
  idle_signal="none"
  if [ -n "$cw" ] && echo "$cw" | "$PY" "$ROOT/tools/aws_cost_lib.py" cw-has-data 2>>"$LOG"; then
    # CloudWatch answered CONCLUSIVELY (has datapoints) -> trust it, idle or busy, no SSH fallback needed.
    # SAMPLE COUNT, NOT "sustained" (2026-09-25 review, LOW): cw-has-data accepts a SINGLE datapoint
    # (tools/aws_cost_lib.py::cw-has-data just checks the list is non-empty) and CloudWatch itself lags 5-10
    # minutes -- the old label "sustained, >= ${IDLE_MINUTES}m of 5-min datapoints" overstated the evidence for
    # exactly that single-datapoint case. cpu-idle already prints "samples=N idle=..." to stderr; capture it
    # once (combined stdout+stderr -- the command itself never writes to stdout) and quote the REAL count.
    cw_idle_out=$(echo "$cw" | "$PY" "$ROOT/tools/aws_cost_lib.py" cpu-idle --threshold "$IDLE_CPU_PCT" 2>&1)
    cw_idle_rc=$?
    echo "$cw_idle_out" >>"$LOG"
    cw_n_samples=$(printf '%s' "$cw_idle_out" | grep -oE 'samples=[0-9]+' | head -1 | cut -d= -f2)
    if [ "$cw_idle_rc" -eq 0 ]; then
      cpu_idle=0
      idle_signal="CloudWatch (${cw_n_samples:-an unstated number of} x 5-min average(s) over the last ${IDLE_MINUTES}m, all < ${IDLE_CPU_PCT}%)"
    fi
  elif [ "$have_ssh" = 1 ]; then
    # CloudWatch had NO datapoints (inconclusive) -> fall back to SSH load-average. `-n` on every remote call in
    # this loop (2026-09-25 review, HIGH #2): belt-and-suspenders alongside the fd-3 loop-read fix above -- see
    # its comment for the production incident (only the first instance's checks ran a cycle) this class of bug
    # caused. `-n` makes ssh itself never touch local stdin, independent of whether the remote command does.
    la=$(ssh -n -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
          ubuntu@"$ip" "uptime" 2>/dev/null)
    if [ -n "$la" ]; then
      load1=$(echo "$la" | grep -oE 'load average: [0-9.]+' | awk '{print $3}')
      ncpu=$(ssh -n -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
              ubuntu@"$ip" "nproc" 2>/dev/null)
      if [ -n "$load1" ] && [ -n "$ncpu" ] \
         && "$PY" "$ROOT/tools/aws_cost_lib.py" loadavg-idle --load1 "$load1" --ncpu "$ncpu" \
              --threshold "$IDLE_CPU_PCT" 2>>"$LOG"; then
        cpu_idle=0
        idle_signal="SSH load-average (single 1-minute sample, NOT a sustained ${IDLE_MINUTES}m signal)"
      fi
    fi
  fi

  runner_flag="--runner-active true"   # inconclusive-by-default: cannot SSH -> assume a runner IS active (keep)
  if [ "$have_ssh" = 1 ]; then
    # CAPTURE THE REAL RC (2026-09-25 review, MEDIUM): pgrep's own "not found" (rc=1) and ssh itself FAILING TO
    # CONNECT (rc=255, e.g. a transient network blip) used to fall into the SAME `else` branch below and read
    # identically as "no runner" -- an unreachable check was stopping an instance this script could not actually
    # verify was idle. 0 = pgrep found a match (runner active); 1 = pgrep ran and found nothing (genuinely no
    # runner); anything else (255 = ssh connection/auth failure, 124 a local timeout, ...) is INCONCLUSIVE and
    # must default to "active" (keep), matching this whole script's existing bias toward NOT stopping on any
    # other inconclusive signal.
    ssh -n -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
         ubuntu@"$ip" "pgrep -f '[r]esearch\.runners' >/dev/null 2>&1" 2>/dev/null
    runner_rc=$?
    case "$runner_rc" in
      0) runner_flag="--runner-active true" ;;
      1) runner_flag="--runner-active false" ;;
      *) runner_flag="--runner-active true"
         echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid: runner-check ssh rc=$runner_rc (not 0/1) -- INCONCLUSIVE, treated as active" >>"$LOG" ;;
    esac
  fi

  if "$PY" "$ROOT/tools/aws_cost_lib.py" idle "$([ "$cpu_idle" = 0 ] && echo 1 || echo 0)" $runner_flag; then
    # TAKE THE NODE OUT OF DISPATCH FOR THE SYNC WINDOW (2026-09-25 review, still-open item: "the node was not
    # taken out of dispatch before the sync"). The re-check right before stop (below) only DETECTS a job that
    # landed while the sync (which can take minutes) was running; this PREVENTS one from being handed out in the
    # first place by removing the node from tools/pool_autodispatch.sh's pool for the duration, closing the race
    # at its source. Restored unconditionally once the decision is made, whether or not this cycle actually
    # stops the instance (registration tracks 'up'/'down', not this one cycle's outcome). A no-op (was_reg=0)
    # for any node never registered for dispatch to begin with (e.g. the single-instance `.aws_gpu` lane).
    was_reg=$(pause_dispatch_for_node "$node_name")
    # TRAP, AT MINIMUM (2026-09-25 review, MEDIUM): if THIS process is killed (SIGINT/SIGTERM) or exits for any
    # other reason before one of the explicit resume_dispatch_for_node calls below runs, resume it here instead
    # -- the graceful-kill half of the durable-restore fix (reregister_stale_paused_nodes above is the half that
    # also covers SIGKILL/OOM/reboot, which no trap can catch). The command string bakes in the CURRENT
    # $node_name/$was_reg BY VALUE (printf %q), not by reference, so a later loop iteration overwriting those
    # variables can never leak into a trap armed for an earlier iteration's node. Disarmed right after each of
    # the explicit resume calls below so a normal exit never double-fires it (harmless if it did -- resume is
    # idempotent -- but disarming keeps the intent clear).
    # `; exit` (no explicit code) matters for INT/TERM: bash does NOT auto-terminate a script after running a
    # trap it set for those signals -- without this, a SIGTERM would run the resume cleanup and then the script
    # would simply keep going, which defeats the point of sending it a termination signal in the first place
    # (e.g. `systemctl stop`). For the plain EXIT case this just re-exits with the same $? already in flight.
    trap "resume_dispatch_for_node $(printf '%q' "$node_name") $(printf '%q' "$was_reg"); exit" EXIT INT TERM
    # SYNC-BEFORE-STOP (2026-09-25, incident-driven -- see sync_node_before_stop's own comment). A `stop` is
    # non-destructive to the EBS volume, but the NODE goes cold and unreachable the instant it stops, so
    # anything written after the last routine pool_sync between here and whenever someone next notices and
    # restarts it is effectively stranded exactly as pool1's last two DA-probe seeds were. Pull first, verify,
    # and only stop once that pull is CONFIRMED -- an unverifiable/failed sync means "not stopping this cycle",
    # never "stop anyway", so the guard fails toward keeping the (billing) instance up, matching this whole
    # script's existing bias toward NOT stopping on any other inconclusive signal.
    if sync_node_before_stop "$node_name" "$ip" "$state_key"; then
      # RE-CHECK RIGHT BEFORE STOP (2026-09-25 review, MEDIUM): the sync above (a strict pool_sync -- main pull
      # + ssh ls + one rsync per isolated revision, up to 180s EACH) can take anywhere from ~1s to several
      # minutes, widening the original pgrep check's check-to-stop window by the same amount. A job can land in
      # that exact window (5b5ea1b7 did, at 09:59:54, right as a revision finished provisioning) -- stopping now
      # would kill it after it has already been removed from the queue. Re-run the SAME no-runner probe used
      # above, immediately before the stop-instances call, so the decision is made on freshness matching the
      # actual action, not on a reading that may now be several minutes stale.
      recheck_runner_flag="--runner-active true"   # same inconclusive-by-default bias as the original check
      recheck_note="a runner appeared during the sync"
      if [ "$have_ssh" = 1 ]; then
        # timeout 30 (2026-09-25 review, LOW/INFO): ConnectTimeout=10 only bounds the TCP/auth handshake -- a
        # remote pgrep that hangs (a wedged shell, a stuck /proc read) after the connection is up would block
        # this re-check (and everything after it in this cycle) indefinitely. A timed-out call falls into the
        # same `*)` INCONCLUSIVE-keep branch below as any other non-0/1 rc, so this changes no decision logic.
        timeout 30 ssh -n -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
             ubuntu@"$ip" "pgrep -f '[r]esearch\.runners' >/dev/null 2>&1" 2>/dev/null
        recheck_rc=$?
        case "$recheck_rc" in
          0) recheck_runner_flag="--runner-active true"; recheck_note="a runner appeared during the sync" ;;
          1) recheck_runner_flag="--runner-active false" ;;
          # Same rc=255-is-not-rc=1 fix as the original check above, applied to the re-check.
          *) recheck_runner_flag="--runner-active true"
             recheck_note="the re-check's ssh was INCONCLUSIVE (rc=$recheck_rc, not 0/1)" ;;
        esac
      fi
      if [ "$recheck_runner_flag" != "--runner-active false" ]; then
        resume_dispatch_for_node "$node_name" "$was_reg"; trap - EXIT INT TERM
        echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid: $recheck_note (re-check after sync, before stop) — NOT stopping this cycle" | tee -a "$LOG"
      else
        # judged idle (2026-09-25 review, INFO: this line used to say "idle >= ${IDLE_MINUTES}m" unconditionally
        # ahead of $idle_signal's own text, which for the SSH-loadavg branch immediately contradicts itself with
        # "single 1-minute sample, NOT a sustained ${IDLE_MINUTES}m signal" one clause later. "judged idle" makes
        # no claim about HOW LONG; $idle_signal alone states the evidence honestly, sustained or not.
        echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid judged idle (signal: $idle_signal), no runner (re-checked after sync) — STOPPING" | tee -a "$LOG"
        aws ec2 stop-instances --region "$REGION" --instance-ids "$iid" --output text 2>&1 | tee -a "$LOG"
        # RESUME AFTER stop-instances, NOT BEFORE (2026-09-25 review, LOW: "resume runs BEFORE stop-instances...
        # reopens the window" -- the OLD ordering restored dispatch eligibility for a node that was about to be
        # (but had not yet been) stopped, so a job could land in the gap between resume and the actual
        # stop-instances call, then be killed under it seconds later. The instance is already stopping/gone by
        # the time dispatch sees this node again on its NEXT cycle either way, so there is no correctness reason
        # to resume any earlier than this.
        resume_dispatch_for_node "$node_name" "$was_reg"; trap - EXIT INT TERM
      fi
    else
      resume_dispatch_for_node "$node_name" "$was_reg"; trap - EXIT INT TERM
      echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid idle but sync-before-stop FAILED — NOT stopping this cycle (will retry)" | tee -a "$LOG"
    fi
  fi
done 3<<<"$ids"
exit 0
