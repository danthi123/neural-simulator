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

sync_node_before_stop() {   # sync_node_before_stop <node-name> <ip> <key> -- pull this node's results BEFORE
  # it is stopped (2026-09-25, incident-driven: pool1 was idle-stopped at 13:06Z with two finished DA LTM-on
  # seeds' last arm+seed JSON written after the last routine pool_sync -- stranded on the stopped node's disk
  # until the owner restarted it by hand ~55 min later). Returns 0 iff the pull is VERIFIED to have succeeded;
  # the caller must not stop the instance on a non-zero return (the root volume is not what's at risk here --
  # unlike aws_pool_node.sh down's DeleteOnTermination=true case -- but a `stop`+cold node still leaves any
  # result written after this point unreachable until someone notices and restarts it, exactly what happened).
  #
  # Prefers tools/pool_sync.sh restricted to this one node (POOL_SYNC_STRICT=1) when the node has a registered
  # dispatch alias in .pool_ssh_config -- identical exclusions/isolated-revision handling to the routine 15-min
  # cadence sync, so this call can never diverge from what "synced" already means elsewhere in this repo. Falls
  # back to the SAME plain rsync pool_sync.sh performs when this node is not a registered pool-dispatch alias
  # (e.g. the single-instance `.aws_gpu` CPU-verify lane, never wired into .pool_ssh_config) -- using the ip/key
  # this script already resolved for the no-runner SSH check, so a non-pool AWS lane is covered too, not just
  # aws_pool_node.sh-managed nodes.
  local node="$1" ip="$2" key="$3"
  if [ -z "$ip" ] || [ -z "$key" ] || [ ! -f "$key" ]; then
    echo "$(date -u '+%FT%TZ') [aws_idle_stop] $node: no verified ssh key/ip on hand -- cannot sync, NOT stopping this cycle" | tee -a "$LOG"
    return 1
  fi
  if [ -n "$node" ] && [ -f "$POOL_SSH_CFG" ] && grep -q "^Host $node\$" "$POOL_SSH_CFG" 2>/dev/null; then
    # </dev/null (2026-09-25 review, HIGH #2): pool_sync.sh's own ssh calls must never inherit the CALLER's
    # stdin -- see the main dispatch loop's fd-3 fix below for the concrete production incident this class of
    # bug caused elsewhere in this same script. This call sits inside the per-instance loop body, so give it
    # its own belt-and-suspenders guard even though the loop's fd-3 fix already keeps `$ids` off fd 0.
    if POOL_SSH_CONFIG="$POOL_SSH_CFG" POOL_NODES="$node" POOL_SYNC_STRICT=1 \
         bash "$ROOT/tools/pool_sync.sh" </dev/null >>"$LOG" 2>&1; then
      return 0
    fi
    echo "$(date -u '+%FT%TZ') [aws_idle_stop] $node: pool_sync --strict FAILED -- NOT stopping this cycle (will retry)" | tee -a "$LOG"
    return 1
  fi
  # CANDIDATE REMOTE DIRS (2026-09-25 review, HIGH #1): a node with no .pool_ssh_config alias is NOT necessarily
  # a tools/aws_pool_node.sh-managed pool node -- the single-instance `.aws_gpu`/.aws_cpuN lanes (tools/
  # aws_cpu_launch.sh + tools/aws_cpu_provision.sh / tools/aws_provision.sh) rsync code to ~/sim, never
  # ~/derisk-pool/sim (see those scripts' own `~/sim/` rsync targets). The OLD single-path fallback always tried
  # ~/derisk-pool/sim: on those lanes the source directory does not exist, rsync exits 23, sync_node_before_stop
  # logged "NOT stopping this cycle" on EVERY cycle forever (only the $50/day cap ever stopped them), and their
  # results under ~/sim were never pulled. FIX: `ssh test -d` each known layout's PROJECT ROOT (not the deeper
  # raw/ results dir -- a genuinely fresh node with zero results yet would otherwise read as a sync FAILURE) and
  # pull from whichever roots actually exist on THIS node -- a node provisioned any other way is still coverable
  # by extending this list, with no change to the decision logic below.
  local remote_dirs rd root status found=0
  if [ -n "${POOL_REMOTE_DIR:-}" ]; then
    remote_dirs="$POOL_REMOTE_DIR"
  else
    remote_dirs="derisk-pool/sim/research/findings/raw/ sim/research/findings/raw/"
  fi
  mkdir -p "$ROOT/research/findings/raw"
  for rd in $remote_dirs; do
    root="${rd%/research/findings/raw/}"
    # ONE round trip per candidate: does the project root exist, and (only if so) does its raw/ results dir
    # exist yet. `-n` (2026-09-25 review, HIGH #2): every remote call in this script must not read the caller's
    # stdin -- see the main dispatch loop's own fd-3 fix below for why.
    status=$(ssh -n -i "$key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
        ubuntu@"$ip" "if [ -d ~/$root ]; then if [ -d ~/$rd ]; then echo has_raw; else echo root_only; fi; else echo no_root; fi" 2>/dev/null)
    case "$status" in
      no_root|"") continue ;;   # this layout is not this node's -- or the probe itself failed; try the next candidate
      root_only) found=1; continue ;;   # this IS the node's layout, but nothing has been written yet -- nothing to pull
      has_raw)
        found=1
        if ! timeout 180 rsync -au --exclude='*.log' --exclude='_provenance/' \
            -e "ssh -i $key -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes" \
            "ubuntu@$ip:~/$rd" "$ROOT/research/findings/raw/" >>"$LOG" 2>&1; then
          echo "$(date -u '+%FT%TZ') [aws_idle_stop] $node: fallback rsync of ~/$rd FAILED -- NOT stopping this cycle (will retry)" | tee -a "$LOG"
          return 1
        fi
        ;;
    esac
  done
  if [ "$found" = 0 ]; then
    echo "$(date -u '+%FT%TZ') [aws_idle_stop] $node: no known project layout found on this node (checked: $remote_dirs) -- cannot verify sync, NOT stopping this cycle" | tee -a "$LOG"
    return 1
  fi
  return 0
}

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
    if echo "$cw" | "$PY" "$ROOT/tools/aws_cost_lib.py" cpu-idle --threshold "$IDLE_CPU_PCT" 2>>"$LOG"; then
      cpu_idle=0
      idle_signal="CloudWatch (sustained, >= ${IDLE_MINUTES}m of 5-min datapoints)"
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
    if ssh -n -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
         ubuntu@"$ip" "pgrep -f '[r]esearch\.runners' >/dev/null 2>&1" 2>/dev/null; then
      runner_flag="--runner-active true"
    else
      runner_flag="--runner-active false"
    fi
  fi

  if "$PY" "$ROOT/tools/aws_cost_lib.py" idle "$([ "$cpu_idle" = 0 ] && echo 1 || echo 0)" $runner_flag; then
    # SYNC-BEFORE-STOP (2026-09-25, incident-driven -- see sync_node_before_stop's own comment above). A `stop`
    # is non-destructive to the EBS volume, but the NODE goes cold and unreachable the instant it stops, so
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
      if [ "$have_ssh" = 1 ]; then
        if ssh -n -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
             ubuntu@"$ip" "pgrep -f '[r]esearch\.runners' >/dev/null 2>&1" 2>/dev/null; then
          recheck_runner_flag="--runner-active true"
        else
          recheck_runner_flag="--runner-active false"
        fi
      fi
      if [ "$recheck_runner_flag" != "--runner-active false" ]; then
        echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid: a runner appeared during the sync (re-check after sync, before stop) — NOT stopping this cycle" | tee -a "$LOG"
      else
        echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid idle >= ${IDLE_MINUTES}m (signal: $idle_signal), no runner (re-checked after sync) — STOPPING" | tee -a "$LOG"
        aws ec2 stop-instances --region "$REGION" --instance-ids "$iid" --output text 2>&1 | tee -a "$LOG"
      fi
    else
      echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid idle but sync-before-stop FAILED — NOT stopping this cycle (will retry)" | tee -a "$LOG"
    fi
  fi
done 3<<<"$ids"
exit 0
