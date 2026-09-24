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
state_iid=""; state_key=""
if [ -f "$GPU_STATE" ]; then
  state_iid=$(awk -F= '/^instance=/{print $2}' "$GPU_STATE" 2>/dev/null)
  state_key=$(awk -F= '/^key=/{print $2}' "$GPU_STATE" 2>/dev/null)
fi

while IFS= read -r iid; do
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
        state_iid="$iid"; state_key=$(awk -F= '/^key=/{print $2}' "$sf" 2>/dev/null); break
      fi
    done
  fi
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
  if [ -n "$cw" ] && echo "$cw" | "$PY" "$ROOT/tools/aws_cost_lib.py" cw-has-data 2>>"$LOG"; then
    # CloudWatch answered CONCLUSIVELY (has datapoints) -> trust it, idle or busy, no SSH fallback needed.
    if echo "$cw" | "$PY" "$ROOT/tools/aws_cost_lib.py" cpu-idle --threshold "$IDLE_CPU_PCT" 2>>"$LOG"; then
      cpu_idle=0
    fi
  elif [ "$have_ssh" = 1 ]; then
    # CloudWatch had NO datapoints (inconclusive) -> fall back to SSH load-average.
    la=$(ssh -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
          ubuntu@"$ip" "uptime" 2>/dev/null)
    if [ -n "$la" ]; then
      load1=$(echo "$la" | grep -oE 'load average: [0-9.]+' | awk '{print $3}')
      ncpu=$(ssh -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
              ubuntu@"$ip" "nproc" 2>/dev/null)
      if [ -n "$load1" ] && [ -n "$ncpu" ] \
         && "$PY" "$ROOT/tools/aws_cost_lib.py" loadavg-idle --load1 "$load1" --ncpu "$ncpu" \
              --threshold "$IDLE_CPU_PCT" 2>>"$LOG"; then
        cpu_idle=0
      fi
    fi
  fi

  runner_flag="--runner-active true"   # inconclusive-by-default: cannot SSH -> assume a runner IS active (keep)
  if [ "$have_ssh" = 1 ]; then
    if ssh -i "$state_key" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
         ubuntu@"$ip" "pgrep -f '[r]esearch\.runners' >/dev/null 2>&1" 2>/dev/null; then
      runner_flag="--runner-active true"
    else
      runner_flag="--runner-active false"
    fi
  fi

  if "$PY" "$ROOT/tools/aws_cost_lib.py" idle "$([ "$cpu_idle" = 0 ] && echo 1 || echo 0)" $runner_flag; then
    echo "$(date -u '+%FT%TZ') [aws_idle_stop] $iid idle >= ${IDLE_MINUTES}m, no runner — STOPPING" | tee -a "$LOG"
    aws ec2 stop-instances --region "$REGION" --instance-ids "$iid" --output text 2>&1 | tee -a "$LOG"
  fi
done <<<"$ids"
exit 0
