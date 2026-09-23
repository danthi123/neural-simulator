#!/usr/bin/env bash
# aws_budget.sh — ENFORCE the AWS daily spend cap for this project's EC2 instances BY TOOLING, not memory
# (owner-approved 2026-09-23: on-demand AWS instances for 6-seed CPU batteries; cap raised same day from an
# initial $15/day to $50/day — see AWS_DAILY_CAP_USD below). This is the single source of truth for "would
# spending more right now take us over budget," and it is wired as a hard pre-launch refusal into
# tools/aws_cpu_launch.sh and the `launch` path of tools/aws_gpu.sh.
#
# All arithmetic lives in tools/aws_cost_lib.py (pure, unit-tested — see tests/test_aws_cost_lib.py); this
# script only talks to the real AWS CLI and prints/acts on that module's verdict. See aws_cost_lib.py's
# docstring for the spend model + its documented approximation (no AWS Cost Explorer — it lags ~24h and
# costs per call, per the build brief).
#
# status/check/enforce below are LEDGERED (tools/aws_spend_ledger.py, fixed 2026-09-23): every call records
# each project instance's cost-so-far-today, and reports the MAX ever recorded per instance today, so a
# stopped/terminated instance can never lose spend it already accrued (see aws_cost_lib.py's
# estimate_spend_with_ledger / stop_candidates_with_ledger). AWS_SPEND_LEDGER overrides the ledger path
# (tests use this — never point it at the shared production queue dir).
#
# Usage:
#   tools/aws_budget.sh check [instance-type]   # exit 1 if we are at/over cap (or would be after +1h of
#                                                #   `instance-type`) — refuses a launch
#   tools/aws_budget.sh status                  # human-readable: today's spend estimate + project instances
#   tools/aws_budget.sh enforce                 # STOP every running project instance once the cap is reached
#
# AWS_DAILY_CAP_USD overrides the default cap. AWS_REGION overrides the default region (us-east-1).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON:-python3}"
REGION="${AWS_REGION:-us-east-1}"
CAP="${AWS_DAILY_CAP_USD:-50}"
# AWS_BUDGET_LOG override exists for tests -- keeps them from writing into the SHARED production queue dir
# (other sessions/agents touch research/queue/ concurrently; see the worktree environment note on this).
LOG="${AWS_BUDGET_LOG:-$ROOT/research/queue/aws_budget.log}"

_fetch_instances() {
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --output json 2>/dev/null
}

cmd="${1:-status}"
case "$cmd" in
  check)
    itype="${2:-}"
    json=$(_fetch_instances)
    if [ -z "$json" ]; then
      echo "⛔ aws_budget: could not reach AWS (describe-instances failed) — refusing to launch until this is resolved" >&2
      exit 1
    fi
    args=(check --cap "$CAP")
    [ -n "$itype" ] && args+=(--type "$itype")
    "$PY" "$ROOT/tools/aws_cost_lib.py" "${args[@]}" <<<"$json"
    exit $?
    ;;
  status)
    json=$(_fetch_instances)
    if [ -z "$json" ]; then
      echo "⛔ aws_budget: could not reach AWS (describe-instances failed)"
      exit 0
    fi
    "$PY" "$ROOT/tools/aws_cost_lib.py" status --cap "$CAP" <<<"$json"
    exit 0
    ;;
  enforce)
    json=$(_fetch_instances)
    if [ -z "$json" ]; then
      echo "[aws_budget] enforce: could not reach AWS (describe-instances failed) — skipping this cycle" | tee -a "$LOG"
      exit 0
    fi
    to_stop=$("$PY" "$ROOT/tools/aws_cost_lib.py" enforce --cap "$CAP" <<<"$json")
    if [ -n "$to_stop" ]; then
      while IFS= read -r iid; do
        [ -z "$iid" ] && continue
        echo "$(date -u '+%FT%TZ') [aws_budget] cap ($CAP USD) reached — stopping $iid" | tee -a "$LOG"
        aws ec2 stop-instances --region "$REGION" --instance-ids "$iid" --output text 2>&1 | tee -a "$LOG"
      done <<<"$to_stop"
    fi
    exit 0
    ;;
  *)
    echo "usage: aws_budget.sh {check [instance-type]|status|enforce}" >&2
    exit 2
    ;;
esac
