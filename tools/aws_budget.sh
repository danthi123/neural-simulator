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
# POOL_SSH_CFG / aws_stop_safety_lib.sh (2026-09-25 review, fix round 3): `enforce` now syncs a node's results
# before stopping it at the cap -- see the `enforce)` case below for why this NEVER blocks the stop itself
# (unlike tools/aws_idle_stop.sh's own opportunistic use of the same library, where an unverified sync DOES
# block). POOL_SSH_CFG override exists for tests, same reasoning as LOG above.
POOL_SSH_CFG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
# AWS_STATE_DIR override exists for tests -- `enforce`'s per-instance node/key lookup globs this directory's
# `.aws_*` state files (the same convention tools/aws_idle_stop.sh reads), which must never resolve to the
# SHARED production research/queue/ (other sessions/agents write state files there concurrently).
AWS_STATE_DIR="${AWS_NODE_STATE_DIR:-$ROOT/research/queue}"
source "$ROOT/tools/aws_stop_safety_lib.sh"

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
      # fd 3 FOR THE LOOP (2026-09-25 review, fix round 3 -- the SAME class of bug tools/aws_idle_stop.sh's own
      # main loop was fixed for): every ssh call the sync below makes must not drain $to_stop off fd 0 for the
      # REST of this loop. Reading the id list from fd 3 makes that structurally impossible regardless of any
      # one remote call's own stdin habits.
      while IFS= read -r iid <&3; do
        [ -z "$iid" ] && continue
        # SYNC BEFORE STOP (2026-09-25 review, still-open item: "aws_budget.sh:72-76 still stops a node at the
        # cap without syncing first"). Best-effort and BOUNDED, and it NEVER blocks the stop itself -- this is
        # the one deliberate difference from tools/aws_idle_stop.sh's own (blocking) use of the same
        # sync_node_before_stop: `enforce` exists to cap SPEND, and an instance that keeps running while a sync
        # is stuck/failing is exactly the outcome the cap is meant to prevent, so it must never be defeated by
        # one. A slower/failed sync here costs at most the same unsynced-results risk aws_idle_stop.sh already
        # accepts when it CANNOT verify a key/ip at all (also logged, never fatal) -- traded deliberately for
        # "the cap always holds."
        node=""; key=""; ip=""
        for sf in "$AWS_STATE_DIR"/.aws_*; do
          [ -f "$sf" ] || continue
          if grep -q "^instance=$iid\$" "$sf" 2>/dev/null; then
            key=$(awk -F= '/^key=/{print $2}' "$sf" 2>/dev/null)
            node=$(basename "$sf" 2>/dev/null); node="${node#.aws_}"
            break
          fi
        done
        if [ -n "$key" ] && [ -f "$key" ]; then
          ip=$(aws ec2 describe-instances --instance-ids "$iid" --region "$REGION" \
                --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
        fi
        if [ -n "$ip" ] && [ "$ip" != "None" ]; then
          was_reg=$(pause_dispatch_for_node "$node")   # see tools/aws_idle_stop.sh's own use -- same reasoning
          if timeout "${AWS_BUDGET_SYNC_TIMEOUT_S:-120}" env AWS_SYNC_LOG="$LOG" POOL_SSH_CONFIG="$POOL_SSH_CFG" \
               bash "$ROOT/tools/aws_stop_safety_lib.sh" --sync "$node" "$ip" "$key" </dev/null >>"$LOG" 2>&1; then
            echo "$(date -u '+%FT%TZ') [aws_budget] $iid ($node): synced before stopping at cap" | tee -a "$LOG"
          else
            echo "$(date -u '+%FT%TZ') [aws_budget] $iid ($node): sync-before-stop failed/timed out -- stopping ANYWAY (hard cap, see comment above)" | tee -a "$LOG"
          fi
          resume_dispatch_for_node "$node" "$was_reg"
        else
          echo "$(date -u '+%FT%TZ') [aws_budget] $iid: no verified ssh key/ip on hand -- cannot sync, stopping ANYWAY (hard cap)" | tee -a "$LOG"
        fi
        echo "$(date -u '+%FT%TZ') [aws_budget] cap ($CAP USD) reached — stopping $iid" | tee -a "$LOG"
        aws ec2 stop-instances --region "$REGION" --instance-ids "$iid" --output text 2>&1 | tee -a "$LOG"
      done 3<<<"$to_stop"
    fi
    exit 0
    ;;
  *)
    echo "usage: aws_budget.sh {check [instance-type]|status|enforce}" >&2
    exit 2
    ;;
esac
