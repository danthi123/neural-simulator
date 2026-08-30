#!/usr/bin/env bash
# Apply the neural-simulator heartbeat as a Hermes cron job.
#
# NOT RUN AUTOMATICALLY by anything in this worktree -- review, then run by hand (this touches the
# owner's global ~/.hermes/, which the task that produced this file was told not to do itself).
#
# What it sets up: a job that runs tools/heartbeat_cmd.sh (via the copied wrapper script) every 15
# minutes -- the same cadence and the same checks (GPU/proc state, unpushed-commit detection,
# tools/workflow_check.sh, tools/parallel_audit.py) as the Claude Code session's heartbeat Monitor
# (CLAUDE.md: "a within-session anti-stall + RUN-STATE heartbeat Monitor ... STATE-CHECKING (emits
# GPU / running-procs / recent-output every ~15 min)"). `--script` mode's DEFAULT behavior (no
# --no-agent) injects the script's stdout into Hermes's own prompt each run, so Hermes actually
# reasons over the audit and can act on it -- matching CLAUDE.md's "⛔ UNDER-PARALLELIZED ... is a
# STALL — launch the listed independent work ... BEFORE holding."
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
REPO=/home/dant123/Projects/sim
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "1/2: copying the heartbeat wrapper into $HERMES_HOME/scripts/ (cron --script is sandboxed there)"
mkdir -p "$HERMES_HOME/scripts"
cp "$HERE/scripts/sim_heartbeat.sh" "$HERMES_HOME/scripts/sim_heartbeat.sh"
chmod +x "$HERMES_HOME/scripts/sim_heartbeat.sh"

echo "2/2: creating the cron job (15-minute cadence, agent mode so Hermes can ACT on the audit)"
hermes cron create "15m" \
  "This is the neural-simulator project's autonomous heartbeat (Hermes-side equivalent of the Claude Code session heartbeat Monitor). The injected script output above reports GPU/process state, unpushed-commit count (ls-remote verified), tools/workflow_check.sh violations, and the tools/parallel_audit.py parallelization verdict. React to it: if it prints any unpushed commits, run tools/push_both.sh now. If workflow_check reports a violation, run the exact fix command it printed. If parallel_audit.py prints UNDER-PARALLELIZED with idle GPU/pool capacity and ready board tasks, launch the listed independent work now rather than just noting it. Otherwise, if the board is saturated and nothing here is actionable, pick and start the next concrete item from GAP_CLOSURE_MISSION.md CURRENT STATE or 'tools/vikunja.sh list-tasks 2' yourself -- never end this turn on a status report alone." \
  --name sim-heartbeat \
  --script sim_heartbeat.sh \
  --workdir "$REPO"

echo "done. Verify with:  hermes cron list   /   hermes cron runs sim-heartbeat"
echo "NOTE: this job's hooks (pre_llm_call live-state injection etc.) only fire if hooks_auto_accept:"
echo "true is set (or --accept-hooks / HERMES_ACCEPT_HOOKS=1) -- see config.hooks.snippet.yaml."
