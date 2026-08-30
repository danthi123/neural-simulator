#!/usr/bin/env bash
# Hermes cron `--script` body for the neural-simulator heartbeat.
#
# WHAT THIS IS FOR: this file must be COPIED (not symlinked -- `hermes cron create --script`
# resolves the path strictly under ~/.hermes/scripts/ via `.resolve().relative_to(scripts_dir)`,
# cron/scheduler.py:4342-4351, so an absolute path or a symlink escaping that directory is
# rejected) to:
#
#   cp hermes-parity/scripts/sim_heartbeat.sh ~/.hermes/scripts/sim_heartbeat.sh
#   chmod +x ~/.hermes/scripts/sim_heartbeat.sh
#
# It deliberately contains NO duplicated logic -- it just execs the repo's own canonical
# heartbeat body (tools/heartbeat_cmd.sh), which already does everything the Claude Code
# session-heartbeat Monitor checks each cycle: GPU/proc state, unpushed-commit detection
# (ls-remote verified, branch-aware), tools/workflow_check.sh (parallelism + lanes + sources +
# cluster rules), and tools/parallel_audit.py (under-parallelization: idle GPU/pool cores next to
# ready board tasks). Routing through ONE script means a fix to tools/heartbeat_cmd.sh is live on
# the next cron tick with no re-copy -- exactly the same "call the live file, never a snapshot"
# reasoning tools/heartbeat_cmd.sh's own header gives for folding parallel_audit.py into itself.
#
# BELT-AND-SUSPENDERS AUTONOMOUS GATE (added for the autonomous-mode build). This script is
# reached by a live `hermes cron` tick, which only fires the moment the gateway process is up --
# i.e. the moment "hermes gateway install"/"start" has run. That is a coarse switch: it says
# nothing about whether HERMES_ACTIVE (Hermes is actually the intended driver, not Claude) or
# GAME_MODE (the owner explicitly paused for a break) hold at tick time, and both can change
# independently of the gateway process's own up/down state. So a stray tick landing while Claude
# is the driver, or while the owner is mid-break, must be a safe no-op, not an action. Checked
# fresh every tick (files, not cached state) -- cheap, no GPU, no network.
REPO=/home/dant123/Projects/sim
if [ ! -f "$REPO/research/queue/HERMES_ACTIVE" ] || [ -f "$REPO/research/queue/GAME_MODE" ]; then
  echo "autonomous paused/inactive -- take no action, end the turn"
  exit 0
fi
exec "$REPO/tools/heartbeat_cmd.sh"
