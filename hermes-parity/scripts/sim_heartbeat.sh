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
exec /home/dant123/Projects/sim/tools/heartbeat_cmd.sh
