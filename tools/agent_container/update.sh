#!/usr/bin/env bash
# tools/agent_container/update.sh -- weekly OS/Node/Python layer rebuild for sim-agent, owner directive
# 2026-09-29 (keep the harness current). Runs ON THE SERVER (homelab-server), against the deployed copy of
# this directory (e.g. ~/sim-agent/). NOT scheduled by this repo or by any agent -- the owner adds a cron line
# or systemd timer themselves (docs/AGENT_HARNESS_SETUP.md says exactly how); this script only knows how to run
# ONE cycle safely when invoked.
#
# Safety rule: a rebuilt image is only SWAPPED IN (`docker compose up -d`, which recreates the container) when
# doing so cannot lose in-progress work -- the harness is `docker pause`d (ground truth: `docker inspect
# .State.Status`, not a sentinel file -- that IS what tools/agent_harness_ctl.sh's `pause` actually does), OR
# the container is running but no `claude` process is active inside it (an idle shell in the tmux session).
# Otherwise the image is still BUILT (so the next cycle doesn't have to redo that work) but the running
# container is left alone, and the deferral is logged.
#
#   bash tools/agent_container/update.sh            # run from the deployed copy's own directory, or:
#   bash tools/agent_container/update.sh /path/to/deployed/tools/agent_container
set -uo pipefail
HERE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
COMPOSE="$HERE/compose.yaml"
LOG="$HERE/update.log"
CONTAINER="${AGENT_CONTAINER_NAME:-sim-agent}"

log() { printf '%s update.sh: %s\n' "$(date -u '+%F %T')" "$*" | tee -a "$LOG" >&2; }

if [ ! -f "$COMPOSE" ]; then
  echo "update.sh: no compose.yaml found at $HERE -- pass the deployed tools/agent_container directory as \$1" >&2
  exit 2
fi

log "checking $CONTAINER's state before deciding whether to recreate"
status=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "absent")
log "current status: $status"

log "building (docker compose build --pull) -- always safe, never touches the running container"
if ! docker compose -f "$COMPOSE" build --pull >> "$LOG" 2>&1; then
  log "BUILD FAILED -- see $LOG; leaving the running container (if any) untouched"
  exit 1
fi

recreate=0
case "$status" in
  absent) log "no existing container -- starting fresh"; recreate=1 ;;
  paused) log "harness is paused -- safe to recreate"; recreate=1 ;;
  running)
    busy="$(docker exec "$CONTAINER" pgrep -f '/claude$' 2>/dev/null || true)"
    if [ -n "$busy" ]; then
      log "DEFERRED: a claude process is active inside $CONTAINER (pid(s): $busy) and the harness is not paused -- image built, NOT recreating this cycle. The owner's schedule will retry next tick."
      recreate=0
    else
      log "running but idle (no active claude process) -- safe to recreate"
      recreate=1
    fi
    ;;
  *) log "unrecognized status '$status' -- treating as safe to recreate"; recreate=1 ;;
esac

if [ "$recreate" = 1 ]; then
  if docker compose -f "$COMPOSE" up -d >> "$LOG" 2>&1; then
    log "recreate complete"
  else
    log "recreate FAILED -- see $LOG (the old container may still be running if compose left it alone)"
    exit 1
  fi
else
  log "cycle complete (build only, no recreate)"
fi
