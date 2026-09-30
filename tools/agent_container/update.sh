#!/usr/bin/env bash
# tools/agent_container/update.sh -- weekly rebuild + smoke test + safe swap for the sim-agent stack, owner
# directive 2026-09-29 (keep the harness current). Runs ON THE SERVER (srv-host-0) as the owner's account, from the
# owner's source checkout on the dataset -- never from ~/apps and never from the agent's own writable repo:
#
#   bash /home/dant123/data/sim-agent/src/tools/agent_container/update.sh
#
# NOT scheduled by this repo or by any agent; the owner installs the timer (docs/AGENT_HARNESS_SETUP.md).
#
# One cycle:
#   1. Preconditions. ~/apps's root compose.yaml includes the sim-agent stack; the build source (this checkout's
#      tools/agent_container + tools/agent_harness_ctl.sh) is a CLEAN git tree -- a modified, untracked or
#      ignored file there would be built into the image (or run by the ctl forced command), so refuse; the
#      ~/apps copy of the fragment matches this checkout's server-stack copy (warn on drift). Disk usage of the
#      ZFS dataset (df reports it against its refquota) and of Docker's data-root is logged, loudly above 85%.
#   2. Tag the current image sim-agent:rollback, then `docker compose build --pull sim-agent` from ~/apps.
#   3. If the image changed: smoke-test the NEW image (`docker compose run --rm --no-deps sim-agent
#      smoke_test.sh`, which includes the in-container network-isolation probes). On failure, point
#      sim-agent:local back at the rollback image and stop -- the running container is never touched.
#   4. Swap the running container onto the new image (`docker compose up -d --no-deps --no-build sim-agent`) only
#      when that can neither lose work nor start it unfiltered:
#        running, no active claude process, sim-agent-fw.service active  -> recreate
#        running, a claude process active                               -> defer (the next cycle retries)
#        paused (a gaming yield froze a session, maybe mid-task)        -> defer: recreating would kill it
#        exited / created / absent                                      -> never started here: the owner
#                                                                          stopped it, or never started it
# Claude Code itself is not updated here: it lives in the home bind mount, and claude-safe-update.sh updates,
# smoke-tests and rolls it back on every `agent start`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SIM_AGENT_SRC_DIR:-$(cd "$HERE/../.." && pwd)}"          # the checkout this script lives in
DATA_DIR="${SIM_AGENT_DATA_DIR:-$(dirname "$SRC_DIR")}"             # ${DATA_ROOT}/sim-agent (the dataset)
APPS_DIR="${SIM_AGENT_APPS_DIR:-$HOME/apps}"
LOG="${SIM_AGENT_UPDATE_LOG:-$DATA_DIR/update.log}"                 # never inside src/: that would dirty it
SERVICE=sim-agent
CONTAINER="${AGENT_CONTAINER_NAME:-sim-agent}"
IMAGE="${SIM_AGENT_IMAGE:-sim-agent:local}"
ROLLBACK_IMAGE="${IMAGE%%:*}:rollback"
FW_UNIT=sim-agent-fw.service
DF="${AGENT_UPDATE_DF:-df}"                     # TEST-ONLY override; NEVER set in production
SYSTEMCTL="${AGENT_UPDATE_SYSTEMCTL:-systemctl}" # TEST-ONLY override; NEVER set in production
DISK_WARN_PCT="${AGENT_UPDATE_DISK_WARN_PCT:-85}"
FRAGMENT_SRC="$SRC_DIR/tools/agent_container/server-stack/sim-agent/compose.yaml"
FRAGMENT_LIVE="$APPS_DIR/sim-agent/compose.yaml"
# Every `docker exec` into the container pins PATH and names its binary absolutely (2026-09-30 security review,
# HIGH): an exec inherits the container's environment, and the home bind mount is model-writable.
EXEC_ENV_PATH="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

log() { printf '%s update.sh: %s\n' "$(date -u '+%F %T')" "$*" | tee -a "$LOG" >&2; }
compose() { (cd "$APPS_DIR" && docker compose "$@"); }
image_id() { docker image inspect -f '{{.Id}}' "$1" 2>/dev/null || true; }

# --- 1. preconditions --------------------------------------------------------------------------------------------
if [ ! -f "$APPS_DIR/compose.yaml" ] || ! grep -qE '^[[:space:]]*-[[:space:]]*sim-agent/compose\.yaml[[:space:]]*(#.*)?$' "$APPS_DIR/compose.yaml"; then
  log "REFUSED: $APPS_DIR/compose.yaml does not include sim-agent/compose.yaml -- install the stack first"
  exit 2
fi
if [ ! -f "$FRAGMENT_LIVE" ]; then
  log "REFUSED: no stack fragment at $FRAGMENT_LIVE"
  exit 2
fi
dirty="$(git -C "$SRC_DIR" status --porcelain --ignored -- tools/agent_container tools/agent_harness_ctl.sh 2>&1)"
if [ $? -ne 0 ] || [ -n "$dirty" ]; then
  log "REFUSED: the build source $SRC_DIR is not a clean git checkout of tools/agent_container +" \
      "tools/agent_harness_ctl.sh (a modified, untracked or ignored file there would be built into the image):"
  printf '%s\n' "$dirty" | head -20 | tee -a "$LOG" >&2
  exit 1
fi
log "build source: $SRC_DIR at $(git -C "$SRC_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo '?')"
if [ -f "$FRAGMENT_SRC" ] && ! cmp -s "$FRAGMENT_SRC" "$FRAGMENT_LIVE"; then
  log "WARNING: $FRAGMENT_LIVE differs from the reviewed copy $FRAGMENT_SRC -- deliberate? If not:" \
      "cp $FRAGMENT_SRC $FRAGMENT_LIVE (the ~/apps copy is what runs)"
fi

# DISK USAGE (2026-09-29 security review, M3): a visibility floor, never a gate. The dataset's refquota is the real
# ceiling (a runaway job hits ENOSPC inside the container, never the shared pool); this makes the approach visible.
disk_roots="${AGENT_UPDATE_DISK_ROOTS:-$DATA_DIR $(docker info -f '{{.DockerRootDir}}' 2>/dev/null)}"
for root in $disk_roots; do
  pct=$("$DF" --output=pcent "$root" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -z "$pct" ]; then
    log "disk usage check: could not read $DF --output=pcent $root -- skipped (non-fatal)"
    continue
  fi
  log "disk usage at $root: ${pct}%"
  if [ "$pct" -ge "$DISK_WARN_PCT" ]; then
    log "WARNING: $root is ${pct}% full (threshold ${DISK_WARN_PCT}%) -- this is a SHARED server; clean up" \
        "the agent's worktrees/results or raise the dataset refquota before writes start failing."
  fi
done

# --- 2. build ----------------------------------------------------------------------------------------------------
old_id="$(image_id "$IMAGE")"
if [ -n "$old_id" ]; then
  docker image tag "$IMAGE" "$ROLLBACK_IMAGE" >> "$LOG" 2>&1 && log "tagged the current image as $ROLLBACK_IMAGE"
fi
log "building (docker compose build --pull $SERVICE) -- never touches the running container"
if ! compose build --pull "$SERVICE" >> "$LOG" 2>&1; then
  log "BUILD FAILED -- see $LOG; $IMAGE and the running container are unchanged"
  exit 1
fi
new_id="$(image_id "$IMAGE")"

# --- 3. smoke-test a changed image before anything can run it ----------------------------------------------------
if [ -n "$new_id" ] && [ "$new_id" = "$old_id" ]; then
  log "image unchanged by the rebuild"
else
  log "smoke-testing the new image (docker compose run --rm --no-deps $SERVICE smoke_test.sh)"
  if ! compose run --rm --no-deps -T "$SERVICE" smoke_test.sh >> "$LOG" 2>&1; then
    if [ -n "$old_id" ]; then
      docker image tag "$ROLLBACK_IMAGE" "$IMAGE" >> "$LOG" 2>&1
      log "SMOKE TEST FAILED -- rolled $IMAGE back to the previous image ($ROLLBACK_IMAGE); the running" \
          "container was not touched. See $LOG."
    else
      log "SMOKE TEST FAILED on the first build -- there is no previous image to roll back to; do not start" \
          "the stack until this passes. See $LOG."
    fi
    exit 1
  fi
  log "smoke test passed"
fi

# --- 4. swap the running container only when it is safe ----------------------------------------------------------
status=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "absent")
[ -n "$status" ] || status=absent
log "current status of $CONTAINER: $status"
case "$status" in
  running)
    current="$(docker inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || true)"
    if [ -n "$current" ] && [ "$current" = "$new_id" ]; then
      log "already running the current image -- cycle complete"
      exit 0
    fi
    busy="$(docker exec -e "$EXEC_ENV_PATH" "$CONTAINER" /usr/bin/pgrep -f '/claude$' 2>/dev/null || true)"
    if [ -n "$busy" ]; then
      log "DEFERRED: a claude process is active inside $CONTAINER (pid(s): $busy) -- image built and smoke-tested," \
          "NOT recreating this cycle; the next one retries."
      exit 0
    fi
    if ! "$SYSTEMCTL" is-active --quiet "$FW_UNIT"; then
      log "REFUSED to recreate: $FW_UNIT is not active (sudo systemctl status $FW_UNIT). The new container would" \
          "refuse to start anyway (firewall_guard.sh); the old one keeps running."
      exit 1
    fi
    log "running but idle (no active claude process), firewall active -- recreating"
    if compose up -d --no-deps --no-build "$SERVICE" >> "$LOG" 2>&1; then
      log "recreate complete (status now: $(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo '?'))"
    else
      log "recreate FAILED -- see $LOG. Roll back by hand: docker image tag $ROLLBACK_IMAGE $IMAGE &&" \
          "(cd $APPS_DIR && docker compose up -d --no-deps --no-build $SERVICE)"
      exit 1
    fi
    ;;
  paused)
    log "DEFERRED: $CONTAINER is paused (a gaming yield may have frozen a session mid-task) -- recreating would" \
        "kill it. Image built and smoke-tested; the next cycle retries."
    ;;
  *)
    log "$CONTAINER is not running ($status) -- not starting it (the owner stopped it, or never started it). The" \
        "new image is ready for: cd $APPS_DIR && docker compose up -d $SERVICE"
    ;;
esac
log "cycle complete"
