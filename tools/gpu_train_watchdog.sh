#!/bin/bash
# Autonomous GPU-crash + training-resume watchdog for a multi-day unattended lmtrain run.
# Runs as a systemd service (root, Restart=always, enabled -> survives its own crashes AND reboots).
#
# What it guarantees ("picks up automatically if it stops for whatever reason"):
#   (1) GPU OFF THE BUS (nvidia-smi -L fails 3x in a row ~= wedged core, reboot-only per gpu_recover.sh)
#         -> `reboot`; lmtrain-resume.service auto-resumes training from checkpoint on boot.
#   (2) Training service FAILED/INACTIVE while the GPU is HEALTHY (e.g. it exhausted StartLimitBurst,
#         or crashed for a non-GPU reason) and NO PAUSE sentinel is present
#         -> `reset-failed` + `start` it again (no reboot needed).
#   (3) Normal crash (non-zero exit, GPU fine) is already handled by the service's own Restart=on-failure.
#   (4) Power/kernel reboot is already handled by the service being `enabled` (boot resume).
#
# Safety (this thing can REBOOT the box, so it is deliberately conservative):
#   - reboots ONLY on 3 consecutive nvidia-smi -L failures (~90s of the GPU genuinely gone; -L works fine
#     during normal training, so this does not fire on a busy-but-healthy GPU),
#   - >= REBOOT_COOLDOWN between reboots,
#   - CIRCUIT BREAKER: if >= MAX_IN_WINDOW reboots happen within WINDOW seconds, the GPU is genuinely failing
#     (not a transient) -> STOP rebooting, drop a STRANDED marker, and just log until a human intervenes,
#   - INITIAL_GRACE at startup so a fresh boot (GPU/driver still coming up) never triggers a reboot,
#   - respects the PAUSE sentinel: never revives training the owner intentionally paused.
# Owner checks in occasionally; the log + STRANDED marker make a genuinely-dead GPU obvious.
set -u

SVC=lmtrain-resume.service
RUNROOT=/home/dant123/Projects/sim/bridges/lmtrain/run3
PAUSE="$RUNROOT/PAUSE"
LOG="$RUNROOT/watchdog.log"
STATE=/var/tmp/gpu_train_watchdog          # /var/tmp persists across reboots
mkdir -p "$STATE"
REBOOTS_F="$STATE/reboot_epochs"           # one epoch per reboot (rolling window)
STRANDED_F="$STATE/STRANDED"

INITIAL_GRACE=180        # let the GPU/driver settle after (re)boot before judging it
POLL=30                  # seconds between checks
FAILS_TO_REBOOT=3        # consecutive nvidia-smi -L failures -> GPU off the bus
REBOOT_COOLDOWN=1200     # >= 20 min between reboots
WINDOW=7200              # circuit-breaker window (2 h)
MAX_IN_WINDOW=4          # >= this many reboots within WINDOW -> strand (flapping/dead GPU)
RESTART_COOLDOWN=300     # >= 5 min between service revive attempts

log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG" 2>/dev/null; }

log "watchdog START (pid $$); initial grace ${INITIAL_GRACE}s"
sleep "$INITIAL_GRACE"

gpu_fail=0
last_restart=0
while true; do
  if nvidia-smi -L >/dev/null 2>&1; then
    gpu_fail=0
    # (2) GPU healthy: revive the training service if it died and wasn't intentionally paused.
    if [ ! -f "$STRANDED_F" ] && [ ! -f "$PAUSE" ]; then
      act=$(systemctl is-active "$SVC" 2>/dev/null || echo unknown)
      if [ "$act" != "active" ]; then
        now=$(date +%s)
        if [ $(( now - last_restart )) -ge "$RESTART_COOLDOWN" ]; then
          log "service $SVC is '$act' + GPU healthy + no PAUSE -> reset-failed + start"
          systemctl reset-failed "$SVC" 2>/dev/null
          systemctl start "$SVC" 2>/dev/null
          last_restart="$now"
        fi
      fi
    fi
  else
    gpu_fail=$(( gpu_fail + 1 ))
    log "nvidia-smi -L FAILED (${gpu_fail}/${FAILS_TO_REBOOT})"
    if [ "$gpu_fail" -ge "$FAILS_TO_REBOOT" ]; then
      now=$(date +%s)
      recent=$(awk -v c=$(( now - WINDOW )) '($1+0)>=c' "$REBOOTS_F" 2>/dev/null | wc -l)
      last_rb=$(tail -1 "$REBOOTS_F" 2>/dev/null || echo 0)
      if [ "$recent" -ge "$MAX_IN_WINDOW" ]; then
        log "CIRCUIT BREAKER: ${recent} reboots within ${WINDOW}s -> GPU genuinely failing. NOT rebooting; STRANDED. Manual GPU fix needed."
        touch "$STRANDED_F"
        gpu_fail=0
      elif [ $(( now - last_rb )) -ge "$REBOOT_COOLDOWN" ]; then
        log "GPU OFF THE BUS (${FAILS_TO_REBOOT}x) -> REBOOT now. lmtrain-resume.service auto-resumes on boot."
        echo "$now" >> "$REBOOTS_F"
        sync
        systemctl reboot
        sleep 120
      else
        log "GPU down but within reboot cooldown ($(( now - last_rb ))s < ${REBOOT_COOLDOWN}s) -> waiting"
      fi
    fi
  fi
  sleep "$POLL"
done
