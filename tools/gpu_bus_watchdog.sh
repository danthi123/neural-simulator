#!/bin/bash
# tools/gpu_bus_watchdog.sh -- HARDWARE-ONLY auto-reboot watchdog for the RTX 3090 "falls off the bus" hang
# documented in docs/GPU_CRASH_RECOVERY.md.
#
# THIS SCRIPT HAS EXACTLY TWO SIDE EFFECTS ON A REAL DETECTION: a wall(1) broadcast, and a reboot. IT NEVER
# RESTARTS TRAINING, A SESSION, A QUEUE, OR ANY OTHER PROCESS. See docs/GPU_CRASH_RECOVERY.md's "2026-09-28:
# the July training auto-resume is retired" section: the EARLIER `gpu-train-watchdog.service` did TWO things
# -- restarted `lmtrain-resume.service` whenever it stopped, AND rebooted the machine after repeated
# GPU-off-the-bus checks. The training-restart half was crash-looping and competing with the local LLM for
# the GPU, so the owner had BOTH halves stopped and disabled on 2026-09-28. That doc explicitly asked for a
# from-scratch replacement that "only reboots and never restarts training" -- that is exactly and only what
# this script does. Do not add a training-restart, session-restart, or queue-restart path to this file; that
# exact feature is what killed the previous watchdog.
#
# CRASH SIGNATURE (docs/GPU_CRASH_RECOVERY.md; kernel log via `journalctl -k`, NOT a userspace log):
#   Primary, most specific: `_scrubWaitAndSave` AND `NV_ERR_TIMEOUT` together on the same kernel-log line
#     (`NVRM: GPU0 _scrubWaitAndSave: Timed out ... [NV_ERR_TIMEOUT]`, also written elsewhere as
#     "API_GPU_ATTACHED_SANITY_CHECK failed"), typically followed by a "GPU lost from the bus" line.
#   Also matched, to be robust to different phrasings of the same event across kernel/driver versions:
#     a line containing "GPU has fallen off the bus", or a line containing "GPU lost from the bus" (either
#     alone is sufficient -- these are treated as their own independent signatures).
#
# DESIGN CHOICE -- live journal follower, not a periodic timer: the crash is rare but high-severity, so this
# runs as a long-lived process over `journalctl -k -b 0 -f` (kernel messages for the CURRENT boot, then
# follow new ones as they arrive) rather than waking up on a timer -- it reacts within seconds of the kernel
# line appearing instead of waiting for the next poll tick. The matching systemd unit
# (tools/systemd/gpu-bus-watchdog.service) sets Restart=always with a RestartSec so a journald hiccup (or the
# quick exit taken below when this boot has already rebooted once) never hot-loops.
#
# AT-MOST-ONCE-PER-BOOT: a marker file under /run (tmpfs -- guaranteed empty on every fresh boot, so "has this
# boot already triggered a reboot?" falls out of plain file existence with no boot-id bookkeeping needed).
# Checked ONCE at start, before touching the journal at all: if present, this invocation logs and exits
# immediately, doing nothing else.
#
# Env overrides -- TEST-ONLY, NEVER set in production (same convention as tools/local_llm/llm.sh's
# LOCAL_LLM_SYSTEMCTL and tools/gpu_queue.sh's GPU_QUEUE_NVIDIA_SMI / GPU_QUEUE_LLM_SH: an env var, defaulting
# to the real command, swapped for a fake script under test):
#   GPU_WATCHDOG_JOURNALCTL   override the journalctl command -- a fake script in tests. Default: journalctl
#   GPU_WATCHDOG_REBOOT_CMD   override the reboot command -- a fake script in tests. Default: reboot
#   GPU_WATCHDOG_DELAY_S      override the reboot delay in seconds. Default: 120 (= 2 minutes; the delay is
#                             deliberate -- this must never reboot instantly)
#   GPU_WATCHDOG_MARKER       override the at-most-once-per-boot marker file path. Default: /run/gpu_bus_watchdog.rebooted
# `wall` itself has no dedicated override: tests prepend a directory holding a fake `wall` to PATH (the same
# approach tests/test_llm_gpu_yield.py already uses for `curl`/`systemd-run`).
set -u

JOURNALCTL="${GPU_WATCHDOG_JOURNALCTL:-journalctl}"
REBOOT_CMD="${GPU_WATCHDOG_REBOOT_CMD:-reboot}"
DELAY_S="${GPU_WATCHDOG_DELAY_S:-120}"
MARKER="${GPU_WATCHDOG_MARKER:-/run/gpu_bus_watchdog.rebooted}"

log() { echo "[$(date -u +%FT%TZ)] gpu_bus_watchdog: $*"; }

# is_crash_line LINE -- true iff LINE matches the documented crash signature (see header comment).
is_crash_line() {
    line="$1"
    case "$line" in
        *_scrubWaitAndSave*)
            case "$line" in
                *NV_ERR_TIMEOUT*) return 0 ;;
            esac
            ;;
    esac
    case "$line" in
        *"GPU has fallen off the bus"*) return 0 ;;
        *"GPU lost from the bus"*) return 0 ;;
    esac
    return 1
}

# trigger_reboot -- the ONLY two real-world side effects this script ever has: a wall(1) broadcast, then
# (after the delay) a reboot. Writes the at-most-once-per-boot marker first so a crash mid-delay can never
# double-fire on the next invocation.
trigger_reboot() {
    mkdir -p "$(dirname "$MARKER")" 2>/dev/null
    : > "$MARKER" 2>/dev/null
    log "crash signature detected -- marker written ($MARKER); broadcasting wall(1); rebooting in ${DELAY_S}s"
    wall "GPU bus hang detected (NVRM _scrubWaitAndSave / NV_ERR_TIMEOUT, GPU lost from the bus) -- this \
machine will reboot automatically in ${DELAY_S} seconds to recover the GPU. This watchdog NEVER restarts \
training or any other process; only the OS reboot is automatic." 2>/dev/null || log "wall(1) failed (no tty listeners?) -- continuing"
    sleep "$DELAY_S"
    log "rebooting now via: $REBOOT_CMD"
    "$REBOOT_CMD"
}

main() {
    if [ -e "$MARKER" ]; then
        log "marker $MARKER already present -- this boot already triggered a reboot; doing nothing"
        exit 0
    fi
    log "watching the kernel log for the GPU-off-the-bus signature ($JOURNALCTL -k -b 0 -f)"
    while IFS= read -r line; do
        if is_crash_line "$line"; then
            trigger_reboot
            exit 0
        fi
    done < <("$JOURNALCTL" -k -b 0 -f 2>/dev/null)
    log "journal stream ended without a match"
}

main "$@"
