#!/usr/bin/env bash
# hermes_takeover.sh — THE owner's one-command handoff between Claude (this repo's usual driver) and
# Hermes (local Qwen, for when Claude usage is exhausted). Run ONE driver at a time.
#
#   bash tools/hermes_takeover.sh on      # HAND OVER TO HERMES: mark Hermes the driver, start the VRAM supervisor,
#                                         #   bring Qwen up. Then run `hermes` to work. (Claude should be idle.)
#   bash tools/hermes_takeover.sh off     # HAND BACK TO CLAUDE: unload Qwen, free the GPU for research runs,
#                                         #   supervisor goes inert. (Run this before you resume with Claude.)
#   bash tools/hermes_takeover.sh status  # who's driving + Qwen/supervisor/GPU state
#
# The supervisor (tools/qwen_supervisor.sh) does the moment-to-moment VRAM dance while ON: it unloads Qwen when a
# LOCAL GPU job runs and reloads it (nudging Hermes to check results) when the local queue is idle. POOL runs never
# trigger it. GAME_MODE (tools/game.sh on) still overrides everything (keeps Qwen down for gaming).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"
ACTIVE="$STATE/HERMES_ACTIVE"
SERVE="$ROOT/tools/qwen_serve.sh"
SUP="$ROOT/tools/qwen_supervisor.sh"

supervisor_up(){ pgrep -f "qwen_supervisor.sh __daemon" >/dev/null 2>&1; }
start_supervisor(){ supervisor_up || { setsid bash "$SUP" __daemon </dev/null >>"$STATE/qwen_supervisor.log" 2>&1 & echo "[takeover] supervisor started (pid $!)"; }; }

case "${1:-status}" in
  on)
    echo "[takeover] handing the project over to HERMES (local Qwen)…"
    # sanity: the DFlash2-capable llama.cpp must exist before we promise a takeover
    if ! bash "$SERVE" status >/dev/null 2>&1; then echo "[takeover] warn: qwen_serve.sh status failed"; fi
    : > "$ACTIVE"
    start_supervisor
    echo "[takeover] HERMES_ACTIVE set + supervisor running. Bringing Qwen up (first run downloads ~9GB)…"
    bash "$SERVE" up || { echo "[takeover] Qwen failed to come up — see $STATE/qwen_server.log. HERMES_ACTIVE still set; fix + re-run 'on'."; exit 1; }
    echo "[takeover] ✅ Hermes is the driver. Wire Hermes once with:  bash tools/hermes_local_setup.sh   (or 'hermes setup')"
    echo "[takeover]    then run:  hermes    — it will use the local Qwen at http://127.0.0.1:${QWEN_PORT:-8033}/v1"
    echo "[takeover] Claude should stay idle while Hermes drives. Hand back later with: bash tools/hermes_takeover.sh off" ;;
  off)
    echo "[takeover] handing the project back to CLAUDE…"
    rm -f "$ACTIVE"
    bash "$SERVE" down
    echo "[takeover] ✅ Qwen unloaded, GPU free for research runs, supervisor is now inert (leave it running; harmless)."
    echo "[takeover]    Resume Claude-side compute if it was paused:  bash tools/gpu_queue.sh resume" ;;
  status)
    echo "[takeover] driver: $([ -f "$ACTIVE" ] && echo HERMES\ \(local\ Qwen\) || echo Claude)"
    echo "[takeover] supervisor: $(supervisor_up && echo running || echo down)"
    bash "$SERVE" status
    echo "[takeover] GAME_MODE: $([ -f "$STATE/GAME_MODE" ] && echo ON || echo off) | GPU_PAUSE: $([ -f "$STATE/GPU_PAUSE" ] && echo ON || echo off)" ;;
  *) echo "usage: bash tools/hermes_takeover.sh {on|off|status}"; exit 2 ;;
esac
