#!/usr/bin/env bash
# game.sh — one-command pause/resume so the owner can free the 3090 + local CPU for gaming
# and resume later, WITHOUT Claude. The minipc pool (pool40/41/42) keeps running either way.
#
#   bash tools/game.sh on          # PAUSE: free the 3090 (kill+re-queue the current GPU job, stop starting new ones)
#                                  #        + set GAME_MODE so any local-CPU runner yields. Minipc pool KEEPS RUNNING.
#   bash tools/game.sh on --force  # also KILL any standalone brain-loading GPU python not under the queue
#   bash tools/game.sh off         # RESUME: clear the pause + GAME_MODE; the GPU queue picks the re-queued job back up
#   bash tools/game.sh status      # show pause state, GAME_MODE, GPU VRAM, and any GPU python still resident
#
# Nothing here needs Claude. The re-queued job loses at most its current-run progress ("not much work").
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDIR="${GPU_QUEUE_DIR:-$ROOT/research/queue}"
GAME="$QDIR/GAME_MODE"          # sentinel any LOCAL-CPU runner can check (kept out of the pool's ssh path)
GPUQ="$ROOT/tools/gpu_queue.sh"

gpu_python_procs() {
  # brain-loading GPU python (cupy), excluding the queue dispatcher itself
  pgrep -af "python .*(research\.runners|webapp)" 2>/dev/null | grep -v "gpu_queue.sh" || true
}
vram() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "n/a"; }

case "${1:-status}" in
  on)
    echo "[game] pausing local compute for gaming/break (minipc pool KEEPS running)…"
    bash "$GPUQ" pause --now || echo "[game] warn: gpu_queue pause returned nonzero (dispatcher may be down); continuing"
    : > "$GAME"
    # unload the Hermes local-Qwen model NOW (frees ~10GB). The supervisor sees GAME_MODE and will NOT reload it
    # or nudge Hermes until you run 'off' — nothing auto-spins-back-up during your break, and GAME_MODE persists
    # across reboot (so a reboot mid-break stays paused; a reboot when NOT paused resumes development normally).
    bash "$ROOT/tools/qwen_serve.sh" down >/dev/null 2>&1 || true
    # If Hermes is the driver, also pause the autonomous heartbeat cron -- belt-and-suspenders alongside
    # sim_heartbeat.sh's own GAME_MODE gate (a tick already in flight when GAME_MODE lands would otherwise
    # still run to completion once).
    if [ -f "$ROOT/research/queue/HERMES_ACTIVE" ]; then
      echo "[game] Hermes is the driver -> pausing the autonomous heartbeat cron too…"
      bash "$ROOT/tools/hermes_autonomous.sh" off >/dev/null 2>&1 || true
    fi
    if [ "${2:-}" = "--force" ]; then
      echo "[game] --force: killing any standalone brain-loading GPU python not under the queue…"
      pgrep -af "python .*(research\.runners|webapp)" 2>/dev/null | grep -v "gpu_queue.sh" \
        | awk '{print $1}' | while read -r p; do kill -TERM "$p" 2>/dev/null && echo "  killed $p"; done
      sleep 3
      pgrep -af "python .*(research\.runners|webapp)" 2>/dev/null | grep -v "gpu_queue.sh" \
        | awk '{print $1}' | while read -r p; do kill -KILL "$p" 2>/dev/null; done
    fi
    echo "[game] GAME_MODE set. VRAM now: $(vram)"
    r="$(gpu_python_procs)"; [ -n "$r" ] && { echo "[game] NOTE: GPU python still resident (re-run 'on --force' to clear):"; echo "$r" | cut -c1-90; } || echo "[game] no brain-loading GPU python resident — 3090 is free to game."
    echo "[game] resume later with:  bash tools/game.sh off"
    ;;
  off)
    echo "[game] resuming…"
    rm -f "$GAME"
    bash "$GPUQ" resume || echo "[game] warn: gpu_queue resume returned nonzero (start it with: bash tools/gpu_queue.sh __daemon &)"
    echo "[game] GAME_MODE cleared; GPU queue will pick the re-queued job back up. VRAM: $(vram)"
    if [ -f "$ROOT/research/queue/HERMES_ACTIVE" ]; then
      echo "[game] Hermes is the driver -> the supervisor will reload Qwen within ~8s; resuming the autonomous heartbeat cron…"
      bash "$ROOT/tools/hermes_autonomous.sh" on || echo "[game] warn: autonomous-mode resume had issues (see above) — Hermes still usable interactively."
    else
      echo "[game] Claude is the driver -> Qwen stays down (GPU free for research); tell Claude 'continue' to resume."
    fi
    ;;
  status)
    echo "[game] GAME_MODE: $([ -f "$GAME" ] && echo ON || echo off)"
    echo "[game] GPU_PAUSE: $([ -f "$QDIR/GPU_PAUSE" ] && echo ON || echo off)"
    echo "[game] VRAM: $(vram)"
    echo "[game] pool dispatcher (minipc): $(pgrep -f pool_autodispatch >/dev/null && echo RUNNING || echo down)"
    r="$(gpu_python_procs)"; [ -n "$r" ] && { echo "[game] GPU python resident:"; echo "$r" | cut -c1-90; } || echo "[game] no brain-loading GPU python resident."
    ;;
  *) echo "usage: bash tools/game.sh {on [--force] | off | status}"; exit 2 ;;
esac
