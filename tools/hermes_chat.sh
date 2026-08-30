#!/usr/bin/env bash
# hermes_chat.sh — free the GPU so Qwen (Hermes' brain) loads for INTERACTIVE use in the webui,
# and hand it back to research when you're done. This is the "load Qwen to switch to Hermes" button.
#
# WHY THIS EXISTS: on one 24GB card, Qwen (~20GB) and a GPU research run cannot coexist, so the
# VRAM supervisor keeps Qwen DOWN whenever a local GPU job is running/queued. To TYPE in the webui
# you need Qwen up, which means the local GPU queue must be clear. This defers (never loses) the
# research queue + stops the running job, so the supervisor loads Qwen; `off` restores the queue.
#
#   bash tools/hermes_chat.sh on      # defer research -> Qwen loads -> chat in the webui
#   bash tools/hermes_chat.sh off     # restore research queue -> runs resume (Qwen will cycle again)
#   bash tools/hermes_chat.sh status  # Qwen up? queue depth? deferred sets?
#
# NOTE on the research LOOP: once you type a kickoff and Hermes launches runs via hermes_gpu_run,
# the queue fills again and the supervisor resumes cycling Qwen around those runs automatically —
# firing each between-runs turn into the webui. `on` is just the bootstrap so you can type; you do
# NOT need to babysit Qwen after that.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"
QUEUE="$STATE/gpu.queue"
SERVE="$ROOT/tools/qwen_serve.sh"

qwen_up(){ curl -sf -m4 http://127.0.0.1:8033/health >/dev/null 2>&1; }

case "${1:-status}" in
  on)
    [ -f "$STATE/HERMES_ACTIVE" ] || { echo "[chat] warn: HERMES_ACTIVE not set — the supervisor won't load Qwen. Run: bash tools/hermes_takeover.sh on"; }
    echo "[chat] freeing the GPU for Qwen (deferring research; nothing lost)…"
    bash "$ROOT/tools/gpu_queue.sh" pause --now >/dev/null 2>&1 || true   # stop+requeue any running job
    if [ -s "$QUEUE" ]; then
      ts=$(date +%s); cp "$QUEUE" "$STATE/gpu.queue.deferred.$ts" 2>/dev/null
      echo "[chat]   deferred $(wc -l < "$QUEUE") job(s) -> gpu.queue.deferred.$ts"
      : > "$QUEUE"
    fi
    bash "$ROOT/tools/gpu_queue.sh" resume >/dev/null 2>&1 || true        # so Hermes' future runs still dispatch
    echo "[chat] waiting for the supervisor to load Qwen…"
    for _ in $(seq 1 20); do qwen_up && { echo "[chat] ✅ Qwen UP — open the webui and type in '🤖 Autonomous research loop'."; exit 0; }; sleep 2; done
    # supervisor may be slow / not running — try a direct load as a fallback
    echo "[chat] supervisor didn't load it in time — loading Qwen directly…"
    bash "$SERVE" up >/dev/null 2>&1 || true
    qwen_up && echo "[chat] ✅ Qwen UP — chat in the webui." || { echo "[chat] ⛔ Qwen still down — see $STATE/qwen_server.log"; exit 1; } ;;
  off)
    echo "[chat] handing the GPU back to research…"
    latest=$(ls -t "$STATE"/gpu.queue.deferred.* 2>/dev/null | grep -v '\.restored$' | head -1)
    if [ -n "$latest" ]; then cat "$latest" >> "$QUEUE"; echo "[chat] restored $(wc -l < "$latest") job(s) -> gpu.queue"; mv "$latest" "$latest.restored" 2>/dev/null; fi
    echo "[chat] research will resume; the supervisor will cycle Qwen around the runs." ;;
  status)
    echo "[chat] Qwen: $(qwen_up && echo UP || echo down) | HERMES_ACTIVE: $([ -f "$STATE/HERMES_ACTIVE" ] && echo ON || echo off) | GAME_MODE: $([ -f "$STATE/GAME_MODE" ] && echo ON || echo off)"
    echo "[chat] gpu.queue depth: $(wc -l < "$QUEUE" 2>/dev/null || echo 0) | deferred sets: $(ls "$STATE"/gpu.queue.deferred.* 2>/dev/null | grep -vc '\.restored$')" ;;
  *) echo "usage: bash tools/hermes_chat.sh {on|off|status}"; exit 2 ;;
esac
