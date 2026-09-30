#!/usr/bin/env bash
# game.sh — one-command pause/resume so the owner can free the 3090 + local CPU for gaming and resume later,
# WITHOUT touching whichever driver (Claude or the local model, tools/local_llm/llm.sh) is currently working.
# The minipc pool (pool40/41/42) keeps running either way.
#
#   bash tools/game.sh on              # PAUSE: free the 3090 (kill+re-queue the current GPU job, stop starting new
#                                       #   ones), SIGSTOP any standalone local research.runners/webapp python (its
#                                       #   pid is recorded so `off` can SIGCONT it), stop the local-llm server if it
#                                       #   was loaded (its profile + last session are recorded so `off` can bring
#                                       #   both back), and set GAME_MODE. Minipc pool KEEPS RUNNING.
#   bash tools/game.sh on --graceful   # same, but the local-llm stop is delayed ~45s (tools/local_llm/llm.sh
#                                       #   __stop_graceful) so a model calling this ON ITSELF, from inside its own
#                                       #   chat turn (see research/coordination/LOCAL_LLM_RUNBOOK.md), can finish
#                                       #   replying before the server actually goes down.
#   bash tools/game.sh off             # RESUME: SIGCONT the frozen pids, resume the GPU queue, reload the local-llm
#                                       #   server (same profile) if it was on, then resume its last session with a
#                                       #   "gaming's over, re-check your last step" prompt -- interactively if this
#                                       #   is run from a terminal, headless (logged) otherwise.
#   bash tools/game.sh status          # one screen: pause state, model on/off, frozen pids, GPU VRAM
#
# GAME_MODE takes precedence over EVERYTHING else in this repo's compute machinery while it is set: no GPU
# dispatch, no local-llm reload, no session auto-resume (see tools/gpu_queue.sh's LOCAL-LLM COORDINATION block
# and tools/local_llm/llm.sh's `yield`). Nothing here needs Claude or the local model to be running.
#
# 2026-09-30: the sim-agent harness was retired; the owner runs Hermes Agent on homelab-server (~/apps/hermes)
# as a general harness; the sim driver is tools/local_llm/llm.sh claude on the workstation. This script no
# longer pauses/unpauses anything on the server.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDIR="${GPU_QUEUE_DIR:-$ROOT/research/queue}"
GAME="$QDIR/GAME_MODE"                  # sentinel any LOCAL-CPU runner (and tools/gpu_queue.sh / llm.sh) checks
GPUQ="$ROOT/tools/gpu_queue.sh"
LLM_SH="${GAME_LLM_SH:-$ROOT/tools/local_llm/llm.sh}"   # override is TEST-ONLY (a fake script); NEVER set in production
LLM_PORT="${LLM_PORT:-8080}"
LLM_UNIT="local-llm"
LOCAL_LLM_SYSTEMCTL=${LOCAL_LLM_SYSTEMCTL:-systemctl}   # TEST-ONLY override (same name/shape as llm.sh's own
                                                         # SYSTEMCTL override); NEVER set in production
FROZEN_PIDS="$QDIR/.game_frozen_pids"   # one SIGSTOP'd pid per line, written by `on`, consumed (SIGCONT'd) by `off`
LLM_STATE="$QDIR/.game_llm_state"       # GAME_LLM_WAS_ON=/GAME_LLM_PROFILE=/GAME_LLM_SESSION=, written by `on`, consumed by `off`
# Mirrors tools/local_llm/llm.sh's own LLM_SESSION_FILE formula (the id of the last `llm claude` session it
# started) -- read directly rather than adding a round-trip through llm.sh for one path.
LLM_SESSION_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/sim-local-llm/last_session"

gpu_python_procs() {
  # brain-loading GPU python (cupy), excluding the queue dispatcher itself
  pgrep -af "python .*(research\.runners|webapp)" 2>/dev/null | grep -v "gpu_queue.sh" || true
}
vram() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "n/a"; }
# llm_unit_active: is the local-llm systemd UNIT actually active? (2026-09-29 fix -- the previous version,
# llm_is_up(), asked "does anything answer curl http://127.0.0.1:$LLM_PORT/health", which is true for ANY
# process bound to that port, not just this unit. Measured: `game on` recorded GAME_LLM_WAS_ON=1 while the
# local-llm unit was inactive, because Unsloth Studio happened to be answering on 8080 during setup -- so
# `game off` would have loaded the 23 GB model and started a SECOND driver session on top of whatever was
# actually running (CLAUDE.md non-negotiable 12: one driver at a time). Ground truth is the unit's own state,
# exactly like llm.sh's cmd_on/cmd_yield already decide "is it loaded" via `$SYSTEMCTL --user is-active`; the
# health endpoint is only ever the right check for "is it done starting up YET" (llm.sh's own wait_up loop,
# used by `on ... --wait` -- an ALREADY-active-unit's own responsiveness, never a stand-in for whether the unit
# itself is the thing running).
llm_unit_active() { "$LOCAL_LLM_SYSTEMCTL" --user is-active --quiet "$LLM_UNIT" 2>/dev/null; }
llm_field() { sed -n "s/^GAME_LLM_$1=//p" "$LLM_STATE" 2>/dev/null | tail -1; }

case "${1:-status}" in
  on)
    graceful=0
    for a in "${@:2}"; do [ "$a" = "--graceful" ] && graceful=1; done
    echo "[game] pausing local compute for gaming/break (minipc pool KEEPS running)…"
    bash "$GPUQ" pause --now || echo "[game] warn: gpu_queue pause returned nonzero (dispatcher may be down); continuing"
    : > "$GAME"

    # SIGSTOP (not kill) any standalone local research.runners/webapp python that ISN'T the job `pause --now`
    # above already reclaimed (that one was killed + re-queued; anything still here was started directly,
    # outside the GPU queue, e.g. a local CPU run) -- SIGSTOP preserves its progress across the break.
    : > "$FROZEN_PIDS"
    gpu_python_procs | awk '{print $1}' | while read -r p; do
      if kill -STOP "$p" 2>/dev/null; then echo "$p" >> "$FROZEN_PIDS"; echo "  frozen (SIGSTOP) $p"; fi
    done

    # Stop the local-llm server, recording enough to bring it back exactly as it was: whether it was up, which
    # profile, and its last session id (llm.sh's own state file) -- `off` uses all three.
    was_on=0; prof=""; sess=""
    if llm_unit_active; then
      was_on=1
      prof="$(bash "$LLM_SH" __current_profile 2>/dev/null || true)"
      sess="$(cat "$LLM_SESSION_FILE" 2>/dev/null || true)"
      if [ "$graceful" = 1 ]; then
        bash "$LLM_SH" __stop_graceful 45
      else
        bash "$LLM_SH" __stop_graceful 0
      fi
    fi
    {
      echo "GAME_LLM_WAS_ON=$was_on"
      echo "GAME_LLM_PROFILE=$prof"
      echo "GAME_LLM_SESSION=$sess"
    } > "$LLM_STATE"

    echo "[game] GAME_MODE set. VRAM now: $(vram)"
    r="$(gpu_python_procs)"; [ -n "$r" ] && { echo "[game] NOTE: GPU python still resident:"; echo "$r" | cut -c1-90; } || echo "[game] no brain-loading GPU python resident — 3090 is free to game."
    echo "[game] resume later with:  bash tools/game.sh off"
    ;;
  off)
    echo "[game] resuming…"
    if [ -s "$FROZEN_PIDS" ]; then
      while read -r p; do
        [ -n "$p" ] || continue
        kill -CONT "$p" 2>/dev/null && echo "  resumed (SIGCONT) $p" || echo "  $p no longer running"
      done < "$FROZEN_PIDS"
    fi
    rm -f "$FROZEN_PIDS"
    rm -f "$GAME"
    bash "$GPUQ" resume || echo "[game] warn: gpu_queue resume returned nonzero (start it with: bash tools/gpu_queue.sh start)"
    echo "[game] GAME_MODE cleared; GPU queue resumed (any re-queued job picks back up). VRAM: $(vram)"

    was_on=0; prof=""; sess=""
    if [ -f "$LLM_STATE" ]; then was_on="$(llm_field WAS_ON)"; prof="$(llm_field PROFILE)"; sess="$(llm_field SESSION)"; fi
    rm -f "$LLM_STATE"
    if [ "$was_on" = 1 ]; then
      echo "[game] reloading the local model (${prof:-default profile})…"
      bash "$LLM_SH" on ${prof:+"$prof"} --wait
      if [ -n "$sess" ]; then
        prompt="Gaming pause is over. Re-check that your last step completed (it may have been cut off), then continue per research/coordination/LOCAL_LLM_RUNBOOK.md."
        if [ -t 1 ]; then
          echo "[game] resuming session $sess interactively…"
          bash "$LLM_SH" claude --resume "$sess" "$prompt"
        else
          echo "[game] resuming session $sess headless (log: research/queue/llm_session.log)…"
          ( bash "$LLM_SH" claude --resume "$sess" -p "$prompt" >> "$QDIR/llm_session.log" 2>&1 & )
        fi
      fi
    else
      echo "[game] the local model was not loaded before the pause -- nothing to reload."
    fi
    ;;
  status)
    echo "[game] GAME_MODE: $([ -f "$GAME" ] && echo ON || echo off)"
    echo "[game] GPU_PAUSE: $([ -f "$QDIR/GPU_PAUSE" ] && echo ON || echo off)"
    if llm_unit_active; then echo "[game] local model: UP ($(bash "$LLM_SH" __current_profile 2>/dev/null))"; else echo "[game] local model: down"; fi
    echo "[game] VRAM: $(vram)"
    echo "[game] pool dispatcher (minipc): $(pgrep -f pool_autodispatch >/dev/null && echo RUNNING || echo down)"
    if [ -s "$FROZEN_PIDS" ]; then echo "[game] frozen (SIGSTOP'd) pids: $(tr '\n' ' ' < "$FROZEN_PIDS")"; else echo "[game] frozen pids: none"; fi
    r="$(gpu_python_procs)"; [ -n "$r" ] && { echo "[game] GPU python resident:"; echo "$r" | cut -c1-90; } || echo "[game] no brain-loading GPU python resident."
    ;;
  *) echo "usage: bash tools/game.sh {on [--graceful] | off | status}"; exit 2 ;;
esac
