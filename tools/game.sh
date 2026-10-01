#!/usr/bin/env bash
# game.sh — one-command pause/resume so the owner can free the 3090 + local CPU for gaming and resume later,
# WITHOUT touching whichever driver (Claude or the local model, tools/local_llm/llm.sh) is currently working.
# The minipc pool (pool40/41/42) keeps running either way.
#
#   bash tools/game.sh on              # PAUSE: free the 3090 (kill+re-queue the current GPU job, stop starting new
#                                       #   ones), SIGSTOP any standalone local research.runners/webapp python (its
#                                       #   pid is recorded so `off` can SIGCONT it), stop the local-llm server if it
#                                       #   was running (its profile + the pause start are recorded for `off`), and
#                                       #   set GAME_MODE. Minipc pool KEEPS RUNNING.
#   bash tools/game.sh on --graceful   # same, but the local-llm stop is delayed ~45s (tools/local_llm/llm.sh
#                                       #   __stop_graceful) so a model calling this ON ITSELF, from inside its own
#                                       #   chat turn (see research/coordination/LOCAL_LLM_RUNBOOK.md), can finish
#                                       #   replying before the server actually goes down.
#   bash tools/game.sh off             # RESUME WHAT WAS RUNNING (owner 2026-09-30): SIGCONT the frozen pids and
#                                       #   resume the GPU queue (the killed job re-runs first). If Hermes had
#                                       #   yielded the GPU (`llm yield`), the window gets back the time the pause
#                                       #   took and the model server stays off -- gpu_queue.sh restarts it when the
#                                       #   jobs finish. Otherwise, if the server was running before, it restarts
#                                       #   with the model UNLOADED (`llm on --lazy`): nothing takes VRAM until
#                                       #   Hermes asks, and Hermes's sim-driver wake gate starts its next run within
#                                       #   ~5 min of the server answering. Pending GPU jobs still go first.
#   bash tools/game.sh status          # one screen: pause state, model on/off, frozen pids, GPU VRAM
#
# GAME_MODE takes precedence over EVERYTHING else in this repo's compute machinery while it is set: no GPU
# dispatch, no local-llm reload, no session auto-resume (see tools/gpu_queue.sh's LOCAL-LLM COORDINATION block
# and tools/local_llm/llm.sh's `yield`). Nothing here needs Claude or the local model to be running.
#
# 2026-09-30: the sim-agent harness was retired; the owner runs Hermes Agent on homelab-server (~/apps/hermes),
# which drives this project from there. This script touches nothing on the server: Hermes notices the model
# server going away and coming back by itself (its sim-driver wake gate), and no local session is resumed here
# any more (that would be a second driver beside Hermes; CLAUDE.md non-negotiable 12).
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
LLM_STATE="$QDIR/.game_llm_state"       # GAME_LLM_WAS_ON=/GAME_LLM_PROFILE=/GAME_LLM_STARTED=, written by `on`, consumed by `off`
YIELD="$QDIR/.llm_yield"                # tools/local_llm/llm.sh yield's marker: a GPU window Hermes opened (gpu_queue.sh)

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

    # Stop the local-llm server, recording enough to bring it back as it was: whether it was up, which profile,
    # and when the pause began (`off` pushes a yielded GPU window's deadline out by the pause's length).
    was_on=0; prof=""
    if llm_unit_active; then
      was_on=1
      prof="$(bash "$LLM_SH" __current_profile 2>/dev/null || true)"
      if [ "$graceful" = 1 ]; then
        bash "$LLM_SH" __stop_graceful 45
      else
        bash "$LLM_SH" __stop_graceful 0
      fi
    fi
    {
      echo "GAME_LLM_WAS_ON=$was_on"
      echo "GAME_LLM_PROFILE=$prof"
      echo "GAME_LLM_STARTED=$(date +%s)"
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

    was_on=0; prof=""; started=""
    if [ -f "$LLM_STATE" ]; then was_on="$(llm_field WAS_ON)"; prof="$(llm_field PROFILE)"; started="$(llm_field STARTED)"; fi
    rm -f "$LLM_STATE"
    # A GPU window Hermes opened (`llm yield`) was paused by `on`: give it back the time the pause took, so the
    # jobs it yielded for still fit before its deadline (digits only before any arithmetic: both values are read
    # from files).
    if [ -f "$YIELD" ] && [[ "$started" =~ ^[0-9]{1,12}$ ]]; then
      deadline="$(sed -n 's/^LLM_YIELD_DEADLINE=//p' "$YIELD" | tail -1)"
      if [[ "$deadline" =~ ^[0-9]{1,12}$ ]]; then
        paused=$(( $(date +%s) - started ))
        [ "$paused" -gt 0 ] && sed -i "s/^LLM_YIELD_DEADLINE=.*/LLM_YIELD_DEADLINE=$(( deadline + paused ))/" "$YIELD" \
          && echo "[game] Hermes's GPU window gets back the $(( paused / 60 )) min the pause took."
      fi
    fi
    bash "$GPUQ" resume || echo "[game] warn: gpu_queue resume returned nonzero (start it with: bash tools/gpu_queue.sh start)"
    echo "[game] GAME_MODE cleared; GPU queue resumed (any re-queued job picks back up). VRAM: $(vram)"

    if [ -f "$YIELD" ]; then
      echo "[game] Hermes had handed the GPU to its queued jobs: they continue now, and the model server comes back by itself when they finish."
    elif [ "$was_on" = 1 ]; then
      # --lazy: the server is back but the model stays out of VRAM until Hermes's next request. If GPU jobs are
      # waiting, llm.sh refuses and gpu_queue.sh starts the server once they drain (the jobs go first).
      echo "[game] restarting the model server (${prof:-default profile}); the model loads when Hermes next asks…"
      bash "$LLM_SH" on ${prof:+"$prof"} --lazy || true
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
