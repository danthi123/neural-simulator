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
# CONTAINERIZED HARNESS (2026-09-29, docs/plans/2026-09-29-local-model-transition-PLAN.md section 9): `on` ALSO
# pauses the sim-agent container on homelab-server (best-effort, via tools/agent_harness_ctl.sh over ssh --
# skipped gracefully if the server/ctl is unreachable, and skipped outright when the container's active
# profile is KIND=cloud, since a cloud backend does not depend on this machine's GPU/model server at all); `off`
# unpauses it and sends a "gaming's over" line into its tmux session, AFTER the local model server is back up
# and answering its health endpoint.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDIR="${GPU_QUEUE_DIR:-$ROOT/research/queue}"
GAME="$QDIR/GAME_MODE"                  # sentinel any LOCAL-CPU runner (and tools/gpu_queue.sh / llm.sh) checks
GPUQ="$ROOT/tools/gpu_queue.sh"
LLM_SH="${GAME_LLM_SH:-$ROOT/tools/local_llm/llm.sh}"   # override is TEST-ONLY (a fake script); NEVER set in production
LLM_PORT="${LLM_PORT:-8080}"
FROZEN_PIDS="$QDIR/.game_frozen_pids"   # one SIGSTOP'd pid per line, written by `on`, consumed (SIGCONT'd) by `off`
LLM_STATE="$QDIR/.game_llm_state"       # GAME_LLM_WAS_ON=/GAME_LLM_PROFILE=/GAME_LLM_SESSION=, written by `on`, consumed by `off`
# Mirrors tools/local_llm/llm.sh's own LLM_SESSION_FILE formula (the id of the last `llm claude` session it
# started) -- read directly rather than adding a round-trip through llm.sh for one path.
LLM_SESSION_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/sim-local-llm/last_session"
CTL_HOST="${AGENT_HARNESS_CTL_HOST:-homelab-agent-ctl}"   # ssh alias for tools/agent_harness_ctl.sh on the server
GAME_SSH="${GAME_SSH:-ssh}"                                # TEST-ONLY override; NEVER set in production
SSH_OPTS=(-o ConnectTimeout=8 -o BatchMode=yes)

# harness_kind: the sim-agent container's active profile KIND (local|cloud), read via agent_harness_ctl.sh
# `state`. Best-effort -- empty (not "cloud") whenever the server/ctl is unreachable or nothing is running, so
# an unreachable harness is treated as "needs a pause" (the safe default: skip pausing only when we POSITIVELY
# know it is a cloud backend, never by default).
harness_kind() {
  "$GAME_SSH" "${SSH_OPTS[@]}" "$CTL_HOST" state 2>/dev/null | sed -n 's/^profile_kind: //p'
}
harness_pause_needed() { [ "$(harness_kind)" != "cloud" ]; }

gpu_python_procs() {
  # brain-loading GPU python (cupy), excluding the queue dispatcher itself
  pgrep -af "python .*(research\.runners|webapp)" 2>/dev/null | grep -v "gpu_queue.sh" || true
}
vram() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "n/a"; }
# Mirrors llm.sh's own is_up(): a plain health check, so game.sh never needs to parse `llm status`'s text output.
llm_is_up() { curl -sf "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1; }
llm_field() { sed -n "s/^GAME_LLM_$1=//p" "$LLM_STATE" 2>/dev/null | tail -1; }

case "${1:-status}" in
  on)
    graceful=0
    for a in "${@:2}"; do [ "$a" = "--graceful" ] && graceful=1; done
    echo "[game] pausing local compute for gaming/break (minipc pool KEEPS running)…"
    if harness_pause_needed; then
      if "$GAME_SSH" "${SSH_OPTS[@]}" "$CTL_HOST" pause >/dev/null 2>&1; then
        echo "[game] sim-agent harness: pause requested"
      else
        echo "[game] sim-agent harness: could not reach $CTL_HOST to pause (unreachable, or not set up yet) -- continuing"
      fi
    else
      echo "[game] sim-agent harness: active profile is KIND=cloud -- no pause needed"
    fi
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
    if llm_is_up; then
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

    # sim-agent harness: unpause + a "gaming's over" line, AFTER the model server (if any) is confirmed
    # healthy above. Always attempted (best-effort) -- agent_harness_ctl.sh's own `unpause` is a no-op if the
    # container was never paused (e.g. it was on a cloud profile, or the server/ctl was unreachable at `on`
    # time), so there is nothing to gate this on here.
    if "$GAME_SSH" "${SSH_OPTS[@]}" "$CTL_HOST" unpause >/dev/null 2>&1; then
      echo "[game] sim-agent harness: unpaused"
      "$GAME_SSH" "${SSH_OPTS[@]}" "$CTL_HOST" say \
        "Gaming pause is over. Re-check your last step, then continue per the runbook." >/dev/null 2>&1 \
        && echo "[game] sim-agent harness: sent a resume prompt into its session" \
        || echo "[game] sim-agent harness: unpaused but could not deliver the resume prompt (container down, or no 'agent' tmux session)"
    else
      echo "[game] sim-agent harness: could not reach $CTL_HOST to unpause (unreachable, or not set up yet)"
    fi
    ;;
  status)
    echo "[game] GAME_MODE: $([ -f "$GAME" ] && echo ON || echo off)"
    echo "[game] GPU_PAUSE: $([ -f "$QDIR/GPU_PAUSE" ] && echo ON || echo off)"
    if llm_is_up; then echo "[game] local model: UP ($(bash "$LLM_SH" __current_profile 2>/dev/null))"; else echo "[game] local model: down"; fi
    echo "[game] VRAM: $(vram)"
    echo "[game] pool dispatcher (minipc): $(pgrep -f pool_autodispatch >/dev/null && echo RUNNING || echo down)"
    if [ -s "$FROZEN_PIDS" ]; then echo "[game] frozen (SIGSTOP'd) pids: $(tr '\n' ' ' < "$FROZEN_PIDS")"; else echo "[game] frozen pids: none"; fi
    r="$(gpu_python_procs)"; [ -n "$r" ] && { echo "[game] GPU python resident:"; echo "$r" | cut -c1-90; } || echo "[game] no brain-loading GPU python resident."
    ;;
  *) echo "usage: bash tools/game.sh {on [--graceful] | off | status}"; exit 2 ;;
esac
