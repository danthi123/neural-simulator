#!/usr/bin/env bash
# openhands_takeover.sh — the owner's one-command handoff to OpenHands (local Qwen dev-agent), for when
# Claude usage is exhausted. Mirrors tools/hermes_takeover.sh's shape; SUPERSEDES it as the recommended
# local fallback once OpenHands is live-verified (tools/hermes_takeover.sh stays intact meanwhile — see
# docs/OPENHANDS_TAKEOVER.md and tools/hermes/loop.py's SUPERSEDED header).
#
#   bash tools/openhands_takeover.sh on      # HAND OVER TO OPENHANDS: mark it the driver, start the
#                                             #   shared VRAM supervisor, launch openhands_loop.py.
#   bash tools/openhands_takeover.sh off     # HAND BACK TO CLAUDE: stop the loop, unload Qwen, GPU free.
#   bash tools/openhands_takeover.sh status  # who's driving + Qwen/supervisor/OpenHands/GPU state
#
# VRAM lifecycle is 100% owned by tools/qwen_supervisor.sh (generalized 2026-09-08 to watch
# OPENHANDS_ACTIVE alongside HERMES_ACTIVE — see its header). Neither this script nor
# tools/openhands_proto/openhands_loop.py call tools/qwen_serve.sh up/down by default: two independent
# deciders unloading/reloading the same model is the exact double-load race qwen_serve.sh's own guard
# comments warn about. This script only ever calls `qwen_serve.sh status` (read-only) plus starts the
# supervisor daemon, which owns up/down.
#
# ⛔ SINGLE ACTIVE DRIVER — do not also run `bash tools/hermes_takeover.sh on` while this is on (or vice
# versa). Hand the current driver back first. Nothing here touches Hermes state (HERMES_ACTIVE,
# tools/hermes/, the hermes-loop/hermes-webui units) — the two takeovers are independent switches that
# happen to share the one Qwen server + the one supervisor daemon.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"
ACTIVE="$STATE/OPENHANDS_ACTIVE"
SERVE="$ROOT/tools/qwen_serve.sh"
SUP="$ROOT/tools/qwen_supervisor.sh"
PROTO="$ROOT/tools/openhands_proto"
VENV_PY="$PROTO/.venv/bin/python"
LOOP_PIDF="$STATE/openhands_loop.pid"
LOOP_LOG="$PROTO/state/openhands_loop.log"

supervisor_up(){ pgrep -f "qwen_supervisor.sh __daemon" >/dev/null 2>&1; }
start_supervisor(){ supervisor_up || { setsid bash "$SUP" __daemon </dev/null >>"$STATE/qwen_supervisor.log" 2>&1 & echo "[takeover] supervisor started (pid $!)"; }; }
loop_running(){ [ -f "$LOOP_PIDF" ] && kill -0 "$(cat "$LOOP_PIDF" 2>/dev/null)" 2>/dev/null; }

case "${1:-status}" in
  on)
    echo "[takeover] handing the project over to OPENHANDS (local Qwen)…"
    if [ ! -x "$VENV_PY" ]; then
      echo "[takeover] ⛔ tools/openhands_proto/.venv not built (it's a gitignored, machine-local build"
      echo "    artifact — never committed). Build it once with:"
      echo "      cd $PROTO && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -U openhands-sdk openhands-tools"
      echo "    then re-run: bash tools/openhands_takeover.sh on"
      exit 1
    fi
    if [ -f "$STATE/HERMES_ACTIVE" ]; then
      echo "[takeover] ⛔ HERMES_ACTIVE is set — Hermes is (or was) the active driver. Hand it back first:"
      echo "      bash tools/hermes_takeover.sh off"
      echo "    then re-run: bash tools/openhands_takeover.sh on"
      exit 1
    fi
    : > "$ACTIVE"
    start_supervisor
    echo "[takeover] OPENHANDS_ACTIVE set + shared VRAM supervisor running (it owns Qwen up/down)."
    mkdir -p "$PROTO/state"
    if loop_running; then
      echo "[takeover] openhands_loop.py already running (pid $(cat "$LOOP_PIDF"))."
    else
      ( cd "$PROTO" && setsid "$VENV_PY" openhands_loop.py </dev/null >>"$LOOP_LOG" 2>&1 & echo $! > "$LOOP_PIDF" )
      sleep 1
      if loop_running; then
        echo "[takeover] openhands_loop.py started (pid $(cat "$LOOP_PIDF"), log: $LOOP_LOG)"
      else
        echo "[takeover] ⛔ openhands_loop.py failed to start — see $LOOP_LOG"; tail -20 "$LOOP_LOG" 2>/dev/null; exit 1
      fi
    fi
    echo "[takeover] Qwen loads once the local GPU queue is idle (note: run 'on' when the GPU is actually"
    echo "[takeover] free for the model — this does NOT preempt a running research job)."
    echo "[takeover] Watch progress:  tail -f $LOOP_LOG   |   bash tools/qwen_serve.sh status   |   bash tools/openhands_takeover.sh status"
    echo "[takeover] Give it an ad-hoc task any time (same persisted conversation either way):"
    echo "[takeover]   $VENV_PY $PROTO/run_turn.py --prompt \"<task>\""
    echo "[takeover] Hand back later with:  bash tools/openhands_takeover.sh off"
    echo "[takeover] Full walkthrough: docs/OPENHANDS_TAKEOVER.md" ;;
  off)
    echo "[takeover] handing the project back to CLAUDE…"
    if loop_running; then
      p=$(cat "$LOOP_PIDF"); kill -TERM "$p" 2>/dev/null
      for _ in 1 2 3 4 5; do kill -0 "$p" 2>/dev/null || break; sleep 1; done
      kill -KILL "$p" 2>/dev/null
    fi
    rm -f "$LOOP_PIDF"
    rm -f "$ACTIVE"
    bash "$SERVE" down
    echo "[takeover] ✅ openhands_loop.py stopped, Qwen unloaded, GPU free for research runs."
    echo "[takeover]    Supervisor left running (inert without a driver sentinel; harmless)." ;;
  status)
    echo "[takeover] driver: $([ -f "$ACTIVE" ] && echo "OpenHands (local Qwen)" || echo "not OpenHands")"
    echo "[takeover] venv: $([ -x "$VENV_PY" ] && echo "built ($VENV_PY)" || echo "NOT built — see 'on' for the build command")"
    echo "[takeover] supervisor: $(supervisor_up && echo running || echo down)"
    echo "[takeover] openhands_loop.py: $(loop_running && echo "running (pid $(cat "$LOOP_PIDF"))" || echo down)"
    bash "$SERVE" status
    echo "[takeover] HERMES_ACTIVE: $([ -f "$STATE/HERMES_ACTIVE" ] && echo ON || echo off) | GAME_MODE: $([ -f "$STATE/GAME_MODE" ] && echo ON || echo off) | GPU_PAUSE: $([ -f "$STATE/GPU_PAUSE" ] && echo ON || echo off)" ;;
  *) echo "usage: bash tools/openhands_takeover.sh {on|off|status}"; exit 2 ;;
esac
