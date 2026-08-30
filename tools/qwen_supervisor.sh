#!/usr/bin/env bash
# qwen_supervisor.sh — VRAM-aware load/unload of the local Qwen (Hermes' brain) around LOCAL gpu_queue jobs.
#
# THE INVARIANT: a LOCAL GPU job and the Qwen server never co-reside — the supervisor unloads Qwen before a job
# can run and reloads it (then nudges Hermes to check results) once the local queue is idle. POOL (mini-PC) runs
# are remote and NEVER affect Qwen. The whole thing is INERT unless HERMES_ACTIVE is set — so while Claude drives,
# Qwen stays down and the GPU is untouched. GAME_MODE (owner gaming/testing) overrides everything: Qwen stays down.
#
#   bash tools/qwen_supervisor.sh __daemon    # the poll loop (run by the systemd user service / hermes_takeover)
#   bash tools/qwen_supervisor.sh status      # one-shot: what would it do right now
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"
ACTIVE="$STATE/HERMES_ACTIVE"       # master switch: Hermes is the driver (Claude usage out)
GAME="$STATE/GAME_MODE"            # owner wants the GPU (gaming/test) — absolute priority
QUEUE="$STATE/gpu.queue"           # local GPU jobs waiting
RUNNING="$STATE/gpu.running"       # local GPU job in flight
JOBRAN="$STATE/.qwen_jobran"       # marker: Qwen was unloaded for a job -> on reload, notify Hermes
TRIGLOCK="$STATE/.qwen_hermes_turn.lock"
LOG="$STATE/qwen_supervisor.log"
POLL="${QWEN_SUPERVISOR_POLL:-8}"
SERVE="$ROOT/tools/qwen_serve.sh"
HERMES="${HERMES_BIN:-/home/dant123/.local/bin/hermes}"

log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
local_gpu_busy(){ [ -s "$QUEUE" ] || [ -f "$RUNNING" ]; }
qwen_running(){ bash "$SERVE" status 2>/dev/null | grep -q '^\[qwen\] UP'; }
qwen_ready(){ bash "$SERVE" status 2>/dev/null | grep -q 'ready=yes'; }

CONTINUE_PROMPT='You are Hermes, now driving the neural-simulator project autonomously because Claude usage is exhausted. A local GPU job just completed and your model was reloaded. FIRST read research/coordination/live_state.md and the CURRENT STATE at the top of GAP_CLOSURE_MISSION.md, then harvest the just-completed run(s), then continue the ordered next action. Obey CLAUDE.md constraints (brain-based-only, one-brain, no-defer, 6-seed, gates are authoritative). Commit BOTH remotes via tools/push_both.sh, never --no-verify. Run GPU work only via tools/hermes_gpu_run.sh (it hands the GPU off from your model safely).'

fire_hermes_continue(){
  # nudge Hermes to take a fresh turn — detached, single-flight (never overlap turns)
  if [ -f "$TRIGLOCK" ] && kill -0 "$(cat "$TRIGLOCK" 2>/dev/null)" 2>/dev/null; then return; fi
  [ -x "$HERMES" ] || { log "hermes bin not executable at $HERMES — skip continue-trigger"; return; }
  ( cd "$ROOT" && setsid "$HERMES" -z "$CONTINUE_PROMPT" </dev/null >>"$STATE/qwen_hermes_turns.log" 2>&1 & echo $! > "$TRIGLOCK" )
  log "fired hermes continue-turn (pid $(cat "$TRIGLOCK" 2>/dev/null))"
}

daemon(){
  log "supervisor up (poll=${POLL}s)"
  while :; do
    if [ ! -f "$ACTIVE" ]; then                       # Claude drives (or nobody) -> Qwen must be down, GPU untouched
      qwen_running && { log "HERMES_ACTIVE off -> unloading Qwen"; bash "$SERVE" down >>"$LOG" 2>&1; }
      sleep "$POLL"; continue
    fi
    if [ -f "$GAME" ]; then                            # owner wants the GPU -> Qwen down, hands off
      qwen_running && { log "GAME_MODE -> unloading Qwen for the owner"; bash "$SERVE" down >>"$LOG" 2>&1; }
      sleep "$POLL"; continue
    fi
    if local_gpu_busy; then                            # a LOCAL job needs the GPU -> Qwen out of the way
      if qwen_running; then log "local GPU job present -> unloading Qwen"; bash "$SERVE" down >>"$LOG" 2>&1; touch "$JOBRAN"; fi
      sleep "$POLL"; continue
    fi
    # idle: no local job, Hermes driving, owner not gaming -> Qwen should be UP for Hermes
    if ! qwen_running; then log "local GPU idle -> loading Qwen for Hermes"; bash "$SERVE" up >>"$LOG" 2>&1; fi
    if [ -f "$JOBRAN" ] && qwen_ready; then rm -f "$JOBRAN"; log "job(s) done + Qwen ready -> nudging Hermes"; fire_hermes_continue; fi
    sleep "$POLL"
  done
}

case "${1:-status}" in
  __daemon) daemon ;;
  status)
    echo "HERMES_ACTIVE: $([ -f "$ACTIVE" ] && echo ON || echo off) | GAME_MODE: $([ -f "$GAME" ] && echo ON || echo off)"
    echo "local_gpu_busy: $(local_gpu_busy && echo yes || echo no) (queue=$(wc -l <"$QUEUE" 2>/dev/null || echo 0), running=$([ -f "$RUNNING" ] && echo yes || echo no))"
    echo "qwen: $(qwen_running && echo UP || echo down)"
    if [ ! -f "$ACTIVE" ]; then echo "verdict: inert (Claude drives) -> keep Qwen down"
    elif [ -f "$GAME" ]; then echo "verdict: GAME_MODE -> keep Qwen down"
    elif local_gpu_busy; then echo "verdict: local job -> Qwen down"
    else echo "verdict: idle -> Qwen up (+ nudge Hermes if a job just finished)"; fi ;;
  *) echo "usage: bash tools/qwen_supervisor.sh {__daemon|status}"; exit 2 ;;
esac
