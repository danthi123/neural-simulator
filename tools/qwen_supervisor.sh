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
# Detect Qwen by the HTTP ENDPOINT, NOT `qwen_serve status` — the latter shells out to nvidia-smi
# (via vram()), which is slow/flaky under GPU contention and made this check intermittently return
# the wrong answer, so the daemon skipped the unload and deadlocked (three times, 2026-08-30). A
# bare curl with a hard timeout can neither hang nor mis-read.
QPORT="${QWEN_PORT:-8033}"
qwen_up(){ curl -sf -m3 "http://127.0.0.1:$QPORT/health" >/dev/null 2>&1; }
qwen_running(){ qwen_up; }
qwen_ready(){ qwen_up; }
# A Hermes turn touches .qwen_llm_active (pre_llm_call hook) on every LLM call. While it's fresh, a
# turn is mid-flight -> do NOT unload Qwen for a queued run (that killed the turn overnight). It goes
# stale a few minutes after the turn's last call, then the pending run is allowed to take the GPU.
GRACE_LLM="${QWEN_TURN_GRACE:-240}"
LLM_ACTIVE="$STATE/.qwen_llm_active"
llm_active(){ local ts age; ts=$(cat "$LLM_ACTIVE" 2>/dev/null) || return 1; [ -n "$ts" ] || return 1; age=$(( $(date +%s) - ts )); [ "$age" -lt "$GRACE_LLM" ] 2>/dev/null; }
# Gateway/webui turns don't fire the pre_llm_call hook, so llm_active can't see them. But a Hermes
# turn launches its run NEAR THE END of the turn, so "gpu.queue unchanged for QUEUE_GRACE seconds"
# reliably means the launching turn has finished -> safe to take the GPU. This is the primary guard
# against unloading Qwen mid-turn (which killed the turn overnight).
QUEUE_GRACE="${QWEN_QUEUE_GRACE:-90}"
queue_settled(){ [ -s "$QUEUE" ] || return 1; local age; age=$(( $(date +%s) - $(stat -c %Y "$QUEUE" 2>/dev/null || echo 0) )); [ "$age" -ge "$QUEUE_GRACE" ] 2>/dev/null; }
# COGNITIVE CONTINUATION: a Hermes turn that does no GPU run (harvest/analysis/edit) leaves nothing
# to trigger the next turn, so the loop dead-ends (the ~30min stall). So when Qwen is idle with NO
# GPU job, we fire the next turn on a cadence. It's SAFE to over-fire: webui_continue does an atomic
# /api/chat/start which 409-skips if a turn is already streaming, so a fire during an active turn is
# a harmless no-op. FIRE_COOLDOWN paces the attempts.
FIRE_COOLDOWN="${QWEN_FIRE_COOLDOWN:-90}"
LASTFIRE="$STATE/.qwen_last_fire"

CONTINUE_PROMPT='You are Hermes, now driving the neural-simulator project autonomously because Claude usage is exhausted. A local GPU job just completed and your model was reloaded. FIRST read research/coordination/live_state.md and the CURRENT STATE at the top of GAP_CLOSURE_MISSION.md, then harvest the just-completed run(s), then continue the ordered next action. Obey CLAUDE.md constraints (brain-based-only, one-brain, no-defer, 6-seed, gates are authoritative). Commit BOTH remotes via tools/push_both.sh, never --no-verify. Run GPU work only via tools/hermes_gpu_run.sh (it hands the GPU off from your model safely).'

fire_hermes_continue(){
  # nudge Hermes to take a fresh turn once a GPU run finished + Qwen reloaded.
  # PREFER the VISIBLE path: drive the single persistent webui conversation the owner watches
  # AND types into (tools/hermes/webui_continue.py -> /api/chat/start on a reused session). This
  # is what makes "monitor + engage the autonomous loop in the web UI" real. It falls back to a
  # headless `hermes -z` turn only if the webui is down / auth fails, so autonomy never stalls.
  # (JOBRAN-clear upstream already makes this fire once per completed run — the single-flight guard.)
  if [ "${HERMES_CONTINUE_VIA_WEBUI:-1}" = "1" ] && command -v python3 >/dev/null 2>&1; then
    # DETACHED (setsid ... &): /api/chat/start is async so this returns fast, but detaching
    # guarantees a slow/unresponsive webui can NEVER block this VRAM-management loop (the earlier
    # freeze lesson). Fire-and-forget; webui_continue.py logs its own success/failure.
    ( cd "$ROOT" && setsid python3 "$ROOT/tools/hermes/webui_continue.py" "$CONTINUE_PROMPT" </dev/null >>"$STATE/qwen_hermes_turns.log" 2>&1 & )
    log "fired hermes continue-turn via WEBUI (visible session, detached)"
    return
  fi
  log "headless hermes -z continuation (webui-drive disabled)"
  if [ -f "$TRIGLOCK" ] && kill -0 "$(cat "$TRIGLOCK" 2>/dev/null)" 2>/dev/null; then return; fi
  [ -x "$HERMES" ] || { log "hermes bin not executable at $HERMES — skip continue-trigger"; return; }
  ( cd "$ROOT" && setsid "$HERMES" -z "$CONTINUE_PROMPT" </dev/null >>"$STATE/qwen_hermes_turns.log" 2>&1 & echo $! > "$TRIGLOCK" )
  log "fired hermes continue-turn headless (pid $(cat "$TRIGLOCK" 2>/dev/null))"
}

down_qwen(){ timeout 40 bash "$SERVE" down >>"$LOG" 2>&1; }   # timeout-bounded: a stuck call can't wedge the loop
up_qwen(){   timeout 600 bash "$SERVE" up   >>"$LOG" 2>&1; }
daemon(){
  log "supervisor up (poll=${POLL}s, endpoint-detect)"
  local last_state="" hb=0
  while :; do
    local state up; up=$(qwen_up && echo up || echo down)
    if [ ! -f "$ACTIVE" ] || [ -f "$GAME" ]; then     # Claude drives OR owner gaming -> Qwen down, GPU untouched
      state="hold"; [ "$up" = up ] && { log "hold ($([ -f "$GAME" ] && echo GAME_MODE || echo HERMES_ACTIVE-off)) -> unloading Qwen"; down_qwen; }
    elif local_gpu_busy; then                          # a LOCAL job needs the full GPU -> Qwen out of the way
      if [ "$up" = down ]; then state="job"            # Qwen already unloaded, the run has the GPU
      elif [ -f "$RUNNING" ]; then state="job"; log "run active but Qwen still up -> unloading"; down_qwen; touch "$JOBRAN"
      elif llm_active; then state="job-wait"           # a turn is streaming (heartbeat, headless path) -> don't cut it
      elif ! queue_settled; then state="job-wait"      # a run was just queued -> the launching turn is finishing; wait
      else state="job"; log "GPU job queued + turn finished (queue settled ${QUEUE_GRACE}s) -> unloading Qwen for the run"; down_qwen; touch "$JOBRAN"; fi
    else                                               # idle: Hermes driving, no GPU job -> Qwen UP + keep the loop turning
      state="idle"
      [ "$up" = down ] && { log "idle -> loading Qwen for Hermes"; up_qwen; up=$(qwen_up && echo up || echo down); }
      if [ "$up" = up ]; then
        now=$(date +%s); lastf=$(cat "$LASTFIRE" 2>/dev/null || echo 0)
        if [ -f "$JOBRAN" ]; then                       # a run just finished -> harvest turn now
          rm -f "$JOBRAN"; log "run(s) done + Qwen ready -> firing Hermes turn"; fire_hermes_continue; echo "$now" > "$LASTFIRE"
        elif [ $(( now - ${lastf:-0} )) -ge "$FIRE_COOLDOWN" ] 2>/dev/null; then   # cognitive continuation (409-skips if a turn is live)
          log "idle -> firing Hermes turn (cognitive continuation)"; fire_hermes_continue; echo "$now" > "$LASTFIRE"
        fi
      fi
    fi
    # Heartbeat: log on every state change, plus a keepalive every ~5min, so a wedge is VISIBLE in the
    # log (silence during a state that demands action = something is wrong) rather than a silent deadlock.
    hb=$((hb+1))
    if [ "$state" != "$last_state" ] || [ "$hb" -ge 38 ]; then
      log "hb state=$state qwen=$up queue=$(wc -l <"$QUEUE" 2>/dev/null || echo 0) running=$([ -f "$RUNNING" ] && echo y || echo n)"
      last_state="$state"; hb=0
    fi
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
