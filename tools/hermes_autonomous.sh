#!/usr/bin/env bash
# hermes_autonomous.sh {on|off|status} — the AUTONOMOUS-MODE switch: makes the 15-min
# sim-heartbeat cron job (hermes-parity/apply_cron.sh) actually TICK, hands-off, by ensuring
# `hermes gateway` -- the process that hosts Hermes's built-in cron ticker (confirmed directly
# against the installed Hermes's own test suite: "the builtin cron ticker only runs inside the
# gateway process", tests/cron/test_87033_cronjob_gateway_liveness.py) -- is installed and running,
# then resumes/pauses the sim-heartbeat job itself.
#
#   bash tools/hermes_autonomous.sh on       # ensure the gateway is up, resume sim-heartbeat
#   bash tools/hermes_autonomous.sh off      # pause sim-heartbeat (gateway stays up; harmless idle)
#   bash tools/hermes_autonomous.sh status   # gateway + cron job state + HERMES_ACTIVE + GAME_MODE
#
# Idempotent and safe from any starting state (fresh boot, mid-session, already-on): every step
# checks live state before acting and prints a clear per-step verdict rather than assuming.
#
# `hermes gateway install` (no --system) is a USER-level systemd unit, no sudo required — this
# script may run it unattended (it is not a privileged/system-wide change, and the owner runs this
# script, not this agent). `hermes cron resume|pause` are pure scheduling switches (no model
# inference, no GPU) — cheap and safe to call at any time, from any driver.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"
HERMES="${HERMES_BIN:-/home/dant123/.local/bin/hermes}"
JOB="sim-heartbeat"
LOG="$STATE/hermes_autonomous.log"
mkdir -p "$STATE"

have_hermes() {
  [ -x "$HERMES" ] || command -v hermes >/dev/null 2>&1
}

# `timeout` execs its argument as an external command -- it cannot invoke a bash function, so
# resolve to whichever form of "$HERMES" is actually runnable (an absolute path, or a PATH lookup)
# once, and call THAT under timeout everywhere below.
hbin() {
  if [ -x "$HERMES" ]; then echo "$HERMES"; else command -v hermes 2>/dev/null; fi
}

# --- gateway: install (first time) or start (already installed), then confirm ------------------
ensure_gateway() {
  if ! have_hermes; then
    echo "  ⛔ hermes binary not found at $HERMES / on PATH -- cannot manage the gateway."
    echo "     fix: confirm the install, or set HERMES_BIN=/path/to/hermes"
    return 1
  fi
  local h out
  h="$(hbin)"
  out=$(timeout 20 "$h" gateway status 2>&1)
  if echo "$out" | grep -qi "gateway service is running"; then
    echo "  ✓ hermes gateway already running"
    return 0
  fi
  if echo "$out" | grep -qi "not installed"; then
    echo "  gateway not installed -- installing as a user service (no sudo; auto-starts on login/boot)…"
    if timeout 60 "$h" gateway install --start-now --start-on-login >>"$LOG" 2>&1; then
      echo "  ✓ hermes gateway installed + started (log: $LOG)"
    else
      echo "  ⛔ 'hermes gateway install --start-now --start-on-login' FAILED — see $LOG"
      echo "     run by hand: hermes gateway install --start-now --start-on-login"
      return 1
    fi
  else
    echo "  gateway installed but not running -- starting…"
    if timeout 30 "$h" gateway start >>"$LOG" 2>&1; then
      echo "  ✓ hermes gateway started (log: $LOG)"
    else
      echo "  ⛔ 'hermes gateway start' FAILED — see $LOG"
      echo "     run by hand: hermes gateway start"
      return 1
    fi
  fi
  out=$(timeout 20 "$h" gateway status 2>&1)
  if echo "$out" | grep -qi "gateway service is running"; then
    return 0
  fi
  echo "  ⛔ gateway still not confirmed running after install/start — run by hand: hermes gateway status"
  return 1
}

resume_heartbeat() {
  if ! have_hermes; then echo "  ⛔ hermes binary not found -- cannot resume '$JOB'"; return 1; fi
  local out
  out=$(timeout 20 "$(hbin)" cron resume "$JOB" 2>&1)
  if echo "$out" | grep -qi "not found"; then
    # auto-create it once (so `start` is truly one-command, no separate apply_cron step needed)
    echo "  ℹ cron job '$JOB' missing -- creating it via hermes-parity/apply_cron.sh…"
    if bash "$ROOT/hermes-parity/apply_cron.sh" >/dev/null 2>&1; then
      out=$(timeout 20 "$(hbin)" cron resume "$JOB" 2>&1)
      echo "$out" | grep -qi "not found" && { echo "  ⛔ still can't create/resume '$JOB' -- run: bash hermes-parity/apply_cron.sh manually"; return 1; }
    else
      echo "  ⛔ apply_cron.sh failed -- create the job manually: bash hermes-parity/apply_cron.sh"; return 1
    fi
  fi
  echo "  ✓ cron '$JOB' resumed (created if it was missing)"
  return 0
}

pause_heartbeat() {
  if ! have_hermes; then echo "  ⛔ hermes binary not found -- cannot pause '$JOB'"; return 1; fi
  local out
  out=$(timeout 20 "$(hbin)" cron pause "$JOB" 2>&1)
  if echo "$out" | grep -qi "not found"; then
    echo "  ℹ cron job '$JOB' does not exist yet (nothing to pause) -- create it with: bash hermes-parity/apply_cron.sh"
    return 0
  fi
  echo "  ✓ cron '$JOB' paused (gateway left running; harmless idle)"
  return 0
}

status_cmd() {
  echo "HERMES_ACTIVE: $([ -f "$STATE/HERMES_ACTIVE" ] && echo ON || echo off)"
  echo "GAME_MODE:     $([ -f "$STATE/GAME_MODE" ] && echo ON || echo off)"
  echo
  if ! have_hermes; then
    echo "hermes binary not found at $HERMES / on PATH -- cannot query gateway/cron."
    return 1
  fi
  local h
  h="$(hbin)"
  echo "-- gateway (must be running for the cron ticker to fire) --"
  timeout 15 "$h" gateway status 2>&1 | sed 's/^/  /'
  echo
  echo "-- cron job '$JOB' --"
  local list_out
  list_out=$(timeout 15 "$h" cron list --all 2>&1)
  if echo "$list_out" | grep -qi "$JOB"; then
    echo "$list_out" | grep -A4 -i "$JOB" | sed 's/^/  /'
  else
    echo "  not found -- create it: bash hermes-parity/apply_cron.sh"
  fi
}

case "${1:-status}" in
  on)
    echo "[autonomous] enabling autonomous mode (gateway + '$JOB' cron)…"
    rc=0
    ensure_gateway || rc=1
    resume_heartbeat || rc=1
    if [ "$rc" -eq 0 ]; then
      echo "[autonomous] ✓ autonomous mode ON — Hermes will act on the heartbeat's own schedule, hands-off."
    else
      echo "[autonomous] ⛔ autonomous mode NOT fully confirmed — see the ⛔ lines above and fix by hand."
    fi
    exit "$rc" ;;
  off)
    echo "[autonomous] disabling autonomous mode ('$JOB' cron paused; gateway left running)…"
    pause_heartbeat
    echo "[autonomous] Hermes will still respond to direct invocation; it just won't self-trigger on the 15-min tick." ;;
  status)
    status_cmd ;;
  *)
    echo "usage: bash tools/hermes_autonomous.sh {on|off|status}"; exit 2 ;;
esac
