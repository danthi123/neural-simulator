#!/usr/bin/env bash
# hermes-sim.sh — THE OWNER'S DESKTOP CONTROL PANEL for the neural-simulator Hermes autonomous
# dev-agent. The canonical, version-controlled copy is tools/hermes_desktop_control.sh in the
# repo; copy it to the Desktop once (it can't be written there from inside a worktree build):
#
#   cp /home/dant123/Projects/sim/tools/hermes_desktop_control.sh ~/Desktop/hermes-sim.sh
#   chmod +x ~/Desktop/hermes-sim.sh
#
# One command, no memorized flags or ~/.hermes internals:
#
#   ~/Desktop/hermes-sim.sh start      GO LIVE. Safe from ANY starting state (idempotent): clears
#                                       any earlier pause (GAME_MODE / GPU_PAUSE), hands the project
#                                       to Hermes, turns on autonomous mode (gateway + the 15-min
#                                       heartbeat cron), brings Qwen up. This is the ONE command for
#                                       "go do overnight work hands-off."
#   ~/Desktop/hermes-sim.sh stop       PAUSE for gaming / a break. Frees the GPU + CPU now, stops
#                                       new local jobs, pauses the autonomous heartbeat. Survives a
#                                       reboot (stays paused until you 'resume').
#   ~/Desktop/hermes-sim.sh resume     RESUME from a 'stop' pause -- does NOT change who drives.
#   ~/Desktop/hermes-sim.sh handback   Hand the project back to CLAUDE: unload Qwen, free the GPU,
#                                       pause the autonomous heartbeat (gateway left running; harmless).
#   ~/Desktop/hermes-sim.sh status     One screen: who's driving, Qwen, supervisor, autonomous cron,
#                                       GPU VRAM, GAME_MODE.
#   ~/Desktop/hermes-sim.sh check      POST-REBOOT / POST-SYSTEM-UPDATE HEALTH GATE. Run this before
#                                       trusting an overnight run right after a CachyOS update or any
#                                       reboot. Green/red per line, read-only, no side effects.
#   ~/Desktop/hermes-sim.sh say "<feedback>"
#                                       Queue feedback for Hermes WITHOUT interrupting it -- surfaces
#                                       once, in its next turn's context.
#   ~/Desktop/hermes-sim.sh logs       Tail the Qwen server / supervisor / autonomous / cron logs.
#
# REPO is hardcoded below -- edit it if this checkout ever moves.
set -uo pipefail
REPO=/home/dant123/Projects/sim
STATE="$REPO/research/queue"
HERMES="${HERMES_BIN:-/home/dant123/.local/bin/hermes}"

step() { echo; echo "── $1 ──"; }
ok()   { printf '  \xe2\x9c\x93 %s\n' "$1"; }
bad()  { printf '  \xe2\x9c\x97 %s\n' "$1"; }

hbin() { if [ -x "$HERMES" ]; then echo "$HERMES"; else command -v hermes 2>/dev/null; fi; }

cmd_status() {
  step "driver + Qwen + supervisor"
  bash "$REPO/tools/hermes_takeover.sh" status
  step "autonomous mode (gateway + heartbeat cron)"
  bash "$REPO/tools/hermes_autonomous.sh" status
}

cmd_start() {
  step "1/4  clearing any earlier pause (gpu_queue resume + GAME_MODE/GPU_PAUSE off)"
  if bash "$REPO/tools/game.sh" off; then ok "pause cleared (or was already clear)"; else bad "game.sh off reported a problem (see output above) -- continuing anyway"; fi

  step "2/4  handing the project to Hermes (HERMES_ACTIVE + VRAM supervisor + Qwen)"
  if bash "$REPO/tools/hermes_takeover.sh" on; then ok "Hermes is the driver, Qwen requested up"; else bad "hermes_takeover.sh on reported a problem (see output above)"; fi

  step "3/4  confirming autonomous mode (gateway + 15-min heartbeat cron)"
  h="$(hbin)"
  if [ -n "$h" ] && timeout 15 "$h" gateway status 2>&1 | grep -qi "gateway service is running"; then
    ok "gateway running -- the heartbeat will tick on its own schedule"
  else
    bad "gateway not confirmed running -- re-run: bash tools/hermes_autonomous.sh on"
  fi

  step "4/4  final status"
  cmd_status
  echo
  echo "GO-LIVE attempted. Fix any ✗ above and re-run 'start' -- every step is idempotent."
  echo "Run '~/Desktop/hermes-sim.sh check' too if this follows a reboot or a system update."
}

cmd_stop()     { step "pausing for gaming / a break"; bash "$REPO/tools/game.sh" on; }
cmd_resume()   { step "resuming from a pause"; bash "$REPO/tools/game.sh" off; }
cmd_handback() { step "handing the project back to Claude"; bash "$REPO/tools/hermes_takeover.sh" off; }
cmd_check()    { bash "$REPO/tools/hermes_health_check.sh"; }

cmd_say() {
  local msg="${1:-}"
  if [ -z "$msg" ]; then echo 'usage: hermes-sim.sh say "<feedback>"'; exit 2; fi
  bash "$REPO/tools/hermes_say.sh" "$msg"
}

cmd_logs() {
  step "qwen server log (last 20 lines)"
  tail -n 20 "$STATE/qwen_server.log" 2>/dev/null || echo "(none yet)"
  step "qwen supervisor log (last 20 lines)"
  tail -n 20 "$STATE/qwen_supervisor.log" 2>/dev/null || echo "(none yet)"
  step "autonomous-mode log (last 20 lines)"
  tail -n 20 "$STATE/hermes_autonomous.log" 2>/dev/null || echo "(none yet)"
  step "hermes cron runs (sim-heartbeat, last 10)"
  h="$(hbin)"
  if [ -n "$h" ]; then timeout 15 "$h" cron runs sim-heartbeat --limit 10 2>&1; else echo "(hermes not found)"; fi
}

usage() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
  start)    cmd_start ;;
  stop)     cmd_stop ;;
  resume)   cmd_resume ;;
  handback) cmd_handback ;;
  status)   cmd_status ;;
  check)    cmd_check ;;
  say)      shift; cmd_say "${1:-}" ;;
  logs)     cmd_logs ;;
  *)        usage ;;
esac
