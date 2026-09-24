#!/usr/bin/env bash
# llm -- run a local model for Claude Code, and get the GPU back for experiments with one command.
#
#   llm on [profile]     load the model (default profile: tools/local_llm/default_profile) as a background service
#   llm off              unload it (frees its VRAM)
#   llm status           what is loaded, and GPU memory in use
#   llm claude [args]    open Claude Code in the current directory, talking to the local model (no Anthropic account)
#   llm run <command>    unload the model, run <command> to completion, reload the model, print how it ended
#
# Profiles (model file, context size, sampling, speculative decoding) live in tools/local_llm/profiles.json and were
# chosen by tools/local_llm/bakeoff.py on this machine: 4 GB of the 24 GB card is left for the displays.
# Add `alias llm='bash ~/Projects/sim/tools/local_llm/llm.sh'` to your shell config to type just `llm`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LLM_PORT:-8080}"
UNIT="local-llm"

profile_cmd() {   # profile_cmd <name> -> the llama-server command line for that profile
  python3 - "$HERE/profiles.json" "$1" "$PORT" <<'EOF'
import json, os, shlex, sys
profiles = {p["name"]: p for p in json.load(open(sys.argv[1]))}
p = profiles.get(sys.argv[2])
if p is None:
    sys.exit("unknown profile %r; known: %s" % (sys.argv[2], ", ".join(profiles)))
cmd = ["llama-server", "-m", os.path.expanduser(p["model"]), "--host", "127.0.0.1", "--port", sys.argv[3],
       "--alias", "local", "-ngl", "99", "-np", "1", "-fa", "on", "-c", str(p["ctx"]), "-ctk", p["kv"], "-ctv", p["kv"],
       "--jinja"] + p.get("extra", [])
print(shlex.join(cmd))
EOF
}

is_up() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

wait_up() {
  for _ in $(seq 1 300); do is_up && return 0; systemctl --user is-active --quiet "$UNIT" || break; sleep 1; done
  echo "the model did not come up; recent log:"; journalctl --user -u "$UNIT" -n 20 --no-pager; return 1
}

cmd_on() {
  local prof="${1:-$(cat "$HERE/default_profile" 2>/dev/null || echo qwen38-27b-iq4nl-mtp)}"
  if systemctl --user is-active --quiet "$UNIT"; then echo "already loaded ($UNIT is running)"; return 0; fi
  local cmd; cmd="$(profile_cmd "$prof")"
  # shellcheck disable=SC2086
  systemd-run --user --quiet --unit="$UNIT" --description="local LLM ($prof)" -p MemoryMax=12G bash -c "exec $cmd"
  echo "loading $prof ..."; wait_up && echo "ready on http://127.0.0.1:$PORT ($prof)"
}

cmd_off() {
  if systemctl --user is-active --quiet "$UNIT"; then systemctl --user stop "$UNIT"; echo "unloaded"; else echo "not loaded"; fi
  systemctl --user reset-failed "$UNIT" 2>/dev/null || true
}

cmd_status() {
  if is_up; then echo "loaded: $(systemctl --user show -p Description --value "$UNIT")"; else echo "not loaded"; fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/GPU memory in use: /'
}

cmd_claude() {
  is_up || cmd_on
  ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" ANTHROPIC_AUTH_TOKEN="local" ANTHROPIC_API_KEY="" \
  ANTHROPIC_MODEL="local" ANTHROPIC_SMALL_FAST_MODEL="local" ANTHROPIC_DEFAULT_HAIKU_MODEL="local" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="local" ANTHROPIC_DEFAULT_OPUS_MODEL="local" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 API_TIMEOUT_MS=1200000 \
    "$(command -v claude || ls -d "$HOME"/.config/Claude/claude-code/*/claude | sort -V | tail -1)" "$@"
}

cmd_run() {
  [ $# -gt 0 ] || { echo "usage: llm run <command ...>"; return 2; }
  local was_up=0 rc=0; systemctl --user is-active --quiet "$UNIT" && was_up=1
  [ "$was_up" = 1 ] && cmd_off
  "$@" || rc=$?
  echo "== finished with exit code $rc at $(date '+%H:%M:%S')"
  [ "$was_up" = 1 ] && cmd_on
  return "$rc"
}

case "${1:-status}" in
  on) shift; cmd_on "$@" ;;
  off) cmd_off ;;
  status) cmd_status ;;
  claude) shift; cmd_claude "$@" ;;
  run) shift; cmd_run "$@" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
