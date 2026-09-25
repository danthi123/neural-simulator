#!/usr/bin/env bash
# llm -- run a local model for Claude Code, and get the GPU back for experiments with one command.
#
#   llm on [profile]     load the model (default profile: tools/local_llm/default_profile) as a background service
#   llm off              unload it (frees its VRAM)
#   llm status           what is loaded, and GPU memory in use
#   llm claude [args]    open Claude Code in the current directory, talking to the local model (no Anthropic account);
#                        LLM_CLAUDE_FULL=1 skips the local-only context trims described at cmd_claude below
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
  python3 - "$HERE/profiles.json" "$1" "$PORT" "$HERE/../.." <<'EOF'
import json, os, shlex, sys
profiles = {p["name"]: p for p in json.load(open(sys.argv[1]))}
p = profiles.get(sys.argv[2])
if p is None:
    sys.exit("unknown profile %r; known: %s" % (sys.argv[2], ", ".join(profiles)))
cmd = ["llama-server", "-m", os.path.expanduser(p["model"]), "--host", "127.0.0.1", "--port", sys.argv[3],
       "--alias", "local", "-ngl", "99", "-np", "1", "-fa", "on", "-c", str(p["ctx"]), "-ctk", p["kv"], "-ctv", p["kv"],
       "--jinja"]
if p.get("chat_template_file"):
    # See tools/local_llm/templates/ -- the stock embedded template rejects a mid-conversation Claude
    # Code "system reminder" message; this profile-specific copy fixes that (tool-call formatting is
    # byte-for-byte unchanged, checked offline by templates/test_templates_offline.py).
    root = os.path.abspath(sys.argv[4])
    cmd += ["--chat-template-file", os.path.join(root, p["chat_template_file"])]
cmd += p.get("extra", [])
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

served_ctx() {   # context size (tokens) of the profile being served: the running unit's, else the default profile's
  local prof
  prof="$(systemctl --user show -p Description --value "$UNIT" 2>/dev/null | sed -n 's/^local LLM (\(.*\))$/\1/p')"
  [ -n "$prof" ] || prof="$(cat "$HERE/default_profile" 2>/dev/null || echo qwen38-27b-iq4nl-mtp)"
  python3 -c 'import json,sys; print(next(p["ctx"] for p in json.load(open(sys.argv[1])) if p["name"] == sys.argv[2]))' \
    "$HERE/profiles.json" "$prof" 2>/dev/null || true
}

# Local-only context trims for `llm claude` (they change nothing for Claude's own sessions or settings files):
#  - CLAUDE_CODE_MAX_CONTEXT_TOKENS = the served profile's ctx (65536 today). Claude Code does not know the window of a
#    model named "local" and otherwise assumes a far larger one, so auto-compact would fire only after llama-server
#    had already overflowed. An explicit CLAUDE_CODE_MAX_CONTEXT_TOKENS in the caller's environment wins. This one is
#    a correctness setting, so it applies even with LLM_CLAUDE_FULL=1 (which only drops the flags below).
#  - --disallowedTools WebSearch ReportFindings: WebSearch is an Anthropic server-side tool that cannot work against
#    llama-server, and ReportFindings only serves /code-review; dropping both removes ~3,000 characters of tool
#    schema from every request (measured 2026-09-25 with a local request recorder).
#  - --strict-mcp-config and bio-research@inline=false: a terminal CLI session loads no MCP servers and no desktop-app
#    plugins today (`claude mcp list` / `claude plugin list` are empty; measured request unchanged), so these only
#    keep it that way if either is added later.
# Hooks, CLAUDE.md, memory and project skills still load: the PreToolUse safety hooks and the LIVE-STATE anchor must
# stay, which is why --bare / --setting-sources / disableAllHooks are NOT used. CLAUDE_CODE_DISABLE_BUNDLED_SKILLS was
# measured and rejected (it grew the request by ~13,000 characters).
# The trim flags go AFTER your arguments because --disallowedTools takes a list: placed first, it would swallow a
# prompt given as a plain argument (`llm claude "fix X"`). If you pass your own --settings, --mcp-config or
# --disallowedTools, run with LLM_CLAUDE_FULL=1 to avoid mixing them.
cmd_claude() {
  is_up || cmd_on
  local ctx trim=() ctxenv=()
  ctx="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-$(served_ctx)}"
  [ -n "$ctx" ] && ctxenv=("CLAUDE_CODE_MAX_CONTEXT_TOKENS=$ctx")
  if [ "${LLM_CLAUDE_FULL:-0}" != 1 ]; then
    trim=(--strict-mcp-config --settings '{"enabledPlugins":{"bio-research@inline":false}}'
          --disallowedTools WebSearch ReportFindings)
  fi
  env "${ctxenv[@]}" \
  ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" ANTHROPIC_AUTH_TOKEN="local" ANTHROPIC_API_KEY="" \
  ANTHROPIC_MODEL="local" ANTHROPIC_SMALL_FAST_MODEL="local" ANTHROPIC_DEFAULT_HAIKU_MODEL="local" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="local" ANTHROPIC_DEFAULT_OPUS_MODEL="local" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 API_TIMEOUT_MS=1200000 \
    "$(command -v claude || ls -d "$HOME"/.config/Claude/claude-code/*/claude | sort -V | tail -1)" "$@" "${trim[@]}"
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
