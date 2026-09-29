#!/usr/bin/env bash
# tools/agent_container/bin/agent-start.sh <profile> -- what `agent start [profile]` (the workstation's fish
# function) sends into the tmux "agent" session's active pane. Never run automatically by the container itself
# (entrypoint.sh only creates an IDLE tmux session) -- only the owner, or this script, starts a model session.
#
# 1. Resolves the profile (arg, else ~/.default_profile, else "workstation") and exports its ANTHROPIC_* env
#    so BOTH the update step's smoke test and the eventual session talk to the same backend.
# 2. Runs claude-safe-update.sh (owner 2026-09-29: update-then-smoke-test-then-rollback, never an unattended
#    unverified build) -- best-effort; a failure there is logged but does not block starting a session on
#    whatever build is already active.
# 3. Hands off to `llm use <profile>` (tools/agent_container/bin/llm), which execs Claude Code in the
#    foreground -- this script's own process becomes that session, so tmux shows it directly.
set -uo pipefail
AGENT_HOME="${AGENT_HOME:-$HOME}"
PROFILE_DIR="${AGENT_PROFILE_DIR:-/usr/local/lib/agent-container/profiles}"

PROFILE="${1:-}"
if [ -z "$PROFILE" ]; then PROFILE="$(cat "$AGENT_HOME/.default_profile" 2>/dev/null || echo workstation)"; fi
pf="$PROFILE_DIR/$PROFILE.env"
if [ ! -f "$pf" ]; then
  echo "agent-start: no such profile '$PROFILE' ($pf not found)" >&2
  echo "available:" >&2
  /usr/local/bin/llm list >&2
  exit 2
fi
echo "$PROFILE" > "$AGENT_HOME/.default_profile"

# shellcheck disable=SC1090
BASE_URL=""; TOKEN_VAR=""; MODEL=""; CONTEXT=""; KIND=""
source "$pf"
export ANTHROPIC_BASE_URL="$BASE_URL"
export ANTHROPIC_MODEL="$MODEL"
if [ -n "$TOKEN_VAR" ]; then export ANTHROPIC_AUTH_TOKEN="${!TOKEN_VAR:-}"; fi

echo "== agent-start: profile=$PROFILE backend=$BASE_URL kind=${KIND:-local} =="
echo "== agent-start: checking for a Claude Code update (see ~/claude_update.log) =="
if ! bash /usr/local/bin/claude-safe-update.sh; then
  echo "== agent-start: claude-safe-update.sh reported a problem (see ~/claude_update.log) -- continuing on the current build =="
fi

echo "== agent-start: launching (profile=$PROFILE) =="
exec /usr/local/bin/llm use "$PROFILE"
