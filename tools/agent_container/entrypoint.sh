#!/usr/bin/env bash
# tools/agent_container/entrypoint.sh -- container start for sim-agent. Runs as the non-root `agent` user
# (tini is PID 1, set in the Dockerfile's ENTRYPOINT, and reaps/forwards signals for everything below).
#
# Responsibilities, in order:
#   1. First boot only: install the Claude Code CLI via its NATIVE installer into the writable agent-home
#      volume (owner 2026-09-29: NOT baked into the read-only image, so its own auto-updater can keep it
#      current -- see claude-safe-update.sh, invoked by agent-start.sh, for the update+smoke+rollback cycle).
#   2. Make sure ~/.ssh exists (700) so the owner's `docker cp` of the container's restricted key (
#      docs/AGENT_HARNESS_SETUP.md step) has somewhere to land.
#   3. Start (or reattach to) one detached tmux session named "agent" -- an IDLE shell. This does NOT start a
#      model session; that is `agent start [profile]` (tools/agent_container/bin/agent-start.sh), run by the
#      owner via the `agent` fish function, or typed inside the session by hand.
#   4. Stay up as the container's foreground process (tini remains PID 1 and reaps tmux's children).
set -euo pipefail

AGENT_HOME="${AGENT_HOME:-/home/agent}"
REPO_DIR="${REPO_DIR:-/work/sim}"
LOG="$AGENT_HOME/entrypoint.log"
mkdir -p "$AGENT_HOME"
log() { echo "$(date -u '+%F %T') entrypoint: $*" | tee -a "$LOG" >&2; }

log "starting (repo=$REPO_DIR home=$AGENT_HOME)"

# --- 1. Claude Code CLI: install into the volume on first boot only --------------------------------------------
# Layout verified live 2026-09-29 (this is the standalone native CLI installer, NOT the desktop app's own
# bundled copy under ~/.config/Claude/claude-code/ that this repo's workstation happens to also have): the
# installer places each version at ~/.local/share/claude/versions/<version> and points the symlink
# ~/.local/bin/claude at the active one. `claude update` / `claude install <version>` manage that symlink
# themselves -- see claude-safe-update.sh for the update+smoke+rollback cycle built on top of it.
CLAUDE_BIN="$AGENT_HOME/.local/bin/claude"
if [ ! -e "$CLAUDE_BIN" ]; then
  log "no Claude Code install found at $CLAUDE_BIN -- running the native installer"
  if HOME="$AGENT_HOME" bash -c 'curl -fsSL https://claude.ai/install.sh | bash' >> "$LOG" 2>&1; then
    log "native installer finished ($("$CLAUDE_BIN" --version 2>/dev/null || echo 'version unknown'))"
  else
    log "WARNING: native installer failed (see $LOG) -- 'claude' will not resolve until this is fixed; " \
        "check network egress to claude.ai (docs/AGENT_HARNESS_SETUP.md's optional egress-restriction step " \
        "must allow it), then re-run: bash /usr/local/lib/agent-container/bin/claude-safe-update.sh"
  fi
else
  log "Claude Code already installed ($("$CLAUDE_BIN" --version 2>/dev/null || echo 'version unknown'))"
fi
# Record the freshly-installed (or already-present) version as "last-good" if nothing is recorded yet, so the
# FIRST `agent start`'s update+smoke+rollback cycle (claude-safe-update.sh) has a baseline to roll back to.
LAST_GOOD="$AGENT_HOME/.claude_last_good_version"
if [ ! -s "$LAST_GOOD" ] && [ -e "$CLAUDE_BIN" ]; then
  cur=$("$CLAUDE_BIN" --version 2>/dev/null | awk '{print $1}')
  [ -n "$cur" ] && echo "$cur" > "$LAST_GOOD" && log "recorded initial last-good version: $cur"
fi

# --- 2. ssh dir ------------------------------------------------------------------------------------------------
mkdir -p "$AGENT_HOME/.ssh"
chmod 700 "$AGENT_HOME/.ssh"
[ -f "$AGENT_HOME/.ssh/config" ] || : > "$AGENT_HOME/.ssh/config"
chmod 600 "$AGENT_HOME/.ssh/config" 2>/dev/null || true

# --- 3. tmux: one idle detached session, never auto-starting the model ----------------------------------------
export TMUX_TMPDIR="${TMUX_TMPDIR:-/tmp}"
if ! tmux has-session -t agent 2>/dev/null; then
  tmux new-session -d -s agent -c "$REPO_DIR" -n main
  tmux set-option -t agent status-left "sim-agent " \; set-option -t agent status-right "%H:%M"
  log "tmux session 'agent' created (idle -- run: agent start [profile] to load a model)"
else
  log "tmux session 'agent' already present (surviving a container restart's volume state? unusual but harmless)"
fi

# --- 4. stay up, OR run a one-off command (`docker compose run --rm sim-agent <cmd...>`, e.g. the smoke test --
#        step 7 of the build) -- entrypoint's own setup (install/ssh-dir/tmux) still runs first either way, so
#        a smoke test that checks `claude --version` sees a real install. ----------------------------------------
if [ "$#" -gt 0 ]; then
  log "one-off command: $*"
  exec "$@"
fi
log "ready. attach with: agent   (fish function on the workstation) / docker exec -it sim-agent tmux attach -t agent"
exec sleep infinity
