#!/bin/bash
# tools/agent_container/entrypoint.sh -- container start for sim-agent. Runs as the non-root `agent` user
# (tini is PID 1, set in the Dockerfile's ENTRYPOINT, and reaps/forwards signals for everything below).
#
# Responsibilities, in order:
#   0. Fail closed: bin/firewall_guard.sh refuses to start the container (exit 1) unless the host firewall
#      (sim-agent.nft) is demonstrably dropping this bridge's traffic -- only when the server stack fragment sets
#      SIM_AGENT_REQUIRE_FIREWALL=1; skipped for a workstation-local build.
#   1. First boot only: install the Claude Code CLI via its NATIVE installer into the writable home bind mount
#      (${DATA_ROOT}/sim-agent/home on the server; owner 2026-09-29: NOT baked into the read-only image, so its
#      own auto-updater can keep it current -- see claude-safe-update.sh, invoked by agent-start.sh, for the
#      update+smoke+rollback cycle).
#   2. Make sure ~/.ssh exists (700). The owner installs the restricted keys into it from the HOST side
#      (docs/AGENT_HARNESS_SETUP.md), never with `docker cp`.
#   3. Start (or reattach to) one detached tmux session named "agent" -- an IDLE shell. This does NOT start a
#      model session; that is `agent start [profile]` (tools/agent_container/bin/agent-start.sh), run by the
#      owner via the `agent` fish function, or typed inside the session by hand.
#   4. Stay up as the container's foreground process (tini remains PID 1 and reaps tmux's children).
#
# NOTHING FROM THE WRITABLE HOME RUNS HERE (2026-09-30 security review, HIGH + MEDIUM). /home/agent is the agent's
# own bind mount, so anything in it is model-controlled: PATH is reset to the image's system directories on the
# next line (never inherited), every interpreter is named by absolute path, the Claude Code binary is never
# executed (its version is read from the installer's symlink), and the tmux shell starts with no config and no
# startup files (tmux -f /dev/null, bash --noprofile --norc), so code a model left in ~/.bashrc, ~/.profile or
# ~/.tmux.conf does not run at boot before anyone starts a session. ~/.local/bin is appended LAST, in the tmux
# shell only, which is the one place `claude` must resolve.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SAFE_PATH="$PATH"

AGENT_HOME="${AGENT_HOME:-/home/agent}"
REPO_DIR="${REPO_DIR:-/work/sim}"
LOG="$AGENT_HOME/entrypoint.log"
mkdir -p "$AGENT_HOME"
log() { echo "$(date -u '+%F %T') entrypoint: $*" | tee -a "$LOG" >&2; }

log "starting (repo=$REPO_DIR home=$AGENT_HOME)"

# --- 0. fail closed if the host firewall is not filtering this bridge ------------------------------------------
if ! /bin/bash /usr/local/lib/agent-container/bin/firewall_guard.sh 2>&1 | tee -a "$LOG" >&2; then
  log "firewall guard FAILED -- not starting. Sleeping 30s before exit so the restart policy does not spin."
  sleep 30
  exit 1
fi

# --- 1. Claude Code CLI: install into the volume on first boot only --------------------------------------------
# Layout verified live 2026-09-29 (this is the standalone native CLI installer, NOT the desktop app's own
# bundled copy under ~/.config/Claude/claude-code/ that this repo's workstation happens to also have): the
# installer places each version at ~/.local/share/claude/versions/<version> and points the symlink
# ~/.local/bin/claude at the active one. `claude update` / `claude install <version>` manage that symlink
# themselves -- see claude-safe-update.sh for the update+smoke+rollback cycle built on top of it.
CLAUDE_BIN="$AGENT_HOME/.local/bin/claude"
# claude_version: the active version, read from the installer's own symlink (~/.local/bin/claude ->
# ~/.local/share/claude/versions/<version>), NEVER by running the binary, which lives in the writable home.
claude_version() {
  local target
  target="$(/usr/bin/readlink "$CLAUDE_BIN" 2>/dev/null || true)"
  target="${target##*/}"
  if [[ "$target" =~ ^[0-9][0-9A-Za-z.+-]*$ ]]; then echo "$target"; else echo "unknown"; fi
}
if [ ! -e "$CLAUDE_BIN" ]; then
  log "no Claude Code install found at $CLAUDE_BIN -- running the native installer"
  if HOME="$AGENT_HOME" /bin/bash -c 'curl -fsSL https://claude.ai/install.sh | /bin/bash' >> "$LOG" 2>&1; then
    log "native installer finished (version $(claude_version))"
  else
    log "WARNING: native installer failed (see $LOG) -- 'claude' will not resolve until this is fixed; " \
        "check HTTPS egress to claude.ai (sim-agent.nft allows 443 to the public internet), then re-run: " \
        "/usr/local/lib/agent-container/bin/claude-safe-update.sh from the tmux session"
  fi
else
  log "Claude Code already installed (version $(claude_version))"
fi
# Record the freshly-installed (or already-present) version as "last-good" if nothing is recorded yet, so the
# FIRST `agent start`'s update+smoke+rollback cycle (claude-safe-update.sh) has a baseline to roll back to.
LAST_GOOD="$AGENT_HOME/.claude_last_good_version"
if [ ! -s "$LAST_GOOD" ] && [ -e "$CLAUDE_BIN" ]; then
  cur="$(claude_version)"
  if [ "$cur" != unknown ]; then echo "$cur" > "$LAST_GOOD" && log "recorded initial last-good version: $cur"; fi
fi

# --- 2. ssh dir ------------------------------------------------------------------------------------------------
mkdir -p "$AGENT_HOME/.ssh"
chmod 700 "$AGENT_HOME/.ssh"
[ -f "$AGENT_HOME/.ssh/config" ] || : > "$AGENT_HOME/.ssh/config"
chmod 600 "$AGENT_HOME/.ssh/config" 2>/dev/null || true

# --- 3. tmux: one idle detached session, never auto-starting the model ----------------------------------------
export TMUX_TMPDIR="${TMUX_TMPDIR:-/tmp}"
if ! /usr/bin/tmux has-session -t agent 2>/dev/null; then
  # The tmux server inherits this PATH (the home's bin LAST) for every shell it starts; -f /dev/null skips any
  # ~/.tmux.conf, and neither the first window nor a later one sources ~/.profile or ~/.bashrc.
  PATH="$SAFE_PATH:$AGENT_HOME/.local/bin" /usr/bin/tmux -f /dev/null new-session -d -s agent -c "$REPO_DIR" \
    -n main "/bin/bash --noprofile --norc"
  /usr/bin/tmux set-option -g default-command "/bin/bash --noprofile --norc" \; \
    set-option -t agent status-left "sim-agent " \; set-option -t agent status-right "%H:%M"
  log "tmux session 'agent' created (idle -- run: agent start [profile] to load a model)"
else
  log "tmux session 'agent' already present (unusual right after a start, but harmless)"
fi

# --- 4. stay up, OR run a one-off command (`cd ~/apps && docker compose run --rm --no-deps sim-agent <cmd...>`,
#        e.g. smoke_test.sh) -- entrypoint's own setup (guard/install/ssh-dir/tmux) still runs first either way,
#        so a smoke test that checks `claude --version` sees a real install. -------------------------------------
if [ "$#" -gt 0 ]; then
  log "one-off command: $*"
  exec "$@"
fi
log "ready. attach with: agent   (fish function on the workstation) / docker exec -it sim-agent tmux attach -t agent"
exec sleep infinity
