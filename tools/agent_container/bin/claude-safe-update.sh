#!/usr/bin/env bash
# tools/agent_container/bin/claude-safe-update.sh -- owner directive 2026-09-29: keep the harness's Claude Code
# CLI current without ever running an untested build unattended. Called by agent-start.sh before every
# `llm use <profile>` (i.e. on every `agent start`).
#
# The native installer's real layout (verified live on this image 2026-09-29): each version is a file at
# ~/.local/share/claude/versions/<version>, and ~/.local/bin/claude is a symlink to the currently-active one.
# `claude update` fetches and SWITCHES TO the latest build in one step -- there is no separate "install but
# don't activate yet" mode to test a build before it goes live, so this script tests AFTER `update` has already
# switched, and uses `claude install <version> --force` to switch back if the new build fails its smoke test
# (the old version's file is still present on disk; `--force` guarantees a real re-point + re-verify rather
# than trusting an "already installed" no-op to also fix up the symlink, at the cost of one redundant
# reinstall on the rare rollback path -- correctness over speed here).
#
# 1. Records the version active BEFORE updating (~/.claude_last_good_version, else `claude --version` itself).
# 2. Runs `claude update`.
# 3. If the active version changed, smoke-tests it: `claude -p "<prompt>"` against whatever backend the CALLER
#    already exported (ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL -- agent-start.sh sources
#    the chosen profile before calling this), with a short timeout.
# 4. Pass -> ~/.claude_last_good_version is promoted to the new version.
#    Fail -> `claude install <old-version> --force` switches back; ~/.claude_last_good_version stays the old
#    version. Either outcome is logged to ~/claude_update.log.
#
# Never touches network settings, never disables the CLI's own auto-updater (DISABLE_AUTOUPDATER is never set
# here, and claude_container_settings.json does not set `disableAutoUpdates`/`autoUpdates: false` either).
set -uo pipefail
AGENT_HOME="${AGENT_HOME:-$HOME}"
LOG="$AGENT_HOME/claude_update.log"
LAST_GOOD="$AGENT_HOME/.claude_last_good_version"
SMOKE_TIMEOUT="${CLAUDE_SMOKE_TIMEOUT_S:-45}"
SMOKE_PROMPT="${CLAUDE_SMOKE_PROMPT:-Reply with exactly one word: OK}"

log() { printf '%s claude-safe-update: %s\n' "$(date -u '+%F %T')" "$*" | tee -a "$LOG" >&2; }

current_version() { claude --version 2>/dev/null | awk '{print $1}'; }

if ! command -v claude >/dev/null 2>&1; then
  log "no 'claude' on PATH -- nothing to update (has entrypoint.sh's installer run? check ~/entrypoint.log)"
  exit 1
fi

before="$(cat "$LAST_GOOD" 2>/dev/null || true)"
[ -n "$before" ] || before="$(current_version)"
if [ -z "$before" ]; then
  log "could not determine the currently active version -- skipping the update this cycle"
  exit 1
fi

log "checking for updates (current: $before)"
if ! claude update >> "$LOG" 2>&1; then
  log "'claude update' failed (offline, or no egress to the update endpoint) -- staying on $before"
  exit 0   # staying on a known-good build is success, not failure, for this script's purposes
fi

after="$(current_version)"
if [ -z "$after" ]; then
  log "WARNING: could not read a version after 'claude update' -- leaving \$LAST_GOOD as-is ($before)"
  exit 1
fi
if [ "$after" = "$before" ]; then
  log "already up to date ($before)"
  echo "$before" > "$LAST_GOOD"
  exit 0
fi

log "updated $before -> $after -- smoke-testing the new build (timeout ${SMOKE_TIMEOUT}s)"
if [ -z "${ANTHROPIC_BASE_URL:-}" ]; then
  log "WARNING: no ANTHROPIC_BASE_URL in this environment -- the smoke test will fail to reach any backend. " \
      "agent-start.sh should source the profile before calling this script."
fi
smoke_rc=0
timeout "$SMOKE_TIMEOUT" claude -p "$SMOKE_PROMPT" >> "$LOG" 2>&1 || smoke_rc=$?
if [ "$smoke_rc" -eq 0 ]; then
  log "smoke test PASSED for $after -- promoting to last-good (was $before)"
  echo "$after" > "$LAST_GOOD"
else
  log "smoke test FAILED for $after (rc=$smoke_rc) -- rolling back to $before via 'claude install $before --force'"
  if claude install "$before" --force >> "$LOG" 2>&1; then
    rolled_back="$(current_version)"
    if [ "$rolled_back" = "$before" ]; then
      log "rollback to $before confirmed active"
    else
      log "WARNING: rollback command succeeded but active version reads '$rolled_back', expected $before -- check manually"
    fi
  else
    log "WARNING: rollback to $before FAILED -- the harness may be left on the untested build $after; check manually"
  fi
  echo "$before" > "$LAST_GOOD"
fi
