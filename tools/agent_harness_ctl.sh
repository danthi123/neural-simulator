#!/usr/bin/env bash
# tools/agent_harness_ctl.sh -- runs ON THE SERVER (homelab-server) as the forced command for a WORKSTATION
# ssh key (owner-applied: an `authorized_keys` line with `command="bash ~/sim-agent/tools/agent_harness_ctl.sh
# ",restrict ssh-ed25519 AAAA...`, see docs/AGENT_HARNESS_SETUP.md). This is the ONLY way the workstation can
# reach into the sim-agent container from the server side -- it allows exactly four things and refuses (and
# logs) everything else:
#
#   pause      docker pause the container after a short grace delay (so an in-flight command -- including the
#              very ssh call that triggered this, from tools/agent_gateway.sh's `yield` -- has time to return
#              normally before everything inside freezes; mirrors tools/local_llm/llm.sh's own
#              cmd_stop_graceful pattern). Scheduled in the background; returns immediately.
#   unpause    docker unpause, only if it is actually paused (a no-op, not an error, otherwise).
#   say '<line>'   tmux send-keys the given text (+ Enter) into the container's `agent` tmux session -- so
#              game.sh/agent_gateway.sh's restore steps can hand the model a "re-check your last step" /
#              "GPU window finished" prompt without needing an interactive attach. Sanitized: printable text
#              only, at most 300 characters. This script only delivers the keystrokes; it never interprets
#              them itself.
#   state      one-line-per-field status: container status (running/paused/absent/...), and -- if running --
#              the active profile + its KIND (tools/agent_container/profiles/*.env's KIND=local|cloud, written
#              by tools/agent_container/bin/llm) and whether a claude process is currently active. Read by
#              tools/agent_gateway.sh's `yield` (skip the pause for a cloud profile) and by tools/game.sh.
set -uo pipefail
CONTAINER="${AGENT_CONTAINER_NAME:-sim-agent}"
DOCKER="${AGENT_CTL_DOCKER:-docker}"          # TEST-ONLY override; NEVER set in production
TMUX_SESSION="${AGENT_CTL_TMUX_SESSION:-agent}"
LOG="${AGENT_CTL_LOG:-$HOME/agent_harness_ctl.log}"
PAUSE_DELAY_S="${AGENT_CTL_PAUSE_DELAY_S:-5}"
MAX_SAY_LEN=300

log() { printf '%s agent_harness_ctl: %s\n' "$(date -u '+%F %T')" "$*" >> "$LOG"; }

CMD="${SSH_ORIGINAL_COMMAND:-}"
log "request: ${CMD:-<empty>}"

read -r SUB REST <<< "$CMD"

container_status() { "$DOCKER" inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "absent"; }

# volume_size_gb <name-substring> -- the named volume's on-disk SIZE in GB (one decimal), via the daemon's own
# accounting (`docker system df -v` -- no raw filesystem access needed, so this works under the same privilege
# level `docker inspect`/`docker exec` already require). Grepped by SUBSTRING since compose prefixes the
# declared volume name with the project name (e.g. "sim-agent_sim-agent-repo"), which can vary by how the stack
# was deployed. Prints nothing (never a bare "0") if the volume cannot be found or its size cannot be parsed --
# 2026-09-29 security review, M3 ("the digest that warns when the volumes exceed a threshold").
volume_size_gb() {
  local name_substr="$1" raw
  raw=$("$DOCKER" system df -v 2>/dev/null | awk -v n="$name_substr" '$1 ~ n {print $NF; exit}')
  [ -n "$raw" ] || return 0
  python3 -c '
import re, sys
m = re.match(r"^([0-9.]+)\s*([a-zA-Z]*)$", sys.argv[1])
if not m:
    sys.exit(0)
val = float(m.group(1)); unit = m.group(2).lower()
mult = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
        "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40, "": 1}.get(unit, 1e9)
print(f"{val * mult / 1e9:.2f}")
' "$raw" 2>/dev/null
}

case "$SUB" in
  pause)
    log "scheduling '$DOCKER pause $CONTAINER' in ${PAUSE_DELAY_S}s"
    nohup bash -c "sleep '$PAUSE_DELAY_S'; '$DOCKER' pause '$CONTAINER' >> '$LOG' 2>&1" >/dev/null 2>&1 &
    disown 2>/dev/null || true
    echo "pause scheduled in ${PAUSE_DELAY_S}s"
    ;;

  unpause)
    st="$(container_status)"
    if [ "$st" = "paused" ]; then
      if "$DOCKER" unpause "$CONTAINER" >> "$LOG" 2>&1; then
        log "unpaused $CONTAINER"; echo "unpaused"
      else
        log "unpause FAILED"; echo "unpause failed" >&2; exit 1
      fi
    else
      log "unpause requested but status=$st (not paused) -- no-op"
      echo "not paused (status=$st) -- nothing to do"
    fi
    ;;

  say)
    msg="$REST"
    # strip one layer of surrounding quotes if the caller sent one (ssh clients commonly do); harmless no-op
    # if there is none.
    msg="${msg%\"}"; msg="${msg#\"}"; msg="${msg%\'}"; msg="${msg#\'}"
    if [ -z "$msg" ]; then echo "usage: say '<one line, <=${MAX_SAY_LEN} chars>'" >&2; exit 2; fi
    if [ "${#msg}" -gt "$MAX_SAY_LEN" ]; then
      log "refused say: ${#msg} chars (max $MAX_SAY_LEN)"; echo "refused: message exceeds $MAX_SAY_LEN chars" >&2; exit 1
    fi
    # REJECT CR/LF EXPLICITLY (2026-09-29 security review, LOW). A \r or \n delivered as a literal keystroke
    # into the container's tmux pane submits whatever was typed there so far to the shell, followed by
    # anything sent after it -- command injection into the model's own terminal. The printable-character check
    # below runs `grep -q` on the message's bytes; grep operates LINE BY LINE, and `-q` only needs ONE line to
    # match -- so a message with an embedded \n silently became several all-printable LINES and passed outright
    # (never actually tested as a single string), and \r itself is classified as [:space:] (POSIX: space, tab,
    # newline, vertical tab, form feed, CR are all "space"), so it passes the class check even alone. This is a
    # dedicated, whole-string check via bash's own pattern matching (never line-split) that closes both gaps at
    # once, run BEFORE the printable check so no embedded newline can reach it.
    case "$msg" in
      *$'\n'*|*$'\r'*)
        log "refused say: message contains a newline/carriage-return"
        echo "refused: message must not contain a newline or carriage return" >&2; exit 1 ;;
    esac
    if ! printf '%s' "$msg" | LC_ALL=C grep -qE '^[[:print:][:space:]]*$'; then
      log "refused say: non-printable characters"; echo "refused: message contains non-printable characters" >&2; exit 1
    fi
    # LITERAL TEXT, then a SEPARATE non-literal Enter (2026-09-29 security review, LOW). Without `-l`, `tmux
    # send-keys` treats each word of its argument as a KEY NAME when one matches (e.g. "C-c", "Enter"), not
    # literal text -- so a `say 'C-c'` sent an actual Ctrl-C keystroke into the model's terminal instead of the
    # three characters it looks like. `-l` (and `--`, so a message starting with `-` is never read as a flag)
    # forces the TEXT argument to be sent exactly as literal bytes; Enter must therefore be its OWN, separate,
    # NON-literal send-keys call -- folding it into the same `-l` invocation would type the four literal
    # characters "Enter" instead of pressing the key.
    if "$DOCKER" exec "$CONTAINER" tmux send-keys -t "$TMUX_SESSION" -l -- "$msg" >> "$LOG" 2>&1 \
       && "$DOCKER" exec "$CONTAINER" tmux send-keys -t "$TMUX_SESSION" Enter >> "$LOG" 2>&1; then
      log "said: $msg"; echo "sent"
    else
      log "say FAILED (container not running, or no '$TMUX_SESSION' tmux session?)"
      echo "failed to send -- is $CONTAINER running with a '$TMUX_SESSION' tmux session?" >&2
      exit 1
    fi
    ;;

  state)
    st="$(container_status)"
    echo "container: $st"
    if [ "$st" = "running" ]; then
      prof="$("$DOCKER" exec "$CONTAINER" cat /home/agent/.active_profile 2>/dev/null || echo none)"
      kind="$("$DOCKER" exec "$CONTAINER" cat /home/agent/.active_profile_kind 2>/dev/null || echo unknown)"
      busy="$("$DOCKER" exec "$CONTAINER" pgrep -f '/claude$' 2>/dev/null || true)"
      cver="$("$DOCKER" exec "$CONTAINER" claude --version 2>/dev/null || echo unknown)"
      echo "active_profile: $prof"
      echo "profile_kind: $kind"
      echo "claude_active: $([ -n "$busy" ] && echo yes || echo no)"
      echo "claude_version: $cver"
    fi
    # VOLUME DISK USAGE -- always reported, regardless of container state (the named volumes persist across
    # pause/stop); consumed by tools/inspection_digest.py to warn when either exceeds a threshold.
    repo_gb="$(volume_size_gb "sim-agent-repo")"
    home_gb="$(volume_size_gb "sim-agent-home")"
    echo "repo_volume_gb: ${repo_gb:-unknown}"
    echo "home_volume_gb: ${home_gb:-unknown}"
    log "state queried (status=$st)"
    ;;

  *)
    log "REFUSED: ${CMD:-<empty>}"
    echo "refused: only pause | unpause | say '<line>' | state are allowed" >&2
    exit 1
    ;;
esac
