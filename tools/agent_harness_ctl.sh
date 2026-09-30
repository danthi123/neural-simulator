#!/usr/bin/env bash
# tools/agent_harness_ctl.sh -- runs ON THE SERVER (homelab-server / srv-host-0) as the forced command for a
# WORKSTATION ssh key (owner-applied: an `authorized_keys` line with `command="bash /home/dant123/data/sim-agent/
# src/tools/agent_harness_ctl.sh",restrict,from="192.168.0.68" ssh-ed25519 AAAA...`, see
# docs/AGENT_HARNESS_SETUP.md -- it runs from the owner's source checkout on the dataset, which the container
# cannot reach, never from the agent's own writable repo). This is the ONLY way the workstation can
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
#              the active profile + its KIND (from the SERVER-LOCAL record `set-active-profile` writes, never
#              from the container) and whether a claude process is currently active; always the ZFS dataset's
#              usage against its refquota, the gitea dataset's usage against its quota (the deploy key can push
#              into it) and whether sim-agent-fw.service is active. Read by
#              tools/agent_gateway.sh's `yield` (skip the pause for a cloud profile), tools/game.sh and
#              tools/inspection_digest.py.
#   set-active-profile <name>   records which profile is active + its KIND (read from this script's OWN
#              sibling profiles/<name>.env on the server), so `state`'s gaming-pause-skip decision never has to
#              trust anything the container itself could write (2026-09-29 re-review, L3). Called by `agent
#              start` from the HOST, over this same trusted ctl channel -- the container has no access to it.
set -uo pipefail
CONTAINER="${AGENT_CONTAINER_NAME:-sim-agent}"
DOCKER="${AGENT_CTL_DOCKER:-docker}"          # TEST-ONLY override; NEVER set in production
TMUX_SESSION="${AGENT_CTL_TMUX_SESSION:-agent}"
LOG="${AGENT_CTL_LOG:-$HOME/agent_harness_ctl.log}"
PAUSE_DELAY_S="${AGENT_CTL_PAUSE_DELAY_S:-5}"
MAX_SAY_LEN=300
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${AGENT_CTL_PROFILE_DIR:-$SELF_DIR/agent_container/profiles}"
ZFS="${AGENT_CTL_ZFS:-zfs}"                        # TEST-ONLY override; NEVER set in production
SYSTEMCTL="${AGENT_CTL_SYSTEMCTL:-systemctl}"      # TEST-ONLY override; NEVER set in production
DATASET="${AGENT_CTL_DATASET:-data/appdata/sim-agent}"
GITEA_DATASET="${AGENT_CTL_GITEA_DATASET:-data/appdata/gitea}"
FW_UNIT=sim-agent-fw.service
# Every `docker exec` into the container pins PATH and names its binary absolutely (2026-09-30 security review,
# HIGH): an exec inherits the container's environment, and the home bind mount is model-writable. Nothing from
# that home is ever executed here: the Claude Code version is read from the installer's symlink, not by running it.
EXEC_ENV_PATH="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# SERVER-LOCAL active-profile record (2026-09-29 re-review, L3). NEVER inside the container's own writable
# home volume: `state` used to `docker exec ... cat /home/agent/.active_profile_kind`, a file the CONTAINER
# itself writes -- a possibly-adversarial model could simply `echo cloud > ~/.active_profile_kind` to claim a
# cloud backend and make tools/game.sh / tools/agent_gateway.sh's `yield` skip pausing it for gaming entirely,
# defeating the owner's own control over their GPU. This file lives on the SERVER's own filesystem, written
# only by THIS script's own `set-active-profile` (below), which determines KIND itself by reading its
# sibling profiles/<name>.env -- never anything the container can reach or influence.
ACTIVE_PROFILE_STATE="${AGENT_CTL_ACTIVE_PROFILE_STATE:-$HOME/.sim_agent_active_profile}"

log() { printf '%s agent_harness_ctl: %s\n' "$(date -u '+%F %T')" "$*" >> "$LOG"; }

# sanitize <text> (2026-09-29 re-review, L3): strips every non-printable/control/escape character, so nothing
# `state` (claude_version is a container-side symlink target) or a logged request can smuggle an ANSI escape sequence
# or other control byte into a terminal that later displays it. Never strips ordinary printable text.
sanitize() { printf '%s' "$1" | LC_ALL=C tr -dc '[:print:]'; }

CMD="${SSH_ORIGINAL_COMMAND:-}"
log "request: ${CMD:-<empty>}"

read -r SUB REST <<< "$CMD"

container_status() { "$DOCKER" inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "absent"; }

# appdata_usage -- prints "<referenced_gb> <refquota_gb|none>" for the sim-agent ZFS dataset (both 2 decimals), or
# nothing if it cannot be read. `zfs get` needs no privilege, so this works as the forced-command account without
# sudo, and it is instant (no du over a multi-GB checkout). REFERENCED is the live data the refquota caps;
# snapshots are excluded on purpose (sanoid prunes them). 2026-09-29 security review M3 ("the digest that warns
# when the volumes exceed a threshold"), rebased from named volumes onto the dataset 2026-09-30.
appdata_usage() {
  "$ZFS" get -Hp -o value referenced,refquota "$DATASET" 2>/dev/null | awk '
    NR == 1 && $1 ~ /^[0-9]+$/ { used = sprintf("%.2f", $1 / 1e9) }
    NR == 2 && $1 ~ /^[0-9]+$/ { quota = ($1 == 0) ? "none" : sprintf("%.2f", $1 / 1e9) }
    END { if (used != "") print used, (quota != "" ? quota : "unknown") }'
}

# gitea_usage -- "<used_gb> <quota_gb|none>" for the gitea dataset (2026-09-30 security review, MEDIUM): the
# container's write deploy key can push unlimited research/draft-* data into it, and it shares the `data` pool with
# every other stack. USED and QUOTA (not refquota) because the recommended cap is `quota`, which counts snapshots.
gitea_usage() {
  "$ZFS" get -Hp -o value used,quota "$GITEA_DATASET" 2>/dev/null | awk '
    NR == 1 && $1 ~ /^[0-9]+$/ { used = sprintf("%.2f", $1 / 1e9) }
    NR == 2 && $1 ~ /^[0-9]+$/ { quota = ($1 == 0) ? "none" : sprintf("%.2f", $1 / 1e9) }
    END { if (used != "") print used, (quota != "" ? quota : "unknown") }'
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
    if "$DOCKER" exec -e "$EXEC_ENV_PATH" "$CONTAINER" /usr/bin/tmux send-keys -t "$TMUX_SESSION" -l -- "$msg" >> "$LOG" 2>&1 \
       && "$DOCKER" exec -e "$EXEC_ENV_PATH" "$CONTAINER" /usr/bin/tmux send-keys -t "$TMUX_SESSION" Enter >> "$LOG" 2>&1; then
      log "said: $msg"; echo "sent"
    else
      log "say FAILED (container not running, or no '$TMUX_SESSION' tmux session?)"
      echo "failed to send -- is $CONTAINER running with a '$TMUX_SESSION' tmux session?" >&2
      exit 1
    fi
    ;;

  set-active-profile)
    # 2026-09-29 re-review, L3: the SERVER-side record of "which profile/kind is active" -- called by `agent
    # start` FROM THE HOST (never by the container, which has no access to this ctl channel at all: it holds
    # only its own separate workstation-gateway key). KIND is determined HERE, from this script's own sibling
    # profiles/<name>.env -- never asked of, or trusted from, the container.
    name="$REST"
    if ! [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
      log "REFUSED set-active-profile: unsafe name '$name'"
      echo "refused: profile name must match ^[A-Za-z0-9_.-]+\$" >&2; exit 1
    fi
    pf="$PROFILE_DIR/$name.env"
    if [ ! -f "$pf" ]; then
      log "REFUSED set-active-profile: no such profile file $pf"
      echo "refused: no such profile '$name' ($pf not found on the server)" >&2; exit 1
    fi
    kind="$(sed -n 's/^KIND=//p' "$pf" | tail -1)"
    { echo "PROFILE=$(sanitize "$name")"; echo "KIND=$(sanitize "${kind:-unknown}")"; } > "$ACTIVE_PROFILE_STATE"
    log "set-active-profile: $name (kind=${kind:-unknown})"
    echo "recorded: $name (kind=${kind:-unknown})"
    ;;

  state)
    st="$(container_status)"
    echo "container: $st"
    if [ "$st" = "running" ]; then
      prof="none"; kind="unknown"
      if [ -f "$ACTIVE_PROFILE_STATE" ]; then
        prof="$(sed -n 's/^PROFILE=//p' "$ACTIVE_PROFILE_STATE" | tail -1)"
        kind="$(sed -n 's/^KIND=//p' "$ACTIVE_PROFILE_STATE" | tail -1)"
        [ -n "$prof" ] || prof="none"
        [ -n "$kind" ] || kind="unknown"
      fi
      busy="$("$DOCKER" exec -e "$EXEC_ENV_PATH" "$CONTAINER" /usr/bin/pgrep -f '/claude$' 2>/dev/null || true)"
      # the active version = the basename of the installer's symlink target (~/.local/share/claude/versions/<v>)
      cver="$("$DOCKER" exec -e "$EXEC_ENV_PATH" "$CONTAINER" /usr/bin/readlink /home/agent/.local/bin/claude 2>/dev/null)"
      cver="${cver##*/}"
      [ -n "$cver" ] || cver=unknown
      echo "active_profile: $(sanitize "$prof")"
      echo "profile_kind: $(sanitize "$kind")"
      echo "claude_active: $([ -n "$busy" ] && echo yes || echo no)"
      echo "claude_version: $(sanitize "$cver")"
    fi
    # APPDATA + FIREWALL -- always reported, regardless of container state (the dataset persists across
    # pause/stop); consumed by tools/inspection_digest.py.
    read -r used_gb quota_gb <<< "$(appdata_usage)"
    echo "appdata_gb: ${used_gb:-unknown}"
    echo "appdata_quota_gb: ${quota_gb:-unknown}"
    read -r g_used_gb g_quota_gb <<< "$(gitea_usage)"
    echo "gitea_gb: ${g_used_gb:-unknown}"
    echo "gitea_quota_gb: ${g_quota_gb:-unknown}"
    fw="$("$SYSTEMCTL" is-active "$FW_UNIT" 2>/dev/null)"
    echo "firewall: $(sanitize "${fw:-unknown}")"
    log "state queried (status=$st)"
    ;;

  *)
    log "REFUSED: ${CMD:-<empty>}"
    echo "refused: only pause | unpause | say '<line>' | state | set-active-profile <name> are allowed" >&2
    exit 1
    ;;
esac
