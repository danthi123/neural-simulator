#!/usr/bin/env bash
# tools/agent_gateway.sh -- runs on the WORKSTATION as the forced command for the sim-agent container's ssh
# key (owner-applied: an `authorized_keys` line with `command="bash /home/dant123/Projects/sim/tools/
# agent_gateway.sh",restrict ssh-ed25519 AAAA...`, see docs/AGENT_HARNESS_SETUP.md). sshd hands us whatever the
# container's ssh client actually asked for in $SSH_ORIGINAL_COMMAND, regardless of the forced command itself
# -- this script is the ENTIRE trust boundary between the container and the workstation's compute (the pool,
# the GPU, the local-model server). It allows exactly five things and refuses (and logs) everything else:
#
#   pool-submit <job args>     -> tools/pool_queue.sh add <job args>       (keeps its own gate-seed guard)
#   gpu-submit '<cmd>'         -> tools/local_llm/gpu_add.sh '<cmd>'      (keeps its own committed-checkout +
#                                                                          gate-seed-prereg checks)
#   results-sync                a READ-ONLY rsync source for research/findings/raw and research/queue (the
#                                container must invoke rsync with --rsync-path=results-sync, see below)
#   status                      gpu_queue + pool_queue + game.sh status, concatenated
#   yield [--max-hours N]       pause the harness (best-effort, via tools/agent_harness_ctl.sh on the server),
#                                stop the local model server, release the GPU queue for the window -- reuses
#                                tools/local_llm/llm.sh's own `yield` (its GAME_MODE / empty-queue / dirty-
#                                worktree refusals all still apply), with the restore step's "resume a
#                                workstation session" behaviour swapped for "unpause the harness + say" (see
#                                that command's --notify-cmd, and tools/gpu_queue.sh's LLM_YIELD_NOTIFY_CMD).
#
# RESULTS-SYNC MECHANICS: rsync's remote command is normally literally "rsync --server ..."; passing
# --rsync-path=results-sync on the CONTAINER side makes rsync run "results-sync --server ..." instead (nothing
# else about rsync's protocol changes), which is what lets this single forced command tell a results-sync pull
# apart from everything else. Example, run FROM the container:
#   rsync -av --rsync-path=results-sync -e "ssh -F ~/.ssh/config workstation-gateway" \
#       workstation-gateway:research/findings/raw/ /work/sim/research/findings/raw/
# Only the SENDER direction (server reads and sends -- i.e. the container reads/pulls) is ever allowed; a
# request missing `--sender` (the container trying to push/write) is refused outright.
set -uo pipefail
ROOT="${AGENT_GATEWAY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="${AGENT_GATEWAY_LOG:-$ROOT/research/queue/agent_gateway.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
CTL_HOST="${AGENT_HARNESS_CTL_HOST:-homelab-agent-ctl}"   # ssh host alias (owner's ~/.ssh/config) for
                                                            # tools/agent_harness_ctl.sh on the server
SSH_BIN="${AGENT_GATEWAY_SSH:-ssh}"                        # TEST-ONLY override; NEVER set in production
SSH_OPTS=(-o ConnectTimeout=8 -o BatchMode=yes)
RSYNC_ALLOWED_ROOTS=("$ROOT/research/findings/raw" "$ROOT/research/queue")

log() { printf '%s agent_gateway: %s\n' "$(date -u '+%F %T')" "$*" >> "$LOG"; }
refuse() { log "REFUSED: ${CMD:-<empty>}  ($1)"; echo "refused: $1" >&2; exit 1; }

CMD="${SSH_ORIGINAL_COMMAND:-}"
log "request: ${CMD:-<empty>}"
[ -n "$CMD" ] || refuse "no command given"

# split into subcommand + the rest of the line (internal spaces in REST are preserved -- `read` with two
# variables puts everything past the first word into the LAST one)
read -r SUB REST <<< "$CMD"

case "$SUB" in
  pool-submit)
    [ -n "$REST" ] || refuse "pool-submit needs arguments (a command and --checked '<reason>')"
    if ! eval "set -- $REST" 2>/dev/null; then refuse "could not parse pool-submit's arguments"; fi
    exec bash "$ROOT/tools/pool_queue.sh" add "$@"
    ;;

  gpu-submit)
    [ -n "$REST" ] || refuse "gpu-submit needs one quoted command argument"
    if ! eval "set -- $REST" 2>/dev/null; then refuse "could not parse gpu-submit's argument"; fi
    [ "$#" -eq 1 ] || refuse "gpu-submit takes exactly one quoted command argument, got $#"
    exec bash "$ROOT/tools/local_llm/gpu_add.sh" "$1"
    ;;

  results-sync)
    [ -n "$REST" ] || refuse "results-sync: no rsync protocol arguments"
    case "$REST" in
      "--server --sender "*) : ;;   # read direction only (server sends = the container reads/pulls)
      *) refuse "results-sync only allows a read (rsync --server --sender ...); refusing: $REST" ;;
    esac
    if ! eval "set -- $REST" 2>/dev/null; then refuse "could not parse results-sync's rsync arguments"; fi
    # rsync's server invocation ends with the source path last -- validate it resolves inside an allowed root.
    last="${*: -1}"
    resolved="$(cd "$ROOT" 2>/dev/null && realpath -m -- "$last" 2>/dev/null)"
    allowed=0
    for a in "${RSYNC_ALLOWED_ROOTS[@]}"; do
      case "$resolved" in "$a"|"$a"/*) allowed=1 ;; esac
    done
    [ "$allowed" = 1 ] || refuse "results-sync path '$last' is outside the allowed roots (research/findings/raw, research/queue)"
    cd "$ROOT" || refuse "could not cd to $ROOT"
    exec rsync "$@"
    ;;

  status)
    bash "$ROOT/tools/gpu_queue.sh" status
    echo
    bash "$ROOT/tools/pool_queue.sh" list
    echo
    bash "$ROOT/tools/game.sh" status
    ;;

  yield)
    # Cloud backends don't touch the workstation's GPU/model server at all -- nothing to yield. Best-effort:
    # if the ctl host cannot even be reached to ask, fall through and yield anyway (the local llm.sh `yield`
    # itself will simply refuse harmlessly if there is nothing to do, e.g. an empty GPU queue).
    kind="$("$SSH_BIN" "${SSH_OPTS[@]}" "$CTL_HOST" state 2>/dev/null | sed -n 's/^profile_kind: //p')"
    if [ "$kind" = "cloud" ]; then
      log "yield: active container profile is KIND=cloud -- no GPU/model-server pause needed"
      echo "cloud backend active -- no GPU yield needed"
      exit 0
    fi
    # Best-effort harness pause BEFORE stopping anything on this end -- "skip gracefully if the server/ctl is
    # unreachable" (plan section 4/9). agent_harness_ctl.sh's own `pause` applies a short grace delay before
    # actually freezing the container, so this ssh call (and the container's own Bash tool call that triggered
    # it) has time to return normally first.
    if ! "$SSH_BIN" "${SSH_OPTS[@]}" "$CTL_HOST" pause >> "$LOG" 2>&1; then
      log "yield: could not reach $CTL_HOST to pause the harness -- continuing anyway"
    fi
    max_hours_args=()
    if [ -n "$REST" ] && eval "set -- $REST" 2>/dev/null; then max_hours_args=("$@"); fi
    # The restore step (tools/gpu_queue.sh's llm_yield_restore_if_drained, once the GPU queue drains) runs this
    # verbatim via `bash -c "$notify" bash "<message>"` -- so "$1" here is deliberately LITERAL (single-quoted
    # semantics via escaping), filled in with the actual "GPU window finished..." message at that later point,
    # never expanded now.
    notify="$SSH_BIN ${SSH_OPTS[*]} $CTL_HOST unpause && $SSH_BIN ${SSH_OPTS[*]} $CTL_HOST say \"\$1\""
    exec bash "$ROOT/tools/local_llm/llm.sh" yield "${max_hours_args[@]}" --notify-cmd "$notify"
    ;;

  *)
    refuse "unknown command '$SUB' (allowed: pool-submit, gpu-submit, results-sync, status, yield)"
    ;;
esac
