#!/usr/bin/env bash
# tools/agent_gateway.sh -- runs on the WORKSTATION as the forced command for the sim-agent container's ssh
# key (owner-applied: an `authorized_keys` line with `command="bash /home/dant123/Projects/sim/tools/
# agent_gateway.sh",restrict ssh-ed25519 AAAA...`, see docs/AGENT_HARNESS_SETUP.md). sshd hands us whatever the
# container's ssh client actually asked for in $SSH_ORIGINAL_COMMAND, regardless of the forced command itself
# -- this script is the ENTIRE trust boundary between the container and the workstation's compute (the pool,
# the GPU, the local-model server). THE CONTAINER'S OWN DRIVER MAY BE ADVERSARIAL (2026-09-29 security review):
# it must not be able to run ANY text it supplies as a shell command, on this workstation or a pool node. It
# allows exactly three things and refuses (and logs) everything else:
#
#   gpu-submit  --sha <40hex> --runner research.runners.<name> [--prereg <path>] -- <args...>
#       STRUCTURED requests only (2026-09-29 security review, H2) -- tools/agent_gateway_submit.py validates the
#       sha (reachable from a refs/remotes/{origin,gitea}/* ref), the runner module (exists AT that sha), every
#       argument (matches a narrow allow-list charset -- no shell metacharacters at all), any --out path
#       (resolves under research/findings/raw/, no ..), and a GATE seed's --prereg (committed at that sha) --
#       THEN builds the actual command line itself. No text the container supplies ever reaches a shell as a
#       command; gpu-submit additionally runs the job bubblewrap-SANDBOXED (see that script's own header).
#   status                      gpu_queue + pool_queue + game.sh status, concatenated
#   yield [--max-hours N]       pause the harness (best-effort, via tools/agent_harness_ctl.sh on the server),
#                                stop the local model server, release the GPU queue for the window -- reuses
#                                tools/local_llm/llm.sh's own `yield` (its GAME_MODE / empty-queue / dirty-
#                                worktree refusals all still apply), with the restore step's "resume a
#                                workstation session" behaviour swapped for "unpause the harness + say" (see
#                                that command's --notify-cmd, and tools/gpu_queue.sh's LLM_YIELD_NOTIFY_CMD).
#
# POOL-SUBMIT IS REFUSED (2026-09-30 security review, HIGH #2). It dispatched agent-authored runner code (any
# research/draft-* commit the gitea deploy key can push) to the pool40/41/42 mini-PCs UNSANDBOXED, with full
# network, under a job account with passwordless sudo: a LAN foothold that bypasses everything sim-agent.nft
# confines, because the code no longer runs in the container. The only compute channels left are the sandboxed GPU
# path (bubblewrap --unshare-all, no network, resource limits) and the model port. Re-enabling it is a CODE CHANGE
# here, reviewed by the owner, and only after every pool node runs agent jobs under a dedicated no-sudo user inside
# the same bubblewrap recipe as gpu-submit (read-only checkout, writable per-job output directory, no network) --
# docs/AGENT_HARNESS_SETUP.md, "Pool nodes". tools/agent_gateway_submit.py's `pool` builder is kept for that day.
#
# RESULTS-SYNC HAS MOVED OFF THIS SCRIPT ENTIRELY (2026-09-29 security review, H3): the old `results-sync`
# subcommand here validated only the LAST rsync protocol argument, so `. /home/dant123/.ssh/ research/findings/
# raw/` could read private keys, and rsync's own `--remove-source-files`/`-L` could delete files or follow
# attacker-planted symlinks. Results now sync via a SEPARATE ssh key whose `authorized_keys` forced command is
# `rrsync -ro -no-del <abs path>/research/findings/raw` directly (rsync's own restricted-rsync wrapper, shipped
# with rsync -- read-only, no delete, confined to that one path by rrsync itself, never by text this script
# would have to parse) -- see docs/AGENT_HARNESS_SETUP.md. This script no longer accepts `results-sync` at all.
#
# H1 (2026-09-29 security review): argument splitting NEVER uses `eval`. tools/agent_gateway_parse.py splits
# $SSH_ORIGINAL_COMMAND's remainder with Python's shlex (word-splitting/quoting only -- no command substitution,
# variable expansion, or any other shell feature) and rejects any token containing a control character.
set -uo pipefail
# SELF_DIR: this script's OWN tools/ directory -- used ONLY to locate agent_gateway_parse.py, a pure, hermetic,
# side-effect-free helper that ships as part of THIS codebase's tooling and is never a thing a test needs to
# stub. ROOT (below) is the repository being operated ON/queued against -- test-overridable, since
# tools/agent_gateway_submit.py (which DOES need stubbing in tests) is resolved relative to it.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AGENT_GATEWAY_ROOT:-$(cd "$SELF_DIR/.." && pwd)}"
LOG="${AGENT_GATEWAY_LOG:-$ROOT/research/queue/agent_gateway.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
CTL_HOST="${AGENT_HARNESS_CTL_HOST:-homelab-agent-ctl}"   # ssh host alias (owner's ~/.ssh/config) for
                                                            # tools/agent_harness_ctl.sh on the server
SSH_BIN="${AGENT_GATEWAY_SSH:-ssh}"                        # TEST-ONLY override; NEVER set in production
SSH_OPTS=(-o ConnectTimeout=8 -o BatchMode=yes)
PYTHON_BIN="${AGENT_GATEWAY_PYTHON:-python3}"              # TEST-ONLY override; NEVER set in production
SUBMIT_PY="$ROOT/tools/agent_gateway_submit.py"

# sanitize <text> (2026-09-29 re-review, L3): strips every non-printable/control/escape character before it
# ever reaches the log file -- $CMD is attacker-controlled (the container's own SSH_ORIGINAL_COMMAND) and is
# logged verbatim on every request, so without this an ANSI escape sequence embedded in a request could alter
# whatever terminal later `tail`s/`cat`s this log (hide text, rewrite the title, or worse on a vulnerable
# emulator). Never strips ordinary printable text.
sanitize() { printf '%s' "$1" | LC_ALL=C tr -dc '[:print:]'; }
log() { printf '%s agent_gateway: %s\n' "$(date -u '+%F %T')" "$(sanitize "$*")" >> "$LOG"; }
refuse() { log "REFUSED: ${CMD:-<empty>}  ($1)"; echo "refused: $1" >&2; exit 1; }

CMD="${SSH_ORIGINAL_COMMAND:-}"
log "request: ${CMD:-<empty>}"
[ -n "$CMD" ] || refuse "no command given"

# split into subcommand + the rest of the line (internal spaces in REST are preserved -- `read` with two
# variables puts everything past the first word into the LAST one)
read -r SUB REST <<< "$CMD"

# parse_rest_into_args: fills the global ARGS array by shlex-splitting $REST (H1 -- NEVER `eval`). Refuses (and
# never returns) on any parse error or any token containing a control character.
parse_rest_into_args() {
  local py_out py_rc
  py_out=$(printf '%s' "$REST" | "$PYTHON_BIN" "$SELF_DIR/agent_gateway_parse.py" 2>&1)
  py_rc=$?
  if [ "$py_rc" -ne 0 ]; then
    refuse "could not parse arguments ($py_out)"
  fi
  ARGS=()
  if [ -n "$py_out" ]; then
    while IFS= read -r tok; do ARGS+=("$tok"); done <<< "$py_out"
  fi
}

case "$SUB" in
  pool-submit)
    refuse "pool-submit is disabled: pool jobs would run unsandboxed, with network, on nodes with passwordless sudo (a LAN foothold outside sim-agent.nft) -- use gpu-submit (sandboxed, no network); see docs/AGENT_HARNESS_SETUP.md, Pool nodes"
    ;;

  gpu-submit)
    [ -n "$REST" ] || refuse "gpu-submit needs a structured request (--sha ... --runner ... -- <args...>)"
    parse_rest_into_args
    exec "$PYTHON_BIN" "$SUBMIT_PY" gpu "${ARGS[@]}"
    ;;

  results-sync)
    refuse "results-sync has moved to a separate, read-only rrsync key (rrsync -ro -no-del research/findings/raw) -- see docs/AGENT_HARNESS_SETUP.md; this gateway no longer handles it"
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
    if [ -n "$REST" ]; then
      parse_rest_into_args   # H1 -- was `eval "set -- $REST"`; same shlex-based parser as every other subcommand
      max_hours_args=("${ARGS[@]}")
    fi
    # The restore step (tools/gpu_queue.sh's llm_yield_restore_if_drained, once the GPU queue drains) runs this
    # verbatim via `bash -c "$notify" bash "<message>"` -- so "$1" here is deliberately LITERAL (single-quoted
    # semantics via escaping), filled in with the actual "GPU window finished..." message at that later point,
    # never expanded now.
    notify="$SSH_BIN ${SSH_OPTS[*]} $CTL_HOST unpause && $SSH_BIN ${SSH_OPTS[*]} $CTL_HOST say \"\$1\""
    exec bash "$ROOT/tools/local_llm/llm.sh" yield "${max_hours_args[@]}" --notify-cmd "$notify"
    ;;

  *)
    refuse "unknown command '$SUB' (allowed: gpu-submit, status, yield)"
    ;;
esac
