#!/usr/bin/env bash
# llm -- run a local model for Claude Code, and get the GPU back for experiments with one command.
#
#   llm on [profile] [--wait]  load the model (default profile: tools/local_llm/default_profile). Refuses while
#                              a tools/gpu_queue.sh job is running/queued (same 3090 -- see COORDINATION below);
#                              --wait blocks until the queue drains instead of refusing.
#   llm off              unload it (frees its VRAM); also cancels any pending gpu_queue auto-restore (below)
#   llm status           what is loaded, and GPU memory in use
#   llm claude [args]    open Claude Code in the current directory, talking to the local model (no Anthropic account);
#                        LLM_CLAUDE_FULL=1 skips the local-only context trims described at cmd_claude below.
#                        Each new session's id is recorded; `llm claude --continue` (or -c) resumes the LAST LOCAL
#                        session, never Claude Code's own "most recent in this folder" (that can be a huge Anthropic
#                        desktop session that overflows the local window); `llm claude --resume <id>` resumes any.
#   llm resume           same as `llm claude --continue`
#   llm run <command>    unload the model, run <command> to completion, reload the model, print how it ended
#   llm yield [--max-hours N]   DRIVER-initiated GPU handoff (default 10h): stop this unit (gracefully) so
#                                tools/gpu_queue.sh can drain its queue, then auto-reload the SAME profile and
#                                resume THIS session headless once the queue empties or the window expires.
#                                Refuses under GAME_MODE, with an empty GPU queue, or with uncommitted tracked
#                                changes in any .claude/worktrees/draft-* worktree.
#
# Profiles (model file, context size, sampling, speculative decoding) live in tools/local_llm/profiles.json and were
# chosen by tools/local_llm/bakeoff.py on this machine: 4 GB of the 24 GB card is left for the displays.
# Add `alias llm='bash ~/Projects/sim/tools/local_llm/llm.sh'` to your shell config to type just `llm`.
#
# UNDER THE HOOD (2026-09-29): the "local-llm" unit runs llama-swap (tools/local_llm/llama-swap.yaml), which
# itself launches the SAME llama-server command as before for whichever profile is selected, auto-unloads it
# after an hour idle (globalTTL, frees the 3090), and serves both loopback and the LAN. See
# docs/LOCAL_MODEL_GUIDE.md to add a model or change the default -- it is a plain-language walkthrough, no
# Claude session needed.
#
# COORDINATION WITH tools/gpu_queue.sh (2026-09-25 auto-swap; retired 2026-09-29 for an explicit GPU YIELD --
# docs/plans/2026-09-29-local-model-transition-PLAN.md section 3). This unit and tools/gpu_queue.sh jobs both want
# the one 3090 and are not coordinated on their own. The ORIGINAL fix had gpu_queue.sh stop this unit itself the
# instant a job was queued -- with the local model now the day-to-day DRIVER, that stop lands mid-session and
# loses whatever it was doing. So a queued GPU job no longer stops this unit automatically: gpu_queue.sh's
# dispatcher just HOLDS the job (never starts it) for as long as this unit is active. The driver frees the card
# only by choosing to, via `llm yield` above: it stops itself (gracefully -- see cmd_stop_graceful) and writes
# research/queue/.llm_yield, naming the session/profile to restore and a deadline; gpu_queue.sh's dispatcher
# drains its queue while that marker exists, then reloads the SAME profile and resumes the SAME session headless.
# GAME_MODE (tools/game.sh) still takes precedence over all of this -- checked by both scripts, and gaming's own
# pause already holds the queue regardless.
#
# A separate, still-live marker: research/queue/.local_llm_was_on, CONTENTS = the profile name. `llm on` (the
# OWNER, not the driver) refusing to load because a GPU job is busy writes this itself, recording the intent to
# load once the queue clears -- `llm off` always removes it (a manual/explicit off cancels that pending
# auto-restore, even mid-job), and gpu_queue.sh's own llm_restore_if_idle consumes it right before its one
# restart attempt (so a crash or a failed restart is never retried forever). Gaming's PAUSE sentinel defers that
# restore rather than skipping it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LLM_PORT:-8080}"
UNIT="local-llm"
SYSTEMCTL=${LOCAL_LLM_SYSTEMCTL:-systemctl}   # TEST-ONLY override (shared name with gpu_queue.sh); NEVER set in production
# 2026-09-29 (serving bake-off, research/serving-bakeoff/tools/local_llm/results/summary.md): the "local-llm" unit
# now runs llama-swap (tools/local_llm/llama-swap.yaml), not a bare llama-server -- llama-swap still spawns the
# SAME llama-server command profile_cmd() builds below (reused byte-for-byte by the config; see __profile_cmd and
# tests/test_llama_swap_config.py), just fronted by a proxy that can hold every profile.json entry, hot-swap
# between them, and auto-unload an idle one (ttl). LLAMA_SWAP_BIN/LLAMA_SWAP_YAML are TEST-ONLY overrides.
LLAMA_SWAP_YAML="${LLAMA_SWAP_YAML:-$HERE/llama-swap.yaml}"
LLAMA_SWAP_BIN="${LLAMA_SWAP_BIN:-$(command -v llama-swap || echo "$HOME/.local/bin/llama-swap")}"
LLAMA_SWAP_PROFILE_TIMEOUT_S="${LLAMA_SWAP_PROFILE_TIMEOUT_S:-620}"   # >= llama-swap.yaml's healthCheckTimeout
# Resolve the SAME shared queue dir gpu_queue.sh uses (its singleton is per-repo, not per-worktree -- see that
# file's SHARED_ROOT comment), so a marker either script writes is visible to the other regardless of which
# worktree invoked them. GPU_QUEUE_DIR overrides it (tests only).
_GIT_COMMON=$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [ -n "$_GIT_COMMON" ] && [ -d "$(dirname "$_GIT_COMMON")" ]; then _LLM_SHARED_ROOT="$(dirname "$_GIT_COMMON")"; else _LLM_SHARED_ROOT="$HERE/../.."; fi
QDIR="${GPU_QUEUE_DIR:-$_LLM_SHARED_ROOT/research/queue}"
LLM_WAS_ON_MARKER="$QDIR/.local_llm_was_on"   # gpu_queue.sh's own `llm on`-refusal marker (its llm_restore_if_idle); CONTENTS = the profile to restore
LLM_YIELD_MARKER="$QDIR/.llm_yield"           # `llm yield`'s own marker; LLM_YIELD_SESSION=/LLM_YIELD_PROFILE=/LLM_YIELD_DEADLINE=<epoch seconds>
GAME_MODE_FILE="$QDIR/GAME_MODE"              # tools/game.sh's sentinel -- absolute priority over a yield
LLM_WAIT_POLL_SEC="${LLM_WAIT_POLL_SEC:-5}"   # `llm on --wait` poll cadence; TEST-ONLY override for a fast test
DPID="$QDIR/gpu_queue.dpid"                   # gpu_queue.sh's own dispatcher-pid + singleton-lock files (read-only from here)
DLOCK="$QDIR/.gpu_daemon.lock"

# Mirrors gpu_queue.sh's own daemon_alive(): a live recorded pid (fast path) OR the singleton lock being held
# (authoritative -- survives a lost/stale dpid, e.g. a SIGKILLed daemon) means a dispatcher is genuinely up.
dispatcher_alive() {
  if [ -f "$DPID" ] && kill -0 "$(cat "$DPID" 2>/dev/null)" 2>/dev/null; then return 0; fi
  [ -f "$DLOCK" ] || return 1
  if ( exec 8>"$DLOCK"; flock -n 8 ) 2>/dev/null; then return 1; fi   # acquired freely => nobody holds it => no daemon
  return 0                                                            # could not acquire => a live daemon holds it
}

# A queued/running GPU job is only real CONTENTION for the card when (a) a dispatcher is actually alive to act
# on it, and (b) gpu.running's recorded pid (if any) is itself still alive. Previously this trusted bare file
# existence, so a stale gpu.running or gpu.queue left behind by a dispatcher that crashed/was killed made `llm
# on` refuse FOREVER even though nothing was ever going to dequeue and contend for the GPU (MEDIUM finding,
# 2026-09-25 fix-round).
gpu_job_busy() {
  dispatcher_alive || return 1
  [ -s "$QDIR/gpu.queue" ] && return 0
  if [ -f "$QDIR/gpu.running" ]; then
    local rp; rp=$(cut -f1 "$QDIR/gpu.running" 2>/dev/null)
    [ -n "$rp" ] && kill -0 "$rp" 2>/dev/null && return 0
  fi
  return 1
}

current_profile() {
  # The profile name embedded in this unit's own Description ("local LLM (<profile>)"), set by cmd_on's
  # systemd-run --description. Used by `cmd_yield` to record WHICH profile to restore in the .llm_yield marker
  # (LOW finding, 2026-09-25 fix-round: a bare marker always restored the DEFAULT profile, even when a
  # different one was actually running). Empty when not up / unparseable.
  local desc
  desc=$("$SYSTEMCTL" --user show -p Description --value "$UNIT" 2>/dev/null || true)
  case "$desc" in
    "local LLM ("*")") desc="${desc#local LLM (}"; echo "${desc%)}" ;;
  esac
}

profile_cmd() {   # profile_cmd <name> -> the llama-server command line for that profile (reused verbatim, with
  # $PORT swapped for llama-swap's own ${PORT} macro, by tools/local_llm/llama-swap.yaml's `cmd:` entries --
  # see the hidden `__profile_cmd` subcommand below and tests/test_llama_swap_config.py, which regenerates
  # every profile's expected command line through THIS function and fails if the committed YAML has drifted.
  # -np 1 was INVESTIGATED, not merely inherited (research/local-llm-prompt-cache branch, 2026-09-25,
  # tools/local_llm/cache_probe.py + tools/local_llm/results/cache_probe.md): a live `llm claude` session
  # reprocesses its ENTIRE prompt every turn instead of reusing the cached prefix.
  #
  # ROUND 1 swept llama-server flags (-np 1 baseline, -np 2 + -kvu, + denser -ctxcp/-cms, + doubled -cram, +
  # --cache-reuse) against the same real multi-turn task: all five showed 0% prompt-cache reuse and concluded
  # this was an upstream llama.cpp hybrid-model (Gated DeltaNet) limitation with nothing to fix. THAT CONCLUSION
  # WAS WRONG -- it was inferred from server-log symptoms, never from a captured real request.
  #
  # ROUND 2 (coordinator challenge: test directly, don't infer) captured real request bodies from a live session
  # (tools/local_llm/capture_session.py) and rendered them through the chat template via llama-server's
  # /apply-template (tools/local_llm/render_and_diff.py): the actual cause was tools/local_llm/templates/
  # qwen38-27b-iq4nl-mtp.jinja itself -- a prior fix for a real "System message must be at the beginning" error
  # merged EVERY mid-conversation "system-reminder" Claude Code sends (e.g. a live token-count line, refreshed
  # every turn) into the ONE leading system block, shifting that block's boundary each turn and invalidating the
  # cached prefix for the entire rest of the conversation (measured: divergence at 76% into turn 1's own
  # rendered prompt). FIXED in the template ("LOCAL FIX 2026-09-25 ROUND 2"): only the leading run of system/
  # developer messages is merged; a later one renders in place instead. Re-measured end-to-end at -np 1 (no
  # flag change): prompt-cache reuse 0% -> 60.6% overall (98.6% on the largest turn), prompt-processing time
  # roughly halved. See tools/local_llm/results/cache_probe.md for both rounds' full numbers.
  #
  # -np 1 itself is kept because round 1's server-flag sweep is still a valid negative result on its own terms
  # (none of -kvu/-ctxcp/-cms/-cram/--cache-reuse moved the needle once the template was still broken, and the
  # -kvu multi-slot variants cost more VRAM); the actual fix needed no -np/-kvu change at all.
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

is_up() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }   # llama-swap's OWN health, not a profile's

wait_up() {
  for _ in $(seq 1 300); do is_up && return 0; "$SYSTEMCTL" --user is-active --quiet "$UNIT" || break; sleep 1; done
  echo "llama-swap did not come up; recent log:"; journalctl --user -u "$UNIT" -n 20 --no-pager; return 1
}

llama_swap_cmd() {   # the launch line for the "local-llm" unit itself: llama-swap, fronting every profile.json
  # entry (tools/local_llm/llama-swap.yaml). -listen 0.0.0.0 (not 127.0.0.1): the server harness needs the LAN
  # address too (owner 2026-09-29); there is no apiKeys: entry, so treat the LAN as trusted, same as any other
  # unauthenticated service on this box. -watch-config: docs/LOCAL_MODEL_GUIDE.md's edits (add/switch a model)
  # take effect without a restart.
  printf '%s -config %s -listen 0.0.0.0:%s -watch-config' "$LLAMA_SWAP_BIN" "$LLAMA_SWAP_YAML" "$PORT"
}

activate_profile() {   # activate_profile <name> -- select it as llama-swap's active profile (pins the "local"
  # model id -- what llm.sh/Claude Code always request, ANTHROPIC_MODEL=local -- to this profile's real model
  # id) and force it to load NOW, synchronously, so `llm on <profile>` keeps its old eager-load meaning instead
  # of silently deferring to the first real request. /upstream/<id>/health is llama-swap's own passthrough
  # (README: "direct access to upstream server"); it waits for the backend's own /health inside
  # llama-swap.yaml's healthCheckTimeout before answering, exactly like any other proxied request would.
  local prof="$1"
  curl -sf -X PUT -H 'Content-Type: application/json' -d "{\"name\":\"$prof\"}" \
    "http://127.0.0.1:$PORT/api/profiles/active" >/dev/null || {
    echo "could not select profile '$prof' (missing from tools/local_llm/llama-swap.yaml's profiles: block?)"; return 1; }
  curl -sf --max-time "$LLAMA_SWAP_PROFILE_TIMEOUT_S" "http://127.0.0.1:$PORT/upstream/$prof/health" >/dev/null || {
    echo "profile '$prof' did not become ready within ${LLAMA_SWAP_PROFILE_TIMEOUT_S}s; recent log:"
    journalctl --user -u "$UNIT" -n 30 --no-pager; return 1; }
}

cmd_on() {
  local wait_for_gpu=0 prof=""
  for a in "$@"; do case "$a" in --wait) wait_for_gpu=1 ;; *) prof="$a" ;; esac; done
  prof="${prof:-$(cat "$HERE/default_profile" 2>/dev/null || echo qwen38-27b-iq4nl-mtp)}"
  if "$SYSTEMCTL" --user is-active --quiet "$UNIT"; then echo "already loaded ($UNIT is running)"; return 0; fi
  if gpu_job_busy; then
    if [ "$wait_for_gpu" = 1 ]; then
      echo "a GPU job is running/queued (tools/gpu_queue.sh status) -- waiting for it to drain before loading $prof (Ctrl-C to give up)…"
      while gpu_job_busy; do sleep "$LLM_WAIT_POLL_SEC"; done
      echo "GPU is clear -- proceeding to load."
    else
      # Record the owner's INTENT to have $prof loaded, so the promise on the next line is actually kept by
      # gpu_queue's own llm_restore_if_idle once the queue drains -- previously this refusal promised an
      # auto-load that never happened unless gpu_queue itself had stopped an ALREADY-running unit for a job (the
      # marker was only ever written there, never here) (MEDIUM finding, 2026-09-25 fix-round).
      printf '%s\n' "$prof" > "$LLM_WAS_ON_MARKER"
      echo "refusing to load $prof: a GPU job is running/queued (see: tools/gpu_queue.sh status)."
      echo "it will load automatically once the queue drains, or run:  llm on $prof --wait"
      return 1
    fi
  fi
  local swap_cmd; swap_cmd="$(llama_swap_cmd)"
  # A unit left FAILED (e.g. a stop that timed out and was SIGKILLed, 2026-09-26) stays loaded, and systemd-run then
  # refuses the same name ("already loaded or has a fragment file"); clear it first. Harmless when there is none.
  "$SYSTEMCTL" --user reset-failed "$UNIT" 2>/dev/null || true
  # LLAMA_SWAP_REPO_ROOT: llama-swap.yaml's `${repo_root}` macro (its --chat-template-file paths) resolves
  # through THIS env var rather than a path baked into the committed YAML, so the same file is correct whether
  # "local-llm" is ever started from the main checkout or from a .claude/worktrees/* dev copy.
  # shellcheck disable=SC2086
  systemd-run --user --quiet --unit="$UNIT" --description="local LLM ($prof)" -p MemoryMax=12G \
    --setenv=LLAMA_SWAP_REPO_ROOT="$(cd "$HERE/../.." && pwd)" bash -c "exec $swap_cmd"
  echo "starting llama-swap, then loading $prof ..."
  wait_up || return 1
  activate_profile "$prof" && echo "ready on http://127.0.0.1:$PORT ($prof)"
}

cmd_off() {
  if "$SYSTEMCTL" --user is-active --quiet "$UNIT"; then "$SYSTEMCTL" --user stop "$UNIT"; echo "unloaded"; else echo "not loaded"; fi
  "$SYSTEMCTL" --user reset-failed "$UNIT" 2>/dev/null || true
  # An explicit/manual off ALWAYS cancels gpu_queue's pending "restore it when the queue drains" intent -- even
  # when this unit was already down for some other reason (the owner's `llm off` must stick; the queue draining
  # afterward must not silently reload it against that choice).
  rm -f "$LLM_WAS_ON_MARKER"
}

# __stop_graceful <delay-seconds>: used by `llm yield` and by `game.sh on --graceful`. delay<=0 stops immediately
# (same body as cmd_off, minus its marker cleanup -- callers that need that do it themselves); delay>0 schedules
# the stop via a one-shot transient systemd timer and returns immediately, so a caller invoking this from INSIDE
# its own chat turn (the model calling `llm yield` on itself) can finish replying before the server actually
# dies. Never blocks, never waits for the stop to land.
cmd_stop_graceful() {
  local delay="${1:-0}"
  if [ "$delay" -gt 0 ] 2>/dev/null && "$SYSTEMCTL" --user is-active --quiet "$UNIT"; then
    systemd-run --user --quiet --on-active="${delay}s" --unit="${UNIT}-graceful-stop-$$" \
      bash -c "\"$SYSTEMCTL\" --user stop \"$UNIT\"; \"$SYSTEMCTL\" --user reset-failed \"$UNIT\" >/dev/null 2>&1 || true"
    echo "graceful stop of $UNIT scheduled in ${delay}s"
  else
    if "$SYSTEMCTL" --user is-active --quiet "$UNIT"; then "$SYSTEMCTL" --user stop "$UNIT"; echo "unloaded"; else echo "not loaded"; fi
    "$SYSTEMCTL" --user reset-failed "$UNIT" 2>/dev/null || true
  fi
}

cmd_status() {
  if is_up; then
    echo "loaded: $("$SYSTEMCTL" --user show -p Description --value "$UNIT")"
    # GET /running (llama-swap API): which profile.json entry is ACTUALLY resident right now -- can legitimately
    # be "none" after llama-swap.yaml's globalTTL idle-unloads it; the next request (or `llm on`) reloads it.
    local running; running="$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/running" 2>/dev/null)"
    [ -n "$running" ] && printf '%s' "$running" | python3 -c '
import json, sys
try: models = json.load(sys.stdin).get("running", [])
except Exception: models = []
print("resident: " + (", ".join("%s (%s)" % (m["model"], m["state"]) for m in models) or "none (idle-unloaded, or not requested yet)"))
' 2>/dev/null
  else
    echo "not loaded"
  fi
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
LLM_SESSION_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/sim-local-llm/last_session"
# Session bookkeeping (2026-09-25): Claude Code's own --continue picks the most recent session in this project folder,
# which is usually the owner's Anthropic desktop session -- a transcript far larger than the local window. So --continue
# / -c is rewritten to --resume <the last session llm itself started>, and every new session gets a recorded id.
llm_session_args() {   # llm_session_args <args...> -> prints NUL-separated args with session handling applied
  local out=() a have_session=0 want_continue=0 next_is_id=0 last=""
  for a in "$@"; do
    if [ "$next_is_id" = 1 ]; then next_is_id=0; last="$a"; out+=("$a"); continue; fi
    case "$a" in
      -c|--continue) want_continue=1 ;;
      -r|--resume) have_session=1; next_is_id=1; out+=("$a") ;;
      --resume=*) have_session=1; last="${a#--resume=}"; out+=("$a") ;;
      --session-id) have_session=1; next_is_id=1; out+=("$a") ;;
      *) out+=("$a") ;;
    esac
  done
  if [ "$want_continue" = 1 ] && [ "$have_session" = 0 ]; then
    last="$(cat "$LLM_SESSION_FILE" 2>/dev/null || true)"
    if [ -n "$last" ]; then out=(--resume "$last" "${out[@]}"); have_session=1
    else echo "llm: no previous local session recorded; starting a new one" >&2; fi
  fi
  if [ "$have_session" = 0 ]; then last="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"; out=(--session-id "$last" "${out[@]}"); fi
  if [ -n "$last" ]; then mkdir -p "$(dirname "$LLM_SESSION_FILE")" && printf '%s\n' "$last" > "$LLM_SESSION_FILE"; fi
  printf '%s\0' "${out[@]}"
}

cmd_claude() {
  is_up || cmd_on || return 1   # don't point Claude at a dead endpoint if cmd_on refused (e.g. a GPU job is busy)
  local sargs=(); mapfile -d '' -t sargs < <(llm_session_args "$@"); set -- "${sargs[@]}"
  local ctx trim=() ctxenv=()
  ctx="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-$(served_ctx)}"
  [ -n "$ctx" ] && ctxenv=("CLAUDE_CODE_MAX_CONTEXT_TOKENS=$ctx")
  if [ "${LLM_CLAUDE_FULL:-0}" != 1 ]; then
    # claude_local_settings.json: the bio-research plugin off (trim) + pre-approved runbook commands and explicit denies
    # (no --no-verify / force-push / merge / AWS / sim/ webapp/ tools/gates/ edits), so a session following
    # research/coordination/LOCAL_LLM_RUNBOOK.md is not stalled by a permission prompt for `bash tools/status.sh`
    # (the 2026-09-25 live test was: every command was permission-blocked in -p mode).
    trim=(--strict-mcp-config --settings "$HERE/claude_local_settings.json"
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
  local was_up=0 rc=0; "$SYSTEMCTL" --user is-active --quiet "$UNIT" && was_up=1
  [ "$was_up" = 1 ] && cmd_off
  "$@" || rc=$?
  echo "== finished with exit code $rc at $(date '+%H:%M:%S')"
  [ "$was_up" = 1 ] && cmd_on
  return "$rc"
}

# cmd_yield [--max-hours N]: the driver's own GPU handoff (default 10h). Refuses (returns 1, changes nothing)
# under GAME_MODE (gaming is the owner's absolute priority, and the marker/session bookkeeping below would only
# race it), with an empty GPU queue (nothing to yield FOR), or with uncommitted tracked changes in any
# `draft-*` worktree (a yield can auto-resume hours later into a checkout the driver never finished; the
# runbook's own hard rule is "commit before you queue", so this just enforces it at the handoff itself). Only
# TRACKED changes count (--untracked-files=no): a scratch/log file mid-draft is not a reason to refuse.
cmd_yield() {
  local max_hours=10 notify_cmd=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --max-hours)
        max_hours="${2:-10}"
        # VALIDATE BEFORE ANY ARITHMETIC (2026-09-29 security review, H2). `$(( ... max_hours * 3600 ))` below
        # is bash ARITHMETIC expansion, which recursively re-expands a variable's VALUE as if it were itself an
        # arithmetic expression -- so `--max-hours 'a[$(cmd)]'` made bash evaluate the embedded command
        # substitution and run `cmd`, not merely fail to parse as a number. A plain digit-only check on the
        # value, BEFORE it ever reaches `$(( ))`, closes this off entirely: no arithmetic context ever sees
        # anything but a string already proven to be 1-2 digits.
        # MAX 12 (2026-09-29 re-review, M2): the digit-only check alone still let a container-triggered yield
        # (tools/agent_gateway.sh's `yield --max-hours N`) claim an arbitrarily long window (up to 99h) over
        # the shared GPU -- a resource-exhaustion concern distinct from the injection this check also closes.
        [[ "$max_hours" =~ ^[0-9]{1,2}$ ]] && [ "$max_hours" -ge 1 ] && [ "$max_hours" -le 12 ] || {
          echo "refusing to yield: --max-hours must be 1-12 (got: $max_hours)"; return 1
        }
        shift 2 ;;
      # --notify-cmd (2026-09-29, tools/agent_gateway.sh's `yield`, docs/plans/2026-09-29-local-model-
      # transition-PLAN.md section 9): the CONTAINERIZED agent has no workstation `llm claude` session to
      # resume, so its restore step is different -- run this shell command instead (its "$1" is filled in by
      # the restorer with the same "GPU window finished..." message the workstation flow would otherwise
      # resume a session with; see tools/gpu_queue.sh's llm_yield_restore_if_drained). Skips capturing
      # LLM_SESSION_FILE below: a container-triggered yield has nothing to do with the workstation's own last
      # local session, and restoring should not accidentally resume it.
      --notify-cmd) notify_cmd="${2:-}"; shift 2 ;;
      *) shift ;;
    esac
  done
  if [ -f "$GAME_MODE_FILE" ]; then
    echo "refusing to yield: GAME_MODE is set (owner gaming/away) -- see: bash tools/game.sh status"; return 1
  fi
  if [ ! -s "$QDIR/gpu.queue" ]; then
    echo "refusing to yield: the GPU queue is empty -- nothing queued to run in the window."; return 1
  fi
  local wt dirty=""
  local draft_root="${LLM_YIELD_DRAFT_ROOT:-$_LLM_SHARED_ROOT}"   # TEST-ONLY override; NEVER set in production
  for wt in "$draft_root"/.claude/worktrees/draft-*; do
    [ -d "$wt" ] || continue
    if [ -n "$(git -C "$wt" status --porcelain --untracked-files=no 2>/dev/null)" ]; then dirty="$dirty $wt"; fi
  done
  if [ -n "$dirty" ]; then
    echo "refusing to yield: uncommitted tracked changes in:$dirty -- commit (through the normal gates) before yielding."
    return 1
  fi
  is_up || { echo "refusing to yield: $UNIT is not loaded -- nothing to yield."; return 1; }
  local sess prof deadline
  if [ -n "$notify_cmd" ]; then
    sess=""   # see --notify-cmd's case comment above: no workstation session bookkeeping for this path
  else
    sess="$(cat "$LLM_SESSION_FILE" 2>/dev/null || true)"
  fi
  prof="$(current_profile)"
  deadline=$(( $(date +%s) + max_hours * 3600 ))
  {
    echo "LLM_YIELD_SESSION=$sess"
    echo "LLM_YIELD_PROFILE=$prof"
    echo "LLM_YIELD_DEADLINE=$deadline"
    [ -n "$notify_cmd" ] && echo "LLM_YIELD_NOTIFY_CMD=$notify_cmd"
  } > "$LLM_YIELD_MARKER"
  echo "yielding for up to ${max_hours}h (session=${sess:-<none>} profile=${prof:-<default>} deadline=$deadline); stopping $UNIT gracefully"
  cmd_stop_graceful 45
}

case "${1:-status}" in
  on) shift; cmd_on "$@" ;;
  off) cmd_off ;;
  status) cmd_status ;;
  claude) shift
    # `llm claude continue` / `llm claude resume` (a bare word, not the flag) means "continue", not a new session
    # whose first prompt is the word "continue" (seen 2026-09-26)
    if [ "$#" -eq 1 ] && { [ "$1" = continue ] || [ "$1" = resume ]; }; then set -- --continue; fi
    cmd_claude "$@" ;;
  resume|continue) shift; cmd_claude --continue "$@" ;;
  __session_args) shift; llm_session_args "$@" | tr '\0' ' '; echo ;;   # TEST-ONLY: show the rewritten args
  run) shift; cmd_run "$@" ;;
  yield) shift; cmd_yield "$@" ;;
  __stop_graceful) shift; cmd_stop_graceful "$@" ;;   # hidden: also used by tools/game.sh on --graceful
  __current_profile) current_profile ;;   # hidden diagnostic: which profile (if any) this unit is currently serving
  __profile_cmd) shift; profile_cmd "$1" ;;   # hidden: exposes profile_cmd() so llama-swap.yaml's authoring step
    # and tests/test_llama_swap_config.py can regenerate a profile's exact llama-server command line without
    # duplicating the flag-building logic above (LLM_PORT=@@PORT@@ ... __profile_cmd <name> is how both do it).
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
