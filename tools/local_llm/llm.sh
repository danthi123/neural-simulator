#!/usr/bin/env bash
# llm -- run a local model for Claude Code, and get the GPU back for experiments with one command.
#
#   llm on [profile] [--wait]  load the model (default profile: tools/local_llm/default_profile). Refuses while
#                              a tools/gpu_queue.sh job is running/queued (same 3090 -- see AUTOSWAP below);
#                              --wait blocks until the queue drains instead of refusing.
#   llm off              unload it (frees its VRAM); also cancels any pending gpu_queue auto-restore (below)
#   llm status           what is loaded, and GPU memory in use
#   llm claude [args]    open Claude Code in the current directory, talking to the local model (no Anthropic account)
#   llm run <command>    unload the model, run <command> to completion, reload the model, print how it ended
#
# Profiles (model file, context size, sampling, speculative decoding) live in tools/local_llm/profiles.json and were
# chosen by tools/local_llm/bakeoff.py on this machine: 4 GB of the 24 GB card is left for the displays.
# Add `alias llm='bash ~/Projects/sim/tools/local_llm/llm.sh'` to your shell config to type just `llm`.
#
# AUTOSWAP (2026-09-25, see docs/LOCAL-MODEL-OFFLOAD.md; fix-round same day): this unit and tools/gpu_queue.sh
# jobs are NOT coordinated on their own and both want the one 3090, so gpu_queue.sh's dispatcher stops this unit
# itself -- DIRECTLY via systemctl, never by shelling out to this script's own `off` (that would delete the very
# marker the stop is about to write, see cmd_off's comment below) -- right before a queued job would contend
# with it for VRAM, and restarts it, with the SAME profile that was running, once the queue drains -- ONLY if it
# found this unit actually running (never fabricates an "it was on" memory) and only after a VERIFIED stop. The
# two scripts share research/queue/.local_llm_was_on as a one-shot marker whose CONTENTS are the profile name to
# restore: `llm off` always removes it (a manual/explicit off cancels the pending auto-restore, even mid-job),
# and gpu_queue consumes it right before its one restart attempt (so a crash or a failed restart is never
# retried forever). A refused `llm on` (GPU busy, no --wait) also WRITES this marker itself, recording the
# owner's intent to have it loaded, so the "it will load automatically once the queue drains" message it prints
# is actually true. Gaming's PAUSE sentinel (tools/game.sh) defers that restore rather than skipping it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LLM_PORT:-8080}"
UNIT="local-llm"
SYSTEMCTL=${LOCAL_LLM_SYSTEMCTL:-systemctl}   # TEST-ONLY override (shared name with gpu_queue.sh); NEVER set in production
# Resolve the SAME shared queue dir gpu_queue.sh uses (its singleton is per-repo, not per-worktree -- see that
# file's SHARED_ROOT comment), so a marker either script writes is visible to the other regardless of which
# worktree invoked them. GPU_QUEUE_DIR overrides it (tests only).
_GIT_COMMON=$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
if [ -n "$_GIT_COMMON" ] && [ -d "$(dirname "$_GIT_COMMON")" ]; then _LLM_SHARED_ROOT="$(dirname "$_GIT_COMMON")"; else _LLM_SHARED_ROOT="$HERE/../.."; fi
QDIR="${GPU_QUEUE_DIR:-$_LLM_SHARED_ROOT/research/queue}"
LLM_WAS_ON_MARKER="$QDIR/.local_llm_was_on"   # gpu_queue.sh's autoswap marker (its llm_stop_for_job/llm_restore_if_idle); CONTENTS = the profile to restore
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
  # systemd-run --description. Used by gpu_queue.sh's llm_stop_for_job to remember WHICH profile to restore
  # (LOW finding, 2026-09-25 fix-round: a bare marker always restored the DEFAULT profile, even when a
  # different one was actually running). Empty when not up / unparseable.
  local desc
  desc=$("$SYSTEMCTL" --user show -p Description --value "$UNIT" 2>/dev/null || true)
  case "$desc" in
    "local LLM ("*")") desc="${desc#local LLM (}"; echo "${desc%)}" ;;
  esac
}

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
  for _ in $(seq 1 300); do is_up && return 0; "$SYSTEMCTL" --user is-active --quiet "$UNIT" || break; sleep 1; done
  echo "the model did not come up; recent log:"; journalctl --user -u "$UNIT" -n 20 --no-pager; return 1
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
  local cmd; cmd="$(profile_cmd "$prof")"
  # shellcheck disable=SC2086
  systemd-run --user --quiet --unit="$UNIT" --description="local LLM ($prof)" -p MemoryMax=12G bash -c "exec $cmd"
  echo "loading $prof ..."; wait_up && echo "ready on http://127.0.0.1:$PORT ($prof)"
}

cmd_off() {
  if "$SYSTEMCTL" --user is-active --quiet "$UNIT"; then "$SYSTEMCTL" --user stop "$UNIT"; echo "unloaded"; else echo "not loaded"; fi
  "$SYSTEMCTL" --user reset-failed "$UNIT" 2>/dev/null || true
  # An explicit/manual off ALWAYS cancels gpu_queue's autoswap "restore it when the queue drains" intent -- even
  # when this unit was already down because gpu_queue itself stopped it for a job (the owner's `llm off` mid-job
  # must stick; the queue draining afterward must not silently reload it against that choice).
  rm -f "$LLM_WAS_ON_MARKER"
}

cmd_status() {
  if is_up; then echo "loaded: $("$SYSTEMCTL" --user show -p Description --value "$UNIT")"; else echo "not loaded"; fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/GPU memory in use: /'
}

cmd_claude() {
  is_up || cmd_on || return 1   # don't point Claude at a dead endpoint if cmd_on refused (e.g. a GPU job is busy)
  ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" ANTHROPIC_AUTH_TOKEN="local" ANTHROPIC_API_KEY="" \
  ANTHROPIC_MODEL="local" ANTHROPIC_SMALL_FAST_MODEL="local" ANTHROPIC_DEFAULT_HAIKU_MODEL="local" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="local" ANTHROPIC_DEFAULT_OPUS_MODEL="local" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 API_TIMEOUT_MS=1200000 \
    "$(command -v claude || ls -d "$HOME"/.config/Claude/claude-code/*/claude | sort -V | tail -1)" "$@"
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

case "${1:-status}" in
  on) shift; cmd_on "$@" ;;
  off) cmd_off ;;
  status) cmd_status ;;
  claude) shift; cmd_claude "$@" ;;
  run) shift; cmd_run "$@" ;;
  __current_profile) current_profile ;;   # hidden: gpu_queue.sh's llm_stop_for_job reads this to remember the profile
  *) sed -n '2,27p' "$0"; exit 2 ;;
esac
