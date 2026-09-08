#!/usr/bin/env bash
# openhands.sh — THE single command for the owner's full OpenHands takeover (2026-09-08).
#
#   bash tools/openhands.sh
#
# Does everything `tools/openhands_takeover.sh on` does (guard against HERMES_ACTIVE, set
# OPENHANDS_ACTIVE, start the shared VRAM supervisor, launch openhands_loop.py), THEN drops the owner
# straight into an interactive REPL: type a task, it runs one turn against the ONE persisted OpenHands
# conversation via the existing run_turn.py and prints its scrollable transcript, then prompts again.
#
# Type /stop or /quit to end the session -- Ctrl-C and Ctrl-D also work. HOWEVER it ends, a `trap` on
# EXIT/INT/TERM runs the exact `tools/openhands_takeover.sh off` teardown (stop the loop, unload Qwen,
# clear OPENHANDS_ACTIVE) before this script exits, so the GPU/driver seat always comes back to Claude
# -- no stranded Qwen holding VRAM, no stale sentinel, even on a crash. (The one exception no process
# can trap: `kill -9` / SIGKILL bypasses all cleanup everywhere, not just here.)
#
# THIS IS A THIN WRAPPER, not a reimplementation: the actual on/off primitive stays
# tools/openhands_takeover.sh (still fine to script directly -- see docs/OPENHANDS_TAKEOVER.md
# "Advanced: the scriptable primitive"), and the actual agent turn stays
# tools/openhands_proto/run_turn.py. This file only adds the interactive loop + the always-runs
# teardown on top of both.
#
# ⛔ Run from the canonical checkout (/home/dant123/Projects/sim), not a worktree -- every script here
# (this one included) resolves its state from ITS OWN path (BASH_SOURCE), not git-common-dir, so a
# worktree copy would read/write a decoupled, worktree-local research/queue/ snapshot instead of the
# real shared driver state (same pre-existing limitation as openhands_takeover.sh and
# hermes_takeover.sh -- see docs/OPENHANDS_TAKEOVER.md).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/research/queue"
PROTO="$ROOT/tools/openhands_proto"
VENV_PY="${OPENHANDS_SH_VENV_PY:-$PROTO/.venv/bin/python}"
# Override seams (test-only -- never set these by hand for real use). Default is always the real
# scriptable primitive / the real run_turn.py; a test substitutes a stand-in to dry-run the REPL and
# trap/teardown control flow without ever touching qwen_serve.sh or the GPU.
TAKEOVER="${OPENHANDS_SH_TAKEOVER:-$ROOT/tools/openhands_takeover.sh}"
LOOP_PIDF="$STATE/openhands_loop.pid"
REPL_LOCK="$STATE/openhands_repl.pid"

# --- refuse a second concurrent REPL front-end -----------------------------------------------------
# The takeover itself (loop/supervisor/Qwen) is already idempotent (tools/openhands_takeover.sh on
# no-ops if already running -- see below), so THIS script always just calls it and attaches. But two
# *interactive* openhands.sh sessions both calling run_turn.py at once would race on the one persisted
# conversation, so that specific case is refused with a clear message instead.
if [ -f "$REPL_LOCK" ] && kill -0 "$(cat "$REPL_LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[openhands] refusing: another 'openhands.sh' session is already attached (pid $(cat "$REPL_LOCK"))."
  echo "[openhands] use that terminal, or 'bash tools/openhands_takeover.sh status' to check the takeover."
  echo "[openhands] if that pid is actually gone: rm -f $REPL_LOCK"
  exit 1
fi
echo $$ > "$REPL_LOCK"

# --- teardown always runs exactly once, on every exit path -----------------------------------------
TEARDOWN_DONE=0
teardown() {
  [ "$TEARDOWN_DONE" = 1 ] && return
  TEARDOWN_DONE=1
  echo
  echo "[openhands] tearing down -> handing the GPU/driver seat back to Claude..."
  bash "$TAKEOVER" off
  rm -f "$REPL_LOCK"
}
trap teardown EXIT
trap 'exit 130' INT   # Ctrl-C -> exit -> the EXIT trap above fires teardown
trap 'exit 143' TERM  # kill -TERM -> same

# --- 1. start the full takeover (idempotent: no-ops / attaches if already active) ------------------
echo "[openhands] starting the full takeover (tools/openhands_takeover.sh on)..."
if ! bash "$TAKEOVER" on; then
  echo "[openhands] takeover failed to start (see above) -- not entering the session."
  exit 1
fi

# --- 2. interactive REPL on the one persisted conversation ------------------------------------------
echo
echo "[openhands] type a task, or /stop to end and hand back to Claude (Ctrl-C / Ctrl-D also work)."
echo

while true; do
  if ! read -r -p "openhands> " task; then
    echo
    echo "[openhands] EOF (Ctrl-D) -> stopping."
    break
  fi
  [ -z "$task" ] && continue
  case "$task" in
    /stop | /quit | /exit)
      break
      ;;
  esac
  "$VENV_PY" "$PROTO/run_turn.py" --prompt "$task"
  echo
done
# falling off the end here runs the EXIT trap -> teardown -> bash "$TAKEOVER" off
