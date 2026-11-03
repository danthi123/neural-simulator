#!/usr/bin/env bash
# brain_gpu.sh — queue a brain/ GPU job (a night, or the weekly test) the house way: through tools/gpu_queue.sh
# (one GPU job at a time), then hand the card over with `tools/local_llm/llm.sh yield` (the queue starts no job while
# the local LLM is up, and restores it once the yield window drains), with the job's host RAM capped
# (tools/memcap.sh). brain/research/2026-11-03-adopt-1.5b-plan.md step 7 / 11.
# 11-03: the job used to wrap itself in `llm.sh run` (unload inside the job) -- the queue never started it while the
# LLM was up, so it waited forever. Yield refuses under GAME_MODE; the job then waits for the game to end.
#
#   bash tools/brain_gpu.sh night  --home DIR [--model 0.1b|0.4b|1.5b|PATH] [--dry-run] [extra brain.night args]
#   bash tools/brain_gpu.sh weekly [--model ...] [--dry-run] [extra brain.eval.weekly args]
#
# --dry-run prints the queued line and queues nothing. Env: BRAIN_GPU_MEM_GB (host RAM cap, default 16),
# BRAIN_GPU_YIELD_H (yield window in hours, default 4); BRAIN_GPU_QUEUE / BRAIN_GPU_LLM override the two tools (tests).
# Nothing here installs a timer; a nightly timer is an ops decision (see the plan's checklist).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_GB="${BRAIN_GPU_MEM_GB:-16}"
# the venv lives in the main checkout; a git worktree has none of its own
GIT_COMMON=$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || [ -z "$GIT_COMMON" ] || PY="$(dirname "$GIT_COMMON")/.venv/bin/python"
what="${1:-}"; shift || true
case "$what" in
  night) module=brain.night ;;
  weekly) module=brain.eval.weekly ;;
  *) echo "usage: brain_gpu.sh night --home DIR [--model M] [--dry-run] | weekly [--model M] [--dry-run]" >&2; exit 2 ;;
esac
dry=0; args=()
for x in "$@"; do
  if [ "$x" = --dry-run ]; then dry=1; else args+=("$x"); fi
done
if [ "$what" = night ] && ! printf '%s\n' "${args[@]}" | grep -qx -- --home; then
  echo "brain_gpu.sh night: --home DIR is required" >&2; exit 2
fi
q=""
for x in "${args[@]}"; do q+=" $(printf '%q' "$x")"; done
line="cd $(printf '%q' "$ROOT") && bash tools/memcap.sh $MEM_GB -- $(printf '%q' "$PY") -u -m $module --device cuda$q"
if [ "$dry" = 1 ]; then echo "$line"; exit 0; fi
bash "${BRAIN_GPU_QUEUE:-$ROOT/tools/gpu_queue.sh}" add "$line"
# after the add: yield refuses with an empty queue. A refusal (game mode, an LLM already down) leaves the job queued.
bash "${BRAIN_GPU_LLM:-$ROOT/tools/local_llm/llm.sh}" yield --max-hours "${BRAIN_GPU_YIELD_H:-4}" \
  || echo "brain_gpu.sh: llm yield refused; the job waits in the queue until the GPU is free" >&2
