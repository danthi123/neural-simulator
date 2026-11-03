#!/usr/bin/env bash
# brain_gpu.sh — queue a brain/ GPU job (a night, or the weekly test) the house way: through tools/gpu_queue.sh
# (one GPU job at a time), with the local LLM yielded for the job's length (tools/local_llm/llm.sh run: unload, run,
# reload) and the job's host RAM capped (tools/memcap.sh). brain/research/2026-11-03-adopt-1.5b-plan.md step 7 / 11.
#
#   bash tools/brain_gpu.sh night  --home DIR [--model 0.1b|0.4b|1.5b|PATH] [--dry-run] [extra brain.night args]
#   bash tools/brain_gpu.sh weekly [--model ...] [--dry-run] [extra brain.eval.weekly args]
#
# --dry-run prints the queued line and queues nothing. Env: BRAIN_GPU_MEM_GB (host RAM cap, default 16).
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
line="cd $(printf '%q' "$ROOT") && bash tools/local_llm/llm.sh run bash tools/memcap.sh $MEM_GB -- $(printf '%q' "$PY") -u -m $module --device cuda$q"
if [ "$dry" = 1 ]; then echo "$line"; exit 0; fi
exec bash "$ROOT/tools/gpu_queue.sh" add "$line"
