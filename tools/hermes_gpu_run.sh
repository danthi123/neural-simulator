#!/usr/bin/env bash
# hermes_gpu_run.sh — the ONE way Hermes (local-Qwen dev agent) should launch a LOCAL GPU job.
#
# WHY a wrapper: Hermes' own brain (Qwen) sits in GPU VRAM. A local GPU experiment needs that VRAM, so the
# supervisor (tools/qwen_supervisor.sh) must UNLOAD Qwen before the job runs and RELOAD it after — at which point
# it nudges Hermes to come back and read the results. So the correct pattern for Hermes is:
#     1. call this to ENQUEUE the job, 2. END your turn. Your model will unload, the job will run on the full GPU,
#     then your model reloads and you are automatically re-invoked to harvest the result.
# Do NOT run GPU python directly (SIM_BACKEND=cupy ...) — that would fight your own model for VRAM. CPU/pool work
# does NOT need this (it never contends with the local GPU): use tools/sweep_pool.sh / gpu is not involved.
#
#   bash tools/hermes_gpu_run.sh "SIM_BACKEND=cupy .venv/bin/python -m research.runners.X --seed 42 --out ..."
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-}"
[ -z "$CMD" ] && { echo "usage: bash tools/hermes_gpu_run.sh '<full GPU command incl. --out>'"; exit 1; }
# enqueue via the shared, VRAM-contention-safe local GPU queue (the supervisor unloads Qwen so this gets the whole card)
bash "$ROOT/tools/gpu_queue.sh" add "$CMD"
echo "[hermes-gpu] queued. NOW END YOUR TURN: your model (Qwen) will unload, the job runs on the full GPU, then"
echo "[hermes-gpu] your model reloads and you're re-invoked to harvest the result (research/coordination/live_state.md"
echo "[hermes-gpu] will tell you what completed). Do not busy-wait; the supervisor handles reload + re-invoke."
