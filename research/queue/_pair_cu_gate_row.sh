#!/bin/bash
# D3 gate row for the pair (Amendment 7 of research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md,
# `--family cu`, the reference 3090). PREPARED, NOT QUEUED. Run it only through tools/gpu_queue.sh (one GPU job at a
# time), from a CLEAN checkout pinned to the revision named below, e.g. (one queue line per seed):
#
#   tools/gpu_queue.sh add 'W=/home/dant123/Projects/sim/.claude/worktrees/_d3_cu_pin; \
#     [ -d "$W" ] || git -C /home/dant123/Projects/sim worktree add --detach "$W" <PIN_SHA>; bash "$W/research/queue/_pair_cu_gate_row.sh" <PIN_SHA> 42'
#
# What it does, and refuses to do:
#   * refuses unless HEAD is exactly <PIN_SHA> and the tracked tree is clean (no local edit reaches a gate row);
#   * links data/corpus from the primary checkout (git-excluded; without it the cross-edge build degrades silently);
#   * waits for RAM headroom (tools/mem_ok.sh), then runs ONE seed of the cu family under a hard cgroup cap
#     (tools/memcap.sh), SIM_BACKEND=cupy, one arm at a time (the runner forces --workers 1 for cu);
#   * writes research/findings/raw/_pair_production_path_cu/cu_seed<N>.json (+ per-arm files) inside that checkout.
set -euo pipefail
PIN="${1:?usage: _pair_cu_gate_row.sh <PIN_SHA> <seed>}"
SEED="${2:?usage: _pair_cu_gate_row.sh <PIN_SHA> <seed>}"
PRIMARY=/home/dant123/Projects/sim
cd "$(dirname "$0")/../.."
HEAD_SHA=$(git rev-parse HEAD)
case "$HEAD_SHA" in
  "$PIN"*) ;;
  *) echo "REFUSED: HEAD $HEAD_SHA is not the pinned revision $PIN" >&2; exit 2 ;;
esac
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "REFUSED: tracked tree is not clean at the pinned revision" >&2; exit 2
fi
mkdir -p data
[ -e data/corpus ] || ln -s "$PRIMARY/data/corpus" data/corpus
for f in tinystories.txt wikitext.txt simplewiki.txt websters1913.json; do
  [ -e "data/corpus/$f" ] || { echo "REFUSED: data/corpus/$f missing" >&2; exit 3; }
done
until bash tools/mem_ok.sh 6 4; do sleep 120; done
mkdir -p research/findings/raw/_pair_production_path_cu
exec bash tools/memcap.sh 6 -- env SIM_BACKEND=cupy "$PRIMARY/.venv/bin/python" -u \
  -m research.runners._pair_production_path_probe --family cu --seed "$SEED" --workers 1 \
  --out research/findings/raw/_pair_production_path_cu
