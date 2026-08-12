#!/bin/bash
# Scaled sweep to LOCATE the deep-context credit wall: push content-vocab K (feedback-alignment stress,
# Bartunov-2018: random FB fails at large output dim) at a moderate lag T=32 (BPTT ceiling still valid),
# plus a K=64/T=64 horizon point. Find the (K,T) where eprop_random DROPS below truefb/bptt AND
# eprop_learnfb opens a gap over eprop_random => the learned-feedback companion becomes load-bearing.
set -u
PY=/home/dant123/Projects/sim/.venv/bin/python
ROOT=/home/dant123/Projects/sim/.claude/worktrees/agent-a40606d7b8b094fbc
cd "$ROOT" || exit 1
export SIM_BACKEND=cupy
run () {
  local K=$1 F=$2 T=$3 tag=$4
  echo "===== SCALE RUN tag=$tag K=$K F=$F T=$T ====="
  $PY -m research.runners._spiking_deepcontext_generation_derisk \
    --seeds 42 43 44 --T $T --K $K --F $F --N 192 --H 96 --D 64 \
    --epochs 60 --n-train 6000 --n-eval 1500 \
    --out "$ROOT/research/findings/raw/_dc_scale_${tag}.json"
}
run 16  16 32 K16_T32
run 32  24 32 K32_T32
run 64  40 32 K64_T32
run 128 60 32 K128_T32
run 64  40 64 K64_T64
echo "===== SCALE SWEEP DONE ====="
