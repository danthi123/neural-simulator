#!/bin/bash
set -e
cd /e/Documents/Projects/sim
FLAGS="--moving-goal --goal-schedule multi --deterministic --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi --enable-cluster-a-closed-loop --enable-cluster-e-topography --enable-dlpfc-wm --enable-pfc-nmda --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --grid-size 32 --n-steps 1800 --spiking-snc --enable-neural-critic"
for s in 42 43; do
  echo "=== NEURAL-CRITIC SMOKE seed $s START $(date) ==="
  python -m research.runners.g11_bg_runner $FLAGS --seed $s \
    --out research/findings/raw/g11_bg/_neuralcritic_gabab_neural_s${s}.json \
    > research/findings/raw/g11_bg/_neuralcritic_smoke_s${s}.log 2>&1
  echo "=== seed $s DONE $(date) rc=$? ==="
done
echo "=== ALL SMOKE SEEDS DONE $(date) ==="
