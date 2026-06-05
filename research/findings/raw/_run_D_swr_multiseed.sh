#!/bin/bash
# D cue-recall arc multi-seed: train v16 seeds 43/44, then run the SWR-on-v16 de-risk (TRUE + permuted anti-cheat)
# on seeds 42/43/44. Pre-staged so it can run after the seed-42 primary de-risk shows GO. GPU.
set -e
cd "$(dirname "$0")/../../.."
V16="--n-train-events 200 --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 --weak-concept-dynamics \
--interleaved --topographic-factor 3.0 --off-target-factor 0.3 --enable-adjective --orthogonal-codes --sparsity 0.05"
for seed in 43 44; do
  if [ ! -f "bridges/v16/seed$seed.simstate.h5" ]; then
    echo "=== train v16 seed $seed ==="
    python -m research.runners.concept_pool_demo --seed $seed $V16 \
      --save-bridge bridges/v16/seed$seed.simstate.h5 --out bridges/v16/seed$seed.json
  fi
done
for seed in 42 43 44; do
  echo "=== seed $seed TRUE consolidation ==="
  python -m research.runners._D_swr_v16_derisk --load-bridge bridges/v16/seed$seed.simstate.h5 --seed $seed --swr-cycles 40
  echo "=== seed $seed PERMUTED anti-cheat ==="
  python -m research.runners._D_swr_v16_derisk --load-bridge bridges/v16/seed$seed.simstate.h5 --seed $seed --swr-cycles 40 --permute
done
echo "=== D SWR multi-seed complete ==="
