#!/bin/bash
# B4 cooled op-point: the FULL battery at the winning config (3-seed migration + lesion + unpaired + moat).
# Fill OPFLAGS + NT with the winning cooled op-point, then run. Each block commits separately (anti-rest).
cd /e/Documents/Projects/sim
OPFLAGS="${OPFLAGS:---td-stdp-w-max 40 --td-to-fs-weight 30 --td-fs-to-strio-weight 20 --td-gabab-prop 0.04 --td-derivative-gain 2 --td-slow-tau-ms 250}"
NT="${NT:-30}"
EXTRA="${EXTRA:-}"   # e.g. --td-csc-to-strio-weight 10  OR  --reward-learning-rate 0.005

# 1) 3-seed migration (the headline)
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk \
    --seeds 42,43,44 $OPFLAGS $EXTRA --n-train $NT --no-gates \
    --out research/findings/raw/_b4_cooled_migration_3seed.json

# 2) cue-pathway LESION anti-cheat (seed 42)
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk \
    --seed 42 --lesion $OPFLAGS $EXTRA --n-train $NT \
    --out research/findings/raw/_b4_cooled_lesion_s42.json

# 3) UNPAIRED anti-cheat (seed 42)
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk \
    --seed 42 --unpaired $OPFLAGS $EXTRA --n-train $NT \
    --out research/findings/raw/_b4_cooled_unpaired_s42.json

# 4) MOAT re-verify (op-point-independent; builder defaults)
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk \
    --seed 42 --moat-only \
    --out research/findings/raw/_b4_cooled_moat_s42.json
