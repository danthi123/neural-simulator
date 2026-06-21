#!/bin/bash
# B4 merged-bridge A-CSC TD cue-shift migration, 3 seeds, best op-point (opsearch) + homeofix lesion.
cd /e/Documents/Projects/sim
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk \
    --seeds 42,43,44 \
    --td-stdp-w-max 60 --td-to-fs-weight 30 --td-fs-to-strio-weight 20 \
    --td-gabab-prop 0.04 --td-derivative-gain 2 --td-slow-tau-ms 250 --n-train 30 \
    --out research/findings/raw/_merged_td_cueshift_migration_3seed.json
