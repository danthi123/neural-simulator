#!/bin/bash
cd /e/Documents/Projects/sim
OP="--td-stdp-w-max 40 --td-to-fs-weight 30 --td-fs-to-strio-weight 20 --td-gabab-prop 0.04 --td-derivative-gain 2 --td-slow-tau-ms 250 --n-train 15"
echo "=== LESION ==="
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk --seed 42 --lesion $OP \
    --out research/findings/raw/_b4_cooled_lesion_s42.json
echo "=== UNPAIRED ==="
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk --seed 42 --unpaired $OP \
    --out research/findings/raw/_b4_cooled_unpaired_s42.json
echo "=== MOAT ==="
SIM_BACKEND=cupy python -u -m research.runners._merged_td_cueshift_consolidation_derisk --seed 42 --moat-only \
    --out research/findings/raw/_b4_cooled_moat_s42.json
echo "=== ANTICHEATS DONE ==="
