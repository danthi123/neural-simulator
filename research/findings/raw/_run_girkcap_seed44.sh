#!/usr/bin/env bash
# #5b R1 close — the GIRK-saturation cap (critic_gabab_max) to rescue the seed-44 SNc-burst δ over-clamp.
# The authoritative SNc-burst δ (the runner's stage-B snc_gap) cleanly discriminates grid from all
# controls (grid 6.67 vs scramble/render/no_learn/lesion ~1.0/0.0) on seeds 42/43; only seed-44 over-fires
# the critic -> g_gabab over-accumulates -> SNc fully clamps at BOTH near+far -> snc_gap inverts to 0.0.
# The cap bounds g_gabab (finite GIRK channels) so a hot critic cannot fully clamp the SNc -> graded δ at
# any rate. Sweep the cap on seed 44; the winner must hold the δ>=1.3 AND not regress seeds 42/43.
set -u
cd "$(dirname "$0")/../../.." || exit 1
export SIM_BACKEND=cupy
RAW=research/findings/raw
SEED="${SEED:-44}"
for cap in ${CAPS:-0.5 1.0 2.0 4.0}; do
  echo "=== seed $SEED critic_gabab_max=$cap ==="
  python -m research.runners._n5_grid_frontend_onbridge_probe --seed "$SEED" --arm grid \
      --readout-only --multi-goal --value-train-trials 40 \
      --grid-drive-scale 2.5 --value-train-w-max 3 --critic-gabab-max "$cap" \
      --out "$RAW/_n5_grid_onbridge_girkcap${cap}_seed${SEED}.json" 2>&1 \
      | grep -E "GABA_B gap|DELTA RESULT|Traceback|Error|Killed" | tail -3
done
echo "=== DONE girk-cap sweep seed $SEED ==="
