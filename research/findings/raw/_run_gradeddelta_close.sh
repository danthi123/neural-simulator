#!/usr/bin/env bash
# #5b R1 close — graded-V-only δ (move a) + optional settle (move b), 3-seed (then 6).
# Each run: grid arm + the full control battery (render/scramble/no_learn/lesion) under --all-arms so
# the controls collapse on the graded-V δ too. GPU (SIM_BACKEND=cupy).
set -u
cd "$(dirname "$0")/../../.." || exit 1
export SIM_BACKEND=cupy
RAW=research/findings/raw
SETTLE="${SETTLE:-0}"      # 0 = move (a) only; >0 adds the move (b) settling window
SEEDS="${SEEDS:-42 43 44}"
SUFFIX="${SUFFIX:-}"       # e.g. _settle80 for the move-(b) JSONs
for s in $SEEDS; do
  echo "=== seed $s (settle=$SETTLE) ==="
  python -m research.runners._n5_grid_frontend_onbridge_probe --seed "$s" --all-arms \
      --readout-only --multi-goal --value-train-trials 40 \
      --grid-drive-scale 2.5 --value-train-w-max 3 --settle-steps "$SETTLE" \
      --out "$RAW/_n5_grid_onbridge_gradeddelta_allarms_seed${s}${SUFFIX}.json" 2>&1 \
      | tail -8
done
echo "=== DONE ($SEEDS) ==="
