#!/usr/bin/env bash
# #5b R1 close — validate the WINNING GIRK-cap (critic_gabab_max=1.0): the authoritative SNc-burst δ
# must hold 3/3 (grid, seeds 42/43/44, gabab_gap True) AND collapse for all controls (seed-44 all-arms).
set -u
cd "$(dirname "$0")/../../.." || exit 1
export SIM_BACKEND=cupy
RAW=research/findings/raw
CAP="${CAP:-1.0}"
# (1) cap=1.0 grid arm on seeds 42/43 (44 already done in the sweep) — confirm no regression.
for s in 42 43; do
  echo "=== VALIDATE cap=$CAP seed $s grid ==="
  python -m research.runners._n5_grid_frontend_onbridge_probe --seed "$s" --arm grid \
      --readout-only --multi-goal --value-train-trials 40 \
      --grid-drive-scale 2.5 --value-train-w-max 3 --critic-gabab-max "$CAP" \
      --out "$RAW/_n5_grid_onbridge_girkcap${CAP}_seed${s}.json" 2>&1 \
      | grep -E "GABA_B gap|DELTA RESULT|Traceback|Error|Killed" | tail -3
done
# (2) cap=1.0 ALL-ARMS on seed 44 — confirm the controls collapse WITH the cap (authoritative snc_gap).
echo "=== VALIDATE cap=$CAP seed 44 ALL-ARMS (controls must collapse) ==="
python -m research.runners._n5_grid_frontend_onbridge_probe --seed 44 --all-arms \
    --readout-only --multi-goal --value-train-trials 40 \
    --grid-drive-scale 2.5 --value-train-w-max 3 --critic-gabab-max "$CAP" \
    --out "$RAW/_n5_grid_onbridge_girkcap${CAP}_allarms_seed44.json" 2>&1 \
    | grep -E "DELTA RESULT|=== |Traceback|Error|Killed" | tail -8
echo "=== DONE validate cap=$CAP ==="
