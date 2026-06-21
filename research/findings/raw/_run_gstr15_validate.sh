#!/usr/bin/env bash
# #5b R1 close — validate graded-strength=15 (the SOURCE fix that un-clamps seed-44's authoritative SNc
# δ): does it hold the δ 3/3 (seeds 42/43 must keep gabab_gap=True at the lower strength) AND collapse for
# the controls (seed-44 all-arms)? If so this is the clean 3/3 close.
set -u
cd "$(dirname "$0")/../../.." || exit 1
export SIM_BACKEND=cupy
RAW=research/findings/raw
ST="${ST:-15}"
for s in 42 43; do
  echo "=== GSTR$ST seed $s grid ==="
  python -m research.runners._n5_grid_frontend_onbridge_probe --seed "$s" --arm grid \
      --readout-only --multi-goal --value-train-trials 40 \
      --grid-drive-scale 2.5 --value-train-w-max 3 --graded-strength "$ST" \
      --out "$RAW/_n5_grid_onbridge_gstr${ST}_seed${s}.json" 2>&1 \
      | grep -E "GABA_B gap|DELTA RESULT|Traceback|Error" | tail -2
done
echo "=== GSTR$ST seed 44 ALL-ARMS (controls must collapse) ==="
python -m research.runners._n5_grid_frontend_onbridge_probe --seed 44 --all-arms \
    --readout-only --multi-goal --value-train-trials 40 \
    --grid-drive-scale 2.5 --value-train-w-max 3 --graded-strength "$ST" \
    --out "$RAW/_n5_grid_onbridge_gstr${ST}_allarms_seed44.json" 2>&1 \
    | grep -E "DELTA RESULT|=== |Traceback|Error" | tail -8
echo "=== DONE gstr$ST validate ==="
