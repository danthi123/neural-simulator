#!/usr/bin/env bash
# #5b R1 close — critic-rate NORMALIZATION via homeostasis + the SETTLED-regime SNc-burst δ
# (delta_snc_graded). The cap-1.0 trade-off (fixes seed 44, breaks seed 42) is because the critic rate is
# seed-variable (~17Hz seed42 vs ~260Hz seed44) -> opposite g_gabab regimes. The principled fix is to
# NORMALIZE the critic rate (intrinsic-threshold homeostasis defending a target rate against the place
# volley) so a single regime works on every seed, and read δ in the settled count-plateau regime
# (delta_snc_graded, which is RPE-shaped and not contaminated like the raw graded-V). Prior doc Attempt 2:
# ema_alpha 0.02 + adapt_rate 0.01 settled the critic 50->30Hz during the value-train.
set -u
cd "$(dirname "$0")/../../.." || exit 1
export SIM_BACKEND=cupy
RAW=research/findings/raw
EMA="${EMA:-0.02}"; ADAPT="${ADAPT:-0.01}"; TGT="${TGT:-0.02}"
SUFFIX="${SUFFIX:-}"
for s in ${SEEDS:-42 43 44}; do
  echo "=== HOMEO seed $s ema=$EMA adapt=$ADAPT tgt=$TGT ==="
  python -m research.runners._n5_grid_frontend_onbridge_probe --seed "$s" --arm grid \
      --readout-only --multi-goal --value-train-trials 40 \
      --grid-drive-scale 2.5 --value-train-w-max 3 \
      --critic-homeo-ema-alpha "$EMA" --critic-homeo-adapt-rate "$ADAPT" --critic-homeo-target-rate "$TGT" \
      --out "$RAW/_n5_grid_onbridge_homeo${SUFFIX}_seed${s}.json" 2>&1 \
      | grep -E "GABA_B gap|DELTA RESULT|Traceback|Error|Killed" | tail -3
done
echo "=== DONE homeo (ema=$EMA adapt=$ADAPT) ==="
