#!/usr/bin/env bash
# #6 SURPASS — log-polar grid-32, the remaining 5 seeds (42 already GO). Sequential, GPU.
# Each seed runs host/sc_popvector(FIX1+log-polar)/sc_popvector_scr, then tags the per-arm JSONs.
set -u
export SIM_BACKEND=cupy
DIR="research/findings/raw/nav_gate_2a"
for s in 43 44 100 101 102; do
  echo "===== #6 log-polar seed $s ====="
  python -m research.runners._nav_sc_popvector_readout_derisk \
    --seed "$s" --grid-size 32 --n-steps 1800 --warmup-steps 600 \
    --fix1 --log-polar --arms host,sc_popvector,sc_popvector_scr \
    --out "$DIR/scpv_logpolar_summary_seed$s.json" > "$DIR/scpv_logpolar_seed$s.log" 2>&1
  for arm in host sc_popvector sc_popvector_scr; do
    if [ -f "$DIR/scpv_${arm}_seed$s.json" ]; then
      cp -f "$DIR/scpv_${arm}_seed$s.json" "$DIR/scpv_logpolar_${arm}_seed$s.json"
    fi
  done
  echo "===== seed $s DONE ====="
done
echo "ALL_REMAINING_SEEDS_DONE"
