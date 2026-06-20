#!/bin/bash
# Grid-32 divnorm re-calibration of the NEURAL pop-vector arm — the step the grid-8 calibration
# could not settle. Tests whether ANY faithful-scale operating point lets the geometry-correct
# read-out track (stronger SC drive via lower gain / higher SC_CORTEX_W). 2 focused points.
set -e
cd E:/Documents/Projects/sim
OUT=research/findings/raw/nav_gate_2a/grid32_s6
# Point A: much lower divnorm gain (less attenuation -> stronger SC margin) at matched drive 18
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
  --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
  --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.005 \
  --out $OUT/scpv_summary_RECAL_g0p005_s42.json
cp $OUT/scpv_sc_popvector_seed42.json $OUT/scpv_RECAL_g0p005_s42.json
echo "RECAL gain0.005 DONE"
