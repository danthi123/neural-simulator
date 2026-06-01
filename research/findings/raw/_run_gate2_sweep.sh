#!/bin/bash
set -e
R="research/findings/raw"
BASE="--seed 42 --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 --weak-concept-dynamics --interleaved --enable-adjective --orthogonal-codes --sparsity 0.03 --topographic-factor 3.0 --off-target-factor 0.3"
for EV in 50 300 500; do
  echo "### TRAIN 28-word topo3.0 ${EV} events ###"
  python -m research.runners.concept_pool_demo_v2 $BASE --n-train-events $EV --save-bridge $R/_gate2_ev${EV}.simstate.h5 --out $R/_gate2_ev${EV}.json 2>&1 | grep -iE "SAVE|interleaved event 1[0-9]00" | tail -2
  python -m research.findings.raw._capture_28concept_activity --m-samples 16 --seed 42 --ckpt $R/_gate2_ev${EV}.simstate.h5 --out $R/_gate2_ev${EV}.npz --sparsity 0.03 2>&1 | tail -1
  echo "--- EV=${EV} separability ---"
  python -m research.findings.raw._gate1_headtohead --npz $R/_gate2_ev${EV}.npz 2>&1 | grep -E "clean|k= 1|SUMMARY"
done
echo "##### SWEEP-DONE #####"
