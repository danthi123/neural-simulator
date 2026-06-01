#!/bin/bash
set -e
R="research/findings/raw"
for SEED in 43 44; do
  BASE="--seed $SEED --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 --weak-concept-dynamics --interleaved --enable-adjective --orthogonal-codes --sparsity 0.01 --topographic-factor 3.0 --off-target-factor 0.3 --n-train-events 300"
  echo "### TRAIN 64-word 300ev seed $SEED ###"
  python -m research.runners.concept_pool_demo_v3 $BASE --save-bridge $R/_scale64_ev300_seed${SEED}.simstate.h5 --out $R/_scale64_ev300_seed${SEED}.json 2>&1 | grep -iE "SAVE" | tail -1
  python -m research.findings.raw._capture_28concept_activity --m-samples 16 --seed $SEED --ckpt $R/_scale64_ev300_seed${SEED}.simstate.h5 --out $R/_scale64_ev300_seed${SEED}.npz --sparsity 0.01 --n-lang 2048 --vocab-mod research.runners.concept_pool_demo_v3 2>&1 | tail -1
  echo "--- seed $SEED ---"
  python -m research.findings.raw._gate1_headtohead --npz $R/_scale64_ev300_seed${SEED}.npz 2>&1 | grep -E "clean|k= 1|SUMMARY"
done
echo "##### SCALE64-MS-DONE #####"
