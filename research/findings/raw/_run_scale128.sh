#!/bin/bash
set -e
R="research/findings/raw"
BASE="--seed 42 --n-lang-input 4096 --n-per-pool 200 --n-fs-per-pool 24 --weak-concept-dynamics --interleaved --enable-adjective --orthogonal-codes --sparsity 0.005 --topographic-factor 3.0 --off-target-factor 0.3 --n-train-events 300"
echo "### TRAIN 128-word 300ev (4096 lang) ###"
python -m research.runners.concept_pool_demo_v4 $BASE --save-bridge $R/_scale128_ev300.simstate.h5 --out $R/_scale128_ev300.json 2>&1 | grep -iE "SAVE|capacity test|Nouns" | tail -2
echo "### CAPTURE + MEASURE 128-word ###"
python -m research.findings.raw._capture_28concept_activity --m-samples 16 --seed 42 --ckpt $R/_scale128_ev300.simstate.h5 --out $R/_scale128_ev300.npz --sparsity 0.005 --n-lang 4096 --vocab-mod research.runners.concept_pool_demo_v4 2>&1 | tail -1
python -m research.findings.raw._gate1_headtohead --npz $R/_scale128_ev300.npz 2>&1 | grep -E "clean|k= 1|SUMMARY"
echo "##### SCALE128-DONE #####"
