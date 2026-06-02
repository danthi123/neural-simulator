#!/bin/bash
set -e
R="research/findings/raw"; G="$R/g11_bg"
C="--n-concepts 64 --n-train-events 400 --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 --top-k 150 --sparsity 0.007"
VD=$(grep -vE '^#|^$' $G/g20_bridgeD_spatial_vocab64.txt | paste -sd, -)
VE=$(grep -vE '^#|^$' $G/g20_bridgeE_functional_vocab64.txt | paste -sd, -)
echo "### retrain bridgeD spatial @ seed 45 ###"
python -m research.runners.concept_pool_sparse_distributed --seed 45 $C --vocab "$VD" --save-bridge $R/_flatdist_bridgeD_seed45.simstate.h5 --out $R/_flatdist_bridgeD_seed45.json 2>&1 | grep -iE "SAVE" | tail -1
echo "### retrain bridgeE functional @ seed 46 ###"
python -m research.runners.concept_pool_sparse_distributed --seed 46 $C --vocab "$VE" --save-bridge $R/_flatdist_bridgeE_seed46.simstate.h5 --out $R/_flatdist_bridgeE_seed46.json 2>&1 | grep -iE "SAVE" | tail -1
echo "##### FLATDIST-DE-DONE #####"
