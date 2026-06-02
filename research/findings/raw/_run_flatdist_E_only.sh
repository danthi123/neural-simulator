#!/bin/bash
set -o pipefail
cd /e/Documents/Projects/sim
R="research/findings/raw"; G="$R/g11_bg"
C="--n-concepts 64 --n-train-events 400 --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 --top-k 150 --sparsity 0.007"
VE=$(grep -vE '^#|^$' $G/g20_bridgeE_functional_vocab64.txt | paste -sd, -)
echo "[$(date +%H:%M:%S)] retrain bridgeE functional @ seed 46 (fresh process, verified-healthy GPU)"
python -m research.runners.concept_pool_sparse_distributed --seed 46 $C --vocab "$VE" \
  --save-bridge "$R/_flatdist_bridgeE_seed46.simstate.h5" --out "$R/_flatdist_bridgeE_seed46.json"
rc=$?
echo "##### BRIDGEE-DONE [$(date +%H:%M:%S)] rc=$rc #####"
