#!/bin/bash
set -e
R="research/findings/raw"; G="$R/g11_bg"
VC=$(grep -vE '^#|^$' $G/g20_bridgeC_adj_vocab64.txt | paste -sd, -)
echo "### retrain bridgeC adj @ seed 44 (clean GPU) ###"
python -m research.runners.concept_pool_sparse_distributed --seed 44 --n-concepts 64 --n-train-events 400 \
  --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 --top-k 150 --sparsity 0.007 --vocab "$VC" \
  --save-bridge $R/_flatdist_bridgeC_seed44.simstate.h5 --out $R/_flatdist_bridgeC_seed44.json 2>&1 | grep -iE "SAVE" | tail -1
echo "### flat-distinct test ###"
rm -f $R/_flatdist_codes.npz
python -m research.findings.raw._insubstrate_flatdistinct_test 2>&1 | grep -E "captured|between-concept|seed |RESULT|VERDICT|CANNOT"
echo "##### FLATDIST-ALL-DONE #####"
