#!/bin/bash
# Flat-distinct cheap-first: retrain bridgeB (verbs) @ seed 43 and bridgeC (adj) @ seed 44 so their sparse
# patterns DIFFER from bridgeA (seed 42). Then bridgeA(42)+bridgeB(43)+bridgeC(44) give DISTINCT FLAT codes
# -> structured SVO composition is a SINGLE-level bind (like the robust within-bridge 64), no nesting wall.
set -e
R="research/findings/raw"; G="$R/g11_bg"
COMMON="--n-concepts 64 --n-train-events 400 --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 --top-k 150 --sparsity 0.007"
VB=$(grep -vE '^#|^$' $G/g20_bridgeB_verbs_vocab64.txt | paste -sd, -)
VC=$(grep -vE '^#|^$' $G/g20_bridgeC_adj_vocab64.txt | paste -sd, -)
echo "### retrain bridgeB verbs @ seed 43 ###"
python -m research.runners.concept_pool_sparse_distributed --seed 43 $COMMON --vocab "$VB" \
    --save-bridge $R/_flatdist_bridgeB_seed43.simstate.h5 --out $R/_flatdist_bridgeB_seed43.json 2>&1 | grep -iE "SAVE|concepts" | tail -2
echo "### retrain bridgeC adj @ seed 44 ###"
python -m research.runners.concept_pool_sparse_distributed --seed 44 $COMMON --vocab "$VC" \
    --save-bridge $R/_flatdist_bridgeC_seed44.simstate.h5 --out $R/_flatdist_bridgeC_seed44.json 2>&1 | grep -iE "SAVE|concepts" | tail -2
echo "##### FLATDIST-RETRAIN-DONE #####"
