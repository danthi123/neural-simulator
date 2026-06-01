#!/bin/bash
ROOT=research/findings/raw/g11_bg
declare -A B=( [B_verbs]=bridgeB_verbs [C_adj]=bridgeC_adj [D_spatial]=bridgeD_spatial [E_functional]=bridgeE_functional )
declare -A V=( [B_verbs]=g20_bridgeB_verbs_vocab [C_adj]=g20_bridgeC_adj_vocab [D_spatial]=g20_bridgeD_spatial_vocab [E_functional]=g20_bridgeE_functional_vocab )
for k in B_verbs C_adj D_spatial E_functional; do
  echo "############## $k ##############"
  python -m research.findings.raw._insubstrate_real_substrate_qa_probe \
    --bridge $ROOT/g20_sparse_bridges/${B[$k]}_sparse.simstate.h5 \
    --vocab  $ROOT/${V[$k]}.txt --seed 42 2>&1 | grep -E "REAL-code|SYNTH-code|RESULT|VERDICT|between-concept"
done
echo "##### ALL 4 DONE #####"
