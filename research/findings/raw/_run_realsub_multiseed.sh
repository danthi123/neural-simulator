#!/bin/bash
ROOT=research/findings/raw/g11_bg
for s in 42 43 44; do
  echo "############## seed $s ##############"
  python -m research.findings.raw._insubstrate_real_substrate_qa_probe \
    --bridge $ROOT/g20_sparse_bridges/bridgeA_nouns_sparse.simstate.h5 \
    --vocab  $ROOT/g20_bridgeA_nouns_vocab.txt --seed $s --n-trials 20 2>&1 \
    | grep -E "REAL-code|SYNTH-code|RESULT|VERDICT|between-concept"
done
echo "##### MULTISEED DONE #####"
