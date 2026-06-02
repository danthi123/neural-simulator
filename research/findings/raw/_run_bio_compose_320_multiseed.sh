#!/bin/bash
# Pre-staged: multi-seed confirm of biological spiking composition at 64-concept (320-tier) scale.
# Launch ONLY if the seed-42 cheap-first (_bio_compose_320tier.log) RESOLVES (>=0.80 + abstention).
# --seed varies the composition RNG (roles/trials/capture-noise) on the SAME seed-42-trained bridge
# (load_checkpoint restores the trained CSR), so this is a genuine composition multi-seed.
set -e
R="research/findings/raw/g11_bg"
for SEED in 43 44; do
  echo "### bio-compose 320-tier (64 concepts) seed $SEED stim=300 ###"
  python -m research.findings.raw._insubstrate_real_substrate_qa_probe \
    --bridge $R/g20_sparse_bridges_320/bridgeA_nouns_sparse64.simstate.h5 \
    --vocab $R/g20_bridgeA_nouns_vocab64.txt \
    --seed $SEED --n-trials 20 --stim-steps 300 --sparsity 0.007 2>&1 \
    | grep -E "concepts|REAL-code|SYNTH-code|RESULT|VERDICT"
done
echo "##### BIO-COMPOSE-MS-DONE #####"
