#!/bin/bash
set -e
R="research/findings/raw"
RECIPE="--seed 42 --n-train-events 150 --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 --weak-concept-dynamics --interleaved --enable-adjective --orthogonal-codes --sparsity 0.03"
echo "### TRAIN baseline topo 3.0/0.3 ###"
python -m research.runners.concept_pool_demo_v2 $RECIPE --topographic-factor 3.0 --off-target-factor 0.3 --save-bridge $R/_gate2_base.simstate.h5 --out $R/_gate2_base.json 2>&1 | grep -iE "TRAIN|SAVE|PASS|accuracy" | tail -4
echo "### TRAIN strong topo 10.0/0.05 ###"
python -m research.runners.concept_pool_demo_v2 $RECIPE --topographic-factor 10.0 --off-target-factor 0.05 --save-bridge $R/_gate2_strong.simstate.h5 --out $R/_gate2_strong.json 2>&1 | grep -iE "TRAIN|SAVE|PASS|accuracy" | tail -4
echo "### CAPTURE + MEASURE baseline ###"
python -m research.findings.raw._capture_28concept_activity --m-samples 16 --seed 42 --ckpt $R/_gate2_base.simstate.h5 --out $R/_gate2_base.npz --sparsity 0.03 2>&1 | tail -1
python -m research.findings.raw._gate1_headtohead --npz $R/_gate2_base.npz 2>&1 | grep -E "clean|k=|SUMMARY"
echo "### CAPTURE + MEASURE strong ###"
python -m research.findings.raw._capture_28concept_activity --m-samples 16 --seed 42 --ckpt $R/_gate2_strong.simstate.h5 --out $R/_gate2_strong.npz --sparsity 0.03 2>&1 | tail -1
python -m research.findings.raw._gate1_headtohead --npz $R/_gate2_strong.npz 2>&1 | grep -E "clean|k=|SUMMARY"
echo "##### GATE2-DONE #####"
