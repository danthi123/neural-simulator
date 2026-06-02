#!/bin/bash
set -o pipefail
cd /e/Documents/Projects/sim
R="research/findings/raw"; G="$R/g11_bg"
C="--n-concepts 64 --n-train-events 400 --n-lang-input 8192 --n-shared-pool 2000 --pattern-size 100 --top-k 150 --sparsity 0.007"
for spec in "F 47" "G 48"; do
  set -- $spec; name=$1; seed=$2
  V=$(grep -vE '^#|^$' "$G/g20_bridge${name}_extra_vocab64.txt" | paste -sd, -)
  echo "[$(date +%H:%M:%S)] retrain bridge${name} @ seed ${seed} (->$(( ${seed} == 47 ? 384 : 448 )) concepts)"
  python -m research.runners.concept_pool_sparse_distributed --seed $seed $C --vocab "$V" \
    --save-bridge "$R/_flatdist_bridge${name}_seed${seed}.simstate.h5" --out "$R/_flatdist_bridge${name}_seed${seed}.json" \
    2>&1 | grep -iE "SAVE" | tail -1
done
echo "[$(date +%H:%M:%S)] running scaling composition test"
python -m research.findings.raw._insubstrate_flatdist_scaling_test 2>&1 \
  | grep -iE "bridges|between-concept|seed [0-9]+:|RESULT|VERDICT"
echo "##### SCALING-FG-DONE [$(date +%H:%M:%S)] #####"
