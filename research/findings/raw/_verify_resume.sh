#!/bin/bash
set -e
R="research/findings/raw"
C="--n-concepts 16 --n-lang-input 2048 --n-shared-pool 2000 --pattern-size 50 --top-k 30 --sparsity 0.05 --seed 42"
echo "### A: from-scratch 100ev ###"
python -m research.runners.concept_pool_sparse_distributed $C --n-train-events 100 --save-bridge $R/_resume_A.h5 2>&1 | grep -E "RESULTS|resume|topographic" | tail -2
echo "### B: RESUME from A, +100ev (total 200 incremental) ###"
python -m research.runners.concept_pool_sparse_distributed $C --n-train-events 100 --resume-from $R/_resume_A.h5 --save-bridge $R/_resume_B.h5 2>&1 | grep -E "RESULTS|resume|loaded" | tail -2
echo "### REF: from-scratch 200ev (one go) ###"
python -m research.runners.concept_pool_sparse_distributed $C --n-train-events 200 2>&1 | grep -E "RESULTS|topographic" | tail -1
echo "##### VERIFY-RESUME-DONE #####"
