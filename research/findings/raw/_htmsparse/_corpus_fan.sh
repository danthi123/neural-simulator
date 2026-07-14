#!/usr/bin/env bash
# grow-to-context (corpus-structured sparse pool) vocab-scale de-risk — one OS proc per (n_subj, seed).
# corpus HTM should stay ~1.0 as vocab grows AND its synapse fraction (vs analytic dense) should DROP (linear-in-corpus).
set -u
OUTDIR=research/findings/raw/_htmsparse
mkdir -p "$OUTDIR"
pids=()
for N in 8 16 32; do
  for S in 42 43 44 100 101 102; do
    python -m research.runners._emerge15_sparse_pool_scale_derisk \
      --seeds "$S" --n-subj "$N" --epochs 80 --variant corpus --window 8 --no-dense \
      --out "$OUTDIR/corpus_n${N}_s${S}.json" > "$OUTDIR/corpus_n${N}_s${S}.log" 2>&1 &
    pids+=($!)
  done
done
echo "launched ${#pids[@]} corpus procs (n=8/16/32 x 6 seeds)"
wait
echo "ALL_CORPUS_DONE"
