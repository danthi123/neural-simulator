#!/usr/bin/env bash
# Batch 3b — MEASURED dense-parity (corpus-w1 == dense on the SAME corpus, not just the analytic count) + the n=64
# scaling point. Fire after the n=32 cells clear (dense at n=8/16 is feasible; n=64 corpus is heavy).
set -u
OUTDIR=research/findings/raw/_htmsparse
pids=()
launch() { python -m research.runners._emerge15_sparse_pool_scale_derisk "$@" & pids+=($!); }
# measured dense-parity (WITH dense; small n only): corpus-w1 HTM must == dense HTM
for S in 42 43; do
  launch --seeds "$S" --n-subj 8  --epochs 80 --variant corpus --window 1 \
     --out "$OUTDIR/corpus_w1_parity_n8_s${S}.json"  > "$OUTDIR/corpus_w1_parity_n8_s${S}.log"  2>&1
  launch --seeds "$S" --n-subj 16 --epochs 80 --variant corpus --window 1 \
     --out "$OUTDIR/corpus_w1_parity_n16_s${S}.json" > "$OUTDIR/corpus_w1_parity_n16_s${S}.log" 2>&1
done
# n=64 scaling point (corpus-only; heavier)
for S in 42 43 44 100 101 102; do
  launch --seeds "$S" --n-subj 64 --epochs 80 --variant corpus --window 1 --no-dense \
     --out "$OUTDIR/corpus_w1_n64_s${S}.json" > "$OUTDIR/corpus_w1_n64_s${S}.log" 2>&1
done
echo "launched ${#pids[@]} batch-3b procs (w1 measured-parity + n=64)"
wait
echo "ALL_W1_BATCH3B_DONE"
