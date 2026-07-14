#!/usr/bin/env bash
# Batch 3 — window=1 anti-cheats (the decisive controls) + n=64 scaling + dense-parity anchor.
#   - dAP-LESION (n=8, w1): should COLLAPSE toward the n-gram floor (the dendritic apical prediction is load-bearing).
#   - PERMUTED corpus (n=8, w1): should COLLAPSE (the pool + winner-chain encode the true word order).
#   - n=64 corpus w1 --no-dense: does HTM stay ~1.0 + synapse fraction keep dropping at 8x the smoke vocab?
#   - n=8 + n=16 dense+corpus w1 (parity): corpus-w1 == the dense pool (measured, not just analytic).
set -u
OUTDIR=research/findings/raw/_htmsparse
mkdir -p "$OUTDIR"
pids=()
launch() { python -m research.runners._emerge15_sparse_pool_scale_derisk "$@" & pids+=($!); }
for S in 42 43 44 100 101 102; do
  launch --seeds "$S" --n-subj 8  --epochs 80 --variant corpus --window 1 --no-dense --lesion \
     --out "$OUTDIR/corpus_w1_lesion_n8_s${S}.json"  > "$OUTDIR/corpus_w1_lesion_n8_s${S}.log"  2>&1
  launch --seeds "$S" --n-subj 8  --epochs 80 --variant corpus --window 1 --no-dense --permute \
     --out "$OUTDIR/corpus_w1_permute_n8_s${S}.json" > "$OUTDIR/corpus_w1_permute_n8_s${S}.log" 2>&1
  launch --seeds "$S" --n-subj 64 --epochs 80 --variant corpus --window 1 --no-dense \
     --out "$OUTDIR/corpus_w1_n64_s${S}.json"        > "$OUTDIR/corpus_w1_n64_s${S}.log"        2>&1
done
# measured dense parity (small n only; dense OOMs at large vocab): corpus-w1 == dense on the SAME corpus
launch --seeds 42 --n-subj 8  --epochs 80 --variant corpus --window 1 \
   --out "$OUTDIR/corpus_w1_parity_n8_s42.json"  > "$OUTDIR/corpus_w1_parity_n8_s42.log"  2>&1
launch --seeds 43 --n-subj 16 --epochs 80 --variant corpus --window 1 \
   --out "$OUTDIR/corpus_w1_parity_n16_s43.json" > "$OUTDIR/corpus_w1_parity_n16_s43.log" 2>&1
echo "launched ${#pids[@]} batch-3 procs (w1 lesion+permute+n64+parity)"
wait
echo "ALL_W1_BATCH3_DONE"
