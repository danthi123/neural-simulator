#!/usr/bin/env bash
# Batch 2 — the CONFOUND-FREE decisive sweep: window=1 (ADJACENT column-pairs only). No subject->branch shortcut;
# exactly the consecutive synapses the HTM winner-chain potentiates = the faithful grow-to-active-context pool.
# Claim: corpus-w1 HTM stays ~1.0 as vocab grows (n=8/16/32) AND its synapse fraction vs analytic dense DROPS.
set -u
OUTDIR=research/findings/raw/_htmsparse
mkdir -p "$OUTDIR"
pids=()
for N in 16 32; do                                   # n=8 window=1 already launched separately (bsfvt9fej)
  for S in 42 43 44 100 101 102; do
    python -m research.runners._emerge15_sparse_pool_scale_derisk \
      --seeds "$S" --n-subj "$N" --epochs 80 --variant corpus --window 1 --no-dense \
      --out "$OUTDIR/corpus_w1_n${N}_s${S}.json" > "$OUTDIR/corpus_w1_n${N}_s${S}.log" 2>&1 &
    pids+=($!)
  done
done
echo "launched ${#pids[@]} window=1 scaling procs (n=16/32 x 6 seeds)"
wait
echo "ALL_W1_SCALING_DONE"
