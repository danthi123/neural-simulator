#!/usr/bin/env bash
# EMERGENT online grow-to-active-context — the decisive 6-seed GO + anti-cheats (n=8) + vocab-scaling (n=16/32).
# GO: online-grown HTM ~1.0 at a cell-level sub-quadratic grown-synapse count, discovered from winner dynamics.
# LESION should collapse; PERMUTE should collapse AND change the grown set (structure from dynamics, not token pre-scan).
set -u
OUTDIR=research/findings/raw/_htmsparse
mkdir -p "$OUTDIR"
pids=()
launch() { python -m research.runners._emerge15_online_grow_derisk "$@" & pids+=($!); }
for S in 42 43 44 100 101 102; do
  launch --seeds "$S" --n-subj 8  --epochs 80 --out "$OUTDIR/online_n8_s${S}.json"          > "$OUTDIR/online_n8_s${S}.log"          2>&1
  launch --seeds "$S" --n-subj 8  --epochs 80 --lesion  --out "$OUTDIR/online_lesion_n8_s${S}.json"  > "$OUTDIR/online_lesion_n8_s${S}.log"  2>&1
  launch --seeds "$S" --n-subj 8  --epochs 80 --permute --out "$OUTDIR/online_permute_n8_s${S}.json" > "$OUTDIR/online_permute_n8_s${S}.log" 2>&1
done
echo "launched ${#pids[@]} online-grow procs (n8 main+lesion+permute x6)"
wait
echo "ALL_ONLINE_N8_DONE"
