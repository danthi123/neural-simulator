#!/usr/bin/env bash
# gap#5 RANK3 gamma-organized recombination — 6-seed {42 43 44 100 101 102}, parallel local numpy (CPU, deterministic).
set -u
cd /home/dant123/Projects/sim
OUT=research/findings/raw/gap5_r4
pids=()
for s in 42 43 44 100 101 102; do
  SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._gap5_gamma_recombination_derisk \
    --seeds "$s" --n-trials 400 --out "$OUT/gamma_recomb_s${s}.json" \
    > "$OUT/gamma_recomb_s${s}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} seeds: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "all done, failures=$fail"
# collect verdicts
for s in 42 43 44 100 101 102; do
  echo "--- seed $s ---"
  grep -E "MAIN recomb_frac" "$OUT/gamma_recomb_s${s}.log" 2>/dev/null | tail -1
done
