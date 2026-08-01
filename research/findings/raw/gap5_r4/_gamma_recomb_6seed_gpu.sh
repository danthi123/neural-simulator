#!/usr/bin/env bash
# gap#5 RANK3 gamma-organized recombination — 6-seed {42 43 44 100 101 102} on GPU, max 3 concurrent (VRAM ~5.4GB each).
# Deterministic reductions via CUBLAS_WORKSPACE_CONFIG. Default config (chain_fwd=24) = the operating point under test.
set -u
cd /home/dant123/Projects/sim
OUT=research/findings/raw/gap5_r4
run_one() {
  local s=$1
  CUBLAS_WORKSPACE_CONFIG=:4096:8 SIM_BACKEND=cupy \
    .venv/bin/python -u -m research.runners._gap5_gamma_recombination_derisk \
    --seeds "$s" --n-trials 400 --out "$OUT/gamma_recomb_s${s}.json" \
    > "$OUT/gamma_recomb_s${s}.log" 2>&1
}
i=0
for s in 42 43 44 100 101 102; do
  run_one "$s" &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi   # throttle: 3 concurrent GPU encodes
done
wait
echo "=== all 6 seeds done ==="
ngo=0
for s in 42 43 44 100 101 102; do
  line=$(grep -E "MAIN recomb_frac" "$OUT/gamma_recomb_s${s}.log" 2>/dev/null | tail -1)
  echo "seed $s: $line"
  echo "$line" | grep -q "RECOMB-GO" && ngo=$((ngo+1))
done
echo "=== 6-SEED GO COUNT: $ngo/6 ==="
