cd /e/Documents/Projects/sim
run() {
  V=$1; s=$2; log="research/findings/raw/_langwikiV_V${V}_s${s}.log"
  grep -qE "language-test" "$log" 2>/dev/null && return
  env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u -m research.runners._reslm_controlled_lag_eprop_derisk \
    --language-test --corpus data/corpus/wikitext.txt --seeds $s --n-sentences 3500 --epochs 12 --n-pool 200 --language-vocab $V \
    > "$log" 2>&1
}
for V in 200 600; do for s in 42 43 44; do
  run $V $s &
  while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 3; done
done; done
wait
echo "VOCAB-SCALE THROTTLED SWEEP DONE"
