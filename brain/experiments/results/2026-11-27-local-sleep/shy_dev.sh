#!/usr/bin/env bash
# Seed-7 0.4B SHY dev sweep (prereg 2026-11-27-local-sleep-shy-prereg-DRAFT.md), CPU, on a pool node. Per night, ONE
# sleep, then the adapters are evaluated at every g of the grid (copy, downscale B, evaluate, restore): 2 local sleeps
# (transport, decay 1e-3, lr 0.1 and 0.3) and the gradient arm (same grid, matched control). local_shuffled is NOT run in dev
# (the pick does not use it); it runs in the six-seed arms only.
# Run on pool40 (23 GB) first:  cd ~/brain-lane/route_repo && bash brain/experiments/results/2026-11-27-local-sleep/shy_dev.sh
# Needs COMMIT (env, else ~/brain-lane/route_repo/COMMIT); exits if empty. Resumable (skips configs whose json exists). OUT sets the output dir. RSS at 0.4B under memcap 20 is UNMEASURED.
set -u
export COMMIT=${COMMIT:-$(cat ~/brain-lane/route_repo/COMMIT 2>/dev/null)}
[ -n "$COMMIT" ] || { echo "COMMIT empty: set COMMIT or write ~/brain-lane/route_repo/COMMIT" >&2; exit 1; }
export HOST=$(hostname)
OUT=${OUT:-$HOME/brain-lane/out/local-sleep-shy-dev}
mkdir -p "$OUT"
export OMP_NUM_THREADS=6
export BRAIN_MODEL_DIR=~/brain-lane/models
export PYTHONPATH=~/brain-lane/route_repo:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
MODEL=$BRAIN_MODEL_DIR/rwkv7-g1d-0.4b-20260210-ctx8192.pth
TIMEV=(); [ -x /usr/bin/time ] && TIMEV=(/usr/bin/time -v)   # puts "Maximum resident set size" in each log (RSS measurement)
GRID=${GRID:-1,0.8,0.6,0.4,0.25}
run() {  # name, then local_sleep args
  local name=$1; shift
  [ -s "$OUT/$name.json" ] && { echo "skip $name"; return; }
  bash tools/memcap.sh 20 -- "${TIMEV[@]}" "$PY" -m brain.experiments.local_sleep --model "$MODEL" --seed 7 --steps 240 \
    --downscale-grid "$GRID" --downscale-factors B \
    --out "$OUT/$name.json.tmp" "$@" > "$OUT/$name.log" 2>&1 && mv "$OUT/$name.json.tmp" "$OUT/$name.json"
}
run gradient --arms gradient
for lr in 0.3 0.1; do
  run "transport_lr${lr}_dec1e-3" --arms local --feedback transport --lr "$lr" --decay 1e-3
done
echo "shy dev done: $(ls "$OUT"/*.json | wc -l) json in $OUT"
