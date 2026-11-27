#!/usr/bin/env bash
# Seed-7 dev sweep for the local sleep rule (prereg 2026-11-27), 0.1B, CPU, on a pool node. One config per process,
# in sequence, one json each. Reference first (no_sleep + gradient: the drift bar), then 3 presets x 4 lr x 2 decay.
# Run on the node:  cd ~/brain-lane/route_repo && bash brain/experiments/results/2026-11-27-local-sleep/dev_sweep.sh
# Skips configs whose json exists (resumable). Set OUT to change the output dir.
set -u
OUT=${OUT:-$HOME/brain-lane/out/local-sleep-dev}
mkdir -p "$OUT"
export OMP_NUM_THREADS=6
export BRAIN_MODEL_DIR=~/brain-lane/models
export PYTHONPATH=~/brain-lane/route_repo:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
MODEL=$BRAIN_MODEL_DIR/rwkv7-g1d-0.1b-20260129-ctx8192.pth
run() {  # name, then local_sleep args
  local name=$1; shift
  [ -s "$OUT/$name.json" ] && { echo "skip $name"; return; }
  bash tools/memcap.sh 13 -- "$PY" -m brain.experiments.local_sleep --model "$MODEL" --seed 7 --steps 240 \
    --out "$OUT/$name.json.tmp" "$@" > "$OUT/$name.log" 2>&1 && mv "$OUT/$name.json.tmp" "$OUT/$name.json"
}
run ref --arms no_sleep,gradient
for fbk in ${FBKS:-mixed transport random}; do
  for lr in 0.01 0.03 0.1 0.3; do
    for dec in 0 1e-3; do
      if [ "$fbk" = random ]; then arm=local_random; else arm=local; fi
      run "${fbk}_lr${lr}_dec${dec}" --arms "$arm" --feedback "$fbk" --lr "$lr" --decay "$dec"
    done
  done
done
echo "sweep done: $(ls "$OUT"/*.json | wc -l) json in $OUT"
