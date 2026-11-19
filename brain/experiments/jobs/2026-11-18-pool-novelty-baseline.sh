#!/bin/bash
# Curiosity: unchanged-brain baseline on the TEST bank, 4 samples per item (brain/experiments/novelty_stage0.py). bash <this> <seed>...
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/novbase; mkdir -p $O; T=$(( $(nproc) - 1 ))
for s in "$@"; do
  $PY -m brain.experiments.novelty_stage0 --model 1.5b --bank test --samples 4 --seed $s --threads $T --out $O/base_s$s.json > $O/base_s$s.log 2>&1
  echo "base s$s rc $?" >> $O/done.txt
done
