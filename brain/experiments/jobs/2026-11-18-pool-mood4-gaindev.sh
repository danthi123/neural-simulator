#!/bin/bash
# brain/research/2026-11-03-mood-state-design.md section 15: 1.5B GAIN dev, seed 42, mixed_dev x 1. bash <this> <gain>
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/mood4; mkdir -p $O
g=$1; T=$(( $(nproc) - 1 ))
$PY -m brain.experiments.mood_probe --model 1.5b --seed 42 --threads $T --gain $g --questions mixed_dev --samples 1 \
  --max-tokens 40 --arms pos,neg,off --no-embedding --out $O/gaindev_1.5b_g${g}_s42.json > $O/gaindev_g$g.log 2>&1
echo "gaindev g$g rc $?" >> $O/done.txt
