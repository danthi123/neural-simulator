#!/bin/bash
# Curiosity stage-1 dev (brain/experiments/inquiry_dev.py), 1.5B, dev bank, seed 1143. bash <this> <arms>
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/inqdev; mkdir -p $O; T=$(( $(nproc) - 1 ))
$PY -m brain.experiments.inquiry_dev --model 1.5b --seed 1143 --arms $1 --samples 2 --threads $T \
  --out $O/inq_$(echo $1 | tr , _).json > $O/inq_$(echo $1 | tr , _).log 2>&1
echo "inq $1 rc $?" >> $O/done.txt
