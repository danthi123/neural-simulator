#!/bin/bash
# Curiosity inquiry gate DEV (brain/experiments/inquiry_gate_dev.py), 1.5B, dev bank. bash <this> <seed>
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/igdev; mkdir -p $O; T=$(( $(nproc) - 1 ))
$PY -m brain.experiments.inquiry_gate_dev --model 1.5b --seed $1 --samples 2 --bar 12.4 --threads $T --out $O/igd_s$1.json > $O/igd_s$1.log 2>&1
echo "igd s$1 rc $?" >> $O/done.txt
