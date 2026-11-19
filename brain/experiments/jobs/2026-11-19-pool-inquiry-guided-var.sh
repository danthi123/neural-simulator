#!/bin/bash
# Curiosity inquiry gate DEV, guided-reply variant (gate arm only), 1.5B, dev bank. bash <this> <seed> <tag> "<guided>"
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/igdev; mkdir -p $O; T=$(( $(nproc) - 1 ))
$PY -m brain.experiments.inquiry_gate_dev --model 1.5b --seed $1 --arms gate --samples 2 --bar 12.4 --threads $T --guided "$3" --out $O/igv_$2_s$1.json > $O/igv_$2_s$1.log 2>&1
echo "igv $2 s$1 rc $?" >> $O/done_var.txt
