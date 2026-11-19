#!/bin/bash
# Curiosity inquiry TEST bank, one arm set, 1.5B. bash <this> <seed> <arms> <tag> ["<guided>"]
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/igtest; mkdir -p $O; T=$(( $(nproc) - 1 ))
G=(); [ -n "$4" ] && G=(--guided "$4")
$PY -m brain.experiments.inquiry_gate_dev --model 1.5b --bank test --seed $1 --arms $2 --samples 4 --bar 12.4 --threads $T "${G[@]}" --out $O/igt_$3_s$1.json > $O/igt_$3_s$1.log 2>&1
echo "igt $3 s$1 rc $?" >> $O/done.txt
