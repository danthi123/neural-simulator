#!/bin/bash
# Part A of brain/research/2026-11-14-key-default-1.5b.md on one pool node: bash <this> <seed> [<seed> ...]
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outA; mkdir -p $O
for s in "$@"; do
  $PY -m brain.experiments.unslept_day_probe --home ~/brain-lane/work_s$s --seed $s --arms c,s,r --out $O/unslept_s$s.json > $O/unslept_s$s.log 2>&1
  echo "s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
