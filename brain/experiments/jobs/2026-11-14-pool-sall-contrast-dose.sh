#!/bin/bash
# brain/research/2026-11-14-sall-contrast-dose-1.5b.md on one pool node: bash <this> <seed> ...
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outSD; mkdir -p $O
for s in "$@"; do for e in contrast; do
  $PY -m brain.experiments.unslept_day_probe --home ~/brain-lane/workSD_s$s --seed $s --arms c,s,r --speaker-contrast 1.5 --out $O/${e}_s$s.json > $O/${e}_s$s.log 2>&1
  echo "$e s$s rc $?" >> $O/done.txt
done; done
echo DONE >> $O/done.txt
