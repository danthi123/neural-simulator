#!/bin/bash
# Curiosity flip guard (prereg Amendment 1): weekly 1.5B off vs --inquiry auto on the pool (CPU): bash <this> <seed> ...
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outIQ; mkdir -p $O
for s in "$@"; do
  for arm in off on; do
    F=""; [ $arm = on ] && F="--inquiry auto"
    $PY -m brain.eval.weekly --model 1.5b --shared-sleep --seed $s $F --homes ~/brain-lane/workIQ_${arm}_s$s --out $O/iq${arm}_s$s.md > $O/iq${arm}_s$s.log 2>&1
    echo "iq $arm s$s rc $?" >> $O/done.txt
  done
done
