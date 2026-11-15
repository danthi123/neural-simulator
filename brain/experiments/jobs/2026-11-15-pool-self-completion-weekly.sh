#!/bin/bash
# report-only: weekly 1.5B off/on --self-completion on the pool (CPU): bash <this> <seed> ...
# (prereg brain/research/2026-11-15-self-completion-weekly-1.5b.md addendum). Installed as ~/brain-lane/sc.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outSC; mkdir -p $O; rm -f $O/done.txt
for s in "$@"; do
  for arm in off on; do
    F=""; [ $arm = on ] && F=--self-completion
    $PY -m brain.eval.weekly --model 1.5b --shared-sleep --seed $s $F --homes ~/brain-lane/workSC_${arm}_s$s --out $O/sc${arm}_s$s.md > $O/sc${arm}_s$s.log 2>&1
    echo "sc $arm s$s rc $?" >> $O/done.txt
  done
done
echo DONE >> $O/done.txt
