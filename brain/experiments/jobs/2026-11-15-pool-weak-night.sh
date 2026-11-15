#!/bin/bash
# Diagnostic for brain/research/2026-11-14-near-miss-guard-1.5b.md (Amendment 3): matched WEAKER NIGHTS without the
# guard, seeds 200-205 on CPU (same lane as the report-only extras, so nmoff/nmon there are the comparison).
#   bash <this> <seed> ...
# Two doses: kl6 (ordinary-text KL weight 6 instead of 3: a uniform pull toward the pretrained cortex) and
# st240 (240 steps instead of 480: a shorter night). Everything else as the extras.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/outWN; mkdir -p $O
for s in "$@"; do
  p=$([ $((s % 2)) -eq 0 ] && echo even || echo odd)
  B="--model 1.5b --seed $s --slept $p --lr 5e-4 --arms c,s,r --self-q"
  $PY -m brain.experiments.mixed_day_probe $B --steps 480 --kl-weight 6 --home ~/brain-lane/workWN_kl6_s$s --out $O/wnkl6_s$s.json > $O/wnkl6_s$s.log 2>&1
  echo "kl6 s$s rc $?" >> $O/done.txt
  $PY -m brain.experiments.mixed_day_probe $B --steps 240 --kl-weight 3 --home ~/brain-lane/workWN_st240_s$s --out $O/wnst240_s$s.json > $O/wnst240_s$s.log 2>&1
  echo "st240 s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
