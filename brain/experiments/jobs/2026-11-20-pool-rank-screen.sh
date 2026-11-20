#!/bin/bash
# R-raw screen (prereg brain/research/2026-11-20-hub-rank-raw-prereg.md section 3) on this node's banked night-7 homes,
# after the running dev life, then the next dev life. usage: rank_then_next.sh <wait seed_arm> <next seed> <next arm> <screen seed>...
W=$1; NS=$2; NA=$3; shift 3
cd ~/brain-lane
until grep -q "life s${W} rc" outLIFE/done.txt 2>/dev/null; do sleep 120; done
PY=~/brain-lane/venv/bin/python; [ -x $PY ] || PY=~/derisk-pool/sim/.venv/bin/python
for S in "$@"; do
  BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib nice -n 5 $PY -m brain.experiments.life_screen rank \
    --home ~/brain-lane/workLIFE_s$S/trunk --seed $S --model 1.5b --night 7 --work ~/brain-lane/workRANK_s$S \
    --out ~/brain-lane/outLIFE/screen_rank_s$S.json > outLIFE/screen_rank_s$S.log 2>&1
  echo "screen rank s$S rc $?" >> outLIFE/done.txt
  rm -rf ~/brain-lane/workRANK_s$S
done
bash ~/brain-lane/life_arm.sh $NS $NA
