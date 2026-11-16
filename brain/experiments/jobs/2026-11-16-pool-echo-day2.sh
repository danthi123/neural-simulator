#!/bin/bash
# Report-only diagnostic: seed 1121 echoes the user verbatim 14/16 on day 2 in both arms
# (results/2026-11-16-reply-rehearse-pool-replication). Is it the night or the cortex alone? Replays day 2 on the slept
# default home in three arms (brain/eval/open_conv_day2.py). pool40 only. Installed as ~/brain-lane/echo.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/hab; : > $O/done5.txt
$PY -m brain.eval.open_conv_day2 --model 1.5b --seed 1121 --home $O/homes_d_s1121/hippocampus_off --out $O/d2_s1121.md > $O/d2_s1121.log 2>&1
echo "d2 s1121 rc $?" >> $O/done5.txt; echo DONE >> $O/done5.txt
