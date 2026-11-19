#!/bin/bash
# Mood iteration 6 (design doc s19 + 19.0): 1.5B calibrate, trace, then the pilot (seed 42, SPAN 1, OFF/ON/CONSTANT).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/arousal6; mkdir -p $O/scratch; T=$(( $(nproc) - 1 ))
$PY -m brain.experiments.arousal_gain calibrate --model 1.5b --threads $T --out $O/calib_1.5b.json --scratch $O/scratch > $O/calib.log 2>&1
echo "calib rc $?" >> $O/done.txt
$PY -m brain.experiments.arousal_gain trace --model 1.5b --threads $T --calib $O/calib_1.5b.json --out $O/trace_1.5b.json --scratch $O/scratch > $O/trace.log 2>&1
echo "trace rc $?" >> $O/done.txt
$PY -m brain.experiments.arousal_gain pilot --model 1.5b --threads $T --calib $O/calib_1.5b.json --seeds 42 --off-split yes --out $O/pilot_s42.json --scratch $O/scratch > $O/pilot.log 2>&1
echo "pilot rc $?" >> $O/done.txt
