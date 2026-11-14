#!/bin/bash
# Report-only follow-ups to brain/research/2026-11-13-gain-refit-1.5b.md (run AFTER the registered verdict, LEAD at
# G* = 0.5): finer grid at the low edge (dev + test), and a wrong-memory control (episode of fact i+1) at gain 0.5 / 1.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/out2; mkdir -p $O
$PY -m brain.experiments.hippo_gain_refit --bank test --gains 0.5 1 --shift 1 --out $O/shift_test.json > $O/shift_test.log 2>&1
$PY -m brain.experiments.hippo_gain_refit --bank test --gains 0.25 0.75 --out $O/fine_test.json > $O/fine_test.log 2>&1
$PY -m brain.experiments.hippo_gain_refit --bank dev --gains 0.25 0.75 --out $O/fine_dev.json > $O/fine_dev.log 2>&1
echo DONE > $O/DONE
