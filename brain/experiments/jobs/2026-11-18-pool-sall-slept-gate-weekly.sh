#!/bin/bash
# Weekly half of jobs/2026-11-18-pool-sall-slept-gate.sh, rerun after the weekly.py __kwdefaults__ crash (11-18).
#   bash <this> <seed> <seed>   (pool40: 42 43, pool41: 44 100, pool42: 101 102).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/sgate
for s in "$@"; do
  $PY -m brain.eval.weekly --model 1.5b --shared-sleep --seed $s --homes $O/wk_d_s$s --out $O/wkd_s$s.md > $O/wkd_s$s.log 2>&1
  echo "wkd2 s$s rc $?" >> $O/done.txt
  $PY -m brain.eval.weekly --model 1.5b --shared-sleep --seed $s --key S_all --key-bar slept --homes $O/wk_s_s$s --out $O/wks_s$s.md > $O/wks_s$s.log 2>&1
  echo "wks2 s$s rc $?" >> $O/done.txt
done
echo DONE2 >> $O/done.txt
