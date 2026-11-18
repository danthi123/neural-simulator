#!/bin/bash
# brain/research/2026-11-18-sall-open-conv-1.5b.md Amendment 2: six-seed gate on 42/43/44/100/101/102, pool CPU.
# Per seed: open_conv d then s (S_all + slept bar), then weekly (report) S0 then S_all + slept bar.
#   bash <this> <seed> <seed>   (pool40: 42 43, pool41: 44 100, pool42: 101 102).
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/sgate; mkdir -p $O
for s in "$@"; do
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_d_s$s --out $O/ocd_s$s.md > $O/ocd_s$s.log 2>&1
  echo "ocd s$s rc $?" >> $O/done.txt
  $PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_s_s$s --key S_all --key-bar slept --out $O/ocs_s$s.md > $O/ocs_s$s.log 2>&1
  echo "ocs s$s rc $?" >> $O/done.txt
done
for s in "$@"; do
  $PY -m brain.eval.weekly --model 1.5b --shared-sleep --seed $s --homes $O/wk_d_s$s --out $O/wkd_s$s.md > $O/wkd_s$s.log 2>&1
  echo "wkd s$s rc $?" >> $O/done.txt
  $PY -m brain.eval.weekly --model 1.5b --shared-sleep --seed $s --key S_all --key-bar slept --homes $O/wk_s_s$s --out $O/wks_s$s.md > $O/wks_s$s.log 2>&1
  echo "wks s$s rc $?" >> $O/done.txt
done
echo DONE >> $O/done.txt
