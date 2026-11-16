#!/bin/bash
# Report-only: reply monitor (Brain(reply_monitor=4), brain/research/2026-11-16-reply-monitor-1.5b.md) on pool seeds
# whose default arm already ran (results/2026-11-16-reply-rehearse-pool-replication): monitor arm only.
#   bash <this> <seed>   (pool40: 1121, pool41: 1124, pool42: 1125). Installed as ~/brain-lane/mon.sh.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/hab; mkdir -p $O; : > $O/done6.txt
s=$1
$PY -m brain.eval.open_conv --model 1.5b --seed $s --homes $O/homes_m_s$s --reply-monitor 4 --out $O/ocm_s$s.md > $O/ocm_s$s.log 2>&1
echo "ocm s$s rc $?" >> $O/done6.txt
echo DONE >> $O/done6.txt
