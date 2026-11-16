#!/bin/bash
# Own-cue frame selection then the teacher gate with it (brain/research/2026-11-15-own-cue-frame-1.5b.md).
#   bash <this> <seed> ...   (pool40: 42 43, pool41: 44 100, pool42: 101 102). Installed as ~/brain-lane/ocf.sh.
# Stage A is greedy and identical on every node (a cross-node consistency check); each node gates its own seeds.
cd ~/brain-lane/src || exit 1
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=~/brain-lane/src:~/brain-lane/pylib
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
O=~/brain-lane/ocf; mkdir -p $O; rm -f $O/done.txt
$PY -m brain.experiments.own_cue_frames --out $O/frames_$(hostname).json > $O/frames.log 2>&1
echo "frames rc $?" >> $O/done.txt
SEL=$($PY -c "import json;print(json.load(open('$O/frames_$(hostname).json'))['selected'])")
DEV=$($PY -c "import json;d=json.load(open('$O/frames_$(hostname).json'));print(d['frames'][d['selected']]['dev_good'])")
echo "selected $SEL dev_good $DEV" >> $O/done.txt
if [ "$DEV" -ge 24 ]; then
  for s in "$@"; do
    $PY -m brain.experiments.teacher_gate_probe --model 1.5b --seed $s --k 4 --own-frame $SEL --out $O/tg_s$s.json > $O/tg_s$s.log 2>&1
    echo "tg s$s rc $?" >> $O/done.txt
  done
else
  echo "A1 fails (dev_good $DEV < 24): no gate run" >> $O/done.txt
fi
echo DONE >> $O/done.txt
