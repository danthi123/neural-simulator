#!/bin/bash
# 6-seed spiking-SC nav A/B: SC-on (--enable-spiking-sc) vs host-reflex (--sc-orienting-reflex)
# the scaffold it replaces. nav_sum = sum final_quarter_mean_distance (LOWER better).
# GPU (CuPy); parallel-3 sharing. Run AFTER the single-seed verdict is GO/competitive.
cd /e/Documents/Projects/sim
D=research/findings/raw/g11_bg
COMMON="--moving-goal --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --grid-size 8 --n-steps 1800 --no-emit-webapp-sidecar"
run() { PYTHONIOENCODING=utf-8 python -X utf8 -m research.runners.g11_bg_runner $COMMON --seed $2 $3 --out $D/_sc6_${1}_seed$2.json > $D/_sc6_${1}_seed$2.log 2>&1; }
for s in 42 43 44 45 46 47; do
  run ON  $s "--enable-spiking-sc" &
  run HR  $s "--sc-orienting-reflex" &
  if (( s % 3 == 44 % 3 )); then wait; fi   # throttle ~ a few parallel
done
wait
echo "=== 6-seed spiking-SC nav A/B (nav_sum, LOWER better) ==="
python -X utf8 -c "
import json,glob,os
def nv(f):
    d=json.load(open(f)); return sum(p['final_quarter_mean_distance'] for p in d.get('phase_stats',[]))
on={};hr={}
for f in glob.glob('$D/_sc6_ON_seed*.json'):
    on[int(f.split('seed')[1].split('.')[0])]=nv(f)
for f in glob.glob('$D/_sc6_HR_seed*.json'):
    hr[int(f.split('seed')[1].split('.')[0])]=nv(f)
ss=sorted(set(on)&set(hr)); 
print('seed   SC-on   host-reflex   SC/host')
for s in ss: print('%4d  %6.3f  %10.3f   %.3f'%(s,on[s],hr[s],on[s]/hr[s] if hr[s] else 0))
if ss:
    mo=sum(on[s] for s in ss)/len(ss); mh=sum(hr[s] for s in ss)/len(ss)
    print('mean  %6.3f  %10.3f   %.3f'%(mo,mh,mo/mh if mh else 0))
"
