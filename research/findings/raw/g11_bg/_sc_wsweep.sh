#!/bin/bash
cd /e/Documents/Projects/sim
D=research/findings/raw/g11_bg
COMMON="--moving-goal --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --grid-size 8 --n-steps 1800 --seed 42 --enable-spiking-sc --no-emit-webapp-sidecar"
for w in 15 40 100; do
  SC_CORTEX_W=$w PYTHONIOENCODING=utf-8 python -X utf8 -m research.runners.g11_bg_runner $COMMON --out $D/_scW${w}_seed42.json > $D/_scW${w}_seed42.log 2>&1 &
done
wait
python -X utf8 -c "
import json
def nv(f):
    d=json.load(open(f)); return sum(p['final_quarter_mean_distance'] for p in d.get('phase_stats',[]))
HOST=4.051
print('=== w_sc_cortex sweep (SC-on, seed 42) vs host-reflex %.3f (LOWER better) ==='%HOST)
best=None
for w in (15,40,100):
    try:
        v=nv('$D/_scW%d_seed42.json'%w); r=v/HOST
        tag='<= host (GO!)' if r<=1.10 else ('competitive' if r<=1.3 else 'still weak')
        print('w_sc_cortex=%3d: nav_sum=%.3f  SC/host=%.3f  %s'%(w,v,r,tag))
        if best is None or v<best[1]: best=(w,v,r)
    except Exception as e:
        print('w=%d: not done/err %s'%(w,e))
if best: print('BEST: w_sc_cortex=%d nav_sum=%.3f (SC/host %.3f). %s'%(best[0],best[1],best[2], 'GO -> 6-seed + anti-cheats' if best[2]<=1.15 else 'sweep higher/lower around best'))
"
