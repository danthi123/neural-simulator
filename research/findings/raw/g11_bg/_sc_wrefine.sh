#!/bin/bash
cd /e/Documents/Projects/sim
D=research/findings/raw/g11_bg
COMMON="--moving-goal --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --grid-size 8 --n-steps 1800 --seed 42 --enable-spiking-sc --no-emit-webapp-sidecar"
for w in 8 12 18; do
  SC_CORTEX_W=$w PYTHONIOENCODING=utf-8 python -X utf8 -m research.runners.g11_bg_runner $COMMON --out $D/_scR${w}_seed42.json > $D/_scR${w}_seed42.log 2>&1 &
done
wait
python -X utf8 -c "
import json
def nv(f):
    d=json.load(open(f)); return sum(p['final_quarter_mean_distance'] for p in d.get('phase_stats',[]))
HOST=4.051
print('=== w_sc_cortex REFINE (seed 42) vs host %.3f; prior: w15=4.573(1.13) ==='%HOST)
res=[(15,4.573)]
for w in (8,12,18):
    try: res.append((w,nv('$D/_scR%d_seed42.json'%w)))
    except Exception as e: print('w=%d err %s'%(w,e))
res.sort(key=lambda x:x[1])
for w,v in sorted(res): print('w=%3d: nav_sum=%.3f SC/host=%.3f'%(w,v,v/HOST))
b=res[0]; print('BEST w=%d nav_sum=%.3f (SC/host %.3f) -> 6-seed at this w'%(b[0],b[1],b[1]/HOST))
"
