#!/bin/bash
cd /e/Documents/Projects/sim
D=research/findings/raw/g11_bg
COMMON="--moving-goal --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --grid-size 8 --n-steps 1800 --no-emit-webapp-sidecar"
declare -a JOBS
for s in 42 43 44 45 46 47; do
  JOBS+=("ON $s --enable-spiking-sc")
  JOBS+=("HR $s --sc-orienting-reflex")
done
JOBS+=("SCRAM 42 --enable-spiking-sc")   # SC_SCRAMBLE handled below
run_one() {
  local tag=$1 seed=$2; shift 2; local flags="$*"
  local env="SC_CORTEX_W=18"
  [ "$tag" = "SCRAM" ] && env="$env SC_SCRAMBLE=1"
  env $env PYTHONIOENCODING=utf-8 python -X utf8 -m research.runners.g11_bg_runner $COMMON --seed $seed $flags --out $D/_sc6_${tag}_seed$seed.json > $D/_sc6_${tag}_seed$seed.log 2>&1
}
i=0
for j in "${JOBS[@]}"; do
  run_one $j &
  i=$((i+1)); if (( i % 3 == 0 )); then wait; fi
done
wait
python -X utf8 -c "
import json,glob
def nv(f):
    d=json.load(open(f)); return sum(p['final_quarter_mean_distance'] for p in d.get('phase_stats',[]))
on={};hr={}
for f in glob.glob('$D/_sc6_ON_seed*.json'): on[int(f.split('seed')[1].split('.')[0])]=nv(f)
for f in glob.glob('$D/_sc6_HR_seed*.json'): hr[int(f.split('seed')[1].split('.')[0])]=nv(f)
ss=sorted(set(on)&set(hr))
print('=== 6-SEED SPIKING-SC NAV A/B (w=18; nav_sum, LOWER better) ===')
print('seed   SC-on   host-reflex   SC/host')
for s in ss: print('%4d  %6.3f  %10.3f   %.3f'%(s,on[s],hr[s],on[s]/hr[s] if hr[s] else 0))
if ss:
    mo=sum(on[s] for s in ss)/len(ss); mh=sum(hr[s] for s in ss)/len(ss); r=mo/mh
    print('MEAN  %6.3f  %10.3f   %.3f'%(mo,mh,r))
    v='N1 CLOSED: spiking SC MATCHES/BEATS host (mean %.2f)'%r if r<=1.10 else ('COMPETITIVE (mean %.2f, within 20%%) = successful biologization, honest small gap'%r if r<=1.20 else 'honest gap %d%% (substrate vs host cheat)'%((r-1)*100))
    print('VERDICT:',v)
try:
    sc=nv('$D/_sc6_SCRAM_seed42.json'); print('ANTI-CHEAT scrambled-retinotopy seed42: nav_sum=%.3f vs SC-on=%.3f -> %s'%(sc,on.get(42,0),'REGRESSES (good, orienting is retinotopic)' if on.get(42,99) and sc>on[42]*1.3 else 'does NOT regress (investigate leak)'))
except Exception as e: print('scramble:',e)
"
