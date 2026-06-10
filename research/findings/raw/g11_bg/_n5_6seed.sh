#!/bin/bash
cd /e/Documents/Projects/sim
D=research/findings/raw/g11_bg
N9="--moving-goal --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --enable-spiking-sc --enable-neural-critic --spiking-snc --spiking-reward-us --neural-place-selforg --selforg-steps 400 --selforg-n-positions 16 --value-train-trials 40 --value-train-pair-steps 60 --value-train-hold-steps 30 --critic-teacher-pa 300 --reward-delay-steps 8 --value-train-stdp-w-max 40 --enable-critic-fs-inhibition --deterministic-selforg --n-steps 1800 --grid-size 8 --no-emit-webapp-sidecar"
run() { local tag=$1 seed=$2 extra=$3 scr=$4
  env SC_CORTEX_W=18 ${scr:+SC_SCRAMBLE=1} PYTHONIOENCODING=utf-8 python -X utf8 -m research.runners.g11_bg_runner $N9 --seed $seed $extra --out $D/_n56_${tag}_seed$seed.json > $D/_n56_${tag}_seed$seed.log 2>&1; }
i=0
for s in 42 43 44 45 46 47; do
  run NEURAL $s "--enable-spiking-sc-approach" "" & i=$((i+1)); (( i % 3 == 0 )) && wait
  run HOST   $s "--perceived-approach-reward" "" & i=$((i+1)); (( i % 3 == 0 )) && wait
done
run SCRAM 42 "--enable-spiking-sc-approach" "scr" &
wait
python -X utf8 -c "
import json,glob
def nv(f):
    d=json.load(open(f)); return sum(p['final_quarter_mean_distance'] for p in d.get('phase_stats',[]))
ne={};ho={}
for f in glob.glob('$D/_n56_NEURAL_seed*.json'): ne[int(f.split('seed')[1].split('.')[0])]=nv(f)
for f in glob.glob('$D/_n56_HOST_seed*.json'): ho[int(f.split('seed')[1].split('.')[0])]=nv(f)
ss=sorted(set(ne)&set(ho))
print('=== 6-SEED N5 nav A/B (neural approach-reward vs host sign(delta ecc); nav_sum LOWER better) ===')
print('seed  NEURAL   HOST    neural/host')
for s in ss: print('%4d %6.3f %7.3f   %.3f'%(s,ne[s],ho[s],ne[s]/ho[s] if ho[s] else 0))
if ss:
    mn=sum(ne[s] for s in ss)/len(ss); mh=sum(ho[s] for s in ss)/len(ss); r=mn/mh
    print('MEAN %6.3f %7.3f   %.3f'%(mn,mh,r))
    print('VERDICT:', 'N5 CLOSED: neural reward MATCHES/BEATS host (mean %.2f)'%r if r<=1.12 else ('N5 COMPETITIVE (mean %.2f, within 20%% = successful biologization, honest gap)'%r if r<=1.20 else 'honest gap %d%% -> tune the nmda_slow tau vs nav cadence'%((r-1)*100)))
try:
    sc=nv('$D/_n56_SCRAM_seed42.json'); print('ANTI-CHEAT scrambled-retinotopy seed42: %.3f vs NEURAL %.3f -> %s'%(sc,ne.get(42,0),'REGRESSES (good: approach reads the retinotopic bump)' if ne.get(42,99) and sc>ne[42]*1.3 else 'no regress (investigate)'))
except Exception as e: print('scramble:',e)
"
