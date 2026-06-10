#!/bin/bash
cd /e/Documents/Projects/sim
D=research/findings/raw/g11_bg
N9="--moving-goal --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --enable-spiking-sc --enable-neural-critic --spiking-snc --spiking-reward-us --neural-place-selforg --selforg-steps 400 --selforg-n-positions 16 --value-train-trials 40 --value-train-pair-steps 60 --value-train-hold-steps 30 --critic-teacher-pa 300 --reward-delay-steps 8 --value-train-stdp-w-max 40 --enable-critic-fs-inhibition --deterministic-selforg --n-steps 1800 --grid-size 8 --seed 42 --no-emit-webapp-sidecar"
SC_CORTEX_W=18 PYTHONIOENCODING=utf-8 python -X utf8 -m research.runners.g11_bg_runner $N9 --enable-spiking-sc-approach --out $D/_n5ab_NEURAL_seed42.json > $D/_n5ab_NEURAL_seed42.log 2>&1 &
SC_CORTEX_W=18 PYTHONIOENCODING=utf-8 python -X utf8 -m research.runners.g11_bg_runner $N9 --perceived-approach-reward --out $D/_n5ab_HOST_seed42.json > $D/_n5ab_HOST_seed42.log 2>&1 &
wait
python -X utf8 -c "
import json
def nv(f):
    d=json.load(open(f)); return sum(p['final_quarter_mean_distance'] for p in d.get('phase_stats',[]))
try:
    n=nv('$D/_n5ab_NEURAL_seed42.json'); h=nv('$D/_n5ab_HOST_seed42.json')
    print('=== N5 nav A/B (seed42; nav_sum LOWER better) ===')
    print('NEURAL approach-reward (approach_n5): %.3f'%n)
    print('HOST sign(delta ecc) reward:          %.3f'%h)
    r=n/h if h else 0; print('neural/host: %.3f'%r)
    print('VERDICT:', 'N5 neural reward MATCHES/BEATS host -> close N5' if r<=1.15 else ('competitive, minor TD tuning' if r<=1.4 else 'TD reward too noisy -> tune nmda_slow tau vs nav cadence / dead-band / read window'))
except Exception as e: print('agg err (runs may not be done):',e)
"
