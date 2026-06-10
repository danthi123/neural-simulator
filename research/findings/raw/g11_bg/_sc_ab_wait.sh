#!/bin/bash
cd /e/Documents/Projects/sim
ON=research/findings/raw/g11_bg/_sc_navON_seed42.json
HR=research/findings/raw/g11_bg/_sc_navHOSTREFLEX_seed42.json
for i in $(seq 1 60); do
  if [ -f "$ON" ] && [ -f "$HR" ]; then break; fi
  sleep 30
done
python -X utf8 -c "
import json
def navsum(f):
    d=json.load(open(f)); return sum(p['final_quarter_mean_distance'] for p in d.get('phase_stats',[]))
try:
    on=navsum('$ON'); hr=navsum('$HR')
    print('=== SPIKING-SC NAV A/B (seed 42, nav_sum=sum final_quarter_mean_distance, LOWER better) ===')
    print('SC-on (spiking SC orienting): %.3f'%on)
    print('host-reflex (the scaffold):   %.3f'%hr)
    r=on/hr if hr else 0
    print('SC/host ratio: %.3f'%r)
    if r<=1.10: print('VERDICT: SC-on MATCHES/BEATS host-reflex (within 10%) -> the spiking SC orienting is a no-regression GO; run 6-seed + anti-cheats.')
    elif r<=1.4: print('VERDICT: SC-on competitive but %d%% worse -> TUNE w_sc_cortex / sc_retina_drive_pa (integration-vs-isolation gap), re-run.'%((r-1)*100))
    else: print('VERDICT: SC-on %d%% worse -> the SC orienting is too weak vs BG+OU; raise w_sc_cortex substantially (the host reflex injects ~150pA; the SC pooling must match) + re-run.'%((r-1)*100))
except Exception as e:
    print('aggregate error (runs may not have finished):', e)
"
