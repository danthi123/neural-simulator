"""Analyze a g11_bg_runner spiking_wta accumulate-then-commit result:
  - cheat-5 SUM metric (sum of per-phase final_quarter_mean_distance; target ~2.34)
  - thal-winner alignment (does the committed action match the clean thalamic winner)
  - accumulate/commit guard (winner's sel ramps, its commit bursts, losers low)
Usage: python research/findings/raw/_n6_analyze.py <result.json>
"""
import json, sys
import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else 'research/findings/raw/_n6_accum_commit_smoke1_seed42.json'
d = json.load(open(p))
ACTIONS = ['N', 'E', 'S', 'W']

print('=== RESULT:', p, '===')
print('readout_source:', d.get('readout_source'), '| use_commit_readout:', d.get('use_commit_readout'))
ps = d['phase_stats']
fq = [x['final_quarter_mean_distance'] for x in ps]
print('per-phase final_quarter_mean_distance:', [round(x, 3) for x in fq])
print('CHEAT-5 SUM (target ~2.34, GATE <= ~3):', round(sum(fq), 3))
print('mean_distance_overall:', round(d['mean_distance_overall'], 3))
print('n_steps_at_goal:', d['n_steps_at_goal'], '/', d['n_steps'])

# Alignment: committed action vs thal winner (guard-logged thal_counts)
tc = d.get('thal_counts', [])
cc = d.get('commit_counts', [])
sc = d.get('sel_counts', [])
al = d.get('action_log', [])
if tc:
    match = n = 0
    for i in range(len(al)):
        if i < len(tc) and max(tc[i]) > 0:
            n += 1
            if al[i] == int(np.argmax(tc[i])):
                match += 1
    print(f'\ncommitted == thal-winner: {match}/{n} = {match/max(n,1)*100:.1f}%')

# Commit selectivity guard
if cc:
    ccarr = np.array(cc)
    win = ccarr.max(axis=1)
    runnerup = np.sort(ccarr, axis=1)[:, -2]
    print(f'commit burst: winner mean {win.mean():.1f}, runner-up mean {runnerup.mean():.1f}, '
          f'separation ratio {win.mean()/max(runnerup.mean(),0.01):.1f}x')
    print(f'frac trials commit fully silent: {(win==0).mean():.3f}')
if sc:
    scarr = np.array(sc)
    win = scarr.max(axis=1)
    runnerup = np.sort(scarr, axis=1)[:, -2]
    print(f'sel accumulator: winner mean {win.mean():.1f}, runner-up mean {runnerup.mean():.1f}')

# Per-substep accumulation/commit trace for sample trials (the ramp + burst guard)
at = d.get('accum_trace', {})
print('\n=== ACCUMULATION/COMMIT TRACE (sample trials) ===')
for tk in sorted(at.keys(), key=lambda x: int(x))[:3]:
    sel = np.array(at[tk]['sel']) if at[tk]['sel'] else None
    com = np.array(at[tk]['commit']) if at[tk]['commit'] else None
    if sel is None:
        continue
    # winner = action with max total sel over the trial
    tot = sel.sum(axis=0)
    win_a = int(np.argmax(tot))
    print(f'-- trial {tk}: winner={ACTIONS[win_a]} (sel total {tot.tolist()}) --')
    # cumulative sel ramp for winner vs mean loser, at checkpoints
    cum = np.cumsum(sel, axis=0)
    loser_idx = [j for j in range(4) if j != win_a]
    for si in [10, 30, 50, 70, 99]:
        if si < len(cum):
            wr = cum[si, win_a]
            lr = cum[si, loser_idx].mean()
            cb = com[si].tolist() if com is not None and si < len(com) else None
            print(f'   substep {si:3d}: winner sel cum={wr:4d}  mean-loser sel cum={lr:5.1f}  commit(inst)={cb}')
