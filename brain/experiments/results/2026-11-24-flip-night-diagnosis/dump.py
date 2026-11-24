# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (host-only read of banked lives)
# Expected-fail: no (a read-out; no model run)
import json,sys
from brain.experiments.strict_scorer import is_first_person as fp
p=sys.argv[1]; nights=[int(x) for x in sys.argv[2].split(',')]
r=json.load(open(p)); L=r['ledger']
print({k:(str(v)[:150]) for k,v in r.items() if k not in('ledger','probes','nights','lived','schedule')})
print('probe keys',list(r['probes'][0].keys()), 'row keys', list(r['probes'][0]['rows'][0].keys()))
print('night keys',list(r['nights'][0].keys()))
for pr in r['probes']:
    if pr.get('line','trunk')!='trunk' or pr.get('tag'): continue
    if pr['night'] in nights and pr['arm'] in ('hippocampus-off','intact'):
        print('---',pr['night'],pr['arm'],pr['summary'])
        for i,x in enumerate(pr['rows']):
            e=L.get(x.get('item') or '')
            print(f" {i:2} {x['kind']:8} {'T' if e and e.get('telling') else '-'} {'FP' if fp(x['brain']) else '  '} {x['you'][:45]:45} -> {x['brain'][:70]}")
