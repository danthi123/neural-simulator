# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-25-sall-live-dev-prereg.md (P2s: pooled intact-frozen right-episode fetches >= lifeKP 45/223)
# Expected-fail: no (a counter; reproduces lifeKP 45/223)
"""Intact-frozen told rows, nights 5-15: fetched the right episode / a wrong one / none."""
import json,sys
def ff(f):
    r=json.load(open(f)); L=r['ledger']; c=dict(right=0,wrong=0,none=0,n=0)
    for p in r['probes']:
        if p['arm']!='intact-frozen' or not (5<=p['night']<=15) or p.get('tag'): continue
        for row in p['rows']:
            e=L.get(row.get('item')) if row.get('item') else None
            if not e or not e.get('telling') or row.get('ok') is None: continue
            c['n']+=1; rc=row.get('recalled')
            if not rc: c['none']+=1
            elif e['telling'].strip() in rc or rc.strip() in e['telling']: c['right']+=1
            else: c['wrong']+=1
    return r['seed'],c
for f in sys.argv[1:]: print(*ff(f))
