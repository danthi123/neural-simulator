# Capability: Self and identity
# Record: brain/research/2026-11-23-biggest-defect-read.md (M3 zero-cost screen); brain/research/2026-11-23-recall-voice-dev-prereg.md §2 rows
# Expected-fail: yes (recorded fetch fields may not separate hurt from helped; verdict DOES NOT SEPARATE)
import json,collections,numpy as np
def auroc(pos,neg):
    pos=np.array(pos,float);neg=np.array(neg,float)
    if len(pos)==0 or len(neg)==0: return float('nan')
    return ((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg))
rows=[]; tot=collections.Counter()
for s in (319,320,321):
    d=json.load(open(f'_scratch/stage2/read/lifeKP_s{s}.json')); led=d['ledger']
    A=collections.defaultdict(dict)
    for p in d['probes']:
        if p.get('line')!='trunk' or p.get('tag') or not 5<=p['night']<=15: continue
        for i,r in enumerate(p['rows']):
            it=r.get('item')
            if it is None or it not in led or not led[it].get('telling'): continue
            A[p['arm']][(p['night'],i,it,r['you'])]=r
    I,O,F=A['intact'],A['hippocampus-off'],A['intact-frozen']
    for k,r in I.items():
        if r.get('ok') is None: continue
        o=O.get(k)
        if o is None or o.get('ok') is None: tot['unpaired']+=1; continue
        tot['n']+=1; tot['intact_right']+=bool(r['ok']); tot['off_right']+=bool(o['ok'])
        cls='hurt' if (not r['ok'] and o['ok']) else 'helped' if (r['ok'] and not o['ok']) else 'same'
        tot[cls]+=1
        if cls!='same': rows.append(dict(seed=s,cls=cls,r=r,f=F.get(k)))
print(dict(tot))
for c in ('hurt','helped'):
    R=[x for x in rows if x['cls']==c]
    print(c,len(R),'per seed',[sum(x['seed']==s for x in R) for s in (319,320,321)],
      'right_ep',sum(bool(x['r'].get('fetched')) and x['r'].get('right_episode') is True for x in R),
      'wrong',sum(bool(x['r'].get('fetched')) and x['r'].get('right_episode') is not True for x in R),
      'nofetch',sum(not x['r'].get('fetched') for x in R),
      'twin',sum(x['f'] is not None and 'fetch' in x['f'] for x in R),
      'same_recalled',sum(x['f'] is not None and x['f'].get('recalled')==x['r'].get('recalled') for x in R))
for fld in ('sim_raw','margin_raw','margin_w'):
    out=[]
    for seeds in ([319],[320],[321],[319,320,321]):
        R=[x for x in rows if x['seed'] in seeds and x['f'] and 'fetch' in x['f'] and x['f']['fetch'].get(fld) is not None]
        out.append(round(auroc([x['f']['fetch'][fld] for x in R if x['cls']=='hurt'],[x['f']['fetch'][fld] for x in R if x['cls']=='helped']),2))
    print(fld,out)
