from load import *; from cf import *; from ode import *
import numpy as np, collections
L=recs('lesion_'); C=recs('cal_')
print('LESION c1 (plateau held, SOM output gated): SOM spikes per record in [200,800) i.e. held-trace self-recruitment')
by=collections.defaultdict(list)
for r in L: by[r['cell']].append(int(r['c1']['som'][200:800].sum()))
for k,v in by.items(): print('  %-14s'%k, v, 'pool Hz/cell max %.2f'%(max(v)/8/0.6))
print('LESION c1: SOM spikes [0,150) / [150,200) mean:', {k: (round(np.mean([r['c1']['som'][:150].sum() for r in L if r['cell']==k]),1), round(np.mean([r['c1']['som'][150:200].sum() for r in L if r['cell']==k]),1)) for k in by})
print('TRACE c1: same:', {k: (round(np.mean([r['c1']['som'][:150].sum() for r in C if r['cell']==k]),1), round(np.mean([r['c1']['som'][150:200].sum() for r in C if r['cell']==k]),1)) for k in by})
# c3: SOM spikes during the newer volley [300,450) and after [450,600) at corner, TRACE and LESION
for nm,RR in (('TRACE',C),('LESION',L)):
    a=[(int(p['som'][300:450].sum()),int(p['som'][450:600].sum()),int(p['som'][150:300].sum())) for r in RR if r['cell']=='sa0p125_ms215' for p in r['pairs']]
    a=np.array(a); print('%s corner c3 SOM pool spikes: X-hold gap [150,300) mean %.2f | Y volley [300,450) mean %.1f | after [450,600) mean %.1f'%(nm,a[:,2].mean(),a[:,0].mean(),a[:,1].mean()))
