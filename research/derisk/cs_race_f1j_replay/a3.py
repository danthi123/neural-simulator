import json,glob,os,numpy as np
from ode import *
R='raw/research/findings/raw/_cs_race/f1i/step0/'
errs=[]
for f in sorted(glob.glob(R+'cal_*.json'))+sorted(glob.glob(R+'lesion_*.json')):
    if f.endswith('prov.json'): continue
    d=json.load(open(f)); ps=d['per_seed'][0]
    c1=ps['c1']; s=c1['som']; w=c1['v_apical']['pulsed_words'][0]
    pw=s['per_word_ms'][w]; v=np.array(pw['v_apical_min']); g=np.array(pw['g_ap_max'])
    T=len(v); D=np.zeros(T); D[:150]=120.0
    vr=replay(-77.28,D,g)
    errs.append((os.path.basename(f), np.abs(vr-v).max(), np.abs(vr[1:]-v[1:]).max()))
e=np.array([x[1] for x in errs]); print('c1 replay max|err| over %d records: max %.4f median %.4f'%(len(e),e.max(),np.median(e)))
print(sorted(errs,key=lambda x:-x[1])[:3])
# c3 pairs
errs=[]
for f in sorted(glob.glob(R+'cal_*.json')):
    if f.endswith('prov.json'): continue
    d=json.load(open(f)); ps=d['per_seed'][0]
    for c3 in ps['c3']:
        s=c3['som']; ws=c3['v_apical']['pulsed_words']
        for j,w in enumerate(ws):
            pw=s['per_word_ms'][w]; v=np.array(pw['v_apical_min']); g=np.array(pw['g_ap_max'])
            T=len(v); D=np.zeros(T)
            if j==0: D[:150]=120
            else: D[300:450]=120
            vr=replay(-77.28,D,g); errs.append(np.abs(vr-v).max())
print('c3 replay max err',max(errs), 'n',len(errs))
