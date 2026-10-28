from load import *
import numpy as np
C={r['name']:r for r in recs('cal_')}; L={r['name']:r for r in recs('lesion_')}
d1=[];d3=[]
for k in C:
    a=C[k]; b=L[k]
    d1.append((int(a['c1']['som'][:200].sum()),int(b['c1']['som'][:200].sum())))
    for p,q in zip(a['pairs'],b['pairs']):
        d3.append((int(p['som'][300:450].sum()),int(q['som'][300:450].sum())))
d1=np.array(d1); d3=np.array(d3)
print('c1 SOM spikes [0,200): TRACE mean %.2f LESION mean %.2f, mean|diff| %.2f, max|diff| %d, corr %.3f'%(d1[:,0].mean(),d1[:,1].mean(),np.abs(d1[:,0]-d1[:,1]).mean(),np.abs(d1[:,0]-d1[:,1]).max(),np.corrcoef(d1.T)[0,1]))
print('c3 SOM spikes in newer volley [300,450): TRACE mean %.2f LESION mean %.2f, mean|diff| %.2f, max|diff| %d'%(d3[:,0].mean(),d3[:,1].mean(),np.abs(d3[:,0]-d3[:,1]).mean(),np.abs(d3[:,0]-d3[:,1]).max()))
