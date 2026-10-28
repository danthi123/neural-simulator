from load import *; from cf import *
import numpy as np
rs=recs('cal_')
pk=[];pkc=[];gterm=[]
for r in rs:
    for p in r['pairs']:
        o=c7(p['vX'],p['vY'])
        if o['d0']:
            g=p['gX'][300:o['term']+1]; pk.append(g.max()); gterm.append(p['gX'][o['term']])
            if r['cell']=='sa0p125_ms215': pkc.append(g.max())
print('older g_ap max over [Y_on, termination]: grid %.2f-%.2f (median %.2f, n=%d); corner %.2f-%.2f'%(min(pk),max(pk),np.median(pk),len(pk),min(pkc),max(pkc)))
