from load import *; from cf import *; from ode import *
import numpy as np, collections
DEC=np.float32(np.exp(-1/10.0))
rs=recs('cal_')
def split(g, edge):
    inc=np.zeros_like(g); inc[0]=g[0]; inc[1:]=g[1:]-g[:-1]*DEC
    inc[inc<1e-6]=0
    gv=np.zeros_like(g); gp=np.zeros_like(g); a=b=0.0
    for t in range(len(g)):
        a=a*DEC+(inc[t] if t<=edge else 0); b=b*DEC+(inc[t] if t>edge else 0)
        gv[t]=a; gp[t]=b
    return gv,gp,inc
fails=[r for r in rs if r['c6']=='FAIL']
early=[r for r in fails if (r['c1']['v'][150:]<VH).argmax()+150<200]
late=[r for r in fails if r not in early]
print('fails',len(fails),'early(<200ms)',len(early),'late',len(late), 'late times',[int((r['c1']['v'][150:]<VH).argmax()+150) for r in late])
cnt=collections.Counter()
rows=[]
for r in fails:
    v=r['c1']['v']; g=r['c1']['g']; T=len(v)
    tf=int((v[150:]<VH).argmax()+150)
    gv,gp,inc=split(g,150)
    D=c1_drive(T)
    sv=c6_pass(replay(-77.28,D,gv)); sp=c6_pass(replay(-77.28,D,gp))
    # remove in-volley only but keep in-volley carry? cf: zero g before edge entirely but keep post as recorded
    g2=g.copy(); g2[:150]=0; s2=c6_pass(replay(-77.28,D,g2))
    cnt['falls_with_volley_SOM_only']+= (not sv); cnt['falls_with_postedge_SOM_only']+=(not sp); cnt['falls_with_g0_before_edge_recorded_after']+=(not s2)
    # terms at edge and at crossing
    l,d,rg,k=terms(v[149],0)
    gaba149=g[150]*(EI-v[149])
    share_v = gv[150:tf+1].sum()/max(g[150:tf+1].sum(),1e-9)
    rows.append((r['name'],tf,round(float(v[149]),1),round(float(g[149]),3),round(float(rg),1),round(float(gaba149),1),round(float(share_v),2),round(float(g[150:tf+1].max()),2)))
print(cnt)
print('name tfall v149 g149 regen@v149(undriven) GABA@150 share_of_g_from_volley_spikes gmax[150,tf]')
for x in rows: print(*x)
sv=np.array([x[6] for x in rows if x[1]<200]); print('early fails: share of post-edge g_ap integral (to fall) from in-volley SOM spikes: median %.2f min %.2f'%(np.median(sv),sv.min()))
