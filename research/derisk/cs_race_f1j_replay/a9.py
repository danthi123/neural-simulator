from load import *; from cf import *; from ode import *
import numpy as np
rs=recs('cal_')
DEC=np.float32(np.exp(-1/10.0))
early=set(r['name'] for r in rs if r['c6']=='FAIL' and (r['c1']['v'][150:]<VH).argmax()+150<200)
def incs(g):
    inc=np.zeros_like(g); inc[0]=g[0]; inc[1:]=g[1:]-g[:-1]*DEC; inc[inc<1e-6]=0; return inc
def rebuild(inc,tau=10.0,mask=None):
    d=np.exp(-1/tau); out=np.zeros_like(inc); a=0.0
    for t in range(len(inc)):
        a=a*d+(inc[t] if (mask is None or mask[t]) else 0); out[t]=a
    return out
def score(c1g, c3g, D1=None, D3=None, label='', step_kw={}):
    c6=e=0
    for r in rs:
        T=len(r['c1']['v']); D=c1_drive(T) if D1 is None else D1(T)
        ok=c6_pass(replay(-77.28,D,c1g(r),**step_kw)); c6+=ok; e+=ok and r['name'] in early
    nd=na=nb=cd=ca=0
    for r in rs:
        for p in r['pairs']:
            T=len(p['vX']); DX,DY=c3_drives(T) if D3 is None else D3(T)
            gX,gY=c3g(r,p)
            o=c7(replay(-77.28,DX,gX,**step_kw),replay(-77.28,DY,gY,**step_kw))
            if o['d0']:
                nd+=1; na+=o['a']; nb+=o['b']
                if r['cell']=='sa0p125_ms215': cd+=1; ca+=o['a']
    print('%-44s C6 %2d/72 early-rescued %2d/45 | C7 def %3d (a) %3d (b) %3d | corner (a) %d/%d'%(label,c6,e,nd,na,nb,ca,cd))
print('--- tail amplitude x tau (current tail starting at the edge)')
for tau in (20,30,50,75):
    for amp in (15,30,60,120,240):
        score(lambda r: r['c1']['g'], lambda r,p:(p['gX'],p['gY']), D1=lambda T,tau=tau,amp=amp: c1_drive(T,tau,amp), D3=lambda T,tau=tau,amp=amp: c3_drives(T,tau,amp), label='tail tau %d amp %d'%(tau,amp))
print('--- NMDA-like conductance tail g_N(t)=gN0*exp(-(t-149)/tauN), Mg block (1 mM), E=0, from the edge')
def nmda_replay(v0,D,g,gN0,tauN,edges):
    v=np.empty(len(D)); x=v0
    for t in range(len(D)):
        gn=sum(gN0*np.exp(-(t-e)/tauN) for e in edges if t>e)
        B=1.0/(1.0+np.exp(-0.062*x)/3.57)
        x=step(x,D[t]+gn*B*(0.0-x),g[t]); v[t]=x
    return v
for tauN in (30,50,75,100):
    for gN0 in (1.0,2.0,4.0):
        c6=e=0
        for r in rs:
            T=len(r['c1']['v']); ok=c6_pass(nmda_replay(-77.28,c1_drive(T),r['c1']['g'],gN0,tauN,[149])); c6+=ok; e+=ok and r['name'] in early
        nd=na=nb=cd=ca=0
        for r in rs:
            for p in r['pairs']:
                T=len(p['vX']); DX,DY=c3_drives(T)
                o=c7(nmda_replay(-77.28,DX,p['gX'],gN0,tauN,[149]),nmda_replay(-77.28,DY,p['gY'],gN0,tauN,[449]))
                if o['d0']:
                    nd+=1; na+=o['a']; nb+=o['b']
                    if r['cell']=='sa0p125_ms215': cd+=1; ca+=o['a']
        print('NMDA tauN %3d gN0 %.1f (g_L units)            C6 %2d/72 early-rescued %2d/45 | C7 def %3d (a) %3d (b) %3d | corner (a) %d/%d'%(tauN,gN0,c6,e,nd,na,nb,ca,cd))
print('--- apical IPSC decay constant (rebuilt from recorded increments)')
for tg in (5.0,10.0,20.0):
    score(lambda r,tg=tg: rebuild(incs(r['c1']['g']),tg), lambda r,p,tg=tg:(rebuild(incs(p['gX']),tg),rebuild(incs(p['gY']),tg)), label='IPSC decay tau %g ms'%tg)
print('--- SOM output timing: drop SOM increments after t0 in the newer volley (c1) / keep c3 as recorded')
for t0 in (150,125,100):
    score(lambda r,t0=t0: rebuild(incs(r['c1']['g']),10.0,np.arange(len(r['c1']['g']))<=t0), lambda r,p:(p['gX'],p['gY']), label='c1 SOM spikes only up to t=%d'%t0)
