from load import *; from cf import *; from ode import *
import numpy as np
exec(open('a9.py').read().split("print('--- tail amplitude")[0])
print('--- drop SOM increments only inside (t0, 200] in c1 (late-volley + post-edge window), keep the rest')
for t0 in (150,140,125,100):
    score(lambda r,t0=t0: rebuild(incs(r['c1']['g']),10.0,~((np.arange(len(r['c1']['g']))>t0)&(np.arange(len(r['c1']['g']))<=200))), lambda r,p:(p['gX'],p['gY']), label='c1 drop SOM spikes in (%d,200]'%t0)
print('--- critical static g_ap (saddle-node) vs self_regen')
vs=np.linspace(-90,10,400001)
for sr in (2.0,2.5,3.0,4.0,6.0):
    fn=f_net(vs,sr=sr); sc=np.flatnonzero(np.sign(fn[:-1])!=np.sign(fn[1:])); sep=vs[sc[1]]; pl=vs[sc[-1]]
    m=(vs>sep)&(vs<pl); print('  sr %.1f: separatrix %.2f plateau %.2f max margin %.2f g_crit %.3f'%(sr,sep,pl,fn[m].max(),(fn[m]/(vs[m]-EI)).max()))
print('--- ligand-gated regen: self_regen(t)=2 + dS*exp(-(t-edge)/tau) after each own drive edge')
def lg_replay(v0,D,g,dS,tau,edge):
    v=np.empty(len(D)); x=v0
    for t in range(len(D)):
        sr=2.0+(dS*np.exp(-(t-edge)/tau) if t>edge else 0.0)
        x=step(x,D[t],g[t],sr=sr); v[t]=x
    return v
for tau in (30,50,75):
  for dS in (1.0,2.0,4.0):
    c6=e=0
    for r in rs:
        T=len(r['c1']['v']); ok=c6_pass(lg_replay(-77.28,c1_drive(T),r['c1']['g'],dS,tau,149)); c6+=ok; e+=ok and r['name'] in early
    nd=na=nb=cd=ca=0
    for r in rs:
        for p in r['pairs']:
            T=len(p['vX']); DX,DY=c3_drives(T)
            o=c7(lg_replay(-77.28,DX,p['gX'],dS,tau,149),lg_replay(-77.28,DY,p['gY'],dS,tau,449))
            if o['d0']:
                nd+=1; na+=o['a']; nb+=o['b']
                if r['cell']=='sa0p125_ms215': cd+=1; ca+=o['a']
    print('  dS %.0f tau %d: C6 %2d/72 early-rescued %2d/45 | C7 def %3d (a) %3d (b) %3d | corner (a) %d/%d'%(dS,tau,c6,e,nd,na,nb,ca,cd))
