from load import *; from cf import *; from ode import *
import numpy as np, collections
rs=recs('cal_')
fails=[r for r in rs if r['c6']=='FAIL']
early=set(r['name'] for r in fails if (r['c1']['v'][150:]<VH).argmax()+150<200)
corner=[r for r in rs if r['cell']=='sa0p125_ms215']
def evaluate(c1fn, c3fn, label):
    c6p=0; c6e=0; c6l=0
    for r in rs:
        T=len(r['c1']['v']); v=c1fn(r,T)
        ok=c6_pass(v); c6p+=ok
        if r['name'] in early: c6e+=ok
        elif r['c6']=='FAIL': c6l+=ok
    nd=na=nb=0; cd=ca=cb=0; ab=0; cab=0
    for r in rs:
        for p in r['pairs']:
            vX,vY=c3fn(r,p)
            o=c7(vX,vY)
            if o['d0']:
                nd+=1; na+=o['a']; nb+=o['b']; ab+=(o['a'] and o['b'])
                if r['cell']=='sa0p125_ms215': cd+=1; ca+=o['a']; cb+=o['b']; cab+=(o['a'] and o['b'])
    print('%-38s C6 %2d/72 (early-fail rescued %2d/45, late %d/6) | C7 grid: def %2d (a) %2d (b) %2d a&b %2d | corner def %d (a) %d (b) %d'%(label,c6p,c6e,c6l,nd,na,nb,ab,cd,ca,cb))
# baseline
evaluate(lambda r,T: replay(-77.28,c1_drive(T),r['c1']['g']),
         lambda r,p: tuple(replay(-77.28,D,g) for D,g in zip(c3_drives(len(p['vX'])),(p['gX'],p['gY']))), 'baseline (box-car)')
print('--- A: drive offset kinetics (box-car -> exponential tail, amp 120 at the edge)')
for tau in (2,5,10,15,20,30,50,100):
    evaluate(lambda r,T,tau=tau: replay(-77.28,c1_drive(T,tau),r['c1']['g']),
             lambda r,p,tau=tau: tuple(replay(-77.28,D,g) for D,g in zip(c3_drives(len(p['vX']),tau),(p['gX'],p['gY']))), 'tail tau=%g ms'%tau)
print('--- B: regenerative plateau conductance (coincidence_plateau_self_regen)')
for sr in (2.0,2.5,3.0,4.0,6.0,8.0,12.0):
    evaluate(lambda r,T,sr=sr: replay(-77.28,c1_drive(T),r['c1']['g'],sr=sr),
             lambda r,p,sr=sr: tuple(replay(-77.28,D,g,sr=sr) for D,g in zip(c3_drives(len(p['vX'])),(p['gX'],p['gY']))), 'self_regen=%g'%sr)
print('--- B2: KIR down-state conductance')
for kg in (1.0,0.5,0.25,0.0):
    evaluate(lambda r,T,kg=kg: replay(-77.28,c1_drive(T),r['c1']['g'],kg=kg),
             lambda r,p,kg=kg: tuple(replay(-77.28,D,g,kg=kg) for D,g in zip(c3_drives(len(p['vX'])),(p['gX'],p['gY']))), 'apical_kir_g=%g'%kg)
print('--- C: SOM lateral-only (no self g_ap; older keeps the recorded g_ap)')
evaluate(lambda r,T: replay(-77.28,c1_drive(T),np.zeros(T)),
         lambda r,p: (replay(-77.28,c3_drives(len(p['vX']))[0],p['gX']), replay(-77.28,c3_drives(len(p['vX']))[1],np.zeros(len(p['vX'])))), 'lateral-only (newer g=0)')
print('--- D: VIP silences SOM during the attended word volley (+k ms)')
DEC=np.float32(np.exp(-1/10.0))
def suppress(g,t0,t1):
    inc=np.zeros_like(g); inc[0]=g[0]; inc[1:]=g[1:]-g[:-1]*DEC; inc[inc<1e-6]=0
    inc[t0:t1+1]=0  # spikes arriving in [t0,t1] removed
    out=np.zeros_like(g); a=0.0
    for t in range(len(g)): a=a*DEC+inc[t]; out[t]=a
    return out
for ext in (0,50,100):
    evaluate(lambda r,T,ext=ext: replay(-77.28,c1_drive(T),suppress(r['c1']['g'],0,150+ext)),
             lambda r,p,ext=ext: (replay(-77.28,c3_drives(len(p['vX']))[0],suppress(p['gX'],300,450+ext)),
                                  replay(-77.28,c3_drives(len(p['vX']))[1],suppress(p['gY'],300,450+ext))), 'VIP: SOM off during volley+%d ms'%ext)
