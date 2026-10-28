from load import *; from cf import *; from ode import *
import collections
rs=recs('cal_')
# baseline check
mm=0
for r in rs:
    T=len(r['c1']['v']); v=replay(-77.28,c1_drive(T),r['c1']['g'])
    if c6_pass(v)!=(r['c6']=='PASS'): mm+=1
print('baseline C6 replay mismatches:',mm)
nd=na=nb=0
for r in rs:
    for p in r['pairs']:
        T=len(p['vX']); DX,DY=c3_drives(T)
        o=c7(replay(-77.28,DX,p['gX']),replay(-77.28,DY,p['gY']))
        if o['d0']: nd+=1; na+=o['a']; nb+=o['b']
print('baseline C7 over grid: defined',nd,'(a)',na,'(b)',nb)
