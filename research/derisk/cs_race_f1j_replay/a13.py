src=open('a9.py').read()
exec(src.split("print('--- tail amplitude")[0])
exec(src[src.index('def nmda_replay'):src.index('for tauN in (30')])
for tauN in (30,50,75):
    for gN0 in (8.0,16.0):
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
        print('NMDA tauN %3d gN0 %4.1f g_L: C6 %2d/72 early-rescued %2d/45 | C7 def %3d (a) %3d (b) %3d | corner (a) %d/%d'%(tauN,gN0,c6,e,nd,na,nb,ca,cd))
