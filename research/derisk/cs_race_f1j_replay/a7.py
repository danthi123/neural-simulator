from load import *; from cf import *; from ode import *
import numpy as np, collections
rs=recs('cal_')
DEC=np.float32(np.exp(-1/10.0))
fails=[r for r in rs if r['c6']=='FAIL']
early=[r for r in fails if (r['c1']['v'][150:]<VH).argmax()+150<200]
tf=np.array([int((r['c1']['v'][150:]<VH).argmax()+150) for r in early]); print('early self-termination: ms after edge min %d med %d max %d'%(tf.min()-150,np.median(tf)-150,tf.max()-150))
# older terminations: time since older drive edge
terms_=[]; bf=[]
for r in rs:
    for p in r['pairs']:
        o=c7(p['vX'],p['vY'])
        if o['d0']: terms_.append(o['term']-150)
        if o['d0'] and not o['b']:
            bf.append(int((p['vY'][450:]<VH).argmax()+450)-450)
terms_=np.array(terms_); print('older termination: ms after the OLDER word drive edge: min %d med %d max %d (n=%d)'%(terms_.min(),np.median(terms_),terms_.max(),len(terms_)))
print('C7(b) fails (newer dies in c3): ms after its own edge:',sorted(bf))
# SOM histogram in c1 by 25ms bins
bins=list(range(0,275,25))
def hist(rr):
    H=np.array([[r['c1']['som'][b:b+25].sum() for b in bins[:-1]] for r in rr]); return H.mean(0)
passes=[r for r in rs if r['c6']=='PASS']
print('bins',bins[:-1]); print('SOM spikes/bin mean, early fails',np.round(hist(early),2)); print('SOM spikes/bin mean, C6 passes',np.round(hist(passes),2))
print('SOM pool rate Hz per cell in [100,150) vs [50,100): fails %.0f vs %.0f'%(hist(early)[4]/8/0.025, hist(early)[2]/8/0.025))
# term table for a representative: corner seed 50 c3 frog as X (UNDEFINED pair) and one c1 fail
def table(v,g,D,ts):
    for t in ts:
        l,d,rg,k=terms(v[t-1],D[t]); gc=g[t]*(EI-v[t-1])
        print('  t=%3d v=%6.1f g_ap=%.3f | drive %5.1f regen %6.1f leak %6.1f kir %5.1f GABA %6.1f (mV-units per tau)'%(t,v[t],g[t],d,rg,l,k,gc))
for r in rs:
    if r['name']=='sa0p125_ms215_s50':
        for p in r['pairs']:
            if p['X']=='frog':
                print('corner s50 frog>vole, X slot:'); T=len(p['vX']); DX,_=c3_drives(T); table(p['vX'],p['gX'],DX,[100,140,149,150,155,160,165,170,175,178,180])
                print('  SOM spikes per ms 130-180:',p['som'][130:180].tolist())
    if r['name']=='sa0p25_ms304_s46':
        print('c1 sa0p25_ms304_s46 (0 SOM spikes after edge):'); T=len(r['c1']['v']); D=c1_drive(T); table(r['c1']['v'],r['c1']['g'],D,[60,100,140,149,150,152,155,160,165,170,175])
