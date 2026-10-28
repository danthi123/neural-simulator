import numpy as np
TAU=15.0; ER=-65.0; SR=2.0; VH=-35.0; VHK=0.2; EE=0.0; KG=1.0; EK=-90.0; KVH=-50.0; KK=8.0; EI=-75.0
def terms(v, D, sr=SR, vh=VH, vhk=VHK, kg=KG):
    leak=-(v-ER)
    regen=sr/(1+np.exp(-vhk*(v-vh)))*(EE-v)
    kir=kg/(1+np.exp((v-KVH)/KK))*(EK-v)
    return leak, D, regen, kir
def step(v, D, g, sr=SR, vh=VH, vhk=VHK, kg=KG, f32=True):
    l,d,r,k=terms(v,D,sr,vh,vhk,kg)
    dvb=l+d+r+k
    gs=(EI-v)*(1-np.exp(np.clip(-g/TAU,-80,0)))
    return v+(1.0/TAU)*dvb+gs
def replay(v0, D, g, **kw):
    v=np.empty(len(D)); x=v0
    for t in range(len(D)):
        x=step(x,D[t],g[t],**kw); v[t]=x
    return v
def f_net(v, sr=SR, vh=VH, vhk=VHK, kg=KG):
    l,d,r,k=terms(v,0.0,sr,vh,vhk,kg); return l+r+k
