import numpy as np
from ode import *
VH=-35.0
def c1_drive(T, tail_tau=0.0, tail_amp=120.0):
    D=np.zeros(T); D[:150]=120.0
    if tail_tau>0:
        t=np.arange(150,T); D[150:]=tail_amp*np.exp(-(t-149)/tail_tau)
    return D
def c3_drives(T, tail_tau=0.0, tail_amp=120.0):
    DX=np.zeros(T); DX[:150]=120; DY=np.zeros(T); DY[300:450]=120
    if tail_tau>0:
        t=np.arange(150,T); DX[150:]=tail_amp*np.exp(-(t-149)/tail_tau)
        t=np.arange(450,T); DY[450:]=tail_amp*np.exp(-(t-449)/tail_tau)
    return DX,DY
def c6_pass(v): return bool(v[150:800].min()>VH)
def c7(vX,vY):
    d0=bool(vX[150:300].min()>VH)
    if not d0: return dict(d0=False)
    below=vX<VH
    term=None
    # first t>=300 such that below[t:950] all True
    for t in range(300,950):
        if below[t:950].all(): term=t; break
    a = term is not None and term-300<=300
    b = bool(vY[450:950].min()>VH)
    return dict(d0=True,a=a,b=b,term=term)
