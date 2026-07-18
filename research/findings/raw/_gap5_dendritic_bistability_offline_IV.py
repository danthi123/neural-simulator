"""Gap #5 dendritic bistability — OFFLINE I-V test (no spiking, seconds). Confirms the kernel's Jahr-Stevens Mg-block
+ a KIR load line gives a robust 3-fixed-point (bistable) regime, while a linear leak does not (Sanders 2013 'perfect
couple'). Reproducible bound on the whole dendritic-bistability arc before any GPU/kernel work.
"""
import numpy as np

def mg_block(v, mg=1.0):  # Jahr-Stevens (sim/kernels.py:275), Mg in mM, v in mV
    return 1.0 / (1.0 + (mg / 3.57) * np.exp(-0.062 * v))

E_e, E_K = 0.0, -90.0
_v = np.linspace(-95, 5, 6000)

def _fps(f):
    s = np.sign(f); cr = np.where(np.diff(s) != 0)[0]
    return [round(0.5 * (_v[i] + _v[i + 1]), 1) for i in cr]

def linear(g_res, g_L, V_L=-65.0):
    return _fps(g_res * mg_block(_v) * (E_e - _v) + g_L * (V_L - _v))

def kir(g_res, gK, v_kir=-50.0, k=8.0):
    gk = gK / (1.0 + np.exp((_v - v_kir) / k))
    return _fps(g_res * mg_block(_v) * (E_e - _v) + gk * (E_K - _v))

def _reg(p):
    if len(p) >= 3: return "BISTABLE"
    if len(p) == 1 and p[0] < -40: return "mono-DOWN"
    if len(p) == 1: return "mono-UP"
    return f"{len(p)}fp"

if __name__ == "__main__":
    print("LINEAR leak (no robust bistable band):")
    for r in (0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        p = linear(r, 1.0); print(f"  g_res/g_L={r:4.1f}: {p} {_reg(p)}")
    print("KIR load line (robust bistable band):")
    for gK in (3.0, 5.0):
        for r in (2.0, 4.0, 6.0, 8.0, 10.0, 14.0):
            p = kir(r, gK); print(f"  g_res={r:4.1f} gK={gK:3.1f}: {p} {_reg(p)}")
