# f1j replay core: vectorized (over records) version of ode.step, plus two candidate companion processes.
# Reference model (ode.py) matches the recorded v_apical within 0.0035 mV on 144 c1 records and 432 c3 slot traces.
# CPU numpy only; reads the f1i step-0 raw records via load.py; writes nothing outside this folder.
import numpy as np
from load import recs
from ode import TAU, ER, SR, VH, VHK, EE, KG, EK, KVH, KK, EI

DEC10 = np.float32(np.exp(-1 / 10.0))  # recorded syn_tau_g_i = 10 ms (sim/config.py:428)
MG = 1.0                                # mM, Jahr-Stevens
V0 = -77.28                             # undriven down state (a2.py)


def mg_block(v, mg=MG):
    return 1.0 / (1.0 + (mg / 3.57) * np.exp(-0.062 * v))


def sig(v):
    return 1.0 / (1.0 + np.exp(-VHK * (v - VH)))


def incs(g):
    """Per-ms SOM->apical conductance increments recovered from the recorded g_ap (decay 10 ms). Exact to 2e-6."""
    inc = np.zeros_like(g)
    inc[..., 0] = g[..., 0]
    inc[..., 1:] = g[..., 1:] - g[..., :-1] * DEC10
    inc[inc < 1e-6] = 0
    return inc


def rebuild(g, tau):
    """Same SOM spike increments, IPSC decay tau (syn_tau_g_i sensitivity)."""
    if tau == 10.0:
        return g
    inc = incs(g)
    d = np.exp(-1.0 / tau)
    out = np.zeros_like(inc)
    a = np.zeros(inc.shape[:-1])
    for t in range(inc.shape[-1]):
        a = a * d + inc[..., t]
        out[..., t] = a
    return out


def replay(D, U, G, *, regen='bdsp', sr=SR,
           nmda=None,            # dict(gN0, tauN, kin='sat'|'lin', alpha=0.5, integ='bounded'|'explicit')
           res=None):            # dict(A=R*self_regen per-step amplitude, tau=80.0) reservoir regen in place of BDSP regen
    """D: (N,T) apical drive (mV units), U: (N,T) sensory-input-on mask (drives NMDA gating), G: (N,T) g_ap.
    Returns v (N,T) and the NMDA conductance trace (N,T) (zeros if none)."""
    N, T = D.shape
    v = np.full(N, V0)
    V = np.empty((N, T))
    GN = np.zeros((N, T))
    s = np.zeros(N)
    if nmda is not None:
        dN = np.exp(-1.0 / nmda['tauN'])
        alpha = nmda.get('alpha', 0.5)
        kin = nmda.get('kin', 'sat')
        if kin == 'sat':
            s_ss = alpha / (1 - dN + alpha * dN)          # gating reached under sustained input
            scale = nmda['gN0'] / s_ss                      # so gN at the drive edge == gN0
        else:                                               # linear integrating synapse, edge value == gN0
            k = nmda['gN0'] * (1 - dN) / (1 - dN ** 150)
    if res is not None:
        dR = np.exp(-1.0 / res.get('tau', 80.0))
        # start the reservoir at its rest steady state
        gres = np.full(N, res['A'] * mg_block(V0) * sig(V0) / (1 - dR))
    for t in range(T):
        leak = -(v - ER)
        if regen == 'bdsp':
            rg = sr * sig(v) * (EE - v)
        elif regen == 'reservoir':
            gres = gres * dR + res['A'] * mg_block(v) * sig(v)       # fused_coincidence_plateau sustain (g_inc = 0)
            rg = gres * mg_block(v) * (EE - v)                       # I = g_eff * mg * (E_e - v), scale folded into A
        else:
            rg = 0.0
        kir = KG / (1 + np.exp((v - KVH) / KK)) * (EK - v)
        dvb = leak + D[:, t] + rg + kir
        gs = (EI - v) * (1 - np.exp(np.clip(-G[:, t] / TAU, -80, 0)))
        ns = 0.0
        if nmda is not None:
            if kin == 'sat':
                s = s * dN
                s = s + alpha * U[:, t] * (1 - s)
                gn = scale * s
            else:
                s = s * dN + k * U[:, t]
                gn = s
            GN[:, t] = gn
            geff = gn * mg_block(v)
            if nmda.get('integ', 'bounded') == 'bounded':
                ns = (EE - v) * (1 - np.exp(np.clip(-geff / TAU, -80, 0)))
            else:
                dvb = dvb + geff * (EE - v)
        v = v + (1.0 / TAU) * dvb + gs + ns
        V[:, t] = v
    return V, GN


class Data:
    def __init__(self):
        self.C = recs('cal_')
        self.L = {r['name']: r for r in recs('lesion_')}
        C = self.C
        self.names = [r['name'] for r in C]
        self.cell = np.array([r['cell'] for r in C])
        self.c6 = np.array([r['c6'] for r in C])
        self.v1 = np.stack([r['c1']['v'] for r in C])
        self.g1 = np.stack([r['c1']['g'] for r in C])
        T1 = self.v1.shape[1]
        self.D1 = np.zeros((len(C), T1)); self.D1[:, :150] = 120.0
        self.U1 = np.zeros((len(C), T1)); self.U1[:, :150] = 1.0
        self.vL1 = np.stack([self.L[n]['c1']['v'] for n in self.names])     # LESION (SOM output gated) c1 trace
        tf = np.array([int((v[150:] < VH).argmax() + 150) if (v[150:800] < VH).any() else -1 for v in self.v1])
        self.fail = self.c6 == 'FAIL'
        self.early = self.fail & (tf < 200)
        self.late = self.fail & (tf >= 200)
        self.pas = self.c6 == 'PASS'
        P = [(r, p) for r in C for p in r['pairs']]
        self.pcell = np.array([r['cell'] for r, p in P])
        self.vX = np.stack([p['vX'] for r, p in P]); self.vY = np.stack([p['vY'] for r, p in P])
        self.gX = np.stack([p['gX'] for r, p in P]); self.gY = np.stack([p['gY'] for r, p in P])
        T3 = self.vX.shape[1]
        self.DX = np.zeros((len(P), T3)); self.DX[:, :150] = 120.0
        self.DY = np.zeros((len(P), T3)); self.DY[:, 300:450] = 120.0
        self.UX = (self.DX > 0).astype(float); self.UY = (self.DY > 0).astype(float)


def c6_pass(V):
    return V[:, 150:800].min(1) > VH


def c7(VX, VY):
    d0 = VX[:, 150:300].min(1) > VH
    above = VX[:, 300:950] >= VH
    anyab = above.any(1)
    last = np.where(anyab, above.shape[1] - 1 - np.argmax(above[:, ::-1], 1), -1)
    term = np.where(anyab, 300 + last + 1, 300)
    valid = ~above[:, -1]
    a = d0 & valid & (term - 300 <= 300)
    b = d0 & (VY[:, 450:950].min(1) > VH)
    return d0, a, b, np.where(valid, term, -1)


SA125 = ('sa0p125_ms215', 'sa0p125_ms304', 'sa0p125_ms429')
CORNER = 'sa0p125_ms215'


def evaluate(dat, *, tau_gi=10.0, **kw):
    g1 = rebuild(dat.g1, tau_gi); gX = rebuild(dat.gX, tau_gi); gY = rebuild(dat.gY, tau_gi)
    V1, GN1 = replay(dat.D1, dat.U1, g1, **kw)
    VX, GNX = replay(dat.DX, dat.UX, gX, **kw)
    VY, GNY = replay(dat.DY, dat.UY, gY, **kw)
    p6 = c6_pass(V1)
    d0, a, b, term = c7(VX, VY)
    out = dict(C6=int(p6.sum()), early=int((p6 & dat.early).sum()), late=int((p6 & dat.late).sum()),
               kept=int((p6 & dat.pas).sum()),
               c7def=int(d0.sum()), c7a=int(a.sum()), c7b=int(b.sum()), c7ab=int((a & b).sum()))
    cm = dat.pcell == CORNER
    out.update(cdef=int(d0[cm].sum()), ca=int(a[cm].sum()), cb=int(b[cm].sum()), cab=int((a & b)[cm].sum()))
    for c in SA125:
        m = dat.cell == c; pm = dat.pcell == c
        out[c] = dict(c6=int(p6[m].sum()), early=int((p6 & dat.early)[m].sum()), n_early=int(dat.early[m].sum()),
                      late=int((p6 & dat.late)[m].sum()), n_late=int(dat.late[m].sum()),
                      d=int(d0[pm].sum()), a=int(a[pm].sum()), b=int(b[pm].sum()), ab=int((a & b)[pm].sum()))
    # closed-loop exposure (soma coupling = soma_g 40 * v_apical_scale 0.05 * max(v - E_rest, 0) pA = 2 pA/mV)
    cpl = lambda v: 2.0 * np.maximum(v - ER, 0)
    out['vol_dcpl_pA'] = float((cpl(V1[:, :150]) - cpl(dat.v1[:, :150])).mean(1).mean())          # mean over volley
    out['vol_dcpl_min_pA'] = float((cpl(V1[:, :150]) - cpl(dat.v1[:, :150])).mean(1).min())
    # post-edge: excess over the LESION held plateau of the same record (what recruited 0 SOM spikes at W_MS<=304)
    ex = cpl(V1[:, 150:800]) - cpl(dat.vL1[:, 150:800])
    out['post_excess_max_pA'] = float(ex.max())
    out['post_excess_ms'] = float((ex > 1.0).sum(1).max())      # longest ms count with > 1 pA excess (any record)
    out['post_excess_200_pA'] = float(ex[:, 50:].max())         # from t=200 on (the LESION SOM-count window)
    # c3: older still above v_hold after the newer volley ends (two held plateaus -> closed-loop SOM recruitment)
    out['older_alive_450'] = int((d0 & (VX[:, 450] > VH)).sum())
    out['older_alive_450_corner'] = int((d0 & (VX[:, 450] > VH))[cm].sum())
    out['_V'] = (V1, VX, VY, d0, a, b, term, p6)
    return out


def fmt(o):
    s = 'C6 %2d/72 early %2d/45 late %d/6 kept %2d/21 | C7 grid def %3d (a) %3d (b) %3d a&b %3d | corner (a) %2d/24 (def %2d, b %2d)' % (
        o['C6'], o['early'], o['late'], o['kept'], o['c7def'], o['c7a'], o['c7b'], o['c7ab'], o['ca'], o['cdef'], o['cb'])
    s += ' || SA.125 C6 ' + ' '.join('%s:%d/8' % (c[-5:], o[c]['c6']) for c in SA125)
    s += ' C7a/def ' + ' '.join('%s:%d/%d b%d' % (c[-5:], o[c]['a'], o[c]['d'], o[c]['b']) for c in SA125)
    return s
