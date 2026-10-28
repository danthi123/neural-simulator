# f1j (2) A-reservoir: the engine's fused_coincidence_plateau form (sim/kernels.py:1426-1460) in place of the BDSP regen.
#   g <- g*exp(-dt/80) + g_inc + A*mg(v)*sig(v)     (coincidence_tau_decay_ms 80, sim/config.py:573; sustain into the slow g only)
#   g_rise <- g_rise*exp(-dt/2) + g_inc ; g_eff = max(g - g_rise, 0) ; regen term = g_eff*mg(v)*(E_e - v), E_e = 0, Mg 1 mM
#   g_inc = 0 (no coincidence trigger; the sensory box-car drive is unchanged), so g_rise = 0 and g_eff = g.
#   A = R*self_regen folds the current->apical scale: engine two-compartment path R = apical_R 0.15 x self_regen 2.0 -> 0.30;
#   BDSP path (no R, mV units) x self_regen 2.0 -> 2.0; plateau-matched A_m = sr*(1-d)/mg(vp)^2 keeps the BDSP plateau
#   fixed point vp = -24.15 mV. KIR, leak, drive, apical GABA, recorded g_ap unchanged.
import numpy as np
from core import *
from ode import f_net
dat = Data()
dR = np.exp(-1 / 80.0)
vs = np.linspace(-90, 20, 220001)
fn = f_net(vs); sc = np.flatnonzero(np.sign(fn[:-1]) != np.sign(fn[1:])); vp = vs[sc[-1]]
A_m = SR * (1 - dR) / mg_block(vp) ** 2
print('BDSP plateau fixed point %.3f mV; plateau-matched A_m = %.4f; engine two-comp literal 0.30 = %.2f x A_m; BDSP-units literal 2.0 = %.1f x A_m' % (vp, A_m, 0.30 / A_m, 2.0 / A_m))
def static(A):
    G = A / (1 - dR) * mg_block(vs) ** 2 * sig(vs)
    f = -(vs - ER) + G * (EE - vs) + KG / (1 + np.exp((vs - KVH) / KK)) * (EK - vs)
    z = np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:])); fps = [round(float(vs[i]), 2) for i in z]
    if len(z) == 3:
        m = (vs > vs[z[1]]) & (vs < vs[z[2]]); gc = (f[m] / (vs[m] - EI)).max()
    else:
        gc = float('nan')
    return fps, gc
def bdsp_gc():
    m = (vs > vs[sc[1]]) & (vs < vs[sc[2]]); return (fn[m] / (vs[m] - EI)).max()
print('BDSP regen: fixed points %s, static g_crit %.4f' % ([round(float(vs[i]), 2) for i in sc], bdsp_gc()))
cases = [('0.50 A_m', 0.5 * A_m), ('0.75 A_m', 0.75 * A_m), ('1.00 A_m (plateau-matched)', A_m), ('1.50 A_m', 1.5 * A_m),
         ('engine two-comp literal 0.30', 0.30), ('3.00 A_m', 3.0 * A_m), ('BDSP-units literal 2.0', 2.0)]
for lab, A in cases:
    fps, gc = static(A)
    print('A %-30s A=%.4f static fixed points %s static g_crit %.4f' % (lab, A, fps, gc))
for tg in (10.0, 12.0):
    o = evaluate(dat, tau_gi=tg)
    print('syn_tau_g_i %g BASELINE BDSP regen        ' % tg, fmt(o))
    for lab, A in cases:
        o = evaluate(dat, tau_gi=tg, regen='reservoir', res=dict(A=A, tau=80.0))
        V1 = o['_V'][0]
        unp = replay(np.zeros_like(dat.D1), np.zeros_like(dat.U1), dat.g1, regen='reservoir', res=dict(A=A, tau=80.0))[0]
        print('syn_tau_g_i %g reservoir %-30s' % (tg, lab), fmt(o),
              '|| c1 v at 800 med %.1f; unpulsed max %.1f; older alive at 450 grid %d corner %d; post-edge excess over LESION max %.1f pA' % (
                  np.median(V1[:, 799]), unp.max(), o['older_alive_450'], o['older_alive_450_corner'], o['post_excess_max_pA']))
