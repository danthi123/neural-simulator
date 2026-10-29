# f1j (2) informational: fine scan of the reservoir scale near its bistability onset, and the A=2.0 explicit-step check.
import numpy as np
from core import *
from ode import f_net
exec(open('a_reservoir.py').read().split("cases = [")[0].split("dat = Data()")[1])  # reuse vs, static(), A_m (no evaluation)
dat = Data()
for k in (0.80, 0.84, 0.88, 0.92, 0.96, 1.00, 1.10, 1.25):
    A = k * A_m; fps, gc = static(A)
    for tg in (10.0,):
        o = evaluate(dat, tau_gi=tg, regen='reservoir', res=dict(A=A, tau=80.0))
        print('A %.2f A_m fps %-24s g_crit %.4f | syn_tau_g_i %g ' % (k, fps, gc, tg) + fmt(o))
# A = 2.0 (BDSP units): per-step explicit factor of the reservoir term at the plateau
G = 2.0 / (1 - dR) * mg_block(-1.0) ** 2 * sig(-1.0)
print('A=2.0: steady regen conductance near -1 mV = %.1f g_L -> explicit per-step factor %.2f (>2 = unstable)' % (G, G / TAU))
V1 = replay(dat.D1[:2], dat.U1[:2], dat.g1[:2], regen='reservoir', res=dict(A=2.0))[0]
print('A=2.0 c1 v trace t=150..170:', np.round(V1[0, 150:170], 1).tolist())
