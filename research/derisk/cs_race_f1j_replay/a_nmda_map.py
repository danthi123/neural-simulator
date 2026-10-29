# f1j (1) informational: finer tauN x gN0 map of the A-NMDA working window (sat kinetics, bounded integration).
# cell = early-rescued/45 . C6/72 . corner C7(a)/24 . grid C7 a&b/216 ; residual gN at the newer's onset (t=300) and offset (t=450)
import numpy as np
from core import *
dat = Data()
TAUS = [40, 50, 60, 75, 90, 100, 125]; GNS = [2, 4, 6, 8, 10, 12, 16]
for tg in (10.0, 12.0):
    print('syn_tau_g_i %g  (cell: early/45 C6/72 corner(a)/24 grid a&b/216)' % tg)
    print('tauN\\gN0 ' + ''.join('%22s' % g for g in GNS))
    for tauN in TAUS:
        row = []
        for gN0 in GNS:
            o = evaluate(dat, tau_gi=tg, nmda=dict(gN0=float(gN0), tauN=float(tauN)))
            row.append('%2d %2d %2d %3d' % (o['early'], o['C6'], o['ca'], o['c7ab']))
        print('%7d ' % tauN + ''.join('%22s' % r for r in row))
print('residual gN (g_L) at t=300 / t=450 = gN0*exp(-(t-149)/tauN):')
for tauN in TAUS:
    print('%4d ' % tauN + '  '.join('%4.1f:%.2f/%.2f' % (g, g * np.exp(-151 / tauN), g * np.exp(-301 / tauN)) for g in GNS))
