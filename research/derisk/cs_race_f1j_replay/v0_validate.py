# f1j step 0: the vectorized replay reproduces the reference ode.replay and the recorded traces.
import numpy as np, time
from core import *
import ode, cf
dat = Data()
t0 = time.time()
o = evaluate(dat)
print('vectorized baseline run %.1fs' % (time.time() - t0))
V1, VX, VY, d0, a, b, term, p6 = o['_V']
print('max|vec - recorded| c1 %.4f mV, c3 X %.4f, Y %.4f' % (np.abs(V1 - dat.v1).max(), np.abs(VX - dat.vX).max(), np.abs(VY - dat.vY).max()))
ref = np.stack([ode.replay(-77.28, dat.D1[i], dat.g1[i]) for i in range(len(dat.names))])
print('max|vec - ode.replay| c1 %.2e' % np.abs(V1 - ref).max())
print('C6 vs recorded status mismatches:', int((p6 != dat.pas).sum()))
print('early', int(dat.early.sum()), 'late', int(dat.late.sum()), 'pass', int(dat.pas.sum()))
# cross-check c7 vectorized vs cf.c7 loop
mm = 0
for i in range(len(VX)):
    r = cf.c7(VX[i], VY[i])
    if r['d0'] != d0[i] or (r['d0'] and (r['a'] != a[i] or r['b'] != b[i])): mm += 1
print('c7 vectorized vs cf.c7 mismatches:', mm)
print('BASELINE', fmt(o))
o12 = evaluate(dat, tau_gi=12.0)
print('BASELINE syn_tau_g_i 12', fmt(o12))
