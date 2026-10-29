# f1j (1)/(3) detail for the window candidates: which records still fail, lin-vs-sat difference, excess time course.
import numpy as np
from core import *
dat = Data()
cands = [(50, 16), (60, 12), (75, 8), (40, 12)]
for tg in (10.0, 12.0):
    for tauN, gN0 in cands:
        o = evaluate(dat, tau_gi=tg, nmda=dict(gN0=float(gN0), tauN=float(tauN)))
        ol = evaluate(dat, tau_gi=tg, nmda=dict(gN0=float(gN0), tauN=float(tauN), kin='lin'))
        V1, VX, VY, d0, a, b, term, p6 = o['_V']
        V1l = ol['_V'][0]
        print('syn_tau_g_i %g tauN %d gN0 %d: max|sat-lin| c1 v %.2f mV (volley %.2f, t>=200 %.3f)' % (
            tg, tauN, gN0, np.abs(V1 - V1l).max(), np.abs(V1 - V1l)[:, :150].max(), np.abs(V1 - V1l)[:, 200:].max()))
        nf = [(dat.names[i], 'early' if dat.early[i] else ('late' if dat.late[i] else 'was-PASS'),
               int((V1[i, 150:800] < VH).argmax() + 150)) for i in np.flatnonzero(~p6)]
        print('   C6 still failing:', nf)
        bf = [(dat.pcell[i], int((VY[i, 450:950] < VH).argmax() + 450)) for i in np.flatnonzero(d0 & ~b)]
        af = [(dat.pcell[i], int(term[i])) for i in np.flatnonzero(d0 & ~a)]
        print('   C7(b) fails (cell, newer fall ms):', bf)
        print('   C7(a) fails (cell, term):', af)
        # older termination time vs baseline at the corner
        ob = evaluate(dat, tau_gi=tg)
        tb = ob['_V'][6]; db = ob['_V'][3]
        cm = (dat.pcell == CORNER) & d0 & db & (term > 0) & (tb > 0)
        print('   corner older termination ms (NMDA) median %d [%d..%d] vs baseline median %d [%d..%d] (n=%d)' % (
            np.median(term[cm]), term[cm].min(), term[cm].max(), np.median(tb[cm]), tb[cm].min(), tb[cm].max(), cm.sum()))
