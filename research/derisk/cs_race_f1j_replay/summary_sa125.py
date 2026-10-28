# f1j: per-cell (W_SA 0.125) early/late self-termination rescue for the requested A-NMDA grid (sat, bounded) at syn_tau_g_i 10/12.
from core import *
dat = Data()
for tg in (10.0, 12.0):
    for nm in [None] + [dict(gN0=float(g), tauN=float(t)) for t in (50, 75, 100, 150) for g in (4, 8, 16)]:
        o = evaluate(dat, tau_gi=tg, nmda=nm)
        lab = 'BASELINE' if nm is None else 'tauN %3d gN0 %2d' % (nm['tauN'], nm['gN0'])
        print('syn_tau_g_i %g %-16s ' % (tg, lab) + ' | '.join('%s early %d/%d late %d/%d C6 %d/8 C7 a %d/%d b %d' % (
            c[-5:], o[c]['early'], o[c]['n_early'], o[c]['late'], o[c]['n_late'], o[c]['c6'], o[c]['a'], o[c]['d'], o[c]['b']) for c in SA125))
