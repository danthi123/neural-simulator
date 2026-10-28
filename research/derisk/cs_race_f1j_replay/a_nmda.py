# f1j (1) A-NMDA: input-driven, decaying, Mg-blocked (1 mM Jahr-Stevens) NMDA conductance on the BDSP apical, E = 0 mV.
# Gating s rises from the slot's own sensory input (ds = -s/tauN + alpha*u*(1-s), alpha 0.5/ms, Wang-2002 style,
# u = 1 while the slot's volley is on), g_N = gN0 * s/s_ss (so g_N at the drive edge == gN0, in g_L units), decays with
# tauN after the edge. Integrated as a bounded exponential-Euler term (engine's apical-GABA idiom). BDSP regen unchanged.
# g_ap fixed at the recorded per-ms values (rebuilt from the recorded SOM increments for syn_tau_g_i 12).
import sys, json
from core import *
dat = Data()
kin = sys.argv[1] if len(sys.argv) > 1 else 'sat'
integ = sys.argv[2] if len(sys.argv) > 2 else 'bounded'
gns = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [4.0, 8.0, 16.0]
taus = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [50.0, 75.0, 100.0, 150.0]
rows = []
for tg in (10.0, 12.0):
    b = evaluate(dat, tau_gi=tg)
    print('syn_tau_g_i %g BASELINE (no NMDA)            ' % tg, fmt(b))
    for tauN in taus:
        for gN0 in gns:
            o = evaluate(dat, tau_gi=tg, nmda=dict(gN0=gN0, tauN=tauN, kin=kin, integ=integ))
            print('syn_tau_g_i %g tauN %3d gN0 %4.1f [%s/%s]' % (tg, tauN, gN0, kin, integ), fmt(o),
                  '|| volley dCpl %.1f pA (min %.1f); post-edge excess over LESION plateau max %.1f pA, >1pA for <= %d ms, from t=200 %.1f pA; older alive at 450: grid %d corner %d' % (
                      o['vol_dcpl_pA'], o['vol_dcpl_min_pA'], o['post_excess_max_pA'], o['post_excess_ms'], o['post_excess_200_pA'], o['older_alive_450'], o['older_alive_450_corner']))
            o.pop('_V'); rows.append(dict(tau_gi=tg, tauN=tauN, gN0=gN0, kin=kin, integ=integ, **o))
json.dump(rows, open('out_a_nmda_%s_%s.json' % (kin, integ), 'w'), indent=1)
