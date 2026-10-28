# f1j (3) closed-loop validity of the replay (g_ap held at the recorded values).
#  A. what the records certify: SOM spikes recruited by a HELD plateau (LESION = SOM output gated) vs TRACE, c1 and c3.
#  B. how far the A-NMDA replay leaves the certified state: soma coupling (2 pA/mV above E_rest) vs the LESION trace,
#     converted to a mention-rate ratio with the f1h held-plateau f-I points (soma_g 20/40/80 at the same -24.15 mV plateau).
#  C. margin tests: SOM tail x{1.25,1.5,2} for C6; newer-volley SOM x{0.9,0.8,0.7,0.5} for the eraser (C7(a)).
import json, glob, os, sys, numpy as np, collections
from core import *
dat = Data()
L = dat.L
print('== A. records: SOM pool spikes recruited by held plateaus (LESION) vs TRACE, per cell')
for cell in sorted(set(dat.cell)):
    c1L = [int(L[n]['c1']['som'][200:800].sum()) for n in dat.names if n.startswith(cell + '_s')]
    c1T = [int(r['c1']['som'][200:800].sum()) for r in dat.C if r['cell'] == cell]
    c3L = [int(p['som'][450:950].sum()) for n in dat.names if n.startswith(cell + '_s') for p in L[n]['pairs']]
    c3T = [int(p['som'][450:950].sum()) for r in dat.C if r['cell'] == cell for p in r['pairs']]
    c3Lg = [int(p['som'][150:300].sum()) for n in dat.names if n.startswith(cell + '_s') for p in L[n]['pairs']]
    print('  %-14s c1 [200,800) LESION sum %3d max %2d | TRACE sum %3d || c3 [450,950) LESION sum %3d (mean %.2f) TRACE sum %3d (mean %.2f) | c3 [150,300) single held older: LESION sum %d' % (
        cell, sum(c1L), max(c1L), sum(c1T), sum(c3L), np.mean(c3L), sum(c3T), np.mean(c3T), sum(c3Lg)))
# f-I from f1h step 0 held plateaus (no SOM; same -24.15 mV plateau; W_IM 40)
R = '/home/dant123/Projects/sim/research/findings/raw/_cs_race/f1h/step0/'
fi = {}
for sg in (20, 40, 80):
    rates = []
    for f in sorted(glob.glob(R + 'cal_sg%d_wim40_s*.json' % sg)):
        if f.endswith('prov.json'): continue
        rates.append(json.load(open(f))['per_seed'][0]['c1']['pulsed_rate_hz'])
    fi[sg * 0.05 * 40.85] = float(np.mean(rates))
RL = 'raw/research/findings/raw/_cs_race/f1i/step0/'
lesion_rate = np.mean([json.load(open(RL + 'lesion_%s.json' % n))['per_seed'][0]['c1']['pulsed_rate_hz'] for n in dat.names if n.startswith('sa0p125')])
print('== f-I (f1h held plateau, W_IM 40): ' + ', '.join('%.1f pA -> %.2f Hz' % kv for kv in sorted(fi.items())) +
      ' ; f1i LESION (soma_g 40) mean %.2f Hz' % lesion_rate)
XI = np.array([0.0] + sorted(fi)); YF = np.array([0.0] + [fi[k] for k in sorted(fi)])
def frate(I):
    slope = (YF[-1] - YF[-2]) / (XI[-1] - XI[-2])
    return np.where(I <= XI[-1], np.interp(I, XI, YF), YF[-1] + slope * (I - XI[-1]))
cpl = lambda v: 2.0 * np.maximum(v - ER, 0)
def exposure(o, label):
    V1 = o['_V'][0]
    for cell in ('sa0p125_ms215', 'sa0p125_ms304'):
        m = dat.cell == cell
        wms = float(cell.split('ms')[1])
        out = []
        for w0, w1 in ((150, 250), (200, 400), (200, 800)):
            rn = frate(cpl(V1[m, w0:w1])).mean(1); rl = frate(cpl(dat.vL1[m, w0:w1])).mean(1)
            ratio = rn / rl
            out.append('[%d,%d) rate x%.2f-%.2f (W_MS-eq %.0f-%.0f)' % (w0, w1, ratio.min(), ratio.max(), wms * ratio.min(), wms * ratio.max()))
        ex = cpl(V1[m, 150:800]) - cpl(dat.vL1[m, 150:800])
        print('   %s %s: excess coupling over LESION max %.1f pA; ' % (label, cell, ex.max()) + '; '.join(out))
def scale_window(g, t0, t1, k, tau_gi):
    inc = incs(g); inc[..., t0:t1] *= k
    d = np.exp(-1.0 / tau_gi); out = np.zeros_like(inc); a = np.zeros(inc.shape[:-1])
    for t in range(inc.shape[-1]):
        a = a * d + inc[..., t]; out[..., t] = a
    return out
def margin(tg, nm, label):
    s = []
    for k in (1.25, 1.5, 2.0):
        g1 = scale_window(dat.g1, 100, 300, k, tg)
        V1, _ = replay(dat.D1, dat.U1, g1, nmda=nm)
        p6 = c6_pass(V1)
        s.append('x%.2f: early %d/45 C6 %d/72 SA.125 ms215 %d/8 ms304 %d/8' % (k, (p6 & dat.early).sum(), p6.sum(),
                 p6[dat.cell == 'sa0p125_ms215'].sum(), p6[dat.cell == 'sa0p125_ms304'].sum()))
    print('   %s C6 SOM-tail [100,300) margin: ' % label + ' | '.join(s))
    s = []
    for k in (0.9, 0.8, 0.7, 0.5):
        gX = scale_window(dat.gX, 300, 600, k, tg); gY = scale_window(dat.gY, 300, 600, k, tg)
        VX, _ = replay(dat.DX, dat.UX, gX, nmda=nm); VY, _ = replay(dat.DY, dat.UY, gY, nmda=nm)
        d0, a, b, term = c7(VX, VY); cm = dat.pcell == CORNER
        s.append('x%.1f: corner (a) %d/24 (def %d) grid a %d/%d b %d' % (k, a[cm].sum(), d0[cm].sum(), a.sum(), d0.sum(), b.sum()))
    print('   %s eraser [300,600) SOM margin: ' % label + ' | '.join(s))
print('== B/C. exposure and margins')
for tg in (10.0, 12.0):
    margin(tg, None, 'syn_tau_g_i %g BASELINE' % tg)
    for tauN, gN0 in ((50, 16), (60, 12), (75, 8), (50, 8), (100, 4)):
        nm = dict(gN0=float(gN0), tauN=float(tauN))
        o = evaluate(dat, tau_gi=tg, nmda=nm)
        lab = 'syn_tau_g_i %g tauN %d gN0 %d' % (tg, tauN, gN0)
        exposure(o, lab)
        margin(tg, nm, lab)
