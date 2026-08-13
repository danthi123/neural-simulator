"""Calibration probe for the GRADED conductance read: collect the pooled g_e[V], g_i[V] and the TRUE logit[V] for
seed 42 (wiring ratio=1.0 so the read-side inhibitory gain is the SOLE compensation), then sweep the inhibitory
read-gain beta OFFLINE to find (a) the beta that maximises the margin<->logit correlation / argmax-agreement and
(b) whether a BALANCED signed margin (df_e*g_e - beta*g_i) beats positive-only (g_e alone). This locates a SINGLE
fixed operating point WITHOUT per-seed tuning (calibrate on 42, the runner then verifies generalisation on 43/44/
100/101/102)."""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("SIM_BACKEND", "numpy")

from sim.backend import to_host, get_backend
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _softmax, _load_eval
from research.runners._wkv_graded_conductance_read_derisk import GradedConductanceLogitRead

SEED = 42
CKPT = f"bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{SEED}.npz"
WIRING_RATIO = 1.0          # <-- no pre-compensation; the read-side beta absorbs ALL of it
N_POS = 80

ro = WKVReadout(CKPT)
ev_ids, vocab = _load_eval(ro, "", 8000, SEED, 64)
s = GradedConductanceLogitRead(ro, SEED, pop=4, ou_std=40.0, read_window=150, hid_gain=120.0,
                               syn_scale=12.0, ratio=WIRING_RATIO, graded_floor_pA=0.0, settle_frac=0.2)
print(f"df_e={s.df_e:.2f} df_i={s.df_i:.2f} v_ref={s.v_ref:.2f}  wiring_ratio={WIRING_RATIO}")

GE = []; GI = []; LG = []; ARG = []
warmup = 3; positions = 0
for ids in ev_ids:
    if len(ids) < warmup + 2:
        continue
    ap = np.zeros(ro.D); an = np.zeros(ro.D)
    for t in range(len(ids) - 1):
        ap, an = ro.advance(ap, an, ids[t])
        if t < warmup:
            continue
        lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
        if ro.unk_idx >= 0:
            lg_supp[ro.unk_idx] = -1e30
        feat = s._hidden_feature(ap, an, ids[t])
        margin, ge_m, gi_m, psp = s._graded_margin(feat, want_diag=True)
        GE.append(ge_m); GI.append(gi_m); LG.append(lg_supp); ARG.append(int(np.argmax(lg_supp)))
        positions += 1
        if positions >= N_POS:
            break
    if positions >= N_POS:
        break

GE = np.array(GE); GI = np.array(GI); LG = np.array(LG); ARG = np.array(ARG)
print(f"collected {len(GE)} positions; g_e mean={GE.mean():.3f} g_i mean={GI.mean():.3f}")
pfull = np.array([_softmax(l) for l in LG])
host_mass = np.array([pfull[i, ARG[i]] for i in range(len(ARG))]).mean()

def evalbeta(beta):
    margin = s.df_e * GE + s.df_i * beta * GI            # df_i<0 so this SUBTRACTS beta*|df_i|*g_i
    win = margin.argmax(axis=1)
    agree = float((win == ARG).mean())
    mass = float(np.array([pfull[i, win[i]] for i in range(len(win))]).mean())
    # per-position pearson of margin vs logit, averaged
    cor = []
    for i in range(len(win)):
        m = margin[i]; l = LG[i]
        msk = l > -1e29
        if m[msk].std() > 1e-9 and l[msk].std() > 1e-9:
            cor.append(np.corrcoef(m[msk], l[msk])[0, 1])
    return agree, mass, float(np.mean(cor))

# positive-only (beta=0)
a0, m0, c0 = evalbeta(0.0)
print(f"\nPOSITIVE-ONLY (beta=0): argmax_agree={a0:.3f} mass={m0:.4f} (fid={m0/host_mass:.3f}) corr={c0:.3f}")
print(f"{'beta':>8} {'agree':>7} {'mass':>7} {'read_fid':>9} {'corr':>7} {'signed>pos?':>11}")
best = None
for beta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]:
    a, m, c = evalbeta(beta)
    fid = m / host_mass
    flag = "YES" if m > m0 * 1.02 else "no"
    print(f"{beta:>8.2f} {a:>7.3f} {m:>7.4f} {fid:>9.3f} {c:>7.3f} {flag:>11}")
    if best is None or c > best[2]:
        best = (beta, a, c, m)
print(f"\nBEST-by-corr: beta={best[0]} agree={best[1]:.3f} corr={best[2]:.3f} mass={best[3]:.4f} (host_mass={host_mass:.4f})")
