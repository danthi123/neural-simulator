"""DIAGNOSTIC probe (not a deliverable): characterize the ~8% near-tie miss structure of the graded conductance read.
Collect per-position substrate conductances (ge, gi per pool) at seed 42, and test cheap ARGMAX-changing corrections
OFFLINE (no substrate re-run) to pick the mechanism grounded in the actual bias structure:
  (0) baseline margin = df_e*ge + df_i*gi                              (reconstructs head_w@h, OMITS head_b)
  (1) + per-pool tonic bias s*head_b  (the missing base-rate prior; a per-pool intrinsic-excitability drive)
  (2) re-fit shared conductance coeffs ce*ge + ci*gi (+ head_b)         (re-calibrate the transfer; seed-independent)
  (3) the CEILING of head_b: argmax(head_w@h + head_b) vs argmax(head_w@h)  (how much of the true argmax IS head_b)
Reports argmax-agreement vs the TRUE logit (head_w@h + head_b) for each, plus corr(margin, head_w@h).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.runners._wkv_fewspike_read_derisk import WKVReadout, _load_eval, _softmax  # noqa: E402
from research.runners._wkv_graded_conductance_read_derisk import GradedConductanceLogitRead  # noqa: E402


def collect(seed, n_pos=160, warmup=3, hid_pop=1):
    ckpt = f"bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz"
    ro = WKVReadout(ckpt)
    ev_ids, vocab = _load_eval(ro, "", 8000, seed, max(64, n_pos // 6))
    s = GradedConductanceLogitRead(ro, seed, pop=4, hid_pop=hid_pop)
    rows = []
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]).copy()          # true logit = head_w@h + head_b
            if ro.unk_idx >= 0:
                lg[ro.unk_idx] = -1e30
            # hidden feature h and head_w@h (the thing the margin reconstructs, WITHOUT head_b)
            state = np.concatenate([ap, an])
            r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[ids[t]]))))
            h = r_h * (ro.Wo_sp @ state)
            hwh = ro.head_w @ h                              # head_w@h  (no bias)
            r = s.read_graded(ap, an, ids[t])
            rows.append(dict(ge=r["ge"].copy(), gi=r["gi"].copy(), logit=lg.copy(),
                             hwh=hwh.copy(), margin=r["margin"].copy()))
            positions += 1
            if positions >= n_pos:
                break
        if positions >= n_pos:
            break
    return ro, s, rows


def _agree(win, tgt):
    return float(np.mean([int(a == b) for a, b in zip(win, tgt)]))


def analyze(seed, ro, s, rows, unk):
    GE = np.array([r["ge"] for r in rows])                  # [N,V]
    GI = np.array([r["gi"] for r in rows])
    LG = np.array([r["logit"] for r in rows])
    HWH = np.array([r["hwh"] for r in rows])
    head_b = ro.head_b.copy()
    if unk >= 0:
        LG[:, unk] = -1e30; HWH[:, unk] = -1e30; head_b = head_b.copy(); head_b[unk] = head_b.min()
    df_e, df_i = s.df_e, s.df_i
    N, V = GE.shape
    tgt = [int(np.argmax(LG[i])) for i in range(N)]         # TRUE argmax (head_w@h + head_b)

    base = df_e * GE + df_i * GI                            # [N,V] reconstructs head_w@h
    win_base = [int(np.argmax(base[i])) for i in range(N)]
    win_hwh = [int(np.argmax(HWH[i])) for i in range(N)]
    tgt_hwh = win_hwh                                        # reconstruction target = argmax(head_w@h)
    # per-position corr(margin, head_w@h): the WITHIN-POSITION ranking fidelity (not the flattened cross-position one)
    pcorr = np.mean([np.corrcoef(base[i], HWH[i])[0, 1] for i in range(N)])
    print(f"[seed {seed}] N={N} V={V} per-position corr(margin, head_w@h)={pcorr:.4f}")
    print(f"  (0) baseline margin argmax_agree vs TRUE logit          = {_agree(win_base, tgt):.4f}")
    print(f"  (R) RECONSTRUCTION fidelity argmax(margin) vs argmax(head_w@h) = {_agree(win_base, tgt_hwh):.4f}")
    print(f"  (3) CEILING argmax(head_w@h) agree vs TRUE = {_agree(win_hwh, tgt):.4f}  "
          f"[omitting head_b costs {1-_agree(win_hwh, tgt):.3f}]")

    # estimate the per-position reconstruction gain alpha (margin ~ alpha*head_w@h): median slope
    alphas = []
    for i in range(N):
        num = float(np.dot(base[i] - base[i].mean(), HWH[i] - HWH[i].mean()))
        den = float(np.dot(HWH[i] - HWH[i].mean(), HWH[i] - HWH[i].mean()))
        if den > 1e-12:
            alphas.append(num / den)
    alpha = float(np.median(alphas))
    print(f"  alpha (margin/head_w@h gain) median = {alpha:.4f}")

    # softmax mass of the TRUE distribution (the recov_argmax metric numerator/denominator)
    P = np.array([_softmax(LG[i]) for i in range(N)])
    mass_argmax = float(np.mean([P[i, tgt[i]] for i in range(N)]))       # denominator (perfect-argmax mass)

    # (1) + per-pool tonic bias s*head_b at the margin scale — sweep by BOTH argmax-agree and MASS (the headline)
    best = (0.0, -1); bestm = (0.0, -1)
    # head_b typical magnitude vs margin spread: scale grid by the margin std so s spans "swamped -> dominant"
    mstd = float(base.std())
    grid = [0.0, 0.02*mstd, 0.05*mstd, 0.1*mstd, 0.2*mstd, 0.5*mstd, 1.0*mstd, 2.0*mstd]
    for sc in grid:
        m = base + sc * head_b[None, :]
        wm = [int(np.argmax(m[i])) for i in range(N)]
        a = _agree(wm, tgt)
        mass = float(np.mean([P[i, wm[i]] for i in range(N)])) / max(1e-9, mass_argmax)   # recov_argmax
        if a > best[1]:
            best = (round(sc, 3), a)
        if mass > bestm[1]:
            bestm = (round(sc, 3), round(mass, 4))
    base_recov = float(np.mean([P[i, win_base[i]] for i in range(N)])) / max(1e-9, mass_argmax)
    print(f"  (1) + s*head_b: best-argmax s={best[0]} agree={best[1]:.4f} | best-MASS s={bestm[0]} "
          f"recov_argmax={bestm[1]} [baseline recov_argmax {base_recov:.4f}]")

    # (1b) GENERALIZATION: pick best s on TRAIN half, apply to HELD-OUT half (single scalar -> should transfer)
    ntr = N // 2; te = np.arange(ntr, N); tr = np.arange(ntr)
    bests = (0.0, -1)
    for sc in grid:
        m = base[tr] + sc * head_b[None, :]
        a = _agree([int(np.argmax(m[i])) for i in range(len(tr))], [tgt[i] for i in tr])
        if a > bests[1]:
            bests = (sc, a)
    sc = bests[0]
    mte = base[te] + sc * head_b[None, :]
    a_te = _agree([int(np.argmax(mte[i])) for i in range(len(te))], [tgt[i] for i in te])
    a_te_base = _agree([win_base[i] for i in te], [tgt[i] for i in te])
    print(f"  (1b) held-out: s(train)={round(sc,3)} -> argmax_agree {a_te:.4f} (baseline {a_te_base:.4f})")

    # (4) PER-POOL HOMEOSTATIC NORMALIZATION (subtract each pool's baseline / std, from TRAIN positions), then add
    # the base-rate tonic head_b. Brain-based: per-pool spike-frequency adaptation removes the position-independent
    # per-pool DC that swamps the discriminative signal; head_b is the base-rate prior as a tonic drive.
    mu = base[tr].mean(axis=0); sd = base[tr].std(axis=0) + 1e-9      # per-pool train stats
    znr = (base - mu[None, :]) / sd[None, :]                          # z-scored margin [N,V]
    # reconstruction after normalization (vs head_w@h)
    rec_z = _agree([int(np.argmax(znr[te][i])) for i in range(len(te))], [tgt_hwh[i] for i in te])
    # sweep head_b scale on TRAIN (z-space), apply held-out; head_b also standardized to z-space scale
    hb_z = (head_b - head_b.mean()) / (head_b.std() + 1e-9)
    bestz = (0.0, -1)
    for sc2 in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        mz = znr[tr] + sc2 * hb_z[None, :]
        a = _agree([int(np.argmax(mz[i])) for i in range(len(tr))], [tgt[i] for i in tr])
        if a > bestz[1]:
            bestz = (sc2, a)
    sc2 = bestz[0]
    mzte = znr[te] + sc2 * hb_z[None, :]
    a_znr = _agree([int(np.argmax(mzte[i])) for i in range(len(te))], [tgt[i] for i in te])
    print(f"  (4) per-pool NORM: reconstruction(vs head_w@h) held-out {rec_z:.4f} | +head_b s={sc2} "
          f"argmax_agree vs TRUE held-out = {a_znr:.4f}  [baseline held-out {a_te_base:.4f}, ceiling {_agree(win_hwh, tgt):.4f}]")
    return alpha, best[0]


if __name__ == "__main__":
    import os
    hp = int(os.environ.get("PROBE_HIDPOP", "1"))
    print(f"### hid_pop={hp} ###")
    for seed in (42, 43):
        ro, s, rows = collect(seed, n_pos=160, hid_pop=hp)
        analyze(seed, ro, s, rows, ro.unk_idx)
        print()
