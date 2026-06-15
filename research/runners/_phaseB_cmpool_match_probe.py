"""Phase-B retinal build — PRE-BUILD probe #2: HOW must the common-mode pool's cortex weights be set?

The prior cm-pool (CYCLE 61, findings table row 3) FAILED ("uniform inhibition removes signal+common-mode
together"). But the whitening-locus probe shows axis1 population-mean subtraction WORKS. Reconcile:

Whitening at the INPUT before the projection W is:
    W @ (Xn - m·1)  =  W@Xn  -  m·(W@1)          where m = per-concept population mean (scalar/concept)
So the inhibition each cortex neuron j must receive is  m · (W@1)[j]  = m · rowsum_j(W).
=> the cm->cortex weight onto neuron j must be PROPORTIONAL to neuron j's hub->cortex row-sum, NOT random.
The prior cm-pool used RANDOM (or uniform) cm->cortex weights, so it subtracted the WRONG direction.

This probe (numpy, real corpus, ON/OFF, high budget) compares:
  (1) ideal axis1 whitening                         -- the target (W@(Xn - rowmean))
  (2) cm-pool, cm->cortex weight = W rowsum (MATCHED) -- the bridge-faithful realization
  (3) cm-pool, cm->cortex weight = RANDOM            -- the prior (failed) realization
  (4) cm-pool, cm->cortex weight = UNIFORM            -- a rank-1 uniform pool
where the cm "activity" = the per-concept pooled hub drive (sum over hubs) = the common mode, and the
subtracted inhibition = cm_activity * cm_to_cortex_weight[j].

If (2) ~= (1) >> (3),(4), then the bridge cm-pool WILL work IF its cm->cortex weights are set to the
hub->cortex row-sums (a deterministic, settable weight pattern -- NOT a sim/ edit, just how I wire it).

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_cmpool_match_probe
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def poisson_spk(rate, gain, rng):
    return rng.poisson(np.maximum(rate, 0.0) * gain).astype(np.float64)


def onoff_code(drive, gain, rng):
    on = np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(drive))])
    off = np.array([poisson_spk(np.maximum(-drive[i], 0.0), gain, rng) for i in range(len(drive))])
    return np.concatenate([on, off], axis=1)


def stat(name, code, S_true, labels):
    p = _pearson_vs_Strue(_cos_sim(code), S_true)
    g, ch = heldout_generalization(code, labels)
    print(f"    [{name:30s}] ON/OFF Pearson={p:+.3f}  gen={g:.3f} (chance {ch:.3f})", flush=True)
    return p


def run(C, labels, S_true, gain=2000.0, k=128):
    C = np.asarray(C, np.float64); labels = np.asarray(labels)
    X = ppmi_matrix(C, 0.75)
    host, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75), labels)
    H = X.shape[1]; rng = np.random.RandomState(7)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, H) / np.sqrt(H)
    print(f"  host=+{host:.3f}  (gain {int(gain)}, k {k})", flush=True)

    geWXn = (W @ Xn.T).T                       # [Nc x k] the raw projected drive
    rowsum = W.sum(1)                          # [k] each cortex neuron's hub->cortex row-sum = W@1
    popmean = Xn.mean(1)                       # [Nc] per-concept population mean (the cm activity ~ this)
    cm_activity = Xn.sum(1)                    # [Nc] pooled hub drive (proportional to popmean * H)

    # (1) ideal axis1 whitening
    stat("ideal axis1 whiten", onoff_code((W @ (Xn - Xn.mean(1, keepdims=True)).T).T, gain, rng), S_true, labels)
    # (2) cm-pool, weight = W rowsum (MATCHED). drive = W@Xn - popmean * rowsum  (== ideal, since
    #     W@(m*1) = m*rowsum). Scale cm so cm_inhibition[j] = popmean * rowsum[j].
    drive2 = geWXn - popmean[:, None] * rowsum[None, :]
    stat("cm-pool weight=W.rowsum (MATCH)", onoff_code(drive2, gain, rng), S_true, labels)
    # (3) cm-pool, weight = RANDOM (the prior failed wiring). inhibition[j] = cm_activity * w_rand[j].
    w_rand = np.abs(rng.randn(k)) / np.sqrt(H)
    drive3 = geWXn - (cm_activity[:, None] * w_rand[None, :]) / H
    stat("cm-pool weight=RANDOM (prior)", onoff_code(drive3, gain, rng), S_true, labels)
    # (4) cm-pool, weight = UNIFORM (rank-1). inhibition[j] = cm_activity * c  (same c for all j).
    c = float(np.mean(rowsum)) if np.mean(rowsum) != 0 else 1.0
    drive4 = geWXn - cm_activity[:, None] * (c / H)
    stat("cm-pool weight=UNIFORM (rank1)", onoff_code(drive4, gain, rng), S_true, labels)
    # point control
    stat("POINT (no whiten, no ON/OFF)", np.array([poisson_spk(np.maximum(geWXn[i], 0.0), gain, rng)
                                                   for i in range(len(X))]), S_true, labels)


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    print("[cm-pool match probe] does cm->cortex weight = hub->cortex ROW-SUM realize the axis1 whitening "
          "(where random/uniform cm weights failed)?", flush=True)
    Cr, lr, Sr = build_real_corpus(42, 500)
    run(Cr, lr, Sr)


if __name__ == "__main__":
    main()
