"""Which whitening AXIS recovers the real category structure — the L1 per-feature (axis-0) centering vs the
common-mode-pool per-concept (axis-1) removal the bridge physically does?

The decisive interpretation cross-check for the Phase-2/3 graded-whitening bridge gate. The numpy retinal
reference (`_phaseB_onoff_whitened_derisk.py`, +0.327 on real) centers `Xn - Xn.mean(0)` = per-HUB mean
across concepts (axis 0) — the standard L1/PCA centering. But the bridge's cm pool (hub_e drives cm densely
=> cm fires ~ the per-concept mean over hubs) and its `host_whitened_drive` reference center
`Xn - Xn.mean(1)` = per-CONCEPT mean across hubs (axis 1) — a DIFFERENT op an instantaneous pool can do.

If axis-1 ≈ axis-0 on the real structure, the bridge architecture is whitening-correct and a gate NEGATIVE is
a transmission/readout issue. If axis-1 ≪ axis-0, the cm-POOL architecture does the WRONG whitening
(per-concept common-mode, not per-feature centering) and the fix is per-HUB adaptation (each hub removes its
own temporal mean = axis-0 streaming), NOT a common-mode pool.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_whitening_axis_probe
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def poisson_spk(rate, gain, rng):
    return rng.poisson(np.maximum(rate, 0.0) * gain).astype(np.float64)


def onoff_code(drive, gain, rng):
    """ON/OFF spiking readout of a signed drive (the bridge's readout shape)."""
    on = np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(drive))])
    off = np.array([poisson_spk(np.maximum(-drive[i], 0.0), gain, rng) for i in range(len(drive))])
    return np.concatenate([on, off], axis=1)


def run_seed(seed, n_hub=500, k=128, gain=500.0):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0))                 # the bridge's log1p drive
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, X.shape[1]) / np.sqrt(X.shape[1])

    # the SAME random projection + ON/OFF readout for every whitening; only the centering axis differs.
    def score_axis(Xw, tag):
        drive = (W @ Xw.T).T
        code = onoff_code(drive, gain, rng)
        p = _pearson_vs_Strue(_cos_sim(code), S_true)
        g, ch = heldout_generalization(code, labels)
        print(f"  [{tag:34s}] Pearson(cos,S)={p:+.3f}  gen={g:.3f} (chance {ch:.3f})", flush=True)
        return p

    print(f"\n[whitening-axis probe seed {seed}] {C.shape[0]}c x {n_hub}h; host PPMI+SVD={host_p:+.3f}  "
          f"(gain={gain}, k={k})", flush=True)
    p_none = score_axis(Xn, "axis-NONE (no centering)")
    p_a0 = score_axis(Xn - Xn.mean(0, keepdims=True), "axis-0 per-FEATURE (L1/numpy-ref)")
    p_a1 = score_axis(Xn - Xn.mean(1, keepdims=True), "axis-1 per-CONCEPT (cm-pool/bridge)")
    return {"seed": seed, "host": host_p, "none": p_none, "axis0": p_a0, "axis1": p_a1}


def main():
    seeds = [42, 43, 44]
    rows = [run_seed(s) for s in seeds]
    a0 = np.mean([r["axis0"] for r in rows]); a1 = np.mean([r["axis1"] for r in rows])
    none = np.mean([r["none"] for r in rows]); host = np.mean([r["host"] for r in rows])
    print(f"\n  MEAN ({len(seeds)} seeds): host {host:+.3f} | none {none:+.3f} | "
          f"axis-0 per-feature {a0:+.3f} | axis-1 per-concept {a1:+.3f}", flush=True)
    if a0 >= 0.30 and a1 < a0 - 0.05:
        print(f"  => AXIS MATTERS: the L1 per-FEATURE centering (axis-0, +{a0:.3f}) recovers more structure than "
              f"the cm-POOL per-CONCEPT removal (axis-1, +{a1:.3f}). The bridge's common-mode POOL does a "
              f"WEAKER/different whitening -> the axis-0 fix is per-HUB adaptation (axis-0 streaming), not a pool.",
              flush=True)
    elif a1 >= 0.30 and abs(a1 - a0) <= 0.05:
        print(f"  => AXES EQUIVALENT: both recover the structure (axis-0 +{a0:.3f} ~= axis-1 +{a1:.3f}); the "
              f"bridge cm-pool whitening is structure-correct -> a gate NEGATIVE is a transmission/readout "
              f"issue, not the whitening axis.", flush=True)
    else:
        print(f"  => axis-0 per-feature +{a0:.3f} vs axis-1 per-concept +{a1:.3f} (host +{host:.3f}); "
              f"axis-1 recovers {100*(a1-none)/max(1e-9,a0-none):.0f}% of the axis-0 gain over none.",
              flush=True)


if __name__ == "__main__":
    main()
