"""Does the BRIDGE's homeostatic threshold adaptation faithfully realize per-hub axis-0 centering — and at
what target firing rate? (Cheap numpy de-risk of the bridge MECHANISM before the expensive bridge build.)

The per-hub-adaptation de-risk (`_phaseB_perhub_adaptation_derisk.py`, 6-seed GO +0.311) subtracted each hub's
MEAN drive. The bridge's homeostasis instead drives each hub to a target FIRING RATE r, which sets its
threshold at the (1-r) QUANTILE of its drive — not the mean. So homeostatic centering ≈ mean-centering ONLY if
(a) the target rate is ~50% (threshold ≈ median ≈ mean) AND (b) there is an ON/OFF split to carry the signed
deviations (a single relu(drive - theta) keeps only the positive half). At the bridge's usual sparse target
rate (~2-5%) the threshold is the top-tail (95th+ pct) and keeps only a few concepts -> loses the structure.

This models homeostasis faithfully: per-hub theta_h = quantile(drive[:,h], 1 - r); ON = relu(drive - theta),
OFF = relu(theta - drive); the code = ON/OFF spike counts of the projected signed-centered drive. Sweep r.
GATE: does homeostatic-percentile centering recover axis-0 (~+0.31, the ideal mean-centered ON/OFF), and at
which target rate? A GO at some r says the bridge homeostasis (with that target rate + an ON/OFF readout) is
worth building; a NEGATIVE (no r recovers it) says homeostasis is the wrong bridge mechanism -> a dedicated
slow per-hub-mean (a guarded sim/ primitive) is needed for axis-0.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_homeostatic_centering_derisk
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
    on = np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(drive))])
    off = np.array([poisson_spk(np.maximum(-drive[i], 0.0), gain, rng) for i in range(len(drive))])
    return np.concatenate([on, off], axis=1)


def run_seed(seed, n_hub=500, k=128, gain=500.0):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, Xn.shape[1]) / np.sqrt(Xn.shape[1])

    def p_of(signed_drive):
        code = onoff_code((W @ signed_drive.T).T, gain, rng)
        return _pearson_vs_Strue(_cos_sim(code), S_true), heldout_generalization(code, labels)[0]

    # ideal axis-0: theta = per-hub MEAN (the validated target).
    p_a0, g_a0 = p_of(Xn - Xn.mean(0, keepdims=True))
    print(f"\n[homeostatic centering seed {seed}] {C.shape[0]}c x {n_hub}h; host={host_p:+.3f}  | ideal axis-0 "
          f"(theta=mean) +{p_a0:.3f}", flush=True)
    out = {"seed": seed, "host": host_p, "axis0_mean": p_a0, "rate": {}}
    for r in (0.05, 0.1, 0.25, 0.5, 0.75):
        theta = np.quantile(Xn, 1.0 - r, axis=0, keepdims=True)   # per-hub (1-r) quantile = homeostatic thresh
        p, g = p_of(Xn - theta)
        out["rate"][r] = p
        print(f"  [homeostatic target-rate r={r:4.2f}  (theta=({1-r:.2f})-quantile)] Pearson={p:+.3f}  gen={g:.3f}  "
              f"(=> {100*p/max(1e-9,p_a0):.0f}% of ideal axis-0)", flush=True)
    return out


def main():
    seeds = [42, 43, 44]
    rows = [run_seed(s) for s in seeds]
    a0 = np.mean([r["axis0_mean"] for r in rows]); host = np.mean([r["host"] for r in rows])
    best_r, best = None, -9
    for r in (0.05, 0.1, 0.25, 0.5, 0.75):
        m = np.mean([row["rate"][r] for row in rows])
        if m > best:
            best, best_r = m, r
    print(f"\n  MEAN ({len(seeds)} seeds): host {host:+.3f} | ideal axis-0 (theta=mean) {a0:+.3f} | "
          f"BEST homeostatic {best:+.3f} (target rate r={best_r})", flush=True)
    if best >= 0.30 and best >= a0 - 0.05:
        print(f"  GO: homeostatic percentile-threshold centering recovers axis-0 ({best:+.3f}, r={best_r}) -> "
              f"the bridge homeostasis (target rate ~{best_r}) + ON/OFF realizes per-hub centering. BUILD it.",
              flush=True)
    elif best >= 0.30:
        print(f"  PARTIAL: homeostatic centering clears +0.30 ({best:+.3f}, r={best_r}) but < ideal axis-0 "
              f"({a0:+.3f}); workable at the right target rate, with a precision gap.", flush=True)
    else:
        print(f"  NEGATIVE: no target rate recovers the structure (best {best:+.3f} at r={best_r}); homeostatic "
              f"percentile-threshold is NOT mean-centering -> a dedicated slow per-hub-mean (sim/ primitive) is "
              f"needed for axis-0, NOT homeostasis.", flush=True)


if __name__ == "__main__":
    main()
