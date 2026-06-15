"""Does per-HUB ADAPTATION (each hub subtracts its own running mean) realize the L1 axis-0 per-feature
centering on the REAL corpus — the brain-based alternative to the common-mode POOL?

The whitening-axis probe (`_phaseB_whitening_axis_probe.py`) showed the L1/numpy-ref centering is per-FEATURE
(axis-0, +0.32 on real) while the bridge's common-mode POOL does per-CONCEPT removal (axis-1, +0.255) — the
wrong axis, capping the cm-pool escape below the +0.30 bar. axis-0 = subtract each hub's mean ACROSS concepts,
which an instantaneous pool can't do (no per-hub cross-concept memory) but per-hub ADAPTATION can: each hub
neuron removes its OWN running/temporal mean (intrinsic spike-frequency adaptation / slow AHP / M-current /
synaptic depression = a per-neuron high-pass filter). This is MORE biological than a pool (every real neuron
adapts) and is the Mikulasch-Priesemann per-neuron predictive-coding form of whitening.

This numpy de-risk models per-hub adaptation as a STREAMING lagged EMA over a shuffled multi-epoch concept
stream: m_h <- (1-a)m_h + a*x_h (updated AFTER read = lagged, causal), output = relu-split(W @ (x - m_lagged)).
GATE: does it recover axis-0 (~+0.32) on real, across biologically-plausible adaptation rates a? A GO says the
corrected bridge architecture is per-hub adaptation (not a cm pool); a NEGATIVE says even axis-0 streaming
adaptation loses it (the lag/finite-tau hurts) -> the pool isn't the only problem.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_perhub_adaptation_derisk
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


def stream_adapt_codes(Xn, W, alpha, gain, n_epochs, seed):
    """Stream the concepts (shuffled per epoch); per-hub lagged EMA adaptation m; the readout for each concept
    is the ON/OFF spike code of W @ (x - m_lagged). Returns the per-concept code from the LAST epoch (the
    converged adaptation state). Causal: m is updated AFTER the read (the adaptation lags the input)."""
    rng = np.random.RandomState(seed)
    Nc, Nh = Xn.shape
    m = np.zeros(Nh)                      # per-hub running mean (the adaptation state)
    codes = np.zeros((Nc, 2 * W.shape[0]))
    for ep in range(n_epochs):
        order = rng.permutation(Nc)
        last = (ep == n_epochs - 1)
        for c in order:
            x = Xn[c]
            adapted = x - m               # per-hub adaptation: subtract the (lagged) running mean
            if last:
                codes[c] = onoff_code((W @ adapted[None, :].T).T, gain, rng)[0]
            m = (1.0 - alpha) * m + alpha * x   # update AFTER read => causal/lagged
    return codes


def run_seed(seed, n_hub=500, k=128, gain=500.0, n_epochs=12):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, Xn.shape[1]) / np.sqrt(Xn.shape[1])

    def p_of(code):
        return _pearson_vs_Strue(_cos_sim(code), S_true), heldout_generalization(code, labels)

    # batch references (ideal axis-0 and the cm-pool axis-1) with the SAME projection + readout.
    p_a0, (g_a0, ch) = p_of(onoff_code((W @ (Xn - Xn.mean(0)).T).T, gain, rng))
    p_a1, _ = p_of(onoff_code((W @ (Xn - Xn.mean(1, keepdims=True)).T).T, gain, rng))
    print(f"\n[per-hub adaptation seed {seed}] {C.shape[0]}c x {n_hub}h; host={host_p:+.3f}  "
          f"(gain {gain}, {n_epochs} epochs)  | batch axis-0={p_a0:+.3f}  axis-1={p_a1:+.3f}", flush=True)

    out = {"seed": seed, "host": host_p, "axis0_batch": p_a0, "axis1_batch": p_a1, "chance": ch, "adapt": {}}
    for alpha in (0.02, 0.05, 0.1, 0.2, 0.5):
        code = stream_adapt_codes(Xn, W, alpha, gain, n_epochs, seed)
        p, (g, _) = p_of(code)
        out["adapt"][alpha] = p
        print(f"  [per-hub adapt alpha={alpha:4.2f}] Pearson={p:+.3f}  gen={g:.3f} "
              f"(=> {100*p/max(1e-9,p_a0):.0f}% of batch axis-0)", flush=True)
    return out


def main():
    seeds = [42, 43, 44]
    rows = [run_seed(s) for s in seeds]
    a0 = np.mean([r["axis0_batch"] for r in rows]); a1 = np.mean([r["axis1_batch"] for r in rows])
    best_alpha = None; best = -9
    for alpha in (0.02, 0.05, 0.1, 0.2, 0.5):
        m = np.mean([r["adapt"][alpha] for r in rows])
        if m > best:
            best, best_alpha = m, alpha
    host = np.mean([r["host"] for r in rows])
    print(f"\n  MEAN ({len(seeds)} seeds): host {host:+.3f} | batch axis-0 {a0:+.3f} | batch axis-1 {a1:+.3f} | "
          f"BEST per-hub adapt {best:+.3f} (alpha={best_alpha})", flush=True)
    if best >= 0.30 and best >= a0 - 0.05:
        print(f"  GO: per-hub ADAPTATION recovers axis-0 ({best:+.3f} ~= batch {a0:+.3f}, clears +0.30) -> the "
              f"corrected bridge architecture = per-hub adaptation (intrinsic), NOT a common-mode pool.",
              flush=True)
    elif best >= a1:
        print(f"  PARTIAL: per-hub adaptation ({best:+.3f}) beats the cm-pool axis-1 ({a1:+.3f}) but < axis-0 "
              f"batch ({a0:+.3f}) -- the streaming lag costs some structure; still the better mechanism.",
              flush=True)
    else:
        print(f"  NEGATIVE: per-hub adaptation ({best:+.3f}) does not beat the cm-pool axis-1 ({a1:+.3f}); "
              f"streaming adaptation isn't the fix.", flush=True)


if __name__ == "__main__":
    main()
