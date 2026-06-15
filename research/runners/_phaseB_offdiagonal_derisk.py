"""Is the residual gap (diagonal-whitening +0.31 -> host +0.44) closeable by OFF-DIAGONAL decorrelation
(full ZCA whitening), or is it PPMI/SVD-specific? (Determines whether the off-diagonal is worth pursuing.)

The arc converged: the DIAGONAL of whitening (per-feature mean-centering = subtractive adaptation [shipped] +
a signed projection = E/I balance [confirmed: bridge +0.155, escapes the excitatory-only +0.045 collapse]) is
built + validated, but caps at ~+0.31 numpy / +0.155 bridge on the real corpus (host PPMI+SVD +0.44). The
research doc flagged the existing on-substrate `graded_lateral` (analog cross-neuron anti-Hebbian whitening,
bridge.py:1789) as the tool for the OFF-DIAGONAL (cross-neuron decorrelation) IF the residual is ever pursued.

This cheap de-risk asks: does FULL whitening (centering + off-diagonal decorrelation = ZCA) close the +0.31 ->
+0.44 gap? GO => the off-diagonal IS the gap -> testing graded_lateral on the bridge (non-gated, existing) is
the next cheap step toward a non-marginal result. NEGATIVE (ZCA ~= centering-only) => the +0.13 is PPMI/SVD-
specific (the log-ratio nonlinearity + low-rank denoising), NOT decorrelation -> neither graded_lateral NOR
dendrites close it; the marginality is intrinsic to the linear-projection family on this corpus.

Compares (real, axis-0 centered, ON/OFF readout, host gain):
  (a) centering only        -- W @ (Xn - mean0)              [the diagonal, ~+0.31]
  (b) ZCA whitened          -- W @ ((Xn-mean0) @ Sigma^-1/2) [diagonal + off-diagonal decorrelation]
  (c) PCA-whitened (k=topV) -- low-rank ZCA (denoise, ~ SVD) [tests if it's the SVD denoising]
  (d) host PPMI+SVD          -- the ceiling +0.44
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_offdiagonal_derisk
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


def zca(Xc, eps=1e-3, rank=None):
    """ZCA whitening of the (already-centered) Xc [Nc x Nh]: Xc @ V diag(1/sqrt(s+eps)) V^T. rank<None => full;
    rank=k => keep top-k components (PCA-whitening / low-rank denoise)."""
    # covariance over features (rows=samples): Sigma = Xc^T Xc / Nc  [Nh x Nh] is huge; instead whiten in the
    # sample space via SVD of Xc (Nc x Nh), Nc=64 small: Xc = U s V^T; whitened = U V^T-ish. Use the economical
    # sample-space whitening: Xw = U @ V_k^T scaled. Simplest faithful: SVD, rescale singular values to 1.
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = len(s) if rank is None else min(rank, len(s))
    s_inv = np.zeros_like(s)
    s_inv[:k] = 1.0 / np.sqrt(s[:k] ** 2 + eps)
    # whitened samples in feature space (ZCA): Xc @ V diag(1/sqrt(s^2+eps)) V^T
    return Xc @ (Vt.T * s_inv) @ Vt


def run_seed(seed, n_hub=500, k=128, gain=500.0):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0)); Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xc = Xn - Xn.mean(0, keepdims=True)
    W = rng.randn(k, n_hub) / np.sqrt(n_hub)

    def p(drive):
        return _pearson_vs_Strue(_cos_sim(onoff_code(drive, gain, rng)), S_true)

    pa = p((W @ Xc.T).T)
    pb = p((W @ zca(Xc).T).T)
    pc = p((W @ zca(Xc, rank=8).T).T)
    pd = p((W @ zca(Xc, rank=16).T).T)
    print(f"\n[off-diagonal seed {seed}] {C.shape[0]}c x {n_hub}h; host PPMI+SVD={host_p:+.3f}\n"
          f"  (a) centering only (diagonal) : {pa:+.3f}\n"
          f"  (b) full ZCA (diag+offdiag)   : {pb:+.3f}\n"
          f"  (c) ZCA rank-8 (denoise)      : {pc:+.3f}\n"
          f"  (d) ZCA rank-16               : {pd:+.3f}", flush=True)
    return {"seed": seed, "host": host_p, "a": pa, "b": pb, "c": pc, "d": pd}


def main():
    rows = [run_seed(s) for s in (42, 43, 44)]
    def m(x): return float(np.mean([r[x] for r in rows]))
    print(f"\n  MEAN (3): host {m('host'):+.3f} | (a)centering {m('a'):+.3f} | (b)full-ZCA {m('b'):+.3f} | "
          f"(c)ZCA-r8 {m('c'):+.3f} | (d)ZCA-r16 {m('d'):+.3f}", flush=True)
    best_off = max(m("b"), m("c"), m("d"))
    if best_off >= m("a") + 0.06:
        print(f"  GO (off-diagonal closes the gap): full/low-rank ZCA ({best_off:+.3f}) beats centering-only "
              f"({m('a'):+.3f}) by >=+0.06 -> the residual IS cross-neuron decorrelation -> test the existing "
              f"graded_lateral (analog off-diagonal whitening) on the bridge (non-gated, no dendrites).",
              flush=True)
    else:
        print(f"  NEGATIVE (off-diagonal is NOT the gap): ZCA ({best_off:+.3f}) ~= centering-only ({m('a'):+.3f}); "
              f"the +0.31->{m('host'):+.3f} residual is PPMI/SVD-specific (log-ratio + denoise), NOT decorrelation "
              f"-> neither graded_lateral NOR dendrites close it; the diagonal front-end is the linear ceiling.",
              flush=True)


if __name__ == "__main__":
    main()
