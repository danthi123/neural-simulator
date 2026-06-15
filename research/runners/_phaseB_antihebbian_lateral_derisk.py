"""Does a ONE-STEP anti-Hebbian LATERAL (the L1 SM rule's missing piece) decorrelate the cortex outputs +
recover toward host -- the cheap-first de-risk for the proper Phase 3 (the deep-research recommendation)?

The L1-SM-on-spiking research (2026-06-15-L1-SM-on-spiking-deep-research.md): plain feedforward STDP is only
the W (Hebbian) half of similarity-matching -> rank-1 collapse (the bridge saw eff-rank 1.5). The missing half
is the recurrent ANTI-HEBBIAN LATERAL M that DECORRELATES the outputs (Pehlevan-Chklovskii 2015/2018; Foldiak
1990). The EXISTING `graded_lateral` (sim/bridge.py:1776) implements exactly `dM ~ <aa^T> - I - lambda*M` on
ANALOG membrane (sidestepping the rate-code wall). Caveat: its target I = FULL ZCA, which the off-diagonal
de-risk showed COLLAPSES (-0.012); low-rank/partial whitening reaches host +0.44 -> a TUNING problem (raise
lambda / lower the target beta for gentler/partial whitening).

This numpy de-risk: feedforward random projection a = W @ (centered input); learn the anti-Hebbian lateral M
online (dM = lr(<aa^T> - beta*I) - lambda*M); read y = a - M@a (one-step lateral); ON/OFF spike code of y.
Sweep beta (the whitening target) so partial whitening (beta<1 or strong lambda) recovers toward host. GATE:
the lateral code BEATS the random projection (no lateral) toward host (+0.44). Anti-cheat: learned M beats
fixed-ZERO M (learning load-bearing); permuted ~0; eff-rank >> 1.5.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_antihebbian_lateral_derisk
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
    _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def poisson_spk(rate, gain, rng):
    return rng.poisson(np.maximum(rate, 0.0) * gain).astype(np.float64)


def onoff_code(drive, gain, rng):
    on = np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(drive))])
    off = np.array([poisson_spk(np.maximum(-drive[i], 0.0), gain, rng) for i in range(len(drive))])
    return np.concatenate([on, off], axis=1)


def learn_lateral(A, beta, lam, lr, n_epochs, seed):
    """Online anti-Hebbian lateral M (k x k, zero diagonal): dM = lr(<aa^T> - beta*I) - lam*M, applied per
    concept over a shuffled stream. Returns the converged M. The one-step output is y = a - M@a."""
    rng = np.random.RandomState(seed * 13 + 1)
    Nc, k = A.shape
    M = np.zeros((k, k))
    for _ep in range(n_epochs):
        for c in rng.permutation(Nc):
            a = A[c]
            y = a - M @ a                      # one-step lateral (the graded_lateral mechanism)
            # anti-Hebbian on the OUTPUT y (decorrelate the outputs toward beta*I):
            M += lr * (np.outer(y, y) - beta * np.eye(k)) - lam * M
            np.fill_diagonal(M, 0.0)           # no self-inhibition (the lateral is off-diagonal)
    return M


def run_seed(seed, n_hub=500, k=128, gain=500.0, n_epochs=8):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0)); Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xw = Xn - Xn.mean(0, keepdims=True)
    W = rng.randn(k, n_hub) / np.sqrt(n_hub)
    A = (W @ Xw.T).T                            # the projected activity (the "cortex g_e"), signed

    def p_of(Y):
        code = onoff_code(Y, gain, rng)
        return (_pearson_vs_Strue(_cos_sim(code), S_true), heldout_generalization(code, labels)[0],
                effective_rank(code))

    p_rand, g_rand, er_rand = p_of(A)           # NO lateral = the random projection (the +0.31 baseline)
    print(f"\n[anti-Hebbian lateral seed {seed}] {C.shape[0]}c x {n_hub}h; host={host_p:+.3f} | "
          f"random (no lateral) +{p_rand:.3f} (eff-rank {er_rand:.1f})", flush=True)
    out = {"seed": seed, "host": host_p, "random": p_rand, "grid": {}}
    best = (-9, None)
    for beta in (0.05, 0.2, 0.5):
        for lam in (0.02, 0.1, 0.3):
            M = learn_lateral(A, beta, lam, lr=0.02, n_epochs=n_epochs, seed=seed)
            Y = A - (M @ A.T).T
            p, g, er = p_of(Y)
            out["grid"][f"b{beta}_l{lam}"] = round(p, 3)
            if p > best[0]:
                best = (p, (beta, lam, g, er))
    bp, (bb, bl, bg, ber) = best
    out["best"] = {"pearson": round(bp, 3), "beta": bb, "lam": bl, "gen": round(bg, 3), "eff_rank": round(ber, 1)}
    print(f"  BEST lateral: Pearson={bp:+.3f} (beta={bb}, lam={bl}, gen={bg:.3f}, eff-rank={ber:.1f}) "
          f"vs random {p_rand:+.3f}", flush=True)
    # anti-cheat: permuted-label on the best lateral code.
    M = learn_lateral(A, bb, bl, lr=0.02, n_epochs=n_epochs, seed=seed); Y = A - (M @ A.T).T
    code = onoff_code(Y, gain, rng)
    rng2 = np.random.RandomState(seed * 999 + 1); perm = rng2.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    out["permuted"] = round(_pearson_vs_Strue(_cos_sim(code), S_perm), 3)
    print(f"  [anti-cheat] permuted={out['permuted']:+.3f} (~0)", flush=True)
    return out


def main():
    rows = [run_seed(s) for s in (42, 43, 44)]
    def m(f): return float(np.mean([f(r) for r in rows]))
    host = m(lambda r: r["host"]); rand = m(lambda r: r["random"])
    best = m(lambda r: r["best"]["pearson"]); perm = m(lambda r: r["permuted"])
    er = m(lambda r: r["best"]["eff_rank"])
    print(f"\n  MEAN (3 seeds): host {host:+.3f} | random(no lateral) {rand:+.3f} | "
          f"BEST anti-Hebbian lateral {best:+.3f} (eff-rank {er:.1f}) | permuted {perm:+.3f}", flush=True)
    if best >= rand + 0.06 and perm <= 0.10:
        print(f"  GO: the anti-Hebbian LATERAL beats the random projection ({best:+.3f} vs {rand:+.3f}) toward "
              f"host ({host:+.3f}), eff-rank {er:.1f} (no rank-1 collapse), permuted-clean ({perm:+.3f}). => the "
              f"SM lateral IS the fix; BUILD it on the bridge via graded_lateral on the cortex (+ tune lambda). "
              f"The learning is load-bearing (random has NO lateral).", flush=True)
    elif best >= rand:
        print(f"  PARTIAL: the lateral helps ({best:+.3f} > random {rand:+.3f}) but by <+0.06 -- the one-step "
              f"approximation is weak; a full recurrent settle (y=(I+M)^-1 a) may be needed before the bridge.",
              flush=True)
    else:
        print(f"  NEGATIVE: the anti-Hebbian lateral does NOT beat the random projection ({best:+.3f} vs "
              f"{rand:+.3f}) -- the one-step lateral isn't the fix; escalate (full recurrent SM) before any edit.",
              flush=True)


if __name__ == "__main__":
    main()
