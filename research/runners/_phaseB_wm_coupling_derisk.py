"""The W-M COUPLING de-risk: is the bridge's failure the COUPLING (W must learn from the M-decorrelated
activity), not M's presence? (The decisive cheap-first test before the (A) guarded sim/ edit.)

The W-M coupling research (2026-06-15-WM-coupling-spiking-SM-deep-research.md): similarity-matching couples
feedforward W and lateral M through the SETTLED, M-DECORRELATED output -- W learns dW ~ y*x^T - y^2*W with
y=(I+M)^-1 Wx, and M adapts FASTER than W. The bridge's graded_lateral IS the correct M, but its feedforward W
is spike-timing STDP (the WRONG rule class) reading spike timing, NOT the decorrelated analog a -> potentiates
every co-active pair -> rank-1 collapse regardless of M. This reproduces EVERY failed Phase-3 attempt.

This numpy de-risk reproduces the bridge in numpy + tests the COUPLING specifically:
  - COUPLED   : W learns from the DECORRELATED a (a = relu(Wx - M@a), settled) -- the SM rule.
  - BROKEN    : W learns from the RAW Wx (M still on for the readout) -- the bridge's spike-STDP analogue.
  - RANDOM    : no W learning (frozen random W) -- the baseline.
GATE: COUPLED reaches low-rank host (+0.44, eff-rank -> ~8, beats random) while BROKEN collapses to rank-1
(reproducing the bridge) => the failure IS the coupling, and the fix (W-from-decorrelated-a) is validated -> the
(A) guarded sim/ edit (a feedforward plasticity mode reading the post-graded_lateral activity) is justified.
Anti-cheat: COUPLED beats RANDOM (learning load-bearing); BROKEN collapse = the load-bearing control; eff-rank
>> 1.5; permuted ~0; no host shortcut (W/M learned online from the bridge-faithful activity).

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_wm_coupling_derisk
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


def settle(Wx, M, k_settle=4):
    """y = (I+M)^-1 Wx approximated by k_settle damped fixed-point steps a <- Wx - M@a (the bridge's one-/few-
    step graded_lateral, not a matrix inverse)."""
    a = Wx.copy()
    for _ in range(k_settle):
        a = Wx - M @ a
    return a


def train_sm(Xw, k, seed, *, coupled, lr_W=0.01, lr_M=0.1, lam=0.1, n_epochs=20, k_settle=4):
    """Online SM. coupled=True: W learns from the decorrelated a (dW ~ a x^T - a^2 W). coupled=False (BROKEN):
    W learns from the RAW Wx (the bridge's spike-STDP analogue, M still decorrelates the readout). M is always
    learned anti-Hebbian + FASTER (lr_M >> lr_W)."""
    rng = np.random.RandomState(seed * 17 + 3)
    Nc, Nh = Xw.shape
    W = rng.randn(k, Nh) / np.sqrt(Nh)
    M = np.zeros((k, k))
    Ik = np.eye(k)
    for _ep in range(n_epochs):
        for c in rng.permutation(Nc):
            x = Xw[c]
            Wx = W @ x
            a = settle(Wx, M, k_settle)                       # decorrelated activity
            M += lr_M * (np.outer(a, a) - lam * Ik) - lr_M * lam * M  # anti-Hebbian, fast
            np.fill_diagonal(M, 0.0)
            nrm = np.linalg.norm(M)
            if nrm > 0.9:
                M *= 0.9 / nrm                                 # bounded M (well-conditioned I+M)
            drive = a if coupled else Wx                       # COUPLED: from decorrelated; BROKEN: from raw
            W += lr_W * (np.outer(drive, x) - (drive ** 2)[:, None] * W)   # Oja-bounded outer product
    return W, M


def run_seed(seed, n_hub=500, k=128, gain=500.0):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0)); Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xw = Xn - Xn.mean(0, keepdims=True)

    def readout(W, M):
        Y = np.array([settle(W @ Xw[c], M, 4) for c in range(len(Xw))])
        code = onoff_code(Y, gain, rng)
        return (_pearson_vs_Strue(_cos_sim(code), S_true), heldout_generalization(code, labels)[0],
                effective_rank(code), code)

    Wc, Mc = train_sm(Xw, k, seed, coupled=True)
    Wb, Mb = train_sm(Xw, k, seed, coupled=False)
    p_c, g_c, er_c, code_c = readout(Wc, Mc)
    p_b, g_b, er_b, _ = readout(Wb, Mb)
    Wr = rng.randn(k, n_hub) / np.sqrt(n_hub); p_r, g_r, er_r, _ = readout(Wr, np.zeros((k, k)))
    rng2 = np.random.RandomState(seed * 991 + 1); perm = rng2.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    pp = _pearson_vs_Strue(_cos_sim(code_c), S_perm)
    print(f"\n[W-M coupling seed {seed}] {C.shape[0]}c x {n_hub}h; host={host_p:+.3f}\n"
          f"  COUPLED (W from decorrelated a): {p_c:+.3f} (gen {g_c:.3f}, eff-rank {er_c:.1f})\n"
          f"  BROKEN  (W from raw Wx, M on)   : {p_b:+.3f} (gen {g_b:.3f}, eff-rank {er_b:.1f})\n"
          f"  RANDOM  (no W learning)        : {p_r:+.3f} (eff-rank {er_r:.1f})  | permuted(coupled) {pp:+.3f}",
          flush=True)
    return {"seed": seed, "host": host_p, "coupled": p_c, "coupled_er": er_c, "broken": p_b, "broken_er": er_b,
            "random": p_r, "permuted": pp}


def main():
    rows = [run_seed(s) for s in (42, 43, 44)]
    def m(f): return float(np.mean([f(r) for r in rows]))
    host = m(lambda r: r["host"]); cp = m(lambda r: r["coupled"]); br = m(lambda r: r["broken"])
    rd = m(lambda r: r["random"]); pp = m(lambda r: r["permuted"])
    cer = m(lambda r: r["coupled_er"]); ber = m(lambda r: r["broken_er"])
    print(f"\n  MEAN (3 seeds): host {host:+.3f} | COUPLED {cp:+.3f} (eff-rank {cer:.1f}) | BROKEN {br:+.3f} "
          f"(eff-rank {ber:.1f}) | RANDOM {rd:+.3f} | permuted {pp:+.3f}", flush=True)
    if cp >= 0.70 * host and cp >= rd + 0.06 and cp >= br + 0.06 and pp <= 0.10:
        print(f"  GO (the failure IS the coupling): COUPLED ({cp:+.3f}, eff-rank {cer:.1f}) reaches >=0.70x host "
              f"({host:+.3f}), BEATS random ({rd:+.3f}) AND the broken-coupling control ({br:+.3f}), permuted-"
              f"clean. => W-from-DECORRELATED-activity is the fix; the (A) guarded sim/ edit (a feedforward "
              f"plasticity mode reading the post-graded_lateral activity) is JUSTIFIED.", flush=True)
    elif cp >= br + 0.06:
        print(f"  PARTIAL: COUPLED ({cp:+.3f}) beats BROKEN ({br:+.3f}) -- coupling helps -- but doesn't reach "
              f"0.70x host ({0.70*host:+.3f}); inspect (settle depth, timescale ratio).", flush=True)
    else:
        print(f"  NEGATIVE: COUPLED ({cp:+.3f}) ~= BROKEN ({br:+.3f}) -- the coupling is NOT the (only) fix; "
              f"re-examine before any sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
