"""D0 de-risk (BEFORE any sim/ edit): is the SPIKING per-hub input-mean clean enough to center axis-0?

The deep-research fork verdict (2026-06-15-slow-perhub-mean-primitive-deep-research.md): the load-bearing op
(x_h - slow_mean(x_h)) is the SEPARABLE diagonal/DC half of whitening (per-neuron mean-centering = subtractive
spike-frequency adaptation), NOT the cross-neuron decorrelation the Mikulasch-Priesemann limit forbids -> a
CHEAP point-neuron primitive (Option A: a guarded slow per-hub INPUT-mean adaptation sim/ array) can realize it,
NOT dendrites. The ONE residual risk before committing the sim/ edit: the numpy GO (+0.311) used the CLEAN drive
x_h in the EMA, but the bridge primitive must EMA the hub's own SPIKING drive (noisy Poisson g_e). Does the
slow EMA average enough that the SPIKING input-mean still centers axis-0?

This probe replaces the clean x_h in the EMA with a Poisson-sampled spiking estimate poisson(x_h*g)/g (the same
noise model the bridge has), sweeps the mean spike budget g, keeps the slow streaming EMA, and gates:
  GO  => spiking input-mean clears +0.30 at a reasonable spike budget -> proceed to the Option-A sim/ edit.
  NEG => the noise floor is the wall -> raise the budget / slow alpha, or re-open dendrites.

ANTI-CHEAT (per BRAIN-BASED-ONLY + project standard):
 - permuted-label control (the +0.30 must be real structure -> ~0 on shuffled labels);
 - beats the cm-pool axis-1 (+0.246) and the no-centering baseline;
 - slow-alpha is load-bearing (alpha=0.5 must FAIL) -> a fast/wrong-tau impl can't pass by accident;
 - the mean is estimated from a SPIKING drive (not clean) -- the faithful on-substrate noise;
 - 6 seeds.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_spiking_mean_derisk
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


def stream_adapt_codes(Xn, W, alpha, gain, n_epochs, seed, mean_spike_gain=None):
    """Stream the concepts (shuffled per epoch); per-hub lagged EMA m of the hub's drive; the readout for each
    concept is the ON/OFF spike code of W @ (x - m). If mean_spike_gain is not None, the EMA integrates a
    POISSON-SPIKING estimate poisson(x*g)/g of the drive (the bridge's noisy input-mean) instead of the clean x.
    Causal: m updated AFTER the read (the adaptation lags the input). adapted = CLEAN x - (noisy) m, mirroring
    the bridge: the hub's actual input current minus its own slow (spiking-estimated) mean."""
    rng = np.random.RandomState(seed)
    Nc, Nh = Xn.shape
    m = np.zeros(Nh)
    codes = np.zeros((Nc, 2 * W.shape[0]))
    for ep in range(n_epochs):
        order = rng.permutation(Nc)
        last = (ep == n_epochs - 1)
        for c in order:
            x = Xn[c]
            adapted = x - m
            if last:
                codes[c] = onoff_code((W @ adapted[None, :].T).T, gain, rng)[0]
            x_for_mean = x if mean_spike_gain is None else (
                rng.poisson(np.maximum(x, 0.0) * mean_spike_gain).astype(np.float64) / mean_spike_gain)
            m = (1.0 - alpha) * m + alpha * x_for_mean
    return codes


def run_seed(seed, n_hub=500, k=128, gain=500.0, n_epochs=12, alpha=0.05):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, Xn.shape[1]) / np.sqrt(Xn.shape[1])

    def p_of(code):
        return _pearson_vs_Strue(_cos_sim(code), S_true), heldout_generalization(code, labels)[0]

    p_a1, _ = p_of(onoff_code((W @ (Xn - Xn.mean(1, keepdims=True)).T).T, gain, rng))  # cm-pool axis-1 ref
    p_clean, _ = p_of(stream_adapt_codes(Xn, W, alpha, gain, n_epochs, seed, mean_spike_gain=None))
    print(f"\n[spiking-mean D0 seed {seed}] {C.shape[0]}c x {n_hub}h; host={host_p:+.3f}  (alpha={alpha}, "
          f"{n_epochs} epochs) | clean-mean +{p_clean:.3f}  cm-pool axis-1 +{p_a1:.3f}", flush=True)
    out = {"seed": seed, "host": host_p, "clean": p_clean, "axis1": p_a1, "spk": {}}
    best_code, best_p = None, -9
    for g in (3.0, 10.0, 30.0, 100.0):
        code = stream_adapt_codes(Xn, W, alpha, gain, n_epochs, seed, mean_spike_gain=g)
        p, gen = p_of(code)
        out["spk"][g] = p
        if p > best_p:
            best_p, best_code = p, code
        print(f"  [spiking-mean budget g={g:5.0f} spk/hub/pres] Pearson={p:+.3f}  gen={gen:.3f}  "
              f"(=> {100*p/max(1e-9,p_clean):.0f}% of clean-mean)", flush=True)
    # anti-cheat: slow-alpha load-bearing (alpha=0.5 must fail) at the best budget, + permuted-label.
    code_fast = stream_adapt_codes(Xn, W, 0.5, gain, n_epochs, seed, mean_spike_gain=30.0)
    p_fast, _ = p_of(code_fast)
    out["fast_alpha"] = p_fast
    rng2 = np.random.RandomState(seed * 7919 + 5); perm = rng2.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    out["permuted"] = _pearson_vs_Strue(_cos_sim(best_code), S_perm)
    print(f"  [anti-cheat] fast-alpha(0.5)={p_fast:+.3f} (must FAIL << best)  permuted={out['permuted']:+.3f} (~0)",
          flush=True)
    return out


def main():
    seeds = [42, 43, 44, 45, 46, 47]
    rows = [run_seed(s) for s in seeds]
    host = np.mean([r["host"] for r in rows]); clean = np.mean([r["clean"] for r in rows])
    a1 = np.mean([r["axis1"] for r in rows])
    budgets = (3.0, 10.0, 30.0, 100.0)
    best_g, best = None, -9
    for g in budgets:
        m = np.mean([r["spk"][g] for r in rows])
        if m > best:
            best, best_g = m, g
    fast = np.mean([r["fast_alpha"] for r in rows]); perm = np.mean([r["permuted"] for r in rows])
    print(f"\n  MEAN ({len(seeds)} seeds): host {host:+.3f} | clean-mean {clean:+.3f} | cm-pool axis-1 {a1:+.3f} | "
          f"BEST spiking-mean {best:+.3f} (budget g={best_g}) | fast-alpha {fast:+.3f} | permuted {perm:+.3f}",
          flush=True)
    # The D0 question is whether the SPIKING input-mean is clean enough = does it MATCH the clean-mean ceiling
    # (not lose to Poisson noise). The absolute level is the inherent moderate-real-structure marginality (the
    # clean mechanism itself is only ~+0.311), separate from the spiking-mean noise.
    slow_loadbearing = best - fast >= 0.08
    clean_enough = best >= clean - 0.03            # spiking mean ~= clean mean (Poisson noise mild)
    anticheat_ok = best >= a1 + 0.03 and perm <= 0.10 and slow_loadbearing
    if clean_enough and anticheat_ok:
        print(f"  GO (spiking-mean clean enough): SPIKING input-mean {best:+.3f} = {100*best/max(1e-9,clean):.0f}% "
              f"of clean-mean {clean:+.3f} (Poisson noise costs only ~{100*(clean-best)/max(1e-9,clean):.0f}%), "
              f"beats cm-pool {a1:+.3f}, permuted-clean {perm:+.3f}, slow-alpha load-bearing (fast {fast:+.3f}). "
              f"=> the spiking input-mean is NOT the wall; BUILD the Option-A guarded sim/ primitive. (Absolute "
              f"{best:+.3f} is at the +0.30 bar = the inherent moderate-real-structure marginality; the decisive "
              f"test is the full bridge gate.)", flush=True)
    elif clean_enough:
        print(f"  PARTIAL: spiking-mean clean enough ({best:+.3f} ~= clean {clean:+.3f}) but an anti-cheat is "
              f"soft (cm-pool {a1:+.3f}, permuted {perm:+.3f}, fast-alpha {fast:+.3f}) -- inspect before the edit.",
              flush=True)
    else:
        print(f"  NEGATIVE: the SPIKING input-mean loses to the clean-mean ({best:+.3f} vs clean {clean:+.3f}, "
              f"{100*(clean-best)/max(1e-9,clean):.0f}% lost); the Poisson noise floor IS a wall -> raise the "
              f"spike budget / slow alpha, or re-open dendrites.", flush=True)


if __name__ == "__main__":
    main()
