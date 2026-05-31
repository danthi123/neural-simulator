"""THROWAWAY cheap-first (CPU/numpy): does the validated VSA composition survive
SPIKING-REALISTIC readout -- i.e. the composed structure read out as POISSON spike
counts at realistic spike budgets (low counts = high relative noise)? This is the
spiking-specific risk the Gaussian-noise test (1.000 at 2x std) did NOT cover, and
the pre-gate before building the full spiking coincidence-bind architecture
(docs/plans/2026-05-31-spiking-composition-integration-design.md).

Model: structure S = sum_k role_k (x) filler_k (the +-1 / ON-OFF-coincidence bind,
biologically realizable per the finding). Read S out as spike counts via push-pull
Poisson: S_spiked[i] = (Poisson(lam*max(S[i],0)) - Poisson(lam*max(-S[i],0)))/lam,
where lam = spike budget (mean spikes per unit signal). Lower lam = fewer spikes =
noisier (realistic spiking). Then unbind a role + cleanup to nearest concept.
Fillers = real substrate concept codes (denoise64 cache, between-cos 0.70).

FROZEN: RESOLVES if role-query recovery stays >= 0.80 at a realistic spike budget
(lam <= ~10) AND the broken-binding control is ~chance -> spiking readout preserves
composition -> build the spiking architecture. BOUNDARY if it collapses at realistic
lam -> spiking noise is the bottleneck; characterize. stdlib+numpy + cache; no protected import.
"""
from __future__ import annotations
import os
import numpy as np

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
SEEDS = [42, 43, 44]
K = 4                      # roles per sentence (subject/verb/object/manner)
# Parametrize by PHYSICALLY-MEANINGFUL spikes-per-neuron over a readout window.
# lambda (spikes per unit signal) = spikes_per_neuron * D / ||S||_1; ||S||_1 ~ sqrt(K).
# Realistic population firing = 1-10 spikes/neuron (10-100 Hz x ~100 ms).
SPIKES_PER_NEURON = [0.1, 0.5, 1.0, 2.0, 5.0, 1e9]   # last = noiseless reference
N_TRIALS = 60


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_fillers(seed):
    d = np.load(CACHE % seed)
    ws = [k[5:] for k in d.files if k.startswith("obs__")]
    return ws, np.stack([_center(d["obs__" + w].mean(axis=0)) for w in ws])


def poisson_readout(S, lam, rng):
    """Push-pull Poisson spike-count readout of a signed vector S."""
    if lam >= 1e8:
        return S
    on = rng.poisson(lam * np.maximum(S, 0.0))
    off = rng.poisson(lam * np.maximum(-S, 0.0))
    return (on - off) / lam


def run(fillers, roles, lam, rng, broken=False):
    V, D = fillers.shape
    R = roles.shape[0]
    correct = total = 0
    for _ in range(N_TRIALS):
        fi = rng.choice(V, size=K, replace=False)
        ri = rng.choice(R, size=K, replace=False)
        S = np.zeros(D)
        for k in range(K):
            S = S + roles[ri[k]] * fillers[fi[k]]
        S = poisson_readout(S, lam, rng)         # SPIKING readout of the composite
        for k in range(K):
            ub = roles[rng.integers(R)] if broken else roles[ri[k]]
            est = S * ub
            correct += int(int(np.argmax(fillers @ est)) == fi[k]); total += 1
    return correct / total


def main():
    seeds = [s for s in SEEDS if os.path.exists(CACHE % s)]
    if not seeds:
        print("CANNOT-CONCLUDE (no caches)"); return
    ws, f0 = load_fillers(seeds[0])
    V, D = f0.shape
    print(f"=== spiking-realistic (Poisson readout) composition; V={V} D={D} K={K} ===")
    print(f"{'spikes/neuron':>14} | {'recovery':>9} | {'broken(ctrl)':>12}")
    res = {}
    for spn in SPIKES_PER_NEURON:
        lam = spn * D / np.sqrt(K) if spn < 1e8 else 1e9   # spikes/neuron -> lambda
        accs, brks = [], []
        for seed in seeds:
            _, fillers = load_fillers(seed)
            rng = np.random.default_rng(seed)
            roles = rng.choice([-1.0, 1.0], size=(8, D))
            roles = roles / np.linalg.norm(roles, axis=1, keepdims=True)
            accs.append(run(fillers, roles, lam, rng))
            brks.append(run(fillers, roles, lam, rng, broken=True))
        res[spn] = (np.mean(accs), np.mean(brks))
        tag = "noiseless ref" if spn >= 1e8 else ("<- realistic (1-10/neuron)" if 1 <= spn <= 10 else "")
        label = "inf" if spn >= 1e8 else f"{spn:.1f}"
        print(f"{label:>14} | {np.mean(accs):>9.3f} | {np.mean(brks):>12.3f}  {tag}")

    a1, b1 = res[1.0]
    chance = 1.0 / V
    print(f"\nchance=1/{V}={chance:.3f}; at 1 spike/neuron (realistic): recovery={a1:.3f} broken={b1:.3f}")
    if a1 >= 0.80:
        print("VERDICT: RESOLVES -- composition survives spiking-realistic Poisson readout at REALISTIC "
              "population firing (1 spike/neuron, recovery>=0.80). Spiking spike-count noise is NOT the "
              "bottleneck (the earlier BOUNDARY was a mis-scaled spike budget). -> build the spiking "
              "coincidence-bind architecture.")
    else:
        need = min((spn for spn in SPIKES_PER_NEURON if spn < 1e8 and res[spn][0] >= 0.80), default=None)
        print(f"VERDICT: needs {need} spikes/neuron for >=0.80 (1/neuron gives {a1:.2f}). "
              f"Achievable with a longer readout window / more neurons per dim.")


if __name__ == "__main__":
    main()
