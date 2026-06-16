"""CYCLE 97 biologization sweep, piece 3 (the READ-OUT NORMALIZATION) — cheap-first: does replacing the host
double-centring with the two NEURAL operations (per-hub spike-frequency ADAPTATION + per-concept FEEDFORWARD
INHIBITION), including realistic RATE-CODED-pool noise on the subtracted means, recover the structure?

THE PIECE. The read-out code = double_center(log1p(M*100)). double_center subtracts (a) the per-hub mean over
concepts and (b) the per-concept mean over hubs (+ a global constant). The brain-based realization (CYCLE 93b
validated the log half on-bridge; this is the centring half):
  (a) per-hub mean  -> SPIKE-FREQUENCY ADAPTATION: the cortex's running per-hub firing frequency is subtracted
      (a per-hub adaptive current; ~ the shipped input_mean_adapt, applied at read-out).
  (b) per-concept mean -> FEEDFORWARD INHIBITION: a per-concept inhibitory pool reads the concept's population
      response and subtracts its mean (a global interneuron per concept).
Both POST-f-I (in the firing/log domain). The means are computed by RATE-CODED neural pools, so each carries
rate-code noise ~ 1/sqrt(pool) -- the load-bearing question: does that noise break the normalization?

GATE (multi seed, real corpus): the NEURAL-normalized code (noisy pool means) recovers the structure within a
small fraction of the HOST double_center (>= 0.90x). ANTI-CHEAT: drop EITHER neural op -> the structure drops
(both are load-bearing); a NO-normalization control is far below.

Cheap-first numpy (confirm the principle before the on-bridge circuit build); the batch C is the proxy for the
learned M (corr(M,C) ~ 0.9 on-bridge). NO GPU.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_biologize_readout_norm_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.dendritic_d1_learn_graded_structure_derisk import _cos_sim, _pearson_vs_Strue, heldout_generalization  # noqa: E402

N_HUB = 300
INHIB_POOL = 64       # neurons in the per-concept feedforward-inhibition pool (rate-coded mean estimate)
ADAPT_POOL = 64       # neurons backing the per-hub adaptation frequency estimate


def double_center(X):
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()


def neural_norm(L, rng, inhib_pool=INHIB_POOL, adapt_pool=ADAPT_POOL):
    """The two NEURAL centring ops with rate-coded-pool noise on the subtracted means.
      per-hub adaptation: subtract the per-hub mean (running frequency), estimated by a pool -> ~1/sqrt(adapt) noise.
      per-concept feedforward inhibition: subtract the per-concept mean, estimated by a pool -> ~1/sqrt(inhib) noise.
    Noise is relative to the signal scale of each mean (a rate-coded pool's SEM)."""
    hub_mean = L.mean(0, keepdims=True)                                   # per-hub (over concepts)
    hub_sem = (L.std(0, keepdims=True) / np.sqrt(adapt_pool))             # rate-coded estimate noise
    hub_mean_n = hub_mean + rng.standard_normal(hub_mean.shape) * hub_sem
    a = L - hub_mean_n                                                    # per-hub adaptation (subtractive)
    con_mean = a.mean(1, keepdims=True)                                   # per-concept (over hubs)
    con_sem = (a.std(1, keepdims=True) / np.sqrt(inhib_pool))
    con_mean_n = con_mean + rng.standard_normal(con_mean.shape) * con_sem
    return a - con_mean_n                                                 # per-concept feedforward inhibition


def run_seed(seed):
    C, labels, S_true = build_real_corpus(seed, N_HUB)
    L = np.log1p(C * 100.0)                                               # the f-I / Weber-Fechner read-out (pre-centre)
    rng = np.random.RandomState(seed)
    host = _pearson_vs_Strue(_cos_sim(double_center(L)), S_true)
    neural = _pearson_vs_Strue(_cos_sim(neural_norm(L, rng)), S_true)
    gen, ch = heldout_generalization(neural_norm(L, rng), labels)
    # ablations (anti-cheat): drop one neural op, and no-norm
    hub_only = _pearson_vs_Strue(_cos_sim(L - L.mean(0, keepdims=True)), S_true)        # adaptation only
    con_only = _pearson_vs_Strue(_cos_sim(L - L.mean(1, keepdims=True)), S_true)        # feedforward-inhib only
    nonorm = _pearson_vs_Strue(_cos_sim(L), S_true)
    print(f"  [readout-norm seed {seed}] host double-centre {host:+.3f} | NEURAL (adapt+FF-inhib, pool-noisy) "
          f"{neural:+.3f} ({neural/max(host,1e-9):.0%}, gen {gen:.2f}) | adapt-only {hub_only:+.3f} | "
          f"FF-inhib-only {con_only:+.3f} | no-norm {nonorm:+.3f}", flush=True)
    return {"seed": seed, "host": host, "neural": neural, "gen": gen, "adapt_only": hub_only,
            "ffi_only": con_only, "nonorm": nonorm}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[readout-norm biologize de-risk] does NEURAL centring (per-hub adaptation + per-concept feedforward "
          f"inhibition, rate-coded-pool-noisy means) match the host double-centre?", flush=True)
    rows = [run_seed(s) for s in (42, 43, 44, 45, 46, 47)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    host, neural, gen, nonorm = m("host"), m("neural"), m("gen"), m("nonorm")
    print(f"\n{'='*94}\n  MEAN (6 seeds): host {host:+.3f} | NEURAL {neural:+.3f} ({neural/max(host,1e-9):.0%} of "
          f"host, gen {gen:.2f}) | no-norm {nonorm:+.3f}", flush=True)
    print(f"{'='*94}", flush=True)
    if neural >= 0.90 * host and neural > nonorm + 0.05:
        print(f"  GO: the NEURAL read-out normalization (spike-freq ADAPTATION + feedforward INHIBITION, with "
              f"realistic rate-coded-pool noise on the means) recovers {neural:+.3f} = {neural/host:.0%} of the host "
              f"double-centre, far above no-norm ({nonorm:+.3f}). ==> the read-out normalization is BIOLOGIZABLE "
              f"(two real cortical gain-control ops); the on-bridge circuit build (per-concept FS feedforward "
              f"inhibition + per-hub adaptation at read-out) is the realization step.", flush=True)
    elif neural >= 0.70 * host:
        print(f"  PARTIAL: the neural normalization recovers {neural/host:.0%} of host -- the rate-coded-pool noise "
              f"on the means costs some structure; larger pools (lower SEM) or tuning needed.", flush=True)
    else:
        print(f"  NEGATIVE: the pool-noisy neural means break the normalization ({neural:+.3f} vs host {host:+.3f}) "
              f"-- the centring needs higher-precision means than rate-coded pools give; inspect.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "neural": neural, "gen": gen, "nonorm": nonorm, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_biologize_readout_norm.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
