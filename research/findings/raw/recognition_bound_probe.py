"""Recognition-bound probe -- the routing from the compositional line's
convergent finding.

The whole compositional line converges on one bound: the substrate's
concept-recognition accuracy. Per-observation recognition of a concept
word (the per-pool argmax of a single 100-step activity capture) is
only about 0.66-0.74, and the substrate's trial-to-trial activity
coefficient of variation is about 1.6.

This probe asks, cheaply, whether that bound is reducible. It reuses
the real captured substrate activity (the activity-level integration
cache: 16 per-neuron observations of each of 16 concept words, three
seeds) -- no new GPU run -- and tests three things:

(a) TEMPORAL AVERAGING. Does averaging the activity over several
    observations of a word, before the per-pool argmax, raise
    recognition? This is the biological analogue of a longer
    integration window / sustained attention -- a noisy rate code
    averaged over time. Swept over K = 1, 2, 4, 8, 16 observations.
(b) WORD FRAGILITY. Does the single-observation recognition error
    concentrate on specific words, or spread uniformly? Concentrated
    error means a few irreducibly-fragile concept representations;
    uniform error means a general noise floor.
(c) CAPTURE DRIFT. Does recognition degrade across the 16-observation
    capture sequence (observation 1 versus observation 16)? The
    substrate is stepped continuously through the capture; if its
    internal state drifts, later observations would be worse, and the
    measured per-observation noise would be partly an artifact of the
    capture protocol rather than intrinsic.

PRE-REGISTERED reading (fixed; never tuned):
- If temporal averaging lifts recognition materially -- to a multi-seed
  mean at or above 0.85 by K = 16 -- then the recognition bound is
  reducible by integration: a longer-integration recognition front-end
  is the build, and the compositional capability can be lifted without
  changing the substrate.
- If averaging does not reach 0.85 and the error is concentrated on
  specific words, the bound is the substrate's concept representation
  itself (specific concepts are irreducibly entangled), and the
  routing is to the concept-pool architecture (the v14/v16 line), not
  to a better readout.

Standalone numpy, ENGINEERING ceiling-clarification (non-load-bearing).
Reuses the activity cache by import. No protected/frozen/moat module
touched. No automatic differentiation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.activity_level_integration import (
    capture_seed, CACHE_DIR,
)
from research.runners.unified_per_regime_monitor_runner import (
    _direct_pool_target,
)

SEEDS = [42, 43, 44]
K_VALUES = [1, 2, 4, 8, 16]
N_SAMPLES = 60               # random K-subsets sampled per word per K
BAR = 0.85                   # pre-registered recognition target for averaging
M_OBS = 16


def recognize(activity, slices, all_pools):
    """The substrate's own readout: the concept pool with the highest
    mean per-neuron firing in this activity vector."""
    best_pool, best_rate = None, -1.0
    for p in all_pools:
        s, e = slices[p]
        rate = float(np.mean(activity[s:e]))
        if rate > best_rate:
            best_rate, best_pool = rate, p
    return best_pool


def run_one_seed(seed, rng):
    print(f"\n--- seed {seed} ---")
    cache_path = os.path.join(CACHE_DIR, f"full_seed{seed}.npz")
    obs, _clean, slices, all_pools, words = capture_seed(
        seed, cache_path, M_OBS)
    target = {w: _direct_pool_target(w) for w in words}

    # (a) Temporal averaging: recognition vs K.
    per_k = {}
    for k in K_VALUES:
        n_ok = n_tot = 0
        for w in words:
            for _ in range(N_SAMPLES):
                idx = rng.choice(M_OBS, size=k, replace=False)
                avg = obs[w][idx].mean(axis=0)
                n_ok += int(recognize(avg, slices, all_pools) == target[w])
                n_tot += 1
        per_k[k] = n_ok / n_tot

    # (b) Per-word single-observation recognition rate.
    per_word = {}
    for w in words:
        hits = sum(recognize(obs[w][i], slices, all_pools) == target[w]
                   for i in range(M_OBS))
        per_word[w] = hits / M_OBS

    # (c) Capture-drift: recognition rate at each observation index.
    per_obs_idx = []
    for i in range(M_OBS):
        hits = sum(recognize(obs[w][i], slices, all_pools) == target[w]
                   for w in words)
        per_obs_idx.append(hits / len(words))

    print(f"  temporal averaging K->recognition: "
          + "  ".join(f"K{k}={per_k[k]:.3f}" for k in K_VALUES))
    fragile = sorted(per_word.items(), key=lambda kv: kv[1])[:4]
    print(f"  most-fragile words (single-obs): "
          + ", ".join(f"{w}={r:.2f}" for w, r in fragile))
    print(f"  capture-drift (obs 0 vs obs 15 recognition): "
          f"{per_obs_idx[0]:.3f} vs {per_obs_idx[-1]:.3f}")
    return {
        "seed": seed,
        "recognition_by_k": {str(k): per_k[k] for k in K_VALUES},
        "per_word_single_obs": per_word,
        "recognition_by_obs_index": per_obs_idx,
    }


def main():
    print("=== recognition-bound probe ===")
    print(f"seeds {SEEDS}; temporal-averaging K={K_VALUES}; "
          f"samples/word/K={N_SAMPLES}; recognition target={BAR}")
    rng = np.random.default_rng(42)
    results = [run_one_seed(s, rng) for s in SEEDS]

    print(f"\n=== MULTI-SEED AGGREGATE ===")
    agg_k = {}
    for k in K_VALUES:
        m = float(np.mean([r["recognition_by_k"][str(k)] for r in results]))
        agg_k[k] = m
        print(f"  K={k:>2}: recognition {m:.4f}")

    # Word fragility: a word is fragile if its mean single-obs
    # recognition is below 0.5 across seeds.
    all_words = list(results[0]["per_word_single_obs"].keys())
    word_mean = {w: float(np.mean([r["per_word_single_obs"][w]
                                   for r in results]))
                 for w in all_words}
    fragile_words = sorted([w for w in all_words if word_mean[w] < 0.5],
                           key=lambda w: word_mean[w])
    robust_words = [w for w in all_words if word_mean[w] >= 0.9]
    drift = [float(np.mean([r["recognition_by_obs_index"][i]
                            for r in results])) for i in range(M_OBS)]

    print(f"  fragile words (mean single-obs recognition < 0.5): "
          f"{len(fragile_words)}/{len(all_words)} "
          f"{[(w, round(word_mean[w], 2)) for w in fragile_words]}")
    print(f"  robust words (>= 0.9): {len(robust_words)}/{len(all_words)}")
    print(f"  capture-drift across obs index: first {drift[0]:.3f} -> "
          f"last {drift[-1]:.3f} (slope "
          f"{(drift[-1] - drift[0]):+.3f})")

    averaging_lifts = agg_k[16] >= BAR
    error_concentrated = len(fragile_words) >= 3

    print(f"\n=== VERDICT ===")
    if averaging_lifts:
        verdict = "TEMPORAL_AVERAGING_LIFTS_RECOGNITION"
        print(f"  Temporal averaging lifts recognition to "
              f"{agg_k[16]:.3f} >= {BAR} by K=16. The recognition bound is "
              f"reducible by integration -- a longer-integration "
              f"recognition front-end can lift the compositional "
              f"capability without changing the substrate.")
    elif error_concentrated:
        verdict = "RECOGNITION_BOUND_IS_WORD_FRAGILITY"
        print(f"  Temporal averaging reaches only {agg_k[16]:.3f} < {BAR} "
              f"by K=16, and the error concentrates on "
              f"{len(fragile_words)} fragile words. The bound is the "
              f"substrate's concept representation itself -- specific "
              f"concepts are irreducibly entangled; routes to the "
              f"concept-pool architecture (the v14/v16 line).")
    else:
        verdict = "RECOGNITION_BOUND_IS_UNIFORM_NOISE_FLOOR"
        print(f"  Temporal averaging reaches {agg_k[16]:.3f} < {BAR} by "
              f"K=16 and the error is spread roughly uniformly -- a "
              f"general noise floor, not specific fragile concepts. "
              f"Routes to reducing the substrate's activity noise.")

    out = {
        "seeds": SEEDS, "k_values": K_VALUES, "n_samples": N_SAMPLES,
        "bar": BAR,
        "recognition_by_k": {str(k): agg_k[k] for k in K_VALUES},
        "word_mean_single_obs": word_mean,
        "fragile_words": fragile_words,
        "n_robust_words": len(robust_words),
        "capture_drift": drift,
        "per_seed": results,
        "verdict": verdict,
    }
    with open("research/findings/raw/recognition_bound_probe.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/recognition_bound_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
