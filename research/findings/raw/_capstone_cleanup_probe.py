"""Capstone cheap test: does the ATTRACTOR cleanup fix the residual that
temporal integration alone could not?

The 64-obs rigorous confirmation showed temporal integration (mean-of-k)
denoises the VARIANCE (CV ~ 1.63/sqrt(k)) but composition PLATEAUS below the
0.80 bar for L>=3 (L3 ~0.69, L5 ~0.57 at k=32) with a SIMPLE argmax cleanup.
The residual is symbol QUALITY/separability, not variance. The May-22
'shortcuts 2+3 coupled' insight: a biological ATTRACTOR cleanup grounds AND
denoises, its fixed points being the clean stored patterns. So swap ONLY the
cleanup (simple argmax -> ResonateFireTPAM annealed attractor settle) at the
best denoiser point (k=32, 64-obs activity) and see if it lifts L=3/L=5.

Isolates the cleanup's effect: same 64-obs activity, same mean-of-k denoiser,
same SpikingPhasorFHRR composition; ONLY the cleanup differs. CPU (64-obs
cache + numpy FHRR + numpy TPAM fast-path). Reuse-by-import byte-unchanged:
spiking_phasor_fhrr, resonate_fire_fhrr (TPAM + validated anneal params),
activity_level_integration cache + deriver. No autograd. No protected/frozen/
moat edit. ASCII.

PRE-REGISTERED (frozen): the attractor cleanup RESOLVES the residual if it
lifts L=3 AND L=5 composition-only to >= 0.80 at k=32 (where simple argmax
gave 0.69 / 0.57). PARTIAL if it lifts them materially (>= +0.10) but not to
0.80. NULL if it does not lift them (the residual is fundamental separability,
an honest biology-translatable boundary). The controller forms the verdict.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import research.findings.raw.activity_level_integration as ali
from research.runners.spiking_phasor_fhrr import (
    SpikingPhasorFHRR, phase_similarity,
)
from research.runners.resonate_fire_fhrr import (
    ResonateFireTPAM, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)

K = 32          # the best denoiser point from the 64-obs run
N_TRIALS = 60   # per load per seed (TPAM settle is heavier than argmax)


def eval_seed(seed, obs, words):
    d_act = obs[words[0]].shape[1]
    deriver = ali.make_deriver(ali.N_DIM, d_act, ali.DERIV_SEED)
    m = obs[words[0]].shape[0]
    half = m // 2
    store_pool = np.arange(0, half)
    query_pool = np.arange(half, m)

    cue_words = [w for w in words
                 if ali._direct_pool_target(w).startswith(
                     ("noun_pool_", "verb_pool_"))]
    filler_words = [w for w in words
                    if ali._direct_pool_target(w).startswith(
                        "adjective_pool_")]

    # Registered vocabulary (cleanup targets), mean over K_VOCAB obs.
    vocab = {fw: deriver(obs[fw][:ali.K_VOCAB].mean(axis=0))
             for fw in filler_words}
    # Attractor cleanup built over the SAME filler vocab (ordered).
    tpam = ResonateFireTPAM([vocab[fw] for fw in filler_words])

    net = SpikingPhasorFHRR(ali.N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    def tp(w):
        return ali._direct_pool_target(w)

    def dk(word, pool):
        idx = qrng.choice(pool, size=min(K, len(pool)), replace=False)
        return deriver(obs[word][idx].mean(axis=0))

    per_load = {}
    for load in ali.LOADS:
        n_arg = n_tpam = n_tot = 0
        for _ in range(N_TRIALS):
            cues = list(qrng.choice(cue_words, size=load, replace=False))
            fills = list(qrng.choice(filler_words, size=load, replace=True))
            composite = net.encode([(dk(c, store_pool), dk(f, store_pool))
                                     for (c, f) in zip(cues, fills)])
            for (c, f) in zip(cues, fills):
                recovered = net.query(composite, dk(c, query_pool))
                # cleanup A: simple argmax over the vocabulary
                sims = {fw: phase_similarity(recovered, vocab[fw])
                        for fw in filler_words}
                arg_best = max(sims, key=sims.get)
                # cleanup B: annealed attractor settle (fast closed-form
                # transfer -- validated equivalent), then argmax overlap
                z, _ = tpam.settle_annealed(
                    recovered, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH,
                    ANNEAL_ITERS, fast=True)
                overlaps = np.abs(tpam.s.conj().T @ z)
                tpam_idx = int(np.argmax(overlaps))
                tpam_best = filler_words[tpam_idx]
                n_arg += int(tp(arg_best) == tp(f))
                n_tpam += int(tp(tpam_best) == tp(f))
                n_tot += 1
        per_load[load] = {"argmax": n_arg / n_tot,
                          "attractor": n_tpam / n_tot, "n": n_tot}
    return {"seed": seed, "per_load": per_load}


def main():
    seeds = ali.SEEDS
    print("=== CAPSTONE cleanup test: simple argmax vs ATTRACTOR cleanup at "
          "k=%d (64-obs activity) ===" % K, flush=True)
    print("anneal theta %.1f->%.1f over %d iters; bar=%.2f loads=%s seeds=%s"
          % (ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS, ali.BAR,
             ali.LOADS, seeds), flush=True)

    loaded = {}
    for s in seeds:
        cp = os.path.join(ali.CACHE_DIR, "denoise64_seed%d.npz" % s)
        if not os.path.exists(cp):
            print("MISSING 64-obs cache %s" % cp, flush=True)
            return 1
        obs, _clean, _sl, _ap, words = ali.capture_seed(s, cp, 64)
        loaded[s] = (obs, words)

    per_seed = [eval_seed(s, *loaded[s]) for s in seeds]
    print("\n=== 3-seed mean (composition; bar 0.80) ===", flush=True)
    agg = {}
    for load in ali.LOADS:
        arg = float(np.mean([r["per_load"][load]["argmax"] for r in per_seed]))
        att = float(np.mean([r["per_load"][load]["attractor"]
                             for r in per_seed]))
        agg[load] = {"argmax": arg, "attractor": att, "delta": att - arg}
        print("  L=%d  argmax=%.3f  attractor=%.3f  delta=%+.3f  %s"
              % (load, arg, att, att - arg,
                 "ATTRACTOR PASS" if att >= ali.BAR else ""), flush=True)
    out = {"probe": "capstone_cleanup_argmax_vs_attractor", "k": K,
           "bar": ali.BAR, "seeds": list(seeds),
           "anneal": [ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS],
           "agg": agg, "per_seed": per_seed}
    op = os.path.join(_HERE, "_capstone_cleanup_probe.json")
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
