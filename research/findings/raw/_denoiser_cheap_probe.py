"""Cheap-first DENOISER probe for biologizing shortcut-2 (the oracle symbol
lookup) -- the activity-grounded-symbol arc.

The May-22 activity-level integration was NEGATIVE: a symbol derived from a
SINGLE observation of the substrate's per-neuron activity is too noisy
(CV ~1.63) to compose (integrated ~0.36 << 0.80). That finding re-specified
the path: a faithful activity-grounded symbol needs a DENOISER. The most
honest, biology-grounded denoiser is TEMPORAL INTEGRATION -- a sustained
encoding window = the MEAN over k observations (CV ~ CV_single / sqrt(k)).

This probe reuses the EXACT activity-level machinery byte-unchanged
(activity_level_integration.py cached activity + make_deriver + the
spiking_phasor_fhrr composition) and inserts ONLY the mean-of-k denoiser at
BOTH storage and query. It sweeps k and measures, per load {2,3,5}, the
multi-seed integrated + composition-only accuracy AND the mean-of-k activity
CV. Pure CPU/cached (no GPU, no substrate re-run). No protected/frozen/moat
edit. No autograd. ASCII only.

PRE-REGISTERED (frozen before reading results):
  - VIABLE (build the denoiser arc) if mean-of-k composition-only accuracy
    reaches >= 0.80 at any tested k AT loads {2,3,5}, OR the k-curve is
    clearly rising toward 0.80 with a feasible extrapolated k (<= ~64) AND
    the CV is falling toward ~0.20 as predicted (1.63/sqrt(k)).
  - NEGATIVE (the substrate noise is not reducible by temporal integration;
    the oracle lookup is irreducible on this substrate by averaging) if the
    k-curve is FLAT (no rise with k) or the CV does not fall as ~1/sqrt(k)
    (i.e. the noise is correlated across observations, not averageable).
  - The controller forms the official verdict + scrutinizes (a rising curve
    that only reaches 0.80 via bootstrap-overlap optimism is NOT viable).
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

K_LIST = [1, 2, 4, 8, 16]
N_TRIALS = 40  # per load per seed (cheap-first estimate; spiking FHRR is time-stepped)


def _meank(deriver, obs_word, k, rng, pool=None, replace=True):
    """Temporal-integration denoiser: derive a symbol from the MEAN of k
    observations (sustained encoding). k=1 reproduces the single-observation
    NEGATIVE baseline. With pool= + replace=False, samples k DISTINCT
    observations from that pool (no bootstrap overlap)."""
    m = obs_word.shape[0]
    if pool is None:
        pool = np.arange(m)
    if replace:
        idx = pool[rng.integers(0, len(pool), size=k)]
    else:
        kk = min(k, len(pool))
        idx = rng.choice(pool, size=kk, replace=False)
    return deriver(obs_word[idx].mean(axis=0)), idx


def _recognizes(obs_word, idx, slices, all_pools, target):
    """Does the mean-of-k activity recognize the target pool (argmax)?"""
    a = obs_word[idx].mean(axis=0)
    return ali.recognized_pool(a, slices, all_pools) == target


def eval_seed_k(seed, k, obs, clean, slices, all_pools, words,
                distinct=False):
    d_act = obs[words[0]].shape[1]
    deriver = ali.make_deriver(ali.N_DIM, d_act, ali.DERIV_SEED)
    m_obs = obs[words[0]].shape[0]
    # Distinct mode: storage symbols sample from the first half of the
    # observations, query symbols from the second half -> storage and query
    # mean-of-k share NO observation (removes the bootstrap-overlap optimism).
    if distinct:
        half = m_obs // 2
        store_pool = np.arange(0, half)
        query_pool = np.arange(half, m_obs)
        store_kw = dict(pool=store_pool, replace=False)
        query_kw = dict(pool=query_pool, replace=False)
    else:
        store_kw = dict(replace=True)
        query_kw = dict(replace=True)

    cue_words = [w for w in words
                 if ali._direct_pool_target(w).startswith(
                     ("noun_pool_", "verb_pool_"))]
    filler_words = [w for w in words
                    if ali._direct_pool_target(w).startswith(
                        "adjective_pool_")]

    # Registered clean-up vocabulary: mean over K_VOCAB obs (unchanged).
    vocab = {fw: deriver(obs[fw][:ali.K_VOCAB].mean(axis=0))
             for fw in filler_words}

    # CV of the mean-of-k symbol's underlying activity (diagnostic).
    n_active = max(1, d_act // len(all_pools))
    cvr = np.random.default_rng(seed + 7)
    cvs = []
    for w in words:
        rows = np.asarray([obs[w][cvr.integers(0, m_obs, size=k)].mean(axis=0)
                           for _ in range(24)])
        cvs.append(ali.activity_cv(rows, n_active))
    cv_meank = float(np.mean(cvs))

    net = SpikingPhasorFHRR(ali.N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    def tp(w):
        return ali._direct_pool_target(w)

    per_load = {}
    for load in ali.LOADS:
        nic = nit = ncc = nct = 0
        for _ in range(N_TRIALS):
            cues = list(qrng.choice(cue_words, size=load, replace=False))
            fills = list(qrng.choice(filler_words, size=load, replace=True))
            facts = []
            enc_syms = []
            for (c, f) in zip(cues, fills):
                cs, ci = _meank(deriver, obs[c], k, qrng, **store_kw)
                fs, fi = _meank(deriver, obs[f], k, qrng, **store_kw)
                facts.append((c, f, ci, fi))
                enc_syms.append((cs, fs))
            composite = net.encode(enc_syms)
            for (c, f, ci, fi) in facts:
                qs, qi = _meank(deriver, obs[c], k, qrng, **query_kw)
                recovered = net.query(composite, qs)
                sims = {fw: phase_similarity(recovered, vocab[fw])
                        for fw in filler_words}
                best = max(sims, key=sims.get)
                hit = (tp(best) == tp(f))
                nic += int(hit)
                nit += 1
                # composition-only: all three mean-of-k activities recognize.
                if (_recognizes(obs[c], ci, slices, all_pools, tp(c)) and
                        _recognizes(obs[f], fi, slices, all_pools, tp(f)) and
                        _recognizes(obs[c], qi, slices, all_pools, tp(c))):
                    ncc += int(hit)
                    nct += 1
        per_load[load] = {
            "integrated": nic / nit,
            "composition_only": (ncc / nct) if nct else float("nan"),
            "n_comp": nct,
        }
    return {"seed": seed, "k": k, "cv_meank": cv_meank, "per_load": per_load}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--distinct", action="store_true",
                    help="storage/query mean-of-k sample DISTINCT, non-"
                         "overlapping observations (removes bootstrap-overlap "
                         "optimism; caps k at m_obs//2)")
    ap.add_argument("--k-list", type=int, nargs="+", default=None)
    ap.add_argument("--capture-obs", type=int, default=0,
                    help="capture N observations via the GPU substrate to a "
                         "denoiseN cache (instead of the 16-obs cache) so "
                         "distinct mean-of-k can reach higher k (pins exact "
                         "k for L=3/L=5 without bootstrap overlap)")
    a = ap.parse_args()
    distinct = bool(a.distinct)
    cap = int(a.capture_obs)
    k_list = a.k_list if a.k_list else ([1, 2, 4, 8] if distinct else K_LIST)

    seeds = ali.SEEDS
    print("=== DENOISER cheap-first probe: temporal integration (mean-of-k) "
          "on activity-grounded symbols %s==="
          % ("[DISTINCT no-overlap] " if distinct else ""), flush=True)
    print("reuse: activity_level_integration cached activity (CV~1.63 single-"
          "obs) + byte-unchanged spiking_phasor_fhrr; bar=%.2f loads=%s seeds=%s"
          % (ali.BAR, ali.LOADS, seeds), flush=True)

    # Load (or capture, if --capture-obs) activity per seed.
    m_obs = cap if cap > 0 else ali.M_OBS
    tag = ("denoise%d" % cap) if cap > 0 else "full"
    loaded = {}
    for s in seeds:
        cache_path = os.path.join(ali.CACHE_DIR, "%s_seed%d.npz" % (tag, s))
        if cap == 0 and not os.path.exists(cache_path):
            print("MISSING cache %s -- cannot run on cache (need --capture-obs "
                  "for a GPU capture)." % cache_path, flush=True)
            return 1
        # capture_seed loads the cache if present, else captures m_obs via the
        # GPU substrate and caches (kill-safe: a re-run skips captured seeds).
        loaded[s] = ali.capture_seed(s, cache_path, m_obs)

    rows = []
    for k in k_list:
        per_seed = [eval_seed_k(s, k, *loaded[s], distinct=distinct)
                    for s in seeds]
        cv = float(np.mean([r["cv_meank"] for r in per_seed]))
        print("\n--- k=%d  (mean-of-k activity CV=%.3f; sqrt-law predict "
              "%.3f) ---" % (k, cv, 1.63 / (k ** 0.5)), flush=True)
        agg = {}
        for load in ali.LOADS:
            ints = [r["per_load"][load]["integrated"] for r in per_seed]
            comps = [r["per_load"][load]["composition_only"]
                     for r in per_seed]
            mi = float(np.mean(ints))
            mc = float(np.mean([c for c in comps if c == c]))
            agg[load] = {"mean_integrated": mi, "mean_comp_only": mc,
                         "per_seed_integrated": ints}
            print("  L=%d  integrated=%.3f  composition-only=%.3f  %s"
                  % (load, mi, mc,
                     "PASS" if mc >= ali.BAR else ""), flush=True)
        rows.append({"k": k, "cv_meank": cv, "agg": agg})

    out = {"probe": "denoiser_temporal_integration_cheap_first",
           "distinct": distinct,
           "bar": ali.BAR, "loads": list(ali.LOADS), "seeds": list(seeds),
           "k_list": k_list, "rows": rows}
    op = os.path.join(_HERE, "_denoiser_cheap_probe%s.json"
                      % ("_distinct" if distinct else ""))
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
