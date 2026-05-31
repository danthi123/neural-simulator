"""THROWAWAY cheap-first probe (CPU/numpy, stdlib+numpy only, NO protected
import): does GRID-LIKE MODULAR REDUNDANT coding escape the DG separation-vs-
reliability BOUNDARY that a SINGLE k-WTA stage could not?

Context: the DG boundary (finding 2026-05-31-DG-...-FUNDAMENTAL-BOUNDARY) is
that ONE competitive sparse-coding k-WTA stage cannot give between-concept
SEPARATION and within-concept RELIABILITY from overlapping inputs (the
substrate's concept activity overlaps at cosine ~0.82). Grid cells achieve
robust high-capacity coding via MULTIPLE INDEPENDENT MODULES + REDUNDANT
decoding (tolerate some modules being noisy, like an error-correcting code).
Hypothesis: M independent k-WTA modules, decoded by MAJORITY across modules,
might thread separation (accumulated across modules) AND reliability (rescued
by redundancy) where M=1 (a single DG) cannot.

Apples-to-apples with the DG boundary: SAME cached substrate activity the DG
probes drove (activity_level_integration_cache/denoise64_seed{N}.npz, obs__<word>,
storage = first 32 obs mean, query = last 32 obs mean), just projected through
modular k-WTA in numpy instead of a spiking DG. M=1 is the single-DG control
(reproduces the boundary).

Matched capacity: total expansion N_TOTAL split into M modules of N_TOTAL/M dims;
total active K_TOTAL split into K_TOTAL/M winners per module. So M=1 = one big
module with K_TOTAL winners (the DG-4000 sparse regime, stable-but-unseparated).

Two decoders:
  - concat-cosine: cosine of the full concatenated binary winner-vectors (a
    concept's query-half matched to the storage concept with max cosine).
  - majority-vote (the REDUNDANT / grid-cell decode): each module votes for the
    storage concept with max winner-overlap; the concept with the most module
    votes wins. This is the error-correcting decode that could escape.

FROZEN three-state (set before the run, never tuned): RESOLVES if some M gives
multi-seed identity accuracy (majority-vote) >= 0.80 AND within >= 0.60 AND
between <= 0.50, where M=1 fails the same bars. BOUNDARY if no M threads both
(modular redundancy does NOT escape -> the boundary is deeper than module count).
DOES-NOT-RESOLVE / CANNOT-CONCLUDE on instrument-invalid.
"""
from __future__ import annotations
import os
import sys
import numpy as np

CACHE_DIR = "research/findings/raw/activity_level_integration_cache"
CACHE_TAG = "denoise64"
SEEDS = [42, 43, 44]
N_TOTAL = 4000
K_TOTAL = 200          # M=1 sparsity 200/4000 = 0.05 (DG stable-but-unseparated regime)
M_LIST = [1, 2, 4, 8, 16, 32]
ID_BAR = 0.80
WITHIN_BAR = 0.60
BETWEEN_BAR = 0.50


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _rectify_norm(v):
    v = np.maximum(v.astype(np.float64), 0.0)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _encode(act, projs, k_mod):
    """Return (concat binary winner-vector, list of per-module winner index-sets)."""
    parts, sets = [], []
    for P in projs:
        s = P @ act
        if k_mod >= s.shape[0]:
            win = np.arange(s.shape[0])
        else:
            win = np.argpartition(-s, k_mod)[:k_mod]
        v = np.zeros(P.shape[0], dtype=np.float64)
        v[win] = 1.0
        parts.append(v)
        sets.append(set(int(i) for i in win))
    return np.concatenate(parts), sets


PROJ_DENSITY = float(os.environ.get("PROJ_DENSITY", "0.05"))  # 0.05 = sparse (DG-like); 1.0 = dense gaussian


def _projection(n_out, d_in, rng):
    """Random projection. Sparse (DG-like, PROJ_DENSITY<1) preserves more input
    correlation -> less separation (faithful-er to the spiking DG); dense (=1)
    decorrelates more."""
    P = rng.standard_normal((n_out, d_in))
    if PROJ_DENSITY < 1.0:
        P = P * (rng.random((n_out, d_in)) < PROJ_DENSITY)
    return P


def eval_seed_M(store_act, query_act, words, M, seed):
    rng = np.random.default_rng(7919 * seed + 31 * M)
    d_act = store_act[words[0]].shape[0]
    n_mod = N_TOTAL // M
    k_mod = max(1, K_TOTAL // M)
    projs = [_projection(n_mod, d_act, rng) for _ in range(M)]

    store_code, store_sets = {}, {}
    query_code, query_sets = {}, {}
    for w in words:
        store_code[w], store_sets[w] = _encode(store_act[w], projs, k_mod)
        query_code[w], query_sets[w] = _encode(query_act[w], projs, k_mod)

    # within-concept reliability (store-half vs query-half concat cosine)
    within = float(np.mean([_cos(store_code[w], query_code[w]) for w in words]))
    # between-concept separation (store vs store, different concepts)
    btw = []
    for i, a in enumerate(words):
        for b in words[i + 1:]:
            btw.append(_cos(store_code[a], store_code[b]))
    between = float(np.mean(btw)) if btw else 0.0

    # identity decode 1: concat cosine (query w -> argmax store c)
    cc_correct = 0
    for w in words:
        sims = [(_cos(query_code[w], store_code[c]), c) for c in words]
        if max(sims, key=lambda t: t[0])[1] == w:
            cc_correct += 1
    id_concat = cc_correct / len(words)

    # identity decode 2: per-module MAJORITY VOTE (redundant / error-correcting)
    mv_correct = 0
    for w in words:
        votes = {}
        for m in range(M):
            qs = query_sets[w][m]
            best_c, best_ov = None, -1
            for c in words:
                ov = len(qs & store_sets[c][m])
                if ov > best_ov:
                    best_ov, best_c = ov, c
            votes[best_c] = votes.get(best_c, 0) + 1
        pred = max(votes.items(), key=lambda t: t[1])[0]
        if pred == w:
            mv_correct += 1
    id_majority = mv_correct / len(words)

    return {"M": M, "n_mod": n_mod, "k_mod": k_mod, "within": within,
            "between": between, "id_concat": id_concat, "id_majority": id_majority}


def load_seed(seed):
    cache = os.path.join(CACHE_DIR, "%s_seed%d.npz" % (CACHE_TAG, seed))
    if not os.path.exists(cache):
        return None
    data = np.load(cache)
    words = [k[len("obs__"):] for k in data.files if k.startswith("obs__")]
    store = {w: _rectify_norm(data["obs__" + w][:32].mean(axis=0)) for w in words}
    query = {w: _rectify_norm(data["obs__" + w][32:].mean(axis=0)) for w in words}
    return words, store, query


def main():
    seeds = [s for s in SEEDS
             if os.path.exists(os.path.join(CACHE_DIR, "%s_seed%d.npz" % (CACHE_TAG, s)))]
    print("=== MODULAR REDUNDANT CODING vs the DG boundary (cheap numpy) ===", flush=True)
    print("seeds=%s N_TOTAL=%d K_TOTAL=%d M_LIST=%s" % (seeds, N_TOTAL, K_TOTAL, M_LIST),
          flush=True)
    if not seeds:
        print("VERDICT: CANNOT-CONCLUDE (no caches)", flush=True)
        return

    # instrument: raw between-concept cosine on the substrate activity (must be ~0.82)
    w0, s0, _ = load_seed(seeds[0])
    raw_btw = np.mean([_cos(s0[a], s0[b]) for i, a in enumerate(w0) for b in w0[i + 1:]])
    print("[instrument] raw substrate between-concept cosine = %.3f (DG-input regime ~0.82); "
          "%d concepts, d_act=%d" % (raw_btw, len(w0), s0[w0[0]].shape[0]), flush=True)

    agg = {M: {"within": [], "between": [], "id_concat": [], "id_majority": []} for M in M_LIST}
    for seed in seeds:
        words, store, query = load_seed(seed)
        for M in M_LIST:
            r = eval_seed_M(store, query, words, M, seed)
            for k in ("within", "between", "id_concat", "id_majority"):
                agg[M][k].append(r[k])

    print("\n%-4s %-7s %-7s | %-8s %-8s | %-10s %-12s" %
          ("M", "n_mod", "k_mod", "within", "between", "id_concat", "id_majority"), flush=True)
    means = {}
    for M in M_LIST:
        n_mod = N_TOTAL // M
        k_mod = max(1, K_TOTAL // M)
        mw = np.mean(agg[M]["within"]); mb = np.mean(agg[M]["between"])
        mc = np.mean(agg[M]["id_concat"]); mm = np.mean(agg[M]["id_majority"])
        means[M] = (mw, mb, mc, mm)
        print("%-4d %-7d %-7d | %-8.3f %-8.3f | %-10.3f %-12.3f" %
              (M, n_mod, k_mod, mw, mb, mc, mm), flush=True)

    # --- frozen verdict ---
    m1_id = max(means[1][2], means[1][3])
    m1_threads = (means[1][3] >= ID_BAR and means[1][0] >= WITHIN_BAR
                  and means[1][1] <= BETWEEN_BAR)
    threads = [M for M in M_LIST if M > 1
               and means[M][3] >= ID_BAR and means[M][0] >= WITHIN_BAR
               and means[M][1] <= BETWEEN_BAR]
    best_id_M = max(M_LIST, key=lambda M: means[M][3])
    print("\n[M=1 single-DG control] id(best of concat/majority) = %.3f within=%.3f between=%.3f"
          % (m1_id, means[1][0], means[1][1]), flush=True)
    print("[best majority id] M=%d -> %.3f (within %.3f, between %.3f)" %
          (best_id_M, means[best_id_M][3], means[best_id_M][0], means[best_id_M][1]), flush=True)
    # INSTRUMENT VALIDITY FIRST: the M=1 reproduce-the-failure control MUST fail
    # for a M>1 pass to mean anything. If M=1 already threads, the probe cannot
    # test modular escape (the clean numpy code is not faithful to the spiking DG).
    if m1_threads:
        print("VERDICT: CANNOT-CONCLUDE (instrument-invalid) -- the M=1 single-DG CONTROL already "
              "threads (within>=%.2f AND between<=%.2f AND id>=%.2f), so it does NOT reproduce the "
              "spiking DG boundary failure. A clean rate-based random-projection k-WTA separates+"
              "stabilizes the substrate activity regardless of M or projection density -> the DG "
              "boundary is SPIKING-DYNAMICS-specific, not a property of sparse-coding the activity. "
              "Modular escape is UNTESTABLE here (need the spiking substrate where M=1 actually fails)."
              % (WITHIN_BAR, BETWEEN_BAR, ID_BAR), flush=True)
    elif threads:
        print("VERDICT: RESOLVES -- modular redundant coding threads the boundary at M=%s "
              "(majority id >= %.2f AND within >= %.2f AND between <= %.2f) where M=1 (%.3f) does not. "
              "-> grid-like modular coding is a candidate biological escape; justifies a spiking "
              "grid-module build (cheap-first PASSED)." % (threads, ID_BAR, WITHIN_BAR, BETWEEN_BAR, m1_id),
              flush=True)
    else:
        print("VERDICT: BOUNDARY -- no M threads majority-id>=%.2f AND within>=%.2f AND between<=%.2f. "
              "Modular REDUNDANCY does NOT escape the DG separation-vs-reliability boundary on this "
              "substrate activity; the boundary is deeper than module count. -> strengthens the honest "
              "conclusion: the overlapping substrate activity cannot be grounded into a separable+stable "
              "symbol by ANY single-stage competitive sparse code (modular or not). Accept the oracle as "
              "an engineering component; advance the validated P4 stack." % (ID_BAR, WITHIN_BAR, BETWEEN_BAR),
              flush=True)


if __name__ == "__main__":
    main()
