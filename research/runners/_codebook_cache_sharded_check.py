"""Bounded check: the codebook cache SHARED across shards (via the existing _dg_index_source graft) stays
byte-identical AND does not multiply RSS by S. Simulates the sim-side diff by binding cached-cleanup methods
onto each shard and pointing every non-base shard's cache at the base (exactly what _dg_index_source already
does for the DG index)."""
from __future__ import annotations
import os, sys, time, json, resource
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.sharded_phasor_store import ShardedPhasorStore
from research.runners.tiered_fact_store import encode_fast


def _ensure_cache(sh):
    src = sh._dg_index_source if getattr(sh, "_dg_index_source", None) is not None else sh
    V = len(src.words)
    if getattr(src, "_cb_cache_V", -1) != V or getattr(src, "_cb_frac", None) is None:
        src._cb_frac = np.stack([np.asarray(src.concepts[w], dtype=float) for w in src.words])
        src._cb_z = np.exp(2j * np.pi * src._cb_frac)
        src._cb_cache_V = V
    sh._cb_frac = src._cb_frac
    sh._cb_z = src._cb_z


def _cleanup_cached(sh, rec_phases, words=None):
    if sh.enable_spiking_cleanup or (sh.enable_sparse_index and words is None) or words is not None:
        return RFPhasorComposer._cleanup(sh, rec_phases, words)
    _ensure_cache(sh)
    scores = np.cos(2.0 * np.pi * (np.asarray(rec_phases)[None, :] - sh._cb_frac)).sum(axis=1)
    return sh.words[int(np.argmax(scores))]


def _cleanup_all_cached(sh, rec, words=None):
    if (sh.enable_sparse_index and words is None) or words is not None:
        return RFPhasorComposer._cleanup_all(sh, rec, words)
    if len(rec) == 0:
        return []
    _ensure_cache(sh)
    rec_z = np.exp(2j * np.pi * np.asarray(rec))
    sims = (rec_z @ sh._cleanup_conj(sh._cb_z).T).real
    return [sh.words[int(j)] for j in np.argmax(sims, axis=1)]


def _bind_cached(store):
    import types
    for sh in store.shards:
        sh._cleanup = types.MethodType(_cleanup_cached, sh)
        sh._cleanup_all = types.MethodType(_cleanup_all_cached, sh)


def main():
    V, K, S, D, seed = 8000, 2000, 40, 128, 42
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(V)]
    agents = [f"w{int(rng.integers(0, V))}" for _ in range(K)]
    actions = [f"w{int(rng.integers(0, V))}" for _ in range(K)]
    patients = [f"w{int(rng.integers(0, V))}" for _ in range(K)]

    base_store = ShardedPhasorStore(n_shards=S, seed=seed, D=D, vocab=vocab, share_codebook=True)
    cached_store = ShardedPhasorStore(n_shards=S, seed=seed, D=D, vocab=vocab, share_codebook=True)
    for a, ac, p in zip(agents, actions, patients):
        for st in (base_store, cached_store):
            sh = st.shard_for(a)
            sh.kb.append(({"agent": a, "action": ac, "patient": p, "polarity": "AFFIRM"}, encode_fast(sh, {"agent": a, "action": ac, "patient": p, "polarity": "AFFIRM"})))
    _bind_cached(cached_store)

    # byte-identity across shards
    mism = 0
    for i in range(min(200, K)):
        a, ac, p = agents[i], actions[i], patients[i]
        if base_store.query_patient(a, ac) != cached_store.query_patient(a, ac):
            mism += 1
        if base_store.ask_yes_no(a, ac, p) != cached_store.ask_yes_no(a, ac, p):
            mism += 1

    # cache-sharing proof: every non-base shard's _cb_z is the SAME object as the base's
    src = cached_store.shards[0]
    _ensure_cache(src)
    for sh in cached_store.shards:
        _ensure_cache(sh)
    shared_ok = all(sh._cb_z is src._cb_z for sh in cached_store.shards)
    n_distinct = len({id(sh._cb_z) for sh in cached_store.shards})

    # latency on a hot-ish shard
    def timeit(store, n=8):
        for i in range(3):
            store.query_patient(agents[i], actions[i])
        lat = []
        for i in range(n):
            t = time.perf_counter(); store.query_patient(agents[i % K], actions[i % K]); lat.append(time.perf_counter() - t)
        return round(float(np.median(lat)) * 1000, 1)

    out = {
        "V": V, "K": K, "S": S, "D": D,
        "byte_mismatches": mism,
        "cache_shared_across_all_shards": shared_ok,
        "n_distinct_codebook_objects": n_distinct,
        "cb_z_MB_per_object": round(src._cb_z.nbytes / 1e6, 1),
        "projected_MB_if_unshared": round(src._cb_z.nbytes * S / 1e6, 1),
        "actual_MB_shared": round(src._cb_z.nbytes / 1e6, 1),
        "latency_baseline_ms": timeit(base_store),
        "latency_cached_ms": timeit(cached_store),
        "peak_rss_MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
