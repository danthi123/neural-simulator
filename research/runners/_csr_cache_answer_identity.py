"""GO-bar gate (2026-06-19): the OneBrainComposer CSR cache (enable_csr_cache, default ON) must be ANSWER-IDENTICAL to
the stock cache-off composer across the FULL who/what matrix INCLUDING the no-confab abstentions, AND cache-invalidation
must be correct (store after a cached read; reconsolidation in-place rewrite; clause + multi-hop). Plus the speedup at
K in {8,16,32}.

This A/B builds a CACHED composer and a STOCK composer at the same seed/D, hears the same facts, and asserts bit-for-bit
identical answers. It is the anti-regression gate for the latency cache: a cache that changed an answer is a bug.

Run:  SIM_BACKEND=cupy python -u -m research.runners._csr_cache_answer_identity        (GPU for speed numbers)
      SIM_BACKEND=numpy python -u -m research.runners._csr_cache_answer_identity       (CPU; answer-identity only)
"""
from __future__ import annotations
import time
import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from research.runners.rf_phasor_composer import Clause
from sim.backend import is_gpu_backend

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]


def _sync():
    if is_gpu_backend():
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def _med(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn(); _sync()
    ts = []
    for _ in range(n):
        _sync(); t0 = time.perf_counter(); fn(); _sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def _pair(seed, D, k_max):
    cached = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=k_max, enable_csr_cache=True)
    stock = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=k_max, enable_csr_cache=False)
    return cached, stock


def _full_matrix(comp, agents, actions, pats):
    """Every who/what/yes-no answer + the no-confab abstentions, as one comparable tuple-of-tuples."""
    rows = []
    for (a, v, p) in zip(agents, actions, pats):
        rows.append(("qp", a, v, comp.query_patient(a, v)))
        rows.append(("qa", v, p, comp.query_agent(v, p)))
        rows.append(("yn", a, v, p, comp.ask_yes_no(a, v, p)))
        rows.append(("rf", a, comp.render_fact(a)))
    # abstentions (unstored cues/facts -- the moat)
    rows.append(("qp_moat", comp.query_patient("apple", "stop")))           # unstored cue -> None
    rows.append(("qa_moat", comp.query_agent("swim", "home")))              # unstored -> None
    rows.append(("yn_moat", comp.ask_yes_no("cat", "go", "west")))          # unstored fact -> unknown/no
    rows.append(("rf_moat", comp.render_fact("river")))                     # unknown agent -> None (if river unstored)
    return tuple(rows)


def main():
    print(f"backend: {'cupy (GPU)' if is_gpu_backend() else 'numpy (CPU)'}")
    SEED, D = 42, 64
    agents = ["dog", "cat", "bird"]; actions = ["go", "come", "look"]; pats = ["north", "east", "south"]

    # ---- (1) ANSWER-IDENTITY across the full matrix incl. abstentions ----
    cached, stock = _pair(SEED, D, k_max=32)
    for c in (cached, stock):
        for (a, v, p) in zip(agents, actions, pats):
            c.store(a, v, p)
    mc = _full_matrix(cached, agents, actions, pats)
    ms = _full_matrix(stock, agents, actions, pats)
    ok1 = (mc == ms)
    print(f"\n(1) ANSWER-IDENTITY (full who/what matrix + abstentions, cached vs stock): "
          f"{'BIT-FOR-BIT IDENTICAL' if ok1 else 'MISMATCH'}")
    if not ok1:
        for a, b in zip(mc, ms):
            if a != b:
                print(f"    cached={a}  stock={b}")
    # explicit abstention report
    print(f"    abstentions: query_patient(apple,stop)={cached.query_patient('apple','stop')!r} "
          f"ask_yes_no(cat,go,west)={cached.ask_yes_no('cat','go','west')!r} "
          f"render_fact(river)={cached.render_fact('river')!r}")

    # ---- (2) CACHE-INVALIDATION: store AFTER a cached read ----
    cached2, stock2 = _pair(SEED, D, k_max=32)
    for c in (cached2, stock2):
        c.store("dog", "go", "north")
    _ = cached2.query_patient("dog", "go")               # prime the cache (n=1 unbind/clean + store CSR)
    for c in (cached2, stock2):
        c.store("cat", "come", "east")                   # n grows -> must invalidate (new cache key) + store dirty
    ok2 = (cached2.query_patient("dog", "go") == stock2.query_patient("dog", "go") == "north"
           and cached2.query_patient("cat", "come") == stock2.query_patient("cat", "come") == "east")
    print(f"\n(2) CACHE-INVALIDATION (store after a cached read): "
          f"{'CORRECT (no stale read)' if ok2 else 'STALE / WRONG'}  "
          f"[dog.go={cached2.query_patient('dog','go')!r}, cat.come={cached2.query_patient('cat','come')!r}]")

    # ---- (3) RECONSOLIDATION (in-place rewrite, same n) must be reflected + not corrupt the unbind/clean cache ----
    cached3, stock3 = _pair(SEED, D, k_max=32)
    for c in (cached3, stock3):
        c.store("dog", "go", "north"); c.store("cat", "come", "east")
    _ = cached3.query_patient("dog", "go")               # prime cache at n=2
    rc = cached3.update_on_mismatch("dog", "go", "south")
    rs = stock3.update_on_mismatch("dog", "go", "south")
    ok3 = (rc["action"] == rs["action"] == "rewrite"
           and cached3.query_patient("dog", "go") == stock3.query_patient("dog", "go") == "south"
           and cached3.count_facts("dog", "go") == 1
           and cached3.query_patient("cat", "come") == "east")        # the OTHER fact is unchanged (cache not corrupted)
    print(f"\n(3) RECONSOLIDATION (in-place rewrite, same n -> reuse unbind/clean, rebuild store): "
          f"{'CORRECT (update reflected, cache intact)' if ok3 else 'WRONG'}  "
          f"[dog.go {cached3.query_patient('dog','go')!r} (was north), cat.come {cached3.query_patient('cat','come')!r}, "
          f"count={cached3.count_facts('dog','go')}]")

    # ---- (4) CLAUSE + MULTI-HOP answer-identity (the per-block/decode paths, cache-on vs cache-off) ----
    cc, sc = _pair(SEED, D, k_max=32)
    for c in (cc, sc):
        c.store("dog", "go", Clause(agent="cat", action="look", patient="south"))
        c.store("cat", "look", "river")
    ok4 = (cc.query_patient("dog", "go") == sc.query_patient("dog", "go") == "cat look south"
           and cc.render_fact("dog") == sc.render_fact("dog") == "dog go cat look south")
    # multi-hop chain (fresh pair to avoid clause confound)
    ch_c, ch_s = _pair(SEED, D, k_max=32)
    for c in (ch_c, ch_s):
        c.store("dog", "go", "cat"); c.store("cat", "go", "north")
    ok4 = ok4 and (ch_c.query_chain("dog", ["go", "go"]) == ch_s.query_chain("dog", ["go", "go"]) == "north"
                   and ch_c.query_chain("dog", ["go", "come"]) is None)
    print(f"\n(4) CLAUSE + MULTI-HOP answer-identity (cache-on vs cache-off): "
          f"{'IDENTICAL' if ok4 else 'MISMATCH'}  "
          f"[clause qp={cc.query_patient('dog','go')!r}, chain={ch_c.query_chain('dog',['go','go'])!r}]")

    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"\n=== ANSWER-IDENTITY + INVALIDATION: {'ALL PASS' if all_ok else 'FAIL'} ===")

    # ---- (5) SPEEDUP at K in {8,16,32} (GPU only) ----
    if is_gpu_backend():
        big = ["dog", "cat", "apple", "river"]; bigv = ["go", "run", "come", "stop"]; bigp = ["north", "south", "east", "west"]
        print("\n(5) SPEEDUP (_read_all_blocks: stock cache-off vs cached, GPU):")
        for K in (8, 16, 32):
            comp = OneBrainComposer(seed=SEED, D=128, vocab=VOCAB, k_max=32, enable_csr_cache=True)
            for i in range(K):
                comp.store(big[i % 4], bigv[(i // 4) % 4], bigp[i % 4])
            _sync()
            comp.enable_csr_cache = False
            t_stock = _med(lambda: comp._read_all_blocks(), n=5, warmup=2)
            comp.enable_csr_cache = True
            comp._read_all_blocks()  # prime
            t_cached = _med(lambda: comp._read_all_blocks(), n=5, warmup=2)
            print(f"    K={K:2d}: stock {t_stock:7.1f} ms | cached {t_cached:7.1f} ms | speedup {t_stock/t_cached:5.1f}x "
                  f"{'PASS(>=4x)' if t_stock/t_cached >= 4.0 else 'FAIL(<4x)'}")
    else:
        print("\n(5) speedup skipped (numpy backend; run with SIM_BACKEND=cupy for GPU timings)")


if __name__ == "__main__":
    main()
