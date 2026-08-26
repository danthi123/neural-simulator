"""Flat-distinct test: does distinct-seed retraining give DISTINCT FLAT codes that compose ROBUSTLY on
STRUCTURED facts (no nesting wall)?

The hierarchical-320 shortcut failed on structured facts (0.0/0.95/1.0 seed-variable -- the nesting wall).
The honest path: retrain bridges with DISTINCT seeds so their codes differ WITHOUT a 2nd binding level.
Here: bridgeA nouns (seed 42, existing), bridgeB verbs (seed 43), bridgeC adj (seed 44). Capture the 192
FLAT codes, verify distinct, then run STRUCTURED SVO composition (agent=noun / action=verb / patient=adj)
over them at composition seeds 42/43/44 -- the SAME structured test that exposed the hierarchical failure.

If structured full-3-slot QA is robust (>= 0.80 at ALL seeds, esp. seed 42 where hierarchical = 0.000) ->
the flat-distinct path is the honest route to robust full-vocab biological composition. Codes cached.
Reuse-by-import; no protected-module change; no autograd. Run (GPU): python -m research.findings.raw._insubstrate_flatdistinct_test
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_real_substrate_qa_probe as Q
import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
import research.runners.concept_pool_sparse_distributed as SP
from sim.backend import get_backend

R = "research/findings/raw"; G = f"{R}/g11_bg"
# (bank, bridge checkpoint, vocab file, build-seed) -- DISTINCT seeds -> distinct patterns -> distinct codes
SPEC = [("noun", f"{G}/g20_sparse_bridges_320/bridgeA_nouns_sparse64.simstate.h5", f"{G}/g20_bridgeA_nouns_vocab64.txt", 42),
        ("verb", f"{R}/_flatdist_bridgeB_seed43.simstate.h5", f"{G}/g20_bridgeB_verbs_vocab64.txt", 43),
        ("adj",  f"{R}/_flatdist_bridgeC_seed44.simstate.h5", f"{G}/g20_bridgeC_adj_vocab64.txt", 44),
        ("spatial", f"{R}/_flatdist_bridgeD_seed45.simstate.h5", f"{G}/g20_bridgeD_spatial_vocab64.txt", 45),
        ("functional", f"{R}/_flatdist_bridgeE_seed46.simstate.h5", f"{G}/g20_bridgeE_functional_vocab64.txt", 46)]
CACHE = f"{R}/_flatdist320_codes.npz"


def _mc(v):
    v = np.asarray(v, dtype=np.float64); v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-12)


def main():
    Q.STIM = 300; Q.SPARSITY = 0.007
    xp, backend = get_backend()
    D = Q.N_POOL
    for _, bp, _, _ in SPEC:
        if not os.path.exists(bp):
            print(f"CANNOT-CONCLUDE: {bp} missing (retrain not done)", flush=True); return

    if os.path.exists(CACHE):
        d = np.load(CACHE); codes = {w: d[w] for w in d.files if not w.startswith("_")}
        words = list(d["_words"]); bank_of = {w: b for w, b in zip(d["_words"], d["_banks"])}
        print(f"=== flat-distinct (loaded {len(words)} cached codes, backend={backend}) ===", flush=True)
    else:
        print(f"=== flat-distinct: capturing 320 FLAT codes (3 distinct-seed bridges, backend={backend}) ===",
              flush=True)
        codes = {}; words = []; bank_of = {}
        for bank, bp, vp, sd in SPEC:
            vw = Q.load_vocab(vp)
            bridge = SP.build_sparse_pool_bridge(seed=sd, n_lang_input=Q.N_LANG, n_shared_pool=Q.N_POOL,
                                                 n_lang_output=Q.N_LANG, verbose=False)
            bridge.load_checkpoint(bp)
            cap = Q.capture_real_codes(bridge, vw, sd, xp)   # FLAT codes (no bridge-role bind)
            for w in vw:
                key = w if w not in codes else f"{bank[0]}_{w}"
                codes[key] = _mc(cap[w]); words.append(key); bank_of[key] = bank
            print(f"  captured {len(vw)} {bank} (total {len(words)})", flush=True)
            del bridge
            try:
                import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        np.savez_compressed(CACHE, _words=np.array(words), _banks=np.array([bank_of[w] for w in words]),
                            **codes)

    import itertools
    btw = [float(np.dot(codes[a], codes[b])) for a, b in itertools.islice(itertools.combinations(words, 2), 30000)]
    print(f"  320-wide between-concept cos: mean {np.mean(btw):.3f}  max {np.max(btw):.3f}  "
          f"({'DISTINCT' if np.max(btw) < 0.9 else 'DUPLICATES REMAIN'})", flush=True)

    nouns = [w for w in words if bank_of[w] == "noun"]
    verbs = [w for w in words if bank_of[w] == "verb"]
    adjs = [w for w in words if bank_of[w] == "adj"]
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    print("  STRUCTURED SVO composition (agent=noun / action=verb / patient=adj), full-3-slot QA:", flush=True)
    results = []
    for seed in [42, 43, 44]:
        rng = np.random.default_rng(seed)
        roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
        roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
        bb, bidx = P.build(seed, D, xp)
        ok = tot = 0
        for _ in range(20):
            f = {"agent": rng.choice(nouns), "action": rng.choice(verbs), "patient": rng.choice(adjs)}
            b = RM.bind_fact_spiking(bb, bidx, f, codes, roles, D, xp)
            g = {r: RM.unbind_spiking(bb, bidx, b, r, roles, codes, words, D, xp) for r in RM.ROLES}
            ok += int(all(g[r] == f[r] for r in RM.ROLES)); tot += 1
        results.append(ok / tot)
        print(f"    seed {seed}: {ok/tot:.3f}", flush=True)
    mean = float(np.mean(results))
    print(f"\nRESULT: structured SVO full-3-slot QA = {results} (mean {mean:.3f})  "
          f"[hierarchical was 0.000/0.950/1.000]", flush=True)
    if min(results) >= 0.80:
        print("VERDICT: RESOLVES -- distinct FLAT codes compose ROBUSTLY on structured facts at ALL seeds "
              "(incl. seed 42 where hierarchical = 0.000). The flat-distinct path avoids the nesting wall. "
              "Full 320-concept robust biological composition (SVO over noun/verb/adj, cleanup over all 320 incl. spatial/functional distractors).", flush=True)
    else:
        print(f"VERDICT: structured QA min {min(results):.2f} -- flat-distinct helps but is not uniformly "
              "robust; characterize the residual.", flush=True)


if __name__ == "__main__":
    main()
