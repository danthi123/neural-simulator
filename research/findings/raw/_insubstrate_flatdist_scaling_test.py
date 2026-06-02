"""Composition SCALING past 320: does the flat-distinct biological composition keep resolving as more
distinct-seed bridges (more concepts) are added? Extends the validated 320 result toward 640.

Same structure as _insubstrate_flatdistinct320_test.py, but the bridge SPEC is the 5 base banks PLUS extra
distinct-seed banks (F@47, G@48, ... synthetic-labelled cz#### concepts). Only bridges whose checkpoint exists
are included, so the SAME test reports composition at whatever scale is currently trained (448 with F+G, 576
with +H+I, 640 with +J). SVO fillers stay noun/verb/adj (the real A/B/C banks); the extra banks are additional
CLEANUP DISTRACTORS -- the scaling-sensitive part (more concepts = wider cleanup = more chance of a wrong
match). Structured SVO + any-bank, cleanup over ALL N concepts, seeds 42/43/44, bar min>=0.80.

Reuse-by-import; regenerates per-bridge patterns from each bridge's build-seed (byte-identical to training);
no protected-module change; no autograd. GPU/CuPy. Run after the extra bridges are trained:
  python -m research.findings.raw._insubstrate_flatdist_scaling_test
"""
from __future__ import annotations
import itertools
import os
import numpy as np

import research.findings.raw._insubstrate_real_substrate_qa_probe as Q
import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
import research.runners.concept_pool_sparse_distributed as SP
from sim.backend import get_backend

R = "research/findings/raw"; G = f"{R}/g11_bg"
# (bank, checkpoint, vocab, build-seed). Base 5 (320) + extra distinct-seed banks toward 640.
SPEC_FULL = [
    ("noun",       f"{G}/g20_sparse_bridges_320/bridgeA_nouns_sparse64.simstate.h5", f"{G}/g20_bridgeA_nouns_vocab64.txt", 42),
    ("verb",       f"{R}/_flatdist_bridgeB_seed43.simstate.h5", f"{G}/g20_bridgeB_verbs_vocab64.txt", 43),
    ("adj",        f"{R}/_flatdist_bridgeC_seed44.simstate.h5", f"{G}/g20_bridgeC_adj_vocab64.txt", 44),
    ("spatial",    f"{R}/_flatdist_bridgeD_seed45.simstate.h5", f"{G}/g20_bridgeD_spatial_vocab64.txt", 45),
    ("functional", f"{R}/_flatdist_bridgeE_seed46.simstate.h5", f"{G}/g20_bridgeE_functional_vocab64.txt", 46),
    ("extraF",     f"{R}/_flatdist_bridgeF_seed47.simstate.h5", f"{G}/g20_bridgeF_extra_vocab64.txt", 47),
    ("extraG",     f"{R}/_flatdist_bridgeG_seed48.simstate.h5", f"{G}/g20_bridgeG_extra_vocab64.txt", 48),
    ("extraH",     f"{R}/_flatdist_bridgeH_seed49.simstate.h5", f"{G}/g20_bridgeH_extra_vocab64.txt", 49),
    ("extraI",     f"{R}/_flatdist_bridgeI_seed50.simstate.h5", f"{G}/g20_bridgeI_extra_vocab64.txt", 50),
    ("extraJ",     f"{R}/_flatdist_bridgeJ_seed51.simstate.h5", f"{G}/g20_bridgeJ_extra_vocab64.txt", 51),
]


def _mc(v):
    v = np.asarray(v, dtype=np.float64); v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-12)


def main():
    Q.STIM = 300; Q.SPARSITY = 0.007
    xp, backend = get_backend()
    spec = [s for s in SPEC_FULL if os.path.exists(s[1])]
    n_concepts = 64 * len(spec)
    print(f"=== flat-distinct SCALING: {len(spec)} bridges -> {n_concepts} concepts (backend={backend}) ===",
          flush=True)
    cache = f"{R}/_flatdist_scaling_{n_concepts}_codes.npz"

    if os.path.exists(cache):
        d = np.load(cache); codes = {w: d[w] for w in d.files if not w.startswith("_")}
        words = [str(w) for w in d["_words"]]; bank_of = {str(w): str(b) for w, b in zip(d["_words"], d["_banks"])}
    else:
        codes = {}; words = []; bank_of = {}
        for bank, bp, vp, sd in spec:
            vw = Q.load_vocab(vp)
            bridge = SP.build_sparse_pool_bridge(seed=sd, n_lang_input=Q.N_LANG, n_shared_pool=Q.N_POOL,
                                                 n_lang_output=Q.N_LANG, verbose=False)
            bridge.load_checkpoint(bp)
            cap = Q.capture_real_codes(bridge, vw, sd, xp)
            for w in vw:
                key = w if w not in codes else f"{bank[0]}_{w}"
                codes[key] = _mc(cap[w]); words.append(key); bank_of[key] = bank
            print(f"  captured {len(vw)} {bank} (total {len(words)})", flush=True)
            del bridge
            try:
                import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        np.savez_compressed(cache, _words=np.array(words), _banks=np.array([bank_of[w] for w in words]), **codes)

    btw = [float(np.dot(codes[a], codes[b]))
           for a, b in itertools.islice(itertools.combinations(words, 2), 40000)]
    print(f"  {len(words)}-wide between-concept cos: mean {np.mean(btw):.3f}  max {np.max(btw):.3f}  "
          f"({'DISTINCT' if np.max(btw) < 0.9 else 'DUPLICATES -> VOID'})", flush=True)
    if np.max(btw) >= 0.9:
        print("VOID: duplicate codes -- distinct-seed assumption violated.", flush=True); return

    nouns = [w for w in words if bank_of[w] == "noun"]
    verbs = [w for w in words if bank_of[w] == "verb"]
    adjs = [w for w in words if bank_of[w] == "adj"]
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    print(f"  STRUCTURED SVO (noun/verb/adj fillers) + ANY-BANK, cleanup over all {len(words)}:", flush=True)
    struct, anyb = [], []
    for seed in [42, 43, 44]:
        rng = np.random.default_rng(seed)
        roles = {r: rng.choice([-1.0, 1.0], size=len(codes[words[0]])) for r in RM.ROLES}
        roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
        D = len(codes[words[0]])
        bb, bidx = P.build(seed, D, xp)
        s_ok = a_ok = 0
        for _ in range(20):
            fs = {"agent": rng.choice(nouns), "action": rng.choice(verbs), "patient": rng.choice(adjs)}
            b = RM.bind_fact_spiking(bb, bidx, fs, codes, roles, D, xp)
            g = {r: RM.unbind_spiking(bb, bidx, b, r, roles, codes, words, D, xp) for r in RM.ROLES}
            s_ok += int(all(g[r] == fs[r] for r in RM.ROLES))
            pk = rng.choice(len(words), 3, replace=False)
            fa = {"agent": words[pk[0]], "action": words[pk[1]], "patient": words[pk[2]]}
            ba = RM.bind_fact_spiking(bb, bidx, fa, codes, roles, D, xp)
            ga = {r: RM.unbind_spiking(bb, bidx, ba, r, roles, codes, words, D, xp) for r in RM.ROLES}
            a_ok += int(all(ga[r] == fa[r] for r in RM.ROLES))
        struct.append(s_ok / 20); anyb.append(a_ok / 20)
        print(f"    seed {seed}: structured={s_ok/20:.3f}  any-bank={a_ok/20:.3f}", flush=True)
    print(f"\nRESULT @ {n_concepts} concepts: structured {struct} (mean {np.mean(struct):.3f}) | "
          f"any-bank {anyb} (mean {np.mean(anyb):.3f})", flush=True)
    if min(struct) >= 0.80 and min(anyb) >= 0.80:
        print(f"VERDICT: RESOLVES at {n_concepts} concepts -- composition SCALES past 320 (cleanup over "
              f"{n_concepts} distinct codes holds multi-seed).", flush=True)
    else:
        print(f"VERDICT: at {n_concepts} concepts a metric dips below 0.80 (structured {min(struct):.2f}, "
              f"any-bank {min(anyb):.2f}) -- the cleanup width limit; characterise.", flush=True)


if __name__ == "__main__":
    main()
