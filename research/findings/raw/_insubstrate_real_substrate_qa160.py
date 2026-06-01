"""Scale-up of the real-substrate spiking QA to the FULL G.20 160-concept ensemble (5 bridges x 32).

Pre-staged: launch ONLY if the single-bridge cheap-first (_insubstrate_real_substrate_qa_probe on
bridgeA) RESOLVES. Captures the REAL concept code for all 160 words across the 5 deployed sparse bridges
(driving lang_input(word) through each TRAINED bridge -> shared_concept_pool activity), pools them into one
160-word vocabulary, then runs the validated spiking SVO fact-memory + wh-QA + abstention control over all
160 -- including CROSS-BRIDGE facts (agent from bridgeA-nouns, patient from bridgeC-adj), the realistic
conversational case. Reports the 160-wide between-concept separability + QA + abstention.

If RESOLVES (QA + abstention >= 0.80 at 160): the largest GENUINE-composition conversational artifact in the
project runs on the REAL deployed substrate. If it degrades: honest boundary (real-substrate cross-bridge
structure), characterized.

Reuse-by-import (capture_real_codes / run_qa from the single-bridge probe; sparse builder); each bridge is
freed after capture to bound GPU memory; no protected-module change; no autograd. load_checkpoint validates
architecture per bridge.

Run (GPU): python -m research.findings.raw._insubstrate_real_substrate_qa160 --seed 42
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.findings.raw._insubstrate_real_substrate_qa_probe as Q1
import research.runners.concept_pool_sparse_distributed as SP
from sim.backend import get_backend

BRIDGES = [
    ("bridgeA_nouns", "g20_sparse_bridges/bridgeA_nouns_sparse.simstate.h5", "g20_bridgeA_nouns_vocab.txt"),
    ("bridgeB_verbs", "g20_sparse_bridges/bridgeB_verbs_sparse.simstate.h5", "g20_bridgeB_verbs_vocab.txt"),
    ("bridgeC_adj", "g20_sparse_bridges/bridgeC_adj_sparse.simstate.h5", "g20_bridgeC_adj_vocab.txt"),
    ("bridgeD_spatial", "g20_sparse_bridges/bridgeD_spatial_sparse.simstate.h5", "g20_bridgeD_spatial_vocab.txt"),
    ("bridgeE_functional", "g20_sparse_bridges/bridgeE_functional_sparse.simstate.h5",
     "g20_bridgeE_functional_vocab.txt"),
]
ROOT = "research/findings/raw/g11_bg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n-facts", type=int, default=2)
    a = ap.parse_args()
    xp, backend = get_backend()
    print(f"=== real-substrate QA at 160 concepts (5 bridges, backend={backend}, seed={a.seed}) ===",
          flush=True)

    all_codes = {}
    all_words = []
    for name, bpath, vpath in BRIDGES:
        bp = os.path.join(ROOT, bpath); vp = os.path.join(ROOT, vpath)
        if not os.path.exists(bp):
            print(f"CANNOT-CONCLUDE: {bp} missing", flush=True); return
        words = Q1.load_vocab(vp)
        bridge = SP.build_sparse_pool_bridge(seed=a.seed, n_lang_input=Q1.N_LANG, n_shared_pool=Q1.N_POOL,
                                             n_lang_output=Q1.N_LANG, verbose=False)
        bridge.load_checkpoint(bp)   # validates architecture
        codes = Q1.capture_real_codes(bridge, words, a.seed, xp)
        # disambiguate any cross-bridge duplicate surface words (e.g. shared tokens) by FULL bridge prefix
        for w in words:
            key = w if w not in all_codes else f"{name}::{w}"
            all_codes[key] = codes[w]; all_words.append(key)
        print(f"  captured {len(words)} from {name} (total {len(all_words)})", flush=True)
        # free this bridge before the next (bound GPU memory)
        del bridge
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    # 160-wide separability (does cross-bridge real-code overlap stay low enough for cleanup?)
    import itertools
    samp = all_words if len(all_words) <= 160 else all_words[:160]
    btw = [float(np.dot(all_codes[i], all_codes[j])) for i, j in itertools.combinations(samp, 2)]
    print(f"  160-wide between-concept cos: mean {np.mean(btw):.3f}  max {np.max(btw):.3f}", flush=True)

    qa, ctrl = Q1.run_qa(all_codes, all_words, a.seed, a.n_trials, a.n_facts, xp)
    print(f"\nRESULT: 160-concept REAL-code QA={qa:.3f}  abstention-control={ctrl:.3f}  "
          f"(chance {1.0/len(all_words):.4f})", flush=True)
    if qa >= 0.80 and ctrl >= 0.80:
        print("VERDICT: RESOLVES -- genuine spiking relational composition + abstention works on the REAL "
              "deployed 160-concept substrate, cross-bridge. Largest genuine-composition conversational "
              "artifact on the real substrate.", flush=True)
    elif qa >= 0.50:
        print(f"VERDICT: PARTIAL -- 160-concept real QA {qa:.2f}; characterize the cross-bridge gap.",
              flush=True)
    else:
        print(f"VERDICT: 160-concept real QA {qa:.2f} -- honest boundary at scale on real codes.", flush=True)


if __name__ == "__main__":
    main()
