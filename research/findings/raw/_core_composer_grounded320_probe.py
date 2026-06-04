"""(A) grounded-320 first increment: validate the promoted CoreSimComposer on the project's REAL production
320-concept codes -- the G.20 sparse-distributed `generate_sparse_patterns` scheme (the actual code scheme behind
the shipped 320-concept ensemble), NOT the synthetic random codes of the vocab-robustness probe.

Each production code is a sparse K-of-N pattern (K active neurons in a pool of N). We convert it to a dense vector
and project it to the composer's dim with a random Gaussian -- exactly how `load_concepts` treats the `denoise64`
concept-pool activity (the composer's native grounded-code path). Then store K=1 facts + a multi-fact KB at V=320,
measure who/what recovery + the no-confab moat. Honest BOUNDARY surfaced if the production sparse-code statistics
(very different ON/OFF balance from the dense denoise64 codes) break the bind/threshold.

This is the cheap-first first rung of direction (A): does the brain composer work on PRODUCTION codes? The heavier
rung is capturing 320 concept-pool activities so the codes are the substrate's OWN (like denoise64 at V=16).

Usage:
  python -m research.findings.raw._core_composer_grounded320_probe --seed 42 --vocab 320 --kb-size 20
"""
import argparse
import json

import numpy as np

from research.runners.core_sim_composition import CoreSimComposer
from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns


def production_codes(vocab, n_pool, pattern_size, proj_dim, seed):
    """The G.20 production sparse patterns -> dense binary -> random-Gaussian projection to proj_dim
    (mirrors load_concepts' treatment of denoise64), unit-normalized."""
    patterns = generate_sparse_patterns(vocab, n_pool, pattern_size, seed)
    b = np.zeros((vocab, n_pool), dtype=np.float64)
    for i, pat in enumerate(patterns):
        b[i, pat] = 1.0
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((n_pool, proj_dim)) / np.sqrt(n_pool)
    x = b @ proj
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--pattern-size", type=int, default=100)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--kb-size", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    codes = production_codes(args.vocab, args.n_pool, args.pattern_size, args.proj_dim, args.seed)
    words = [f"c{i:03d}" for i in range(args.vocab)]
    concepts = {w: codes[i] for i, w in enumerate(words)}

    c = CoreSimComposer(seed=args.seed, proj_dim=args.proj_dim, concepts=concepts)
    m = np.stack([c.concepts[w] for w in c.words])
    g = m @ m.T
    iu = np.triu_indices(len(c.words), 1)
    bc = g[iu]
    print(f"[grounded320] V={len(c.words)} sparse K={args.pattern_size}/N={args.n_pool} -> D={args.proj_dim} "
          f"between-cos mean={bc.mean():.3f} max={bc.max():.3f}")

    rng = np.random.default_rng(args.seed + 1)
    kb = max(1, args.kb_size)
    facts, cues, guard = [], set(), 0
    while len(facts) < kb and guard < 100000:
        guard += 1
        a, ac, p = (str(x) for x in rng.choice(c.words, size=3, replace=False))
        if (a, ac) in cues:
            continue
        cues.add((a, ac))
        facts.append((a, ac, p))
    c.kb = []
    for a, ac, p in facts:
        c.store(a, ac, p)
    okw = oka = okab = 0
    for a, ac, p in facts:
        okw += int(c.query_patient(a, ac) == p)
        oka += int(c.query_agent(ac, p) == a)
    for _ in range(kb):
        g2, a2, ac2 = 0, None, None
        while g2 < 1000:
            g2 += 1
            a2, ac2 = (str(x) for x in rng.choice(c.words, size=2, replace=False))
            if (a2, ac2) not in cues:
                break
        okab += int(c.query_patient(a2, ac2) is None)

    res = {
        "seed": args.seed, "vocab": len(c.words), "n_pool": args.n_pool, "pattern_size": args.pattern_size,
        "proj_dim": args.proj_dim, "kb_size": kb, "code_source": "G20_sparse_distributed_production",
        "between_cos_mean": float(bc.mean()), "between_cos_max": float(bc.max()),
        "what_correct": okw, "who_correct": oka, "abstain_correct": okab,
        "what_rate": okw / kb, "who_rate": oka / kb, "abstain_rate": okab / kb,
    }
    print(f"[grounded320] kb={kb}  what {okw}/{kb}  who {oka}/{kb}  abstain {okab}/{kb}")
    print("[grounded320] " + json.dumps(res))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[grounded320] wrote {args.out}")


if __name__ == "__main__":
    main()
