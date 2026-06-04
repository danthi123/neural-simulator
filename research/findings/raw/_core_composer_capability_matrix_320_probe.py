"""(A) grounded-320, rung 3: the full CAPABILITY MATRIX of the composer at production 320-word vocab on the real
G.20 production codes -- flat fact, one-attribute, two-attribute, embedded clause, negation/yes-no. This completes
the picture: rung 1 (vocab-robust) + rung 2 (full agent flat loop) covered flat memory; this checks the
COMPOSITION-DEPTH categories at scale, where an honest boundary is expected (the consolidation flagged
two-attribute as a K=5-load BOUNDARY even at V=16; the capacity cost model notes composition-depth ops want D to
grow with vocabulary). Either outcome is a real finding -- a clean PASS or a confirmed honest boundary at scale.

Usage:
  python -m research.findings.raw._core_composer_capability_matrix_320_probe --seed 42 --vocab 320 --n 6
"""
import argparse
import json

import numpy as np

from research.runners.core_sim_composition import CoreSimComposer, Clause
from research.findings.raw._core_composer_grounded320_probe import production_codes
from research.findings.raw._core_composer_v320_capacity_probe import make_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--pattern-size", type=int, default=100)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n", type=int, default=6, help="trials per category")
    ap.add_argument("--rho", type=float, default=0.0,
                    help="if >0, use SYNTHETIC correlated codes at this between-cos (e.g. 0.80 = the captured-code "
                         "regime) instead of the production sparse codes -- to PREDICT item-3's outcome on real codes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.rho > 0.0:
        codes = make_codes(args.vocab, args.proj_dim, args.seed, rho=args.rho)
        code_source = f"synthetic_rho_{args.rho:.2f}"
    else:
        codes = production_codes(args.vocab, args.n_pool, args.pattern_size, args.proj_dim, args.seed)
        code_source = "G20_sparse_distributed_production"
    words = [f"c{i:03d}" for i in range(args.vocab)]
    concepts = {w: codes[i] for i, w in enumerate(words)}
    c = CoreSimComposer(seed=args.seed, proj_dim=args.proj_dim, concepts=concepts)
    rng = np.random.default_rng(args.seed + 1)

    def pick(k):
        return [str(x) for x in rng.choice(c.words, size=k, replace=False)]

    score = {}

    # flat: store(a, ac, p) -> query_patient == p
    ok = 0
    for _ in range(args.n):
        a, ac, p = pick(3)
        c.kb = []
        c.store(a, ac, p)
        ok += int(c.query_patient(a, ac) == p)
    score["flat"] = (ok, args.n)

    # one-attribute: store(a, ac, (adj, noun)) -> "adj noun"
    ok = 0
    for _ in range(args.n):
        a, ac, adj, noun = pick(4)
        c.kb = []
        c.store(a, ac, (adj, noun))
        ok += int(c.query_patient(a, ac) == f"{adj} {noun}")
    score["one_attr"] = (ok, args.n)

    # two-attribute: store(a, ac, ((adj1,adj2), noun)) -> adjs sorted by vocab index + noun
    ok = 0
    for _ in range(args.n):
        a, ac, adj1, adj2, noun = pick(5)
        c.kb = []
        c.store(a, ac, ((adj1, adj2), noun))
        expected = " ".join(sorted([adj1, adj2], key=c.words.index) + [noun])
        ok += int(c.query_patient(a, ac) == expected)
    score["two_attr"] = (ok, args.n)

    # embedded clause: store(a, ac, Clause(a2, ac2, p2)) -> "a2 ac2 p2"
    ok = 0
    for _ in range(args.n):
        a, ac, a2, ac2, p2 = pick(5)
        c.kb = []
        c.store(a, ac, Clause(a2, ac2, p2))
        ok += int(c.query_patient(a, ac) == f"{a2} {ac2} {p2}")
    score["clause"] = (ok, args.n)

    # negation / yes-no: affirmed -> "yes", negated -> "no"
    ok = 0
    for _ in range(args.n):
        a, ac, p = pick(3)
        c.kb = []
        c.store(a, ac, p, polarity="AFFIRM")
        ok += int(c.ask_yes_no(a, ac, p) == "yes")
        a, ac, p = pick(3)
        c.kb = []
        c.store(a, ac, p, polarity="NEGATE")
        ok += int(c.ask_yes_no(a, ac, p) == "no")
    score["negation_yesno"] = (ok, 2 * args.n)

    m = np.stack([c.concepts[w] for w in c.words])
    bc = (m @ m.T)[np.triu_indices(len(c.words), 1)]
    res = {"seed": args.seed, "vocab": args.vocab, "code_source": code_source,
           "proj_dim": args.proj_dim, "between_cos_mean": float(bc.mean()), "between_cos_max": float(bc.max()),
           "scores": {k: {"correct": v[0], "total": v[1], "rate": v[0] / v[1]} for k, v in score.items()}}
    line = "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in score.items())
    print(f"[matrix320] seed {args.seed} V={args.vocab}  {line}")
    print("[matrix320] " + json.dumps(res))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[matrix320] wrote {args.out}")


if __name__ == "__main__":
    main()
