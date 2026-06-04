"""(item-3 cheap-first de-risk) Does ZCA-decorrelating the REAL V=16 denoise64 captured codes (between-cos ~0.80)
preserve the composer's capability AND fix the cos-0.80-driven two-attribute boundary? Build the composer with
decorrelate False vs True, measure between-cos, run flat / one-attr / two-attr / negation. Validates the
decorrelation linchpin (stage 1.5 + item-2) on REAL captured codes before any heavy 320 capture.

  python -m research.findings.raw._decorrelate_v16_probe --proj-dim 800
"""
import argparse

import numpy as np

from research.runners.core_sim_composition import CoreSimComposer


def run_matrix(comp, n, rng):
    def pick(k):
        return [str(x) for x in rng.choice(comp.words, size=k, replace=False)]

    score = {}
    ok = 0
    for _ in range(n):
        a, ac, p = pick(3); comp.kb = []; comp.store(a, ac, p); ok += int(comp.query_patient(a, ac) == p)
    score["flat"] = (ok, n)
    ok = 0
    for _ in range(n):
        a, ac, adj, noun = pick(4); comp.kb = []; comp.store(a, ac, (adj, noun))
        ok += int(comp.query_patient(a, ac) == f"{adj} {noun}")
    score["one_attr"] = (ok, n)
    ok = 0
    for _ in range(n):
        a, ac, adj1, adj2, noun = pick(5); comp.kb = []; comp.store(a, ac, ((adj1, adj2), noun))
        exp = " ".join(sorted([adj1, adj2], key=comp.words.index) + [noun])
        ok += int(comp.query_patient(a, ac) == exp)
    score["two_attr"] = (ok, n)
    ok = 0
    for _ in range(n):
        a, ac, p = pick(3); comp.kb = []; comp.store(a, ac, p, polarity="AFFIRM"); ok += int(comp.ask_yes_no(a, ac, p) == "yes")
        a, ac, p = pick(3); comp.kb = []; comp.store(a, ac, p, polarity="NEGATE"); ok += int(comp.ask_yes_no(a, ac, p) == "no")
    score["negation"] = (ok, 2 * n)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    for dec in (False, True):
        try:
            comp = CoreSimComposer(seed=args.seed, proj_dim=args.proj_dim, decorrelate=dec)
        except FileNotFoundError:
            print("[dec16] denoise64 cache missing; skip", flush=True); return
        m = np.stack([comp.concepts[w] for w in comp.words]); g = m @ m.T
        bc = g[np.triu_indices(len(comp.words), 1)]
        rng = np.random.default_rng(args.seed + 1)
        score = run_matrix(comp, args.n, rng)
        line = "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in score.items())
        print(f"[dec16] decorrelate={dec} D={args.proj_dim} between-cos={bc.mean():.3f}  {line}", flush=True)


if __name__ == "__main__":
    main()
