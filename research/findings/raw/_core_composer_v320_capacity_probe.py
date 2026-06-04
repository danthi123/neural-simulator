"""Cheap-first capacity probe (in-scope VALIDATION of the consolidation deliverable, NOT a months-scale
scaling commitment): does the PROMOTED CoreSimComposer -- the consolidated on-brain conversational composer
(research/runners/core_sim_composition.py) -- inherit the validated V=320 vocab-robustness of the
_insubstrate coincidence bind/unbind (pillar n=111: spiking cleanup recovery 1.000 up to V=320)?

Inject `vocab` distinct concept codes via the `concepts=` hook (bypassing the V=16 denoise64 cache), store
K=1 SVO facts on the real ~6400-neuron Izhikevich bridge, measure who/what recovery + the no-confab
abstention moat. GPU (real run). Honest boundary surfaced if the promoted module does NOT hold at V=320.

Code regime: near-orthogonal random unit codes (the clean mechanism-headroom END; between-cos ~0.04). The
hard correlated-code end (denoise64 was between-cos ~0.70) is the documented next rung if this passes.

Usage:
  python -m research.findings.raw._core_composer_v320_capacity_probe --seed 42 --vocab 320
"""
import argparse
import json

import numpy as np

from research.runners.core_sim_composition import CoreSimComposer


def make_codes(n, d, seed, rho=0.0):
    """n random unit codes of dim d. rho=0 -> near-orthogonal (between-cos ~0, easy end);
    rho>0 -> inject a shared component so between-cos ~ rho (the hard correlated regime,
    e.g. rho~0.6 mimics the denoise64 substrate codes' ~0.70 between-cos)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    if rho > 0.0:
        s = rng.standard_normal(d)
        s = s / np.linalg.norm(s)
        x = np.sqrt(1.0 - rho) * x + np.sqrt(rho) * s[None, :]
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--rho", type=float, default=0.0,
                    help="shared-component correlation; 0=near-orthogonal (easy), ~0.6=denoise64-like (hard)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.proj_dim
    words = [f"c{i:03d}" for i in range(args.vocab)]
    codes = make_codes(args.vocab, d, args.seed, rho=args.rho)
    concepts = {w: codes[i] for i, w in enumerate(words)}

    c = CoreSimComposer(seed=args.seed, proj_dim=d, concepts=concepts)

    # between-code cosine after the composer's centering (honesty: which code regime is this?)
    m = np.stack([c.concepts[w] for w in c.words])
    g = m @ m.T
    iu = np.triu_indices(len(c.words), 1)
    bc = g[iu]
    print(f"[probe] V={len(c.words)} D={d} between-cos mean={bc.mean():.3f} max={bc.max():.3f}")

    rng = np.random.default_rng(args.seed + 1)
    ok_w = ok_a = ok_abs = tot_abs = 0
    for _ in range(args.n_facts):
        a, ac, p = (str(x) for x in rng.choice(c.words, size=3, replace=False))
        c.kb = []
        c.store(a, ac, p)
        ok_w += int(c.query_patient(a, ac) == p)
        ok_a += int(c.query_agent(ac, p) == a)
        # no-confab moat: a never-stored agent+action -> must abstain (None)
        a2, ac2 = (str(x) for x in rng.choice(c.words, size=2, replace=False))
        if a2 != a:
            ok_abs += int(c.query_patient(a2, ac2) is None)
            tot_abs += 1

    res = {
        "seed": args.seed, "vocab": len(c.words), "proj_dim": d, "n_facts": args.n_facts, "rho": args.rho,
        "between_cos_mean": float(bc.mean()), "between_cos_max": float(bc.max()),
        "what_correct": ok_w, "who_correct": ok_a, "abstain_correct": ok_abs, "abstain_total": tot_abs,
        "what_rate": ok_w / args.n_facts, "who_rate": ok_a / args.n_facts,
        "abstain_rate": (ok_abs / tot_abs) if tot_abs else None,
    }
    print(f"[probe] what {ok_w}/{args.n_facts}  who {ok_a}/{args.n_facts}  abstain {ok_abs}/{tot_abs}")
    print("[probe] " + json.dumps(res))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
