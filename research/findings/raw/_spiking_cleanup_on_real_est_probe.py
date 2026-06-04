"""(item 2 integration de-risk) Does the spiking cleanup recover the COMPOSER'S REAL `est` as well as numpy argmax,
per capability category? Build a V=320 composer (production-scheme codes, already near-orthogonal cos~0.05), store
facts of each kind, capture the REAL est for each role via `composer._unbind_onoff` (no composer rewrite, no slow
full-matrix), and clean each est up with (a) numpy argmax [the composer's current cleanup] and (b) the spiking
matched-filter cleanup bridge built from the SAME codebook. Reports per-role recovery + the est cue-cosine
distribution -- predicting exactly which roles the spiking cleanup preserves vs regresses on real est.

  python -m research.findings.raw._spiking_cleanup_on_real_est_probe --seed 42 --proj-dim 800
"""
import argparse
import json

import numpy as np

from research.runners.core_sim_composition import CoreSimComposer, Clause
from research.findings.raw._core_composer_grounded320_probe import production_codes
from research.findings.raw._spiking_cleanup_core_probe import build_cleanup_bridge, cleanup_spiking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=15)
    ap.add_argument("--n-attr", type=int, default=8)
    ap.add_argument("--w-match", type=float, default=40.0)
    ap.add_argument("--w-inh", type=float, default=0.0, help="lateral inhibition (WTA) = a normalization on near-ortho codes")
    ap.add_argument("--concept-bias", type=float, default=-150.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    codes_in = production_codes(args.vocab, 2000, 100, args.proj_dim, args.seed)
    words = [f"c{i:03d}" for i in range(args.vocab)]
    concepts = {w: codes_in[i] for i, w in enumerate(words)}
    comp = CoreSimComposer(seed=args.seed, proj_dim=args.proj_dim, concepts=concepts)

    # cleanup bridge from the composer's ACTUAL concept codebook (M concepts, D dim)
    M = len(comp.words); D = comp.D
    code_mat = np.stack([comp.concepts[w] for w in comp.words])
    cbridge, cidx = build_cleanup_bridge(args.seed, code_mat, args.w_match, args.w_inh)
    widx = {w: i for i, w in enumerate(comp.words)}

    rng = np.random.default_rng(args.seed + 1)

    def pick(k):
        return [str(x) for x in rng.choice(comp.words, size=k, replace=False)]

    # (est, true_word, category) tuples captured from the composer's REAL unbind
    items = []
    for _ in range(args.n_flat):
        a, ac, p = pick(3)
        comp.kb = []; comp.store(a, ac, p)
        bound = comp.kb[0][1]
        for role, true in (("agent", a), ("action", ac), ("patient", p)):
            e_on, e_off = comp._unbind_onoff(bound, role)
            items.append((e_on - e_off, true, "flat"))
    for _ in range(args.n_attr):
        a, ac, adj1, adj2, noun = pick(5)
        comp.kb = []; comp.store(a, ac, ((adj1, adj2), noun))
        bound = comp.kb[0][1]
        for role, true in (("patient", noun), ("attribute", adj1), ("attribute2", adj2)):
            e_on, e_off = comp._unbind_onoff(bound, role)
            items.append((e_on - e_off, true, "two_attr"))

    # clean up each REAL est: numpy argmax vs spiking matched filter
    by_cat = {}
    for est, true, cat in items:
        cos = float(code_mat[widx[true]] @ est / (np.linalg.norm(est) + 1e-12))
        np_win = comp.words[int(np.argmax(code_mat @ est))]
        rates = cleanup_spiking(cbridge, cidx, D, M, est, args.concept_bias)
        sp_win = comp.words[int(np.argmax(rates))]
        d = by_cat.setdefault(cat, {"n": 0, "np_ok": 0, "sp_ok": 0, "cos": []})
        d["n"] += 1; d["np_ok"] += int(np_win == true); d["sp_ok"] += int(sp_win == true); d["cos"].append(cos)

    res = {"seed": args.seed, "vocab": args.vocab, "proj_dim": args.proj_dim, "by_category": {}}
    for cat, d in by_cat.items():
        res["by_category"][cat] = {"n": d["n"], "numpy": d["np_ok"] / d["n"], "spiking": d["sp_ok"] / d["n"],
                                   "mean_est_cos": float(np.mean(d["cos"]))}
        print(f"[realest] {cat}: n={d['n']}  est_cos={np.mean(d['cos']):.3f}  numpy={d['np_ok']}/{d['n']}  "
              f"spiking={d['sp_ok']}/{d['n']}", flush=True)
    print("[realest] " + json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
