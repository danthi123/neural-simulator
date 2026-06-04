"""(iii) vocab scaling: the unified-agent benchmark (numpy NestedCompositionAgent, with the clause-depth2 fix) at
GROWING vocabulary, fixed dimension D. The frozen test set uses only the core fact-bearing words, so a larger
vocabulary adds DISTRACTOR concepts that stress the clean-up / resonator at fixed D -- the capacity curve. Maps
WHERE each category degrades (the cost model predicts the two-attribute F=3 resonator is the lone bottleneck,
needing D ∝ M²; everything else holds at fixed D).

Reuse-by-import (run_seed + aggregate from the benchmark). numpy/CPU.

  python -m research.runners.unified_agent_capacity_curve --seeds 42 43 --sizes 320 640 1280
"""
from __future__ import annotations
import argparse
import json

from research.runners.unified_agent_benchmark import run_seed, aggregate


def _split(vocab):
    """Keep the benchmark's 200:60:60 = 320 ratio at any multiple of 320 (n_noun:n_verb:n_adj = 5:1.5:1.5)."""
    k = vocab / 320.0
    return int(round(200 * k)), int(round(60 * k)), int(round(60 * k))


def main():
    ap = argparse.ArgumentParser(description="Unified-agent capacity curve: benchmark at growing vocab, fixed D.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 640, 1280])
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"=== unified-agent CAPACITY CURVE | D={args.D} | seeds={args.seeds} | sizes={args.sizes} ===", flush=True)
    print("  (frozen test set fixed; larger vocab = more distractor concepts at fixed D)\n", flush=True)
    rows = []
    for vocab in args.sizes:
        nn, nv, na = _split(vocab)
        rs = [run_seed(s, D=args.D, n_noun=nn, n_verb=nv, n_adj=na) for s in args.seeds]
        agg, gok, gtot = aggregate(rs)
        rows.append({"vocab": vocab, "n_noun": nn, "n_verb": nv, "n_adj": na,
                     "aggregate": agg, "overall": [gok, gtot]})
        line = "  ".join(f"{c}={agg[c][2]*100:.0f}%" for c in agg)
        print(f"  vocab={vocab:>5} ({nn}n/{nv}v/{na}a):  {line}  OVERALL={gok/gtot*100:.0f}%", flush=True)

    # which categories hold vs degrade across the curve
    cats = list(rows[0]["aggregate"].keys())
    print("\n  --- per-category across the curve (hold vs degrade) ---", flush=True)
    for c in cats:
        series = "  ".join(f"{r['vocab']}:{r['aggregate'][c][2]*100:.0f}%" for r in rows)
        held = all(r["aggregate"][c][2] >= 0.999 for r in rows)
        print(f"    {c:<16} {series}   {'HOLDS' if held else 'DEGRADES'}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"D": args.D, "seeds": args.seeds, "rows": rows}, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
