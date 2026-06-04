"""Capacity-curve probe: does the spiking unified agent hold accuracy as the VOCABULARY grows at fixed
dimension? Turns "future scaling may need more than 320 concepts" into a concrete cost model -- which category
degrades first (the clean-up is expected robust; the F=3 two-attribute resonator the first to need more D), and
how wall-clock scales with vocabulary. The cheap-first measurement BEFORE any GPU / fast-mode scaling investment.

The benchmark facts use only the core words, so growing the vocabulary adds DISTRACTOR concepts that stress the
clean-up / resonator / decode at fixed dimension -- exactly the capacity question.

  SIM_BACKEND=numpy python -m research.runners._capacity_curve_probe
  SIM_BACKEND=numpy python -m research.runners._capacity_curve_probe --D 4096 --vocabs 1280 2560
"""
from __future__ import annotations
import argparse
import json
import time

from research.runners.spiking_unified_agent import run_core_benchmark


def main():
    ap = argparse.ArgumentParser(description="Capacity curve: agent accuracy + cost vs vocabulary at fixed D.")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocabs", type=int, nargs="+", default=[320, 640, 1280, 2560])
    ap.add_argument("--out", default="research/findings/raw/capacity_curve_probe.json")
    args = ap.parse_args()

    print(f"=== capacity curve | D={args.D} | seed={args.seed} | vocabs={args.vocabs} ===\n", flush=True)
    rows = {}
    for V in args.vocabs:
        n_noun, n_verb, n_adj = round(V * 200 / 320), round(V * 60 / 320), round(V * 60 / 320)  # keep ratio
        t0 = time.perf_counter()
        res, _ = run_core_benchmark(n_dim=args.D, seed=args.seed, n_noun=n_noun, n_verb=n_verb, n_adj=n_adj)
        dt = time.perf_counter() - t0
        gok = sum(v[0] for v in res.values())
        gtot = sum(v[1] for v in res.values())
        rows[V] = {"n": [n_noun, n_verb, n_adj], "res": res, "overall": [gok, gtot], "secs": dt}
        cats = "  ".join(f"{c}={res[c][0]}/{res[c][1]}" for c in res)
        print(f"  vocab={V:5} ({n_noun}n/{n_verb}v/{n_adj}a): {cats} | overall {gok}/{gtot}={gok/gtot*100:.0f}% "
              f"| {dt:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)

    print("\n  --- capacity analysis (per category, accuracy vs vocab; cost) ---", flush=True)
    for c in rows[args.vocabs[0]]["res"]:
        curve = [(V, rows[V]["res"][c][0] / rows[V]["res"][c][1]) for V in args.vocabs]
        first_drop = next((V for V, r in curve if r < 0.999), None)
        tag = f"(degrades at vocab {first_drop})" if first_drop else "(holds across the sweep)"
        print(f"    {c:<14} " + "  ".join(f"{V}:{r*100:.0f}%" for V, r in curve) + f"   {tag}", flush=True)
    secs = [(V, rows[V]["secs"]) for V in args.vocabs]
    print(f"    {'wall-clock':<14} " + "  ".join(f"{V}:{s:.0f}s" for V, s in secs), flush=True)
    print("\n  -> the category that degrades first is where dimension D must grow with vocabulary (and where a "
          "GPU/fast-mode\n     scaling foundation pays off); the categories that hold scale cheaply.", flush=True)


if __name__ == "__main__":
    main()
