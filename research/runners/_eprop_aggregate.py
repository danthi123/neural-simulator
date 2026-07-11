"""Aggregate the e-prop recurrent-learning de-risk per-seed JSONs into a mean+-std table: for each arm (mode), the
plastic-minus-fixed CE by context depth (esp. DEEP 6+, where the fixed reservoir was n-gram-level) and the aggregate,
averaged across seeds. Usage: python -m research.runners._eprop_aggregate "research/findings/raw/_eprop/wiki_np300_s*.json" --label lr002
"""
import argparse, glob, json
import numpy as np

BUCKETS = ["1", "2", "3", "4-5", "6-9", "10-99"]


def deep_margin(bm, mode):
    """mean over deep buckets (6-9,10-99) of (mode.ce - fixed.ce); negative = mode better."""
    fx = bm["fixed"]["by_depth"]; b = bm[mode]["by_depth"]
    vals = [b[k]["ce"] - fx[k]["ce"] for k in ("6-9", "10-99") if k in b and k in fx]
    return float(np.mean(vals)) if vals else float("nan")


def by_depth_margin(bm, mode):
    fx = bm["fixed"]["by_depth"]; b = bm[mode]["by_depth"]
    return {k: (b[k]["ce"] - fx[k]["ce"]) for k in BUCKETS if k in b and k in fx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"NO FILES match {args.glob}"); return
    modes = None; per_mode_deep = {}; per_mode_agg = {}; per_mode_depth = {}; seeds = []
    for f in files:
        d = json.load(open(f))
        for seed, rec in d["per_seed"].items():
            bm = rec["by_mode"]
            if "fixed" not in bm:
                continue
            seeds.append(seed)
            if modes is None:
                modes = [m for m in bm if m != "fixed"]
            for m in modes:
                if m not in bm:
                    continue
                per_mode_deep.setdefault(m, []).append(deep_margin(bm, m))
                per_mode_agg.setdefault(m, []).append(bm[m]["aggregate_ce"] - bm["fixed"]["aggregate_ce"])
                dm = by_depth_margin(bm, m)
                per_mode_depth.setdefault(m, {})
                for k, v in dm.items():
                    per_mode_depth[m].setdefault(k, []).append(v)
    n = len(set(seeds))
    print(f"\n===== {args.label or args.glob}  ({n} seeds: {sorted(set(seeds))}) =====")
    print(f"{'mode':12s} {'agg-vs-fixed':>14s} {'DEEP(6+)':>14s}   per-depth mean (neg=better)")
    for m in (modes or []):
        if m not in per_mode_deep:
            continue
        ag = np.array(per_mode_agg[m]); dp = np.array(per_mode_deep[m])
        depth_str = " ".join(f"d{k}:{np.mean(per_mode_depth[m][k]):+.3f}" for k in BUCKETS if k in per_mode_depth[m])
        print(f"{m:12s} {np.mean(ag):+.3f}+-{np.std(ag):.3f}  {np.mean(dp):+.3f}+-{np.std(dp):.3f}   {depth_str}")
        if m in ("plastic", "adaptive", "symmetric"):
            npos = int(np.sum(dp < 0))
            print(f"{'':12s}   -> DEEP better than fixed on {npos}/{len(dp)} seeds")


if __name__ == "__main__":
    main()
