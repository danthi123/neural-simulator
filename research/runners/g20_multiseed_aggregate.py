"""Multi-seed aggregator for the G.20 32-concept tier.

Reads shared_pool_n32{,_seedN}.json files across multiple seeds and
produces a cross-seed summary: per-word PASS counts, robust/fragile/fail
classification, mean PASS rate.

Usage:
  python -m research.runners.g20_multiseed_aggregate \\
      --seeds 42,43,44,45 \\
      --out research/findings/raw/g11_bg/g20_n32_multiseed.json
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_seed_result(out_dir: Path, seed: int) -> dict | None:
    """Load a seed's 32-concept JSON. Seed 42 is in shared_pool_n32.json;
    others are in shared_pool_n32_seed{N}.json."""
    if seed == 42:
        path = out_dir / "shared_pool_n32.json"
    else:
        path = out_dir / f"shared_pool_n32_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def aggregate(seeds: list[int], out_dir: Path) -> dict:
    """Aggregate across seeds, classify each word."""
    per_word = defaultdict(lambda: {
        "top1_count": 0,
        "top5_count": 0,
        "ranks": [],
        "total_seeds": 0,
    })
    n_top1_per_seed = {}
    n_top5_per_seed = {}
    seeds_loaded = []

    for seed in seeds:
        d = load_seed_result(out_dir, seed)
        if not d:
            continue
        seeds_loaded.append(seed)
        n_top1_per_seed[seed] = d["n_top1"]
        n_top5_per_seed[seed] = d["n_top5"]
        for r in d["results"]:
            w = r["word"]
            per_word[w]["total_seeds"] += 1
            per_word[w]["ranks"].append(r["rank"])
            if r["top1"]:
                per_word[w]["top1_count"] += 1
            if r["top5"]:
                per_word[w]["top5_count"] += 1

    n = len(seeds_loaded)
    robust_top1 = sorted([w for w, info in per_word.items()
                           if info["top1_count"] == n and n > 0])
    fragile_top1 = sorted([w for w, info in per_word.items()
                            if 0 < info["top1_count"] < n])
    fail_top1 = sorted([w for w, info in per_word.items()
                         if info["top1_count"] == 0
                         and info["total_seeds"] > 0])
    robust_top5 = sorted([w for w, info in per_word.items()
                           if info["top5_count"] == n and n > 0])
    fail_top5 = sorted([w for w, info in per_word.items()
                         if info["top5_count"] == 0
                         and info["total_seeds"] > 0])

    return {
        "seeds": seeds_loaded,
        "n_top1_per_seed": n_top1_per_seed,
        "n_top5_per_seed": n_top5_per_seed,
        "per_word": {w: {
            "top1_count": info["top1_count"],
            "top5_count": info["top5_count"],
            "total_seeds": info["total_seeds"],
            "ranks": info["ranks"],
        } for w, info in per_word.items()},
        "robust_top1": robust_top1,
        "fragile_top1": fragile_top1,
        "fail_top1": fail_top1,
        "robust_top5": robust_top5,
        "fail_top5": fail_top5,
    }


def print_summary(agg: dict) -> None:
    seeds = agg["seeds"]
    if not seeds:
        print("(no seeds loaded)")
        return
    n = len(seeds)
    total_words = 32 * n
    sum_top1 = sum(agg["n_top1_per_seed"].values())
    sum_top5 = sum(agg["n_top5_per_seed"].values())
    print(f"=== G.20 32-concept multi-seed ({n} seeds) ===")
    print(f"  seeds loaded: {seeds}")
    print(f"  per-seed top-1: ", end="")
    print(", ".join(f"{s}={agg['n_top1_per_seed'][s]}/32"
                     for s in seeds))
    print(f"  per-seed top-5: ", end="")
    print(", ".join(f"{s}={agg['n_top5_per_seed'][s]}/32"
                     for s in seeds))
    print(f"  combined top-1: {sum_top1}/{total_words} "
          f"({100*sum_top1/total_words:.1f}%)")
    print(f"  combined top-5: {sum_top5}/{total_words} "
          f"({100*sum_top5/total_words:.1f}%)")
    print(f"  per-bridge mean top-1: {sum_top1/n:.1f}/32 "
          f"({100*sum_top1/(32*n):.1f}%)")
    print(f"  per-bridge mean top-5: {sum_top5/n:.1f}/32 "
          f"({100*sum_top5/(32*n):.1f}%)")
    print(f"\n  ROBUST top-1 ({len(agg['robust_top1'])}): "
          f"{', '.join(agg['robust_top1'])}")
    print(f"  FRAGILE top-1 ({len(agg['fragile_top1'])}): "
          f"{', '.join(agg['fragile_top1'])}")
    if agg["fail_top1"]:
        print(f"  FAIL top-1 ({len(agg['fail_top1'])}): "
              f"{', '.join(agg['fail_top1'])}")
    print(f"\n  ROBUST top-5 ({len(agg['robust_top5'])}): "
          f"{len(agg['robust_top5'])}/32")
    if agg["fail_top5"]:
        print(f"  FAIL top-5 ({len(agg['fail_top5'])}): "
              f"{', '.join(agg['fail_top5'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43,44,45")
    p.add_argument("--out-dir", type=str,
                    default="research/findings/raw/g11_bg")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)

    agg = aggregate(seeds, out_dir)
    print_summary(agg)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(agg, indent=2, default=str))
        print(f"\n[OUT] -> {args.out}")


if __name__ == "__main__":
    main()
