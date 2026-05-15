"""Multi-seed aggregator for multi-bridge concept pool training.

Reads seed${N}_set${M}.json files and produces a cross-seed comparison
table: which words are robust (PASS across all seeds) vs fragile
(PASS at some seeds but not others).

Usage:
    python -m research.runners.multibridge_multiseed_aggregate \
        --seeds 42,43,44,45,46 --sets set2,set3,set4,set5
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_seed_set(out_dir: Path, seed: int, set_name: str) -> dict | None:
    """Load a single (seed, set) JSON result. Returns None if missing."""
    path = out_dir / f"seed{seed}_{set_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def aggregate_set(out_dir: Path, seeds: list[int], set_name: str) -> dict:
    """Aggregate one set's results across multiple seeds.

    Returns dict with:
      - per_word: word -> {pass_count, total_seeds, ratios, pools}
      - n_pass_per_seed: seed -> int
      - cross_seed_robust: list of words with PASS across ALL seeds
      - cross_seed_fragile: list of words with PASS at <all seeds
    """
    per_word = defaultdict(lambda: {
        "pass_count": 0,
        "total_seeds": 0,
        "ratios": [],
        "off_targets": set(),
    })
    n_pass_per_seed = {}
    seeds_loaded = []

    for seed in seeds:
        result = load_seed_set(out_dir, seed, set_name)
        if not result:
            continue
        seeds_loaded.append(seed)
        n_pass_per_seed[seed] = result["n_pass"]
        for word, info in result["results"].items():
            target_rate = info["target_rate"]
            max_off = info["max_off_target"]
            max_off_pool = info["max_off_target_pool"]
            target_pool = info["target"]
            ratio = target_rate / max_off if max_off > 0 else float('inf')
            per_word[word]["total_seeds"] += 1
            if target_rate > max_off:
                per_word[word]["pass_count"] += 1
            per_word[word]["ratios"].append(ratio)
            per_word[word]["off_targets"].add(max_off_pool)

    cross_seed_robust = sorted([
        w for w, info in per_word.items()
        if info["pass_count"] == len(seeds_loaded) and len(seeds_loaded) > 0
    ])
    cross_seed_fragile = sorted([
        w for w, info in per_word.items()
        if 0 < info["pass_count"] < len(seeds_loaded)
    ])
    cross_seed_fail = sorted([
        w for w, info in per_word.items()
        if info["pass_count"] == 0 and info["total_seeds"] > 0
    ])

    return {
        "set_name": set_name,
        "seeds_loaded": seeds_loaded,
        "per_word": {w: {
            "pass_count": info["pass_count"],
            "total_seeds": info["total_seeds"],
            "ratios": info["ratios"],
            "off_targets": sorted(info["off_targets"]),
        } for w, info in per_word.items()},
        "n_pass_per_seed": n_pass_per_seed,
        "cross_seed_robust": cross_seed_robust,
        "cross_seed_fragile": cross_seed_fragile,
        "cross_seed_fail": cross_seed_fail,
    }


def print_summary(agg: dict) -> None:
    """Print human-readable aggregator summary."""
    s = agg["set_name"]
    seeds = agg["seeds_loaded"]
    print(f"=== {s} ===")
    print(f"  seeds loaded: {seeds}")
    if not seeds:
        print(f"  (no results found)")
        return
    print(f"  PASS per seed: ", end="")
    print(", ".join(f"{seed}={agg['n_pass_per_seed'][seed]}/16"
                     for seed in seeds))
    # Combined Phase 1 rate
    total_pass = sum(agg['n_pass_per_seed'].values())
    total_words = 16 * len(seeds)
    print(f"  combined PASS: {total_pass}/{total_words} "
          f"({100*total_pass/total_words:.1f}%)")
    # Robust / fragile / fail breakdown
    n_robust = len(agg["cross_seed_robust"])
    n_fragile = len(agg["cross_seed_fragile"])
    n_fail = len(agg["cross_seed_fail"])
    n_total = n_robust + n_fragile + n_fail
    print(f"  cross-seed: {n_robust} robust, {n_fragile} fragile, "
          f"{n_fail} fail (of {n_total} words)")
    if agg["cross_seed_robust"]:
        print(f"  ROBUST ({n_robust}): {', '.join(agg['cross_seed_robust'])}")
    if agg["cross_seed_fragile"]:
        print(f"  FRAGILE ({n_fragile}): {', '.join(agg['cross_seed_fragile'])}")
    if agg["cross_seed_fail"]:
        print(f"  FAIL ({n_fail}): {', '.join(agg['cross_seed_fail'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43",
                    help="Comma-separated seed list")
    p.add_argument("--sets", type=str,
                    default="set2,set3,set4,set5",
                    help="Comma-separated set names")
    p.add_argument("--out-dir", type=str,
                    default="research/findings/raw/g11_bg/concept_pool_demo")
    p.add_argument("--out", type=str, default=None,
                    help="Optional JSON output path")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    set_names = [s.strip() for s in args.sets.split(",")]
    out_dir = Path(args.out_dir)

    print(f"=== Multi-seed multi-bridge aggregator ===")
    print(f"  seeds: {seeds}")
    print(f"  sets:  {set_names}")
    print(f"  dir:   {out_dir}")
    print()

    all_aggs = {}
    for set_name in set_names:
        agg = aggregate_set(out_dir, seeds, set_name)
        all_aggs[set_name] = agg
        print_summary(agg)
        print()

    # Overall summary
    print(f"=== OVERALL ===")
    total_pass_overall = 0
    total_words_overall = 0
    total_robust = 0
    total_fragile = 0
    total_fail = 0
    for s, agg in all_aggs.items():
        for seed in agg["seeds_loaded"]:
            total_pass_overall += agg["n_pass_per_seed"][seed]
            total_words_overall += 16
        total_robust += len(agg["cross_seed_robust"])
        total_fragile += len(agg["cross_seed_fragile"])
        total_fail += len(agg["cross_seed_fail"])
    if total_words_overall > 0:
        print(f"  total PASS: {total_pass_overall}/{total_words_overall} "
              f"({100*total_pass_overall/total_words_overall:.1f}%)")
    print(f"  cross-seed: {total_robust} robust, "
          f"{total_fragile} fragile, {total_fail} fail")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert sets to lists for JSON
        out_data = {
            "seeds": seeds,
            "sets": set_names,
            "per_set": all_aggs,
            "totals": {
                "pass": total_pass_overall,
                "total": total_words_overall,
                "robust": total_robust,
                "fragile": total_fragile,
                "fail": total_fail,
            },
        }
        out_path.write_text(json.dumps(out_data, indent=2, default=str))
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
