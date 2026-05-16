"""Sparse pattern-set seed-quality analysis (research, CPU-only).

The corrected failure root-cause (2026-05-16-cross-benchmark-failure-
analysis): failure is INDEX-intrinsic -- specific positions in the
per-seed `generate_sparse_patterns` set are weak (idx-12 @ 64-tier;
idx 10/42 in the 320 benchmarks; 1-2 weak idxs at the 32-tier for
seeds 43/46). This asks, with a large sample and a falsifiable
predictor:

  1. Across many seeds, what is the pattern-set max-overlap
     distribution, and which seeds are "cleanest" (fewest
     high-overlap outlier patterns -> predicted fewest failures)?
  2. VALIDATION: does the structural outlier metric correctly RANK
     the KNOWN 160 multi-seed result (seeds 42/44/45 = 100% per-bridge
     vs 43/46 = 96.9%/93.8%)? If the metric predicts that ordering,
     it is trustworthy and its 64-tier seed recommendation can be
     handed to the flagged recovery task (one targeted run instead of
     a multi-hour seed search).

Pure: only `generate_sparse_patterns` + numpy. No GPU, no retrain,
no implementation. Output INFORMS the flagged task; builds nothing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from research.runners.concept_pool_sparse_distributed import (
    generate_sparse_patterns,
)


def pattern_set_quality(n_concepts: int, n_pool: int, k: int,
                          seed: int) -> dict:
    """Max-pairwise-overlap distribution + outlier count for one
    seed's pattern set. 'outliers' = patterns whose max overlap with
    any other pattern exceeds mean + 2*std (the failure-prone ones)."""
    pats = generate_sparse_patterns(n_concepts, n_pool, k, seed)
    S = [set(p) for p in pats]
    maxov = np.array([
        max(len(S[i] & S[j]) for j in range(n_concepts) if j != i)
        for i in range(n_concepts)
    ], dtype=float)
    thr = maxov.mean() + 2.0 * maxov.std()
    outliers = sorted(int(i) for i in np.where(maxov > thr)[0])
    return {
        "seed": seed, "mean_maxov": float(maxov.mean()),
        "std_maxov": float(maxov.std()), "max_maxov": float(maxov.max()),
        "outlier_thr": float(thr), "n_outliers": len(outliers),
        "outlier_idxs": outliers,
    }


# Known empirical anchor: 160 tier (32-concept) per-bridge top-1.
# From 2026-05-16-G20-sparse-160-multiseed-VALIDATED.md.
KNOWN_160 = {42: 1.000, 43: 0.969, 44: 1.000, 45: 1.000, 46: 0.938}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-lo", type=int, default=42)
    p.add_argument("--seed-hi", type=int, default=101)
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    seeds = list(range(args.seed_lo, args.seed_hi + 1))

    # --- VALIDATION at 32-concept against KNOWN 160 multi-seed ---
    print("=== VALIDATION: structural metric vs known 160 (32-conc) ===")
    val = []
    for s in sorted(KNOWN_160):
        q = pattern_set_quality(32, args.n_pool, args.k, s)
        val.append((s, q["n_outliers"], q["max_maxov"], KNOWN_160[s]))
        print(f"  seed {s}: n_outliers={q['n_outliers']} "
              f"max_ov={q['max_maxov']:.0f} | KNOWN per-bridge="
              f"{KNOWN_160[s]:.3f}")
    # Spearman-ish: does higher n_outliers/max_ov => lower accuracy?
    import statistics
    by_acc = sorted(val, key=lambda r: r[3])           # worst acc first
    by_ov = sorted(val, key=lambda r: (-r[1], -r[2]))  # worst struct first
    rank_match = [r[0] for r in by_acc] == [r[0] for r in by_ov]
    # softer: do the two 100% seeds have <= outliers than the two <100%?
    clean = [r for r in val if r[3] >= 0.999]
    weak = [r for r in val if r[3] < 0.999]
    sep = (max((r[1] for r in clean), default=0)
           <= min((r[1] for r in weak), default=99)) if weak else None
    print(f"  exact-rank-match={rank_match} | "
          f"clean-vs-weak outlier separation={sep} "
          f"(clean max_outliers={max((r[1] for r in clean),default=0)}, "
          f"weak min_outliers={min((r[1] for r in weak),default='NA')})")

    # --- 64-concept tier: rank all seeds, recommend the cleanest ---
    print(f"\n=== 64-concept tier seed-quality (seeds "
          f"{args.seed_lo}-{args.seed_hi}) ===")
    rows = [pattern_set_quality(64, args.n_pool, args.k, s)
            for s in seeds]
    ranked = sorted(rows, key=lambda r: (r["n_outliers"],
                                          r["max_maxov"]))
    print("  cleanest 8 seeds (fewest outlier patterns):")
    for r in ranked[:8]:
        print(f"    seed {r['seed']:3d}: n_outliers="
              f"{r['n_outliers']} max_ov={r['max_maxov']:.0f} "
              f"outlier_idxs={r['outlier_idxs']}")
    s42 = next(r for r in rows if r["seed"] == 42)
    print(f"  (seed 42 = the 320-tier seed: n_outliers="
          f"{s42['n_outliers']} idxs={s42['outlier_idxs']} "
          f"-- idx-12 was the empirical 64-tier failure)")
    best = ranked[0]
    zero = [r["seed"] for r in ranked if r["n_outliers"] == 0]
    print(f"\n=== RECOMMENDATION (for the flagged recovery task) ===")
    print(f"  cleanest seed: {best['seed']} "
          f"(n_outliers={best['n_outliers']}, "
          f"max_ov={best['max_maxov']:.0f})")
    print(f"  zero-outlier seeds: {zero if zero else 'NONE in range'}")
    print(f"  -> the flagged per-seed recovery should try seed "
          f"{best['seed']} FIRST (one targeted 320 retrain) instead "
          f"of a blind multi-seed sweep.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "validation_32c": [
                {"seed": s, "n_outliers": no, "max_ov": mo,
                 "known_perbridge": acc} for s, no, mo, acc in val],
            "validation_exact_rank_match": rank_match,
            "validation_clean_weak_separation": sep,
            "tier64": rows,
            "recommendation": {"cleanest_seed": best["seed"],
                                "zero_outlier_seeds": zero},
        }, open(args.out, "w"), indent=2)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
