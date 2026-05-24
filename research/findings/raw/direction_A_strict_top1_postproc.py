"""Direction A STRICT TOP-1 post-processor (STRENGTHEN fix per
adversarial reviewer VERDICT BLOCK).

The reviewer found a critical methodology defect: the top-3 readout
is DEGENERATE with the engram-captures-all-3-slot-pools mechanism.
At seed 42 cached result (0.875 top-3 acc), strict top-1 accuracy
is only 2/8 = 25% -- the positional cue is NOT doing real work; the
test collapses to multitag set-membership.

This post-processor reads the cached trials JSON (which already has
topK_words for each sequence) and computes:
  (a) strict top-1 acc: true_slot_word == topK_words[0]
  (b) per-rank distribution: which rank (0/1/2) does the true word
      fall in?

Re-evaluation uses the SAME pre-registered 0.80 multi-seed bar; no
bar tuning, no test redesign, just a tighter metric on the existing
data.

NUMPY/stdlib only; ~1 s wall.
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "direction_A_ec_context_cache")
OUT_JSON = os.path.join(_HERE, "direction_A_strict_top1_postproc.json")
BAR = 0.80


def analyze_per_seq(per_seq):
    """For each seq entry, compute strict top-1 + rank of true word."""
    n_top1 = 0
    n_total = len(per_seq)
    rank_dist = Counter()
    per_seq_strict = []
    for p in per_seq:
        true = p["true_slot3"]
        topK = p["topK_words"]
        # Strict top-1
        top1 = (topK[0] == true) if len(topK) > 0 else False
        if top1: n_top1 += 1
        # Rank
        if true in topK:
            rank = topK.index(true)
        else:
            rank = -1  # not in top-K
        rank_dist[rank] += 1
        per_seq_strict.append({
            "seq_idx": p["seq_idx"], "sequence": p["sequence"],
            "true_slot3": true, "topK_words": topK,
            "strict_top1_correct": top1, "true_rank": rank,
        })
    return {
        "n_top1": n_top1, "n_total": n_total,
        "strict_top1_acc": n_top1 / n_total if n_total > 0 else 0.0,
        "rank_distribution": dict(rank_dist),
        "per_seq_strict": per_seq_strict,
    }


def main():
    print(f"=== Direction A STRICT TOP-1 post-processor ===",
          flush=True)
    print(f"  (STRENGTHEN fix per reviewer VERDICT BLOCK)",
          flush=True)
    print(f"  Pre-registered bar: {BAR} multi-seed (frozen)",
          flush=True)

    # Find all per-seed trials caches
    seed_results = []
    for seed in [42, 43, 44, 45, 46]:
        trials_p = os.path.join(
            CACHE_DIR, f"trials_full_seed{seed}.json")
        if not os.path.exists(trials_p):
            print(f"  seed {seed}: no trials cache; skip", flush=True)
            continue
        with open(trials_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_seq = data.get("per_seq", [])
        analysis = analyze_per_seq(per_seq)
        analysis["seed"] = seed
        analysis["original_top3_acc"] = data.get("slot3_accuracy")
        seed_results.append(analysis)
        print(f"\n  seed {seed}:", flush=True)
        print(f"    top-3 (original): "
              f"{analysis['original_top3_acc']:.3f}",
              flush=True)
        print(f"    strict top-1: {analysis['strict_top1_acc']:.3f}"
              f" ({analysis['n_top1']}/{analysis['n_total']})",
              flush=True)
        print(f"    rank distribution: "
              f"{dict(sorted(analysis['rank_distribution'].items()))}",
              flush=True)
        for p in analysis["per_seq_strict"]:
            mark = "TOP1" if p["strict_top1_correct"] else (
                "in-top3" if p["true_rank"] >= 0 else "MISS")
            print(f"      seq {p['seq_idx']:1d}: true={p['true_slot3']:6s} "
                  f"rank={p['true_rank']:2d} {mark}", flush=True)

    if not seed_results:
        print("\n  [FATAL] no seed caches found; cannot post-process",
              flush=True)
        return 1

    n_seeds = len(seed_results)
    if n_seeds < 3:
        print(f"\n  [WARN] only {n_seeds} seed(s) cached; "
              f"multi-seed verdict needs 3+ seeds", flush=True)

    top3_accs = [r["original_top3_acc"] for r in seed_results
                  if r["original_top3_acc"] is not None]
    top1_accs = [r["strict_top1_acc"] for r in seed_results]
    top3_mean = float(np.mean(top3_accs)) if top3_accs else 0.0
    top1_mean = float(np.mean(top1_accs)) if top1_accs else 0.0

    print(f"\n=== MULTI-SEED (n={n_seeds}) ===", flush=True)
    print(f"  top-3 mean (original):  {top3_mean:.3f}", flush=True)
    print(f"  strict top-1 mean:      {top1_mean:.3f}", flush=True)
    print(f"  delta:                  {top3_mean - top1_mean:+.3f}",
          flush=True)
    chance_top3 = 3.0 / 16.0
    chance_top1 = 1.0 / 16.0
    print(f"  chance top-3:           {chance_top3:.3f}", flush=True)
    print(f"  chance top-1:           {chance_top1:.3f}", flush=True)

    print(f"\n=== VERDICT (STRENGTHEN-only fix; bar UNCHANGED at "
          f"{BAR}) ===", flush=True)
    top1_pass = top1_mean >= BAR
    top1_above_chance = top1_mean > 2 * chance_top1
    if top1_pass:
        verdict = "STRICT_TOP1_PASS"
        print(f"  Multi-seed strict top-1 >= {BAR} -- the ec_context"
              f" positional cue is GENUINELY load-bearing; the "
              f"mechanism produces real positional binding.",
              flush=True)
    elif top1_above_chance:
        verdict = "STRICT_TOP1_ABOVE_CHANCE_BELOW_BAR"
        print(f"  Multi-seed strict top-1 {top1_mean:.3f} > "
              f"2*chance {2*chance_top1:.3f} but < {BAR}; partial "
              f"signal -- the ec_context cue contributes but doesn't"
              f" reliably select the slot-i word as top-1.",
              flush=True)
    else:
        verdict = "STRICT_TOP1_AT_CHANCE"
        print(f"  Multi-seed strict top-1 {top1_mean:.3f} at chance "
              f"({chance_top1:.3f}); the positional cue is NOT "
              f"doing positional work -- the apparent top-3 PASS is"
              f" degenerate (collapses to multitag set-membership).",
              flush=True)

    out = {
        "bar": BAR, "chance_top1": chance_top1,
        "chance_top3": chance_top3,
        "n_seeds_analyzed": n_seeds,
        "top3_mean_original": top3_mean,
        "strict_top1_mean": top1_mean,
        "per_seed": seed_results,
        "verdict": verdict,
        "interpretation": (
            "STRICT TOP-1 metric introduced by adversarial reviewer "
            "VERDICT BLOCK (2026-05-24); top-3 readout is degenerate "
            "with the engram-captures-all-slot-words mechanism. "
            "Strict top-1 = true positional binding test."
        ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
