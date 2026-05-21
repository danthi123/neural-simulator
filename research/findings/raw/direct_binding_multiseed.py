"""Multi-seed validation of direct binding at 800ev Phase-1.

Single-seed (42) showed 15/16 = 93.8% direct binding accuracy on the
16-word task at 800ev (vs 11/16 = 68.8% at 200ev). This multi-seed
expansion validates across seeds 42 + 43 + 44.

PROTOCOL:
For each seed in {42, 43, 44}:
  Load 800ev Phase-1 checkpoint
  Query all 16 trained words via measure_pool_firing
  Report n_correct/16
Then aggregate: total direct binding accuracy across 3 × 16 = 48
queries.

DECISION RULE:
If ALL 3 seeds >= 0.80 (frozen direct_retain_min bar): TRUSTWORTHY
direct binding capability VALIDATED at biological scale on the unified
substrate.
If aggregate >= 0.80 but variance: trustworthy in aggregate but
variability needs noting.
If any seed < 0.80: capability not yet trustworthy multi-seed.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the seed-42 diagnostic's test_one_checkpoint helper.
from importlib import util as _import_util
_diag_path = os.path.join(_HERE, "direct_binding_phase1_comparison.py")
_spec = _import_util.spec_from_file_location("_db", _diag_path)
_db = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_db)
test_one_checkpoint = _db.test_one_checkpoint


SEEDS = [42, 43, 44]
CACHE_DIR = "research/findings/raw/unified_per_regime/phase1_800ev"


def main():
    per_seed_results = []
    for seed in SEEDS:
        label = f"800ev seed {seed}"
        result = test_one_checkpoint(seed, CACHE_DIR, label)
        per_seed_results.append({
            "seed": seed,
            "n_correct": result["n_correct"],
            "n_total": result["n_total"],
            "accuracy": result["accuracy"],
            "per_word": result["per_word"],
        })

    # Aggregate
    total_correct = sum(r["n_correct"] for r in per_seed_results)
    total = sum(r["n_total"] for r in per_seed_results)
    aggregate_acc = total_correct / total

    print("\n=== MULTI-SEED AGGREGATE (800ev) ===")
    for r in per_seed_results:
        print(f"  seed {r['seed']}: {r['n_correct']}/{r['n_total']} = {100.0*r['accuracy']:.1f}%")
    print(f"  AGGREGATE: {total_correct}/{total} = {100.0*aggregate_acc:.1f}%")

    all_above_bar = all(r["accuracy"] >= 0.80 for r in per_seed_results)
    aggregate_above = aggregate_acc >= 0.80
    if all_above_bar:
        print(
            "  --> TRUSTWORTHY DIRECT BINDING VALIDATED: all 3 seeds "
            "individually >= 0.80 frozen bar."
        )
    elif aggregate_above:
        n_below = sum(1 for r in per_seed_results if r["accuracy"] < 0.80)
        print(
            f"  --> Aggregate >= 0.80 but {n_below}/3 seeds individually below "
            "0.80; mixed validation."
        )
    else:
        print(
            "  --> Aggregate < 0.80; trustworthy direct binding NOT validated "
            "multi-seed (single-seed 42 may have been outlier)."
        )

    results = {
        "seeds": SEEDS,
        "phase1_events_per_word": 800,
        "n_words_per_seed": 16,
        "per_seed": per_seed_results,
        "aggregate": {
            "total_correct": total_correct,
            "total": total,
            "accuracy": aggregate_acc,
            "all_seeds_above_0.80": all_above_bar,
        },
    }
    out = "research/findings/raw/direct_binding_multiseed.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
