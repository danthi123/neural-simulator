"""Multi-seed direct binding diagnostic at 400ev.

Mirrors `direct_binding_multiseed.py` (800ev version) but points at
the 400ev cache dir for the Direction B Probe-2 multi-seed expansion.

Aggregates n_correct / n_total across all 3 seeds; reports per-seed
+ aggregate; applies the frozen 0.80 bar (set in advance; never
moved).
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the existing test_one_checkpoint helper byte-unchanged
from importlib import util as _import_util
_diag_path = os.path.join(_HERE, "direct_binding_phase1_comparison.py")
_spec = _import_util.spec_from_file_location("_db", _diag_path)
_db = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_db)
test_one_checkpoint = _db.test_one_checkpoint


SEEDS = [42, 43, 44]
CACHE_DIR = "research/findings/raw/unified_per_regime/phase1_400ev"


def main():
    per_seed_results = []
    for seed in SEEDS:
        label = f"400ev seed {seed}"
        result = test_one_checkpoint(seed, CACHE_DIR, label)
        per_seed_results.append({
            "seed": seed,
            "n_correct": result["n_correct"],
            "n_total": result["n_total"],
            "accuracy": result["accuracy"],
            "per_word": result["per_word"],
        })

    total_correct = sum(r["n_correct"] for r in per_seed_results)
    total = sum(r["n_total"] for r in per_seed_results)
    aggregate_acc = total_correct / total

    print("\n=== MULTI-SEED AGGREGATE (400ev) ===")
    for r in per_seed_results:
        print(f"  seed {r['seed']}: {r['n_correct']}/{r['n_total']} = {100.0*r['accuracy']:.1f}%")
    print(f"  AGGREGATE: {total_correct}/{total} = {100.0*aggregate_acc:.1f}%")

    all_above_bar = all(r["accuracy"] >= 0.80 for r in per_seed_results)
    aggregate_above = aggregate_acc >= 0.80
    if all_above_bar:
        print("  --> TRUSTWORTHY DIRECT BINDING VALIDATED at 400ev: "
              "all 3 seeds >= 0.80 frozen bar.")
    elif aggregate_above:
        n_below = sum(1 for r in per_seed_results if r["accuracy"] < 0.80)
        print(f"  --> Aggregate >= 0.80 but {n_below}/3 seeds below 0.80; "
              "mixed validation.")
    else:
        print("  --> Aggregate < 0.80; trustworthy direct binding NOT validated "
              "multi-seed at 400ev.")

    results = {
        "seeds": SEEDS,
        "phase1_events_per_word": 400,
        "n_words_per_seed": 16,
        "per_seed": per_seed_results,
        "aggregate": {
            "total_correct": total_correct,
            "total": total,
            "accuracy": aggregate_acc,
            "all_seeds_above_0.80": all_above_bar,
        },
    }
    out = "research/findings/raw/direct_binding_multiseed_400ev.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
