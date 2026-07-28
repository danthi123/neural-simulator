"""Aggregate per-seed G2 (or later) result JSONs into mean/std per epoch.

Usage:
    python -m research.runners.aggregate_seeds \\
        research/findings/raw/g2-seed*.json \\
        --out research/findings/raw/g2-aggregate.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def aggregate(result_paths):
    """Load per-seed JSONs and compute per-epoch summary stats.

    Returns dict with:
        per_seed: list of dicts with seed, per_epoch (list of dicts), final_test_acc
        mean_test_acc_by_epoch: list of floats
        std_test_acc_by_epoch: list of floats
        mean_train_acc_by_epoch: list of floats
        delta_epoch0_to_epoch_last: mean change in test_acc
        n_seeds, n_epochs
    """
    per_seed = []
    for p in sorted(result_paths):
        data = json.load(open(p))
        per_seed.append({
            "path": str(p),
            "seed": data.get("seed"),
            "epochs": data["epochs"],
            "final_test_acc": data["epochs"][-1]["test_accuracy"],
        })

    assert per_seed, "No result files found"
    n_epochs = len(per_seed[0]["epochs"])
    assert all(len(s["epochs"]) == n_epochs for s in per_seed), \
        "Per-seed JSONs have different epoch counts"

    test_matrix = np.array([[ep["test_accuracy"] for ep in s["epochs"]] for s in per_seed])
    train_matrix = np.array([[ep["train_accuracy"] for ep in s["epochs"]] for s in per_seed])

    mean_test = test_matrix.mean(axis=0).tolist()
    std_test = test_matrix.std(axis=0).tolist()
    mean_train = train_matrix.mean(axis=0).tolist()

    delta = (test_matrix[:, -1] - test_matrix[:, 0]).tolist()

    return {
        "n_seeds": len(per_seed),
        "n_epochs": n_epochs,
        "seeds": [s["seed"] for s in per_seed],
        "mean_test_acc_by_epoch": mean_test,
        "std_test_acc_by_epoch": std_test,
        "mean_train_acc_by_epoch": mean_train,
        "per_seed_final_test_acc": [s["final_test_acc"] for s in per_seed],
        "per_seed_delta_test_acc_first_to_last": delta,
        "mean_delta_first_to_last": float(np.mean(delta)),
        "per_seed": per_seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="Per-seed result JSONs (glob patterns expanded)")
    ap.add_argument("--out", default="aggregate.json",
                    help="Output JSON path")
    args = ap.parse_args()

    expanded = []
    for p in args.paths:
        expanded.extend(sorted(glob.glob(p)))
    expanded = [p for p in expanded if Path(p).is_file()]
    assert expanded, f"No files matched: {args.paths}"

    agg = aggregate(expanded)

    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n=== Aggregate of {agg['n_seeds']} seeds, {agg['n_epochs']} epochs ===")
    print(f"Seeds: {agg['seeds']}")
    print(f"\nPer-epoch mean test accuracy:")
    for i, m in enumerate(agg["mean_test_acc_by_epoch"]):
        std = agg["std_test_acc_by_epoch"][i]
        print(f"  epoch {i}: {m:.3f} +/- {std:.3f}")
    print(f"\nPer-seed final test accuracy: "
          f"{[round(x, 3) for x in agg['per_seed_final_test_acc']]}")
    print(f"Per-seed delta (last - first): "
          f"{[round(x, 3) for x in agg['per_seed_delta_test_acc_first_to_last']]}")
    print(f"Mean delta first->last: {agg['mean_delta_first_to_last']:+.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
