"""
Permuted-label control: tests whether word-action mapping is real
learning vs random architectural structure.

For each W->A confusion matrix, compute accuracy under all 24
permutations of (token -> action) mapping. If the network learned
real labels, the TRUE mapping should be the BEST permutation. If
"learning" is just architectural noise, true mapping is one of 24
random options.

Usage:
    python -m research.runners.permuted_label_check
    # auto-scans research/findings/raw/g11_bg/text_eval_*_seed*.json

    python -m research.runners.permuted_label_check --pattern "text_eval_arch_*_seed*.json"
    # filter to specific runs

    python -m research.runners.permuted_label_check --out report.md

Output: markdown table per condition with true acc, best perm, excess.
Aligned ratio (true == best across seeds) is the key metric.

  aligned 6/6  -> definitive real learning (1/24 chance per seed = 1/4096 chance)
  aligned 4/6  -> probably real learning
  aligned 0/6  -> no real word-action learning (16% expected by chance is very low,
                  so 0/6 suggests structure is ANTI-correlated with labels in some sense)
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"

WORDS = ["north", "east", "south", "west"]
TRUE_MAP = {"north": "N", "east": "E", "south": "S", "west": "W"}
ACTIONS = ("N", "E", "S", "W")


def acc_for_mapping(cm: dict, mapping: dict) -> float:
    """Compute W->A accuracy for a given (word -> action) mapping."""
    correct = 0
    total = 0
    for word, row in cm.items():
        target = mapping[word]
        for action, count in row.items():
            count = int(count)
            total += count
            if action == target:
                correct += count
    return correct / max(total, 1)


def best_permutation(cm: dict) -> tuple[float, tuple]:
    """Find the highest-accuracy permutation. Returns (acc, perm)."""
    best_acc = 0.0
    best_perm = None
    for perm in itertools.permutations(ACTIONS):
        mapping = dict(zip(WORDS, perm))
        acc = acc_for_mapping(cm, mapping)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_acc, best_perm


def parse_condition_seed(filename: str) -> Optional[tuple[str, int]]:
    """Extract (condition_label, seed) from text_eval filename.

    Examples:
        text_eval_v2_swr500_seed42.json -> ('v2_swr500', 42)
        text_eval_R3R6_100ep_HebOff_v2_seed44.json -> ('R3R6_100ep_HebOff_v2', 44)
        text_eval_h4_isolation_seed100.json -> ('h4_isolation', 100)
        text_eval_arch_motor50_seed42.json -> ('arch_motor50', 42)
    """
    m = re.match(r"text_eval_(.+)_seed(\d+)\.json$", filename)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="text_eval_*_seed*.json",
                    help="Glob pattern for input files (default: all evals)")
    ap.add_argument("--out", default=None,
                    help="Output markdown file (default: stdout)")
    args = ap.parse_args()

    files = sorted(RAW_DIR.glob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern}")
        return

    # Group by condition
    by_condition = defaultdict(list)
    for path in files:
        parsed = parse_condition_seed(path.name)
        if parsed is None:
            continue
        cond, seed = parsed
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        cm = (data.get("word_to_action_eval") or {}).get("confusion_matrix")
        if not cm or len(cm) != 4:
            continue
        # Normalize key/value to int
        cm_norm = {w: {a: int(cm.get(w, {}).get(a, 0)) for a in ACTIONS}
                   for w in WORDS}
        by_condition[cond].append((seed, cm_norm))

    out = []
    out.append("# Permuted-label control across conditions")
    out.append("")
    out.append("For each seed: compute W->A accuracy under all 24 permutations of")
    out.append("(token -> action). If the network learned real labels, the TRUE")
    out.append("mapping should be the BEST permutation.")
    out.append("")
    out.append("| condition | seed | true | best | excess | best perm | aligned? |")
    out.append("|---|---|---|---|---|---|---|")
    summary = {}
    for cond in sorted(by_condition.keys()):
        runs = sorted(by_condition[cond])
        excesses = []
        aligned = 0
        true_accs = []
        best_accs = []
        for seed, cm in runs:
            true_acc = acc_for_mapping(cm, TRUE_MAP)
            best_acc, best_perm = best_permutation(cm)
            excess = best_acc - true_acc
            is_aligned = (best_perm == ACTIONS)
            true_accs.append(true_acc)
            best_accs.append(best_acc)
            excesses.append(excess)
            if is_aligned:
                aligned += 1
            out.append(
                f"| {cond} | {seed} | {100*true_acc:.0f}% | {100*best_acc:.0f}% | "
                f"+{100*excess:.1f}pp | {''.join(best_perm)} | "
                f"{'YES' if is_aligned else 'no'} |"
            )
        summary[cond] = {
            "n": len(runs),
            "true_mean": statistics.mean(true_accs) if true_accs else None,
            "best_mean": statistics.mean(best_accs) if best_accs else None,
            "excess_mean": statistics.mean(excesses) if excesses else None,
            "aligned": aligned,
        }

    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| condition | n | true mean | best mean | excess | aligned/n |")
    out.append("|---|---|---|---|---|---|")
    for cond, s in summary.items():
        if s["n"] == 0:
            continue
        out.append(
            f"| {cond} | {s['n']} | {100*s['true_mean']:.1f}% | "
            f"{100*s['best_mean']:.1f}% | +{100*s['excess_mean']:.1f}pp | "
            f"{s['aligned']}/{s['n']} |"
        )

    out.append("")
    out.append("**Interpretation:**")
    out.append("- aligned/n = 6/6: definitive real learning")
    out.append("- aligned/n >= 4/6: probably real")
    out.append("- aligned/n = 0/6: no real word-action learning; structure is mis-")
    out.append("  aligned with labels (just architectural noise)")

    output = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote permuted-label report to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
