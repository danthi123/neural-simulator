"""Meta-analysis of all text I/O experiments.

Aggregates results across all text_eval_*.json files in the findings raw
directory. Produces a comparison table sorted by W->A accuracy and a
summary of per-direction patterns across all runs.

Useful for tracking the project's overall progress and identifying which
variants have been tested and what their results were.

Usage:
  python -m research.runners.text_io_meta_analysis
  python -m research.runners.text_io_meta_analysis --dir custom/path
  python -m research.runners.text_io_meta_analysis --min-trials 50  # filter smokes
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional


def _binom_p_above(correct: int, n: int, p_chance: float = 0.25) -> float:
    """One-sided p-value for X >= correct under Binomial(n, p_chance)."""
    if n == 0:
        return 1.0
    mu = n * p_chance
    sigma = math.sqrt(n * p_chance * (1 - p_chance))
    if sigma < 1e-9:
        return 1.0 if correct < mu else 0.0
    z = (correct - 0.5 - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


def parse_result_json(path: Path) -> Optional[dict]:
    """Parse a text_eval_*.json result file. Returns None if not parseable."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if "image_to_word_eval" not in d or "word_to_action_eval" not in d:
        return None
    iw = d["image_to_word_eval"]
    wa = d["word_to_action_eval"]

    # Try to derive a label from filename
    label = path.stem.replace("text_eval_", "").replace("_seed", " seed=")

    # Parse per-direction W->A from confusion matrix
    word_to_action_letter = {"north": "N", "east": "E", "south": "S", "west": "W"}
    wa_per = {}
    for word in ["north", "east", "south", "west"]:
        row = wa.get("confusion_matrix", {}).get(word, {})
        total = sum(row.values()) if row else 0
        correct = row.get(word_to_action_letter[word], 0) if row else 0
        if total > 0:
            wa_per[word] = correct / total
        else:
            wa_per[word] = None

    return {
        "label": label,
        "path": str(path),
        "iw_correct": iw.get("correct", 0),
        "iw_n": iw.get("n_trials", 0),
        "iw_acc": iw.get("accuracy", 0.0),
        "iw_p": _binom_p_above(iw.get("correct", 0), iw.get("n_trials", 0)),
        "wa_correct": wa.get("correct", 0),
        "wa_n": wa.get("n_trials", 0),
        "wa_acc": wa.get("accuracy", 0.0),
        "wa_p": _binom_p_above(wa.get("correct", 0), wa.get("n_trials", 0)),
        "wa_per": wa_per,
        "training_correct_rate": (
            d.get("training_stats", [{}])[0].get("correct_move_rate", None)
            if d.get("training_stats") else None
        ),
        "n_episodes": d.get("n_episodes"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str,
                    default="research/findings/raw/g11_bg/",
                    help="directory containing text_eval_*.json files")
    ap.add_argument("--min-trials", type=int, default=50,
                    help="filter results with fewer than this many eval trials")
    ap.add_argument("--include", type=str, default="text_eval_",
                    help="filename prefix filter (default text_eval_)")
    args = ap.parse_args()

    base = Path(args.dir)
    json_files = sorted([
        p for p in base.glob("*.json")
        if p.name.startswith(args.include) and p.is_file()
    ])

    results = []
    for p in json_files:
        r = parse_result_json(p)
        if r is None:
            continue
        if r["iw_n"] < args.min_trials and r["wa_n"] < args.min_trials:
            continue
        results.append(r)

    if not results:
        print("No results found.")
        return

    # Sort by W->A accuracy descending
    results.sort(key=lambda r: r["wa_acc"], reverse=True)

    # Print summary table
    print("=" * 100)
    print(f"TEXT I/O META-ANALYSIS — {len(results)} runs in {base}")
    print("=" * 100)
    print(f"{'Run label':<48} {'I->W':<14} {'W->A':<14} {'p_WA':<8} train")
    print("-" * 100)
    for r in results:
        iw_str = f"{r['iw_correct']}/{r['iw_n']}={r['iw_acc']*100:.0f}%"
        wa_str = f"{r['wa_correct']}/{r['wa_n']}={r['wa_acc']*100:.0f}%"
        train_str = (f"{r['training_correct_rate']*100:.0f}%"
                     if r['training_correct_rate'] is not None else "?")
        print(f"{r['label'][:46]:<48} {iw_str:<14} {wa_str:<14} "
              f"{r['wa_p']:<8.3f} {train_str}")

    # Aggregate statistics
    print()
    print("=" * 100)
    print("AGGREGATE STATISTICS")
    print("=" * 100)

    iw_accs = [r["iw_acc"] for r in results]
    wa_accs = [r["wa_acc"] for r in results]
    n_iw = sum(r["iw_n"] for r in results)
    n_wa = sum(r["wa_n"] for r in results)
    sum_iw_correct = sum(r["iw_correct"] for r in results)
    sum_wa_correct = sum(r["wa_correct"] for r in results)

    print(f"  Mean I->W accuracy across runs:      {sum(iw_accs)/len(iw_accs)*100:.1f}%")
    print(f"  Mean W->A accuracy across runs:      {sum(wa_accs)/len(wa_accs)*100:.1f}%")
    print(f"  Pooled I->W: {sum_iw_correct}/{n_iw} = {sum_iw_correct/n_iw*100:.1f}%  (p={_binom_p_above(sum_iw_correct, n_iw):.4f})")
    print(f"  Pooled W->A: {sum_wa_correct}/{n_wa} = {sum_wa_correct/n_wa*100:.1f}%  (p={_binom_p_above(sum_wa_correct, n_wa):.4f})")
    print(f"  Best W->A:                          {max(wa_accs)*100:.1f}%")
    print(f"  Best I->W:                          {max(iw_accs)*100:.1f}%")

    # Per-direction analysis (W->A) across all runs that have per_direction data
    print()
    print("PER-DIRECTION W->A ACCURACY (across runs with valid per-word data):")
    for word in ["north", "east", "south", "west"]:
        accs = [r["wa_per"].get(word) for r in results
                if r["wa_per"].get(word) is not None]
        if accs:
            mean = sum(accs) / len(accs)
            print(f"  {word:<10} mean={mean*100:.0f}%  range=[{min(accs)*100:.0f}%, {max(accs)*100:.0f}%]  n_runs={len(accs)}")

    # List runs that beat 28.5% W->A
    print()
    above_baseline = [r for r in results if r["wa_acc"] > 0.285]
    if above_baseline:
        print(f"RUNS BEATING 28.5% W->A BASELINE: {len(above_baseline)}")
        for r in above_baseline:
            print(f"  - {r['label']:<60} W->A {r['wa_acc']*100:.1f}% (p={r['wa_p']:.3f})")
    else:
        print("NO RUNS HAVE BEATEN THE 28.5% W->A BASELINE.")


if __name__ == "__main__":
    main()
