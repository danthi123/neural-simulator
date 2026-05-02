"""Diagnostic post-hoc analyzer for text_eval result JSONs.

Given a result file written by `text_eval_embodied.py` (or compatible),
prints a structured breakdown:

- Headline: I->W and W->A accuracy with chance baseline + significance estimate
- Per-direction accuracy: which words/images are easiest/hardest
- Confusion-matrix asymmetry: are errors systematic (e.g., everything
  predicted as 'east') or distributed?
- Training-phase trajectory: did correct-moves climb, plateau, or drop?
- Comparison: against any prior result file passed via --baseline
- Verdict: which decision-tree branch the result falls in

Usage:
  python -m research.runners.text_eval_analyze <result.json> [--baseline <prior.json>]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CHANCE_4 = 0.25
DIRECTIONS = ["north", "east", "south", "west"]


def _binom_p_value_above_chance(correct: int, n: int, p_chance: float = CHANCE_4) -> float:
    """One-sided binomial p-value: P(X >= correct | X ~ Binomial(n, p_chance)).
    Uses normal approximation when n is large enough."""
    if n == 0:
        return 1.0
    mu = n * p_chance
    sigma = math.sqrt(n * p_chance * (1 - p_chance))
    if sigma < 1e-9:
        return 1.0 if correct < mu else 0.0
    # Continuity correction: P(X >= k) ~= P(Z >= (k - 0.5 - mu) / sigma)
    z = (correct - 0.5 - mu) / sigma
    # Survival function via erfc
    return 0.5 * math.erfc(z / math.sqrt(2))


def analyze(result_path: Path, baseline_path: Path | None = None) -> dict:
    d = json.loads(result_path.read_text())

    iw = d.get("image_to_word_eval", {})
    wa = d.get("word_to_action_eval", {})
    train = d.get("training_stats", []) or []

    iw_correct = iw.get("correct", 0)
    iw_n = iw.get("n_trials", 0)
    iw_acc = iw.get("accuracy", 0.0)
    wa_correct = wa.get("correct", 0)
    wa_n = wa.get("n_trials", 0)
    wa_acc = wa.get("accuracy", 0.0)

    # Significance vs chance
    iw_p = _binom_p_value_above_chance(iw_correct, iw_n)
    wa_p = _binom_p_value_above_chance(wa_correct, wa_n)

    # Per-direction breakdown
    iw_per = {}
    for word in DIRECTIONS:
        row = iw.get("confusion_matrix", {}).get(word, {})
        total = sum(row.values()) if row else 0
        correct = row.get(word, 0) if row else 0
        iw_per[word] = {
            "correct": correct, "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
        }
    wa_per = {}
    # word_to_action confusion is keyed by word, value is action-letter map
    word_to_action_letter = {"north": "N", "east": "E", "south": "S", "west": "W"}
    for word in DIRECTIONS:
        row = wa.get("confusion_matrix", {}).get(word, {})
        total = sum(row.values()) if row else 0
        correct = row.get(word_to_action_letter[word], 0) if row else 0
        wa_per[word] = {
            "correct": correct, "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    # Most-predicted bucket (helps detect "always predicts north" failure modes)
    iw_pred_dist = {w: 0 for w in DIRECTIONS}
    for tgt_row in iw.get("confusion_matrix", {}).values():
        for pred, cnt in tgt_row.items():
            if pred in iw_pred_dist:
                iw_pred_dist[pred] += cnt
    wa_pred_dist = {a: 0 for a in ["N", "E", "S", "W"]}
    for tgt_row in wa.get("confusion_matrix", {}).values():
        for pred, cnt in tgt_row.items():
            if pred in wa_pred_dist:
                wa_pred_dist[pred] += cnt

    # Training trajectory
    if train and isinstance(train, list):
        first_ep = train[0]
        train_summary = {
            "n_episodes": first_ep.get("n_episodes"),
            "n_total_steps": first_ep.get("n_total_steps"),
            "n_correct_moves": first_ep.get("n_correct_moves"),
            "correct_move_rate": first_ep.get("correct_move_rate"),
            "elapsed_seconds": first_ep.get("elapsed_seconds"),
        }
    else:
        train_summary = None

    # Decision-tree verdict (matches user's stated thresholds)
    max_acc = max(iw_acc, wa_acc)
    if max_acc >= 0.35:
        verdict = "WIN: >=35% on at least one task -- candidate for flagship"
    elif max_acc >= 0.30:
        verdict = "MATCH BASELINE: 30-35% -- fix worked, no improvement; explore architectural changes"
    elif max_acc >= 0.20:
        verdict = "PARTIAL REGRESSION: 20-30% -- reset_steps fix didn't fully restore; also revert T1.1 (stim_steps)"
    else:
        verdict = "DEEP REGRESSION: <20% -- bisect more carefully"

    out = {
        "result_file": str(result_path),
        "image_to_word": {
            "correct": iw_correct, "n": iw_n, "accuracy": iw_acc,
            "p_value_vs_chance": iw_p,
            "per_word": iw_per,
            "predicted_dist": iw_pred_dist,
        },
        "word_to_action": {
            "correct": wa_correct, "n": wa_n, "accuracy": wa_acc,
            "p_value_vs_chance": wa_p,
            "per_word": wa_per,
            "predicted_dist": wa_pred_dist,
        },
        "training": train_summary,
        "verdict": verdict,
    }

    if baseline_path is not None and baseline_path.exists():
        b = json.loads(baseline_path.read_text())
        b_iw = b.get("image_to_word_eval", {})
        b_wa = b.get("word_to_action_eval", {})
        out["baseline_comparison"] = {
            "baseline_file": str(baseline_path),
            "image_to_word_delta": iw_acc - b_iw.get("accuracy", 0.0),
            "word_to_action_delta": wa_acc - b_wa.get("accuracy", 0.0),
            "baseline_iw": b_iw.get("accuracy", 0.0),
            "baseline_wa": b_wa.get("accuracy", 0.0),
        }

    return out


def _print_report(report: dict) -> None:
    print("=" * 70)
    print(f"RESULT: {report['result_file']}")
    print("=" * 70)

    iw = report["image_to_word"]
    wa = report["word_to_action"]
    print(f"\nImage -> Word: {iw['correct']}/{iw['n']} = {iw['accuracy']:.1%} "
          f"(chance 25%, p={iw['p_value_vs_chance']:.4f})")
    print("  Per-word accuracy:")
    for word in DIRECTIONS:
        p = iw["per_word"][word]
        print(f"    {word:>6}: {p['correct']}/{p['total']} = {p['accuracy']:.1%}")
    print(f"  Predicted distribution: {iw['predicted_dist']}")

    print(f"\nWord -> Action: {wa['correct']}/{wa['n']} = {wa['accuracy']:.1%} "
          f"(chance 25%, p={wa['p_value_vs_chance']:.4f})")
    print("  Per-word accuracy:")
    for word in DIRECTIONS:
        p = wa["per_word"][word]
        print(f"    {word:>6}: {p['correct']}/{p['total']} = {p['accuracy']:.1%}")
    print(f"  Predicted distribution: {wa['predicted_dist']}")

    if report.get("training"):
        t = report["training"]
        if t["correct_move_rate"] is not None:
            print(f"\nTraining: {t['n_episodes']} ep x ?? steps, "
                  f"{t['n_correct_moves']}/{t['n_total_steps']} = "
                  f"{t['correct_move_rate']:.1%} correct moves "
                  f"({t['elapsed_seconds']:.0f}s)")

    if "baseline_comparison" in report:
        b = report["baseline_comparison"]
        print(f"\nvs baseline ({b['baseline_file']}):")
        print(f"  I->W: {b['baseline_iw']:.1%} -> {iw['accuracy']:.1%} "
              f"(delta {b['image_to_word_delta']:+.1%})")
        print(f"  W->A: {b['baseline_wa']:.1%} -> {wa['accuracy']:.1%} "
              f"(delta {b['word_to_action_delta']:+.1%})")

    print(f"\nVERDICT: {report['verdict']}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=str, help="path to text_eval result JSON")
    ap.add_argument("--baseline", type=str, default=None,
                    help="path to baseline result JSON for comparison")
    ap.add_argument("--json", action="store_true",
                    help="print JSON instead of formatted report")
    args = ap.parse_args()

    rp = Path(args.result)
    bp = Path(args.baseline) if args.baseline else None

    if not rp.exists():
        ap.error(f"result file not found: {rp}")

    report = analyze(rp, bp)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
