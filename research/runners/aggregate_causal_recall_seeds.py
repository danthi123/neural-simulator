"""Aggregate Liu 2012-style causal recall results across seeds.

Reads research/findings/raw/g11_bg/causal_seed*.json and reports
multi-seed PASS rate for word-driven recall + causal recall.

The Liu 2012 paper (Optogenetic stimulation of a hippocampal engram
activates fear memory recall): tested whether stimulating only the
tagged hippocampal engram cells produces the memory-encoded behavior
without the original sensory cue. PASS = causal stimulation produces
target-action motor activity above other-pool activity.

Our equivalent: train word->motor binding, tag CA3 ensemble,
stimulate ONLY the engram → measure if motor_target is preferentially
activated.

Usage:
    python -m research.runners.aggregate_causal_recall_seeds \\
        [--seeds 42,43,44] \\
        [--out research/findings/2026-05-11-causal-recall-multiseed.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(seed: int, root: Path):
    fp = root / f"causal_seed{seed}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def aggregate(seeds, root):
    rows = []
    for s in seeds:
        r = load_result(s, root)
        if r is None:
            print(f"  seed {s}: causal_seed{s}.json missing", file=sys.stderr)
            continue
        row = {
            "seed": s,
            "target_action": r["target_action"],
            "train_events": r["train_events"],
            "n_engram_neurons": r["n_engram_neurons"],
            "target_baseline": r["target_baseline"],
            "target_word": r["target_word"],
            "target_causal": r["target_causal"],
            "other_baseline": r["other_baseline"],
            "other_word": r["other_word"],
            "other_causal": r["other_causal"],
            "word_ratio": r["target_word"] / max(r["other_word"], 1e-6),
            "causal_ratio": r["causal_recall_ratio"],
            "pass_word": r["pass_word_driven"],
            "pass_causal": r["pass_causal_recall"],
            "overall_passed": r["overall_passed"],
            "total_seconds": r["total_seconds"],
        }
        rows.append(row)
    return rows


def render_markdown(rows):
    if not rows:
        return "# No results found\n"
    n_seeds = len(rows)
    n_pass = sum(1 for r in rows if r["overall_passed"])
    n_pass_w = sum(1 for r in rows if r["pass_word"])
    n_pass_c = sum(1 for r in rows if r["pass_causal"])
    avg_word_ratio = sum(r["word_ratio"] for r in rows) / n_seeds
    avg_causal_ratio = sum(r["causal_ratio"] for r in rows) / n_seeds
    avg_total = sum(r["total_seconds"] for r in rows) / n_seeds

    md = []
    md.append("# Liu 2012-style causal recall multi-seed\n\n")
    md.append("**Date:** 2026-05-11\n")
    md.append(
        "**Catalog:** D.14 (engram tagging, "
        "Tonegawa lab 2012-2015)\n"
    )
    md.append(
        "**Reference:** Liu et al. 2012 Nature — "
        "Optogenetic stimulation of a hippocampal engram activates "
        "fear memory recall.\n"
    )
    md.append(f"**Seeds:** {[r['seed'] for r in rows]}\n")
    md.append(f"**Verdict:** {n_pass}/{n_seeds} OVERALL PASS "
              f"({n_pass_w}/{n_seeds} word-driven, "
              f"{n_pass_c}/{n_seeds} causal)\n\n")

    md.append("## Per-seed results\n\n")
    md.append("| Seed | Tag size | Word ratio | Word | "
              "Causal ratio | Causal | Overall | Wall |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(
            f"| {r['seed']} | {r['n_engram_neurons']} | "
            f"{r['word_ratio']:.2f}x | "
            f"{'PASS' if r['pass_word'] else 'FAIL'} | "
            f"{r['causal_ratio']:.2f}x | "
            f"{'PASS' if r['pass_causal'] else 'FAIL'} | "
            f"{'PASS' if r['overall_passed'] else 'FAIL'} | "
            f"{r['total_seconds']:.0f}s |\n"
        )
    md.append("\nTargets: word-driven target/other > 1.3x; "
              "causal target/other > 1.5x.\n\n")

    md.append("## Multi-seed averages\n\n")
    md.append(f"- Word-driven target/other ratio: "
              f"{avg_word_ratio:.2f}x (training worked)\n")
    md.append(f"- Causal target/other ratio: "
              f"{avg_causal_ratio:.2f}x (engram drives motor)\n")
    md.append(f"- Mean wall clock: {avg_total:.0f} sec/seed\n\n")

    md.append("## Interpretation\n\n")
    if n_pass == n_seeds:
        md.append(
            f"**Liu 2012-style causal recall CONFIRMED at "
            f"multi-seed.** All {n_seeds} seeds show:\n\n"
            "1. Word-driven recall: lang_input drive produces "
            "preferentially activates motor_target (training "
            "worked end-to-end).\n"
            "2. Causal recall: stimulating ONLY the CA3 engram "
            "tag (no lang_input) activates motor_target above "
            "other pools.\n\n"
            "This is the project's behavioral confirmation of the "
            "Tonegawa-2012-style optogenetic memory recall result. "
            "The hippocampal engram is sufficient (not just "
            "necessary) for memory expression.\n"
        )
    elif n_pass_w == n_seeds and n_pass_c < n_seeds:
        md.append(
            f"**Word-driven recall confirmed ({n_pass_w}/{n_seeds}); "
            f"causal recall partial ({n_pass_c}/{n_seeds}).** "
            "Training works (lang_input properly drives motor); "
            "engram tag stimulation doesn't reliably propagate "
            "to motor_target across all seeds. May need stronger "
            "engram tag drive_pA, denser ca3->ca1 path, or "
            "longer training.\n"
        )
    elif n_pass_w < n_seeds:
        md.append(
            f"**Word-driven recall partial "
            f"({n_pass_w}/{n_seeds}).** Training itself isn't "
            "reliable across seeds — even with lang_input drive, "
            "motor_target doesn't preferentially activate. "
            "Likely needs more training events or stronger "
            "lang_input -> motor pathway.\n"
        )
    else:
        md.append(
            f"**Mostly FAIL ({n_pass}/{n_seeds}).** "
            "Neither word-driven nor causal recall reliable. "
            "Architectural issue.\n"
        )

    return "".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument(
        "--raw-root", type=str,
        default="research/findings/raw/g11_bg",
    )
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    root = Path(args.raw_root)
    rows = aggregate(seeds, root)
    md = render_markdown(rows)

    # ASCII-safe stdout summary
    n_pass = sum(1 for r in rows if r["overall_passed"])
    print(f"Seeds: {[r['seed'] for r in rows]}")
    print(f"PASS: {n_pass}/{len(rows)}")
    for r in rows:
        print(f"  seed={r['seed']}: "
              f"{'PASS' if r['overall_passed'] else 'FAIL'} "
              f"(word_ratio={r['word_ratio']:.2f}x, "
              f"causal_ratio={r['causal_ratio']:.2f}x)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n[OUT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
