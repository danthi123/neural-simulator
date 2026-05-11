"""Aggregate P4.1 positional binding results across seeds.

Reads research/findings/raw/g11_bg/p41_seed*.json and reports
multi-seed PASS rate. Mirrors aggregate_two_concept_seeds.py.

Usage:
    python -m research.runners.aggregate_positional_seeds \\
        [--seeds 42,43,44] \\
        [--out research/findings/2026-05-11-P41-positional-multiseed.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(seed: int, root: Path):
    fp = root / f"p41_seed{seed}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def aggregate(seeds, root):
    rows = []
    for s in seeds:
        r = load_result(s, root)
        if r is None:
            print(f"  seed {s}: result file missing", file=sys.stderr)
            continue
        # Extract per-pair cosines
        pair_dict = {(p["a"], p["b"]): p["cosine"]
                     for p in r["pair_cosines"]}
        row = {
            "seed": s,
            "build_seconds": r["build_seconds"],
            "n_neurons": r["n_neurons"],
            "n_synapses": r["n_synapses"],
            "apple_pos_diff": pair_dict.get(
                ("apple_pos0", "apple_pos2"), 1.0),
            "alice_pos_diff": pair_dict.get(
                ("alice_pos0", "alice_pos2"), 1.0),
            "word_diff_pos0": pair_dict.get(
                ("apple_pos0", "alice_pos0"), 1.0),
            "word_diff_pos2": pair_dict.get(
                ("apple_pos2", "alice_pos2"), 1.0),
            "cross": pair_dict.get(
                ("apple_pos0", "alice_pos2"), 1.0),
            "pass_position": r["pass_position"],
            "pass_word": r["pass_word"],
            "overall_passed": r["overall_passed"],
        }
        rows.append(row)
    return rows


def render_markdown(rows):
    if not rows:
        return "# No results found\n"
    n_seeds = len(rows)
    n_pass = sum(1 for r in rows if r["overall_passed"])
    avg_apple_pos = sum(r["apple_pos_diff"] for r in rows) / n_seeds
    avg_alice_pos = sum(r["alice_pos_diff"] for r in rows) / n_seeds
    avg_word_pos0 = sum(r["word_diff_pos0"] for r in rows) / n_seeds
    avg_word_pos2 = sum(r["word_diff_pos2"] for r in rows) / n_seeds
    avg_cross = sum(r["cross"] for r in rows) / n_seeds

    md = []
    md.append("# P4.1 positional binding multi-seed result\n\n")
    md.append(f"**Date:** 2026-05-11\n")
    md.append(f"**Phase:** P4.1 (item-in-context binding) of realigned plan v3\n")
    md.append(f"**Catalog:** D.01 + D.02 + D.11\n")
    md.append(f"**Seeds:** {[r['seed'] for r in rows]}\n")
    md.append(f"**Verdict:** {n_pass}/{n_seeds} PASS\n\n")

    md.append("## Per-seed results\n\n")
    md.append("| Seed | apple-pos | alice-pos | word@pos0 | word@pos2 | cross | Overall |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(
            f"| {r['seed']} | "
            f"{r['apple_pos_diff']:.3f} | {r['alice_pos_diff']:.3f} | "
            f"{r['word_diff_pos0']:.3f} | {r['word_diff_pos2']:.3f} | "
            f"{r['cross']:.3f} | "
            f"{'PASS' if r['overall_passed'] else 'FAIL'} |\n"
        )
    md.append("\nAll cosines should be < 0.4. PASS means architecture "
              "distinguishes the (word, position) tuples.\n\n")

    md.append("## Multi-seed averages\n\n")
    md.append(f"- Same word, different position: "
              f"apple={avg_apple_pos:.3f}, alice={avg_alice_pos:.3f}\n")
    md.append(f"- Different word, same position: "
              f"@pos0={avg_word_pos0:.3f}, @pos2={avg_word_pos2:.3f}\n")
    md.append(f"- Different word, different position (cross): "
              f"{avg_cross:.3f}\n\n")

    md.append("## Interpretation\n\n")
    if n_pass == n_seeds:
        md.append(
            "**P4.1 substrate confirmed at multi-seed.** The "
            "architecture cleanly distinguishes (word, position) "
            "tuples — same word at different positions get distinct "
            "CA3 ensembles, and different words at the same position "
            "also get distinct ensembles. Word-order-dependent "
            "meaning is mechanistically supported.\n\n"
        )
        md.append(
            "Downstream impact: P5 ventral semantic stream + P6 "
            "Broca's can now learn to distinguish 'alice ate apple' "
            "from 'apple ate alice' via their distinct (word, "
            "position) CA3 ensemble sequences.\n"
        )
    elif n_pass >= int(n_seeds * 0.5):
        md.append(
            f"**Partial pass ({n_pass}/{n_seeds}).** Some seeds "
            "produce indistinct (word, position) ensembles. Possible "
            "causes: insufficient DG separation for the combined "
            "(word, position) input pattern; ec_context_to_dg weight "
            "may need tuning; or position 2 / position 0 ec_context "
            "patterns overlap too much. Investigation needed.\n"
        )
    else:
        md.append(
            f"**Mostly FAIL ({n_pass}/{n_seeds}).** The architecture "
            "doesn't reliably produce distinct (word, position) "
            "ensembles. Likely needs DG capacity scaling or "
            "ec_context redesign.\n"
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
              f"(apple-pos={r['apple_pos_diff']:.3f}, "
              f"word@pos0={r['word_diff_pos0']:.3f})")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n[OUT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
