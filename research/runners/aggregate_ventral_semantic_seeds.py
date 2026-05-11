"""Aggregate P5 ventral semantic stream results across seeds.

Reads research/findings/raw/g11_bg/p5*seed*.json and reports
multi-seed PASS rate for comprehension + naming + overall.
Supports any iteration prefix (p5_seed, p5_iterA_seed, p5_iterB_seed,
p5_iterC_seed) via the --prefix flag.

Mirrors aggregate_positional_seeds.py.

Usage:
    python -m research.runners.aggregate_ventral_semantic_seeds \\
        --prefix p5_iterB_seed --seeds 42,43,44 \\
        --out research/findings/2026-05-11-P5-iterB-multiseed.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(seed: int, prefix: str, root: Path):
    fp = root / f"{prefix}{seed}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def aggregate(seeds, prefix, root):
    rows = []
    for s in seeds:
        r = load_result(s, prefix, root)
        if r is None:
            print(f"  seed {s}: {prefix}{s}.json missing", file=sys.stderr)
            continue
        row = {
            "seed": s,
            "build_seconds": r.get("build_seconds", 0.0),
            "total_seconds": r.get("total_seconds", 0.0),
            "n_neurons": r.get("n_neurons", 0),
            "n_synapses": r.get("n_synapses", 0),
            "n_train_events": r.get("n_train_events", 0),
            "n_replay_cycles": r.get("n_replay_cycles", 0),
            "apple_tag_size": r.get("apple_tag_size", 0),
            "river_tag_size": r.get("river_tag_size", 0),
            "apple_self": r["comprehension"]["apple_self_cosine"],
            "apple_river": r["comprehension"]["apple_river_cosine"],
            "pass_comprehension": r["comprehension"]["passed"],
            "baseline_spikes": r["naming"]["baseline_lang_out_spikes"],
            "causal_spikes": r["naming"]["causal_lang_out_spikes"],
            "naming_ratio": r["naming"]["ratio"],
            "pass_naming": r["naming"]["passed"],
            "overall_passed": r["overall_passed"],
        }
        rows.append(row)
    return rows


def render_markdown(rows, prefix):
    if not rows:
        return "# No results found\n"
    n_seeds = len(rows)
    n_pass = sum(1 for r in rows if r["overall_passed"])
    n_pass_comp = sum(1 for r in rows if r["pass_comprehension"])
    n_pass_name = sum(1 for r in rows if r["pass_naming"])
    avg_apple_self = sum(r["apple_self"] for r in rows) / n_seeds
    avg_apple_river = sum(r["apple_river"] for r in rows) / n_seeds
    avg_ratio = sum(r["naming_ratio"] for r in rows) / n_seeds
    avg_total = sum(r["total_seconds"] for r in rows) / n_seeds

    md = []
    md.append(f"# P5 ventral semantic stream multi-seed result ({prefix})\n\n")
    md.append("**Date:** 2026-05-11\n")
    md.append("**Phase:** P5 of realigned plan v3\n")
    md.append("**Catalog:** G.11 (dual-stream, Hickok & Poeppel) + "
              "G.13 (Wernicke's area)\n")
    md.append(f"**Seeds:** {[r['seed'] for r in rows]}\n")
    md.append(f"**Verdict:** {n_pass}/{n_seeds} OVERALL PASS "
              f"({n_pass_comp}/{n_seeds} comprehension, "
              f"{n_pass_name}/{n_seeds} naming)\n\n")

    md.append("## Per-seed results\n\n")
    md.append("| Seed | apple_self | apple_river | Comp | "
              "Naming ratio | Naming | Overall | Wall |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(
            f"| {r['seed']} | "
            f"{r['apple_self']:.3f} | {r['apple_river']:.3f} | "
            f"{'PASS' if r['pass_comprehension'] else 'FAIL'} | "
            f"{r['naming_ratio']:.2f}x | "
            f"{'PASS' if r['pass_naming'] else 'FAIL'} | "
            f"{'PASS' if r['overall_passed'] else 'FAIL'} | "
            f"{r['total_seconds']:.0f}s |\n"
        )
    md.append("\nTargets: apple_self > 0.5 AND apple_river < 0.4 (Comp); "
              "naming ratio > 1.3x (Naming).\n\n")

    md.append("## Multi-seed averages\n\n")
    md.append(f"- Comprehension cosines: same-concept "
              f"{avg_apple_self:.3f}, cross-concept "
              f"{avg_apple_river:.3f}\n")
    md.append(f"- Naming ratio (causal/baseline): {avg_ratio:.2f}x\n")
    md.append(f"- Mean wall clock: {avg_total:.0f} sec/seed\n\n")

    md.append("## Interpretation\n\n")
    if n_pass == n_seeds:
        md.append(
            f"**P5 ventral semantic stream CONFIRMED at multi-seed.** "
            f"All {n_seeds} seeds pass both comprehension and naming "
            f"criteria. The architecture cleanly maps lang_input -> "
            f"semantic_cortex (comprehension) and engram tag -> "
            f"lang_output (naming) per catalog G.11/G.13.\n\n"
        )
        md.append(
            "Downstream impact: P6 Broca's compositional syntax can "
            "now build on top of working word<->meaning translation.\n"
        )
    elif n_pass >= int(n_seeds * 0.5):
        md.append(
            f"**Partial pass ({n_pass}/{n_seeds}).** Some seeds "
            "produce working comprehension/naming, others don't. "
            "Likely needs higher training events or further tuning.\n"
        )
    else:
        same_gt_cross = sum(1 for r in rows
                            if r["apple_self"] > r["apple_river"])
        md.append(
            f"**FAIL ({n_pass}/{n_seeds}).** Architecture doesn't "
            f"reliably produce word<->meaning binding.\n\n"
        )
        if same_gt_cross >= n_seeds * 0.66:
            md.append(
                f"NOTE: same-concept > cross-concept in "
                f"{same_gt_cross}/{n_seeds} seeds — methodology "
                "picks up some signal, but magnitude is below "
                "absolute threshold (>0.5).\n"
            )
        else:
            md.append(
                "Same-concept signal not reliably above cross-concept "
                "— deeper architectural issue.\n"
            )

    return "".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", type=str, default="p5_seed",
                    help="File prefix (p5_seed, p5_iterA_seed, "
                         "p5_iterB_seed, p5_iterC_seed)")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument(
        "--raw-root", type=str,
        default="research/findings/raw/g11_bg",
    )
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    root = Path(args.raw_root)
    rows = aggregate(seeds, args.prefix, root)
    md = render_markdown(rows, args.prefix)

    # ASCII-safe stdout summary
    n_pass = sum(1 for r in rows if r["overall_passed"])
    print(f"Prefix: {args.prefix}")
    print(f"Seeds: {[r['seed'] for r in rows]}")
    print(f"PASS: {n_pass}/{len(rows)}")
    for r in rows:
        print(f"  seed={r['seed']}: "
              f"{'PASS' if r['overall_passed'] else 'FAIL'} "
              f"(self={r['apple_self']:.3f}, "
              f"cross={r['apple_river']:.3f}, "
              f"ratio={r['naming_ratio']:.2f}x)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n[OUT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
