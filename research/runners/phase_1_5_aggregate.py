"""Aggregate multi-seed Phase 1.5 unified eval suite results.

Reads per-seed JSONs produced by continual_eval_suite, computes
mean/std score per benchmark and aggregate across benchmarks, and
writes findings markdown.

Each per-seed JSON has structure:
    {
        "seed": int,
        "benchmarks": [
            {"name": str, "score": float, "pass": bool, "details": {...}},
            ...
        ],
        "aggregate": {"score": float, "all_pass": bool, ...}
    }

Usage:
    python -m research.runners.phase_1_5_aggregate \\
        research/findings/raw/g11_bg/g11_seed*_phase_1_5_unified_scaled_*.json \\
        --out research/findings/raw/g11_bg/phase_1_5_aggregate.json \\
        --findings-md research/findings/2026-05-XX-Phase-1.5-multi-seed.md \\
        --label "Phase 1.5 unified eval suite (scaled, 6 seeds)"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _safe_mean(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def _safe_std(xs: list[float]) -> float:
    return float(stdev(xs)) if len(xs) > 1 else 0.0


def aggregate(result_paths: list[str]) -> dict[str, Any]:
    """Aggregate per-seed Phase 1.5 JSONs into per-benchmark + overall stats."""
    per_seed = []
    for p in sorted(result_paths):
        if p.endswith(".cmd.json"):
            continue
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue
        per_seed.append({
            "path": str(p),
            "seed": data.get("seed"),
            "benchmarks": data.get("benchmarks", []),
            "aggregate": data.get("aggregate", {}),
        })

    if not per_seed:
        raise ValueError("No result files found (after filtering .cmd.json)")

    # Group by benchmark name
    per_benchmark: dict[str, list[dict]] = {}
    for s in per_seed:
        for b in s["benchmarks"]:
            name = b.get("name", "unknown")
            per_benchmark.setdefault(name, []).append({
                "seed": s["seed"],
                "score": b.get("score", 0.0),
                "pass": b.get("pass", False),
                "details": b.get("details", {}),
            })

    # Per-benchmark aggregation
    benchmark_summary = {}
    for name, runs in per_benchmark.items():
        # Skip benchmarks that are placeholders (Tier 2.2/2.3 pending)
        active_runs = [r for r in runs
                       if r["details"].get("status") not in
                          ("tier_2_2_pending", "tier_2_3_pending",
                           "not_yet_implemented")
                       and "error" not in r["details"]]
        if not active_runs:
            benchmark_summary[name] = {
                "n_seeds": 0, "skipped": True,
                "reason": runs[0]["details"].get("status", "no active runs")
                          if runs else "no runs",
            }
            continue

        scores = [r["score"] for r in active_runs]
        n_pass = sum(1 for r in active_runs if r["pass"])
        benchmark_summary[name] = {
            "n_seeds": len(active_runs),
            "score_mean": _safe_mean(scores),
            "score_std": _safe_std(scores),
            "score_min": min(scores),
            "score_max": max(scores),
            "n_pass": n_pass,
            "pass_rate": n_pass / len(active_runs),
            "per_seed": [{"seed": r["seed"], "score": r["score"],
                          "pass": r["pass"]} for r in active_runs],
        }

    # Overall aggregate (mean of mean across active benchmarks)
    active_means = [b["score_mean"] for b in benchmark_summary.values()
                    if not b.get("skipped", False)]
    overall_mean = _safe_mean(active_means)
    overall_std = _safe_std(active_means)
    n_pass_threshold = sum(1 for b in benchmark_summary.values()
                            if not b.get("skipped", False)
                            and b.get("pass_rate", 0.0) >= 0.5)
    n_active = sum(1 for b in benchmark_summary.values()
                    if not b.get("skipped", False))

    summary = {
        "n_seeds": len(per_seed),
        "seeds": [s["seed"] for s in per_seed],
        "benchmarks": benchmark_summary,
        "overall": {
            "aggregate_score_mean": overall_mean,
            "aggregate_score_std": overall_std,
            "n_active_benchmarks": n_active,
            "n_benchmarks_majority_pass": n_pass_threshold,
            # Master plan threshold: aggregate >= 0.7 = BIOLOGY-GROUNDED
            # CONTINUAL LEARNING VALIDATED
            "master_plan_pass": overall_mean >= 0.70,
        },
        "per_seed_aggregates": [
            {"seed": s["seed"], "agg_score": s["aggregate"].get("score", 0.0),
             "all_pass": s["aggregate"].get("all_pass", False)}
            for s in per_seed
        ],
    }
    return summary


def write_findings_md(summary: dict[str, Any], out_path: str, label: str):
    """Render aggregate as a findings markdown doc."""
    md = []
    md.append(f"# Multi-seed Phase 1.5 aggregate: {label}\n\n")
    md.append(f"**N seeds:** {summary['n_seeds']} "
              f"(seeds: {summary['seeds']})\n\n")
    md.append("---\n\n## Overall\n\n")
    o = summary["overall"]
    md.append(f"- Aggregate score (mean across active benchmarks): "
              f"**{o['aggregate_score_mean']:.2f}** "
              f"± {o['aggregate_score_std']:.2f}\n")
    md.append(f"- N active benchmarks: {o['n_active_benchmarks']}\n")
    md.append(f"- N benchmarks majority-passing (>=50% seeds): "
              f"{o['n_benchmarks_majority_pass']}\n")
    md.append(f"- **Master plan threshold (>=0.70):** "
              f"{'PASS' if o['master_plan_pass'] else 'FAIL'}\n\n")

    md.append("---\n\n## Per-benchmark results\n\n")
    md.append("| Benchmark | N | Score Mean ± Std | Range | Pass Rate |\n")
    md.append("|---|---|---|---|---|\n")
    for name, stats in sorted(summary["benchmarks"].items()):
        if stats.get("skipped"):
            md.append(f"| {name} | 0 | (skipped: {stats['reason']}) | - | - |\n")
        else:
            md.append(f"| {name} | {stats['n_seeds']} | "
                      f"**{stats['score_mean']:.2f}** ± "
                      f"{stats['score_std']:.2f} | "
                      f"{stats['score_min']:.2f}-{stats['score_max']:.2f} | "
                      f"{stats['n_pass']}/{stats['n_seeds']} "
                      f"({stats['pass_rate']:.0%}) |\n")
    md.append("\n")

    md.append("---\n\n## Per-seed aggregate\n\n")
    md.append("| Seed | Aggregate Score | All Pass |\n|---|---|---|\n")
    for ps in summary["per_seed_aggregates"]:
        md.append(f"| {ps['seed']} | {ps['agg_score']:.2f} | "
                  f"{'YES' if ps['all_pass'] else 'NO'} |\n")
    md.append("\n")

    md.append("---\n\n## Per-benchmark per-seed\n\n")
    for name, stats in sorted(summary["benchmarks"].items()):
        if stats.get("skipped"):
            continue
        md.append(f"### {name}\n\n")
        md.append("| Seed | Score | Pass |\n|---|---|---|\n")
        for r in stats["per_seed"]:
            md.append(f"| {r['seed']} | {r['score']:.2f} | "
                      f"{'YES' if r['pass'] else 'NO'} |\n")
        md.append("\n")

    md.append("---\n\n*Generated by `research.runners.phase_1_5_aggregate` "
              "from per-seed continual_eval_suite JSONs.*\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="Glob(s) of per-seed Phase 1.5 stats JSONs")
    ap.add_argument("--out", type=str, default=None,
                    help="Path to write aggregate JSON")
    ap.add_argument("--findings-md", type=str, default=None,
                    help="Path to write a findings-style markdown report")
    ap.add_argument("--label", type=str, default="Phase 1.5 multi-seed",
                    help="Display label for the findings doc title")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        matched = sorted(glob.glob(p))
        if matched:
            paths.extend(matched)
        else:
            paths.append(p)
    paths = [p for p in paths if not p.endswith(".cmd.json")]

    if not paths:
        print("[FAIL] No result paths matched.")
        return 2

    print(f"[AGGREGATE] {len(paths)} files")
    for p in paths:
        print(f"  {p}")
    print()

    summary = aggregate(paths)

    print(f"\n[SUMMARY] N={summary['n_seeds']} seeds")
    print(f"  Aggregate score: "
          f"{summary['overall']['aggregate_score_mean']:.2f} ± "
          f"{summary['overall']['aggregate_score_std']:.2f}")
    print(f"  Master plan threshold (>=0.70): "
          f"{'PASS' if summary['overall']['master_plan_pass'] else 'FAIL'}")
    for name, stats in sorted(summary["benchmarks"].items()):
        if stats.get("skipped"):
            print(f"  {name}: skipped ({stats['reason']})")
        else:
            print(f"  {name}: {stats['score_mean']:.2f} ± "
                  f"{stats['score_std']:.2f} "
                  f"({stats['n_pass']}/{stats['n_seeds']} pass)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2,
                                              default=str),
                                    encoding="utf-8")
        print(f"\n[OUT] {args.out}")

    if args.findings_md:
        write_findings_md(summary, args.findings_md, args.label)
        print(f"[FINDINGS] {args.findings_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
