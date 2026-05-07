"""Aggregate multi-seed chat_demo / chat_synonym_demo results.

Reads per-seed stats JSONs produced by chat_demo / chat_synonym_demo /
chat_continual_demo, computes mean/std accuracy and surfaces per-direction
breakdown, then writes an aggregate JSON + a markdown findings doc.

Usage:
    # Run a single demo across N seeds and aggregate:
    for seed in 42 43 44 100 101 102; do
        python -m research.runners.chat_demo \\
            --seed $seed --train-events 200 \\
            --out-stats research/findings/raw/g11_bg/chat_demo_seed$seed.json
    done

    python -m research.runners.chat_demo_aggregate \\
        research/findings/raw/g11_bg/chat_demo_seed*.json \\
        --out research/findings/raw/g11_bg/chat_demo_aggregate.json \\
        --findings-md research/findings/2026-05-07-chat-demo-multi-seed.md \\
        --label "Tier 1 chat demo"

The aggregate works for chat_demo (4-word vocab), chat_synonym_demo (8-word
synonym vocab — also reports primary-vs-synonym split if the seed JSONs
have those fields), and chat_continual_demo (Phase 1.4 BRANCH A -- reports
retention ratio).
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
    """Load per-seed chat-demo JSONs and compute summary stats.

    Detects which demo type produced each file by looking for distinctive
    fields (synonym demo has primary_accuracy / synonym_accuracy; continual
    demo has retention_ratio).
    """
    per_seed = []
    for p in sorted(result_paths):
        data = json.load(open(p))
        entry = {"path": str(p), "seed": data.get("seed")}

        if "primary_accuracy" in data and "synonym_accuracy" in data:
            entry["demo_type"] = "synonym"
            entry["accuracy"] = data.get("accuracy", 0.0)
            entry["primary_accuracy"] = data["primary_accuracy"]
            entry["synonym_accuracy"] = data["synonym_accuracy"]
            entry["per_action_correct"] = data.get("per_action_correct", {})
            entry["per_action_total"] = data.get("per_action_total", {})
        elif ("retention_ratio" in data or "retention" in data
              or "primary_post_b" in data or "primary_b_acc" in data):
            # chat_continual_demo writes either {primary_post_a, primary_post_b,
            # retention_ratio, synonym_learning} OR {primary_a_acc, primary_b_acc,
            # retention, synonym_acc} depending on its version. Accept both.
            entry["demo_type"] = "continual"
            entry["primary_post_a"] = (
                data.get("primary_post_a") or data.get("primary_a_acc") or 0.0
            )
            entry["primary_post_b"] = (
                data.get("primary_post_b") or data.get("primary_b_acc") or 0.0
            )
            entry["retention_ratio"] = (
                data.get("retention_ratio") or data.get("retention") or 0.0
            )
            entry["synonym_learning"] = (
                data.get("synonym_learning") or data.get("synonym_acc") or 0.0
            )
            entry["accuracy"] = entry["primary_post_b"]
        else:
            entry["demo_type"] = "tier1"
            entry["accuracy"] = data.get("accuracy", 0.0)
            entry["correct"] = data.get("correct", 0)
            entry["total"] = data.get("total", 0)

        per_seed.append(entry)

    if not per_seed:
        raise ValueError("No result files found")

    demo_types = {s["demo_type"] for s in per_seed}
    if len(demo_types) > 1:
        # Mixed types — aggregate only the common "accuracy" field.
        print(f"[WARN] Mixed demo types in input: {demo_types}; "
              f"aggregating common 'accuracy' field only.")

    accs = [s["accuracy"] for s in per_seed]
    summary = {
        "n_seeds": len(per_seed),
        "demo_types": sorted(demo_types),
        "seeds": [s["seed"] for s in per_seed],
        "accuracy_mean": _safe_mean(accs),
        "accuracy_std": _safe_std(accs),
        "accuracy_min": min(accs),
        "accuracy_max": max(accs),
        "per_seed": per_seed,
    }

    # Synonym-specific aggregation
    syn_seeds = [s for s in per_seed if s["demo_type"] == "synonym"]
    if syn_seeds:
        prim = [s["primary_accuracy"] for s in syn_seeds]
        syn = [s["synonym_accuracy"] for s in syn_seeds]
        summary["synonym_demo"] = {
            "n_seeds": len(syn_seeds),
            "primary_acc_mean": _safe_mean(prim),
            "primary_acc_std": _safe_std(prim),
            "synonym_acc_mean": _safe_mean(syn),
            "synonym_acc_std": _safe_std(syn),
        }

    # Continual-specific aggregation
    cont_seeds = [s for s in per_seed if s["demo_type"] == "continual"]
    if cont_seeds:
        ret = [s["retention_ratio"] for s in cont_seeds]
        summary["continual_demo"] = {
            "n_seeds": len(cont_seeds),
            "retention_mean": _safe_mean(ret),
            "retention_std": _safe_std(ret),
            "retention_min": min(ret),
            "retention_max": max(ret),
            "n_pass_above_80": sum(1 for r in ret if r >= 0.80),
        }

    return summary


def write_findings_md(summary: dict[str, Any], out_path: str, label: str):
    """Render aggregate as a findings markdown doc."""
    md = []
    md.append(f"# Multi-seed chat demo aggregate: {label}\n\n")
    md.append(f"**Demo types:** {', '.join(summary['demo_types'])}\n\n")
    md.append(f"**N seeds:** {summary['n_seeds']} "
              f"(seeds: {summary['seeds']})\n\n")
    md.append("---\n\n## Overall accuracy\n\n")
    md.append(f"- Mean: **{summary['accuracy_mean']:.1%}**\n")
    md.append(f"- Std:  {summary['accuracy_std']:.1%}\n")
    md.append(f"- Range: {summary['accuracy_min']:.1%} – "
              f"{summary['accuracy_max']:.1%}\n\n")

    if "synonym_demo" in summary:
        s = summary["synonym_demo"]
        md.append("## Synonym demo split\n\n")
        md.append(f"- Primary words (north/east/south/west): "
                  f"**{s['primary_acc_mean']:.1%}** ± {s['primary_acc_std']:.1%}\n")
        md.append(f"- Synonym words (up/right/down/left):    "
                  f"**{s['synonym_acc_mean']:.1%}** ± {s['synonym_acc_std']:.1%}\n\n")

    if "continual_demo" in summary:
        c = summary["continual_demo"]
        md.append("## Continual learning retention\n\n")
        md.append(f"- Mean retention: **{c['retention_mean']:.1%}** "
                  f"± {c['retention_std']:.1%}\n")
        md.append(f"- Range: {c['retention_min']:.1%} – {c['retention_max']:.1%}\n")
        md.append(f"- Seeds passing >= 80% retention: "
                  f"{c['n_pass_above_80']}/{c['n_seeds']}\n\n")

    md.append("---\n\n## Per-seed table\n\n")
    md.append("| Seed | Accuracy | Notes |\n|---|---|---|\n")
    for s in summary["per_seed"]:
        notes_parts = []
        if s["demo_type"] == "synonym":
            notes_parts.append(f"PRI={s['primary_accuracy']:.1%}")
            notes_parts.append(f"SYN={s['synonym_accuracy']:.1%}")
        if s["demo_type"] == "continual":
            notes_parts.append(f"retention={s['retention_ratio']:.1%}")
        notes = ", ".join(notes_parts) if notes_parts else "-"
        md.append(f"| {s['seed']} | {s['accuracy']:.1%} | {notes} |\n")
    md.append("\n")

    md.append("---\n\n*Generated by `research.runners.chat_demo_aggregate` "
              "from per-seed stats JSONs.*\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="Glob(s) of per-seed chat demo stats JSONs")
    ap.add_argument("--out", type=str, default=None,
                    help="Path to write aggregate JSON")
    ap.add_argument("--findings-md", type=str, default=None,
                    help="Path to write a findings-style markdown report")
    ap.add_argument("--label", type=str, default="chat demo",
                    help="Display label for the findings doc title")
    args = ap.parse_args()

    # Expand globs
    paths = []
    for p in args.paths:
        matched = sorted(glob.glob(p))
        if matched:
            paths.extend(matched)
        else:
            paths.append(p)  # let json.load fail with a clear error

    if not paths:
        print("[FAIL] No result paths matched.")
        return 2

    print(f"[AGGREGATE] {len(paths)} files:")
    for p in paths:
        print(f"  {p}")
    print()

    summary = aggregate(paths)

    print(f"\n[SUMMARY] N={summary['n_seeds']} seeds")
    print(f"  Demo types: {summary['demo_types']}")
    print(f"  Accuracy:  {summary['accuracy_mean']:.1%} +/- "
          f"{summary['accuracy_std']:.1%}  "
          f"(range {summary['accuracy_min']:.1%}-{summary['accuracy_max']:.1%})")
    if "synonym_demo" in summary:
        s = summary["synonym_demo"]
        print(f"  Primary:   {s['primary_acc_mean']:.1%} +/- "
              f"{s['primary_acc_std']:.1%}")
        print(f"  Synonym:   {s['synonym_acc_mean']:.1%} +/- "
              f"{s['synonym_acc_std']:.1%}")
    if "continual_demo" in summary:
        c = summary["continual_demo"]
        print(f"  Retention: {c['retention_mean']:.1%} +/- "
              f"{c['retention_std']:.1%}  "
              f"({c['n_pass_above_80']}/{c['n_seeds']} pass >= 80%)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
        print(f"\n[OUT] {args.out}")

    if args.findings_md:
        write_findings_md(summary, args.findings_md, args.label)
        print(f"[FINDINGS] {args.findings_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
