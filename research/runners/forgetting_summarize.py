"""Phase 1.4 forgetting eval -- 6-seed result summarizer.

Reads forgetting_seed{N}.json files produced by
continual_forgetting_eval.py and produces a markdown summary with:
- Per-seed Phase A baseline, Phase B retention, retention_pct
- Pass/fail per seed (>= 80%, 50-80%, < 50%)
- 6-seed mean + std + n_pass

Output: stdout markdown table, plus optional --out-md file write.

Usage:
    python -m research.runners.forgetting_summarize \\
        --pattern "research/findings/raw/g11_bg/forgetting_seed*.json"

    # Default seeds 42 43 44 100 101 102:
    python -m research.runners.forgetting_summarize
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]


def load_seed(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def extract_metrics(data: dict) -> dict:
    """Pull retention metrics from a forgetting JSON."""
    out = {
        "phase_a_acc": None,
        "phase_b_acc": None,
        "retention_pct": None,
        "synonym_b_acc": None,
        "sanity_failed": False,
    }
    if data.get("sanity_check_failed"):
        out["sanity_failed"] = True
    metrics = data.get("metrics", {})
    if metrics:
        out["phase_a_acc"] = metrics.get("primary_a_acc")
        out["phase_b_acc"] = metrics.get("primary_b_acc")
        out["retention_pct"] = metrics.get("retention_pct")
        out["synonym_b_acc"] = metrics.get("synonym_b_acc")
        return out
    # Fallback: pull from checkpoints if metrics missing
    cps = data.get("checkpoints", [])
    for cp in cps:
        if cp.get("name") == "after_phase_a":
            wa = cp.get("primary_wa", {})
            out["phase_a_acc"] = wa.get("accuracy")
        if cp.get("name") == "after_phase_b":
            wa_p = cp.get("primary_wa", {})
            wa_s = cp.get("synonym_wa", {})
            out["phase_b_acc"] = wa_p.get("accuracy")
            out["synonym_b_acc"] = wa_s.get("accuracy")
    if out["phase_a_acc"] and out["phase_a_acc"] > 0 and out["phase_b_acc"]:
        out["retention_pct"] = (
            out["phase_b_acc"] / out["phase_a_acc"]
        ) * 100
    return out


def grade(retention_pct: Optional[float]) -> str:
    if retention_pct is None:
        return "?"
    if retention_pct >= 80:
        return "[OK] PASS (>=80%)"
    if retention_pct >= 50:
        return "[~] MODERATE (50-80%)"
    return "[X] FAIL (<50%)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--dir", type=str,
                    default="research/findings/raw/g11_bg",
                    help="Directory containing forgetting_seed{N}.json")
    ap.add_argument("--filename-template", type=str,
                    default="forgetting_seed{seed}.json")
    ap.add_argument("--out-md", type=str, default=None,
                    help="Write markdown summary to this path")
    args = ap.parse_args()

    base = Path(args.dir)
    rows = []
    for seed in args.seeds:
        path = base / args.filename_template.format(seed=seed)
        data = load_seed(path)
        if data is None:
            rows.append({"seed": seed, "missing": True})
            continue
        m = extract_metrics(data)
        m["seed"] = seed
        m["missing"] = False
        rows.append(m)

    # Compute aggregate
    valid = [r for r in rows
             if not r.get("missing")
             and not r.get("sanity_failed")
             and r.get("retention_pct") is not None]

    md = []
    md.append("# Phase 1.4 catastrophic forgetting -- 6-seed summary\n")
    md.append(f"Source: {args.dir}/{args.filename_template}\n\n")
    md.append("| Seed | Phase A acc | Phase B acc | Retention | "
              "Synonym new | Status |\n")
    md.append("|------|-------------|-------------|-----------|"
              "-------------|--------|\n")
    for r in rows:
        if r.get("missing"):
            md.append(f"| {r['seed']} | -- | -- | -- | -- | "
                      f"MISSING |\n")
            continue
        if r.get("sanity_failed"):
            paa = r.get("phase_a_acc")
            paa_s = f"{paa:.1%}" if paa is not None else "?"
            md.append(f"| {r['seed']} | {paa_s} | -- | -- | -- | "
                      f"SANITY FAIL |\n")
            continue
        paa = r.get("phase_a_acc", 0) or 0
        pba = r.get("phase_b_acc", 0) or 0
        ret = r.get("retention_pct")
        syn = r.get("synonym_b_acc", 0) or 0
        md.append(
            f"| {r['seed']} | {paa:.1%} | {pba:.1%} | "
            f"{ret:.0f}% | {syn:.1%} | {grade(ret)} |\n"
        )

    if valid:
        retentions = [r["retention_pct"] for r in valid]
        mean_ret = statistics.mean(retentions)
        std_ret = statistics.stdev(retentions) if len(retentions) > 1 else 0
        n_pass_80 = sum(1 for r in retentions if r >= 80)
        n_pass_50 = sum(1 for r in retentions if r >= 50)
        n_total = len(args.seeds)
        n_valid = len(valid)
        # 4/6 threshold scales for partial results -- use proportion:
        # need 2/3 seeds at threshold to declare branch.
        pass_threshold_count = max(1, int(round(n_total * 2 / 3)))
        md.append("\n## Aggregate\n\n")
        md.append(f"- Valid seeds: **{n_valid}/{n_total}**\n")
        md.append(f"- Mean retention: **{mean_ret:.0f}%** "
                  f"(+/- {std_ret:.0f}%)\n")
        md.append(f"- Seeds passing 80% threshold: "
                  f"**{n_pass_80}/{n_total}**\n")
        md.append(f"- Seeds passing 50% threshold: "
                  f"**{n_pass_50}/{n_total}**\n")
        md.append(f"- Pass threshold for branch verdict: "
                  f"**{pass_threshold_count}/{n_total}**\n")
        md.append("\n## Verdict\n\n")
        if n_valid < n_total:
            md.append(f"[~] **PRELIMINARY** ({n_valid}/{n_total} seeds "
                      f"complete). Final verdict pending remaining seeds.\n\n")
            # Show preliminary direction based on what we have
            if n_pass_80 == n_valid:
                md.append("Direction: trending BRANCH A (all completed "
                          "seeds at >= 80% retention).\n")
            elif n_pass_50 == n_valid:
                md.append("Direction: trending BRANCH B (all completed "
                          "seeds at >= 50% retention).\n")
            else:
                md.append("Direction: mixed; some seeds at <50% retention.\n")
        elif n_pass_80 >= pass_threshold_count:
            md.append(f"[OK] **BRANCH A**: catastrophic forgetting NOT "
                      f"occurring under standard test. Biology-grounded "
                      f"continual learning preserves old knowledge "
                      f"(>= {pass_threshold_count}/{n_total} retention "
                      f">= 80%).\n")
            md.append("\nNext: proceed to Phase 1.2 Tier 2.3 or Phase 2.1.\n")
        elif n_pass_50 >= pass_threshold_count:
            md.append(f"[~] **BRANCH B**: moderate forgetting. Some "
                      f"retention loss but not catastrophic ("
                      f">= {pass_threshold_count}/{n_total} at >= 50%, "
                      f"< {pass_threshold_count}/{n_total} at >= 80%). "
                      f"Phase 1.4b mitigations (heterosynaptic LTD, "
                      f"replay-based interleaving, sleep consolidation) "
                      f"recommended before Phase 2.\n")
            md.append("\nNext: design + implement Phase 1.4b. "
                      "See docs/plans/2026-05-06-Phase-1.4-decision-tree.md.\n")
        else:
            md.append("[X] **BRANCH C**: catastrophic forgetting confirmed. "
                      "Biology-grounded mechanisms alone insufficient at "
                      "this scale.\n")
            md.append("\nNext: advance Phase 1.3 (hippocampus consolidation) "
                      "to head of queue. If Phase 1.3 also fails, surface "
                      "to user (may need EWC or other non-strict-bio "
                      "mitigation).\n")
    else:
        md.append("\n## Aggregate\n\n")
        md.append("No valid seeds (all missing or sanity-failed).\n")

    output = "".join(md)
    print(output)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(output)
        print(f"\nWrote: {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
