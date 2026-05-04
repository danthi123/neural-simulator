"""Universal result aggregator - replaces N specific aggregator scripts
with a parameterized engine.

Tonight's investigation needed 4 separate aggregators:
  - swr_aggregate.py (4 SWR conditions across seeds)
  - swr_per_seed.py (per-seed cross-condition deltas)
  - permuted_label_check.py (aligned ratio per-condition)
  - biology_sweep_summary.py (biology sweep result)

All do similar things: glob `text_eval_*.json`, group by condition,
compute aligned ratio + accuracy stats, render markdown table. The
specific aggregators differ only in:
  1. Which file patterns to scan
  2. How to extract condition name from filename
  3. Which metrics to compute
  4. How to format the output

This module provides:
  - ResultSet: load JSONs, extract metrics, group by condition
  - AggregateConfig: declares which patterns to load + how to label them
  - Markdown rendering with parameterized columns
  - Permuted-label aligned ratio out of the box

Usage as library:
    from research.result_aggregator import AggregateConfig, ResultSet
    cfg = AggregateConfig(
        conditions={
            "v2 baseline": "text_eval_R3R6_100ep_HebOff_v2_seed{seed}.json",
            "v2 + SWR":    "text_eval_v2_swr500_seed{seed}.json",
        },
        seeds=[42, 43, 44, 100, 101, 102],
    )
    rs = ResultSet.load(cfg)
    print(rs.render_summary_markdown())
    print(rs.render_per_seed_markdown())
    print(rs.render_aligned_markdown())

Usage as CLI (with shipping config):
    python -m research.result_aggregator --config swr-investigation
    python -m research.result_aggregator --pattern "text_eval_biology_*"
    python -m research.result_aggregator --config biology --output report.md

Anti-shortcut: aligned ratio (TRUE = best perm) is always computed
alongside true accuracy. The single aggregator can't fool itself by
displaying only "true accuracy" without showing the alignment context.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"

WORDS = ["north", "east", "south", "west"]
TRUE_MAP = {"north": "N", "east": "E", "south": "S", "west": "W"}
ACTIONS = ("N", "E", "S", "W")


def _acc_for_mapping(cm: Dict[str, Dict[str, int]],
                     mapping: Dict[str, str]) -> float:
    correct = total = 0
    for word, row in cm.items():
        target = mapping[word]
        for action, count in row.items():
            count = int(count)
            total += count
            if action == target:
                correct += count
    return correct / max(total, 1)


def _best_permutation(cm: Dict[str, Dict[str, int]]) -> Tuple[float, tuple]:
    best_acc = 0.0
    best_perm = None
    for perm in itertools.permutations(ACTIONS):
        mapping = dict(zip(WORDS, perm))
        acc = _acc_for_mapping(cm, mapping)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_acc, best_perm


@dataclass
class RunResult:
    """One eval JSON's parsed metrics."""
    condition: str
    seed: int
    true_acc: float
    best_acc: float
    best_perm: str
    aligned: int  # 1 if best_perm == TRUE, else 0
    i2w_acc: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateConfig:
    """Declares which files to load and how to label them.

    conditions: maps condition_label -> filename template with {seed} placeholder
    seeds: list of seed integers
    raw_dir: directory to glob in (default: research/findings/raw/g11_bg)
    """
    conditions: Dict[str, str]
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 100, 101, 102])
    raw_dir: Path = RAW_DIR


# Built-in named configs for common aggregations
BUILTIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "swr-investigation": {
        "conditions": {
            "v2 baseline":         "text_eval_R3R6_100ep_HebOff_v2_seed{seed}.json",
            "v2 + SWR (default)":  "text_eval_v2_swr500_seed{seed}.json",
            "v2 + SWR balanced":   "text_eval_h1_balanced_seed{seed}.json",
            "PFC isolation (H4)":  "text_eval_h4_isolation_seed{seed}.json",
            "H4 dose-1000":        "text_eval_h4_dose1000_seed{seed}.json",
        },
    },
    "fundamentals": {
        "conditions": {
            "heb_only":          "text_eval_arch_heb_only_seed{seed}.json",
            "drive_5x":          "text_eval_arch_drive_5x_seed{seed}.json",
            "stdp_wmax_10":      "text_eval_arch_stdp_wmax_10_seed{seed}.json",
            "heb_drive":         "text_eval_arch_heb_drive_seed{seed}.json",
            "heb_stdp":          "text_eval_arch_heb_stdp_seed{seed}.json",
            "drive_stdp":        "text_eval_arch_drive_stdp_seed{seed}.json",
        },
    },
    "biology": {
        "conditions": {
            "baseline (random+STDP, no FS)": "text_eval_minimal_iso_seed{seed}.json",
            "+FS only":                       "text_eval_biology_fs_only_seed{seed}.json",
            "+Topo only":                     "text_eval_biology_topo_only_seed{seed}.json",
            "+Topo +FS":                      "text_eval_biology_topo_fs_seed{seed}.json",
        },
    },
    # Post-biology-sweep follow-ups (auto-launched by
    # wait_biology_then_decide.ps1 when biology sweep aligns >= 4/6).
    # Identifies minimum sufficient biology dose.
    "minimum_biology": {
        "conditions": {
            "+Topo weak (1.3/0.8)":       "text_eval_minbio_topo_weak_seed{seed}.json",
            "+FS minimal (1 PV-FSI)":     "text_eval_minbio_fs_minimal_seed{seed}.json",
            "+Topo strong (2.0/0.5)":     "text_eval_minbio_topo_strong_seed{seed}.json",
            "+Combo weak (both halved)":  "text_eval_minbio_combo_weak_seed{seed}.json",
        },
    },
    # Eval methodology sanity check: hand-built perfect language->motor
    # weights, no training. Tests whether eval can detect a known-correct
    # mapping. Auto-launched if biology sweep gives 0/6 across all conds.
    "sanity_check": {
        "conditions": {
            "density 0.30 (matches default)":    "text_eval_sanity_check_density030_seed{seed}.json",
            "density 1.0 (full connectivity)":   "text_eval_sanity_check_density100_seed{seed}.json",
        },
    },
}


@dataclass
class ResultSet:
    """Aggregated results across conditions."""
    config: AggregateConfig
    results: List[RunResult]

    @classmethod
    def load(cls, config: AggregateConfig) -> "ResultSet":
        results: List[RunResult] = []
        for cond_label, pattern in config.conditions.items():
            for seed in config.seeds:
                path = config.raw_dir / pattern.format(seed=seed)
                if not path.exists():
                    continue
                try:
                    d = json.loads(path.read_text())
                except Exception:
                    continue
                cm_raw = (d.get("word_to_action_eval") or {}).get("confusion_matrix")
                if not cm_raw or len(cm_raw) != 4:
                    continue
                cm = {w: {a: int(cm_raw.get(w, {}).get(a, 0)) for a in ACTIONS}
                      for w in WORDS}
                true_acc = _acc_for_mapping(cm, TRUE_MAP)
                best_acc, best_perm = _best_permutation(cm)
                aligned = 1 if best_perm == ACTIONS else 0
                i2w = (d.get("image_to_word_eval") or {}).get("accuracy")
                results.append(RunResult(
                    condition=cond_label, seed=seed,
                    true_acc=true_acc, best_acc=best_acc,
                    best_perm="".join(best_perm) if best_perm else "?",
                    aligned=aligned,
                    i2w_acc=i2w,
                ))
        return cls(config=config, results=results)

    def by_condition(self) -> Dict[str, List[RunResult]]:
        out: Dict[str, List[RunResult]] = {}
        for r in self.results:
            out.setdefault(r.condition, []).append(r)
        return out

    def render_summary_markdown(self) -> str:
        """Headline summary table - one row per condition with aligned ratio."""
        out: List[str] = []
        out.append("| Condition | n | true mean | best mean | excess | "
                   "**aligned/n** | I->W mean |")
        out.append("|---|---|---|---|---|---|---|")
        by_cond = self.by_condition()
        for cond_label in self.config.conditions.keys():
            rows = by_cond.get(cond_label, [])
            if not rows:
                out.append(f"| {cond_label} | 0 | - | - | - | (no data) | - |")
                continue
            true_mean = statistics.mean(r.true_acc for r in rows)
            best_mean = statistics.mean(r.best_acc for r in rows)
            excess = best_mean - true_mean
            aligned = sum(r.aligned for r in rows)
            n = len(rows)
            i2w_vals = [r.i2w_acc for r in rows if r.i2w_acc is not None]
            i2w_mean_str = (f"{100*statistics.mean(i2w_vals):.1f}%"
                            if i2w_vals else "-")
            verdict = ("**REAL LEARNING**" if aligned >= 4
                       else ("**probably real**" if aligned >= 2
                             else "noise"))
            out.append(
                f"| {cond_label} | {n} | {100*true_mean:.1f}% | "
                f"{100*best_mean:.1f}% | +{100*excess:.1f}pp | "
                f"**{aligned}/{n}** ({verdict}) | {i2w_mean_str} |"
            )
        return "\n".join(out)

    def render_per_seed_markdown(self) -> str:
        """Per-seed cross-condition table."""
        cond_labels = list(self.config.conditions.keys())
        out: List[str] = []
        out.append("| seed | " + " | ".join(cond_labels) + " |")
        out.append("|---" + "|---" * len(cond_labels) + "|")
        by_cond_seed: Dict[Tuple[str, int], RunResult] = {}
        for r in self.results:
            by_cond_seed[(r.condition, r.seed)] = r
        for seed in self.config.seeds:
            cells = [str(seed)]
            for cl in cond_labels:
                r = by_cond_seed.get((cl, seed))
                if r is None:
                    cells.append("-")
                else:
                    al = "*" if r.aligned else ""
                    cells.append(f"{100*r.true_acc:.0f}% {al}".strip())
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    def render_aligned_table(self) -> str:
        """Per-seed aligned-flag table - definitive learning-vs-noise."""
        out: List[str] = []
        out.append("| condition | seed | true | best | best_perm | aligned |")
        out.append("|---|---|---|---|---|---|")
        for r in self.results:
            mark = "**YES**" if r.aligned else "no"
            out.append(
                f"| {r.condition} | {r.seed} | {100*r.true_acc:.1f}% | "
                f"{100*r.best_acc:.1f}% | {r.best_perm} | {mark} |"
            )
        return "\n".join(out)

    def verdict(self) -> str:
        """One-line verdict on whether real learning achieved."""
        by_cond = self.by_condition()
        winners = [k for k, rs in by_cond.items()
                   if sum(r.aligned for r in rs) >= 4 and len(rs) >= 6]
        if winners:
            return ("**Real word-action learning achieved.** "
                    f"Conditions with aligned >= 4/6: {', '.join(winners)}")
        partials = [k for k, rs in by_cond.items()
                    if sum(r.aligned for r in rs) >= 2]
        if partials:
            return ("**Partial signal.** Some seeds aligned in: "
                    f"{', '.join(partials)}")
        any_data = any(rs for rs in by_cond.values())
        if not any_data:
            return "(no data yet)"
        return ("**No real learning.** All conditions show 0-1 aligned of N. "
                "Architecture noise, not word-action learning.")


def render_full_report(rs: ResultSet, title: str = "Result aggregation") -> str:
    out: List[str] = []
    out.append(f"# {title}")
    out.append("")
    out.append("**Headline:** " + rs.verdict())
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(rs.render_summary_markdown())
    out.append("")
    out.append("## Per-seed cross-condition")
    out.append("")
    out.append(rs.render_per_seed_markdown())
    out.append("")
    out.append("* marks seeds where TRUE = best permutation (aligned).")
    out.append("")
    out.append("## Aligned details")
    out.append("")
    out.append(rs.render_aligned_table())
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(BUILTIN_CONFIGS.keys()),
                    help="Built-in config (swr-investigation, fundamentals, "
                         "biology, minimum_biology, sanity_check)")
    ap.add_argument("--pattern", action="append", default=None,
                    help="Custom condition label=pattern (repeatable). "
                         "E.g. --pattern 'mine=text_eval_my_*_seed{seed}.json'")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output markdown path (default: stdout)")
    ap.add_argument("--title", default="Result aggregation")
    args = ap.parse_args()

    if args.config:
        conditions = BUILTIN_CONFIGS[args.config]["conditions"]
    elif args.pattern:
        conditions = {}
        for spec in args.pattern:
            if "=" not in spec:
                raise ValueError(f"--pattern spec must be label=glob, got: {spec}")
            label, pat = spec.split("=", 1)
            conditions[label.strip()] = pat.strip()
    else:
        ap.error("must provide --config or one or more --pattern")

    cfg = AggregateConfig(
        conditions=conditions,
        seeds=args.seeds,
        raw_dir=args.raw_dir,
    )
    rs = ResultSet.load(cfg)
    report = render_full_report(rs, title=args.title)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out} ({len(report)} bytes)")
    else:
        print(report)


if __name__ == "__main__":
    main()
