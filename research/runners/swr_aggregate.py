"""
Aggregate all SWR-related runs into a single summary table.

Scans research/findings/raw/g11_bg/ for:
  * text_eval_v2_swr500_seed*.json    — v2 + standard SWR replay
  * text_eval_h1_balanced_seed*.json  — H1 balanced replay
  * text_eval_h4_isolation_seed*.json — H4 PFC bypass isolation

Prints a markdown table comparing W->A, I->W, per-direction breakdown
across seeds and conditions.

Used by the autonomous overnight loop to keep the findings doc
auto-updated as new seeds complete. Idempotent — can run any time.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"

CONDITIONS = {
    "v2 baseline":         "text_eval_R3R6_100ep_HebOff_v2_seed{seed}.json",
    "v2 + SWR (default)":  "text_eval_v2_swr500_seed{seed}.json",
    "v2 + SWR balanced":   "text_eval_h1_balanced_seed{seed}.json",
    "PFC isolation (H4)":  "text_eval_h4_isolation_seed{seed}.json",
    "H4 dose (1000/dir)":  "text_eval_h4_dose1000_seed{seed}.json",
    # Fundamentals sweep (2026-05-03 afternoon) — pivoting from
    # v2-tweak structural changes to biology-correct fundamentals
    # after permuted-label control showed 0/29 aligned in v2-variant
    # evals. Tests Hebbian re-enable + reduced decay, stronger drive,
    # higher stdp_w_max.
    "fund heb_only":                   "text_eval_arch_heb_only_seed{seed}.json",
    "fund drive_5x":                   "text_eval_arch_drive_5x_seed{seed}.json",
    "fund stdp_wmax_10":               "text_eval_arch_stdp_wmax_10_seed{seed}.json",
    "fund heb_drive":                  "text_eval_arch_heb_drive_seed{seed}.json",
    "fund heb_stdp":                   "text_eval_arch_heb_stdp_seed{seed}.json",
    "fund drive_stdp":                 "text_eval_arch_drive_stdp_seed{seed}.json",
}

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]


def load_run(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def per_direction_w2a(data: dict) -> dict:
    """Returns {north: pct, east: pct, south: pct, west: pct} for W->A."""
    cm = (data.get("word_to_action_eval") or {}).get("confusion_matrix") or {}
    out = {}
    for word in ("north", "east", "south", "west"):
        row = cm.get(word, {}) or {}
        target = word[0].upper()
        total = sum(int(v) for v in row.values())
        correct = int(row.get(target, 0))
        out[word] = correct / total if total > 0 else None
    return out


def per_direction_i2w(data: dict) -> dict:
    """Returns {north: pct, ...} for I->W (rows = ground truth, diagonal cell = correct)."""
    cm = (data.get("image_to_word_eval") or {}).get("confusion_matrix") or {}
    out = {}
    for word in ("north", "east", "south", "west"):
        row = cm.get(word, {}) or {}
        total = sum(int(v) for v in row.values())
        correct = int(row.get(word, 0))
        out[word] = correct / total if total > 0 else None
    return out


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.1f}%"


def format_summary_table(by_condition: dict) -> str:
    """Build a markdown table summarizing the seed runs.

    by_condition: { condition_label: [(seed, data) for seed in seeds] }
    """
    lines = []
    lines.append("| Condition | seed 42 | seed 43 | seed 44 | seed 100 | seed 101 | seed 102 | mean ± std |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for cond_label, runs_by_seed in by_condition.items():
        # Format each cell as "I/W: 39% / W/A: 22%"
        cells = []
        accs_iw, accs_wa = [], []
        for seed in DEFAULT_SEEDS:
            data = runs_by_seed.get(seed)
            if data is None:
                cells.append("—")
                continue
            iw = (data.get("image_to_word_eval") or {}).get("accuracy")
            wa = (data.get("word_to_action_eval") or {}).get("accuracy")
            if iw is not None: accs_iw.append(iw)
            if wa is not None: accs_wa.append(wa)
            cells.append(
                f"I={fmt_pct(iw).rstrip('%')}/W={fmt_pct(wa).rstrip('%')}"
            )
        # Mean ± std
        if accs_wa:
            mean_wa = statistics.mean(accs_wa)
            std_wa = statistics.stdev(accs_wa) if len(accs_wa) > 1 else 0.0
            mean_iw = statistics.mean(accs_iw) if accs_iw else None
            std_iw = statistics.stdev(accs_iw) if len(accs_iw) > 1 else 0.0
            mean_cell = (
                f"I={fmt_pct(mean_iw)}±{100*std_iw:.1f}/" if mean_iw else "I=—/"
            ) + f"W={fmt_pct(mean_wa)}±{100*std_wa:.1f}"
        else:
            mean_cell = "—"

        lines.append(f"| {cond_label} | " + " | ".join(cells) + f" | {mean_cell} |")

    return "\n".join(lines)


def format_per_direction_table(condition: str, runs_by_seed: dict) -> str:
    """Per-direction W->A breakdown for one condition across seeds."""
    lines = [f"### {condition} — W->A per direction", ""]
    lines.append("| direction | " + " | ".join(f"seed {s}" for s in DEFAULT_SEEDS) + " | mean |")
    lines.append("|---|" + "|".join("---" for _ in DEFAULT_SEEDS) + "|---|")
    for word in ("north", "east", "south", "west"):
        cells = []
        accs = []
        for seed in DEFAULT_SEEDS:
            data = runs_by_seed.get(seed)
            if data is None:
                cells.append("—")
                continue
            pd = per_direction_w2a(data)
            v = pd.get(word)
            if v is not None:
                accs.append(v)
            cells.append(fmt_pct(v))
        mean_cell = fmt_pct(statistics.mean(accs)) if accs else "—"
        lines.append(f"| {word} | " + " | ".join(cells) + f" | **{mean_cell}** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="Output markdown file (default: stdout)")
    args = ap.parse_args()

    by_condition = {}
    for cond_label, pattern in CONDITIONS.items():
        runs_by_seed = {}
        for seed in DEFAULT_SEEDS:
            path = RAW_DIR / pattern.format(seed=seed)
            data = load_run(path)
            if data is not None:
                runs_by_seed[seed] = data
        by_condition[cond_label] = runs_by_seed

    out_lines = []
    out_lines.append("# SWR investigation — auto-aggregated summary")
    out_lines.append("")
    out_lines.append("Auto-generated by `research/runners/swr_aggregate.py`. Do not hand-edit.")
    out_lines.append("")
    out_lines.append("Each cell shows `I=<i2w>/W=<w2a>` accuracy. Mean ± std taken across available seeds (note that conditions with no completed runs show `—`).")
    out_lines.append("")
    out_lines.append("## Headline")
    out_lines.append("")
    out_lines.append(format_summary_table(by_condition))
    out_lines.append("")
    for cond_label, runs in by_condition.items():
        if any(data is not None for data in runs.values()):
            out_lines.append("")
            out_lines.append(format_per_direction_table(cond_label, runs))
    output = "\n".join(out_lines) + "\n"

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote summary to {args.out} ({len(output)} bytes)")
    else:
        print(output)


if __name__ == "__main__":
    main()
