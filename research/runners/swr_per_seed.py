"""
Per-seed cross-condition analysis.

For each seed in {42, 43, 44, 100, 101, 102}, compare:
  * v2 baseline           (no SWR)
  * v2 + SWR (default)    (frequency-weighted replay)
  * v2 + SWR balanced     (H1)
  * PFC bypass isolation  (H4)

So the user can see at a glance:
  "For seed X, what does each condition give for W->A?"

This is more illuminating than aggregate means alone since the conditions
share architecture but differ in training procedure. Per-seed comparison
isolates the EFFECT OF THE PROCEDURE while controlling for seed-specific
random init.
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"

CONDITIONS = [
    ("v2 baseline",        "text_eval_R3R6_100ep_HebOff_v2_seed{seed}.json"),
    ("v2 + SWR default",   "text_eval_v2_swr500_seed{seed}.json"),
    ("v2 + SWR balanced",  "text_eval_h1_balanced_seed{seed}.json"),
    ("H4 PFC isolation",   "text_eval_h4_isolation_seed{seed}.json"),
]
SEEDS = [42, 43, 44, 100, 101, 102]


def acc_for(path: Path):
    if not path.exists(): return None
    try:
        d = json.loads(path.read_text())
        return d.get("word_to_action_eval", {}).get("accuracy")
    except Exception:
        return None


def main():
    print("# Per-seed W->A across conditions")
    print()
    # Header
    cols = ["seed"] + [c[0] for c in CONDITIONS]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    # Rows
    rows_data = {}
    for seed in SEEDS:
        row = [str(seed)]
        rows_data[seed] = {}
        for label, pat in CONDITIONS:
            path = RAW_DIR / pat.format(seed=seed)
            acc = acc_for(path)
            rows_data[seed][label] = acc
            row.append(f"{100*acc:.0f}%" if acc is not None else "-")
        print("| " + " | ".join(row) + " |")
    # Mean +/- std footer
    means_row = ["**mean +/- std**"]
    for label, _ in CONDITIONS:
        accs = [rows_data[s][label] for s in SEEDS if rows_data[s][label] is not None]
        if not accs:
            means_row.append("-")
        else:
            m = statistics.mean(accs) * 100
            s = (statistics.stdev(accs) if len(accs) > 1 else 0) * 100
            means_row.append(f"**{m:.1f}% +/- {s:.1f}%** (n={len(accs)})")
    print("| " + " | ".join(means_row) + " |")

    print()
    print("## Per-seed delta vs baseline")
    print()
    print("| seed | v2+SWR delta | H1 balanced delta | H4 isolation delta |")
    print("|---|---|---|---|")
    for seed in SEEDS:
        base = rows_data[seed]["v2 baseline"]
        if base is None:
            print(f"| {seed} | — | — | — |")
            continue
        cells = []
        for cond in ("v2 + SWR default", "v2 + SWR balanced", "H4 PFC isolation"):
            v = rows_data[seed][cond]
            if v is None:
                cells.append("n/a")
            else:
                d = (v - base) * 100
                sign = "+" if d >= 0 else ""
                cells.append(f"{sign}{d:.0f}pp")
        print(f"| {seed} | {' | '.join(cells)} |")

    # Pairwise paired-difference test for SWR vs baseline (most rigorous test)
    print()
    print("## Paired-difference test: v2+SWR vs v2 baseline")
    print()
    base_swr_pairs = [
        (rows_data[s]["v2 baseline"], rows_data[s]["v2 + SWR default"])
        for s in SEEDS
        if rows_data[s]["v2 baseline"] is not None
        and rows_data[s]["v2 + SWR default"] is not None
    ]
    if len(base_swr_pairs) >= 3:
        diffs = [swr - base for base, swr in base_swr_pairs]
        m = statistics.mean(diffs) * 100
        s = (statistics.stdev(diffs) if len(diffs) > 1 else 0) * 100
        print(f"n={len(base_swr_pairs)}")
        print(f"mean delta (SWR - baseline): {m:+.2f}pp +/- {s:.2f}pp")
        if len(diffs) > 1 and s > 0:
            t = (m / 100) / ((s / 100) / (len(diffs) ** 0.5))
            print(f"paired t-statistic: {t:.2f}")


if __name__ == "__main__":
    main()
