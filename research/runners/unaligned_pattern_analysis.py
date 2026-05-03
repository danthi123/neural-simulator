"""
Pattern analysis: across the 22+ runs that show aligned=0/N, what is
the SHAPE of the misalignment?

For each W->A confusion matrix:
  1. Find the best permutation (true=best perm).
  2. Record which (word, action) pairs are swapped relative to TRUE.
  3. Count: across all conditions/seeds, which (word, action) "swaps"
     are most frequent? If e.g. "north -> east" appears as the best
     mapping in 60% of seeds, the cascade has a structural N->E bias.

Also analyzes:
  - Per-cell average rate (which (word, action) cell has highest
    average count across all conditions, regardless of true labels)
  - Which cardinal action gets the most predictions overall, regardless
    of which word is driving (cascade default-fire pattern)

Output: markdown report identifying the dominant structural biases.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"

WORDS = ["north", "east", "south", "west"]
TRUE_MAP = {"north": "N", "east": "E", "south": "S", "west": "W"}
ACTIONS = ("N", "E", "S", "W")


def acc_for_mapping(cm: dict, mapping: dict) -> float:
    correct = total = 0
    for word, row in cm.items():
        target = mapping[word]
        for action, count in row.items():
            count = int(count)
            total += count
            if action == target:
                correct += count
    return correct / max(total, 1)


def best_permutation(cm: dict) -> tuple[float, tuple]:
    best_acc = 0.0
    best_perm = None
    for perm in itertools.permutations(ACTIONS):
        mapping = dict(zip(WORDS, perm))
        acc = acc_for_mapping(cm, mapping)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_acc, best_perm


def parse_filename(filename: str) -> Optional[tuple[str, int]]:
    m = re.match(r"text_eval_(.+)_seed(\d+)\.json$", filename)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="Output markdown file (default: stdout)")
    args = ap.parse_args()

    files = sorted(RAW_DIR.glob("text_eval_*_seed*.json"))

    # Aggregate confusion matrices by (cm_norm, condition_seed_pair)
    all_cms = []
    best_perm_counter = Counter()
    word_to_action_count = defaultdict(lambda: defaultdict(int))  # word -> {action -> count summed across runs}

    for path in files:
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        cond, seed = parsed
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        cm_raw = (d.get("word_to_action_eval") or {}).get("confusion_matrix")
        if not cm_raw or len(cm_raw) != 4:
            continue
        cm = {w: {a: int(cm_raw.get(w, {}).get(a, 0)) for a in ACTIONS}
              for w in WORDS}
        all_cms.append((cond, seed, cm))
        _, best_perm = best_permutation(cm)
        if best_perm:
            best_perm_counter[best_perm] += 1
        # Sum word -> action counts across all runs
        for w in WORDS:
            for a in ACTIONS:
                word_to_action_count[w][a] += cm[w][a]

    out = []
    out.append("# Unaligned-structure pattern analysis")
    out.append("")
    out.append(f"Analyzed {len(all_cms)} W->A confusion matrices across "
               f"{len(set(c for c, _, _ in all_cms))} conditions.")
    out.append("")

    out.append("## Most common best-permutation across all runs")
    out.append("")
    out.append(f"If learning was real, the TRUE permutation `(N, E, S, W)` "
               f"would dominate. Instead:")
    out.append("")
    out.append("| best perm | count | % | mapping (north->_, east->_, south->_, west->_) |")
    out.append("|---|---|---|---|")
    total = sum(best_perm_counter.values())
    for perm, n in best_perm_counter.most_common(15):
        pct = 100 * n / total
        is_true = perm == ACTIONS
        marker = " (TRUE)" if is_true else ""
        out.append(f"| `{''.join(perm)}` | {n} | {pct:.1f}% | "
                   f"north->{perm[0]}, east->{perm[1]}, south->{perm[2]}, west->{perm[3]}{marker} |")
    if ACTIONS not in best_perm_counter:
        out.append(f"| **TRUE (NESW)** | **0** | **0.0%** | (never the best!) |")

    # Most common single (word -> action) pair in best perms
    out.append("")
    out.append("## Per-word: where does each word's signal go?")
    out.append("")
    out.append("Across all runs, count what action the BEST permutation")
    out.append("assigned to each word.")
    out.append("")
    word_pos_counter = {w: Counter() for w in WORDS}
    for perm, n in best_perm_counter.items():
        for i, w in enumerate(WORDS):
            word_pos_counter[w][perm[i]] += n
    out.append("| word | true action | most common best-perm action | count | % |")
    out.append("|---|---|---|---|---|")
    for w in WORDS:
        true_a = TRUE_MAP[w]
        cc = word_pos_counter[w]
        for a, count in cc.most_common(1):
            pct = 100 * count / total
            mark = " (matches true)" if a == true_a else " (DIFFERENT from true)"
            out.append(f"| {w} | {true_a} | {a}{mark} | {count} | {pct:.1f}% |")
        # Show all 4 actions
        out.append(f"|  |  | (all 4 actions: " +
                   ", ".join(f"{a}={cc.get(a, 0)}" for a in ACTIONS) + f") |")

    # Per-cell average across all runs (regardless of best perm)
    out.append("")
    out.append("## Per-cell average count across all runs")
    out.append("")
    out.append("Total counts in each (word, action) cell, summed across all")
    out.append("runs. If cascade has structural biases, certain cells will")
    out.append("dominate regardless of which word is driving.")
    out.append("")
    out.append("| | -> N | -> E | -> S | -> W | row total |")
    out.append("|---|---|---|---|---|---|")
    for w in WORDS:
        row = word_to_action_count[w]
        row_total = sum(row.values())
        cells = []
        for a in ACTIONS:
            c = row[a]
            pct = 100 * c / row_total if row_total else 0
            mark = "**" if a == TRUE_MAP[w] else ""
            cells.append(f"{mark}{c} ({pct:.1f}%){mark}")
        out.append(f"| {w} | " + " | ".join(cells) + f" | {row_total} |")

    # Action totals (how often each action is predicted overall)
    action_totals = {a: sum(word_to_action_count[w][a] for w in WORDS)
                     for a in ACTIONS}
    grand_total = sum(action_totals.values())
    out.append("")
    out.append("## Action prediction frequency (overall cascade bias)")
    out.append("")
    out.append("How often is each action predicted, summed across all words?")
    out.append("If the architecture were unbiased, each action would be ~25%.")
    out.append("")
    out.append("| action | total predictions | % |")
    out.append("|---|---|---|")
    for a in ACTIONS:
        c = action_totals[a]
        pct = 100 * c / grand_total
        out.append(f"| {a} | {c} | **{pct:.1f}%** |")

    # Implication
    out.append("")
    out.append("## Implication")
    out.append("")
    most_common_perm, mc_n = best_perm_counter.most_common(1)[0]
    out.append(f"The most common best permutation is `{''.join(most_common_perm)}` "
               f"({mc_n}/{total} runs = {100*mc_n/total:.1f}%).")
    out.append("")
    out.append("If a single permutation appeared as best across many seeds, the")
    out.append("architecture has a CONSISTENT structural bias that overrides")
    out.append("training. If best perms are scattered (each ~5-10% frequency),")
    out.append("the bias is seed-dependent (each random init creates its own")
    out.append("private misalignment).")
    out.append("")
    if mc_n / total > 0.20:
        out.append("**Result: dominant structural bias** — same permutation in")
        out.append(f"{100*mc_n/total:.0f}% of runs strongly suggests an")
        out.append("architectural pattern that's reproducible across seeds.")
    else:
        out.append("**Result: scattered (seed-dependent) bias** — no single")
        out.append("permutation dominates. Each seed builds its own private")
        out.append("misalignment from random init dynamics.")

    output = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote pattern analysis to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
