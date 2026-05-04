"""
Biology sweep summary — aggregates the 4 minimal-arch conditions into
one comparable table.

Conditions:
  1. baseline:  random init, no FS         (text_eval_minimal_iso_seed*.json)
  2. fs_only:   random init, +motor FS     (text_eval_biology_fs_only_seed*.json)
  3. topo_only: topo init 1.5/0.7, no FS   (text_eval_biology_topo_only_seed*.json)
  4. topo_fs:   topo init + FS             (text_eval_biology_topo_fs_seed*.json)

Plus the anti-cheat control:
  anticheat:    topo init + freeze STDP    (text_eval_biology_anticheat_seed*.json)
                — should NOT align if topographic factor is biology-mild

Headline metric: aligned ratio (TRUE labeled mapping is the BEST of 24
permutations across seeds). >= 4/6 = real learning, joint random p < 1e-3.

Usage:
    python -m research.runners.biology_sweep_summary
    python -m research.runners.biology_sweep_summary --out report.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"

WORDS = ["north", "east", "south", "west"]
TRUE_MAP = {"north": "N", "east": "E", "south": "S", "west": "W"}
ACTIONS = ("N", "E", "S", "W")

CONDITIONS = [
    ("baseline (random+STDP, no FS)", "text_eval_minimal_iso_seed{seed}.json",
     "Condition 1: cascade-free, random language->motor weights, "
     "STDP only. The current minimal-iso run."),
    ("+FS (random+STDP+motor lat-inhib)", "text_eval_biology_fs_only_seed{seed}.json",
     "Condition 2: adds PV-FS interneurons providing cross-pool "
     "lateral inhibition. Tests whether competition between motor "
     "pools is the missing ingredient."),
    ("+Topo (topo+STDP, no FS)", "text_eval_biology_topo_only_seed{seed}.json",
     "Condition 3: topographic prior 1.5/0.7 (Pulvermuller mid-biology) "
     "applied to language->motor weights at init. STDP refines from "
     "structured prior."),
    ("+Topo+FS (combined biology fix)", "text_eval_biology_topo_fs_seed{seed}.json",
     "Condition 4: full biology-grounded combination. Topography + "
     "lateral inhibition + STDP. The hypothesized real-learning config."),
]

ANTI_CHEAT_PATTERN = "text_eval_biology_anticheat_seed{seed}.json"
SEEDS = [42, 43, 44, 100, 101, 102]


def acc_for_mapping(cm, mapping):
    correct = total = 0
    for word, row in cm.items():
        target = mapping[word]
        for action, count in row.items():
            count = int(count)
            total += count
            if action == target:
                correct += count
    return correct / max(total, 1)


def best_permutation(cm):
    best_acc = 0.0
    best_perm = None
    for perm in itertools.permutations(ACTIONS):
        mapping = dict(zip(WORDS, perm))
        acc = acc_for_mapping(cm, mapping)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_acc, best_perm


def load_condition(pattern):
    """Load (seed, true_acc, best_acc, best_perm, aligned) tuples."""
    rows = []
    for seed in SEEDS:
        path = RAW_DIR / pattern.format(seed=seed)
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
        true_acc = acc_for_mapping(cm, TRUE_MAP)
        best_acc, best_perm = best_permutation(cm)
        aligned = 1 if best_perm == ACTIONS else 0
        rows.append({
            "seed": seed,
            "true_acc": true_acc,
            "best_acc": best_acc,
            "best_perm": "".join(best_perm) if best_perm else "?",
            "aligned": aligned,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="Output markdown file")
    args = ap.parse_args()

    out = []
    out.append("# Biology-grounded sweep — results summary")
    out.append("")
    out.append("Tests biology-corrected additions to the minimal language->motor")
    out.append("isolation arch (cascade-free): topographic prior, motor PV-FS")
    out.append("lateral inhibition, and the combination.")
    out.append("")
    out.append("**Headline metric:** aligned ratio. TRUE labeled mapping is the")
    out.append("BEST of 24 permutations across seeds.")
    out.append("- aligned 6/6 = definitive real learning (joint p ~ 6e-9)")
    out.append("- aligned 4/6 = probably real (joint p ~ 7e-4)")
    out.append("- aligned 0-1/6 = no real word-action learning (architecture noise)")
    out.append("")

    # Anti-cheat first
    out.append("## Anti-cheat control: topo init + freeze STDP")
    out.append("")
    out.append("Tests whether topographic prior ALONE (without STDP) solves the")
    out.append("task. If aligned >= 1/1, the prior is too strong and biases the")
    out.append("answer rather than nudging the learner.")
    out.append("")
    anti_rows = load_condition(ANTI_CHEAT_PATTERN)
    if not anti_rows:
        out.append("No anti-cheat results yet.")
    else:
        out.append("| seed | true | best | best perm | aligned |")
        out.append("|---|---|---|---|---|")
        for r in anti_rows:
            mark = "**ALIGNED (CHEAT!)**" if r["aligned"] else "no"
            out.append(f"| {r['seed']} | {100*r['true_acc']:.1f}% | "
                       f"{100*r['best_acc']:.1f}% | {r['best_perm']} | {mark} |")
        if any(r["aligned"] for r in anti_rows):
            out.append("")
            out.append("**WARNING: anti-cheat triggered.** Topographic prior alone")
            out.append("aligned without STDP. Reduce factor to 1.3/0.8 and re-run.")

    out.append("")
    out.append("## Main conditions: aligned ratio across 6 seeds")
    out.append("")
    out.append("| Condition | n | true mean | best mean | excess | **aligned/n** |")
    out.append("|---|---|---|---|---|---|")

    summary_data = {}
    for label, pattern, _desc in CONDITIONS:
        rows = load_condition(pattern)
        if not rows:
            out.append(f"| {label} | 0 | - | - | - | (no data) |")
            continue
        true_mean = statistics.mean(r["true_acc"] for r in rows)
        best_mean = statistics.mean(r["best_acc"] for r in rows)
        excess = best_mean - true_mean
        aligned = sum(r["aligned"] for r in rows)
        n = len(rows)
        flag = "**REAL LEARNING**" if aligned >= 4 else (
            "**probably real**" if aligned >= 2 else "noise")
        out.append(f"| {label} | {n} | {100*true_mean:.1f}% | "
                   f"{100*best_mean:.1f}% | +{100*excess:.1f}pp | "
                   f"**{aligned}/{n}** ({flag}) |")
        summary_data[label] = {"rows": rows, "aligned": aligned, "n": n}

    out.append("")
    out.append("## Per-seed details")
    out.append("")
    for label, pattern, desc in CONDITIONS:
        rows = load_condition(pattern)
        if not rows:
            continue
        out.append(f"### {label}")
        out.append(f"_{desc}_")
        out.append("")
        out.append("| seed | true | best | best perm | aligned |")
        out.append("|---|---|---|---|---|")
        for r in rows:
            mark = "**YES**" if r["aligned"] else "no"
            out.append(f"| {r['seed']} | {100*r['true_acc']:.1f}% | "
                       f"{100*r['best_acc']:.1f}% | {r['best_perm']} | {mark} |")
        out.append("")

    # Verdict
    out.append("## Verdict")
    out.append("")
    if not summary_data:
        out.append("Insufficient data to draw conclusions.")
    else:
        any_aligned_4 = any(d["aligned"] >= 4 for d in summary_data.values())
        any_aligned_2 = any(d["aligned"] >= 2 for d in summary_data.values())
        all_zero = all(d["aligned"] == 0 for d in summary_data.values())
        if any_aligned_4:
            winners = [k for k, v in summary_data.items() if v["aligned"] >= 4]
            out.append("**Real word-action learning achieved.** "
                       f"Conditions with aligned >= 4/6: {', '.join(winners)}")
            out.append("")
            out.append("Next steps:")
            out.append("- Identify the smallest sufficient biology fix (topo only? FS only? both?)")
            out.append("- Re-introduce cascade with reduced default weights, see if alignment survives")
            out.append("- Scale up to v2 architecture with the validated biology pieces")
        elif any_aligned_2:
            partial = [k for k, v in summary_data.items() if v["aligned"] >= 2]
            out.append("**Partial signal.** Some seeds aligned in: "
                       f"{', '.join(partial)}")
            out.append("")
            out.append("Suggests biology fix is ALMOST sufficient. Next:")
            out.append("- Lengthen training (3000 events/dir vs 1000)")
            out.append("- Verify with different seed sets")
            out.append("- Check if outliers reflect specific seed properties")
        elif all_zero:
            out.append("**No real learning in any condition.** The architecture")
            out.append("cannot align word-action mapping with task labels even")
            out.append("with biology-corrected topographic prior + lateral inhibition.")
            out.append("")
            out.append("This rules out the 'cascade is the problem' hypothesis.")
            out.append("Implication: the issue is more fundamental.")
            out.append("")
            out.append("Hypotheses to test next:")
            out.append("- Eval methodology itself may have a bug "
                       "(test by training a TRIVIAL model that should obviously align)")
            out.append("- Sparse-code overlap fundamentally prevents discrimination")
            out.append("- Plasticity rules (STDP+R-STDP) cannot create the discriminative")
            out.append("  pattern from sparse inputs in this many trials")
            out.append("- Need supervised gradient learning instead of STDP")
        else:
            out.append("Mixed results — needs interpretation.")

    output = "\n".join(out) + "\n"

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote summary to {args.out} ({len(output)} bytes)")
    else:
        print(output)


if __name__ == "__main__":
    main()
