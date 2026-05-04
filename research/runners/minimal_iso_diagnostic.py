"""Diagnostic on the minimal-isolation 'below chance' result (16.7% mean
across 3 seeds, vs 25% chance baseline).

The minimal-iso architecture (no cascade, just language_input -> motor_X
under paired-stim training) was expected to show "the cascade was the
interference; without it the architecture should learn cleanly" — but
instead returned BELOW CHANCE alignment. Why?

This script analyzes the existing minimal_iso_seed{42,43,44} JSON output
files to understand:

1. Per-direction accuracy: which directions are systematically wrong?
2. Best-permutation analysis: what mapping IS the architecture learning?
3. Cross-seed consistency: is the misalignment systematic or random?
4. Specific patterns: does "north <-> south" inversion hold? Is "E" a sink?
5. Comparison: how does this differ from v2 baseline cascade-included data?

Usage:
    python -m research.runners.minimal_iso_diagnostic
    python -m research.runners.minimal_iso_diagnostic --out research/findings/2026-05-04-minimal-iso-diagnostic.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple


WORDS = ["north", "east", "south", "west"]
ACTIONS = ["N", "E", "S", "W"]
TRUE_MAP = {"north": "N", "east": "E", "south": "S", "west": "W"}


def _load(seed: int, raw_dir: Path) -> Dict:
    p = raw_dir / f"text_eval_minimal_iso_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _confusion(d: Dict) -> Dict[str, Dict[str, int]]:
    return d["word_to_action_eval"]["confusion_matrix"]


def _per_word_acc(cm: Dict[str, Dict[str, int]],
                   mapping: Dict[str, str]) -> Dict[str, float]:
    """Accuracy per word under given mapping."""
    out = {}
    for w in WORDS:
        total = sum(cm[w].values())
        out[w] = cm[w][mapping[w]] / total if total > 0 else 0.0
    return out


def _all_permutation_accs(cm: Dict[str, Dict[str, int]]) -> List[Tuple[float, Dict[str, str]]]:
    """Try all 24 permutations of (word -> action), return (acc, mapping)."""
    perms = []
    for perm in itertools.permutations(ACTIONS):
        mapping = dict(zip(WORDS, perm))
        correct = sum(cm[w][mapping[w]] for w in WORDS)
        total = sum(sum(cm[w].values()) for w in WORDS)
        acc = correct / total if total > 0 else 0.0
        perms.append((acc, mapping))
    perms.sort(key=lambda t: -t[0])
    return perms


def _mapping_str(m: Dict[str, str]) -> str:
    return "->".join([f"{w[0].upper()}:{m[w]}" for w in WORDS])


def _analyze_seed(seed: int, raw_dir: Path) -> Dict:
    d = _load(seed, raw_dir)
    if d is None:
        return None
    cm = _confusion(d)

    # True accuracy
    true_acc = sum(cm[w][TRUE_MAP[w]] for w in WORDS) / sum(
        sum(cm[w].values()) for w in WORDS
    )
    per_word_true = _per_word_acc(cm, TRUE_MAP)

    # Best permutation
    perms = _all_permutation_accs(cm)
    best_acc, best_map = perms[0]

    # North-South inversion check
    n_to_s_rate = cm["north"]["S"] / sum(cm["north"].values())
    s_to_n_rate = cm["south"]["N"] / sum(cm["south"].values())
    ns_inversion = (n_to_s_rate > 0.30) and (s_to_n_rate > 0.30)

    # E-sink check (multiple words map to E)
    e_sink_count = sum(1 for w in WORDS
                       if cm[w]["E"] / sum(cm[w].values()) > 0.30)

    # Action bias: which action is most often picked across all words
    action_totals = {a: 0 for a in ACTIONS}
    grand_total = 0
    for w in WORDS:
        for a in ACTIONS:
            action_totals[a] += cm[w][a]
            grand_total += cm[w][a]
    action_distribution = {a: action_totals[a] / grand_total
                           for a in ACTIONS}

    return {
        "seed": seed,
        "true_acc": true_acc,
        "per_word_true": per_word_true,
        "best_acc": best_acc,
        "best_map": best_map,
        "best_map_str": _mapping_str(best_map),
        "ns_inversion": ns_inversion,
        "n_to_s_rate": n_to_s_rate,
        "s_to_n_rate": s_to_n_rate,
        "e_sink_count": e_sink_count,
        "action_distribution": action_distribution,
        "confusion": cm,
        "top_5_perms": [(round(acc, 3), _mapping_str(m))
                         for acc, m in perms[:5]],
    }


def _format_confusion(cm: Dict[str, Dict[str, int]]) -> str:
    """Render a confusion matrix as a markdown table."""
    lines = []
    lines.append("| word \\ action | N | E | S | W | total |")
    lines.append("|---|---|---|---|---|---|")
    for w in WORDS:
        row = [str(cm[w][a]) for a in ACTIONS]
        total = sum(cm[w].values())
        lines.append(f"| {w} | " + " | ".join(row) + f" | {total} |")
    return "\n".join(lines)


def _write_report(analyses: List[Dict], out_path: Path):
    out = []
    out.append("# Minimal-isolation diagnostic — why is it below chance?")
    out.append("")
    out.append("**Date:** 2026-05-04")
    out.append("**Source:** research/findings/raw/g11_bg/text_eval_minimal_iso_seed{42,43,44}.json")
    out.append("**Question:** Minimal-iso (cascade-stripped) gave 16.7% mean — "
               "below 25% chance. Why?")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## TL;DR")
    out.append("")

    # Compute headline patterns
    seeds = [a["seed"] for a in analyses]
    true_accs = [a["true_acc"] for a in analyses]
    best_accs = [a["best_acc"] for a in analyses]
    ns_inversions = sum(1 for a in analyses if a["ns_inversion"])
    e_sinks = sum(1 for a in analyses if a["e_sink_count"] >= 2)

    out.append(f"- TRUE accuracy: {true_accs} (mean {sum(true_accs)/len(true_accs):.1%})")
    out.append(f"- Best permutation accuracy: {best_accs} (mean {sum(best_accs)/len(best_accs):.1%})")
    out.append(f"- North-South inversion present in {ns_inversions}/{len(seeds)} seeds")
    out.append(f"  (north->S rate > 30% AND south->N rate > 30%)")
    out.append(f"- E-as-sink (>= 2 words mapping to E): {e_sinks}/{len(seeds)} seeds")
    out.append("")

    # Are best permutations CONSISTENT across seeds?
    best_maps = [a["best_map_str"] for a in analyses]
    if len(set(best_maps)) == 1:
        out.append(f"**Best permutation is IDENTICAL across all seeds: "
                   f"`{best_maps[0]}`** — this would be a strong systematic")
        out.append("signal of architecture bias.")
    else:
        out.append(f"**Best permutations vary by seed** — random architecture noise:")
        for a in analyses:
            out.append(f"  - seed {a['seed']}: `{a['best_map_str']}` "
                       f"(acc {a['best_acc']:.1%})")
    out.append("")

    # Action distribution
    out.append("## Action distribution (motor pool firing rate, all words combined)")
    out.append("")
    out.append("If the architecture has structural bias toward one motor pool, "
               "this will be skewed away from 25% per pool.")
    out.append("")
    out.append("| seed | N | E | S | W |")
    out.append("|---|---|---|---|---|")
    for a in analyses:
        ad = a["action_distribution"]
        row = [f"{ad[x]:.1%}" for x in ACTIONS]
        out.append(f"| {a['seed']} | " + " | ".join(row) + " |")
    out.append("")

    # Per-seed details
    out.append("## Per-seed details")
    out.append("")
    for a in analyses:
        out.append(f"### Seed {a['seed']}")
        out.append("")
        out.append(f"**TRUE-mapping accuracy:** {a['true_acc']:.1%}")
        out.append("")
        out.append(f"Per-word TRUE accuracy:")
        for w in WORDS:
            out.append(f"- {w} -> {TRUE_MAP[w]}: {a['per_word_true'][w]:.1%}")
        out.append("")
        out.append(f"**Best permutation:** `{a['best_map_str']}` "
                   f"(accuracy {a['best_acc']:.1%})")
        out.append("")
        out.append(f"Top 5 permutations:")
        for acc, mp in a["top_5_perms"]:
            out.append(f"- `{mp}`: {acc:.1%}")
        out.append("")
        out.append(f"Confusion matrix:")
        out.append("")
        out.append(_format_confusion(a["confusion"]))
        out.append("")
        out.append(f"NS-inversion: north->S {a['n_to_s_rate']:.1%}, "
                   f"south->N {a['s_to_n_rate']:.1%}")
        out.append("")

    # Interpretation
    out.append("## Interpretation")
    out.append("")
    out.append("### Why below chance?")
    out.append("")
    out.append("Pure random would give 25% TRUE accuracy in expectation. "
               "Below-chance means the network is making CORRELATED wrong "
               "answers — it's actively picking the wrong motor pool for at "
               "least some words. The mechanism could be:")
    out.append("")
    out.append("1. **Reward eligibility window mismatch**: paired-stim training "
               "reinforces (lang_active, motor_target) pairs, but if the eval "
               "drives lang_active and the WINNER motor pool fires AFTER the "
               "intended one, eligibility might consolidate the wrong pair.")
    out.append("")
    out.append("2. **Lateral inhibition asymmetry** (architecture has none, but "
               "reset windows might differ across motor pools).")
    out.append("")
    out.append("3. **Sparse code overlap**: if 'north' and 'south' patterns "
               "share many active neurons, training on 'north' partially "
               "decreases weights from those overlapping neurons to motor_S "
               "(via STDP LTD on uncorrelated firing).")
    out.append("")
    out.append("### Cross-seed pattern stability")
    out.append("")
    if len(set(best_maps)) == 1:
        out.append("Same best permutation across seeds = ARCHITECTURE-LEVEL "
                   "structural bias. The bias persists regardless of init "
                   "randomness. Likely culprits: vocab_to_drive_pattern "
                   "deterministic per word + shared CSR initialization seed "
                   "for connectivity (which IS seed-dependent but the same "
                   "active-neuron set per word means the same biased "
                   "co-firing happens).")
    elif len(set(best_maps)) >= 2:
        out.append("Best permutations differ by seed = SEED-DEPENDENT noise "
                   "with weak structural alignment. Each random init creates "
                   "its own private bias. This pattern is consistent with the "
                   "permuted-label control test result (0/45 prior runs had "
                   "TRUE labels as the best of 24 permutations).")
    out.append("")

    # What this means for the biology sweep
    out.append("## Implication for biology sweep")
    out.append("")
    out.append("The biology sweep (in flight) tests three fixes on this same "
               "minimal architecture:")
    out.append("")
    out.append("- **+FS only** (motor PV-FS lateral inhibition): if the issue is "
               "pure WTA selection, this should help. Predicts: aligned ratio "
               "moves from 0/3 toward 4/6+.")
    out.append("")
    out.append("- **+Topo only** (Wernicke->motor topographic prior): if the "
               "issue is sparse-code-overlap-induced LTD on the right pairs, "
               "the topographic prior gives STDP a head start with correct "
               "weights. Predicts: aligned ratio moves toward 4/6+.")
    out.append("")
    out.append("- **+Topo +FS** (combined): if BOTH are needed.")
    out.append("")
    out.append("If the NS-inversion pattern is structural (same best perm "
               "across seeds), the topographic prior should specifically fix "
               "it because the prior tells the network 'north -> motor_N' as "
               "starting weights.")
    out.append("")
    out.append("If the pattern is seed-dependent random noise, then biology "
               "fixes might still help by providing systematic structure that "
               "STDP can refine, even if the underlying issue is overlap-driven.")
    out.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path,
                    default=Path("research/findings/raw/g11_bg"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path,
                    default=Path("research/findings/2026-05-04-minimal-iso-diagnostic.md"))
    args = ap.parse_args()

    analyses = []
    for seed in args.seeds:
        a = _analyze_seed(seed, args.raw_dir)
        if a is not None:
            analyses.append(a)
        else:
            print(f"WARNING: no data for seed {seed}", flush=True)

    if not analyses:
        print("No data to analyze. Exiting.", flush=True)
        return

    _write_report(analyses, args.out)

    # Also print a TL;DR to stdout
    print("=" * 60)
    print("MINIMAL-ISO DIAGNOSTIC TL;DR")
    print("=" * 60)
    true_accs = [a["true_acc"] for a in analyses]
    best_accs = [a["best_acc"] for a in analyses]
    print(f"TRUE accuracy: {[f'{x:.1%}' for x in true_accs]} "
          f"(mean {sum(true_accs)/len(true_accs):.1%})")
    print(f"Best perm accuracy: {[f'{x:.1%}' for x in best_accs]} "
          f"(mean {sum(best_accs)/len(best_accs):.1%})")
    best_maps = [a["best_map_str"] for a in analyses]
    if len(set(best_maps)) == 1:
        print(f"Best perm IDENTICAL across seeds: {best_maps[0]} "
              f"(architecture bias)")
    else:
        print(f"Best perms vary across seeds (seed-dependent noise):")
        for a in analyses:
            print(f"  seed {a['seed']}: {a['best_map_str']}")
    ns_count = sum(1 for a in analyses if a["ns_inversion"])
    print(f"NS-inversion (north->S AND south->N > 30%): "
          f"{ns_count}/{len(analyses)} seeds")
    print("=" * 60)


if __name__ == "__main__":
    main()
