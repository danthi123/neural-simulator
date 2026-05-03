"""
SWR investigation — decision script. Run after H1 + H4 land to figure
out what to do next.

Reads the aggregated results, applies the decision tree from
docs/plans/2026-05-03-autonomous-overnight-plan.md, and prints a
markdown recommendation.

Decision tree:
  IF H4 isolation gives 80%+:
    The cascade interferes during full training; PFC bypass is
    capable in isolation. Pivot: REVERSE CURRICULUM.
    -> design + run text_train_curriculum --reverse-curriculum
  ELIF H4 isolation gives 50-79%:
    Moderate. H1 outcome matters more.
    IF H1 rescues W->A to ~baseline:
      The replay-distribution bias was the issue; balanced replay fixes it.
      Pivot: ship as default, move on.
    ELSE:
      Multiple compounding issues. Investigate deeper.
  ELIF H4 isolation gives ~28%:
    Architecture itself limits W->A. Pivot: STRUCTURAL CHANGES.
    -> larger language regions (256->512), larger motor pools
       (10->50), different motor readout (population vector
       instead of argmax).

This script doesn't take action — it prints what to do. Human-in-loop
for the actual launch decisions when there's enough time pressure to
matter; auto-launch for things that can run unattended.
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "research" / "findings" / "raw" / "g11_bg"

CONDITIONS = {
    "v2 baseline":         "text_eval_R3R6_100ep_HebOff_v2_seed{seed}.json",
    "v2 + SWR (default)":  "text_eval_v2_swr500_seed{seed}.json",
    "v2 + SWR balanced":   "text_eval_h1_balanced_seed{seed}.json",
    "PFC isolation (H4)":  "text_eval_h4_isolation_seed{seed}.json",
}

SEEDS = [42, 43, 44, 100, 101, 102]


def load_seeds(pattern: str) -> list:
    out = []
    for s in SEEDS:
        p = RAW_DIR / pattern.format(seed=s)
        if p.exists():
            try:
                d = json.loads(p.read_text())
                wa = (d.get("word_to_action_eval") or {}).get("accuracy")
                if wa is not None:
                    out.append((s, wa))
            except Exception:
                pass
    return out


def stat(vals):
    if not vals: return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0
    return m, s


def main():
    print("# SWR investigation — decision recommendation")
    print()
    results = {}
    for label, pat in CONDITIONS.items():
        runs = load_seeds(pat)
        n = len(runs)
        if n == 0:
            results[label] = {"n": 0}
            continue
        accs = [r[1] for r in runs]
        m, s = stat(accs)
        results[label] = {"n": n, "mean": m, "std": s, "seeds": [r[0] for r in runs]}
    # Print state
    print("## Current state")
    for label, r in results.items():
        if r["n"] == 0:
            print(f"  - {label}: NO DATA")
        else:
            print(f"  - {label}: n={r['n']}, W→A {100*r['mean']:.1f}% ± {100*r['std']:.1f}% (seeds {r['seeds']})")
    print()

    # Apply decision tree
    print("## Recommendation")
    base = results.get("v2 baseline", {})
    swr = results.get("v2 + SWR (default)", {})
    h1 = results.get("v2 + SWR balanced", {})
    h4 = results.get("PFC isolation (H4)", {})

    if h4.get("n", 0) >= 6:
        h4_mean = h4["mean"]
        if h4_mean >= 0.80:
            print("**H4 → reverse curriculum.**")
            print()
            print(f"PFC bypass isolation gives {100*h4_mean:.1f}% W→A — well above the 28.5% "
                  "v2 baseline. The architecture supports the language→motor mapping; "
                  "cascade interference is what's limiting v2.")
            print()
            print("Pivot: implement reverse curriculum. Train language pathway in "
                  "isolation FIRST (Phase 0), THEN unfreeze cascade (Phase 1+2).")
            print("This should retain H4's high W→A while still letting the cascade "
                  "learn navigation.")
        elif h4_mean >= 0.50:
            print("**H4 moderate → H1 outcome decides.**")
            print()
            print(f"PFC bypass isolation gives {100*h4_mean:.1f}% — better than baseline but not "
                  "approaching the upper bound we'd hope for. The architecture has "
                  "limited capacity even without cascade interference.")
            print()
            if h1.get("n", 0) >= 6:
                h1_mean = h1["mean"]
                if h1_mean >= 0.27:
                    print("H1 balanced replay achieves ~baseline. The replay-distribution-"
                          "bias mechanism is the issue. Ship balanced replay as default.")
                else:
                    print(f"H1 balanced replay gives {100*h1_mean:.1f}% — doesn't rescue. "
                          "Multiple compounding issues. Investigate deeper.")
            else:
                print("Wait for H1 to complete before deciding.")
        else:
            print("**H4 low → architectural change.**")
            print()
            print(f"PFC bypass isolation gives {100*h4_mean:.1f}% — at or below baseline. The "
                  "architecture itself caps W→A. Pivot to structural changes:")
            print("- Increase language_input from 256 to 512 neurons")
            print("- Increase motor_X pool size from 10 to 50 neurons")
            print("- Replace argmax motor readout with population-vector decoding")
    else:
        print(f"H4 incomplete (n={h4.get('n', 0)}/6). Wait for full eval.")
    print()


if __name__ == "__main__":
    main()
