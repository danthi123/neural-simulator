#!/usr/bin/env python
"""Neural reward+dopamine nav de-risk analysis (2026-06-08).

Compares the FULLY-BIOLOGIZED reward+dopamine (N5 coordinate-free perceived-approach
reward + the SPIKING-SNc actor-critic dopamine, Stage A) against the cheat baseline
(coordinate Manhattan reward + raw-scalar dopamine) in the biologized flagship
multi-goal config. This is the FIRST full-nav test of --spiking-snc.

ACCEPTANCE: the neural reward+DA does NOT regress the nav score. The win is that the
dopamine RPE is computed by SNc neurons FIRING (brain-based), not a host scalar — at
no nav cost. A regression IS a reportable finding (the measured cost of brain-basing
the dopamine), NOT something to hide.

Metric = sum_finalQ = sum over phases of phase_stats[*].final_quarter_mean_distance
(the project's flagship multi-goal score; LOWER better).
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = sorted({int(m.group(1)) for f in glob.glob(os.path.join(HERE, "_biorda_neural_s*.json"))
                if (m := re.search(r"_biorda_neural_s(\d+)\.json$", f))})


def sum_finalq(path):
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ps = d.get("phase_stats") or []
    vals = [p.get("final_quarter_mean_distance") for p in ps
            if p.get("final_quarter_mean_distance") is not None]
    return sum(float(v) for v in vals) if vals else None


def main():
    print("\nNeural reward+DA de-risk — N5 perceived-reward + spiking-SNc vs cheat (coord reward + raw DA)")
    print("  sum_finalQ = sum of per-phase final-quarter mean distance (LOWER better); flagship multi-goal\n")
    neur, cheat, deltas = [], [], []
    for s in SEEDS:
        n = sum_finalq(os.path.join(HERE, f"_biorda_neural_s{s}.json"))
        c = sum_finalq(os.path.join(HERE, f"_biorda_cheat_s{s}.json"))
        if n is None or c is None:
            print(f"  seed {s:>4}: neural {('--' if n is None else f'{n:.2f}')}  "
                  f"cheat {('--' if c is None else f'{c:.2f}')}   (incomplete)")
            continue
        neur.append(n); cheat.append(c); d = n - c; deltas.append(d)
        tag = "no-regress" if d <= 0.5 else ("~parity" if d <= 1.5 else "REGRESSION")
        print(f"  seed {s:>4}: neural {n:5.2f}  cheat {c:5.2f}  Δ(neural-cheat) {d:+5.2f}   {tag}")
    if neur:
        mn = sum(neur) / len(neur); mc = sum(cheat) / len(cheat); md = sum(deltas) / len(deltas)
        n_ok = sum(1 for d in deltas if d <= 0.5)
        print(f"\n  mean neural {mn:.2f}  |  mean cheat {mc:.2f}  |  mean Δ {md:+.2f}  over {len(neur)} seeds")
        print(f"  {n_ok}/{len(deltas)} seeds no-regression (Δ ≤ 0.5)")
        if md <= 0.5:
            print("\n  VERDICT: NO REGRESSION — the spiking-SNc neural dopamine + coord-free perceived reward\n"
                  "  navigate as well as the cheat. The dopamine RPE is now computed by SNc neurons FIRING\n"
                  "  (brain-based), and the reward is coordinate-free — at no nav cost. Spiking-SNc Stage A GO in full nav.")
        else:
            print("\n  VERDICT: MEASURABLE COST — brain-basing the dopamine costs Δ nav score. Report HONESTLY\n"
                  "  (this is the measured price of the neural realization; diagnose: tonic/gain tuning vs a real limit).")
    else:
        print("\n(awaiting seeds.)")


if __name__ == "__main__":
    main()
