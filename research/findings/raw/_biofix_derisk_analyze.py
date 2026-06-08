#!/usr/bin/env python
"""POST-FIX de-risk re-run analysis (2026-06-08). Same as _biorda_derisk_analyze
but reads the _biofix_* outputs (re-run from the fixed bridge: the per-synapse
gate arrays cp_d1_d2_sign/cp_transmission_gain/cp_plasticity_rate_gain no longer
under-run, so reward-modulated plasticity actually applies). Verdict question:
with plasticity working, does the brain-based reward+dopamine (N5 + spiking-SNc)
still regress vs the cheat, or was the entire ~6x gap the silent bug?
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = sorted({int(m.group(1)) for f in glob.glob(os.path.join(HERE, "_biofix_neural_s*.json"))
                if (m := re.search(r"_biofix_neural_s(\d+)\.json$", f))})


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
    print("\nPOST-FIX de-risk — N5 perceived-reward + spiking-SNc vs cheat (reward-modulated plasticity now WORKS)")
    print("  sum_finalQ = sum of per-phase final-quarter mean distance (LOWER better); flagship multi-goal")
    print("  PRE-FIX (BUGGED): neural 23.15 vs cheat 3.83 (Delta +19.32) — confounded by the silent crash\n")
    neur, cheat, deltas = [], [], []
    for s in SEEDS:
        n = sum_finalq(os.path.join(HERE, f"_biofix_neural_s{s}.json"))
        c = sum_finalq(os.path.join(HERE, f"_biofix_cheat_s{s}.json"))
        if n is None or c is None:
            print(f"  seed {s:>4}: neural {('--' if n is None else f'{n:.2f}')}  "
                  f"cheat {('--' if c is None else f'{c:.2f}')}   (incomplete)")
            continue
        neur.append(n); cheat.append(c); d = n - c; deltas.append(d)
        tag = "no-regress" if d <= 0.5 else ("~parity" if d <= 1.5 else "REGRESSION")
        print(f"  seed {s:>4}: neural {n:5.2f}  cheat {c:5.2f}  Delta(neural-cheat) {d:+5.2f}   {tag}")
    if neur:
        mn = sum(neur) / len(neur); mc = sum(cheat) / len(cheat); md = sum(deltas) / len(deltas)
        n_ok = sum(1 for d in deltas if d <= 0.5)
        print(f"\n  mean neural {mn:.2f}  |  mean cheat {mc:.2f}  |  mean Delta {md:+.2f}  over {len(neur)} seeds")
        print(f"  {n_ok}/{len(deltas)} seeds no-regression (Delta <= 0.5)")
        print(f"\n  cheat: PRE-FIX 3.83 -> POST-FIX {mc:.2f} (flagship re-validation — should be <= 3.83 / improved)")
        print(f"  neural: PRE-FIX 23.15 -> POST-FIX {mn:.2f}")
        if md <= 0.5:
            print("\n  VERDICT: the ~6x 'regression' was the BUG. With plasticity working, brain-based")
            print("  reward+dopamine (SNc firing the RPE) navigates ON PAR with the host shortcut. Spiking-SNc Stage A GO.")
        elif md <= 3.0:
            print("\n  VERDICT: the bug explained MOST of the gap; a smaller REAL cost remains (the honest")
            print("  brain-based-vs-shortcut delta). Report the true number; diagnose the residual (tonic/gain).")
        else:
            print("\n  VERDICT: a large gap PERSISTS post-fix — now a genuine result (not the bug). Diagnose")
            print("  (isolation N5 vs SNc, gain sweep) before concluding substrate limit.")
    else:
        print("\n(awaiting seeds.)")


if __name__ == "__main__":
    main()
