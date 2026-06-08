#!/usr/bin/env python
"""Rank 2 GENERALIZE2 analysis (2026-06-08): does the learned-from-vision
(dx,dy)->action map navigate to goals it was NEVER trained on, reflex OFF?

generalize2 schedule (n-steps 6000, wean @2000-3000):
  phase 0 (0-700)    goal (6,6) far          TRAIN (reflex on)
  phase 1 (700-1400) goal (1,6) far_west     TRAIN
  phase 2 (1400-2100)goal (1,1) sw           TRAIN (wean begins @2000)
  phase 3 (2100-3000)goal (6,1) far_se       TRAIN (wean completes @3000)
  phase 4 (3000-4000)goal (4,6) mid_top      TEST  NEW non-corner, reflex OFF
  phase 5 (4000-5000)goal (1,4) mid_left     TEST  NEW non-corner, reflex OFF
  phase 6 (5000-6000)goal (6,4) mid_right    TEST  NEW non-corner, reflex OFF

Generalization metric = mean of phases 4-6 final_quarter_mean_distance
(the pure learned circuit on never-trained goals). LOWER better.
Context: reflex single-goal precision ceiling ~2.0; position-invariant
IT-only floor ~6.1 (== cannot navigate, near random walk).
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = sorted({int(m.group(1)) for f in glob.glob(os.path.join(HERE, "_rank2_generalize2_s*.json"))
                if (m := re.search(r"_rank2_generalize2_s(\d+)\.json$", f))})

TRAIN_PHASES = (0, 1, 2, 3)
TEST_PHASES = (4, 5, 6)
REFLEX_CEILING = 2.0
IT_FLOOR = 6.1


def load(s):
    p = os.path.join(HERE, f"_rank2_generalize2_s{s}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fq(ps, i):
    if i < len(ps) and ps[i].get("final_quarter_mean_distance") is not None:
        return float(ps[i]["final_quarter_mean_distance"])
    return None


def main():
    print("\nRank 2 GENERALIZE2 — learned-from-vision map on NEVER-TRAINED goals (reflex OFF)\n")
    print(f"  reflex precision ceiling ~{REFLEX_CEILING} | position-invariant IT floor ~{IT_FLOOR} (== random walk)\n")
    test_means = []
    for s in SEEDS:
        d = load(s)
        if d is None:
            print(f"  seed {s:>4}: NOT-DONE")
            continue
        ps = d.get("phase_stats") or []
        train = [fq(ps, i) for i in TRAIN_PHASES]
        test = [fq(ps, i) for i in TEST_PHASES]
        if any(v is None for v in test):
            print(f"  seed {s:>4}: incomplete phases")
            continue
        tr_mean = sum(v for v in train if v is not None) / max(1, sum(1 for v in train if v is not None))
        te_mean = sum(test) / len(test)
        test_means.append(te_mean)
        # how far between IT floor and reflex ceiling did the NEW-goal nav land? 1.0 = at reflex, 0.0 = at IT floor
        frac = (IT_FLOOR - te_mean) / (IT_FLOOR - REFLEX_CEILING)
        verdict = "GENERALIZES" if te_mean <= 4.5 else ("partial" if te_mean <= 5.4 else "FAILS (~IT floor)")
        print(f"  seed {s:>4}: train(0-3) {tr_mean:.2f} | NEW-goal(4-6) {te_mean:.2f}  "
              f"[{test[0]:.2f}/{test[1]:.2f}/{test[2]:.2f}]  {frac*100:.0f}% toward reflex  -> {verdict}")
    if test_means:
        mean = sum(test_means) / len(test_means)
        std = (sum((v - mean) ** 2 for v in test_means) / len(test_means)) ** 0.5
        n_gen = sum(1 for v in test_means if v <= 4.5)
        print(f"\n  NEW-goal mean = {mean:.2f} +/- {std:.2f}  over {len(test_means)} seeds; "
              f"{n_gen}/{len(test_means)} generalize (<=4.5)")
        frac = (IT_FLOOR - mean) / (IT_FLOOR - REFLEX_CEILING)
        print(f"  {frac*100:.0f}% of the way from IT-floor (cannot-navigate) to reflex-precision, on goals NEVER trained.")
        if n_gen == len(test_means):
            print("\n  VERDICT: the learned-from-vision (dx,dy)->action circuit GENERALIZES to never-trained\n"
                  "  goals with the teaching reflex OFF. Position-preserving dorsal code confirmed: the map\n"
                  "  is goal-agnostic (offset->action), not a memorized per-goal policy.")
        else:
            print("\n  VERDICT: PARTIAL/seed-dependent generalization (honest negative on the failing seeds).")
    else:
        print("\n(awaiting seeds.)")


if __name__ == "__main__":
    main()
