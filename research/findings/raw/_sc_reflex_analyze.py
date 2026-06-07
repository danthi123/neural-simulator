#!/usr/bin/env python
"""Analyze the SC-orienting reflex de-risk smoke (seed 42, grid-8, multi-goal).

Reads the 3 conditions and reports the cheat-5 score (sum over phases of
final_quarter_mean_distance; LOWER = better navigation):
  A = SC reflex, heuristic OFF (the test)
  B = heuristic ON              (the cheat baseline; ~4 expected)
  C = floor, both OFF           (visual cortex alone; ~18-22 expected)

Verdict heuristic:
  GO       — A navigates: A is much closer to B than to C
             (A <= midpoint(B,C) AND A <= ~2x B). The SC reflex drives
             navigation from vision (no coords); proceed to multi-seed + Rank 2.
  BOUNDARY — A sits at the floor (A >= ~0.7 * C): the rendered salience is too
             coarse / the reflex can't drive a clean cardinal. Cheap pivot.
  MIXED    — in between; inspect per-phase + run multi-seed before deciding.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = [
    ("A  SC reflex (heur OFF)", "_sc_reflex_A_s42.json"),
    ("B  heuristic ON (cheat)", "_sc_reflex_B_heuron_s42.json"),
    ("C  floor (both OFF)", "_sc_reflex_C_floor_s42.json"),
]


def cheat5_score(path):
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ps = d.get("phase_stats") or []
    vals = [p.get("final_quarter_mean_distance") for p in ps
            if isinstance(p, dict) and p.get("final_quarter_mean_distance") is not None]
    if not vals:
        return None
    return sum(vals), len(vals), vals


def main():
    scores = {}
    print("\nSC-orienting reflex de-risk — seed 42, grid-8, multi-goal (cheat-5 sum-finalQ, LOWER better)\n")
    for label, fn in CONDS:
        s = cheat5_score(os.path.join(HERE, fn))
        scores[label[0]] = s[0] if s else None
        if s:
            per = " ".join(f"{v:.2f}" for v in s[2])
            print(f"  {label:26} sum-finalQ = {s[0]:6.2f}   per-phase [{per}]")
        else:
            print(f"  {label:26} NOT-DONE")

    A, B, C = scores.get("A"), scores.get("B"), scores.get("C")
    if None in (A, B, C):
        print("\n(awaiting all 3 conditions; re-run when the smoke completes.)")
        return
    mid = 0.5 * (B + C)
    print(f"\n  A={A:.2f}  B={B:.2f}  C={C:.2f}   midpoint(B,C)={mid:.2f}")
    if A <= mid and A <= 2.0 * B:
        verdict = ("GO — the SC reflex navigates from vision (no coords); A is near the "
                   "heuristic baseline, well below the floor. Proceed to multi-seed + Rank 2.")
    elif A >= 0.7 * C:
        verdict = ("BOUNDARY — A sits at the floor; the rendered salience is too coarse / the "
                   "reflex can't drive a clean cardinal. Cheap pivot (N2 render fidelity / sharper salience).")
    else:
        verdict = "MIXED — between baselines; inspect per-phase and run multi-seed before deciding."
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
