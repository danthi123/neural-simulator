#!/usr/bin/env python
"""Multi-seed analysis of the SC-orienting reflex de-risk (grid-8, multi-goal).

Condition A = SC reflex, heuristic OFF (the test); C = floor, both OFF.
B = heuristic ON (cheat) reference at seed 42. Score = cheat-5 sum-finalQ
(sum over phases of final_quarter_mean_distance; LOWER = better).

GO (multi-seed): A navigates robustly across seeds — every seed's A is well
below its floor C (A <= ~0.5 * C) and near the cheat baseline. The agent
orients to the goal from VISION (no coordinates) via an innate collicular
reflex. -> write the finding, extend to 6 seeds, then Rank 2 (dorsal/PPC
learned read-out + transmission_gate wean).
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
# Auto-detect seeds from the A-condition files present (handles 3- or 6-seed).
SEEDS = sorted({int(m.group(1)) for f in glob.glob(os.path.join(HERE, "_sc_reflex_A_s*.json"))
                if (m := re.search(r"_sc_reflex_A_s(\d+)\.json$", f))})
if not SEEDS:
    SEEDS = [42, 43, 44]


def score(fn):
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ps = d.get("phase_stats") or []
    vals = [p.get("final_quarter_mean_distance") for p in ps
            if isinstance(p, dict) and p.get("final_quarter_mean_distance") is not None]
    return sum(vals) if vals else None


def main():
    print("\nSC-orienting reflex de-risk — multi-seed (cheat-5 sum-finalQ, LOWER better)\n")
    b42 = score("_sc_reflex_B_heuron_s42.json")
    print(f"  reference  B (heuristic cheat, seed 42) = {b42:.2f}\n" if b42 else "  reference  B = NOT-DONE\n")
    print(f"  {'seed':>4}  {'A (SC reflex)':>14}  {'C (floor)':>10}  {'A/C':>6}  verdict")
    print("  " + "-" * 52)
    a_vals, n_go = [], 0
    for s in SEEDS:
        a = score(f"_sc_reflex_A_s{s}.json")
        c = score(f"_sc_reflex_C_floor_s{s}.json")
        if a is None or c is None:
            print(f"  {s:>4}  {('%.2f'%a) if a else '—':>14}  {('%.2f'%c) if c else '—':>10}  {'—':>6}  PENDING")
            continue
        ratio = a / c if c else float("inf")
        navigates = a <= 0.5 * c
        n_go += navigates
        a_vals.append(a)
        print(f"  {s:>4}  {a:>14.2f}  {c:>10.2f}  {ratio:>6.2f}  {'NAVIGATES' if navigates else 'AT-FLOOR'}")
    print("  " + "-" * 52)
    if len(a_vals) == len(SEEDS):
        mean_a = sum(a_vals) / len(a_vals)
        print(f"\n  mean A = {mean_a:.2f}   (cheat ref B-42 = {b42:.2f})   {n_go}/{len(SEEDS)} seeds navigate")
        if n_go == len(SEEDS):
            print("\nVERDICT: GO multi-seed — the SC orienting reflex navigates from VISION "
                  "(no coords) on every seed, ~near the cheat and far below the floor. The "
                  "perceptual cold-start is broken biologically. -> finding + 6-seed + Rank 2.")
        else:
            print(f"\nVERDICT: PARTIAL — {n_go}/{len(SEEDS)} navigate; inspect the at-floor seed(s).")
    else:
        print("\n(awaiting seeds; re-run when the multi-seed batch completes.)")


if __name__ == "__main__":
    main()
