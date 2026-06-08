#!/usr/bin/env python
"""Plain Rank 2 (learned-from-vision, NO teacher) multi-seed analysis.

The REAL Rank 2 result (the supervised teacher was a seed-42 artifact + counterproductive
on good seeds). Post-wean = last-quarter mean distance (single-goal, reflex weaned @2000).
LOWER better; reflex single-goal ~2.0 is the precision ceiling; IT-only floor ~6.1.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = sorted({int(m.group(1)) for f in glob.glob(os.path.join(HERE, "_rank2_R2_s*.json"))
                if (m := re.search(r"_rank2_R2_s(\d+)\.json$", f))})


def postwean(s):
    p = os.path.join(HERE, f"_rank2_R2_s{s}.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ps = d.get("phase_stats") or []
    if ps and ps[-1].get("final_quarter_mean_distance") is not None:
        return float(ps[-1]["final_quarter_mean_distance"])
    return None


def main():
    print("\nPlain Rank 2 (learned-from-vision, no teacher) — post-wean last-quarter (LOWER better)\n")
    print(f"  reflex single-goal ceiling ~2.0 | IT-only floor ~6.1\n")
    vals = []
    for s in SEEDS:
        pw = postwean(s)
        if pw is None:
            print(f"  seed {s:>4}: NOT-DONE")
            continue
        vals.append(pw)
        tag = "near-reflex" if pw <= 2.7 else ("outlier" if pw >= 3.5 else "mid")
        print(f"  seed {s:>4}: {pw:.2f}   ({tag})")
    if vals:
        mean = sum(vals) / len(vals)
        n_near = sum(1 for v in vals if v <= 2.7)
        print(f"\n  mean = {mean:.2f}  over {len(vals)} seeds; {n_near}/{len(vals)} near-reflex (<=2.7)")
        print("\n  Plain R2 = the durable learned-from-vision circuit. Near-reflex on most seeds = "
              "a strong perception-biologization (the learned where->action map consolidates from "
              "vision, self-sufficient post-wean, where the position-invariant IT->cortex collapsed).")
    else:
        print("\n(awaiting seeds.)")


if __name__ == "__main__":
    main()
