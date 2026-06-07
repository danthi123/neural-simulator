#!/usr/bin/env python
"""Analyze the Rank 2 de-risk (does the learned-from-vision circuit hold post-wean?).

Single-goal, reflex weaned @2000 (full-off 3000). Post-wean metric = last-quarter
mean distance (steps 4500-6000), i.e. the durable heuristic-/reflex-free performance.
HOLD ~1-2 = the learned circuit navigates self-sufficiently; COLLAPSE ~5-6 = it didn't
consolidate.

  R2   = reflex teaches the LEARNED-from-vision sensory->cortex, then weans.
  CTRL = reflex teaches IT-only (position-INVARIANT), then weans (N1-scaffold-fragile).

GO       — R2 holds (post-wean <= ~2.5) AND beats CTRL: the position-preserving
           learned circuit consolidates where IT could not. -> multi-seed.
BOUNDARY — R2 collapses (>= ~4): the learned-from-vision circuit didn't consolidate
           in this budget; characterize (denser teaching / different code).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def postwean(fn):
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ps = d.get("phase_stats") or []
    if ps and ps[-1].get("final_quarter_mean_distance") is not None:
        return float(ps[-1]["final_quarter_mean_distance"])
    dl = d.get("distance_log") or []
    if not dl:
        return None
    tail = dl[len(dl) * 3 // 4:]
    return float(sum(tail) / len(tail)) if tail else None


def main():
    r2 = postwean("_rank2_R2_s42.json")
    ctrl = postwean("_rank2_CTRL_itonly_s42.json")
    print("\nRank 2 de-risk — seed 42, single-goal, reflex weaned @2000 (post-wean = last-quarter mean dist, LOWER better)\n")
    print(f"  R2   (learned-from-vision, weaned) post-wean = {('%.2f'%r2) if r2 is not None else 'NOT-DONE'}")
    print(f"  CTRL (IT-only, weaned)             post-wean = {('%.2f'%ctrl) if ctrl is not None else 'NOT-DONE'}")
    if r2 is None or ctrl is None:
        print("\n(awaiting both conditions; re-run when the de-risk completes.)")
        return
    print(f"\n  R2={r2:.2f}  CTRL={ctrl:.2f}")
    if r2 <= 2.5 and r2 < ctrl:
        print("\nVERDICT: GO (seed 42) — the learned-from-vision circuit HOLDS post-wean and beats "
              "IT-only. The position-preserving learned read-out consolidates where IT could not. "
              "-> multi-seed (43,44 then 6).")
    elif r2 >= 4.0:
        print("\nVERDICT: BOUNDARY (seed 42) — R2 collapses post-wean; the learned-from-vision circuit "
              "didn't consolidate in this budget. Inspect: denser teaching / longer wean / code.")
    else:
        print("\nVERDICT: MARGINAL (seed 42) — R2 between hold and floor; inspect trajectory + multi-seed.")


if __name__ == "__main__":
    main()
