"""Session D: redesigned gate metrics for sensorimotor learning.

The strict Q1->Q4 >= 1.5 and P1 finalQ < 3.0 gates used in G6/G7/G8/G9 turned
out to misfire repeatedly:
  - G6: agent converges in <100 steps so Q1 is already near Q4 -- strict
    improvement delta can't fire even though learning is real.
  - G7/G8/G9/SessionC: 300-step phase-2 budget is too tight for biological
    weight-learning timescales.

This module implements two metrics that are actually what biology produces:

  1. time_to_proficiency (TTP):
     "First step at which fraction-of-steps-within-dist-D-of-goal exceeds
      threshold, over a sliding window."
     This is a *rate of acquisition* metric. If TTP is small, learning is
     fast. If it's None, learning hasn't happened. No fragile delta
     arithmetic on quartiles.

  2. random_start_generalization (RSG):
     "After training on (start, goal), freeze weights, test from M random
      start positions; report mean end-of-eval-episode distance."
     This separates 'learned a controller' from 'memorized one trajectory'.
     A lookup table cannot generalize. A real controller can.

  3. proficiency_fraction (PF):
     "Fraction of steps within dist-D of goal, over a given interval."
     Simple scalar summary. Random baseline on 8x8 grid with dist<=2 target
     is ~0.25-0.30 (by area of the near-goal zone over full grid).

These can be computed retrospectively from existing `distance_log` arrays,
and RSG requires a small additional evaluator runner (see gate_metrics_rsg.py).

Biology anchors:
  - Skinner-box rat task: proficiency in 50-200 trials typical (Staddon 2003).
    Translates to TTP ~ tens-of-trials on our setup.
  - Rodent Morris water maze: generalization across start positions after
    ~20 training trials (Vorhees & Williams 2006). We can test this
    directly via RSG.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


def time_to_proficiency(
    distance_log: list | np.ndarray,
    proficiency_dist: int = 2,
    window_size: int = 50,
    threshold: float = 0.5,
    start_step: int = 0,
) -> Optional[int]:
    """Return first step at which the sliding-window fraction-of-steps-
    within `proficiency_dist` of goal exceeds `threshold`.

    Args:
        distance_log: per-step Manhattan distance array (length N).
        proficiency_dist: the "near-goal" radius. 2 = dist<=2 counts as near.
        window_size: sliding window in steps.
        threshold: fraction required within window (0.5 = half the time).
        start_step: ignore everything before this step (for per-phase TTP).

    Returns:
        int step index (0-based) at which threshold is first met, or None
        if never met within the window coverage.
    """
    arr = np.asarray(distance_log)
    if len(arr) <= window_size:
        return None

    near = (arr <= proficiency_dist).astype(np.float32)
    # Sliding-window mean via cumulative sum
    cumsum = np.cumsum(near)
    # window_mean[t] = mean(near[t-window_size+1 : t+1]) for t >= window_size-1
    window_means = (cumsum[window_size - 1 :] - np.concatenate([[0], cumsum[: -window_size]])) / window_size

    # Map back to actual step index (step = t)
    for t_local, m in enumerate(window_means):
        step = t_local + window_size - 1  # step at end of the window
        if step < start_step:
            continue
        if m >= threshold:
            return step
    return None


def proficiency_fraction(
    distance_log: list | np.ndarray,
    proficiency_dist: int = 2,
    start_step: int = 0,
    end_step: Optional[int] = None,
) -> float:
    """Fraction of steps in [start_step, end_step) at which dist <= proficiency_dist."""
    arr = np.asarray(distance_log)
    if end_step is None:
        end_step = len(arr)
    sl = arr[start_step:end_step]
    if sl.size == 0:
        return 0.0
    return float(np.mean(sl <= proficiency_dist))


def random_baseline_proficiency(grid_size: int, proficiency_dist: int = 2, goal=(6, 6)) -> float:
    """Analytical fraction-of-cells within dist<=D of goal (for random-walk baseline)."""
    total = grid_size * grid_size
    near = 0
    for x in range(grid_size):
        for y in range(grid_size):
            if abs(x - goal[0]) + abs(y - goal[1]) <= proficiency_dist:
                near += 1
    return near / total


def summarize_g_run(
    run_json_path: str | Path,
    proficiency_dist: int = 2,
    window_size: int = 50,
    threshold: float = 0.5,
) -> dict:
    """Apply TTP + PF to a G-runner output JSON. Works for G6/G7/G8/G9 raw files.

    Handles moving-goal runs (phase_stats present) by computing per-phase
    metrics, so TTP is measured separately for each goal phase.
    """
    d = json.load(open(run_json_path))
    dist_log = d.get("distance_log", [])
    n_steps = d.get("n_steps", len(dist_log) - 1)
    grid_size = d.get("grid_size", 8)
    phase_stats = d.get("phase_stats") or []

    out = {
        "file": str(run_json_path),
        "seed": d.get("seed"),
        "n_steps": n_steps,
        "proficiency_dist": proficiency_dist,
        "window_size": window_size,
        "threshold": threshold,
        "random_baseline_PF": random_baseline_proficiency(grid_size, proficiency_dist),
    }

    if len(phase_stats) <= 1:
        # Fixed-goal: single TTP + single PF
        ttp = time_to_proficiency(dist_log, proficiency_dist, window_size, threshold)
        pf = proficiency_fraction(dist_log, proficiency_dist)
        out["fixed_goal"] = {
            "TTP": ttp,
            "TTP_fraction_of_episode": (ttp / n_steps) if ttp is not None else None,
            "PF_overall": pf,
            "acquired": ttp is not None,
        }
    else:
        # Moving-goal: per-phase TTP + PF
        phases = []
        for p_idx, p in enumerate(phase_stats):
            ps = p["step_start"]
            pe = p["step_end"]
            sub = dist_log[ps : pe + 1]  # inclusive of start dist
            ttp_local = time_to_proficiency(sub, proficiency_dist, window_size, threshold)
            ttp_abs = (ps + ttp_local) if ttp_local is not None else None
            pf = proficiency_fraction(sub, proficiency_dist)
            phases.append(
                {
                    "phase": p_idx,
                    "goal": p.get("goal"),
                    "step_start": ps,
                    "step_end": pe,
                    "TTP_within_phase": ttp_local,
                    "TTP_absolute": ttp_abs,
                    "PF_phase": pf,
                    "acquired": ttp_local is not None,
                }
            )
        out["moving_goal_phases"] = phases
        out["n_phases_acquired"] = sum(1 for p in phases if p["acquired"])
    return out


# ------------------------- Simple CLI ------------------------------

def _pretty_print_summary(s: dict):
    print(f"\n  {Path(s['file']).name}  (seed={s['seed']}, n_steps={s['n_steps']})")
    print(f"    random baseline PF = {s['random_baseline_PF']:.3f}  (window={s['window_size']}, threshold={s['threshold']})")
    if "fixed_goal" in s:
        fg = s["fixed_goal"]
        ttp_str = f"{fg['TTP']}" if fg["TTP"] is not None else "never"
        print(f"    FIXED   TTP={ttp_str}  PF={fg['PF_overall']:.3f}  acquired={fg['acquired']}")
    else:
        for p in s["moving_goal_phases"]:
            ttp_str = (
                f"{p['TTP_within_phase']} (abs {p['TTP_absolute']})"
                if p["TTP_within_phase"] is not None
                else "never"
            )
            print(
                f"    PHASE{p['phase']} goal={p['goal']}  TTP={ttp_str}  "
                f"PF={p['PF_phase']:.3f}  acquired={p['acquired']}"
            )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute redesigned gate metrics on G-runner output JSONs.")
    ap.add_argument("paths", nargs="+", help="Glob(s) or file(s)")
    ap.add_argument("--dist", type=int, default=2, help="proficiency_dist (default: 2)")
    ap.add_argument("--window", type=int, default=50, help="window_size (default: 50)")
    ap.add_argument("--threshold", type=float, default=0.5, help="threshold (default: 0.5)")
    args = ap.parse_args()

    import glob
    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        print("No files matched.")
        raise SystemExit(1)

    all_summaries = []
    for f in files:
        s = summarize_g_run(f, args.dist, args.window, args.threshold)
        _pretty_print_summary(s)
        all_summaries.append(s)

    # Aggregate: acquired count
    acquired_count = sum(
        (1 if s.get("fixed_goal", {}).get("acquired") else 0)
        + s.get("n_phases_acquired", 0)
        for s in all_summaries
    )
    print(f"\n  Total phases acquired across {len(all_summaries)} runs: {acquired_count}")
