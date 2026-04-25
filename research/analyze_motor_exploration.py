"""Compare baseline vs exploration-noise G9 results.

Reads JSONs from research/findings/raw/g9_motor_exploration/ and produces
a per-condition comparison table. Specifically:

- Per-motor activity in each phase (silent-motor detection)
- Phase-1 final-quarter mean distance (readaptation success)
- Steps spent at goal in each phase
- Plastic-weight final std (learning evidence)

This is the "did exploration noise break the silent-motor trap?" analyzer.
"""
import json
import sys
from pathlib import Path

import numpy as np


def load_run(json_path: Path) -> dict:
    """Read one G9 run JSON and extract the comparison metrics."""
    d = json.load(open(json_path))
    motor_counts = np.asarray(d["motor_counts"])  # (n_steps, n_motor)
    action_log = np.asarray(d["action_log"])
    goal_change_steps = d["goal_change_steps"]

    if not goal_change_steps:
        ph0_end = len(motor_counts)
        ph1_start = ph0_end
    else:
        ph0_end = goal_change_steps[0]
        ph1_start = ph0_end

    ph0_motor = motor_counts[:ph0_end]
    ph1_motor = motor_counts[ph1_start:]

    ph0_actions = action_log[:ph0_end]
    ph1_actions = action_log[ph1_start:]

    n_motor = motor_counts.shape[1]
    motor_names = ["N", "E", "S", "W"][:n_motor]

    ph0_total_per_motor = ph0_motor.sum(axis=0)
    ph1_total_per_motor = ph1_motor.sum(axis=0)
    ph0_action_counts = [int((ph0_actions == a).sum()) for a in range(n_motor)]
    ph1_action_counts = [int((ph1_actions == a).sum()) for a in range(n_motor)]

    phase_stats = d["phase_stats"]
    return {
        "seed": d["seed"],
        "rate_hz": d.get("motor_exploration_rate_hz", 0.0),
        "ph0_finalQ": phase_stats[0]["final_quarter_mean_distance"]
                       if len(phase_stats) > 0 else None,
        "ph1_finalQ": phase_stats[1]["final_quarter_mean_distance"]
                       if len(phase_stats) > 1 else None,
        "ph0_meanD": phase_stats[0]["mean_distance"]
                       if len(phase_stats) > 0 else None,
        "ph1_meanD": phase_stats[1]["mean_distance"]
                       if len(phase_stats) > 1 else None,
        "ph0_steps_at_goal": phase_stats[0]["n_steps_at_goal"]
                       if len(phase_stats) > 0 else None,
        "ph1_steps_at_goal": phase_stats[1]["n_steps_at_goal"]
                       if len(phase_stats) > 1 else None,
        "ph0_action_counts": ph0_action_counts,
        "ph1_action_counts": ph1_action_counts,
        "ph0_total_motor_spikes": ph0_total_per_motor.tolist(),
        "ph1_total_motor_spikes": ph1_total_per_motor.tolist(),
        "ph0_silent_motors": [motor_names[i] for i in range(n_motor)
                              if ph0_total_per_motor[i] == 0],
        "ph1_silent_motors": [motor_names[i] for i in range(n_motor)
                              if ph1_total_per_motor[i] == 0],
        "plastic_w_final_std": d["plastic_weight_final_std"],
    }


def fmt_actions(counts):
    return f"[N={counts[0]:4d} E={counts[1]:4d} S={counts[2]:4d} W={counts[3]:4d}]"


def main():
    raw_dir = Path("research/findings/raw/g9_motor_exploration")
    if not raw_dir.exists():
        print(f"Missing: {raw_dir}")
        sys.exit(1)

    runs = []
    for jp in sorted(raw_dir.glob("g9_*.json")):
        runs.append((jp.name, load_run(jp)))

    print(f"\n{'='*92}")
    print(f"  Session G: Motor Exploration Noise — Per-Run Comparison")
    print(f"{'='*92}")

    # Group by rate, sort by seed
    by_rate = {}
    for name, r in runs:
        by_rate.setdefault(r["rate_hz"], []).append((name, r))
    for rate in sorted(by_rate.keys()):
        rate_label = f"baseline (rate={rate})" if rate == 0 else f"treatment (rate={rate} Hz)"
        print(f"\n## {rate_label}")
        print(f"{'-'*92}")
        for name, r in sorted(by_rate[rate], key=lambda x: x[1]["seed"]):
            print(f"\n  seed={r['seed']}:")
            print(f"    Phase 0 (goal acquired): finalQ={r['ph0_finalQ']:.2f}  "
                  f"meanD={r['ph0_meanD']:.2f}  atGoal={r['ph0_steps_at_goal']}")
            print(f"      actions:      {fmt_actions(r['ph0_action_counts'])}")
            print(f"      motor spikes: {r['ph0_total_motor_spikes']}")
            if r["ph0_silent_motors"]:
                print(f"      *** SILENT MOTORS: {r['ph0_silent_motors']} ***")
            print(f"    Phase 1 (readaptation):  finalQ={r['ph1_finalQ']:.2f}  "
                  f"meanD={r['ph1_meanD']:.2f}  atGoal={r['ph1_steps_at_goal']}")
            print(f"      actions:      {fmt_actions(r['ph1_action_counts'])}")
            print(f"      motor spikes: {r['ph1_total_motor_spikes']}")
            if r["ph1_silent_motors"]:
                print(f"      *** SILENT MOTORS: {r['ph1_silent_motors']} ***")
            print(f"    plastic_w_final_std={r['plastic_w_final_std']:.3f}")

    # Aggregate metrics
    print(f"\n\n{'='*92}")
    print(f"  Aggregate Summary")
    print(f"{'='*92}")
    print(f"\n  {'Condition':<28} | {'Ph0 finalQ':>11} | {'Ph1 finalQ':>11} | "
          f"{'Ph1 atGoal':>11} | {'silent_motor_seeds':>20}")
    print(f"  {'-'*28}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}-+-{'-'*20}")
    for rate in sorted(by_rate.keys()):
        rate_label = f"baseline" if rate == 0 else f"explore@{rate:.0f}Hz"
        rs = [r for _, r in by_rate[rate]]
        ph0fq = np.mean([r["ph0_finalQ"] for r in rs])
        ph1fq = np.mean([r["ph1_finalQ"] for r in rs])
        ph1at = np.mean([r["ph1_steps_at_goal"] for r in rs])
        any_silent_ph1 = sum(1 for r in rs if r["ph1_silent_motors"])
        print(f"  {rate_label:<28} | {ph0fq:>11.2f} | {ph1fq:>11.2f} | "
              f"{ph1at:>11.1f} | {any_silent_ph1:>4d}/{len(rs)}")

    print(f"\nPass criteria for treatment to dissolve the silent-motor trap:")
    print(f"  1. Ph1 finalQ < 4 in >=2/3 seeds")
    print(f"  2. 0/3 seeds with any silent motor in phase 1")
    print(f"  3. Baseline reproduces prior finding (>= 1 seed silent-motor)")


if __name__ == "__main__":
    main()
