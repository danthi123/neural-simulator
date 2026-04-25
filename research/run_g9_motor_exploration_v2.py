"""Session G v2: Escalated motor exploration noise probe.

V1 result: rate=15 + argmax fires every motor but W still wins argmax only 1
time in 1500 phase-1 steps. Eligibility forms, but the motor-selection layer
is too entrenched on the phase-1 winner to give the new-correct motor a
chance to actually drive the agent west.

V2 tries two stronger interventions:

  - **first_spike** action selection: picks the motor with earliest spike
    in the 50-150 ms readout window (vs. argmax which sums spike counts).
    This is much more noise-sensitive — even a single spurious spike from
    exploration noise can win the argmin over a weakly-firing competitor.
    Biology-canonical (lateral-inhibition WTA).

  - **rate=30**: doubles the spurious spike rate so each motor receives
    ~3 noise spikes per 100ms readout window. Brings W's baseline from
    ~2 spikes/step to ~3 spikes/step, much closer to the trained motors'
    ~4-5 spikes. May let W occasionally cross the argmax threshold.

Two conditions × 3 seeds = 6 runs. ~80 min wall.
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.runners.g9_runner import run_g9_episode


GOAL_SCHEDULE = [(0, (6, 6)), (300, (1, 6))]
N_STEPS = 1800
SEEDS = [42, 43, 44]

# (action_selection, motor_exploration_rate_hz) pairs
CONDITIONS = [
    ("first_spike", 15.0),
    ("argmax", 30.0),
]


def main():
    out_dir = Path("research/findings/raw/g9_motor_exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t_total = time.time()

    for action_sel, rate in CONDITIONS:
        for seed in SEEDS:
            label = f"{action_sel}_rate{int(rate)}"
            out_path = out_dir / f"g9_v2_{label}_seed{seed}.json"
            print(f"\n{'='*72}")
            print(f"  action_selection={action_sel}  rate={rate} Hz  "
                  f"seed={seed}  steps={N_STEPS}")
            print(f"{'='*72}", flush=True)

            t0 = time.time()
            run_g9_episode(
                out_path=str(out_path),
                seed=seed,
                n_steps=N_STEPS,
                start_pos=(1, 1),
                goal_pos=(6, 6),
                goal_schedule=GOAL_SCHEDULE,
                learning_rate=0.01,
                reward_eligibility_tau_ms=500.0,
                reward_hold_steps=10,
                action_selection=action_sel,
                motor_exploration_rate_hz=rate,
                verbose=True,
            )
            elapsed = time.time() - t0

            data = json.load(open(out_path))
            ph0 = data["phase_stats"][0]
            ph1 = data["phase_stats"][1]
            row = {
                "action_selection": action_sel, "rate_hz": rate,
                "seed": seed,
                "phase0_finalQ": ph0["final_quarter_mean_distance"],
                "phase0_actions": ph0["action_counts"],
                "phase1_finalQ": ph1["final_quarter_mean_distance"],
                "phase1_actions": ph1["action_counts"],
                "phase1_steps_at_goal": ph1["n_steps_at_goal"],
                "elapsed_s": round(elapsed, 1),
            }
            summary_rows.append(row)
            print(f"  -> phase0 finalQ={ph0['final_quarter_mean_distance']:.2f}  "
                  f"phase1 finalQ={ph1['final_quarter_mean_distance']:.2f}  "
                  f"phase1 actions={ph1['action_counts']}  ({elapsed:.0f}s)",
                  flush=True)

    csv_path = out_dir / "v2_summary.csv"
    with open(csv_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                row_out = {k: (str(v) if isinstance(v, list) else v)
                           for k, v in r.items()}
                writer.writerow(row_out)

    print(f"\n{'='*72}")
    print(f"  V2 SUMMARY ({time.time() - t_total:.0f}s wall)")
    print(f"{'='*72}")
    for action_sel, rate in CONDITIONS:
        rs = [r for r in summary_rows
              if r["action_selection"] == action_sel and r["rate_hz"] == rate]
        ph1fq = sum(r["phase1_finalQ"] for r in rs) / len(rs)
        ph1at = sum(r["phase1_steps_at_goal"] for r in rs) / len(rs)
        print(f"\n  {action_sel} + rate={rate}Hz:")
        for r in rs:
            print(f"    seed={r['seed']:2d}: ph1 finalQ={r['phase1_finalQ']:.2f}  "
                  f"atGoal={r['phase1_steps_at_goal']:4d}  "
                  f"actions={r['phase1_actions']}")
        print(f"    avg ph1 finalQ: {ph1fq:.2f}  avg ph1 atGoal: {ph1at:.1f}")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
