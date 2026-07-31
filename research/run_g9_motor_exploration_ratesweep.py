"""Session G follow-up: Motor exploration rate-sensitivity sweep.

Runs after the main run_g9_motor_exploration.py result if and only if the
15 Hz treatment cleared the silent-motor trap. Characterizes how the
exploration rate affects readaptation:
  - Too low: doesn't break silent-motor trap (silent motors stay silent)
  - Optimal: enough exploration for eligibility, not enough to hurt action
  - Too high: dominates action selection, agent flails

Same scenario as run_g9_motor_exploration.py:
  - Phase 0: goal (6,6) for 300 steps
  - Phase 1: goal (1,6) for 1500 steps  (silent-motor trap territory)
  - 3 seeds: 42, 43, 44

Conditions: rates {5, 30, 60} Hz (15 Hz already covered by main probe).
  - 5 Hz:  ~0.25 spurious spikes/motor per 100ms readout window
  - 30 Hz: ~1.5 spurious spikes/motor — comparable to baseline motor rate
  - 60 Hz: ~3 spurious spikes/motor — likely dominates action selection

Total wall: ~45-60 minutes (9 runs × 1800 steps).
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
EXPLORATION_RATES = [5.0, 30.0, 60.0]


def main():
    out_dir = Path("research/findings/raw/g9_motor_exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t_total = time.time()

    for rate in EXPLORATION_RATES:
        rate_label = f"explore{int(rate)}"
        for seed in SEEDS:
            out_path = out_dir / f"g9_{rate_label}_seed{seed}.json"
            print(f"\n{'='*72}")
            print(f"  motor_exploration_rate_hz={rate}  seed={seed}  steps={N_STEPS}")
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
                action_selection="argmax",
                motor_exploration_rate_hz=rate,
                verbose=True,
            )
            elapsed = time.time() - t0

            data = json.load(open(out_path))
            ph0 = data["phase_stats"][0]
            ph1 = data["phase_stats"][1]
            row = {
                "rate_hz": rate, "seed": seed,
                "phase0_finalQ_dist": ph0["final_quarter_mean_distance"],
                "phase0_actions": ph0["action_counts"],
                "phase1_finalQ_dist": ph1["final_quarter_mean_distance"],
                "phase1_actions": ph1["action_counts"],
                "phase1_steps_at_goal": ph1["n_steps_at_goal"],
                "elapsed_s": round(elapsed, 1),
            }
            summary_rows.append(row)
            print(f"  -> phase0 finalQ={ph0['final_quarter_mean_distance']:.2f}  "
                  f"phase1 finalQ={ph1['final_quarter_mean_distance']:.2f}  "
                  f"phase1 actions={ph1['action_counts']}  ({elapsed:.0f}s)",
                  flush=True)

    csv_path = out_dir / "ratesweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                row_out = {k: (str(v) if isinstance(v, list) else v)
                           for k, v in r.items()}
                writer.writerow(row_out)

    print(f"\n{'='*72}")
    print(f"  RATE SWEEP SUMMARY ({time.time() - t_total:.0f}s wall)")
    print(f"{'='*72}")
    for rate in EXPLORATION_RATES:
        rs = [r for r in summary_rows if r["rate_hz"] == rate]
        ph1fq = sum(r["phase1_finalQ_dist"] for r in rs) / len(rs)
        ph1at = sum(r["phase1_steps_at_goal"] for r in rs) / len(rs)
        print(f"\n  rate={rate} Hz:")
        for r in rs:
            print(f"    seed={r['seed']:2d}: ph1 finalQ={r['phase1_finalQ_dist']:.2f}  "
                  f"atGoal={r['phase1_steps_at_goal']:4d}  "
                  f"actions={r['phase1_actions']}")
        print(f"    avg ph1 finalQ: {ph1fq:.2f}  avg ph1 atGoal: {ph1at:.1f}")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
