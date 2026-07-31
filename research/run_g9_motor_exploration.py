"""Session G: Motor exploration noise probe.

Tests whether injecting Poisson spikes into motor neurons during the stimulus
window breaks the silent-motor trap that defeated:
  - Route A (parallel-subprocess seed-replicas)
  - Route C (5,068-neuron reservoirs)
  - NE sensitivity sweep (tonic excitability drive at sensitivity 0..150)
  - PFC bistability tuning (homeostatic floor blocks persistent activity)

Hypothesis: Phase-1-silent motors can't acquire eligibility under STDP. Without
eligibility, no reward-mediated weight update reaches them. Adding stochastic
spike input ensures every motor fires occasionally; STDP can then form
positive eligibility traces with co-firing hidden neurons; reward then
converts those into weight changes. This is the classical exploration-noise
fix (e.g. epsilon-greedy / entropy-regularization).

Scenario: the canonical relaxed moving-goal task from Session D.A.4
  - Phase 1 (steps 0-300):    goal (6, 6)
  - Phase 2 (steps 300-1800): goal (1, 6)  — silent-motor trap territory
  - 3 seeds: 42, 43, 44

Conditions:
  - baseline:  motor_exploration_rate_hz = 0   (replicates D.A.4 result)
  - treatment: motor_exploration_rate_hz = 15  (smoke-tested working level)

Output: JSON per (seed, condition) plus a summary CSV.
Total wall: ~12-18 minutes (6 runs × 1800 steps).
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
EXPLORATION_RATES = [0.0, 15.0]


def main():
    out_dir = Path("research/findings/raw/g9_motor_exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t_total = time.time()

    for rate in EXPLORATION_RATES:
        rate_label = f"explore{int(rate)}" if rate > 0 else "baseline"
        for seed in SEEDS:
            out_path = out_dir / f"g9_{rate_label}_seed{seed}.json"
            print(f"\n{'='*72}")
            print(f"  motor_exploration_rate_hz={rate}  seed={seed}  "
                  f"steps={N_STEPS}")
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
                "rate_hz": rate,
                "seed": seed,
                "phase0_mean_dist": ph0["mean_distance"],
                "phase0_finalQ_dist": ph0["final_quarter_mean_distance"],
                "phase0_actions": ph0["action_counts"],
                "phase1_mean_dist": ph1["mean_distance"],
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

    # Write CSV summary
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                # action_counts is a list; serialize as string
                row_out = {k: (str(v) if isinstance(v, list) else v)
                           for k, v in r.items()}
                writer.writerow(row_out)

    # Print summary
    print(f"\n{'='*72}")
    print(f"  SESSION G: Motor exploration probe ({time.time() - t_total:.0f}s wall)")
    print(f"{'='*72}")
    for rate in EXPLORATION_RATES:
        rate_label = f"explore{int(rate)}" if rate > 0 else "baseline"
        rows = [r for r in summary_rows if r["rate_hz"] == rate]
        ph1_finalQ = [r["phase1_finalQ_dist"] for r in rows]
        ph1_at_goal = [r["phase1_steps_at_goal"] for r in rows]
        all_motors_active = sum(
            1 for r in rows if all(c > 0 for c in r["phase1_actions"])
        )
        print(f"\n  {rate_label} (rate={rate} Hz):")
        for r in rows:
            print(f"    seed={r['seed']:2d}: phase1 finalQ={r['phase1_finalQ_dist']:.2f}  "
                  f"steps_at_goal={r['phase1_steps_at_goal']:4d}  "
                  f"actions={r['phase1_actions']}")
        print(f"    avg phase1 finalQ: {sum(ph1_finalQ)/len(ph1_finalQ):.2f}")
        print(f"    avg phase1 steps_at_goal: {sum(ph1_at_goal)/len(ph1_at_goal):.1f}")
        print(f"    seeds with all 4 motors active in phase1: {all_motors_active}/{len(rows)}")
    print(f"\nResults in {out_dir}/")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
