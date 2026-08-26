"""Session G v5: Proportional action selection.

V1-V4 cumulative: motor exploration noise (V1) breaks the silent-motor
*firing* layer but argmax dominates action selection. V2 (first_spike,
rate=30) and V3 (positive-only) and V4 (action attribution) all fail
because the underlying problem is "the trained-winner motor gets selected
99%+ of the time, so the silent motor never receives reward-driven
weight updates that would let it grow."

V5 hypothesis: change action selection from argmax to **proportional
sampling** — pick motor m with probability ∝ (spike_count[m] + 1). The
+1 floor ensures even zero-count motors get nonzero selection probability.

This is biologically real: rate-coded motor selection in cortex/basal
ganglia uses softmax-like rules where firing rate maps to probability of
action initiation, not winner-take-all. (Cisek 2007, Klaus 2017).

Two conditions × 3 seeds = 6 runs. ~80 min wall.

Conditions:
  - prop_rate0:  proportional, no exploration (cleanest test)
  - prop_rate15: proportional + exploration (combined)
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
CONDITIONS = [
    ("prop_rate0", {
        "action_selection": "proportional",
        "motor_exploration_rate_hz": 0.0,
    }),
    ("prop_rate15", {
        "action_selection": "proportional",
        "motor_exploration_rate_hz": 15.0,
    }),
]


def main():
    out_dir = Path("research/findings/raw/g9_motor_exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t_total = time.time()

    for label, cond_kwargs in CONDITIONS:
        for seed in SEEDS:
            out_path = out_dir / f"g9_v5_{label}_seed{seed}.json"
            print(f"\n{'='*72}")
            print(f"  {label}  seed={seed}  steps={N_STEPS}  {cond_kwargs}")
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
                verbose=True,
                **cond_kwargs,
            )
            elapsed = time.time() - t0

            data = json.load(open(out_path))
            ph0 = data["phase_stats"][0]
            ph1 = data["phase_stats"][1]
            row = {
                "label": label, "seed": seed,
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

    csv_path = out_dir / "v5_summary.csv"
    with open(csv_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                row_out = {k: (str(v) if isinstance(v, list) else v)
                           for k, v in r.items()}
                writer.writerow(row_out)

    print(f"\n{'='*72}")
    print(f"  V5 SUMMARY ({time.time() - t_total:.0f}s wall)")
    print(f"{'='*72}")
    for label, _ in CONDITIONS:
        rs = [r for r in summary_rows if r["label"] == label]
        if not rs:
            continue
        ph1fq = sum(r["phase1_finalQ"] for r in rs) / len(rs)
        ph1at = sum(r["phase1_steps_at_goal"] for r in rs) / len(rs)
        print(f"\n  {label}:")
        for r in rs:
            print(f"    seed={r['seed']:2d}: ph1 finalQ={r['phase1_finalQ']:.2f}  "
                  f"atGoal={r['phase1_steps_at_goal']:4d}  "
                  f"actions={r['phase1_actions']}")
        print(f"    avg ph1 finalQ: {ph1fq:.2f}  avg ph1 atGoal: {ph1at:.1f}")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
