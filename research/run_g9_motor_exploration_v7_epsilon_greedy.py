"""Session I: ε-greedy action selection + V1 motor exploration.

V1-V6 cumulative findings:
  - V1 motor exploration noise breaks the silent-motor *firing* invariant
    (every motor fires) but argmax still locks onto trained-winner.
  - V2-V5 explored shallower interventions on action selection / reward sign.
    None help.
  - V6 weight reset on goal change. KEY FINDING: even alpha=1.0 (full reset
    of hidden→motor weights to random initial values), seed 42 still has
    W=0 in phase 1. This means the *reservoir hidden state itself is
    structurally biased* toward patterns aligned with random initial
    hidden→E weights, NOT just trained E weights.

V7 hypothesis: bypass the reservoir-bias problem entirely with **ε-greedy
action selection**. With probability ε, the runner picks a uniformly random
action regardless of motor spike counts. This guarantees W gets selected
~ε/4 of the time, allowing reward+eligibility to accumulate W weight
changes regardless of how the reservoir biases hidden→motor activations.

This is the canonical RL exploration mechanism. Schultz/Sutton/Watkins;
universally acknowledged. Biologically grounded in tonic dopamine driving
behavioral variability (Shadmehr 2010).

Two epsilon levels × 3 seeds = 6 runs. ~80 min wall.
All conditions use motor_exploration_rate_hz=15 + argmax + bipolar reward.

Conditions:
  - eps_01: epsilon_greedy=0.1 (W selected ~2.5% / 37 times in phase 1)
  - eps_02: epsilon_greedy=0.2 (W selected ~5% / 75 times in phase 1)
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
    ("eps_01", {"epsilon_greedy": 0.1}),
    ("eps_02", {"epsilon_greedy": 0.2}),
]


def main():
    out_dir = Path("research/findings/raw/g9_motor_exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t_total = time.time()

    for label, cond_kwargs in CONDITIONS:
        for seed in SEEDS:
            out_path = out_dir / f"g9_v7_{label}_seed{seed}.json"
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
                action_selection="argmax",
                motor_exploration_rate_hz=15.0,
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

    csv_path = out_dir / "v7_summary.csv"
    with open(csv_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                row_out = {k: (str(v) if isinstance(v, list) else v)
                           for k, v in r.items()}
                writer.writerow(row_out)

    print(f"\n{'='*72}")
    print(f"  V7 (Session I) SUMMARY ({time.time() - t_total:.0f}s wall)")
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
    print(f"Compare to V1 baseline (rate=15, no eps): avg phase-1 finalQ 6.40")


if __name__ == "__main__":
    main()
