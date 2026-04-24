"""G8 probe: does the goal-context channel enable moving-goal readaptation?

Runs two conditions x 3 seeds each on the G7 NO-GO scenario (goal moves at step 300):
  - goal_context_enabled=True  (PFC-like context signal active)
  - goal_context_enabled=False (ablation, equivalent to G6/G7 architecture)

Reports per-condition phase statistics so we can compare readaptation.
"""
import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.runners.g8_runner import run_g8_episode


GOAL_SCHEDULE = [
    (0, (6, 6)),       # Phase 1: goal at (6, 6)
    (300, (1, 6)),     # Phase 2: goal jumps to (1, 6) — NW instead of NE
]

CONDITIONS = [
    ("goal_context_on", True),
    ("goal_context_off", False),
]

SEEDS = [42, 43, 44]

OUTPUT_DIR = Path("research/findings/raw/g8")


def run_one(cond_name, enabled, seed):
    out_path = OUTPUT_DIR / f"g8_{cond_name}_seed{seed}.json"
    return run_g8_episode(
        out_path=str(out_path),
        seed=seed, n_steps=600, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        goal_schedule=GOAL_SCHEDULE,
        learning_rate=0.01,
        lr_schedule="decay_after_goal", lr_decay_factor=0.25,
        epsilon_start=0.1, epsilon_end=0.0, epsilon_decay_steps=150,
        reset_epsilon_on_goal_change=True,
        negative_reward_rule="B",
        goal_context_enabled=enabled,
        verbose=True,
    )


def main():
    all_results = {}
    for cond_name, enabled in CONDITIONS:
        cond_results = []
        for seed in SEEDS:
            print(f"\n{'='*70}")
            print(f"CONDITION: {cond_name}  SEED: {seed}")
            print(f"{'='*70}")
            r = run_one(cond_name, enabled, seed)
            cond_results.append(r)
        all_results[cond_name] = cond_results

    # Compare phase 0 (before goal change) vs phase 1 (after)
    print(f"\n{'='*70}")
    print("G8 PROBE SUMMARY")
    print(f"{'='*70}")

    summary = {}
    for cond_name, results_list in all_results.items():
        print(f"\n{cond_name}:")
        cond_summary = {"phase_0": [], "phase_1": []}
        for r in results_list:
            seed = r["seed"]
            phases = r["phase_stats"]
            if len(phases) < 2:
                print(f"  seed={seed}: only {len(phases)} phase(s), skipping")
                continue
            p0 = phases[0]
            p1 = phases[1]
            cond_summary["phase_0"].append({
                "seed": seed,
                "mean_dist": p0["mean_distance"],
                "final_q_dist": p0["final_quarter_mean_distance"],
                "at_goal": p0["n_steps_at_goal"],
            })
            cond_summary["phase_1"].append({
                "seed": seed,
                "mean_dist": p1["mean_distance"],
                "final_q_dist": p1["final_quarter_mean_distance"],
                "at_goal": p1["n_steps_at_goal"],
            })
            print(f"  seed={seed}:")
            print(f"    Phase 0 (goal={p0['goal']}): mean_dist={p0['mean_distance']:.2f}  "
                  f"final_q={p0['final_quarter_mean_distance']:.2f}  at_goal={p0['n_steps_at_goal']}")
            print(f"    Phase 1 (goal={p1['goal']}): mean_dist={p1['mean_distance']:.2f}  "
                  f"final_q={p1['final_quarter_mean_distance']:.2f}  at_goal={p1['n_steps_at_goal']}")
        summary[cond_name] = cond_summary

    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATE (mean across 3 seeds)")
    print(f"{'='*70}")
    for cond_name, data in summary.items():
        p0_final = sum(d["final_q_dist"] for d in data["phase_0"]) / max(len(data["phase_0"]), 1)
        p1_final = sum(d["final_q_dist"] for d in data["phase_1"]) / max(len(data["phase_1"]), 1)
        p0_at_goal = sum(d["at_goal"] for d in data["phase_0"]) / max(len(data["phase_0"]), 1)
        p1_at_goal = sum(d["at_goal"] for d in data["phase_1"]) / max(len(data["phase_1"]), 1)
        print(f"  {cond_name}:")
        print(f"    Phase 0 final_q_dist mean: {p0_final:.2f}  at_goal mean: {p0_at_goal:.1f}")
        print(f"    Phase 1 final_q_dist mean: {p1_final:.2f}  at_goal mean: {p1_at_goal:.1f}")
        print(f"    READAPTATION DELTA (P1 final vs P0 final): {p1_final - p0_final:+.2f}")

    # Save everything
    out_summary = OUTPUT_DIR / "g8_probe_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w") as f:
        json.dump({"by_condition": summary}, f, indent=2, default=str)
    print(f"\nSummary written to {out_summary}")


if __name__ == "__main__":
    main()
