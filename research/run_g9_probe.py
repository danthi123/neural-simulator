"""G9 probe: sim-native R-STDP on fixed-goal and moving-goal scenarios.

Phase 1 (quick validation): 3 seeds, fixed goal (6,6), argmax. Does the
sim-native R-STDP path learn at all, and does it match G6's performance?

Phase 2 (the G7 test): 3 seeds, moving goal (G7 scenario), both argmax
and first-spike WTA. Can the sim-native path readapt where the runner-
side path could not?
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.runners.g9_runner import run_g9_episode


FIXED_GOAL_CONFIG = {
    "goal_schedule": None,   # single goal (6,6) throughout
    "label": "fixed",
}

MOVING_GOAL_CONFIG = {
    "goal_schedule": [(0, (6, 6)), (300, (1, 6))],
    "label": "moving",
}


def run_phase(phase_name, scenario, action_selections, seeds, out_dir,
              enable_neuromod_gating=False, neuromod_tag=""):
    """Run a phase: scenario × action_selections × seeds.

    neuromod_tag: suffix appended to output filenames when neuromod is on,
    so "neuromod" runs don't clobber base runs.
    """
    all_results = {}
    for action_sel in action_selections:
        for seed in seeds:
            suffix = f"_{neuromod_tag}" if neuromod_tag else ""
            out_path = out_dir / f"g9_{scenario['label']}_{action_sel}{suffix}_seed{seed}.json"
            print(f"\n{'='*70}")
            print(f"G9 {phase_name}: scenario={scenario['label']} action={action_sel} "
                  f"seed={seed} neuromod={enable_neuromod_gating}")
            print(f"{'='*70}")
            r = run_g9_episode(
                out_path=str(out_path),
                seed=seed, n_steps=600,
                start_pos=(1, 1), goal_pos=(6, 6),
                goal_schedule=scenario["goal_schedule"],
                learning_rate=0.01,
                reward_eligibility_tau_ms=500.0,
                reward_hold_steps=10,
                action_selection=action_sel,
                enable_neuromod_gating=enable_neuromod_gating,
                neuromod_tau_ms=100.0,
                neuromod_strength=0.5,
                verbose=True,
            )
            all_results[(scenario["label"], action_sel, seed)] = r
    return all_results


def summarize(phase_name, all_results):
    print(f"\n{'='*70}")
    print(f"{phase_name.upper()} SUMMARY")
    print(f"{'='*70}")
    # Group by (scenario, action_sel)
    grouped = {}
    for (scen, act, seed), r in all_results.items():
        grouped.setdefault((scen, act), []).append(r)

    for (scen, act), results in sorted(grouped.items()):
        print(f"\n  Scenario: {scen}  Action: {act}")
        for r in results:
            seed = r["seed"]
            phases = r.get("phase_stats", [])
            if len(phases) == 1:
                p0 = phases[0]
                print(f"    seed={seed}: mean_dist={r['mean_distance_overall']:.2f}  "
                      f"quarters={[round(q, 2) for q in r['mean_distance_quarters']]}  "
                      f"at_goal={r['n_steps_at_goal']}  actions={r['action_counts']}")
            else:
                for p in phases:
                    print(f"    seed={seed} phase{p['phase']} goal={p['goal']}: "
                          f"mean_dist={p['mean_distance']:.2f}  finalQ={p['final_quarter_mean_distance']:.2f}  "
                          f"at_goal={p['n_steps_at_goal']}  actions={p['action_counts']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["fixed", "moving", "both"], default="both")
    parser.add_argument("--action", choices=["argmax", "first_spike", "both"], default="both")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--neuromod", action="store_true",
                        help="Enable neuromodulatory gain gating (Session C).")
    parser.add_argument("--tag", default="",
                        help="Filename suffix tag (e.g. 'neuromod' for neuromod runs).")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    action_selections = ["argmax", "first_spike"] if args.action == "both" else [args.action]

    out_dir = Path("research/findings/raw/g9")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    scenarios = []
    if args.scenario in ("fixed", "both"):
        scenarios.append(FIXED_GOAL_CONFIG)
    if args.scenario in ("moving", "both"):
        scenarios.append(MOVING_GOAL_CONFIG)

    for scenario in scenarios:
        phase_results = run_phase(
            f"scenario={scenario['label']}", scenario,
            action_selections, seeds, out_dir,
            enable_neuromod_gating=args.neuromod,
            neuromod_tag=args.tag,
        )
        all_results.update(phase_results)
        summarize(f"scenario={scenario['label']}", phase_results)


if __name__ == "__main__":
    main()
