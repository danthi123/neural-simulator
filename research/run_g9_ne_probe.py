"""Session E.1 validation probe: does NE excitability_drive escape the
silent-motor trap?

Setup matches `research/run_g9_relaxed_moving.py` (1800 steps, phase-1 goal
(6,6), phase-2 goal (1,6) at step 300, argmax) but adds a noradrenaline
modulator wired to:
  - rule: from_error_persistence (rises when |reward error| stays high)
  - target: excitability_drive on group:motor (boosts ALL motor neurons
    uniformly, breaking the eligibility-gated trap because silent motors
    finally get a chance to fire)

If NE breaks the trap:
  - Phase 2 PF (fraction-of-steps within Manhattan dist <= 2 of new goal)
    should rise above the relaxed-probe baseline (0.001-0.018).
  - At least 2/3 seeds should show TTP within phase 2.

If it doesn't break the trap:
  - Probe records the negative result (informative; lets us tune
    sensitivity / threshold / decay tau).

3 seeds, ~30-45 min wall time on RTX 3090.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.runners.g9_runner import run_g9_episode
from sim.neuromodulators import (
    NeuromodulatorConfig,
    ProductionRule,
    ModulatorTarget,
)


GOAL_SCHEDULE = [(0, (6, 6)), (300, (1, 6))]
N_STEPS = 1800


def _ne_configs():
    """Two-modulator config: DA replicates legacy reward path, NE provides
    the new silent-motor escape via from_error_persistence + excitability_drive
    on motor neurons."""
    return [
        NeuromodulatorConfig(
            name="dopamine",
            baseline=0.0,
            decay_tau_ms=500.0,
            production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
            targets=[],  # legacy reward path still does the heavy lifting
        ),
        NeuromodulatorConfig(
            name="noradrenaline",
            baseline=0.05,
            decay_tau_ms=3000.0,        # slow tonic dynamics (matches LC kinetics)
            concentration_min=0.0,
            concentration_max=2.0,
            production_rules=[
                ProductionRule(
                    rule_type="from_error_persistence",
                    sensitivity=1.0,
                    threshold=0.4,       # only sustained errors matter
                    window_ms=2000.0,    # longer window than reward variability
                )
            ],
            targets=[
                ModulatorTarget(
                    target_type="excitability_drive",
                    scope="group:motor",
                    sensitivity=120.0,    # 120 pA at conc=1.0 -> meaningful boost
                ),
            ],
        ),
    ]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds", default="42,43,44",
        help="Comma-separated seeds to run sequentially in this process. "
             "For parallel execution across seeds, launch multiple processes "
             "each with --seeds=<one>.",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = Path("research/findings/raw/g9")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        out_path = out_dir / f"g9_ne_relaxed_seed{seed}.json"
        print(f"\n{'='*70}")
        print(f"E.1 NE PROBE: seed={seed}  n_steps={N_STEPS}  phase2=1500 steps")
        print(f"{'='*70}")
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
            nm_configs=_ne_configs(),
            verbose=True,
        )


if __name__ == "__main__":
    main()
