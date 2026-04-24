"""Session D.A.5: random-start generalization probe.

Train G9 fixed-goal on (start=(1,1), goal=(6,6)) for 600 steps, then freeze
plastic weights and test from 20 random start positions. If the trained
policy is a true controller (generalizes over input state), tail_mean_dist
should be low across random starts. If it's a memorized trajectory,
random-start evals will land anywhere.

Random-walk baseline: mean Manhattan on 8x8 grid is ~5.5 after a short
trajectory from random start. A trained controller should get significantly
below this.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.runners.g9_runner import run_g9_episode


def main():
    out_dir = Path("research/findings/raw/g9")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in (42, 43, 44):
        out_path = out_dir / f"g9_rsg_seed{seed}.json"
        print(f"\n{'='*70}")
        print(f"RSG PROBE: seed={seed}  (600 train + 20 random-start evals x 30 steps)")
        print(f"{'='*70}")
        run_g9_episode(
            out_path=str(out_path),
            seed=seed,
            n_steps=600,
            start_pos=(1, 1),
            goal_pos=(6, 6),
            goal_schedule=None,       # fixed goal throughout
            learning_rate=0.01,
            reward_eligibility_tau_ms=500.0,
            reward_hold_steps=10,
            action_selection="argmax",
            eval_random_starts=20,
            eval_steps_per_start=30,
            verbose=True,
        )


if __name__ == "__main__":
    main()
