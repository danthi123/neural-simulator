"""Session D.A.4: relaxed moving-goal probe.

The strict 300-step phase-2 budget in G7/G8/G9/C gave us 4 NO-GOs in a
row. Retrospective TTP analysis (gate_metrics.py) shows phase-0
acquisition takes TTP ~50-200 steps across runners, so 300 phase-2
steps is barely 1-2x the acquisition timescale — not enough time for
the system to:
  1. notice reward has flipped sign,
  2. depress phase-1-winning synapses,
  3. build new eligibility on phase-2-correct motors,
  4. cross argmax/first-spike threshold,
  5. consolidate.

This probe extends phase 2 to 1500 steps (5x the original). If
readaptation happens, TTP_phase1 is a measure of *how slow* biological
learning of reversal is in this sim. If it doesn't happen even at 1500
steps, something structural (not just time) is missing.

Runs 3 seeds with argmax (the consolidation-favoring action selection)
on the same (6,6) -> (1,6) goal-change pattern as G9.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.runners.g9_runner import run_g9_episode


# Phase 1: goal (6,6), 300 steps (same as original G9)
# Phase 2: goal (1,6), 1500 steps (5x relaxed)
GOAL_SCHEDULE = [(0, (6, 6)), (300, (1, 6))]
N_STEPS = 1800


def main():
    out_dir = Path("research/findings/raw/g9")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in (42, 43, 44):
        out_path = out_dir / f"g9_moving_relaxed_argmax_seed{seed}.json"
        print(f"\n{'='*70}")
        print(f"RELAXED MOVING-GOAL: seed={seed} (n_steps={N_STEPS}, phase2=1500 steps)")
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
            verbose=True,
        )


if __name__ == "__main__":
    main()
