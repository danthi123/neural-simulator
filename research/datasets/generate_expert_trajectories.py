"""Generate expert trajectories from a hand-coded greedy policy.

Used as input data for `g11_bg_trajectory_train.py` (imitation-learning
mode). The hand-policy is Manhattan-greedy: always pick an action that
strictly reduces Manhattan distance to the goal; tie-break randomly.

Output JSON shape:
    {
      "name": "expert_8x8_v1",
      "grid_size": 8,
      "trajectories": [
        {"trajectory_id": 0, "goal": [6,6],
         "steps": [{"state":[1,1],"action":0,"reward":0.0}, ...]},
        ...
      ]
    }

action ∈ {0=N, 1=E, 2=S, 3=W} matching ACTION_DELTAS in g11_bg_runner.py.
reward ∈ {-1, 0, +1} with the same semantics as the runner: +1 if action
shrinks Manhattan distance, -1 if it grows, 0 if unchanged (boundary).

Usage:
    python -m research.datasets.generate_expert_trajectories \\
        --grid-size 8 --n-trajectories 100 --max-steps 50 \\
        --output research/datasets/expert_8x8_v1.json --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

ACTION_NAMES = ["N", "E", "S", "W"]
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pick_greedy_action(
    state: Tuple[int, int],
    goal: Tuple[int, int],
    rng: random.Random,
) -> int:
    """Pick a Manhattan-greedy action; tie-break uniformly."""
    x, y = state
    gx, gy = goal
    candidates = []
    if gy > y: candidates.append(0)  # N
    if gx > x: candidates.append(1)  # E
    if gy < y: candidates.append(2)  # S
    if gx < x: candidates.append(3)  # W
    if not candidates:
        # at goal — pick a random no-op-like action (will get reward=0)
        return rng.randint(0, 3)
    return rng.choice(candidates)


def generate_trajectory(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid_size: int,
    max_steps: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate one expert trajectory from start to goal."""
    x, y = start
    gx, gy = goal
    steps = []
    for _ in range(max_steps):
        if (x, y) == (gx, gy):
            break  # reached the goal — stop
        action = pick_greedy_action((x, y), goal, rng)
        dx, dy = ACTION_DELTAS[action]
        new_x = max(0, min(grid_size - 1, x + dx))
        new_y = max(0, min(grid_size - 1, y + dy))
        d_before = manhattan((x, y), goal)
        d_after = manhattan((new_x, new_y), goal)
        if d_after < d_before:
            reward = 1.0
        elif d_after > d_before:
            reward = -1.0
        else:
            reward = 0.0
        steps.append({
            "state": [x, y],
            "action": int(action),
            "reward": float(reward),
        })
        x, y = new_x, new_y
    return steps


def generate_dataset(
    grid_size: int,
    n_trajectories: int,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Generate `n_trajectories` random (start, goal) trajectories."""
    rng = random.Random(seed)
    trajectories = []
    for tid in range(n_trajectories):
        # Random start + goal that aren't equal
        while True:
            start = (rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1))
            goal = (rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1))
            if start != goal:
                break
        steps = generate_trajectory(start, goal, grid_size, max_steps, rng)
        trajectories.append({
            "trajectory_id": tid,
            "goal": list(goal),
            "start": list(start),
            "steps": steps,
        })
    return {
        "name": f"expert_{grid_size}x{grid_size}_n{n_trajectories}",
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "max_steps_per_trajectory": max_steps,
        "seed": seed,
        "policy": "manhattan_greedy",
        "trajectories": trajectories,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--n-trajectories", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    dataset = generate_dataset(
        grid_size=args.grid_size,
        n_trajectories=args.n_trajectories,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2))

    n_steps = sum(len(t["steps"]) for t in dataset["trajectories"])
    print(f"Generated {len(dataset['trajectories'])} trajectories "
          f"({n_steps} total steps) -> {args.output}")


if __name__ == "__main__":
    main()
