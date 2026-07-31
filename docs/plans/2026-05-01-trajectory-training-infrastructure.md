---
type: plan
status: live
date: 2026-05-01
---

# Train.1 — Trajectory Training Infrastructure (Imitation Learning via STDP+Reward)

**Date:** 2026-05-01
**Status:** DESIGN
**Purpose:** Allow the sim to learn from pre-recorded expert trajectories instead of (or in addition to) live RL. Foundation for "train it on datasets/behavior" capability per user directive.

## Goal

A new runner mode `g11_bg_trajectory_train.py` that:
- Loads a JSON file of expert trajectories `[{state, action, reward, next_state}]`
- For each step, drives `cortex_X` corresponding to the recorded action
- Sets reward signal from the recorded reward
- Runs the simulation step (STDP + reward modulation update weights)
- Bypasses heuristic AND BG action selection — actions are imposed, not sampled

This is **imitation learning via biology-grounded plasticity**: the agent's BG cascade learns to associate (state → action) by being driven through expert trajectories with reward feedback.

## Use cases

1. **Bootstrapping**: train a sim from scratch on expert trajectories, then run live for the first time. Tests whether STDP+reward can absorb supervised data.
2. **Generalization tests**: train on dataset A, eval on task B. Tests the sim's generalization capability.
3. **Behavioral cloning baseline**: a way to compare against "raw RL from scratch" results.
4. **Foundation for richer datasets**: animal behavior datasets (e.g. mouse navigation), human-recorded LLM trajectories.

## Biology grounding

This is consistent with how real animals learn:
- **Imitation**: monkeys learning from observation (mirror neurons; Rizzolatti & Craighero 2004)
- **Supervised motor learning**: cerebellar PF→PC LTD with climbing-fiber teaching signal (Albus 1971; we already have this in F v2)
- **Replay**: hippocampal trajectory replay during NREM (Foster & Wilson 2006)

The "trajectory" framing is the same as our existing sleep replay infrastructure — we already have `successful_trajectories` as a buffer that gets replayed during NREM. Train.1 is essentially "replay-only mode": run only the replay phase, no live wake mode.

## Implementation

### v1 minimal (this design)

Create `research/runners/g11_bg_trajectory_train.py`. Reuse most of `g11_bg_runner.py`'s setup (regions, pathways, bridge init), but replace the env loop with a trajectory-driven loop.

### Input format

JSON file with shape:
```json
{
  "name": "expert_navigation_8x8",
  "grid_size": 8,
  "trajectories": [
    {
      "trajectory_id": 0,
      "goal": [6, 6],
      "steps": [
        {"state": [1, 1], "action": 0, "reward": 0.0},
        {"state": [1, 2], "action": 1, "reward": 1.0},
        ...
      ]
    },
    ...
  ]
}
```

`action` is 0-3 (NESW). `reward` matches our existing reward log format. `state` is (x, y).

### Generator: expert hand-policy

For testing v1, generate expert trajectories from a hand-coded greedy policy:
- For each (start, goal) pair, simulate the optimal Manhattan path
- Tag each step with the action taken and a +1/-1 reward based on whether distance shrunk

```python
def generate_expert_trajectory(start, goal, grid_size, max_steps=50):
    """Greedy Manhattan-optimal path from start to goal."""
    x, y = start
    gx, gy = goal
    steps = []
    while (x, y) != (gx, gy) and len(steps) < max_steps:
        # pick best action
        if abs(gx - x) > abs(gy - y):
            action = 1 if gx > x else 3  # E or W
        else:
            action = 0 if gy > y else 2  # N or S
        # apply
        dx, dy = ACTION_DELTAS[action]
        new_x, new_y = x + dx, y + dy
        d_before = abs(gx - x) + abs(gy - y)
        d_after = abs(gx - new_x) + abs(gy - new_y)
        reward = 1.0 if d_after < d_before else -1.0 if d_after > d_before else 0.0
        steps.append({"state": [x, y], "action": action, "reward": reward})
        x, y = new_x, new_y
    return steps
```

### Training loop

```python
def train_on_trajectories(trajectories, n_epochs, bridge, region_indices, ...):
    for epoch in range(n_epochs):
        for traj in trajectories:
            for step in traj.steps:
                # Drive cortex_X corresponding to imposed action
                cortex_letter = ACTION_NAMES[step.action]
                bridge.cp_external_input_current[region_indices[f'cortex_{cortex_letter}']] = HEURISTIC_DRIVE_PA
                # Set reward from trajectory
                bridge.core_config.current_reward_signal = step.reward
                # Run stim window (200 sim steps with this drive + reward)
                for _ in range(n_stim_steps):
                    bridge._run_one_simulation_step()
                # No action sampling — we use the imposed action
                # No env step — the imposed state is given
```

### CLI

```bash
python -m research.runners.g11_bg_trajectory_train \
    --trajectories expert_8x8.json \
    --n-epochs 10 \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry \
    --enable-striatal-pv-fsi --enable-cluster-a-closed-loop \
    --enable-cluster-e-topography \
    --output-checkpoint trained_state.h5
```

After training, the bridge state can be saved (existing checkpoint mechanism) and loaded into a fresh `g11_bg_runner.py` run for live evaluation.

### Eval

After 10 epochs of training on expert trajectories:
1. Save bridge weights
2. Load into live runner
3. Run cheat-5 multi-goal det without heuristic (`--heuristic-strength 0`)
4. Compare to fresh-init agent on same task

Expected outcomes:
- **Best case**: trained agent beats fresh-init on cheat-5 (imitation learning works)
- **Likely case**: trained agent matches a heuristic-driven agent (imitation absorbed the policy)
- **Worst case**: trained agent doesn't transfer (learned weights are too specific to imposed-action regime)

## Test plan

### Unit tests

- `generate_expert_trajectory`: optimal path length matches Manhattan distance
- `train_on_trajectories`: bridge weights change after training (compared to before)
- JSON I/O: serialize + deserialize round-trip

### Integration test 1: single-trajectory overfit

Train for 100 epochs on a single (start=(1,1), goal=(6,6)) trajectory. Then eval: agent drops at (1,1) with goal (6,6); does it follow the trained path?

Validation: agent reaches goal in ~ optimal step count (Manhattan distance + small slack).

### Integration test 2: multi-goal generalization

Train on 100 trajectories spanning 16 (start, goal) pairs. Eval on 4 held-out goals. Validation: agent reaches held-out goals at near-expert performance.

### Integration test 3: cheat-5 transfer

Train on 1000 expert trajectories sampled from random (start, goal) on 8×8. Eval on standard cheat-5 multi-goal det. Compare to fresh-init agent.

## Effort estimate

**v1 minimal:** ~3-5 days.
- Day 1: trajectory generator + JSON I/O + basic training loop
- Day 2: integration tests 1 + 2
- Day 3: integration test 3 + hyperparameter tuning
- Day 4-5: documentation, checkpoint hookup, findings doc

## Out of scope (defer)

- Multimodal trajectory data (image + audio + state) — needs Cluster K first
- Human-recorded trajectories from real datasets (animal behavior, etc.) — requires data acquisition pipeline
- Adversarial / preference-labeled trajectories
- LLM-generated trajectories

## Files to create

- `research/runners/g11_bg_trajectory_train.py` (NEW, ~300 LOC)
- `research/datasets/expert_8x8_v1.json` (NEW, generated from hand-policy)
- `tests/test_trajectory_training.py` (NEW, ~150 LOC)
- `docs/plans/2026-05-01-trajectory-training-infrastructure.md`: this design

## Decision: implement after Cluster G eval

Sequencing per architecture roadmap:
1. Cluster G v1 (running now)
2. If G v1 lands clean, Cluster K v1 next (visual cortex foundation)
3. After K v1 (or in parallel for non-conflicting parts): Train.1 trajectory training

Alternative: implement Train.1 NOW since it doesn't depend on K and is self-contained. Pro: immediate value, smaller scope, parallelizable. Con: less novel than K.

**Decision: implement Train.1 NEXT after Cluster G eval lands.** It's smaller and unblocks the user's "train it on datasets/behavior" directive.
