---
type: plan
status: live
date: 2026-04-20
---

# Design: G5 — Sensorimotor loop

**Date:** 2026-04-20 (sketched overnight)
**Status:** Draft — prerequisite is at least G2 PARTIAL
**Scope:** Gate G5 — the brain's motor output modulates the next sensory input, forming a closed loop. Verify that intrinsic plasticity / dynamics produce different trajectories from the same initial state. The artificial-life foothold.

---

## 1. Context

G1 (pipeline), G2 (sim-local learning), G3 (persistence), G4 (generalization) are all offline-learning gates where the sim consumes a static dataset. G5 breaks the offline-ness: a single long-running episode where **what the brain does changes what it sees next**.

## 2. Task

Smallest viable closed loop: **1D gridworld navigation.**

- State: position `x ∈ {0, ..., 15}` on a 16-cell line.
- Sensor: `x` encoded as a 64-dim Poisson rate vector — neurons have Gaussian receptive fields over positions, so neurons tuned to the current `x` fire at ~30 Hz, distant neurons at ~1 Hz.
- Motor: 2 output action neurons (move left / move right). Read spike counts in the readout window; argmax → action.
- Environment update: `x += {-1, +1}[action]`, clipped to `[0, 15]`. Optionally wraps around.
- Episode: 200 timesteps (i.e. 200 actions). Each action gets a 150 ms sensory window + 50 ms motor readout.

No reward signal. No training. Just: let the brain move around and measure what it does.

## 3. Architecture

Reuse the 264-neuron reservoir from G1.v3. Add:
- **64 input neurons**: frontal sensor (already exist — re-interpret the input group as position-coded).
- **2 output neurons**: motor, downstream of hidden.
  - `move_left` = neuron index 264
  - `move_right` = neuron index 265
- **Hidden → output** projection: sparse, fixed weights (seeded differently per run) OR optional plasticity.

Total: 266 neurons.

## 4. Protocol

```
for step in range(n_steps):
    rate_vec = position_to_rates(x)   # 64-dim
    present(rate_vec, STIMULUS_MS)    # 150 ms stim
    count_motor = read output neurons in [100, 150] ms
    action = argmax(count_motor)       # 0 = left, 1 = right
    x = clip(x + (1 if action==1 else -1), 0, 15)
    record_trajectory.append(x)
    record_motor_counts.append(count_motor)
```

Run for 200 steps per seed. Dump trajectory.

## 5. Success criteria

**GO:**
- Seeds {42, 43, 44} produce **statistically different trajectories** (Kolmogorov-Smirnov or simple trajectory-distance metric).
- Each seed's motor output changes systematically with position (correlation > 0.1 between local sensory activity and motor choice).

**NO-GO:**
- All seeds produce identical trajectories → there's no intrinsic variation (bug in RNG seeding).
- Or the brain is unresponsive — motor output is constant regardless of sensory input.

**PARTIAL:**
- Different trajectories but the mapping from sensor to motor looks random (correlation close to 0). This would still satisfy "motor changes what it sees next" as a minimal loop, but wouldn't be *meaningful* sensorimotor behaviour.

## 6. Why this is the artificial-life foothold

Offline gates can be satisfied by a static classifier — the brain is treated as a feature extractor. G5 is different: the **sequence** of sensory inputs is caused by the brain's own outputs. This is the precondition for:
- Intrinsic motivation (brain seeks certain sensory patterns)
- Embodiment (brain's "body state" is recursively defined)
- Development (repeated interaction shapes trajectories, which shapes plasticity, which shapes interaction…)

Even without reward or task, running G5 lets us measure whether the brain has **persistent behavioural structure** that different seeds reliably produce.

## 7. Deliverables

- `research/runners/g5_runner.py` — episode loop + trajectory recording.
- `research/findings/2026-04-20-g5.md` — trajectory plots (matplotlib), statistical analysis of per-seed variation.
- One test: `tests/test_g5_runner_smoke.py` — run 10 steps, verify position changes and motor counts are recorded.

## 8. Branch decision

Stays on `main`. Uses existing sim primitives (reservoir, rate_vector stimulus, spike count readout). The sensorimotor wrapper is a thin loop on top of the existing runner infrastructure.

## 9. What's NOT in G5

- Learning during the loop (that's G5.v2 — intrinsic motivation / reward).
- Multi-dim state or 2D gridworld (G6+).
- Goal-directed behaviour (G6+).
- Physical embodiment simulation (out of scope for this project).
