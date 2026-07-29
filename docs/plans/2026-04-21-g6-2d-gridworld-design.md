# Design: G6 — 2D gridworld sensorimotor navigation

**Date:** 2026-04-21 (same day, autonomous continuation after G5.v3 GO)
**Status:** Approved (operator pre-authorized autonomous progression)
**Scope:** G6 — scale the signed-perceptron sensorimotor approach from G5.v3 to a 2D gridworld with 4-way movement. Real test of whether the reservoir + signed-delta readout head generalizes past a single degree of freedom.

---

## 1. Why this next

G5.v3 closed GO on 1D navigation: 2/3 seeds clear the Q1→Q4 improvement threshold and all 3 reach goal. The mechanism (signed perceptron on hidden→motor, runner-side, reservoir frozen) is simple and clean. But 1D has only 2 possible actions — at minimum even random tie-breaking gets to goal 50% of the time on any given step given some reasonable bias. A 4-direction 2D world is a meaningfully harder learnability test:

- Action space × 2 (2 → 4).
- Sensor state × N (1 position → up to 64 positions on an 8×8 grid).
- Reward ambiguity: on the "wrong" axis, all 4 actions can produce reward=0 (no change in L2 distance) because the world only moves 1 cell per action. Need Manhattan-based reward with proper cardinal decomposition.

## 2. Architecture

**Network** — reuse G5.v3's reservoir with a widened motor head:

- 64 input neurons (2D position-coded Poisson, see §3.1)
- 160 hidden excitatory (trait 0)
- 40 hidden inhibitory (trait 1)
- **4 motor neurons** (trait 0) — one per cardinal direction: N=0, E=1, S=2, W=3
- Total: 268 neurons.

**Connectivity** — same densities/weights as G5.v3:
- input → hidden: 0.5 density, weight ~Normal(1.5, 0.3)
- hidden → hidden: 0.1 density, weights depend on exc/inh trait
- hidden → motor: 0.5 density, weight ~Normal(1.0, 0.2)
- All plastic=False EXCEPT hidden→motor (plastic=True, runner-managed)

**Sim config** — identical to G5.v3: Izhikevich, STDP off, reward_modulation off, homeostasis off, prop=1.0, OU σ=60 pA. `strict_step_errors=True`.

## 3. Task

**World**: 8×8 grid. `pos = (x, y)` with `0 ≤ x, y ≤ 7`. Goal at `(6, 6)`. Start at `(1, 1)`.

**Per step** (same cadence as G5.v3):
1. Present 150 ms stimulus (position-coded).
2. Read motor spike counts in [50, 150] ms.
3. `action = argmax(motor_counts)` (tie-break deterministic RNG per step — same as G5.v3).
4. Apply move: N=(0,+1), E=(+1,0), S=(0,−1), W=(−1,0). Clip to `[0, 7]` per axis.
5. `reward = sign(dist_before − dist_after)` where dist = Manhattan `|Δx| + |Δy|`.
6. Perceptron delta on hidden→motor synapses, same rule as G5.v3 but over 4 posts instead of 2:
   - if reward > 0: target = chosen action; potentiate `h → chosen`, depress `h → other three`.
   - if reward < 0: target = any of the 3 non-chosen directions. **Problem:** which one? See §4.
   - if reward == 0: skip (agent at boundary OR moved perpendicular to goal).

### 3.1 Position encoding

Split the 64 input neurons into two 32-neuron sub-populations:
- Neurons 0..31 tuned to `x` with Gaussian receptive fields over `x ∈ [0, 7]`.
- Neurons 32..63 tuned to `y` with Gaussian receptive fields over `y ∈ [0, 7]`.

Keep the `_position_to_rates` helper from G5, call it twice per step and concatenate. Receptive field σ = 1.5, peak 30 Hz, floor 1 Hz.

## 4. Design wrinkle — reward=-1 target ambiguity

In 1D, "reward<0 means I should have gone the other way" unambiguously identifies the target (only one other action). In 4-way 2D, reward<0 means "the direction I went wasn't the best" — but the correct target could be any of the other three. Plausibly only one (the true goal-direction), but sometimes two (if goal is on a diagonal).

**Three candidate fixes**, ranked by complexity:

**A. Skip negative-reward updates entirely.** Only apply perceptron delta on `reward > 0`. Slower convergence but always unambiguous.

**B. Split negative reward across the 3 non-chosen motors.** For reward < 0, potentiate each of the 3 non-chosen by `lr/3 × h_act`, depress the chosen by `lr × h_act`. Preserves the "the chosen action was wrong" signal, hedges on which specific alternative was right.

**C. Use the two cardinal components of the goal direction.** Compute `sign(goal_x − x)` and `sign(goal_y − y)`. Those identify up to 2 "correct" directions. Potentiate each, depress the other two. This uses goal-direction info the runner already has; cheap.

**Pick: B** for G6.v1. Why:
- Preserves the "perceptron learns from trial-and-error" quality. C leaks an oracle-like signal (the runner knows the goal direction).
- A may be too slow — 2D has 4× the action space, we don't want to cut training signal in half too.
- B is a clean extension of G5.v3's rule with minimal additional reasoning.

If B converges slowly, fall back to C (or explicit A-vs-B-vs-C comparison).

## 5. Success criteria

**GO:**
- Mean Manhattan distance in Q4 < Q1 by ≥1.5 units, in ≥ 2/3 seeds.
- At least one seed reaches goal (dist == 0) at least 5 times per episode.
- Reservoir drift still 0.

**NO-GO:**
- Q1 → Q4 delta < 0 (agent gets worse) in all 3 seeds.
- Zero goal reaches across all seeds.

**PARTIAL:**
- One seed learns clearly; others flat. Acceptable; document and try alternate rule (C).

## 6. Episode length + cadence

- `n_steps = 600`. Bigger grid (8×8 = 64 cells Manhattan diameter = 14) needs more room than the 1D (16 cells diameter = 15 but only 2 actions).
- Per step: still 150 ms stim + implicit gap (covered by the perceptron-update logic, which doesn't need its own reward-delivery window since the sim isn't doing the learning).
- Seeds {42, 43, 44}.
- Estimated runtime per seed: ~3–4 min (50% more than G5.v3 because of extra steps + slightly larger network). 3 seeds ~10–15 min total.

## 7. Hyperparameters

- `learning_rate = 0.01` (same as G5.v3)
- `lr_schedule = "decay_after_goal"` with factor 0.25 (assuming G5.v3's new LR schedule holds up — if the decay probe currently running shows improvement for seed 44 without hurting 42/43, use it here too).
- `w_max = 3.0`
- Plastic hidden→motor density 0.5 → ~100 plastic synapses (vs ~50 in G5.v3 because of double motor count).

## 8. Artifacts to add

- `research/runners/g6_runner.py` — ~400 lines (~20% bigger than G5.v3 because of 2D encoder + 4-way motor + negative-reward splitting).
- `tests/test_g6_runner_smoke.py` — short episode, verify trajectory stays in bounds + reservoir drift 0.
- `research/findings/2026-04-21-g6.md` — results.
- `research/findings/raw/g6-seed{42,43,44}.json`.

## 9. Branch decision

**Stays on `main`.** Same "runner-only, no sim internals" discipline. Reuses G5.v3's reservoir config and signed-perceptron update logic with the 4-motor variant of the rule.

## 10. Out of scope for G6

- Moving goal (G6.v2 or G7).
- Multiple goals or sub-goal hierarchy.
- Memory / recurrent sensory state (just current position).
- Obstacles.
- Variable episode length / terminate on goal.

All save for later gates once this base converges.
