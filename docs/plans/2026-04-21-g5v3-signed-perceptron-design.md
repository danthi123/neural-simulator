---
type: plan
status: live
date: 2026-04-21
---

# Design: G5.v3 — Signed perceptron on hidden→motor, reward-driven

**Date:** 2026-04-21
**Status:** Approved (autonomous continuation, operator pre-authorized)
**Scope:** G5.v3 — reward-driven learning in the sensorimotor loop that *actually* converges to the goal, after G5.v2 hit the sim's unsigned-eligibility ceiling.

---

## 1. Gap from G5.v2

G5.v2 used the sim's native reward modulation. The sim's eligibility trace stores `|Δw|` (unsigned). A signed reward signal multiplying an unsigned eligibility produces a monodirectional update — reward < 0 uniformly depresses, reward > 0 uniformly potentiates. When the agent got stuck far from goal, reward stayed negative, all plastic weights decayed to zero, motor went silent, agent pinned at x=0. Degenerate attractor.

## 2. Core design choice

**External signed perceptron, sim plasticity fully disabled.**

The sim becomes a pure forward pass (reservoir dynamics + motor read-out). Learning lives entirely in the runner. Per step:

1. Present position-encoded stimulus.
2. Read hidden spike counts and motor spike counts.
3. Decide action = argmax(motor counts).
4. Update world: `x := clip(x ± 1, 0, 15)`.
5. Compute `reward = sign(dist_before − dist_after)` — `+1` if we got closer, `-1` if we got further, `0` if no change (boundary clip).
6. Apply signed perceptron delta to hidden→motor weights:
   - `target_action = chosen_action` if `reward > 0`, else `other_action` if `reward < 0`, else skip update.
   - `ΔW[h, target_action] += lr × hidden_active[h]`
   - `ΔW[h, other_action] −= lr × hidden_active[h]`
   - Clip to `[0, w_max]` (keep connections non-negative).

**Why this works where G5.v2 didn't:**
- Update is *signed per synapse* (some potentiated, others depressed in the same step).
- Reward direction determines which motor should have fired, not whether weights should grow.
- Reservoir stays fixed, so the feature representation doesn't drift out from under the perceptron.
- Bypasses the sim's reward path entirely, respecting the "don't change the biological science layer" constraint.

## 3. Architecture

Same 266-neuron network as G5 / G5.v2 (64 input + 160 hidden exc + 40 hidden inh + 2 motor). Same fixed reservoir weights per seed. Only change: runner owns the weight updates on hidden→motor.

- **Sim config:** `enable_stdp = False`, `enable_reward_modulation = False`, all other plasticity off. Sim is purely dynamical.
- **Plastic mask:** not strictly needed since sim plasticity is off, but still set (plastic=True for hidden→motor, False for the rest) so that if someone later enables STDP they don't accidentally modify the reservoir.
- **Runner responsibility:** computes and applies `ΔW` directly to `cp_connections.data` at the specific entries corresponding to hidden→motor synapses.

### Index bookkeeping

At setup, compute `i2m_flat_indices` — the indices into `cp_connections.data` where pre ∈ hidden_idx and post ∈ motor_idx. Store as a CuPy array. Also store `i2m_pre` (for each synapse, which hidden neuron is the pre) and `i2m_post_local` (0 or 1 for which motor neuron is the post). Update logic indexes into these.

### Update formula (vectorised on GPU)

```
hidden_active: shape (n_hidden,) int   — spike counts during stimulus window
target = chosen if reward>0 else other  (int in {0, 1})
delta[k]  = +lr * hidden_active[i2m_pre[k]] if i2m_post_local[k] == target
           else -lr * hidden_active[i2m_pre[k]]
cp_connections.data[i2m_flat_indices] += delta
clip to [0, w_max]
```

If `reward == 0`, skip entirely (no update, no contested target).

## 4. Success criteria

**GO:**
- `mean_distance_quarters[3] − mean_distance_quarters[0] ≤ −1.0` in at least 2 of 3 seeds, AND
- At least one seed reaches `dist == 0` (at goal) at least once during the episode.

**NO-GO:**
- Mean-distance delta ≥ 0 in all seeds (no improvement anywhere).
- Agent pinned at a boundary for > 80% of steps in all seeds.

**PARTIAL:**
- One seed shows clear improvement; the others flat.

## 5. Hyperparameters (probe defaults)

- Episode length: 400 steps.
- Learning rate: 0.01 (tunable).
- `hidden_active` normalization: raw spike count; if too noisy, switch to `/ max(spike_count)`.
- Weight clip: `[0, 3.0]` (same as G5.v2).
- Seeds: {42, 43, 44}.

## 6. Test plan

- `tests/test_g5_v3_runner_smoke.py` — 40-step episode, verify:
  - Trajectory recorded.
  - Weights actually change post-update (non-trivial delta).
  - Only hidden→motor weights change — reservoir weights untouched.
- Then run the full 3-seed probe.

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Learning rate too high → weights saturate → motor always one side | Start at 0.01, clip to [0, 3.0], monitor weight histogram per quarter |
| Reservoir doesn't have class-discriminative features → no amount of readout learning helps | Verify at probe: hidden firing patterns at different positions should differ (check by dumping 1 hidden spike matrix per quarter) |
| Tie-break bias (silent motor → action 0 = left) still dominates | Reduce or eliminate silent steps by ensuring motor drive is sufficient; if persistent, switch argmax to a stochastic choice among equally-high counts |
| 400 steps too short to see convergence | Budget allows up to 800 if needed; check quarter analysis |

## 8. Branch decision

**Stays on `main`.** No sim-internals changes. Runner-only code addition. Existing biological-experiment suite untouched.

## 9. What's NOT in G5.v3

- 2D or larger gridworlds.
- Discount factors / value functions / eligibility traces over time.
- Stochastic policy sampling (stays argmax for now).
- Learning rate schedules.
- Goal position changes during episode.

These are G6+ if G5.v3 goes.
