# Session G v4 (contingency): Action-Attribution Eligibility

> **Status:** Drafted as contingency. Only launch if V3 (positive-only reward
> + exploration) also fails to break the silent-motor trap.

## Motivation

V1+V2+V3 isolated three layers of the silent-motor trap:
- **Layer A (eligibility absence):** silent motors can't form eligibility traces
- **Layer B (action selection lock-in):** entrenched winners dominate argmax
- **Layer C (action-blind reward):** global reward hits all eligibility,
  not just the chosen motor's

V1/V2 (motor exploration noise + WTA variants) fix Layer A but leave B/C.
V3 (positive-only reward) addresses Layer C partially — at least no longer
*depresses* W's eligibility — but still doesn't *selectively reinforce* it.

V4 directly attacks Layer C with surgical action attribution: when an
action `a` is chosen, zero eligibility for synapses targeting non-`a`
motors before the reward signal applies. The reward then updates *only*
synapses leading to the motor that actually executed the action.

This is a runner-side hack, but it's the cleanest test of the hypothesis.
A biologically-realistic version would use lateral inhibition or selective
DA delivery — but the math is the same.

## Implementation

In `research/runners/g9_runner.py`:

1. After computing `i2m_flat_indices`, also compute per-motor index arrays:

```python
i2m_per_motor = []  # list of cupy int64 arrays
for m_neuron in layout["motor_idx"]:
    mask_m = i2m_mask & (post_h == m_neuron)
    i2m_per_motor.append(cp.asarray(np.where(mask_m)[0], dtype=cp.int64))
```

2. New kwarg `action_attribution_eligibility=False` (default = backward compat).

3. In the per-step loop, after picking `action` and before the reward-hold
   steps, if the kwarg is True:

```python
if action_attribution_eligibility:
    for m_idx in range(n_motor):
        if m_idx == action:
            continue
        bridge.cp_eligibility_trace[i2m_per_motor[m_idx]] = 0.0
```

4. Save `action_attribution_eligibility` to results JSON.

## V4 Probe Design

`research/run_g9_motor_exploration_v4.py`:
- 3 conditions × 3 seeds = 9 runs
- All conditions use motor_exploration_rate_hz=15, argmax
- (a) `attr_off`: action_attribution_eligibility=False, positive_only_reward=False (= V1 baseline reproduction)
- (b) `attr_on_bipolar`: action_attribution_eligibility=True, positive_only_reward=False
- (c) `attr_on_posonly`: action_attribution_eligibility=True, positive_only_reward=True

Pass criteria for (b) or (c):
1. All 4 motors active in Phase 1
2. Phase 1 finalQ < 4 in ≥2/3 seeds
3. W usage rises substantially over Phase 1 (vs baseline near-zero)

## Why this should work

When E wins and goes wrong way (under bipolar reward + attribution):
- Negative reward applies ONLY to hidden→E synapses
- Hidden→W eligibility (from noise) is preserved
- Over time, hidden→W's *positive* eligibility gets selectively potentiated
  when W happens to win and go right way
- Hidden→E's *negative* eligibility gets depressed each time E goes wrong
- Net: E weights shrink, W weights grow

Combined with positive-only reward, this is even cleaner: W only grows,
E stays put when wrong, E only grows when correct (rare in phase 1 since
correct is now W).

## Trade-offs

- This is RUNNER-SIDE attribution, not biologically grounded. A real brain
  uses lateral inhibition or selective DA delivery.
- Requires knowing the chosen action (which the runner does have).
- Doesn't help if the architecture has multiple "competing populations"
  beyond what the runner explicitly tracks.

## Estimated wall time

9 runs × 14 min = 126 min (~2 hours). Worth it if V3 fails.
