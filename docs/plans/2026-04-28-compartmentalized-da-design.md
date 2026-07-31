---
type: plan
status: live
date: 2026-04-28
---

# Cheat #5 closure attempt: Compartmentalized Dopamine — Design

**Status:** design, on standby. Activated only if options 1 and 2 both fail.
**Survey context:** [`2026-04-28-cheat5-real-options-survey.md`](2026-04-28-cheat5-real-options-survey.md) — option 3 of 3.

## Goal

Replace the global scalar `current_reward_signal` with a **per-action vector** `da_per_action[N, E, S, W]`. Cross-projection synapses are tagged with the *target* action they contribute to (cortex_X → str_D1_Y synapses get tagged with action Y). The plasticity rule reads `eligibility[i] * da_per_action[synapse_action_tag[i]]` instead of `eligibility[i] * reward_scalar`. This gives each cross-projection synapse a teaching signal specific to the action it contributes to, not a global "good/bad."

## Why this might work

v4 showed cross-projections develop a uniform weight pattern (mean ~11, std ~0.5 across seeds). One interpretation: the global DA signal can't differentiate "synapse contributing to N when N was the right action" from "synapse contributing to E when N was the right action". Both get the same +1 reward signal during eligibility hold, so they evolve in lockstep.

With per-action DA, only the synapses whose *target* matches the chosen action receive the reward; synapses pointing to other actions stay quiet (or receive no signal during this trial). Over many trials with varied actions, each cross-projection synapse develops a weight specific to its target.

This is the most biologically faithful of the three options — real DA is spatially structured, not a global scalar. But it's also the most invasive to existing code.

## Architecture

### Per-synapse action tag

Add `cp_synapse_action_tag` (int8 GPU array, shape (nnz,)). For each synapse:
- Same-action `cortex_X → str_D1_X`: tag = X (action index 0-3).
- Cross-projection `cortex_X → str_D1_Y` for X≠Y: tag = Y (the *target* action).
- Lateral inhibition `str_D1_X → str_D1_Y`: tag = -1 (no DA gating; static pathway).
- All other synapses: tag = -1 (no DA gating; uses global reward as before).

Built once at network construction. Determined by `RegionPathway.from_region` and `to_region` plus a parser to extract the action from the region name (e.g. `"str_D1_E"` → action E → tag = 1).

### Per-action DA vector

Replace `bridge.core_config.current_reward_signal: float` with `bridge.core_config.current_reward_signal_per_action: list[float]` (length 4) when `enable_compartmentalized_da` is set.

Backward compatibility: keep `current_reward_signal` working as before; when compartmentalized DA is on, the runner sets `current_reward_signal_per_action[chosen_action] = reward` and the others to 0. When OFF, the existing behavior (scalar signal applied uniformly) holds.

### Plasticity rule update

In the per-step plasticity rule (in `bridge._run_one_simulation_step` or wherever STDP+reward applies), gate the reward term by tag:

```python
# Currently: weight_update_from_reward = eligibility[i] * current_reward_signal
# With compartmentalized DA:
tag = cp_synapse_action_tag[i]
if tag >= 0:
    da = current_reward_signal_per_action[tag]
else:
    da = current_reward_signal  # fall back to global for untagged synapses
weight_update_from_reward = eligibility[i] * da
```

This is per-synapse logic. Implementing efficiently requires a vectorized GPU kernel — likely a small fused kernel that takes the tag array, the DA-per-action vector, and the eligibility, and produces the per-synapse update.

### Setting DA per action

In the runner's per-trial reward block, replace:
```python
bridge.core_config.current_reward_signal = float(reward)
```
with:
```python
if cfg.enable_compartmentalized_da:
    da_vec = [0.0, 0.0, 0.0, 0.0]
    da_vec[action_idx] = float(reward)
    bridge.core_config.current_reward_signal_per_action = da_vec
else:
    bridge.core_config.current_reward_signal = float(reward)
```

After the reward hold steps, reset both to zero.

## Why this is hardest

Three things make this option more invasive than 1 and 2:

1. **Touches the inner plasticity hot path.** Options 1 and 2 add side machinery; option 3 *changes* the per-synapse update rule that runs every sim step. Bigger correctness risk; needs careful testing.
2. **Per-synapse tags require parser machinery.** Walking pathways and extracting action indices from region names is straightforward but error-prone. Need tests that verify tags are assigned correctly.
3. **DA-per-action vs global reward both supported.** Backward compatibility means dual code paths. Risk of subtle divergence when both code paths exist.

## Implementation outline

1. Add `cp_synapse_action_tag` allocation in bridge (default -1 for all synapses; populated by a builder that walks `RegionPathway` data after `inject_explicit_wiring`).
2. Add `cfg.enable_compartmentalized_da: bool = False`.
3. Add `bridge.core_config.current_reward_signal_per_action` (4-vector, default all-zeros).
4. Modify the inner plasticity reward kernel to gate by tag.
5. CLI flag `--enable-compartmentalized-da`.
6. Update the runner's per-trial reward block.
7. Tests:
   - `test_compartmentalized_da_tags_assigned` — verify each cross-projection synapse has the correct target-action tag.
   - `test_compartmentalized_da_only_target_action_updates` — set DA for action E only, verify only synapses tagged E receive plasticity updates.
   - `test_compartmentalized_da_default_off` — flag off → identical to current behavior.
   - `test_compartmentalized_da_pretraining_smoke` — end-to-end short pretraining, verify behavior changes are bounded (not e.g. NaN explosions).

## Predicted effect

Cross-projection synapses develop weight distributions that differ across target actions. STDP+reward can refine "context X → action Y" independently for each Y. This is the *minimum* mechanism that lets cross-projections encode action-transition information.

## Risks

- **Reward signal too sparse**: with per-action DA, each synapse only sees a teaching signal on a fraction of trials (the trials where its target action was chosen). If pretraining steps are too few, surviving cross-projection synapses may not have integrated enough signal. Need longer pretraining than option 1 might.
- **Reward attribution is wrong**: assigning DA to the *target* action assumes the synapse contributed to that action. But during cross-action competition (lateral inhibition), the chosen action *suppresses* the others. A cross-projection synapse pointing to a *suppressed* action may have *prevented* its activation — for which the right teaching signal is the inverse. This is a subtle question. The simple version (DA = chosen action's reward) is a starting point; if it doesn't work, may need more complex attribution.
- **Implementation complexity**: changing the inner plasticity kernel is invasive. Need careful before/after comparison on a v3-only smoke run to confirm the dual-path machinery doesn't break the global-DA path.

## Validation tiers

Same as options 1 and 2:
- Tier 1: 1 seed × 1K pretraining + 1800 eval. Compartmentalized DA on, structural pruning off, patch-matrix off.
- Tier 2: 3 seeds × 5K pretraining + 1800 eval.
- Tier 3: 6 seeds × 30K pretraining + 1800 eval (only if Tier 2 ≤ 4.5).

May also try option 3 + option 1 (compartmentalized DA + structural pruning) if option 3 alone is partial — a fully complete biology-grounded model.

## Out of scope

- Spatially-structured DA (different DA *concentrations* in different brain regions). The action-specific DA we're modeling is closer to the "DA fluctuates with reward" picture; real DA also has spatial structure (D1 vs D2 differential receptor density, regional axon termination patterns, etc.). Out of scope.
- Tonic vs phasic DA differentiation. We model only phasic. Tonic DA is its own subsystem.
- Action-specific *reward gradient* (RPE per action). Currently we use scalar reward; per-action RPE would be a further refinement.

## Done criteria

- [ ] All 4 unit tests pass
- [ ] Tier 1 smoke verifies tag assignment is correct (cross-projection synapses tagged with target action)
- [ ] Tier 2 signal check
- [ ] Tier 3 if promising
- [ ] Findings doc + propagation

## Combinations worth testing

If options 1 and 2 both partial, the natural follow-ups are combinations:
- Option 3 + Option 1 (compartmentalized DA + structural pruning)
- Option 3 + Option 2 (compartmentalized DA + patch-matrix)
- All three (the "full biology stack")

These are out-of-scope for the initial cheat-5 closure attempt — try each option in isolation first.
