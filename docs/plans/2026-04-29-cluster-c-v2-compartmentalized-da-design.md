---
type: plan
status: live
date: 2026-04-29
---

# Cluster C v2 — Compartmentalized DA Design

**Date:** 2026-04-29
**Goal:** Replace the scalar reward signal (and Cluster C v1's single tonic-DA modulator) with **per-action DA channels**, so plasticity at action-X synapses is driven only by action-X DA. Catalog T2.D / cheat-5 option 3.
**Why:** Cluster C v1 implements tonic + phasic DA, but DA still applies uniformly to all synapses regardless of target action. Real BG has compartmentalized DA — DA axons targeting striatal patches matched by action selectivity. Per-action DA decouples reward credit assignment by action channel.

## Architecture

### v1 (current)
- Single `dopamine` neuromodulator
- Scalar concentration → uniform `plasticity_rate_multiplier` across all synapses
- Applied to `weight_updates` regardless of synapse's target

### v2 (this design)
- 4 per-action DA modulators: `dopamine_N`, `dopamine_E`, `dopamine_S`, `dopamine_W`
- Each synapse tagged with `action_tag ∈ {0, 1, 2, 3}` based on its target region's action channel
- Reward modulation: weight_updates[synapse_i] *= plasticity_rate_multiplier(dopamine_{action_tag_i})
- Reward signal arrives per-action: when reward fires, only the *selected* action's DA bursts (per Schultz / Montague specificity)

## Implementation pieces

### 1. Per-synapse action tag

New GPU array: `cp_synapse_action_tag: int32[capacity]`. Populated at wiring-injection time:
- For synapses targeting `str_D1_X / str_D2_X / cortex_X / motor_X / thal_X / gpi_X / gpe_X / etc`: tag = action index (N=0, E=1, S=2, W=3)
- For synapses targeting non-action-specific regions (`stn`, `dopamine`, `dopamine_X`, `sensory`, `place_cells`): tag = -1 (no action)
- `BrainRegion` adds optional `action_index: Optional[int] = None` field; populated at builder time.

### 2. Per-action DA modulators

Register 4 modulators in `g11_bg_runner` when `--enable-compartmentalized-da` is on:

```python
def _default_per_action_dopamine_config(action: str, action_index: int):
    return NeuromodulatorConfig(
        name=f"dopamine_{action}",
        baseline=0.5, decay_tau_ms=200,
        concentration_min=0.0, concentration_max=2.0,
        targets=[
            ModulatorTarget(target_type="plasticity_rate",
                          scope=f"action:{action_index}", sensitivity=+1.0),
        ],
        production_rules=[
            ProductionRule(
                rule_type="from_action_specific_reward",
                sensitivity=+1.0, threshold=0.0,
                # source_action: this DA only responds when reward arrives
                # AND the agent's selected action matches.
            ),
        ],
    )
```

### 3. New `ModulatorTarget.scope = "action:N"`

Add scope value `"action:N"` (where N ∈ {0,1,2,3}) to indicate "synapses with action_tag == N." `compute_plasticity_rate_multiplier()` becomes per-synapse instead of scalar — returns a cp array indexed by synapse, OR a dict `{action_index: float}` for per-action multipliers.

Cleanest API: `compute_per_synapse_plasticity_rate_multiplier(action_tag_array) -> cp.ndarray`.

### 4. New `ProductionRule.rule_type = "from_action_specific_reward"`

Reads `bridge.core_config.current_reward_signal` AND `bridge.core_config.last_selected_action` (new field). Produces sensitivity * (reward - baseline) ONLY if the action matches.

### 5. Bridge: per-synapse plasticity_rate

In bridge.py reward modulation block:
```python
# Currently (v1):
da_signal = self.neuromodulator_manager.get_concentration("dopamine") - dopamine_baseline
weight_updates = effective_reward_lr * da_signal * eligibility[:n]

# v2 (compartmentalized):
if compartmentalized:
    # per-synapse signal based on action tag
    da_signal_per_synapse = compute_per_synapse_da_signal(self.cp_synapse_action_tag[:n])
    weight_updates = effective_reward_lr * da_signal_per_synapse * eligibility[:n]
else:
    # v1 path
    da_signal = ... (scalar)
    weight_updates = ... (scalar)
```

### 6. Runner reports selected action

Already done — runner stores `action_log[step]`. Just need to expose `bridge.core_config.last_selected_action` so the production rule can read it.

## Validation

### Smoke test
- `--enable-compartmentalized-da` + `--enable-tonic-da` (or as standalone replacement for v1):
  - 4 DA modulators register
  - cp_synapse_action_tag populated correctly
  - smoke run completes

### Cheat-5 multi-goal eval (n=3)

Compare:
- baseline (post-R, no Cluster C)
- + Cluster C v1 (single tonic DA)
- + Cluster C v2 (compartmentalized)
- A + C v2 (closed loop + per-action DA — strongest hypothesis)

If v2 helps where v1 doesn't, that confirms per-action specificity is necessary. If neither helps, signal that the bottleneck is elsewhere (architecture, eval methodology, or cross-projection initialization).

## Estimated effort

- 4-6 hours (substantial: new GPU array, new manager methods, new production rule type, bridge change)
- 4 conditions × 3 seeds eval: ~120 min

## Composition

Cluster C v2 + Cluster A is the strongest theoretical combo:
- A provides closed-loop teaching signal for STDP at cross-action synapses
- C v2 provides action-specific DA so cross-action plasticity is gated to "did this specific action just succeed?"

If A alone or A+C v1 doesn't close cheat-5, this is the next attempt.

## Decision tree (after current evals)

```
Cluster A eval → +A helps mean ≤ -1.0?
├── YES: tier-3 (6-seed) validation → DONE if validates
├── PARTIAL: try A + C v1 (current combo eval)
│   ├── helps: tier-3 validation
│   └── doesn't: build C v2 (this plan), test A + C v2
└── NO: Cluster A alone doesn't help; investigate why
    └── try A + C v2 anyway, then if no help, deeper architectural investigation
```

This plan is the FALLBACK / NEXT STEP if the current evals don't show GO signal.
