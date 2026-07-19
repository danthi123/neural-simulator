# Per-Action Dopamine Targeting — Same Trade-off as WTA

**Date:** 2026-04-26
**Status:** PARTIAL — kept opt-in. Same exploitation/exploration trade-off as motor lateral inhibition.
**Companion:** [2026-04-26-wta-lateral-inhibition-mixed.md](2026-04-26-wta-lateral-inhibition-mixed.md), [2026-04-25-phase-b-acid-test-real-win.md](2026-04-25-phase-b-acid-test-real-win.md)

## TL;DR

Implemented per-action dopamine targeting via eligibility-trace gating: when the agent selects action X and reward != 0, eligibility traces are zeroed on cortex→str_D1_Y synapses for Y != X *before* reward is delivered. This means reward only credits the cortex→D1 pathway for the action that was actually taken, rather than smearing across all eligible pathways.

Acid test (3 seeds × 1800 steps moving goal) shows the **same exploitation/exploration trade-off** as the WTA finding:

| Variant | Phase 0 finalQ | Phase 1 finalQ | BG-active |
|---|---:|---:|---:|
| Baseline (no-DA / no-WTA) | 3.48 | **1.76** | 22-24% |
| Motor WTA | **2.40 (-31%)** | 2.46 (+40%) | 22-24% |
| **Per-action DA** | **2.04 (-41%)** | 2.61 (+48%) | 22-24% |

**Decision:** kept `--per-action-da` flag opt-in, default OFF. Two independent sharpening mechanisms produced the same pattern; this is structural, not a tuning issue.

## Implementation

`research/runners/g11_bg_runner.py` lines ~410-435:

```python
# At init: pre-compute per-action mask of cortex→D1_X synapses
coo = bridge.cp_connections.tocoo()
post_neurons = coo.col
synapse_post_action = cp.full(n_synapses, -1, dtype=cp.int8)
for action_idx, action_name in enumerate(ACTION_NAMES):
    d1_indices = region_indices_cp[f"str_D1_{action_name}"]
    mask_d1 = cp.isin(post_neurons, d1_indices)
    synapse_post_action[mask_d1] = action_idx
# Cache: per-action mask of "synapses NOT going to action X's D1 pool"
d1_synapse_other_action_masks = {
    a_idx: ((synapse_post_action >= 0) & (synapse_post_action != a_idx))
    for a_idx in range(N_ACTIONS)
}

# Per-trial, before reward hold:
if abs(reward) > 0:
    other_mask = d1_synapse_other_action_masks[action_idx][:cp_connections.nnz]
    bridge.cp_eligibility_trace[:cp_connections.nnz][other_mask] = 0.0
    # ... reward hold steps ...
```

D2 (indirect path) is left untouched — only direct path D1 gets per-action gating. Structural plasticity disabled to keep mask aligned with synapse count.

## Why phase 0 improves

Without per-action DA: when agent picks action N (which causes only cortex_N→D1_N firing) and gets reward, eligibility on cortex_E→D1_E (which had eligibility from cortex_E firing earlier in the trial because goal direction had E component) ALSO gets the reward boost.

With per-action DA: only cortex_N→D1_N gets the boost. Credit is precise. Agent's policy converges faster on the correct action representation.

## Why phase 1 worsens

After goal change from (6,6) to (1,6): agent's learned weights favor N+E (old goal direction). Agent picks N often (N is partially correct for new goal too). Each time agent picks N and gets +1 reward (because N keeps it close to y=6), only cortex_N→D1_N gets reinforced.

But to reach goal=(1,6), the agent ALSO needs to learn cortex_W→D1_W (go left). The old weights make cortex_W weak; the agent rarely picks W; when it does pick W and gets +1, only cortex_W→D1_W gets the boost. But the increment is small (small initial weights × small eligibility) and the boost rate is too slow to overcome the entrenched cortex_N→D1_N strength.

**Without per-action DA**: every reward also boosts cortex_E→D1_E and cortex_W→D1_W if those cortex pools fired during the trial (which they do whenever the goal direction has E or W component). So the broadcast-DA case has a built-in exploration mechanism — eligibility from cortex pools that fired but weren't the chosen action still gets credited.

In other words: broadcast DA is implicitly an exploration regularizer. Removing it removes that regularizer, hence slower readaptation.

## Synthesis with WTA finding

WTA and per-action DA are different mechanisms for the same goal: sharpen the credit signal so the system commits faster to the correct policy. Both produce the same trade-off because they both reduce exploration noise — the agent locks in faster but can't unlock when the world changes.

This is the **classic exploration-exploitation dilemma** showing up directly in the architecture. Real brains solve it via DA/NE-mediated meta-learning: high uncertainty (NE) and unexpected reward signals (DA) modulate sharpening strength on the fly.

## Next experiment: adaptive sharpening

Implementing adaptive per-action DA: gate strength on recent reward stability.
- Recent reward EMA high & stable → tight gating (exploit)
- Recent reward dropped or oscillating → relax gating toward broadcast (explore)

This is a simple form of NE-like meta-modulation without requiring full neuromodulator subsystem extension. Implementation in next finding doc.

## Files

- `research/runners/g11_bg_runner.py:410-435,547-557`: per-action DA implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_perDA.json`: 3-seed acid test data

## Per-seed details

| Seed | Phase 0 finalQ | Phase 1 finalQ |
|------|---:|---:|
| 42   | 1.60 | 3.37 |
| 43   | 1.88 | 2.07 |
| 44   | 2.65 | 2.39 |
| **avg** | **2.04** | **2.61** |
