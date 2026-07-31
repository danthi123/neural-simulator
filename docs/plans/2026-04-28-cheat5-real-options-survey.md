---
type: plan
status: live
date: 2026-04-28
---

# Cheat #5 "real" closure — three architectural options

**Status:** survey. Cheat #5 is currently closed *by design* (2026-04-28, see [`research/findings/2026-04-28-cheat5-v4-results.md`](../../research/findings/2026-04-28-cheat5-v4-results.md)) — v3 MSN lateral inhibition + same-action-only routing achieves the equivalent functional outcome of biological winner-take-all in our reduced model. This survey enumerates what would be needed to close cheat #5 *for real* (i.e., make BG cross-projections behaviorally useful).
⛔ SUPERSEDED THE SAME DAY — the "closed by design" framing above was replaced by **ON HOLD pending a multi-cluster biology buildout** in `research/findings/2026-04-28-cheat5-post-v4-reframe.md` (multi-goal eval was the methodological correction). The v1-v4 NO-GO measurements stand; the conclusion drawn from them was too quick.

**Background:** v1 (curriculum-staged), v2 (zero-init), v3.1 (cross + lateral inhibition + adult thaw), v4 (developmental pretraining) all NO-GO. v4 confirmed the most damning finding: pretraining DID develop cross weights consistently across 3 seeds (mean ~11, std ~0.5 — robust signal, not noise), but the resulting connectivity *degrades* eval performance even on Phase 0. The topology, not the training, is the limit.

## Why three options, in this order

The three missing architectural pieces, in *increasing strength of evidence* given the v4 failure pattern:

1. **Structural plasticity** (axon pruning + synaptogenesis)
2. **Sparse heterogeneous initial topology** (patch-matrix striatum)
3. **Compartmentalized dopamine** (action-specific DA pulses)

I'll explain the ranking, then sketch each.

### Why structural plasticity is #1

v4's smoking gun: pretraining converged to a **uniform, suspiciously consistent** weight distribution (3 seeds → mean ~11, std ~0.5). With dense 100% cross-connectivity and weight-only learning, every cross-pair gets *some* weight that has to be useful, which is over-constrained. Real BG handles this by *eliminating* connections that fire unhelpfully — it's a sparsifying mechanism, not a weight-tuning one. Adding pruning to the existing weight-plasticity machinery is the most direct fix and plausibly addresses all three symptoms (topology becomes sparse and useful; surviving synapses self-select for action transitions that matter; the "everything has to work" over-constraint dissolves).

### Why patch-matrix is #2

Patch-matrix would solve the topology problem by initialization rather than at runtime — a sparser, more anatomically faithful starting point. But it doesn't address the *learning-time* problem: even with sparse init, weight-only refinement still has to find a good operating point in a fixed substrate, and v4 suggests that's hard. Patch-matrix is also more invasive (requires rewriting `build_bg_brain_regions` and reasoning about which cortical pools project where), and harder to validate biologically without published density data we can fit. Likely valuable but secondary.

### Why compartmentalized DA is #3

Compartmentalized DA is the most invasive change to existing systems — currently we have one global `current_reward_signal` that all eligibility traces respond to. To make DA action-specific, we'd need per-action eligibility windows, per-action DA pulses, and reasoning about which actions overlap in time. The mechanism is real biology, but it's not the *most likely* missing piece given v4's evidence (which points more to topology than to DA precision). Worth keeping in reserve as the third lever.

## Sketch of each option

### Option 1: Structural plasticity (axon pruning)

**Core idea:** During pretraining (critical period), each synapse maintains a *survival score* that accumulates over time based on its contribution to reward. Synapses whose contribution stays consistently low get *pruned* — their weight clamps to 0 permanently, and they're excluded from future plasticity updates. After pretraining ends, pruning stops (critical period closes) and the surviving sparse network is frozen for eval.

**Mechanism (proposed):**
- New per-synapse GPU array `cp_synapse_alive` (bool, default True).
- New per-synapse GPU array `cp_synapse_survival_score` (float).
- Per-trial update during pretraining: `survival_score[i] += eligibility[i] * reward * α_survival`. Synapses that fire in lockstep with reward accumulate positive survival score; those that fire against reward accumulate negative.
- Pruning rule: when `survival_score[i] < threshold` AND `weight[i] < weight_floor`, set `alive[i] = False` and `weight[i] = 0`. Pruned synapses don't contribute to forward dynamics or receive plasticity updates.
- New flag `--enable-structural-pruning` (default off).
- During pretraining: pruning ON. After pretraining ends: `set_plasticity_gate("bg_cross_projections", 0.0)` AND pruning auto-stops (since plasticity is off).

**Predicted effect:** dense initial cross-connectivity gets sparsified over the pretraining period. Useful cross-pairs survive (they correlate with reward); noise pairs prune away. After freeze, the surviving connectivity is sparse and reward-aligned — exactly what real developmental pruning produces.

**Risk:** if pruning is too aggressive, it eliminates everything (collapse to zero connectivity). If too conservative, no useful sparsification. Needs hyperparameter tuning.

### Option 2: Sparse heterogeneous initial topology (patch-matrix)

**Core idea:** Replace the dense 4×4 cortex→striatum cross-connectivity with a sparse, structured initial pattern that mirrors real patch-matrix anatomy. Cortex pools project to *specific subsets* of striatal pools based on a fixed cross-projection density (e.g., 25% — each cortex pool reaches only 1 of 3 cross-action striatal targets at random) plus a same-action-dominant overlay.

**Mechanism (proposed):**
- New `--cross-projection-density` flag (default 1.0 = current dense; 0.25 = sparse patch-matrix-like).
- In `build_bg_brain_regions`, when adding cross pathways, randomly select a subset of (from_action, to_action) pairs to instantiate based on density.
- The selection must be deterministic given the seed so reruns reproduce the same topology.
- Combine with v3.1 or v4 training regimes to test whether sparse init alone is enough.

**Predicted effect:** smaller cross-projection footprint means fewer noise channels for the cascade to suppress. Each surviving cross-pair carries more "weight" (information value) and is easier for STDP+reward to refine usefully.

**Risk:** if density is too low, we lose the cross-action coupling that motivates cheat #5 in the first place. If too high, no improvement over v3.1.

### Option 3: Compartmentalized dopamine

**Core idea:** Per-action DA pulses. Currently `bridge.core_config.current_reward_signal` is a scalar that all eligibility traces respond to uniformly. Replace with a per-action vector — DA[N], DA[E], DA[S], DA[W] — and route each pulse only to synapses tagged with the matching action.

**Mechanism (proposed):**
- New per-action eligibility tag on synapses: `synapse_action_tag[i]` ∈ {0,1,2,3} indicating which action this synapse contributes to (cortex_X→str_Y synapses get tagged with the *target* action Y).
- Reward signal becomes a 4-vector. After action Y is selected and reward computed, DA[Y] += reward; DA[≠Y] = 0.
- In the per-step plasticity rule, gate `eligibility[i] * reward[synapse_action_tag[i]]` instead of `eligibility[i] * reward_scalar`.
- Combine with v3.1 or v4 to test whether action-specific DA fixes cross-projection refinement.

**Predicted effect:** cross-projection synapses receive teaching signals specific to the action they contribute to, not a global "good/bad" signal. STDP+reward can finally differentiate "context X → action Y" from "context X → action Z".

**Risk:** invasive change to the existing reward / eligibility / plasticity machinery. Needs careful validation that single-action behavior (where this should reduce to v3) is unchanged.

## Decision policy

**Try option 1 first.** Run Tier 1 wiring smoke (~25 min single seed) → Tier 2 signal check (3 seeds × 5K pretraining + structural pruning, ~4h batch) → Tier 3 if promising. Same decision matrix as v4:

| Tier 2 mean sum (n=3) | Action |
|---|---|
| ≤ 4.5 | Tier 3 (overnight 6-seed) → if GO, cheat #5 closed for real |
| 4.5–6.0 | review per-seed; consider tuning pruning thresholds and re-running |
| > 6.0 | option 1 fails; pivot to option 2 |

**If option 1 fails, pivot to option 2.** Plan + design ready by then (this survey + the option 2 design doc to be written during option 1's smoke).

**If option 2 fails, pivot to option 3.** Plan + design ready (option 3 design doc to be written in parallel).

**If all three fail:** the closure-by-design landed earlier today stands as the final answer for this codebase. Document the negative finding triplet as a strong "cross-projections cannot be made functional in our reduced substrate" result, and close the chapter.

## Implementation order

1. **Now → option 1 design doc** (`docs/plans/2026-04-28-structural-plasticity-design.md`)
2. **Now → option 1 implementation plan** (`docs/plans/2026-04-28-structural-plasticity-implementation.md`) — TDD bite-sized
3. **Now → implement option 1** via subagents (TDD per task)
4. **Then → option 1 Tier 1 smoke** (single seed, ~25 min)
5. **In parallel during smoke → option 2 design doc** (`docs/plans/2026-04-28-patch-matrix-striatum-design.md`)
6. **In parallel during smoke → option 3 design doc** (`docs/plans/2026-04-28-compartmentalized-da-design.md`)
7. **Then → option 1 Tier 2 signal check** (3 seeds, ~4h)
8. **Decide:** GO → Tier 3 overnight; PARTIAL → tune; NO-GO → option 2

Total estimated wall-clock to first verdict: ~3-5h for option 1 implementation + smoke + Tier 2.
