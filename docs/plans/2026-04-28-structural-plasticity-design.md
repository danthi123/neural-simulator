# Cheat #5 closure attempt: Structural Plasticity (axon pruning) — Design

**Status:** design. Implementation plan to follow.
**Survey context:** [`2026-04-28-cheat5-real-options-survey.md`](2026-04-28-cheat5-real-options-survey.md) — option 1 of 3, ranked most likely to succeed.

## Goal

Add experience-dependent **synapse pruning** as a new biology-grounded plasticity mechanism. During the developmental critical period (pretraining), each plastic synapse maintains a *survival score* tracking its contribution to reward; synapses whose contribution stays consistently low get *pruned* — weight clamps to 0 permanently and they're excluded from future plasticity updates. After the critical period closes (eval starts), pruning auto-stops and the surviving sparse network is frozen.

This addresses v4's smoking gun: weight-only refinement converged to a uniform connectivity (mean ~11, std ~0.5 across 3 seeds), suggesting the over-constrained dense topology is the limit. Adding pruning lets the network *carve* useful sparsity from random init.

## Architecture

Three new GPU arrays + one rule + two flags:

```
cp_synapse_alive       (bool[nnz])  — default True; pruned synapses set False permanently
cp_synapse_survival    (f32[nnz])   — running score; pruning decision depends on this
cfg.pruning_alpha      (float)      — survival score update rate
cfg.pruning_threshold  (float)      — survival score below which pruning may fire
cfg.pruning_weight_floor (float)    — weight below which pruning may fire
--enable-structural-pruning         — opt-in flag (off by default)
```

### Per-step rule (during pretraining only)

After the existing eligibility-trace and reward-driven weight update:

1. **Update survival scores** (vectorised on GPU):
   ```
   survival_score[i] += pruning_alpha * eligibility_trace[i] * current_reward_signal
   ```
   Reward-aligned synapses (eligibility positive when reward arrives) accumulate positive survival. Reward-misaligned synapses (eligibility positive when punishment arrives, or vice versa) accumulate negative.

2. **Pruning decision** (per-step on GPU):
   ```
   prune_now = (survival_score[i] < pruning_threshold) AND (weight[i] < pruning_weight_floor) AND alive[i]
   if prune_now:
       alive[i] = False
       weight[i] = 0
   ```
   Two conditions to prune: low survival (synapse hasn't earned its keep) AND low weight (synapse is barely contributing anyway). Both must hold — protects against pruning a useful but currently-quiet synapse.

3. **Forward pass + plasticity gate**:
   - Forward: `effective_weight[i] = weight[i] * alive[i]` — pruned synapses contribute zero current.
   - Plasticity: existing `cp_plasticity_gain[i] *= alive[i]` — pruned synapses receive zero updates from STDP, eligibility, etc.
   - Both are simple multiplicative gates; no branching needed.

### Critical period closure

When pretraining ends, three things happen:
1. Existing curriculum init forces `bg_cross_projections` plasticity gate to 0.0 (already happens at [g11_bg_runner.py:1220](../../research/runners/g11_bg_runner.py#L1220)).
2. Pruning auto-stops because `eligibility_trace * reward_signal` updates are gated by the same plasticity gate (no eligibility → no survival update → no pruning fires).
3. The `alive` array stays frozen at whatever shape pretraining produced. Pruned synapses stay pruned forever.

This means we don't need a separate flag to disable pruning at the boundary — the existing plasticity gate machinery handles it.

## What this gives us

After pretraining + freeze:
- `alive` array is True for surviving cross-projection synapses, False for pruned.
- The cross-projection sub-network has been *sparsified* by experience: dense 100% → some-density-X% (X to be measured).
- Surviving synapses' weights are at whatever STDP+reward shaped them to.
- During eval, only surviving synapses contribute current and only their weights matter.

The hypothesis is: a sparse, experience-shaped sub-network behaves better in eval than the dense, uniformly-shaped one v4 produced.

## What we expect to measure

- **Pruning fraction**: what % of cross-projection synapses get pruned during pretraining? Reasonable target: 30-70% pruned (if 0%, pruning rule too conservative; if 100%, too aggressive).
- **Surviving weight distribution**: should NOT be the uniform mean ~11 std ~0.5 we saw in v4. Should be more bimodal — some strong (useful, survived) and some near zero (pruned).
- **Eval performance**: same metric as before, sum of `final_quarter_mean_distance` over phases. Decision matrix unchanged from v4.

## Hyperparameters and defaults

Initial guesses (tune in Tier 2 if needed):
- `pruning_alpha`: 0.001 (slow accumulation; need ~1000 trials to integrate enough signal to make pruning decisions)
- `pruning_threshold`: -1.0 (survival score needs to be negative to consider pruning — synapse has been actively counterproductive)
- `pruning_weight_floor`: 1.0 (weight must have already shrunk significantly via normal STDP+reward; pruning eliminates the long tail of dying synapses)

Why these values: cross-projection weights start at 0.0, grow via STDP. Synapses that fire usefully grow weight AND positive survival; synapses that fire harmfully don't grow weight (or shrink) AND accumulate negative survival. Pruning fires when both signals agree the synapse is useless.

## Implementation strategy

Like the v4 pretraining helper: pragmatic insertion, not a refactor of `bridge.py`'s core update path. Specifically:

1. **Allocate the new arrays** in `bridge.initialize` or wherever `cp_synapse_weights` (== `cp_connections.data`) is allocated. Both new arrays size = `nnz`.
2. **New bridge method** `bridge.update_pruning(eligibility_trace, reward_signal, alpha, threshold, weight_floor)` — does steps 1+2 of the rule above. Called explicitly by the runner after the existing per-trial reward update, only when `enable_structural_pruning` is True AND we're in pretraining phase.
3. **Forward + plasticity gating**:
   - Forward: in the existing synaptic-current kernel (`fused_conductance_decay_and_current` or wherever `weight[i]` is read), multiply by `alive[i]`. If hard to inject, can do a one-shot `weight[i] *= alive[i]` after each pruning event — same effect, slightly less efficient.
   - Plasticity: existing `cp_plasticity_gain[i]` array can be scaled by `alive[i]`. Single line addition: after each pruning event, `cp_plasticity_gain[i] = cp_plasticity_gain[i] * alive[i]`. Idempotent.
4. **Hook into v4 pretraining loop**: in `_run_pretraining_phase`, after the existing per-trial reward+plasticity, call `bridge.update_pruning(...)` if `enable_structural_pruning`. Pass the cross-projection synapse indices so we only prune cross-projections (not the same-action `cortex_to_d1` pathway, which is supposed to remain dense).

The "only prune cross-projections" detail is important: same-action cortex→striatum is the workhorse routing; we don't want to sparsify it. We use the existing `_plasticity_gate_to_synapses["bg_cross_projections"]` index set as the prunable subset.

## What this preserves vs changes

**Preserves:**
- All existing v3 lateral inhibition behavior (lateral pathways are tagged `plastic=False` so they never enter the survival/prune path).
- All existing same-action cortex→striatum routing (it's a different gate, different synapse subset).
- All existing eligibility and reward machinery — pruning piggybacks on these signals, doesn't replace them.
- The flagship config without `--enable-structural-pruning` is bit-identical to today's flagship.

**Changes:**
- New arrays in `bridge` state.
- New method on `bridge`.
- New flag in the runner.
- The pretraining helper grows ~30 lines to call the new pruning method.
- New CLI flags exposed.

## Testing

Same TDD pattern as v4: write failing test, minimal impl, verify pass, commit. Tests:

1. `test_structural_pruning_arrays_allocated` — when `enable_structural_pruning=True`, bridge has `cp_synapse_alive` and `cp_synapse_survival` arrays of correct size.
2. `test_structural_pruning_default_off` — when flag not set, the arrays are not allocated; current behavior unchanged.
3. `test_pruning_only_targets_cross_projections` — pruning rule only applies to synapses in `bg_cross_projections` gate's index set; same-action `cortex_to_d1` synapses are never pruned.
4. `test_pruning_eliminates_low_survival_low_weight_synapses` — synapses with negative survival AND low weight get pruned (alive=False, weight=0); high survival OR high weight survive.
5. `test_pruned_synapses_dont_receive_plasticity_updates` — once pruned, `cp_plasticity_gain[i] = 0`, so subsequent STDP/reward updates don't change weight.
6. `test_pretraining_with_pruning_smoke` — end-to-end Tier 1: 1 goal × 1000 trials of pretraining with `--enable-structural-pruning`; verify some non-zero fraction of cross-projection synapses get pruned, eval still completes without crash.

## Decision matrix (unchanged from v4)

| Tier 2 mean sum (n=3) | Verdict |
|---|---|
| ≤ 4.5 | promising → Tier 3 (overnight 6-seed) |
| 4.5–6.0 | review per-seed; tune pruning thresholds and re-run |
| > 6.0 | NO-GO → pivot to option 2 (patch-matrix) |

GO criterion at Tier 3: same as v4 — sum ≤ 4.1 with both phases ≤ 2.5.

## Out of scope

- Synaptogenesis (the inverse process — *adding* new synapses based on activity). Real biology has both directions; we model only pruning. Adding synaptogenesis would require a different mechanism (likely a periodic structural-plasticity step that adds candidate synapses based on co-activation). Defer.
- Pruning of same-action `cortex_to_d1` synapses. Out of scope by design — this rewrites a fundamental routing layer that's working.
- Structural plasticity of inhibitory pathways (lateral inhibition, FS interneurons, etc.). Out of scope — these are `plastic=False` and shouldn't be touched.
- Hardware-aware optimization (e.g., compaction of pruned synapses to skip them in iteration). Future perf work; for now alive-mask multiplication is sufficient.

## Done criteria

- [ ] Tests 1-6 pass
- [ ] Tier 1 smoke shows sparsification (> 0% pruned, < 100% pruned) and eval completes
- [ ] Tier 2 (3 seeds × 5K pretraining + pruning) returns a 3-seed result against the matrix
- [ ] Tier 3 (only if Tier 2 promising) — 6-seed validation
- [ ] Findings doc + propagation per the v4 template
