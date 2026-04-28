# Cheat #5 closure attempt: Patch-Matrix Striatum (sparse heterogeneous initial topology) — Design

**Status:** design, on standby. Activated only if option 1 (structural plasticity) fails Tier 2 with mean sum > 6.0.
**Survey context:** [`2026-04-28-cheat5-real-options-survey.md`](2026-04-28-cheat5-real-options-survey.md) — option 2 of 3.

## Goal

Replace the dense 100% cortex→striatum cross-connectivity with a sparse, structured initial topology that mirrors real patch-matrix anatomy. Rather than every cortical action pool projecting to every striatal action pool (the 4×4 dense pattern of v3.1+v4), each cortical pool projects to a *sampled subset* of cross-action targets — set at initialization and held fixed by topology.

## Why this might work

v4 showed dense cross-connectivity converges to a uniform weight distribution that degrades eval performance. The hypothesis: the substrate is *over-parameterized* — every cross-pair has to carry useful information, and there isn't enough information to go around. A sparser initial topology forces the surviving cross-projections to specialize, similar to how option 1 (pruning) does it dynamically. The difference: option 2 sets the topology at build time and never changes it.

## Architecture

### Topology change

Currently in `build_bg_brain_regions`, each (cortex_action, str_action) pair where `cortex_action != str_action` gets a cross-pathway with `density=1.0` (every cortex neuron connects to every striatal neuron in the target pool). For 4 actions × 3 cross-targets × 2 (D1, D2) = 24 dense pathways.

**New:** introduce a `cross_projection_density` parameter ∈ [0, 1] that controls the *pathway-level* presence:
- `cross_projection_density=1.0` — current behavior (all 24 cross-pathways added).
- `cross_projection_density=0.5` — randomly sample half of the 24 cross-pathways at build time; the others are simply not instantiated.
- `cross_projection_density=0.25` — keep 6 of 24 (one per direction × D1/D2 average).

**Within each surviving pathway, density stays 1.0** (every cortex neuron in the source pool connects to every striatal neuron in the target). This keeps the existing connection-builder code unchanged — only the *list of pathways* shrinks.

### Determinism

Topology must be reproducible. The pathway-selection RNG is seeded from `cfg.heterogeneity_seed` (or a new dedicated seed `cfg.cross_projection_topology_seed`). Different seeds get different sparse patterns; same seed gets the same pattern. Document that "topology and weights are independent" — same eval seed across two different topology seeds tests robustness; same topology seed across two eval seeds tests topology-conditional reproducibility.

### Same-action paths unchanged

Same-action `cortex_X → str_X` pathways (8 total) are untouched at full density. Patch-matrix only sparsifies the 24 cross pathways.

### Variants worth considering

- **Pure random sparse:** randomly sample N of 24. Simple but doesn't mirror biology.
- **Direction-aware sparse:** prefer cross-pathways between adjacent actions (N↔E, E↔S, S↔W, W↔N) over opposite (N↔S, E↔W). Mirrors how real motor cortex codes neighboring directions in correlated populations.
- **D1/D2 asymmetric sparse:** different density for D1 vs D2 cross-pathways. Real BG has direct/indirect pathway asymmetries that might map onto this.

Start with **pure random sparse** for simplicity. If pure random shows partial signal (sum 4.5–6), try direction-aware.

## Implementation outline

1. Add `cfg.cross_projection_density: float = 1.0` to `CoreSimConfig`.
2. Add `cfg.cross_projection_topology_seed: int = 0` to `CoreSimConfig` (so users can vary topology independent of eval seed).
3. In `research/runners/g11_bg_runner.py:build_bg_brain_regions`, modify the cross-pathway loop:
   ```python
   import random
   topology_rng = random.Random(cross_projection_topology_seed or seed)
   all_cross_pairs = [(c, s) for c in ACTION_NAMES for s in ACTION_NAMES if c != s]
   n_keep = int(round(len(all_cross_pairs) * cross_projection_density))
   selected_cross = set(topology_rng.sample(all_cross_pairs, n_keep))
   for cortex_action in ACTION_NAMES:
       for str_action in ACTION_NAMES:
           if cortex_action == str_action:
               # same-action pathway, always added
               ...
           elif (cortex_action, str_action) in selected_cross:
               # this cross-pathway survives the topology sample
               ...
           # else: pathway not instantiated
   ```
4. Add CLI flags: `--cross-projection-density`, `--cross-projection-topology-seed`.
5. New tests: verify number of pathways at density=1.0/0.5/0.25; verify same seed produces same selection.

## Pretraining + this option

Patch-matrix can stack with option 1 (structural pruning) AND with v4 developmental pretraining. Three meaningful combinations to try:

1. **Patch-matrix alone** (no v4, no pruning) — pure topology test. Use `cross_projection_density=0.25`, no `--developmental-pretraining`. Compare against v3.1's 8.92.
2. **Patch-matrix + v4 pretraining** — sparse topology, weights shaped during critical period, then frozen. The full developmental story.
3. **Patch-matrix + structural pruning** — option 1+2 combo. Sparse init + further dynamic sparsification.

Try (1) first as a clean isolation test, then (2), then (3) if neither isolates the win.

## Predicted effect

A 25%-density patch-matrix (6 of 24 cross-pathways) reduces the cross-projection footprint by 75%. The surviving cross-pairs each get more "information value" — STDP+reward has fewer noise channels to differentiate. If the v4 failure was indeed over-parameterization, this should help. If it was something else (e.g., the global DA signal can't differentiate actions), this won't help — that's option 3 territory.

## Validation tiers (mirrors v4)

- **Tier 1**: 1 seed × 1000 pretraining steps + 1800 eval, density=0.25. Verify build doesn't crash, fewer pathways instantiated as expected.
- **Tier 2**: 3 seeds × 5K pretraining + 1800 eval, density=0.25. Decision matrix unchanged.
- **Tier 3**: 6 seeds × 30K pretraining + 1800 eval, density=0.25 (only if Tier 2 ≤ 4.5).

## Risks

- **Density too low**: 1 of 24 = effectively no cross-projections, equivalent to v3.
- **Density too high**: 23 of 24 = effectively v3.1, no improvement.
- **Wrong topology seed**: a particular sparse pattern may be lucky/unlucky. Need to test multiple topology seeds to know if the variance is from eval seeds vs topology seeds. Tier 3 should fix the topology seed and vary the eval seed; a separate ablation study would vary topology seed.
- **Combining with option 1 (pruning)**: if both are on, sparse init + pruning could lead to over-sparsification (collapse to zero connectivity). Watch for `cross_alive_count == 0` after pretraining.

## What this preserves vs changes

**Preserves:** all existing v3 lateral inhibition, v3.1 cross-projection dynamics (when density=1.0), v4 developmental pretraining. Default `cross_projection_density=1.0` is bit-identical to today's behavior.

**Changes:** a small build-time pathway selection step. No runtime changes. No new GPU arrays. Lower implementation cost than option 1.

## Out of scope

- Within-pathway density (random subset of synapses within each pathway) — Watts-Strogatz-style sparsity inside a pool. Defer until pathway-level sparsity is shown insufficient.
- Continuous direction tuning of cortical pools (real motor cortex has continuous distributions; we have discrete N/E/S/W). Different problem.
- Patch-matrix compartmentalization of *striatum itself* (real striatum has anatomically distinct patch + matrix sub-regions). Even bigger architectural change.

## Done criteria

- [ ] CLI flags wired
- [ ] Topology determinism test
- [ ] Tier 1 smoke — verify pathway count matches density × 24
- [ ] Tier 2 signal check (3 seeds × 5K pretraining)
- [ ] Tier 3 if promising
- [ ] Findings doc + propagation
