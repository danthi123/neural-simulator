---
type: finding
status: live
claim_check: measured
date: 2026-09-09
mechanism: order-STDP cortical consolidation — region-growth-compensating in-degree law
lane: memory
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_replay_cortical_consolidation_gate_v10_indegree_law.py
artifacts:
  - research/findings/raw/order_recalib/v10_indegree_law_decisive_numpy.json
builds_on:
  - research/findings/2026-09-08-order-consolidation-v9-true-sparsity-scaleup-NO-GO-readout-threshold-not-mechanism.md
---

# Board #130 sparsity scale-up: the v9 NO-GO's OWN named fix (a region-growth-compensating in-degree law) is a clean 6-seed GO — the fixed in-degree was a constant standing in for a missing homeostatic process

## What v9 found, and what it named
The v9 finding (`877a07ef9`, NO-GO 0/6 at a 2x region scale-up toward true CA1 sparsity) root-caused its own
failure precisely: the ONE full-region, non-assembly-scoped wiring pathway `cortical_association` (spans the WHOLE
cue -> target region) kept a FIXED absolute in-degree (48) while its candidate pool DOUBLED at the new scale,
DILUTING the driven-cue signal. The order-STDP consolidation MECHANISM itself stayed intact
(`order_stdp_attribution=1.0`, `stdp_off_order_margin=0.0` exact) — only a diluted READOUT failed a fixed
behavioral threshold. v9 named two independently-sufficient, stackable fixes; this is the FIRST: a
region-growth-compensating in-degree LAW that preserves the driven-FRACTION probability as the region grows
(this project's own "wall reframe": what homeostatic/companion process did biology run that we replaced with a
constant?).

## The v10 mechanism (additive, byte-identical off)
`research/runners/_replay_cortical_consolidation_gate_v10_indegree_law.py` gives `cortical_association` a
scale-invariant in-degree law: `indegree(n_cue) = round(48 * n_cue / 1200)`, opt-in via a new
`GateConfig.indegree_law` field. `fixed` (default) reproduces v9's constant 48 -- byte-identical -- and is
ANCHORED at v8/v9's own baseline (`n_cue = 1200`), so both laws agree EXACTLY at that scale by construction (a
stronger guarantee than merely "off equals old behaviour"). `scale_invariant` scales the in-degree with the
candidate-pool size so the driven fraction is scale-invariant. No `sim/` edit; v1/v6/v7/v8/v9 untouched.

## Result — 6-seed decisive, GO
`research/findings/raw/order_recalib/v10_indegree_law_decisive_numpy.json` (numpy, seeds 42/43/44/100/101/102,
`--indegree-law scale_invariant`, `region_scale_beyond_v8 = 2`):

- **VERDICT GO** on the inherited bar (intact_beats_shuffled_order margin >= +0.01 AND both_memories_recovered
  AND stdp_owns_order AND lesions ~0, on >= 5/6 seeds): **6/6** — `n_seed_order_go = 6`, `n_beats_shuffled_order = 6`,
  `n_both_recovered = 6`.
- `mean_order_recovery_margin = +0.0229` <!--derived--> (clears +0.01); at the 2x scale the scale-invariant law set
  `cortical_association_indegree = 96` (= 48 * 2400/1200), vs the fixed 48 that diluted the signal in v9.
- ANTI-CHEATS INTACT: `mean_stdp_off_order_margin = 0.0` (exact -- STDP-off collapses the effect, so it is
  genuinely order-STDP-driven, not a wiring artifact) and `aggregate_order_stdp_attribution = 1.0` (the
  order-STDP mechanism owns the recovered order). The A/B correct-rates recovered by MAGNITUDE (~quadrupled in
  the 2-seed pre-smoke) rather than needing the behavioral threshold moved -- so v9's SECOND named fix
  (re-deriving the fixed 0.015/0.03 <!--derived--> thresholds as a fraction of the substrate's own baseline) was NOT needed here.

## What this closes, and the honest residual
The v9 NO-GO was a diluted-readout artifact of a fixed connectivity constant at a grown scale, NOT a mechanism
failure -- and its own named region-growth in-degree law resolves it, 6/6, with the consolidation mechanism
provably intact. **Honest residual (not a wall):** the `assembly:region ratio` at this rung is 17% (0.1667 <!--derived-->; artifact `assembly_region_ratio_ca1_cue_target`),
a step toward the biological CA1 target of ~1-5% <!--derived--> but not yet there -- further sparsity is future scale-up rungs,
each of which this scale-invariant law should now carry (the driven-fraction is preserved by construction). The
in-degree law is a developmental connectivity rule (biological), not a host readout patch. Not a phenomenal claim.
