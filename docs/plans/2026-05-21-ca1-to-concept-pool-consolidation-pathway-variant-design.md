---
type: plan
status: live
date: 2026-05-21
---

# Experimental substrate variant: ca1 -> concept-pool consolidation pathway

> **For Claude / autonomous continuation:** This is the pre-registered
> design for the experimental variant the consolidation probe's
> terminal finding identified. It is a net-new arc that touches no
> protected module. After the variant test, propagate honestly; a
> positive result motivates (but does not itself perform) the
> owner-decision roll-in to the main substrate.

## Why this variant exists

The renewed-focus compositional investigation (three cheap-first
probes, 2026-05-21) localized the eight-architecture convergent
ceiling to a single structural cause: the unified substrate has
hippocampo-cortical consolidation pathways `ca1 -> motor` and
`ca1 -> language_output` (built for the validated direct word-to-motor
task) but **no `ca1 -> concept-pool` pathway**. A compositional
(noun, adjective) binding is encoded as a hippocampal engram tag, but
it cannot be consolidated into the cortical concept pools where
compositional readout must land -- the wire does not exist.

This variant adds that wire and tests whether compositional
consolidation then works.

## The change (net-new; no protected module modified)

`build_biological_brain_regions` (in the protected `text_minimal_isolation.py`)
returns `(regions, pathways)` lists. `_build_bridge_with_phase1_recipe`
sets `cfg.region_pathways = list(pathways)`. The variant builder calls
`build_biological_brain_regions` with identical kwargs -- byte-unchanged
-- then **appends** new `RegionPathway` objects to the returned
`pathways` list before building the bridge. The builder is not
modified; the runner augments its output.

The appended pathways, one per concept pool (4 noun + 4 verb + 4
adjective = 12), mirror the existing `ca1 -> motor` pathway exactly:

```
RegionPathway(from_region="ca1", to_region=<concept_pool>,
              density=0.20, weight_mean=2.0, weight_jitter=0.3,
              plastic=True, plasticity_gate="ca1_to_concept_pool")
```

This is the SAME projection class the substrate already has as
`ca1 -> motor` (density 0.20, weight 2.0, jitter 0.3, plastic),
extended to the concept-pool cortex. Biologically: ca1 / subiculum
projects broadly to association cortex; the substrate already models
this for motor cortex; the variant extends it to the concept-pool
cortex. A moderate prior weight (2.0, matching `ca1 -> motor`) is
required because the project's own v16 findings established that
STDP pathways grown from zero stay functionally silent -- the
validated `ca1 -> motor` consolidation works precisely because it has
a moderate prior weight that replay then sharpens.

## The test

1. Build the variant substrate (augmented pathways present from
   construction, exactly as `ca1 -> motor` is).
2. Train Phase-1 on the variant at the standard 200-event recipe,
   single seed 42 (the variant's extra synapses mean the existing
   cached checkpoint cannot be reused -- fresh training is required;
   ~27 min). The `ca1 -> concept-pool` pathway coexists with Phase-1
   exactly as `ca1 -> motor` already coexists with motor-pool
   Phase-1 training in the validated substrate.
3. Encode 4 compositional (noun, adjective) bindings as engram tags
   (reused `_encode_facts`, byte-unchanged).
4. Measure tag-stimulated concept-pool firing PRE-consolidation
   (0 replay cycles).
5. Run replay consolidation: `set_sleep_gates` + open the new
   `ca1_to_concept_pool` gate + `run_concept_replay_phase` (reused
   byte-unchanged). Measure at 20 and 60 cumulative replay cycles.
6. Compare pre vs post.

## Pre-registered decision rule -- with the critical anti-cheat

A moderate-weight `ca1 -> concept-pool` pathway will, on its own,
make tag stimulation drive the concept pools off the noise floor --
that is just the weight-2.0 prior transmitting, and it proves
nothing. **Lifting off the noise floor is NECESSARY BUT NOT
SUFFICIENT.** The real test is SELECTIVITY:

- **PASS (missing-pathway hypothesis confirmed)**: after
  consolidation, the bound adjective's pool is the strongest
  adjective pool on a majority of bindings (>= 3/4), AND the
  permuted-tag control holds -- stimulating a different binding's tag
  surfaces THAT binding's adjective, not the cued one. Selectivity
  must EMERGE from consolidation: pre-consolidation the readout is
  diffuse (the generic weight-2.0 prior fires all pools roughly
  equally); post-consolidation it must become bound-adjective-
  selective. If selectivity is already present pre-consolidation,
  the result is VOID (the prior, not consolidation, is doing the
  work).

- **NEGATIVE**: after 60 replay cycles the bound adjective's pool is
  not selectively strongest, or the permuted control fails. Even with
  the pathway present, replay-driven consolidation does not establish
  selective compositional retrieval. This is a deeper substrate
  finding (the consolidation learning rule, not just the wiring, is
  insufficient for compositional bindings).

The permuted-tag control is the load-bearing anti-cheat: a generic
weight-2.0 pathway cannot pass it; only genuine consolidation-
sharpened selectivity can.

## Honest ceiling (binding)

- A PASS confirms the missing pathway is the fix and motivates a full
  pre-registered multi-seed arc. It would still NOT be fluent
  open-ended language -- it would be reliable cue-to-attribute
  compositional recall, the capability eight arcs could not reach.
- A PASS does NOT itself roll the pathway into the main substrate.
  That modifies the protected, validated `build_biological_brain_regions`
  and risks the validated direct-binding capability -- an owner
  architectural decision, to be made with this variant's evidence.
- A NEGATIVE is an honest finding and routes to the consolidation
  learning rule.
- No bar tuned. The variant runner touches no protected/frozen/moat
  module: `build_biological_brain_regions`, `text_minimal_isolation.py`,
  `abstention_gate.py`, every frozen verdict module -- all
  byte-unchanged. Reuse-by-import for the encode, replay, gate, and
  measurement helpers. No autograd. Honest propagation both remotes.

## Next step

Write `research/findings/raw/ca1_concept_pool_variant.py` (the variant
builder + Phase-1 training + the pre-registered test), run it on seed
42, apply the decision rule, propagate.
