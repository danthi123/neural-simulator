# Acetylcholine-staged recurrent excitation: consolidation variant design

> **For Claude / autonomous continuation:** Pre-registered design for
> the next experimental variant. Supersedes the "dedicated region"
> route (route 1 of the ca1-variant findings) -- external biology
> research redirected it to the more faithful mechanism below. Net-new;
> touches no protected module. Continue straight into build + run +
> propagate; no hand-back.

## What the research changed

The ca1-variant arc (2026-05-22) found the missing `ca1 -> concept-pool`
wire is necessary but not sufficient: the concept pools' deliberately
weak internal dynamics (a v14/v16 Phase-1-stability choice) cannot
ignite into a consolidated attractor from hippocampal drive. The
finding framed this as a trainability-vs-consolidatability tension and
proposed a dedicated separate region as a workaround.

External biology research (Hasselmo and colleagues; SPEAR model --
Separate Phases of Encoding And Retrieval) shows the tension is not
fundamental and biology does not work around it -- it dissolves it
with acetylcholine:

- During ENCODING (high ACh): muscarinic acetylcholine causes
  **selective presynaptic inhibition of recurrent / intracortical
  excitatory synapses**, while sparing afferent input. With recurrent
  excitation suppressed, many overlapping patterns can be trained
  into the same network without crosstalk. Hasselmo's mathematical
  point: in every associative-memory model, the weight matrix must be
  modified "without allowing the new patterns to activate excitatory
  recurrent synapses of the network during encoding" -- otherwise each
  retrieved memory becomes spuriously associated with new input
  ("runaway synaptic modification", "severe breakdown of function").
- During CONSOLIDATION (low ACh, <1/3 of waking levels in slow-wave
  sleep): the presynaptic inhibition is released, recurrent excitation
  is restored, and "attractor dynamics and pattern completion" can
  consolidate memories into stable cortical representations.

This is exactly the project's situation. The concept pools' weak
recurrence is a crude PERMANENT encoding-mode. The "canon amplifies
bias" Phase-1 collapse the project documented for strong concept-pool
dynamics IS Hasselmo's encoding-interference cascade -- recurrent
synapses active during multi-concept encoding. The project
independently rediscovered the interference problem and solved it the
crude way (permanently weak recurrence), which also kills
consolidation.

The faithful fix: STAGE the concept pools' recurrent excitation.
Suppressed (encoding mode) during Phase-1 training; released
(consolidation mode) during replay consolidation and the subsequent
readout. The same pools, two modes -- as biology does it.

(Inhibition-stabilized-network theory is the corroborating second
mechanism: strong recurrent excitation is stable when matched by
strong recurrent inhibition. The concept pools carry FS interneurons,
so released recurrent excitation has an inhibitory partner. If the
released recurrence runs away, ISN tuning -- matched inhibition -- is
the documented next step.)

## The change (net-new; no protected module modified)

Reuse the ca1-variant substrate exactly: `build_biological_brain_regions`
byte-unchanged plus the 12 appended `ca1 -> concept-pool` consolidation
pathways. Phase-1 is identical to the ca1-variant -- so the
ca1-variant's existing Phase-1 checkpoint
(`phase1_ca1variant/seed42.simstate.h5`) is reused; NO retraining.

After loading the Phase-1 checkpoint, BEFORE consolidation, install
staged recurrent excitation into each concept pool: for each noun /
verb / adjective pool, generate a recurrent excitatory connectivity
among the pool's excitatory neurons (density 0.10, weight 2.0 --
mirroring the validated motor-pool canon recurrence, which is an
attractor-capable, ISN-stable regime) and install it via
`bridge.set_pathway_weights(..., add_missing=True)`. This is the "low
ACh release of recurrent excitation": the recurrence is absent during
Phase-1 (the checkpoint was trained without it -> Phase-1 stability
preserved by construction) and present for consolidation + readout.

`set_pathway_weights` is the documented post-build pathway-installation
API; it touches no protected, frozen, or moat module.

## The test

1. Build the ca1-variant substrate; load its Phase-1 checkpoint.
2. Encode 4 compositional (noun, adjective) bindings (reused
   `_encode_facts`, byte-unchanged).
3. Measure tag-stimulated concept-pool firing PRE-recurrence-install
   (this reproduces the ca1-variant noise-floor baseline).
4. Install the staged recurrent excitation (the "low ACh" release).
5. Measure tag-stimulated firing PRE-consolidation but
   POST-recurrence-install (isolates what the recurrence prior does
   on its own -- the anti-cheat baseline).
6. Run replay consolidation; measure at 20 and 60 cumulative cycles.
7. Compare.

## Pre-registered decision rule (anti-cheat baked in; never tuned)

- **PASS**: after consolidation, the bound adjective's pool is the
  strongest adjective pool on >= 3/4 bindings AND the permuted-tag
  control holds (>= 3/4) -- stimulating a different binding's tag
  surfaces THAT binding's adjective. Selectivity must EMERGE FROM
  CONSOLIDATION: if the post-recurrence-install / pre-consolidation
  measurement (step 5) is already selective >= 3/4, the result is
  VOID -- the recurrence prior, not consolidation, did the work.
- **NEGATIVE**: after 60 replay cycles the bound adjective's pool is
  not selectively strongest, or the permuted control fails.
- **RUNAWAY (a distinct NEGATIVE sub-case)**: if installing the
  recurrence drives the concept pools into saturation (firing rates
  uniformly very high across all pools, no discrimination possible),
  the released recurrence is unstable without matched inhibition.
  This is an honest finding that routes to ISN tuning (co-install
  matched recurrent inhibition) rather than a clean negative on the
  staging hypothesis.

The permuted-tag control is the load-bearing anti-cheat: a generic
recurrence prior amplifies whatever the `ca1 -> concept-pool` wire
delivers; only genuine consolidation-sharpened selectivity makes the
amplified attractor track the stimulated tag.

## Honest ceiling (binding)

- A PASS is the first demonstration that the compositional binding
  consolidates into the cortical concept representation -- the
  capability eight arcs plus four probes could not reach. It would
  still NOT be fluent open-ended language; it would be reliable
  cue-to-attribute compositional recall at small load, single seed,
  pending multi-seed validation.
- A NEGATIVE or RUNAWAY is an honest finding and routes to the next
  biology-grounded refinement (ISN matched inhibition; or the
  dedicated-region route as a fallback).
- No bar tuned. No protected / frozen / moat module modified.
  Reuse-by-import for the substrate builder, encode, replay, gate,
  measurement helpers. `set_pathway_weights` is the documented
  post-build install API. No autograd. Honest propagation both
  remotes.

## Next step

Write `research/findings/raw/ach_staged_recurrence_variant.py` (reuse
`build_variant_bridge` + the ca1-variant Phase-1 checkpoint; add the
staged-recurrence install; run the pre-registered test), run it on
seed 42, apply the decision rule, propagate. Continue straight through
-- no hand-back.
