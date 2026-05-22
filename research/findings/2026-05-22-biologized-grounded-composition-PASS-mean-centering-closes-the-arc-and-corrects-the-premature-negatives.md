# Fully-biologized grounded composition = PASS: mean-centering closes the biologization arc, and honestly corrects two premature NEGATIVEs

## Status

A compositional pipeline that is biology-grounded end-to-end -- with no
oracle symbol table -- clears the project's frozen 0.80 compositional
bar at multi-seed mean 0.98 across loads {2,3,5}. This is the
constructive close of the FHRR-biologization arc. It was reached by a
cheap smell-test that caught an error: two NEGATIVEs propagated earlier
today concluded the composition symbol could not be grounded in the
substrate's activity, and that conclusion was premature. This document
records the PASS and corrects the record.

## The error, and the smell-test that caught it

Two NEGATIVEs were propagated today:

- "Shortcut 2 NEGATIVE -- the oracle supplies orthogonality the
  substrate cannot": the substrate's concept representations overlap by
  a mean pairwise similarity of about 0.45; FHRR/VSA needs
  near-orthogonal symbols; therefore the symbol cannot be grounded.
- "Fully-biologized grounded composition NEGATIVE": routing the
  grounded symbol through dentate-gyrus pattern separation reached only
  0.07-0.19 symbol overlap, the attractor clean-up stayed degenerate, a
  sweep confirmed the 0.07 floor.

Both measured something real -- the substrate's concept representations
do overlap by 0.45, and dentate-gyrus separation does floor at 0.07.
But both then concluded the symbol could not be grounded, and that
conclusion tested only two transforms (an oracle-lookup replacement,
and dentate-gyrus separation). The "post-hoc transform route is closed"
claim was over-broad.

A smell-test of that over-broad claim tested the obvious untested
transform: removing the common-mode. The 0.45 overlap is a similarity
between full activity vectors; if the concepts share a large baseline
("common-mode") activity, that baseline alone produces a high cosine
similarity even when the concept-specific parts are orthogonal.
Subtracting the across-concept mean activity -- subtractive
normalisation, a recognised cortical computation implemented by pooled
inhibition -- removes the common-mode.

The result was decisive (multi-seed):

```
concept-symbol mean pairwise similarity
  raw                       0.45
  mean-centered            -0.05      (the random-symbol level)
attractor clean-up identifies a clean grounded symbol as itself
  raw                       1/16      (degenerate)
  mean-centered            15-16/16   (works)
```

The 0.45 overlap was almost entirely common-mode. The concept-specific
activity is near-orthogonal. Removing the common-mode exposes it, and
the grounded symbols come out near-orthogonal -- composable.

## The result

The fully-biologized grounded compositional pipeline, re-run with
mean-centering as the grounding transform
(`biologized_grounded_composition.py --grounding meancenter`):

```
            integrated mean    composition-only mean
L=2         0.987              0.999
L=3         0.981              0.994
L=5         0.982              0.994

VERDICT -> PASS (all loads clear the frozen 0.80 bar)
```

The pipeline, end to end:

1. RECOGNISE a concept word by averaging the substrate's per-neuron
   activity over K=8 observations (the longer-integration rate readout,
   ~0.93 recognition; the recognition-bound probe established this).
2. The GROUNDED SYMBOL of a concept = a phasor derived from that
   concept's consolidated activity with the across-concept common-mode
   subtracted. It is a deterministic function of the substrate's own
   activity -- not an oracle vector. The orthogonality is the
   substrate's own concept-specific structure, exposed by common-mode
   removal.
3. COMPOSE with the resonate-and-fire FHRR layer (binding on
   resonate-and-fire neurons).
4. CLEAN UP with the attractor (annealed settle over the grounded
   symbols).

Every stage is biological -- a longer-integration rate readout,
common-mode subtraction, resonate-and-fire FHRR composition, an
attractor clean-up. There is no oracle symbol table.

## Smell test (a PASS scrutinised harder than a FAIL)

- Genuine, not tuned: mean-centering is parameter-free (subtract the
  mean). K=8 was set by the recognition-bound probe. The 0.80 bar is
  frozen. The mean-centering fix was found by a smell-test, not a sweep.
- Mean-centering is biological, not a cheat: subtractive normalisation
  / common-mode rejection is a well-characterised cortical computation
  -- pooled feedforward and feedback inhibition compute a population
  average and subtract it. It is arguably more biological than the
  random-projection symbol deriver. The grounded symbol remains a
  deterministic function of the substrate's own concept activity; the
  orthogonality emerges from the substrate's concept-specific structure,
  it is not assigned.
- composition-only is 0.99 -- the composition itself is essentially
  perfect on the mean-centered grounded symbols; the integrated
  shortfall from 1.0 is the substrate's recognition error propagating,
  the expected recognition-bounded behaviour.
- No answer leakage: the grounded symbols are derived from activity;
  task labels never feed the pipeline. The clean-up scores against the
  ground-truth filler only as the scoring oracle.

A dedicated adversarial review is the pre-registered next step before
this rolls into a capability-status claim, exactly because this PASS
overturns propagated NEGATIVEs and must be scrutinised independently.

## What this corrects

- The shortcut-2 NEGATIVE's conclusion ("the symbol cannot be grounded;
  the oracle's function -- orthogonality -- the substrate cannot
  supply") is OVERTURNED. The substrate can supply near-orthogonal
  grounded symbols; its concept-specific activity is near-orthogonal
  once the common-mode is removed. The 0.45-overlap measurement was
  real; the conclusion drawn from it was premature because common-mode
  removal was not tested.
- The fully-biologized-grounded-composition NEGATIVE (the dentate-gyrus
  version) is superseded: it was specific to dentate-gyrus separation
  as the grounding transform, which floors at 0.07; mean-centering, the
  correct transform, reaches -0.05.
- The "compositional-biologization line at terminus" framing is
  withdrawn: the line closes POSITIVELY. All three engineered shortcuts
  of the phase-coded composition layer are now biologized -- neurons
  (resonate-and-fire), clean-up (attractor + familiarity gate), and
  symbols (grounded in mean-centered substrate activity) -- and the
  end-to-end pipeline composes at 0.98 multi-seed.

Correction notices are added to the two superseded NEGATIVE findings
docs, pointing here. The honest lesson: a cheap probe with an
incomplete set of tested transforms can produce a premature NEGATIVE;
the standing smell-test discipline -- test the obvious untested
alternative before a NEGATIVE calcifies -- caught it within the same
session.

## Honest scope

This is a multi-seed (3 seeds) PASS on the project's compositional task
at loads {2,3,5}, computed from the real substrate activity captured in
the activity cache. It is biology-grounded end-to-end with no oracle
symbol table. Honest caveats: 3 seeds; the recognition front-end is
temporal averaging over K=8 observations (the recognition-bound probe's
mechanism), recognition-bounded at ~0.93; the two fragile words
("go", "stop") remain the substrate's hardest concepts. It is small-load
cue-to-attribute compositional retrieval -- not fluent open-ended
language. A dedicated adversarial review precedes any capability claim.

## Files / evidence

- Runner: `research/findings/raw/biologized_grounded_composition.py`
  (`--grounding meancenter`)
- Result: `research/findings/raw/biologized_grounded_composition_meancenter.json`
- The superseded NEGATIVEs (with correction notices):
  `2026-05-22-biologization-shortcut-2-NEGATIVE-...md`,
  `2026-05-22-biologized-grounded-composition-NEGATIVE-...md`
