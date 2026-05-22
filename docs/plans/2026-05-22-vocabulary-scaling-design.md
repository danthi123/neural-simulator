# Vocabulary scaling of the biologized compositional capability -- design

## Context

The project's compositional retrieval capability is now validated and
thoroughly characterised. It exists (the identity-level integration,
multi-seed 0.96-0.99); it is biologized end-to-end (resonate-and-fire
neurons, an attractor clean-up with a separate familiarity gate, and
symbols grounded in the substrate's own common-mode-removed concept
activity -- the fully-biologized grounded pipeline clears the frozen
0.80 bar at multi-seed 0.98, adversarially reviewed CLEAR); and its
load scaling is characterised (the composition layer holds up to 96
bound facts in one composite, with the required phasor dimension
growing linearly with load -- load is not the bottleneck).

Every result so far is at a 16-concept vocabulary -- the 16 concept
words of the v14/v16 substrate captured in the activity cache. The one
scaling axis not yet addressed is the vocabulary itself.

## The question

Does the biologized grounded-composition pipeline still clear the
frozen 0.80 compositional bar at a substantially larger vocabulary, and
how does the recognition bound -- which the compositional line
identified as the limiting factor -- behave as the vocabulary grows?

There are two distinct sub-questions, and they must not be conflated:

1. The composition algebra. The capacity curve already showed the FHRR
   composition scales with phasor dimension; a larger vocabulary means
   a larger clean-up set, which raises the dimension needed only
   logarithmically. So the composition algebra is expected to scale to
   a large vocabulary cheaply. This sub-question is low-risk.
2. The substrate. A larger vocabulary needs more concept
   representations, and the limiting factor is whether the substrate's
   concepts stay separable and recognisable as their number grows. The
   compositional line found that the 16 v14/v16 concepts overlap by
   0.45 (mostly common-mode, removable) and recognise at about 0.93
   under temporal averaging. Whether those properties hold at, say, 64
   or 160 concepts is the real open question.

## Substrate options

The project already has larger-vocabulary substrates (per CLAUDE.md's
validated-asset inventory):

- The v17 28-word concept-pool architecture. Honest caveat from the
  record: v17's direct-binding recognition is weak (about 50% at
  28 words) -- a documented structural imbalance. Using v17 would mean
  the recognition bound dominates from the start, which is informative
  but pessimistic.
- The G.20 sparse-distributed ensemble. This is the project's
  validated large-vocabulary substrate: sparse scattered K-of-N
  concept codes, validated at 100% per-bridge discrimination for 64
  concepts and 98.4% for the 320-concept tier. Sparse-distributed
  codes are, by construction, more separable than the v14/v16
  contiguous-pool codes -- which is precisely the property the
  compositional line found limiting.

The G.20 sparse ensemble is the right substrate: it is the project's
own validated answer to vocabulary scaling, and its sparse codes
directly address the concept-separability limit the compositional line
identified. (A useful expected cross-check: the compositional line
found the v14/v16 concepts' overlap is mostly removable common-mode;
sparse-distributed codes should show low overlap without needing the
common-mode removal -- or confirm common-mode removal still helps.)

## Approach

Reuse, byte-unchanged: the activity-capture path (drive a concept,
record per-neuron concept-population activity) and the biologized
grounded-composition pipeline (longer-integration recognition +
common-mode-removed grounded symbol + resonate-and-fire FHRR + attractor
clean-up). Only the substrate and the vocabulary change.

Cheapest-to-falsify first: a single G.20 sparse bridge at 64 concepts
(the validated 100%-discrimination tier), not the full 160/320-concept
multi-bridge ensemble. Capture per-neuron concept activity from that
bridge (a GPU substrate run -- modest, one bridge), then run the
biologized grounded-composition pipeline on the captured activity. If
64 concepts hold, the 160/320 ensemble is the follow-on; if 64
concepts already fail, that is the honest finding and there is no
point capturing the larger tiers.

## Pre-registered test (fixed before the run, never tuned)

- The biologized grounded-composition pipeline, run on per-neuron
  activity captured from a 64-concept G.20 sparse bridge, against the
  frozen 0.80 compositional bar, multi-seed, loads {2,3,5}.
- PASS: integrated multi-seed mean >= 0.80. The biologized grounded
  compositional capability scales to a 64-concept vocabulary; proceed
  to the 160/320-concept ensemble.
- Recognition is reported separately and honestly: the per-observation
  and temporally-averaged concept-recognition accuracy at 64 concepts,
  compared to the ~0.66/0.93 measured at 16 concepts. If recognition
  degrades materially with vocabulary size, that is the honest finding
  -- it localises the vocabulary-scaling bound to substrate recognition,
  consistent with the compositional line's convergent result.
- NEGATIVE: integrated below 0.80. The honest finding is which stage
  costs it -- composition (unlikely, given the capacity curve),
  recognition, or concept separability at 64 concepts.

## Honest ceiling

A PASS at 64 concepts would show the biologized compositional capability
is not confined to a toy vocabulary. It would still be small-load
cue-to-attribute compositional retrieval -- not fluent open-ended
language. A NEGATIVE would honestly localise the vocabulary-scaling
bound. Either outcome is propagated. No protected, frozen, or validated
substrate module is modified; the G.20 sparse builder and the
biologized pipeline are reused as-is; no automatic differentiation.

## Sequence

This design doc, then writing-plans for the implementation, then a
subagent-driven build of the capture-and-run, then the GPU substrate
capture run (controller-monitored to completion), then the standard
discipline -- smell-test a PASS harder than a FAIL, dedicated
adversarial review before any capability claim, honest propagation to
both git remotes.
