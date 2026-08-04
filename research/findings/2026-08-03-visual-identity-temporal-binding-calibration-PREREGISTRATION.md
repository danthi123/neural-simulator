---
type: preregistration
status: retired
date: 2026-08-03
mechanism: visual-identity-temporal-binding
runner: research/runners/_laneD_visual_identity_temporal_binding_gate.py
---

# Visual identity temporal binding: calibration preregistration

**Filed after non-scientific smoke seed `220` was used for implementation checks
and before any scientific seed was run.** Seed `220` is reserved for smoke only
and cannot produce a formal verdict. The original scientific partition was
retired before execution; all partitions below are untouched.

## Functional requirement

The brain must associate different continuous views of one object with a stable
identity without receiving the object's label during learning or inference.
Identity should survive new combinations of position, scale, lighting, and
noise, while scrambled images and temporally shuffled experience should not
produce the same result.

## Mechanism under test

Graded fixed Gabor/V1-complex responses drive a first-spike-latency V1
population. Its spikes reach an identity population through permanences stored
in `SimulationBridge.cp_connections.data`. Local presynaptic traces,
postsynaptic spike winners, and short-lived postsynaptic persistence bind
adjacent frames from a continuous track. Local usage homeostasis prevents a
small set of identity units from monopolizing all objects. Inference receives
V1 spikes only; object labels remain outside the network for scoring.

## Seed and phase lock

- Non-scientific smoke only: `220`.
- Calibration, consumed and closed after NO-GO: `224`, `225`.
- Development, locked: `226`, `227`, `322`.
- Held out, locked: `323`, `324`, `325`.

The calibration tuple `224`, `225` was consumed and produced a recorded NO-GO.
This preregistration is retired; the runner now rejects those seeds and keeps
development and held-out phases closed. See
`2026-08-03-visual-identity-temporal-binding-calibration-NO-GO.md`.

## Fixed protocol and controls

Each seed uses the runner's frozen default configuration and six matched arms:
intact, exact-frame-multiset temporal shuffle, persistence lesion, presynaptic
trace lesion, homeostasis lesion, and learning off. Separate controls scramble
held images and test V1 and identity latency populations under graded drive,
flat drive, neural-drive lesion, and fast-spiking-pathway lesion.

Training and held transform combinations are disjoint. The temporal control
changes only frame order and preserves the exact frame multiset.

## Fixed validity preconditions

A result is `UNDEFINED` unless scientific partitions are fresh and disjoint
from smoke and earlier visual gates; train and held transforms are disjoint;
the shuffle preserves its exact frame multiset; labels never enter learning or
inference; V1 and identity winners come from first-spike activity rather than a
host top-k; and every scored numeric measurement is finite.

## Fixed scientific criteria

Every item must pass on both calibration seeds:

1. Intact held-view identity decoding is at least `0.50` against four-way
   chance `0.25`, and held-to-train cosine margin is positive.
2. Intact cosine margin is strictly greater than temporal shuffle, persistence
   lesion, and presynaptic-trace lesion.
3. Intact identity-unit usage coefficient of variation is strictly lower than
   the homeostasis lesion.
4. Learning-off decoding is at least `0.25` below intact decoding, and its
   held-to-train cosine margin is strictly below intact.
5. Pixel-scrambled held-view decoding is at most four-way chance `0.25`, and is
   at least `0.25` below intact decoding.
6. Intact local learning changes at least one substrate permanence; learning
   off changes exactly zero permanences.
7. V1 and identity latency populations each produce exactly their configured
   winner count under graded drive and zero winners under neural-drive lesion.
   Lesioning each population's fast-spiking pathway must strictly increase the
   number of columns that fire relative to its intact pathway.
8. Identity first-spike winners overlap the graded host reference by at least
   `0.80`. The host reference is measurement only and never selects a winner.
9. Flat identity drive overlap with graded winners is at most
   `k_win / n_col + 0.25`.

## Host boundary and scaffolds

The host still computes fixed Gabor responses, divisive normalization,
overlap-to-current scaling, presynaptic traces, postsynaptic persistence
current, first-spike readout, seeded same-step tie resolution, synthetic track
boundaries, and label-based scoring. These are explicit research scaffolds.
The host may not rank activations to select V1 or identity winners, provide an
object label to learning or inference, or choose the identity assembly.

This gate tests a small invariant visual representation, not natural vision or
general object understanding. A failure must be recorded without tuning these
seeds or weakening the causal controls.
