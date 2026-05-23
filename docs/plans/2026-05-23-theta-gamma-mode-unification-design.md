# Theta-gamma mode-unification: cheap-first numpy probe — order-bearing AND order-invariant as modes of one theta-gamma encoded code

## Status

Design + cheap-first numpy probe of the algebra. Pre-registered. The
catalog-documented Lisman-Idiart N.16 mechanism the owner explicitly
flagged on 2026-05-19 ("the key catalog-documented interconnection
the project never built"; "order-bearing vs order-invariant are
operating MODES of one theta-gamma code, not two stores") as
load-bearing for the conversational path. This is the FHRR-arc
cheap-first pattern reapplied: probe the algebra in numpy first; if
the algebra works the spiking implementation is a separate
pre-registered next step; if it doesn't the algebra needs refinement
before any spiking commitment.

## Background

The renewed-focus compositional investigation produced a validated
first compositional capability (the activity-grounded biologized
pipeline at 16/64/160 concepts, with the FHRR-biologization arc
delivering an end-to-end biology-faithful pipeline that clears the
frozen 0.80 bar at multi-seed 0.98 on the 16-concept substrate, the
64-concept K=16 PASS, the 160-concept per-bridge characterisation).
But the substrate retrieval capability is order-INVARIANT by design
(the consolidated activity for each concept averages over
observations; sequences collapse to bags). The earlier
necessity-instrument line ran TERMINAL on the conjunction
(consolidation-lesion-necessary AND episodic-serial-order-recoverable
in one regime) because that conjunction IS the complementary-
learning-systems division of labour and cannot co-hold under a
static two-store retrieval.

The owner's 2026-05-19 scientific reframe named the biological
resolution: order-bearing AND order-invariant are operating MODES of
one theta-gamma code (GABAergic regime), not two separate stores.
One shared ~125 ms theta rhythm time-multiplexes write vs read
phases (SPEAR -- already built and tested); within a theta cycle the
gamma rhythm (~25 ms, ~40 Hz) segments the cycle into ~5-7 slots; an
item placed in gamma slot k at theta cycle t encodes both its
IDENTITY (which item) and its ORDER (which slot, i.e. which phase
position). The same code can be READ in two complementary modes:

- ORDER-BEARING: unbind each gamma-slot position from the code to
  recover the item at that position; sequence the recovered items
  by slot to reproduce the ordered sequence.
- ORDER-INVARIANT: marginalise the readout over the gamma-slot
  positions; the score for each candidate item summed across all
  slot positions recovers the unordered set {item_k} without
  position information.

This is the structural mechanism N.16 from the project's catalog
(Lisman & Idiart 1995, Lisman & Jensen 2013) and what Buzsaki frames
as theta-gamma coding. SPEAR built the encode-vs-retrieve timing
multiplexing; mode-unification is the orthogonal readout mechanism
that turns one code into two complementary information products.

## The question this probe asks

Does the FHRR phase-coded vector-symbolic algebra support both
order-bearing AND order-invariant readout from the SAME encoded code
at usable accuracy, where the encoding represents an ordered
K-item sequence by binding each item to its gamma-slot position
phasor and bundling?

This is a pure algebra question. If the algebra works the spiking
implementation (a separate pre-registered step) is meaningful; if
the algebra doesn't even work the spiking implementation cannot.

## The mechanism

Reuses the validated FHRR primitives byte-unchanged.

Encode a K-item ordered sequence ``(item_1, item_2, ..., item_K)``
from a vocabulary of N distinct concepts:

```
C = bundle_k=1..K [ bind(item_k, position_k) ]
```

where ``position_k`` is a fixed deterministic phasor representing
the k-th gamma slot in the theta cycle (5-7 slot phasors, the
biologically grounded number). Position phasors are near-orthogonal
random phase vectors (the standard FHRR position-encoding pattern).

Order-bearing readout, for each slot k:

```
candidate_item_k = nearest_match( unbind(C, position_k), vocab )
```

Sequence the recovered items by slot index k = 1..K. PASS iff the
recovered sequence equals the encoded sequence (every position
correct).

Order-invariant readout, scoring each item w in the vocabulary:

```
score(w) = sum_k=1..K  inner_product( unbind(C, position_k), w )
                                              = inner_product( C, bundle_k bind(w, position_k) )
```

Top-K items by score = the recovered unordered set. PASS iff the
recovered set equals the encoded set (every item present,
regardless of order).

The KEY pre-registered question: BOTH readouts must work on the SAME
encoding C. The mode-unification claim is that one code supports
both information products at usable accuracy.

## Pre-registered test (fixed; never tuned)

Same FHRR phasor dimension as the validated compositional pipeline
(N_DIM = 512). Vocabulary size = 32 (mid-tier between the 16 and 64
the vocab-scaling thread covered). Compositional loads {2, 3, 5}
matching the project's standard. Multi-seed (42, 43, 44). 200
random sequences per load per seed.

PRE-REGISTERED reading:
- PASS: both order-bearing AND order-invariant multi-seed-mean
  accuracy >= 0.80 at every load {2, 3, 5}. The mode-unification
  claim is supported by the FHRR algebra. The spiking biologized
  implementation is the next pre-registered step.
- NEGATIVE_ORDER_BEARING_ONLY: order-bearing PASSes but order-
  invariant misses. The algebra supports order-bearing readout via
  position-unbind but not the marginal scoring -- the unification
  claim's "one code" fails on one side.
- NEGATIVE_ORDER_INVARIANT_ONLY: order-invariant PASSes but order-
  bearing misses. The algebra supports set-recovery via marginal
  scoring but the per-position unbind crosstalks too much --
  unification fails on the other side.
- NEGATIVE_BOTH: neither mode clears the bar. The FHRR algebra at
  this dim does not support unified bidirectional readout at these
  loads on these vocab sizes.

## Soundness considerations

A PASS at this probe must survive these adversarial checks before
being claimed:

1. The vocabulary phasors and position phasors are generated
   deterministically from the seed (no per-trial regeneration); same
   vocab + same positions across all 200 trials per (seed, load).
   The mode-unification is on a fixed encoding regime, not a
   per-trial tuned one.

2. The order-bearing readout uses ``nearest_match`` over the WHOLE
   N-vocab vocabulary at each position; no oracle restriction to
   the true items (a real readout has no privileged access to the
   sequence's contents).

3. The order-invariant readout's top-K is over the WHOLE vocabulary;
   the score does NOT use the true item identities.

4. The position phasors are fixed across loads -- at load L=5, the
   first 5 positions of the 7 gamma slots are used; at L=2 the
   first 2; at L=3 the first 3. The same position phasors mean a
   given test is comparable across loads.

5. The probe is numpy-only; no protected, frozen, or moat module
   modified; no automatic differentiation; the FHRR primitives are
   reused by import byte-unchanged from
   ``research/runners/spiking_phasor_fhrr.py`` (the validated
   compositional module).

6. The frozen 0.80 compositional bar is the SAME bar the vocab-
   scaling thread used. Not redefined; not scaled per readout mode.

## Implementation outline (TDD plan to follow separately if PASS)

This first probe is a single focused script -- no multi-task TDD
plan needed for a numpy algebra check. Soundness via a smell-test
recompute after the run plus a one-shot adversarial review of the
script before it is propagated as anything more than a probe.

Single-file probe: `research/findings/raw/theta_gamma_mode_unification_probe.py`.
Reads the FHRR primitives from `spiking_phasor_fhrr` (existing,
validated). Pre-registered constants verbatim:

```
N_DIM = 512
N_VOCAB = 32
LOADS = [2, 3, 5]
N_TRIALS = 200
N_GAMMA_SLOTS = 7        # the biological 7 slots per theta cycle
SEEDS = [42, 43, 44]
BAR = 0.80               # the project's frozen compositional bar
```

The probe builds a deterministic per-seed vocab + position phasor
set, runs N_TRIALS encode/decode trials per (load, seed), aggregates
the two readout-mode accuracies per load multi-seed, applies the
pre-registered reading, writes JSON.

After PASS or NEGATIVE_*: a focused findings doc + capability_status
entry (status BOUNDARY if PARTIAL, status VALIDATED if both modes
PASS subject to a fresh adversarial review on the probe code).

## Honest scope

A cheap-first algebra probe on the catalog-documented Lisman-Idiart
N.16 mechanism. Numpy only; no GPU; no spiking implementation; no
new substrate code. Whatever the verdict, it is a foundational check
on whether the project's chosen vector-symbolic algebra supports the
biological theta-gamma mode-unification at usable accuracy on
relevant vocabularies and compositional loads. A PASS justifies the
substantial multi-week biologized-spiking implementation as the
next pre-registered step (with the same biology-faithful discipline
the FHRR-biologization arc used). A NEGATIVE is itself a clean
biology-translatable finding (the FHRR algebra at this dimension
does not support unified readout; the substrate would need a
different code or a higher dimension).

No protected, frozen, or moat module modified. No automatic
differentiation. The frozen 0.80 compositional bar is unchanged.
Reuse-by-import only for the FHRR primitives.
