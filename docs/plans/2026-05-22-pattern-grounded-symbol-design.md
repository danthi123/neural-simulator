---
type: plan
status: live
date: 2026-05-22
---

# Pattern-grounded compositional symbols: a focused next step within the biologization arc

## Status

Design. Pre-registered. A new step on the same biologization arc that
produced the activity-grounded compositional pipeline (which clears the
frozen 0.80 compositional bar on the validated 16-concept substrate at
multi-seed 0.98, and clears it at multi-seed 0.84/0.81 at compositional
loads 2 and 3 on the trained 64-concept sparse-distributed substrate
but ceilings between loads 3 and 4 -- the spiking-symbol noise floor's
capacity reduction relative to the pure algebra).

## Background, in plain words

A short orienting paragraph for an informed reader who hasn't tracked
the arc.

The project's compositional pipeline takes one concept's name, looks
up the substrate's response to that name (a per-neuron firing pattern
in a shared pool of 2000 neurons -- the "captured activity"), and
derives a complex-valued phase vector from it (a "phasor"). Two such
phasors are combined into a single compositional phasor by elementwise
phase arithmetic; a query phasor extracts a stored binding back out;
an attractor clean-up step picks the nearest stored phasor as the
answer. This is a biological realisation of phase-coded
vector-symbolic composition (Frady & Sommer 2019).

Each concept's response sits on a fixed sparse set of pool neurons --
the concept's K-of-N pattern (here K=100 neurons of N=2000). On the
trained substrate the language-input drive selectively excites the
concept's own pattern, the strong feedback inhibition shapes the
result, and the captured firing-rate vector encodes the concept's
identity well enough that recognition is essentially perfect
(temporally averaged accuracy 1.000 multi-seed). But the captured
activity is also noisy at the per-observation level: each pool neuron
either fires or does not on a given trial, with rates around 1-2
spikes per 100ms. That spiking noise propagates into the
derived-phasor symbol and limits how many bindings can be packed into
one compositional phasor before the symbols crosstalk past the
attractor's basin -- which is what produces the load ceiling.

## The question this step asks

If we replace the noisy activity-derived symbol with a symbol derived
DIRECTLY from the concept's K-of-N pattern (a clean binary vector that
does not change from trial to trial), does the compositional load
ceiling rise -- and by how much?

The reference curve is the load-ceiling map from the activity-grounded
run on the same trained substrate, same seeds, same loads:

```
L=2 0.8417 PASS   L=3 0.8139 PASS   L=4 0.7988 miss
L=5 0.7573        L=6 0.7225        L=7 0.6721
```

The pure phasor algebra (numpy reference probe, same phasor dimension
512) clears the bar past load 96 -- so there is large headroom for a
cleaner symbol to climb into.

## The mechanism

The symbol derivation path is the only thing that changes; everything
else (recognition, the compositional pipeline, the attractor clean-up,
the frozen 0.80 bar, the per-seed and per-load test grid) is reused
unchanged.

In the activity-grounded path the grounded symbol for concept w is

  phi(w) = deriver( consolidated(w) - common_mode )

where consolidated(w) averages the captured activity vector across the
registration observations of w, common_mode is the mean across all
concepts (subtractive normalisation -- a real cortical computation),
and deriver is a fixed linear projection from the per-neuron activity
space to a 512-dimensional phase vector.

In the pattern-grounded path the grounded symbol for concept w is

  phi(w) = deriver( pattern_vector(w) )

where pattern_vector(w) is the concept's K-of-N pattern represented as
a binary 0/1 vector over the same 2000-neuron pool (1 at the K=100
pattern neurons, 0 elsewhere), and deriver is the SAME fixed linear
projection.

Two design choices to be precise about:

1. The pattern is read from the substrate's own stored concept code
   (the per-concept sparse pattern that the training stage shaped the
   language-input pathway to evoke). The recognition front-end
   continues to read from captured activity -- recognition's job is to
   map an observed noisy activity vector back to its concept identity;
   only the symbol derivation step uses the pattern.

2. The deriver is unchanged -- the same fixed-seed linear projection
   the activity-grounded path uses. Same input dimensionality, same
   output dimensionality, same construction; only the input value
   (binary pattern vs noisy activity) differs.

## The honest oracle-adjacency caveat

The K-of-N pattern is the substrate's own concept code, not a freely
chosen vector. It is stored in the trained connectivity (the
topographic prior plus the training stage shape the language-input
pathway so the pattern is exactly what that drive evokes), and the
recognition front-end has to extract the concept's identity from
noisy activity in order to even know which pattern to read. So
pattern-grounded symbols ARE substrate-grounded in a real sense -- a
brain that has a stable cell ensemble for a concept can read out
"this ensemble's identity" cleanly, and that identity is itself a
biological signal, not a hand-supplied lookup.

But pattern-grounded is also one step CLOSER to the oracle-lookup
shortcut than activity-grounded is. The activity-grounded path
explicitly says "the symbol is whatever the substrate's
trial-to-trial activity actually produces"; the pattern-grounded path
abstracts past that trial-to-trial variability to the underlying
identity-defining ensemble. On a strict biological reading the brain
has both: noisy spiking activity AND a stable ensemble identity each
concept maps to (engram cells, in Tonegawa's language). Pattern-
grounding uses the ensemble identity directly as the symbol's input
and so trades some biological faithfulness (the per-observation noise
is hidden) for cleaner symbols. The honest framing: pattern-grounded
sits between activity-grounded (more faithful, noisier) and
oracle-lookup (no biology) -- closer to the latter than the former,
but still substrate-grounded.

This caveat is recorded up front so a PASS at pattern-grounded is
read for what it is: a refinement that uses the substrate's stable
ensemble identity as the symbol's input, not a biological compositional
result at the same fidelity as activity-grounded.

## Pre-registered test (fixed; never tuned)

Same biologized grounded-composition pipeline as the decisive
trained-substrate runner. Same recognition front-end (temporally
averaged nearest-match in the captured activity space). Same FHRR
operations (resonate-and-fire bind / unbind / bundle). Same attractor
clean-up with the separate familiarity gate. Same multi-seed
{42, 43, 44}. Same compositional loads {2, 3, 5}. Same FHRR phasor
dimension (512). Same 0.80 compositional bar.

The ONLY change: the grounded symbol for each concept is derived from
the concept's K-of-N pattern vector instead of from its mean-centred
captured activity. The pattern vectors are read deterministically from
the substrate's per-concept pattern store (generated by the substrate
builder at the seeded RNG; the substrate trained on those exact
patterns).

PRE-REGISTERED reading:

- PASS: integrated multi-seed mean >= 0.80 at all loads {2, 3, 5}.
  Pattern-grounded clears the bar where activity-grounded missed at
  L=5. The honest interpretation: the spiking-symbol noise floor IS
  the load-ceiling cause; replacing the noisy symbol with the clean
  pattern symbol raises the ceiling. Subject to the oracle-adjacency
  caveat above, a PASS demonstrates that compositional capability at
  64 concepts and loads up to 5 is achievable with the trained
  substrate's own stable ensemble identity as the symbol input.

- NEGATIVE: integrated below 0.80 at some load. The spiking-symbol
  noise is NOT the (only) ceiling cause. The honest finding is that
  the limit is deeper -- attractor crosstalk at 64 concepts even with
  clean symbols, or another mechanism. Sharpens the diagnosis.

Either outcome is a clean, publishable, biology-translatable result.

## Soundness considerations

A PASS at pattern-grounded must SURVIVE these adversarial checks
before being claimed:

1. The pattern vectors are NOT being treated as oracle-like
   per-concept symbols freely chosen for separability. They are read
   from the substrate's per-concept pattern store, identical to what
   the substrate was trained on; they are sparse random K-of-N codes
   with the natural pairwise overlap that comes from random sampling
   (about 5 in 100, by birthday calculation).

2. The recognition front-end remains the load-bearing identity
   extractor: it must continue to read noisy activity and resolve to
   the correct concept. If recognition were skipped (the symbol
   chosen by oracle from the task labels), that would be the oracle
   shortcut. The recognition step is NOT changed; pattern-grounded
   only changes the symbol's INPUT after recognition.

3. The frozen 0.80 bar is unchanged, the loads, the seeds, the
   pipeline shape are all unchanged. A PASS or NEGATIVE here is
   strictly comparable to the activity-grounded decisive run.

4. The control: the FRESH (untrained) substrate's recognition is
   weaker; pattern-grounded with untrained recognition would inherit
   that weakness. We run pattern-grounded on the TRAINED substrate
   only -- the same one the activity-grounded decisive run used --
   so that the comparison is symbol-derivation only, with the
   recognition front-end identical.

5. The runner must NOT short-circuit recognition. The recognition
   output (the recognised concept name) must remain the only handle
   that selects which pattern vector to ground the symbol from. If
   the runner were to use the true label rather than the recognised
   label, that would be an answer leak. The dedicated adversarial
   review must run this exploit-class check explicitly.

## Implementation outline (TDD plan to follow separately)

Task 0: grounding pin -- a test that asserts the validated
encoding constants and the frozen 0.80 bar are unchanged.

Task 1: a small `pattern_vector(concept_idx, pool_size, pattern)` helper
that builds the binary 0/1 vector over the pool from a stored sparse
pattern. Pure function; unit-tested.

Task 2: the pattern-grounded runner
`research/findings/raw/vocabulary_scaling_run_pattern_grounded.py` --
a focused byte-reuse extension of the trained-substrate runner that
substitutes the symbol-derivation step with pattern_vector + deriver.
Reuses the entire pipeline, capture, recognition, run_pipeline,
multi-seed aggregate, smell-test by import. The genuinely-new code is
the new `_ground_symbols_pattern` function and the runner's main()
orchestration.

Task 3: soundness tests. The load-bearing property is that the
grounded symbol's input differs from the activity-grounded path AND
that the recognition front-end remains the only handle on which
pattern is read. The tests pin both.

Task 4: dedicated adversarial reviewer (fresh agent, full tool
access, RUNS the exploit-class checks): no answer leak (the true
label is never used to index the pattern store); recognition genuinely
load-bearing; the deriver is identical to the activity-grounded path;
no protected module modified; no automatic differentiation; the
frozen bar immovable.

Task 5: CONTROLLER-ONLY decisive run. Reuses the trained activity
cache (recognition still reads from it). Multi-seed {42, 43, 44},
loads {2, 3, 5}. GPU not strictly required (the pipeline is numpy on
the cache), but the same harness-tracked background pattern applies if
needed. Mandatory anti-cheat smell-test (separate tool) recomputes
the verdict from the recording. Honest propagation either way; on a
PASS a fresh adversarial review before any capability claim.

After Task 5, regardless of outcome, the load-ceiling characterisation
is re-run on the pattern-grounded pipeline to extend the comparison
across loads {2..7}, completing the comparison against the
activity-grounded reference curve.

## Honest scope

A focused next step on the biologization arc. Whatever the verdict, it
is one further test in a continuing biologization line, not a final
answer to the larger question of how a brain composes. The completed
twice-reviewed 16-concept FHRR-biologization arc (multi-seed 0.98)
stands; the trained-substrate 64-concept BOUNDARY result (multi-seed
0.84 / 0.81 at loads 2-3, ceiling between 3 and 4) stands. The
oracle-adjacency caveat above is recorded up front and any PASS is
read with that caveat in mind. The frozen 0.80 bar is never moved;
reuse-by-import only; no protected, frozen, or moat module modified;
no automatic differentiation.
