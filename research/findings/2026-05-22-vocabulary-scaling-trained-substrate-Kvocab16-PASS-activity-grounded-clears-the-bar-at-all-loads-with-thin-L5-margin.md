# Vocabulary scaling on the trained 64-concept substrate: activity-grounded pipeline clears the 0.80 bar multi-seed at all loads {2,3,5} with K=16 observations -- refined PASS with thin L=5 margin

## Status

Refined capability finding. The activity-grounded biologized
grounded-composition pipeline, on the trained 64-concept sparse-
distributed substrate, clears the frozen 0.80 compositional bar
multi-seed at every tested compositional load {2, 3, 5} when the
symbol-consolidation step averages all 16 cached observations per
concept (K_VOCAB=16, the cache maximum). Multi-seed integrated means
0.933 / 0.924 / 0.864 at L=2 / 3 / 5; per-seed L=5 values 0.898,
0.817, 0.877 — every seed individually above the bar. The mandatory
anti-cheat smell-test (the sweep's built-in sanity check) passed
byte-for-byte: K=8 reproduces the trained-substrate decisive
recording exactly (0.8417 / 0.8139 / 0.7560). A dedicated adversarial
reviewer (fresh agent, full tool access, RAN every check) returned
VERDICT CLEAR on all ten exploit-class checks, with one honest
non-blocking caveat preserved here.

## Background

The trained-substrate decisive run (K_VOCAB=8, the original default)
produced a BOUNDARY result: passed the strict 0.80 bar at loads 2 and
3 multi-seed (0.842, 0.814), missed at load 5 (0.756, miss by 0.044).
The load-ceiling characterisation pinned the ceiling between loads 3
and 4. A subsequent pattern-grounded variant (testing the spiking-
noise hypothesis) decisively FAILED to chance and its built-in
diagnostic localised the actual cause precisely to symbol geometry:
the algebra requires near-orthogonal signed symbols; the activity-
grounded path satisfies this via mean-centring (subtractive
normalisation / pooled inhibition); the pattern-grounded path does
not. The remaining gap from the activity-grounded L=5 ceiling toward
the geometry-clean reference (~1.000) was hypothesised to be residual
spiking-symbol noise on top of clean geometry, that more observations
per concept (longer temporal integration) should close.

This refined finding is the test of that pre-registered hypothesis on
the existing cached observations.

## What was run

`research/findings/raw/vocabulary_scaling_kvocab_sweep_probe.py`. A
cheap CPU sweep on the existing trained activity cache; no GPU, no
re-train, no new capture. K_VOCAB sweeps over the natural log2
ladder {1, 2, 4, 8, 16}; K_RECOG (recognition window) is held fixed
at 8; loads and seeds unchanged from the decisive run; the biologized
pipeline is imported byte-unchanged from the adversarially-reviewed
trained-substrate runner. K_VOCAB=16 = use all 16 cached observations
per concept (the cache maximum, not a cherry-picked tuning point).

## Result

```
                     multi-seed integrated mean
                     L=2     L=3     L=5
K_VOCAB= 1           0.023   0.036   0.032     (~chance, 1 obs too noisy)
K_VOCAB= 2           0.443   0.381   0.338
K_VOCAB= 4           0.792   0.783   0.703
K_VOCAB= 8           0.842   0.814   0.756     (decisive recording -- BOUNDARY)
K_VOCAB=16           0.933   0.924   0.864     PASS at every load

K_VOCAB=16 per-seed L=5: [seed 42] 0.898  [seed 43] 0.817  [seed 44] 0.877
```

Sanity contract: K=8 multi-seed means reproduce the trained-substrate
decisive recording byte-for-byte (0.8417 / 0.8139 / 0.7560 — exact
match). The sweep is deterministic from the cache; the decisive
result is reproducible.

The curve is monotonic in K and matches the classic noise-averaging
signature (accuracy rising as the symbol's effective signal-to-noise
ratio rises with more averaged observations). The L=5 lift from K=8
to K=16 is +0.108.

## Mandatory smell-test (PASS scrutinised harder than NEGATIVE)

The result was scrutinised per the discipline. The sweep tool's
built-in sanity check (K=8 byte-for-byte vs the decisive recording)
passed exactly. A dedicated adversarial reviewer ran ten
exploit-class checks (is K=16 config-tuning a NEGATIVE; is the bar
unchanged; is K=8 sanity exact; is the pipeline byte-unchanged; is
there an answer leak; per-seed PASS at L=5; does the claim overstate;
consistency with the geometry-NEGATIVE; no autograd; overall
smell-test) and returned CLEAR with no defect. Key points the
reviewer verified:

- The K_VOCAB ladder is natural log2 doubling, not a tuned point;
  K=16 is the cache MAXIMUM. The pre-registered noise-bounded
  hypothesis was committed in `a54928a` (the candidate-2 NEGATIVE
  propagation) at 23:57 BEFORE the sweep result in `01bfb32` at 00:09
  the next day. The pre-registration predates the result.
- The bar is unchanged (BAR=0.80 in the imported pipeline; never
  redefined or scaled in the sweep).
- The pipeline is byte-unchanged (`git diff-tree 01bfb32` shows only
  the new probe file; no modification to the runner or any protected
  module; the no-confab moat is 7/7 green).
- K_VOCAB and K_RECOG are independent; the sweep holds K_RECOG=8 so
  the recognition path is identical to the decisive run.
- Per-seed L=5 PASS at K=16: all three seeds individually above the
  0.80 bar (0.898, 0.817, 0.877; minimum +0.017 above bar).
- Consistency with the geometry-NEGATIVE: the activity-grounded path
  always had signed mean-centred geometry by construction; this
  sweep reduces residual spiking-symbol noise ON TOP of correct
  geometry. Nothing walked back.

## The honest non-blocking caveat

The L=5 margin is thin. Multi-seed mean 0.864 (below the
pre-registered "above 0.90" lift target the noise-bounded hypothesis
spelled out, though clearly above the 0.80 bar). The lowest per-seed
L=5 value is 0.817, only +0.017 above the bar. The PASS is real and
multi-seed (every seed individually clears the bar at every load),
but the L=5 margin is narrow and would benefit from confirmation at
additional seeds or a longer integration budget.

K=16 is the cache MAXIMUM. The curve at K > 16 is not tested (the
cache has only 16 observations per concept). Re-running the trained
substrate with a deeper M_OBS would test whether the L=5 mean keeps
climbing toward the ~1.0 reference, or whether ~0.86 is the true
ceiling at this substrate density. Both are honest possibilities
inside the noise-bounded interpretation.

## What this is, and what it is not

This is a refined CAPABILITY finding on the activity-grounded
biologized grounded-composition pipeline at a 64-concept vocabulary.
The capability is: with the validated G.20 sparse-distributed
substrate trained per the validated encoding, and with the symbol
consolidation averaging all 16 cached observations per concept, the
biologized pipeline clears the frozen 0.80 bar multi-seed at every
tested compositional load {2, 3, 5}.

It is NOT a claim that 64-concept composition is solved in general,
or that the pipeline composes at unbounded loads. The honest map of
the activity-grounded pipeline on this substrate, now reading all
results together:

- Pure compositional algebra (no biology, same phasor dimension 512):
  clears the 0.80 bar past load 96.
- Activity-grounded, K=16 (this finding): clears the bar at loads
  {2, 3, 5} multi-seed; L=5 margin thin (0.864 mean; lowest seed
  0.817).
- Activity-grounded, K=8 (original decisive): BOUNDARY (loads 2-3
  PASS; L=5 misses at 0.756).
- Pattern-grounded (no spiking noise, broken geometry): chance.
- Mean-centered-pattern (no noise, fixed geometry, oracle-adjacent
  diagnostic only): ~1.000 (mechanism pinpoint, not propagated as
  capability).

The biology-translatable insight set is unchanged from the
geometry-NEGATIVE finding and sharpened by this PASS: the
compositional algebra requires mean-centred signed symbols (the
geometric load-bearing condition), the activity-grounded path
satisfies that requirement via mean-centring (subtractive
normalisation / pooled inhibition), and the residual ceiling at K=8
is residual spiking-symbol noise that LONGER TEMPORAL INTEGRATION
(more observations averaged) closes -- exactly the kind of operation
a brain naturally performs when reading a noisy population code.

## Next step

The vocabulary-scaling thread on the activity-grounded biologized
pipeline is now thoroughly characterised: validated at 16 concepts
(multi-seed 0.98), cleared the strict bar at 64 concepts multi-seed
{2,3,5} at K=16 with a thin L=5 margin, with the geometric and noise
mechanisms precisely pinned. Open follow-up options, in order of
cheap-first:

(a) **Re-run the trained substrate with a deeper observation budget
    (M_OBS > 16)** to extend the K curve past 16 and pin down whether
    the activity-grounded L=5 ceiling sits near ~0.86 or keeps
    climbing toward 1.0. This is a focused GPU re-run on the existing
    substrate; medium-cost.

(b) **Extend the K=16 result to the broader load-ceiling map** (re-
    run the load-ceiling probe at K=16, loads {2..7}) to characterise
    how far the noise-averaged activity-grounded path extends past
    load 5. Cheap CPU; no GPU.

(c) **Scale to 160/320-concept ensembles** (the next vocabulary tier
    the design doc names) on the trained substrate at K=16, as the
    next pre-registered vocab-scaling step. Medium-cost GPU per
    bridge.

(d) The owner's standing broader-horizon items (SPEAR theta-gamma
    multiplexing, generative replay, integrated closed loop): the
    larger arcs, NOT auto-launched.

The natural next pre-registered step within this thread is (b) — a
cheap re-run of the load-ceiling probe at K=16 that fully maps the
activity-grounded ceiling at the cache's full integration budget.

## Honest scope

A refined capability finding produced by a cheap CPU follow-up on
the existing trained activity cache, adversarially reviewed CLEAR,
with one honest non-blocking caveat (thin L=5 margin) preserved
front and centre. No protected, frozen, or moat module modified;
the substrate runner, the substrate cache, the biologized pipeline,
and the frozen 0.80 bar are all unchanged. No automatic
differentiation. The completed twice-reviewed 16-concept
biologization arc (multi-seed 0.98) stands; the trained-substrate
K=8 BOUNDARY result stands; the geometry-NEGATIVE and its
mean-centered-pattern diagnostic stand. The no-confab moat is 7/7
green.

## Files / evidence

- Probe: `research/findings/raw/vocabulary_scaling_kvocab_sweep_probe.py`
- Result: `research/findings/raw/vocabulary_scaling_kvocab_sweep_probe.json`
- Trained activity cache (read for recognition + symbol consolidation):
  `research/findings/raw/vocabulary_scaling_trained_cache/trained_full_seed{42,43,44}.npz`
- Adversarial review verdict: VERDICT CLEAR on ten exploit-class
  checks (no config-tuning, bar unchanged, K=8 sanity exact, pipeline
  byte-unchanged, no answer leak, K_VOCAB/K_RECOG independent,
  per-seed PASS, claim limits stated, geometry-NEGATIVE consistency,
  no autograd, smell-test honest).
- The pre-registration commit (predates result): `a54928a` (records
  the noise-bounded hypothesis + K_VOCAB sweep plan).
- The K=8 BOUNDARY this refines:
  `research/findings/2026-05-22-vocabulary-scaling-trained-substrate-BELOW-BAR-with-loads-2-3-PASS-and-load-5-ceiling.md`
- The geometry-NEGATIVE that pinpointed the mechanism:
  `research/findings/2026-05-22-pattern-grounded-NEGATIVE-symbol-geometry-not-spiking-noise-is-the-load-ceiling.md`
