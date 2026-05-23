# Biologized spiking theta-gamma mode-unification: decisive verdict NEGATIVE_ORDER_INVARIANT_ONLY -- order-invariant readout PASSes multi-seed 1.000; the TPAM attractor (the FHRR-biologization arc's validated identification mechanism for fact composition) does NOT transfer to per-slot mode-unification identification at the 32-concept scale; diagnostic localises the failure precisely to the TPAM attractor (the algebra and the per-slot unbinds are clean)

## Status

Decisive multi-seed run of the biologized spiking theta-gamma mode-
unification implementation, completed 2026-05-23. Per the pre-
registered trichotomy from the design doc: VERDICT =
NEGATIVE_ORDER_INVARIANT_ONLY -- order-invariant readout clears the
frozen 0.80 bar multi-seed-mean at every load (1.0000 / 1.0000 /
0.9817 at L=2 / 3 / 5); order-bearing readout via TPAM attractor
settle + argmax decisively misses (0.5283 / 0.3350 / 0.0867 at L=2 /
3 / 5). A built-in diagnostic localises the failure precisely to
the TPAM attractor mechanism (a SIMPLER biology-grounded order-
bearing decoder -- argmax of phase-similarities to grounded vocab
symbols, no TPAM -- gives 1.000 across all loads/seeds on the same
substrate, same encoded codes, same unbinds). The biology-
translatable insight set: the algebra-PASS holds on the biologized
substrate (the per-slot unbinds are clean); the FHRR-biologization
arc's TPAM attractor (validated 0.98 on fact composition at 16
concepts) does NOT transfer to per-slot mode-unification
identification at this scale.

## Background

The cheap-first numpy algebra-PASS probe (2026-05-23, multi-seed
1.000 across both readouts at every load) and the comprehensive
characterisation (algebra capacity envelope wide across load, noise,
vocab; algebra survives substrate-realistic noise std=1.6) justified
the biologized spiking implementation as the next pre-registered
step. The design doc records the pre-registered trichotomy verdict
labels and the build's reuse-by-import-only architecture: validated
trained substrate + activity capture + FHRR-biologization arc's
resonate-and-fire neurons + ResonateFireTPAM attractor + the K=16
PASS recipe; genuinely-new code is gamma-slot position phasors and
the two-readout orchestration.

## What was run

`research/findings/raw/biologized_spiking_mode_unification_runner.py`,
adversarially reviewed CLEAR on 11 exploit-class checks before
launch. One bridge (bridgeA_nouns, 32 concepts), multi-seed (42,
43, 44), K=16 PASS recipe, frozen 0.80 bar, compositional loads
{2, 3, 5}, 200 trials per load per seed.

Substrate, training, and per-concept activity capture: reused the
existing 160-ensemble bridgeA_nouns cache files (byte-identical to
what the pre-registered runner would have produced: same bridge,
same seeds, same `train_substrate` + `capture_concept_activity` +
all parameters). The pre-registered runner was then re-launched
with the cache present so it skipped the build+train+capture step
and ran the pipeline pure-CPU; the recorded result is the
biologized pipeline running on the validated trained substrate's
captured activity.

## Result (pre-registered; multi-seed; frozen 0.80 bar)

```
                                multi-seed-mean
                                order-bearing       order-invariant
L=2                             0.5283   miss       1.0000   PASS
L=3                             0.3350   miss       1.0000   PASS
L=5                             0.0867   miss       0.9817   PASS

Per-seed L=5:
  bridgeA_nouns seed 42         TPAM-OB 0.21        OI 1.00
  bridgeA_nouns seed 43         TPAM-OB 0.18        OI 0.99
  bridgeA_nouns seed 44         TPAM-OB 0.02        OI 0.94
```

Order-invariant PASSes essentially perfect at every cell. Order-
bearing via TPAM is far below the bar at every cell and decays
sharply with load (0.53 -> 0.34 -> 0.09). VERDICT per the pre-
registered trichotomy: NEGATIVE_ORDER_INVARIANT_ONLY (order-
invariant clears the bar; order-bearing misses).

## Diagnostic localising the failure

The TPAM-OB number is below chance for large loads (1/32 single-
argmax chance is 0.03; correct K-tuple chance is 0.03^K, so K=5
chance is ~3e-8; observed 0.087 multi-seed at L=5 is well above
that, indicating partial signal but TPAM-specific corruption).

A direct probe of TPAM behaviour on the smoke-scale cache (8 vocab)
showed the TPAM consistently converges to vocab item 7 regardless
of the per-slot unbind input (overlaps ~410 for item 7 vs ~50-300
for the true items). At the full 32-concept scale the diagnostic
repeated the same pattern: TPAM systematically biased toward a
spurious mass-attractor.

A SIMPLER order-bearing decoder -- per-slot argmax of
`phase_similarity(unbind_k, grounded[w])` over the full vocabulary,
no TPAM -- gave 1.000 accuracy across every load/seed cell on the
SAME unbinds the TPAM decoder operated on:

```
                     TPAM-OB (pre-registered) | simple-OB (diagnostic)
seed 42 L=2                    0.57                      1.00
seed 42 L=3                    0.46                      1.00
seed 42 L=5                    0.21                      1.00
seed 43 L=2                    0.57                      1.00
seed 43 L=3                    0.41                      1.00
seed 43 L=5                    0.18                      1.00
seed 44 L=2                    0.36                      1.00
seed 44 L=3                    0.18                      1.00
seed 44 L=5                    0.02                      1.00
```

This is reported only as a mechanism-pinpointing diagnostic, NOT as
a capability claim. The pre-registered runner's verdict per the
pre-registered TPAM decoder stands as NEGATIVE_ORDER_INVARIANT_ONLY.
The diagnostic shows:
- The encoded code C and per-slot unbinds are clean (top-1 by
  phase-similarity is the correct item at every slot).
- The FHRR algebra works on the biologized substrate (the bind /
  unbind / bundle primitives carry the encoded sequence faithfully).
- The biologized identification step (TPAM attractor settle + argmax)
  is the failing component: spurious mass-attractor at 32-concept
  vocabulary scale on the project's grounded symbols.

## Why TPAM works for fact composition but not mode-unification

The FHRR-biologization arc validated TPAM at 0.98 multi-seed on
FACT composition at the 16-concept tier. The mode-unification at
32 concepts collapses. Two structural differences:

1. In fact composition, the TPAM was built over FILLER WORDS only
   (a partition of the vocabulary). Each query unbinds with a CUE
   (also in vocabulary but in the cue partition) and recovers a
   filler. TPAM has a small basin set (filler vocabulary only).
2. In mode-unification, the TPAM is built over the FULL vocabulary
   (32 concepts). Each query unbinds with a POSITION (not in
   vocabulary; a separate gamma-slot phasor) and recovers an item
   from the full vocabulary. TPAM has 32 basins.

At 32 basins on grounded symbols whose mean pairwise overlap is
-0.05 (per the FHRR-biologization arc's measurement), the basin
structure has a mass-attractor that captures most queries
regardless of input. The annealed-threshold settle (with its fixed
high theta) cannot escape this basin.

Biologically: the cortical attractor network the TPAM models has a
known SCALING limit -- as the number of stored patterns grows,
spurious mass-attractors appear and can capture queries. The
literature notes this for Hopfield-class networks (Amit & Treves
1989 capacity-edge analysis). The FHRR-biologization arc validated
TPAM at small vocabularies where this isn't yet an issue; mode-
unification at 32 concepts crosses the threshold for these
particular grounded-symbol overlap properties.

## What this means

The biologized spiking implementation realises the algebraic
half of the mode-unification claim cleanly: the ORDER-INVARIANT
readout (marginal-sum of per-slot phase-similarities to grounded
vocab symbols) clears the frozen 0.80 bar multi-seed at every
tested load on the biologized substrate. The ORDER-BEARING readout
via the pre-registered TPAM attractor decoder does NOT transfer
from fact composition to per-slot mode-unification identification
at this scale; the diagnostic localises the failure precisely to
the TPAM attractor's spurious-mass-attractor regime on 32 grounded
symbols.

The honest biology-translatable insight: the FHRR-biologization
arc's identification mechanism (the TPAM attractor) has a per-
vocab-size scaling limit that the fact-composition validation
(at 16-concept tier) did not surface. Per-slot mode-unification
identification at 32 concepts is where the limit bites. A simpler
biology-grounded alternative (parallel population matching =
argmax of phase-similarities to stored vocab symbols) gives 1.000
on the same data, but using it for the pre-registered verdict
would be post-hoc decoder-substitution and was not pre-registered.

## What this is, and what it is not

This is the pre-registered NEGATIVE_ORDER_INVARIANT_ONLY verdict on
the biologized spiking mode-unification implementation, with a
clean diagnostic localising the failure to the TPAM attractor
mechanism (not the algebra, not the substrate, not the per-slot
unbind quality).

It is NOT a claim that biologized spiking mode-unification is
impossible. The algebra works on the substrate; the unbinds are
clean; one of the two readouts PASSes. The failure is in one
specific component (the TPAM attractor) that was validated for a
narrower compositional regime and doesn't scale to this one.

It is NOT a claim that the project should switch to argmax-of-
similarities as the identification mechanism. That would be a new
pre-registered step with its own design + adversarial review +
oracle-adjacency analysis (argmax over substrate-stored grounded
symbols is biology-grounded but it does re-introduce some of the
"stored-vocabulary clean-up" structure the FHRR-biologization arc
worked to avoid).

It IS a clean biology-translatable result: the FHRR-biologization
arc's TPAM attractor identification mechanism has a per-vocab-size
scaling limit; per-slot mode-unification identification at 32
concepts is past that limit; marginal-sum-of-similarities readout
(the order-invariant decoder) does NOT need an attractor and works
at the biologization tier directly. The substrate side and the
algebra side both behave as the FHRR-biologization arc's framework
predicted; the identification side has a refined ceiling.

## Next step

The pre-registered next-step trichotomy in the design and
implementation plans is honored by this honest negative. The
substantive follow-up options (in order of cheap-first):

(a) **Diagnose the TPAM-attractor scaling limit precisely.** A cheap
    probe could measure TPAM identification accuracy at vocab sizes
    {8, 16, 24, 32, 48, 64} on grounded symbols to map where the
    mass-attractor regime emerges. ~minutes CPU, reuses the same
    pipeline. This would yield a precise biology-translatable
    statement about the project's biologized attractor mechanism's
    per-vocab capacity ceiling.

(b) **A new pre-registered runner using the simpler parallel-
    population-matching decoder for order-bearing.** This is a
    different biology-grounded identification mechanism (argmax of
    phase-similarities to grounded vocab symbols), with its own
    pre-registered test + adversarial review. It would test whether
    a different biology-grounded identification mechanism delivers
    mode-unification both-readouts PASS on the biologized substrate.

(c) **Surface for owner steer on the broader-horizon arcs.** The
    mode-unification thread has now produced the algebra-PASS
    (cheap probe), the capacity-envelope characterisation
    (cheap probe), and the biologization NEGATIVE_ORDER_INVARIANT_
    ONLY with the failure mechanism precisely localised. The
    project's brain-analogue goal might be better served by moving
    to generative replay (which builds on top of mode-unification
    once the order-bearing side is solved) or to the broader
    integrated closed loop the owner named.

## Honest scope

A pre-registered decisive run on the biologized spiking
implementation with a clean honest NEGATIVE_ORDER_INVARIANT_ONLY
verdict, the failure mechanism localised by a built-in diagnostic
to the TPAM attractor. The frozen 0.80 bar was not moved. The
adversarial reviewer ran 11 exploit-class checks before launch and
returned CLEAR. The mandatory smell-test was the diagnostic that
isolated the TPAM as the failing component (predicted the verdict
exactly; reuses the existing 160-ensemble bridgeA_nouns cache
byte-identical to what the pre-registered runner would produce).
No protected, frozen, or moat module modified. No automatic
differentiation. No-confab moat 7/7 green. The algebra-PASS pillar
and characterisation findings stand; the FHRR-biologization arc's
prior validated 0.98 multi-seed on fact composition at 16 concepts
stands (this NEGATIVE is on a different compositional task at a
larger vocabulary scale, not a regression).

## Files / evidence

- Runner: `research/findings/raw/biologized_spiking_mode_unification_runner.py`
- Helper: `research/findings/raw/biologized_spiking_mode_unification_helpers.py`
- Soundness tests (14/14 green):
  `tests/test_biologized_spiking_mode_unification_pin.py`,
  `tests/test_biologized_spiking_mode_unification_helpers.py`,
  `tests/test_biologized_spiking_mode_unification.py`
- Result:
  `research/findings/raw/biologized_spiking_mode_unification_runner_full.json`
- Activity caches (byte-identical to the 160-ensemble bridgeA_nouns
  cache; reused unchanged):
  `research/findings/raw/biologized_spiking_mode_unification_cache/full_seed{42,43,44}.npz`
- Design + plan:
  `docs/plans/2026-05-23-biologized-spiking-mode-unification-design.md`,
  `docs/plans/2026-05-23-biologized-spiking-mode-unification-implementation.md`
- Adversarial review verdict: VERDICT CLEAR on 11 exploit-class
  checks before launch.
- Prior arcs this builds on:
  - Algebra-PASS:
    `research/findings/2026-05-23-theta-gamma-mode-unification-cheap-numpy-probe-ALGEBRA-PASS-Lisman-Idiart-N16-realisable-on-FHRR.md`
  - Characterisation:
    `research/findings/2026-05-23-theta-gamma-mode-unification-characterisation-capacity-envelope-wide-on-all-three-axes-algebra-survives-substrate-realistic-noise.md`
  - The TPAM identification mechanism this finds the scaling limit
    of: the FHRR-biologization arc's
    `research/findings/2026-05-22-attractor-cleanup-biologization-shortcut-3-RESOLVED-abstention-is-a-separate-familiarity-signal-not-a-basin-property.md`
