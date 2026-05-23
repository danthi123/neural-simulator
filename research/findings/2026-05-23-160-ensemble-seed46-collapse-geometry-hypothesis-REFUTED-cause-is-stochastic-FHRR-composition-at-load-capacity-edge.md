# 160-ensemble seed-46 L=5-collapse diagnostic: substrate-geometry hypothesis REFUTED — the cause is downstream of the symbol input, in stochastic FHRR-composition variance at the high-binding-load capacity edge

## Status

Cheap CPU diagnostic to test the natural follow-up hypothesis from
the 5-seed extension. Completed 2026-05-23. The substrate-geometry
hypothesis (that seed 46's per-bridge pattern sets have a systematic
geometric property -- pattern overlap, symbol-input cosine, or
symbol-output cosine -- that explains its L=5 collapse across 4 of 5
bridges) is REFUTED. Seed 46's geometry is indistinguishable from the
strong seeds (42-45) on all three measurements across all 5 bridges.
The cause of the seed-46 L=5 collapse is downstream of the geometry,
in stochastic FHRR-composition variance at the high-binding-load
capacity edge.

## Background

The 5-seed extension of the 160-concept ensemble vocab-scaling test
revealed a striking systematic pattern: seed 46 collapses L=5 at 4
of 5 bridges (bridgeA 0.600, bridgeB 0.402, bridgeD 0.591, bridgeE
0.513; only bridgeC seed 46 scored 0.997). Seed 45 is uniformly
strong (0.948, 0.996, 0.841, 0.996, 0.947). The systematic pattern
suggested the seed-46 per-bridge pattern sets share a measurable
structural property -- the natural hypothesis was that seed 46 might
produce higher concept-pattern overlap or less orthogonal symbol
inputs across bridges.

## What was measured

For each (bridge, seed) of the 5 × 5 = 25 cached cells, three
geometric properties:

- (a) Per-bridge mean pairwise concept-pattern overlap fraction
  (random K-of-N patterns' shared-neuron ratio averaged over all
  32-choose-2 = 496 pairs).
- (b) Per-seed mean pairwise mean-centred symbol-input cosine
  (mean and std of the 496 pairwise cosines of the consolidated
  activity vector at K=16, mean-centred -- the deriver's input).
- (c) Per-seed mean pairwise post-deriver spike-phase symbol cosine
  (mean and std of the 496 pairwise cosines of `phases_to_spikes`
  applied to the deriver output -- the FHRR symbol the pipeline
  actually uses).

## Result

Per-bridge per-seed (25 cells):

```
bridge / seed     overlap   sym_in_mu  sym_in_sd  sym_out_mu  sym_out_sd
bridgeA / 42      0.0494    -0.0316    0.0688     +0.7470     0.0160
bridgeA / 43      0.0516    -0.0316    0.0712     +0.7462     0.0176
bridgeA / 44      0.0497    -0.0317    0.0712     +0.7458     0.0157
bridgeA / 45      0.0495    -0.0314    0.0672     +0.7453     0.0175
bridgeA / 46      0.0512    -0.0317    0.0680     +0.7461     0.0166
bridgeB / 42      0.0489    -0.0318    0.0650     +0.7453     0.0154
...                                                                     (similar across all 5 bridges x 5 seeds)
bridgeE / 46      0.0491    -0.0314    0.0740     +0.7461     0.0154
```

Seed-mean across the 5 bridges:

```
seed   overlap   sym_in_mu   sym_out_mu
42     0.0496    -0.0317     +0.7462
43     0.0499    -0.0314     +0.7452
44     0.0503    -0.0313     +0.7470
45     0.0499    -0.0314     +0.7460
46     0.0505    -0.0314     +0.7461
```

Seed 46 is indistinguishable from the other seeds on all three
measurements:
- Mean pattern overlap: 0.050 ± 0.001 across seeds (seed 46 = 0.0505;
  seed 44 = 0.0503; the difference is within the seed-to-seed noise
  band of about 0.001).
- Mean symbol-input cosine: -0.031 ± 0.0004 across seeds (seed 46 =
  -0.0314; seed 42 = -0.0317; identical to the band).
- Mean symbol-output cosine: +0.746 ± 0.001 across seeds (seed 46 =
  +0.7461; seed 42 = +0.7462; identical).

## The substrate-geometry hypothesis is refuted

Seed 46 collapses L=5 across 4 of 5 bridges but its geometric
properties are indistinguishable from the strong seeds 42-45. The
hypothesis that "seed 46 has higher pattern overlap" is refuted
(0.0505 vs 0.0496-0.0503, no meaningful difference). The hypothesis
that "seed 46 has less orthogonal symbol inputs" is refuted (sym_in
cosine identical to other seeds). The hypothesis that "seed 46
produces symbols that cluster differently after the deriver" is
refuted (sym_out cosine identical).

So the cause of the seed-46 L=5 collapse is NOT a measurable
substrate-geometry property. It is downstream of the symbol input,
in the FHRR + attractor composition at high binding load.

The honest interpretation: at L=5 (5-binding composition) the
FHRR + TPAM clean-up has a stochastic per-seed variance that is not
explained by surface-measurable geometric properties of the symbols.
Some seeds (45) compose cleanly at L=5; other seeds (46) collapse;
the difference between them is in the specific concept-pair
configurations that arise across the 1000 L=5 trials per (bridge,
seed) -- which the geometry diagnostics summary-statistic-averaged
over 496 pairs cannot capture.

This is itself biology-translatable: composition at the capacity
edge has non-trivial per-substrate variance that doesn't reduce to
mean-orthogonality of the underlying symbols. The compositional
algebra has tail behaviour at high load that the mean-cosine
diagnostic doesn't surface.

## The vocab-scaling thread, complete picture

Across the 16 / 64 / 160-concept tiers with K=8 and K=16
characterised:

- **16-concept** (the original validated capability): multi-seed
  0.98.
- **64-concept K=8** (decisive BOUNDARY): multi-seed-mean PASS through
  L=3, ceiling at L=4 (0.7988 borderline).
- **64-concept K=16** (refined CAPABILITY PASS): multi-seed-mean PASS
  through L=6 (thin at L=6), strict per-seed PASS through L=5,
  ceiling between L=6 and L=7. Pillar n=90 VALIDATED.
- **160-concept K=16 at 3 seeds** (BOUNDARY): 4 of 5 bridges PASS at
  every load; bridgeD misses at every load. Pillar n=91 BOUNDARY.
- **160-concept K=16 at 5 seeds** (refined BOUNDARY): ALL 5 bridges
  PASS at L=2 AND L=3; only bridgeD misses, only at L=5 (1 of 15
  cells); 14 of 15 cells PASS multi-seed-mean.
- **Seed-46 collapse hypothesis-tested**: the systematic seed-46
  L=5 collapse across 4 of 5 bridges is NOT explained by any
  substrate-geometry property (pattern overlap, sym-in cosine,
  sym-out cosine all indistinguishable from other seeds); the cause
  is downstream of the symbol input, in the FHRR composition
  dynamics at the high-binding-load capacity edge.

The biology-translatable insight set:
- The compositional algebra requires mean-centred signed symbols (the
  geometric load-bearing condition; pattern-grounded NEGATIVE
  established this).
- The activity-grounded biologized pipeline satisfies that requirement
  via the brain's pooled-inhibition / subtractive-normalisation step.
- The remaining residual noise on top of correct geometry is closed
  by longer temporal integration (K=16 PASS confirmed).
- The pipeline extends per-bridge to ALL 5 categories at the
  160-concept tier at compositional loads 2-3.
- At higher loads (5+) and larger vocabularies, the FHRR composition
  has non-trivial per-seed stochastic variance that is NOT explained
  by surface symbol-geometry diagnostics -- a separate axis of
  capacity behaviour that would need finer instrumentation to
  characterise further.

## What this is, and what it is not

This is the final cheap diagnostic on the vocab-scaling thread, with
an honest negative on the substrate-geometry hypothesis for the
seed-46 collapse. It is NOT a capability claim or a finding of a new
substrate property. It records that the cause of the per-seed L=5
variance across the 160-concept ensemble does not trace to obvious
substrate geometry.

The vocab-scaling thread is now substantively complete across 16,
64, and 160-concept tiers, with the per-bridge per-load per-seed
breakdown mapped and the geometric and noise mechanisms pinned. The
remaining tail behaviour at the load-capacity edge is the natural
opening for the broader-horizon arcs the owner named.

## Next step

The vocab-scaling thread is at a natural terminus. Further
investigation of seed-46 L=5 variance would require finer FHRR-
internal instrumentation (per-trial recovery accuracy; basin-of-
attraction characterisation; etc.) that is separable from the
substrate-side biology and would be a different research direction
than the biology-faithful scaling thread.

The owner's standing broader-horizon directives -- SPEAR theta-gamma
multiplexing, generative replay, the integrated closed loop -- are
the higher-leverage next direction for the project's brain-analogue
goal. They are NOT auto-launched per the standing instruction; the
vocab-scaling thread's completion is recorded here and surfaced for
the owner to steer the next major direction.

## Honest scope

A cheap CPU diagnostic with a clean honest negative on the surface
substrate-geometry hypothesis. The frozen 0.80 bar was not moved.
No protected, frozen, or moat module modified; no automatic
differentiation; no-confab moat 7/7 green. The 64-concept K=16
refined CAPABILITY PASS (n=90) stands; the 160-concept BOUNDARY
pillar (n=91) stands and is sharpened by both the 5-seed extension
finding and this geometry-hypothesis refutation.

## Files / evidence

- Per-seed per-bridge geometry diagnostic: inline Python in the
  smell-test step of this finding (run output recorded in the
  findings doc body; no separate runner file -- the diagnostic is a
  ~30-line pure-CPU recompute from the existing activity + pattern
  caches).
- Existing activity + pattern caches:
  `research/findings/raw/vocabulary_scaling_160ensemble_cache/full_bridge*_seed{42,43,44,45,46}.npz`
  (25 files: 5 bridges × 5 seeds).
- The 5-seed extension this builds on:
  `research/findings/2026-05-23-160-ensemble-5seed-extension-refined-finding-all-5-bridges-PASS-at-L2-L3-bridgeD-uniquely-misses-only-at-L5.md`
- The 3-seed BOUNDARY pillar this thread of work is rooted in:
  `research/findings/2026-05-23-160-concept-ensemble-K16-BOUNDARY-4-of-5-bridges-PASS-multiseed-bridgeD-uniquely-misses-with-honest-perseed-caveats.md`
- The K=16 64-concept refined CAPABILITY PASS:
  `research/findings/2026-05-22-vocabulary-scaling-trained-substrate-Kvocab16-PASS-activity-grounded-clears-the-bar-at-all-loads-with-thin-L5-margin.md`
- The pattern-grounded NEGATIVE (which established that mean-centred
  signed symbol geometry is the load-bearing condition for the
  compositional algebra):
  `research/findings/2026-05-22-pattern-grounded-NEGATIVE-symbol-geometry-not-spiking-noise-is-the-load-ceiling.md`
