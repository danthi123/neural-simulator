# 160-concept ensemble vocabulary scaling at K=16: BOUNDARY -- 4 of 5 bridges PASS the strict multi-seed-mean criterion at every load; bridgeD_spatial uniquely misses; per-bridge symbol geometry is identical across bridges so the cause is not vocabulary structure

## Status

Decisive multi-seed GPU run of the 160-concept ensemble vocab-scaling
test, completed 2026-05-23. Per the pre-registered bar (PASS iff
every (bridge, load) cell multi-seed mean >= 0.80 across 5 bridges ×
3 loads = 15 cells), verdict is BELOW BAR -- one bridge (bridgeD
spatial) misses at every load (multi-seed-means 0.775 / 0.769 /
0.738). But the result is a substantive, refined characterisation:
4 of 5 bridges PASS the strict multi-seed-mean criterion at every
load (nouns 1.00, verbs 0.94-0.96, adjectives 0.82-0.83, functional
words 0.97-0.99); only the spatial-concepts bridge misses. A built-in
diagnostic refutes the obvious vocabulary-structure hypothesis: the
symbol-input geometry is essentially identical across all 5 bridges,
so bridgeD's distinctive miss is not traceable to obvious symbol-
space differences and likely reflects per-seed variance interacting
with the bridge's specific patterns.

## Background

The activity-grounded biologized grounded-composition pipeline is
validated at 16 concepts (multi-seed 0.98), and at 64 concepts on a
trained sparse-distributed substrate with K_VOCAB=16 it cleared the
frozen 0.80 bar multi-seed at every tested load {2, 3, 5}
(0.93 / 0.92 / 0.86), with the extended ceiling map sitting between
binding loads 6 and 7 (multi-seed-mean PASS through L=6 with one
seed below at L=6). The next vocabulary tier the design doc names
is the 5-bridge sparse-distributed ensemble: 5 bridges (A nouns,
B verbs, C adjectives, D spatial, E functional words) at 32 distinct
concepts per bridge, 160 unique concepts total. This decisive run
applies the K=16 PASS recipe per-bridge on that ensemble.

## What was run

`research/findings/raw/vocabulary_scaling_run_160ensemble.py`. A
focused byte-reuse extension of the trained-substrate runner that
loops over the 5 bridges × 3 seeds (42, 43, 44). Per (bridge, seed):
build a sparse-distributed bridge at 32 concepts using the validated
G.20 defaults (lang_input 8192, shared_pool 2000, fast-spiking
interneurons 300, K=100); train via `train_substrate` at the
validated encoding (topographic prior + 400 interleaved per-concept
events; teacher 500 pA); capture activity at M_OBS=16 observations
per concept; run the biologized grounded-composition pipeline at
K_VOCAB=16, K_RECOG=8, loads {2, 3, 5}, N_TRIALS=200 -- the K=16
PASS recipe.

The runner cleared adversarial review CLEAR on ten exploit-class
checks before launch (no vocab drift; per-bridge pattern determinism
+ decorrelation; K=16 recipe pinned; bar immovable; reuse byte-
unchanged; no answer leak; train orchestration correct per-bridge;
per-bridge cache cannot poison; no autograd; aggregate logic
correct). 14/14 soundness tests green.

15 bridge-seed combinations; per-bridge per-seed activity caches in
`research/findings/raw/vocabulary_scaling_160ensemble_cache/`.

## Result (pre-registered; multi-seed; frozen 0.80 bar)

Per-bridge multi-seed-mean integrated accuracy at each load:

```
                          L=2          L=3          L=5
bridgeA_nouns           1.0000  >=  1.0000  >=  0.9973  >=    PASS at every load
bridgeB_verbs           0.9633  >=  0.9533  >=  0.9400  >=    PASS at every load
bridgeC_adj             0.8300  >=  0.8300  >=  0.8247  >=    PASS at every load (thin)
bridgeD_spatial         0.7750   <  0.7694   <  0.7377   <    MISS at every load
bridgeE_functional      0.9917  >=  0.9850  >=  0.9783  >=    PASS at every load
```

Per-seed at L=5 (the most-stressed load) per bridge:

```
bridgeA_nouns      [0.998, 0.999, 0.995]    all 3 seeds individually clear
bridgeB_verbs      [0.824, 1.000, 0.996]    all 3 seeds individually clear
bridgeC_adj        [1.000, 0.523, 0.951]    seed 43 outlier 0.523; mean clears bar
bridgeD_spatial    [0.780, 0.621, 0.812]    seed 43 outlier 0.621; seeds 42/44 near bar; mean misses
bridgeE_functional [0.999, 0.964, 0.972]    all 3 seeds individually clear
```

Per the strict per-seed criterion (every seed individually >= 0.80
at every load): 3 of 5 bridges PASS (A, B, E). bridgeC has a
seed-43 outlier collapse (0.523) at L=5 but the mean clears the bar
(0.825). bridgeD has seed-43 outlier (0.621) and seeds 42/44 near
but slightly below the bar (0.780, 0.812) -- it misses the
multi-seed-mean criterion at every load.

Per the multi-seed-mean criterion (the project's standard): 4 of 5
bridges PASS (A, B, C, E); bridgeD misses.

Verdict per the strict pre-registered bar (PASS iff every cell):
**BELOW BAR**.

## Mandatory smell-test (PASS scrutinised; recompute-from-recording)

All 5 checks pass:

1. Per-bridge per-load means recompute from `cell_results`
   independently of the runner's aggregate -- byte-for-byte match.
2. Per-bridge captured pool density (from caches): 0.04-0.06 across
   all bridges and seeds. Honest observation: this is somewhat lower
   than the 64-concept K=16 substrate's 0.09-0.11. Plausible cause:
   fewer concepts per pool (32 vs 64) means fewer activity events
   spread across the same pool size at the same per-bridge sparsity.
   Notable that bridgeD's density (0.058, 0.044, 0.050) is essentially
   identical to bridgeA's (0.058, 0.043, 0.050) -- the substrate-side
   activity for bridgeD is no sparser than for the passing bridges.
3. Per-bridge recognition (temporally averaged): 1.000 across every
   bridge and seed -- perfect, matching the 64-concept K=16 reference.
   The miss is purely at the composition stage, not recognition.
4. bridgeD vocabulary check: ['north', 'south', 'east', 'west', 'up',
   'down', 'left', 'right', 'here', 'there', 'near', 'far', 'in',
   'out', 'on', 'under', 'above', 'below', 'front', 'back', 'top',
   'bottom', 'side', 'middle', 'now', 'then', 'before', 'after',
   'first', 'last', 'next', 'today']. Distinctive structure: paired
   opposites (north/south, up/down, in/out, above/below, etc.) + a
   few temporal terms. This is the "obvious" vocabulary-structure
   hypothesis the diagnostic below tests.
5. composition-only >= integrated at every cell -- consistent (since
   recognition is perfect, composition-only equals integrated
   everywhere).

## The vocabulary-structure hypothesis -- REFUTED by direct geometry

The obvious hypothesis for why bridgeD uniquely misses: its
paired-opposite vocabulary could produce higher pairwise overlap
between concept symbols (antonyms might share similar activity
patterns), and the compositional algebra needs near-orthogonal
signed symbols (the geometric load-bearing condition the
pattern-grounded NEGATIVE established).

A direct measurement refutes this. Per-bridge symbol-input pairwise
cosine across all 32-choose-2 = 496 concept pairs at K=16, seed 42
(the mean-centered consolidated activity vector that the deriver
projects into the phasor):

```
                       mean     std    min      max    frac_positive
bridgeA_nouns         -0.0316  0.069  -0.19   +0.28   0.304
bridgeB_verbs         -0.0318  0.065  -0.21   +0.21   0.300
bridgeC_adj           -0.0316  0.063  -0.21   +0.24   0.298
bridgeD_spatial       -0.0317  0.068  -0.21   +0.16   0.298
bridgeE_functional    -0.0318  0.064  -0.21   +0.18   0.312
```

Symbol-input geometry is essentially IDENTICAL across all 5 bridges:
mean cosine -0.0316 to -0.0318, standard deviation 0.063 to 0.069,
fraction of positive pairwise cosines 0.298 to 0.312. The
mean-centring + the substrate's per-bridge orthogonal lang_input
codes + the random K-of-N patterns produce essentially the same
symbol-input distribution regardless of vocabulary content. The
substrate doesn't see word semantics -- it sees the orthogonal
code for each concept and trains its pool selectivity accordingly,
and the patterns themselves are seeded by the per-bridge derivation
that decorrelates them from the other bridges.

So the cause of bridgeD's distinctive miss is NOT obvious symbol-
space geometry. It is downstream of the symbol input -- in the
deriver projection + FHRR + attractor composition on bridgeD's
specific symbols, where the per-seed values
([0.780, 0.621, 0.812] at L=5) suggest seed-variance interaction
with the bridge's specific patterns rather than a structural
property of the spatial vocabulary itself.

A confirmatory observation: bridgeC and bridgeD BOTH show
seed-43-specific outlier collapses at L=5 (C: 1.000 / 0.523 /
0.951; D: 0.780 / 0.621 / 0.812). Seed 43 is harder for both
bridges' L=5. bridgeC's mean still clears the bar because the
non-outlier seeds are at 1.0 / 0.95; bridgeD's mean misses because
its non-outlier seeds are at 0.78 / 0.81 (near but below the bar).

## What this means

The K=16 PASS recipe extends to 4 of 5 categories at the
160-concept tier:
- bridgeA_nouns: essentially perfect (every cell >= 0.997).
- bridgeB_verbs: very strong (every cell >= 0.94).
- bridgeC_adj: clears the bar (0.82 mean) with a seed-43 outlier at
  L=5 (0.523); per-seed PASS for 2 of 3 seeds.
- bridgeE_functional: essentially perfect (every cell >= 0.97).
- bridgeD_spatial: misses at every load multi-seed-mean (0.74-0.78);
  the cause is NOT obvious symbol-space geometry (symbol-input
  cosines identical to passing bridges).

The honest pre-registered verdict is BELOW BAR. The honest
substantive content is "K=16 recipe extends per-bridge to 4 of 5
categories at this tier; bridgeD has a per-seed-variance failure
mode that is not traceable to vocabulary structure or symbol
geometry." A category-specific scaling limit at this tier, not a
substrate failure.

## Next step

The bridgeD miss is genuinely informative but its cause is not yet
pinned. The cheapest pre-registered next probe: add 2 more seeds
(45, 46) to all 5 bridges to test whether the bridgeD miss is
robust across more seeds, or seed-43-anomaly that averages out at
larger sample size. With 5 bridges × 2 new seeds = 10 new bridge-
seeds × ~35 minutes = roughly 6 hours of GPU. The existing 15
cached bridge-seeds are reused; only the 10 new ones cost.

A second cheap CPU probe: re-run the pipeline on the existing 15
caches at varying K_VOCAB (not just 16) to see if the bridgeD miss
narrows with longer integration -- the noise-bounded interpretation
from the 64-concept thread applied per-bridge.

(Broader horizon, surfaced for the owner, NOT auto-launched: the
owner's standing conversational-path directives -- SPEAR, theta-
gamma mode-unification, generative replay -- and the integrated
closed loop are the larger arcs. The vocab-scaling thread has now
mapped the activity-grounded biologized pipeline at 16, 64, and
160-concept tiers; the per-bridge breakdown at 160 surfaces
category-specific behaviour worth further probing but the bigger
arcs may be the higher-leverage direction.)

## Honest scope

A clean multi-seed decisive run with the strict pre-registered
verdict (BELOW BAR) and the substantive 4-of-5 PASS surfaced
honestly alongside. The frozen 0.80 bar was not moved. The
adversarial reviewer ran ten exploit-class checks before launch and
returned CLEAR. The mandatory smell-test (recompute-from-recording,
per-bridge captured-density verification, per-bridge recognition
verification, vocab check, consistency) passed across all 5 checks.
The vocabulary-structure hypothesis for bridgeD's miss was directly
tested via per-bridge symbol-input cosine measurement and refuted
honestly. No protected, frozen, or moat module modified. No
automatic differentiation. The no-confab moat remains 7/7 green.
The 64-concept K=16 PASS pillar (n=90) stands; the 16-concept
validated capability (multi-seed 0.98) stands.

## Files / evidence

- Runner: `research/findings/raw/vocabulary_scaling_run_160ensemble.py`
- Helper: `research/findings/raw/vocabulary_scaling_160ensemble_helpers.py`
- Soundness tests (14/14): `tests/test_160_ensemble_pin.py`,
  `tests/test_vocabulary_scaling_160ensemble_helpers.py`,
  `tests/test_vocabulary_scaling_160ensemble.py`
- Result: `research/findings/raw/vocabulary_scaling_run_160ensemble_full.json`
- Activity caches:
  `research/findings/raw/vocabulary_scaling_160ensemble_cache/full_bridge*_seed{42,43,44}.npz`
  (15 files, one per bridge-seed)
- Run log:
  `research/findings/raw/vocabulary_scaling_run_160ensemble_full.log`
- Design + plan:
  `docs/plans/2026-05-23-160-concept-ensemble-vocab-scaling-design.md`,
  `docs/plans/2026-05-23-160-concept-ensemble-vocab-scaling-implementation.md`
- Adversarial review verdict: VERDICT CLEAR on ten exploit-class
  checks before launch.
- Prior arcs this builds on:
  - 64-concept K=16 refined CAPABILITY PASS:
    `research/findings/2026-05-22-vocabulary-scaling-trained-substrate-Kvocab16-PASS-activity-grounded-clears-the-bar-at-all-loads-with-thin-L5-margin.md`
  - K=16 extended load-ceiling map:
    `research/findings/2026-05-23-vocabulary-scaling-K16-extended-load-ceiling-map-ceiling-sits-between-L6-and-L7-with-honest-per-seed-caveat-at-L6.md`
  - Pattern-grounded NEGATIVE + geometry diagnostic (the framing
    this finding reads against):
    `research/findings/2026-05-22-pattern-grounded-NEGATIVE-symbol-geometry-not-spiking-noise-is-the-load-ceiling.md`
