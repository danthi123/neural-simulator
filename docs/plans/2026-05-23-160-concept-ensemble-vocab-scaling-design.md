---
type: plan
status: live
date: 2026-05-23
---

# 160-concept ensemble vocabulary scaling: 5 sparse-distributed bridges × 32 concepts per bridge at K=16, per-bridge biologized grounded-composition tested against the frozen 0.80 bar

## Status

Design, pre-registered. The next vocabulary tier the vocab-scaling
design doc names, executed on the project's validated 5-bridge
sparse-distributed concept ensemble at K_VOCAB=16 (the temporal-
integration budget that produced the refined PASS at 64 concepts).

## Background

The activity-grounded biologized grounded-composition pipeline
(longer-integration concept recognition + common-mode-removed
grounded symbols + resonate-and-fire phase-coded composition +
attractor clean-up with separate familiarity gate) is now thoroughly
characterised on the project's 64-concept sparse-distributed
substrate:

- Validated at the 16-concept tier: multi-seed mean 0.98 (the
  original twice-reviewed validation).
- At 64 concepts on the trained substrate with K_VOCAB=16 (use all
  16 cached observations per concept): refined CAPABILITY PASS,
  multi-seed-mean clears the frozen 0.80 bar at every tested load
  {2, 3, 5}; strict per-seed PASS through L=5; multi-seed-mean PASS
  extends through L=6 with one seed below the bar at L=6;
  ceiling between binding loads 6 and 7. Adversarially reviewed
  CLEAR on ten exploit-class checks.

The honest geometric mechanism is pinned: the compositional algebra
requires mean-centred signed symbols; the activity-grounded path
satisfies this via the brain's pooled-inhibition / subtractive-
normalisation step on the consolidated activity. Above the right
geometry, longer temporal integration closes residual spiking-symbol
noise — the K_VOCAB=16 result demonstrates this directly.

The next vocabulary tier the design doc names is the 5-bridge
ensemble: 5 sparse-distributed bridges (A nouns, B verbs, C
adjectives, D spatial terms, E functional words) at 32 distinct
concepts per bridge, 160 unique concepts in total. The project's
existing G.20 5-bridge ensemble (shipped 2026-05-15) has per-bridge
100% discrimination at this tier validated end-to-end with the
existing fixed vocabulary specification.

## The question this step asks

Does the activity-grounded biologized grounded-composition pipeline
at K_VOCAB=16 clear the frozen 0.80 compositional bar PER-BRIDGE on
each of the 5 validated sparse-distributed bridges (32 concepts per
bridge, 160 unique concepts total), multi-seed, at the same
compositional loads the 64-concept thread used?

This is the per-bridge compositional scaling question — does the
biologized pipeline's K=16 PASS at 64 concepts on a single bridge
extend to each of the 5 individually-validated 32-concept bridges?
A PASS means the activity-grounded pipeline is robust across the
project's full 5-bridge concept ensemble; a NEGATIVE means there
is per-bridge variation in scaling and the honest finding is which
bridge or category misses.

The question this step does NOT ask: cross-bridge compositional
binding. The validated G.20 ensemble has a cross-bridge sentence
demo (sentences spanning multiple bridges) using tag-based encoding,
but cross-bridge composition through the biologized grounded-
composition pipeline (with each bridge's symbols in a shared phasor
space) requires distinct architectural decisions and is a separate
further step. This design covers ONLY the per-bridge compositional
capability.

## Scope and reuse

What is reused, byte-unchanged:

- The validated 5-bridge vocabulary specification (the existing
  `research/runners/g20_vocab_spec` module that defines the 5
  bridges' 32-concept vocabularies and asserts global uniqueness
  across all 160 concepts).
- The validated sparse-distributed substrate builder
  (`build_sparse_pool_bridge` from
  `research/runners/concept_pool_sparse_distributed`).
- The validated G.20 encoding: `apply_sparse_topographic_prior` +
  `train_concept_sparse`, with the same validated default factors
  (topographic factor 10.0, off-target factor 0.1, training teacher
  500 pA, 400 interleaved events per concept).
- The biologized grounded-composition pipeline: `run_pipeline`,
  `recognition_accuracy`, `_ground_symbols`, `_cosine`,
  `partition_cue_filler`, all imported byte-unchanged from
  `vocabulary_scaling_run.py`.
- The `train_substrate` helper from
  `vocabulary_scaling_run_trained.py`, byte-unchanged.
- The activity-capture helper `capture_concept_activity` from
  `vocabulary_scaling_run.py`, byte-unchanged.
- The `_save_cache`/`_load_cache` helpers, byte-unchanged.

What is genuinely new:

- A multi-bridge orchestration runner that loops over the 5 bridges
  per seed: build each bridge with the bridge's 32-concept vocab
  + patterns, train it via the validated encoding, capture activity
  with M_OBS=16 (matching the K=16 budget), run the biologized
  pipeline at K_VOCAB=16 loads {2, 3, 5}, aggregate per-bridge
  per-seed results.
- A per-bridge per-seed cache (separate cache directory; cache key
  = bridge name + seed) for kill-safe resume at per-bridge
  granularity.
- A per-bridge sparse-pattern derivation matched to each bridge's
  32-concept vocabulary (the existing
  `generate_sparse_patterns(n_concepts=32, ...)` reused).

## Substrate sizing per bridge

Each of the 5 bridges is built at 32 concepts using the validated
G.20 sparse-distributed defaults:
- `n_lang_input = 8192`
- `n_shared_pool = 2000`
- `n_shared_fs = 300`
- `pattern_size (K) = 100`
- `sparsity = 0.01` (the same orthogonal-drive sparsity the
  64-concept thread used; stride 8192/32 = 256, so n_active =
  round(0.01 × 8192) = 82 < 256 — geometry holds with the same
  margin)

These match the substrate sizing the K=16 PASS used at 64 concepts.

## Pre-registered test (fixed; never tuned)

Same biologized pipeline. Same frozen 0.80 bar. Same multi-seed
grid {42, 43, 44}. Same compositional loads {2, 3, 5}. Same K_RECOG
= 8 and K_VOCAB = 16 (the K=16 PASS recipe). Same M_OBS = 16 cached
observations per concept. Same partition_cue_filler split (first 16
of each bridge's 32 concepts are cues; last 16 are fillers).

PRE-REGISTERED reading:

- **PASS**: integrated multi-seed mean >= 0.80 at every load
  {2, 3, 5} on every bridge (5 bridges × 3 loads = 15 cells; all
  cells must clear). The activity-grounded biologized compositional
  capability extends to the 160-concept ensemble per-bridge at K=16.

- **NEGATIVE**: integrated below 0.80 at some load on some bridge.
  The honest finding is which bridge or category misses, and the
  per-bridge characterisation maps it (some categories may scale
  more cleanly than others, with specific reasons that connect back
  to the concept-content of each bridge's vocabulary).

The strict per-seed criterion is also recorded honestly alongside
the multi-seed-mean criterion (matching the K=16 thread's
characterisation), with any one-seed-below caveats explicitly
preserved.

## Soundness considerations

A PASS at this tier must survive these adversarial checks before
being claimed:

1. Each bridge is genuinely trained per the validated G.20
   encoding (the same byte-unchanged path the 64-concept thread
   uses). The dedicated adversarial review must verify the training
   stage is not a no-op for any bridge (e.g. silent gate failures
   on bridge B that wouldn't be caught by aggregate metrics).

2. The vocabulary partition is the one
   `research/runners/g20_vocab_spec` defines (with its global
   uniqueness assertion). The runner does NOT regenerate vocabs or
   patterns with a different seed; the substrate that is captured
   from is the substrate that was trained on.

3. The K_VOCAB=16 recipe is identical to the 64-concept K=16 PASS;
   no per-bridge tuning of K, K_RECOG, n_trials, deriver seed, or
   any pipeline parameter. The bar is unchanged.

4. The recognised concept name (the OUTPUT of recognition) is the
   only handle that names which concept's pattern / activity is
   read at any step. The true label NEVER indexes anything in the
   compositional or symbol-derivation path. (Inherited from the
   K=16 PASS's adversarial review; must be re-verified at the
   multi-bridge orchestration level.)

5. The reuse is genuinely byte-unchanged: the runner's `git diff`
   must add only new files (the multi-bridge runner, soundness
   tests, a brief design doc cross-reference); no modification to
   any protected, frozen, moat, or previously-reviewed module.

6. No automatic differentiation; the no-confab moat remains 7/7
   green.

7. The smell-test must scrutinise a PASS HARDER than a NEGATIVE:
   recompute every bridge's per-load means from the recording;
   recompute the captured pool density from each bridge's activity
   cache and confirm it sits in the same regime as the 64-concept
   trained substrate (well above the untrained near-silent
   baseline); confirm per-bridge per-load values are non-degenerate
   (not all identical, not all 0 or 1).

## Implementation outline (TDD plan to follow separately)

Task 0: grounding pin (constants + bar unchanged across the multi-
bridge tier; existing 5-bridge vocab spec total concept count =
160; per-bridge concept count = 32).

Task 1: a small `bridge_vocab_and_patterns(bridge_name, seed,
n_pool, k)` helper that pulls each bridge's 32-word vocab from
`g20_vocab_spec` and generates its sparse K-of-N patterns via the
validated `generate_sparse_patterns`. Pure function;
deterministic; unit-tested (pin shape, count, uniqueness across
all 160 concepts).

Task 2: the runner
`research/findings/raw/vocabulary_scaling_run_160ensemble.py` — a
focused multi-bridge extension of the trained-substrate runner:
the per-bridge orchestration loop builds, trains via
`train_substrate`, captures via `capture_concept_activity` at
M_OBS=16, runs `run_pipeline` at K_VOCAB=16 loads {2,3,5}; per-
bridge per-seed activity cache; aggregate verdict over 5 × {2,3,5}
= 15 cells; --smoke mode for a tiny end-to-end check (reduced
bridge sizes, 2 bridges instead of 5, tiny vocab subset; toy
numbers NOT propagated).

Task 3: soundness tests. Pin the multi-bridge load-bearing
properties: each bridge's vocab matches g20_vocab_spec; per-bridge
patterns are reproducible from seed; the runner does not regenerate
vocabs from a different seed; the train -> capture handoff on a
tiny bridge produces well-formed cached activity.

Task 4: dedicated adversarial reviewer (fresh agent, full tool
access, RUNS exploit-class checks) on the multi-bridge runner
BEFORE the decisive GPU run.

Task 5: CONTROLLER-ONLY decisive GPU run. Wall-clock estimate:
approximately 35 minutes per bridge per seed (build + train +
capture), 5 bridges × 3 seeds = 15 bridge-seed combinations,
roughly 9 hours total. Kill-safe at per-bridge per-seed
granularity (each bridge-seed cache file is independent). Pipeline
runs are cheap CPU after the captures are cached. Mandatory anti-
cheat smell-test (recompute from the recording; per-bridge
captured pool density check; per-load consistency). Honest
propagation either way; on a PASS a dedicated fresh adversarial
review before any capability-pillar claim.

## Wall-clock and resource estimate

Per bridge per seed:
- Build: about 1 minute
- Train (32 concepts × 400 events × 31 steps = 397 thousand steps
  on an 18.7 thousand-neuron bridge at GPU speed): about 29 minutes
  (scaling linearly from the 64-concept thread's 58 min/seed)
- Capture (32 concepts × 16 obs × 120 steps = 61 thousand steps):
  about 5 minutes
- Total per bridge per seed: about 35 minutes

5 bridges × 3 seeds × 35 minutes = approximately 9 hours total
GPU. Pipeline composition runs (per bridge per seed × 3 loads)
are cheap CPU after captures are cached: a few minutes total.

The run is kill-safe at per-bridge per-seed granularity — a kill
mid-run loses only the in-flight bridge-seed's training; a re-launch
resumes from the next un-cached bridge-seed. Within a bridge-seed
the existing cache pattern is seed-granular; that matches the
existing 64-concept thread's discipline.

## Honest scope

A focused next step on the vocab-scaling thread. Whatever the
verdict, it is one further test in a continuing line — not a final
answer to the larger question of how a brain composes at scale.
The completed twice-reviewed 16-concept arc (multi-seed 0.98)
stands; the 64-concept K=16 refined CAPABILITY PASS (multi-seed-
mean PASS at loads 2-3-5; characterised ceiling between L=6-7)
stands. Whatever this 160-concept step delivers is read in the
context of those prior validated results.

A PASS here is a meaningful capability extension to the full
5-bridge concept ensemble at K=16; a NEGATIVE is an honest per-
bridge characterisation of where the activity-grounded biologized
pipeline scales cleanly and where it does not. The cross-bridge
compositional question is explicitly out of scope for this step;
that is a separate larger design.

Frozen bar never tuned; reuse-by-import only; no protected,
frozen, or moat module modified; no automatic differentiation; the
no-confab moat must remain 7/7 green throughout. The biology-
translatable insight set from the K=16 thread (mean-centring as
the geometric load-bearing condition; longer integration as the
noise-bounded ceiling-closing mechanism) is the framing this 160-
concept tier reads against.
