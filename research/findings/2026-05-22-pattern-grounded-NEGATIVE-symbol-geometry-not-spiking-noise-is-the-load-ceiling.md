# Pattern-grounded compositional symbols on the trained 64-concept substrate: NEGATIVE at chance; the load ceiling's cause is symbol geometry, not spiking-symbol noise

## Status

The decisive multi-seed run of candidate 2 of the vocabulary-scaling
NEGATIVE branch. Pre-registered verdict: NEGATIVE (multi-seed
integrated 0.038 / 0.033 / 0.029 at loads {2, 3, 5} -- essentially
chance for the 32-filler argmax). The pre-registered hypothesis behind
candidate 2 -- that the spiking-symbol noise is the load-ceiling cause
-- is REFUTED.

A built-in diagnostic test (mean-centered pattern indicators, as a
confirmatory measurement only -- not a capability claim) pinpoints the
actual cause precisely: symbol geometry. The compositional algebra
(spiking-phasor FHRR + threshold-phasor associative-memory clean-up)
requires near-orthogonal symbols with both positive and negative
pairwise correlations; the raw binary K-of-N pattern indicator
violates this by construction. The activity-grounded path satisfies
it via the mean-centring step a brain naturally performs (subtractive
normalisation / pooled inhibition).

## Background

The trained-substrate decisive run on the 64-concept sparse-distributed
substrate cleared the frozen 0.80 compositional bar at loads 2 and 3
multi-seed (0.842, 0.814) and missed at load 5 (0.756). A
load-ceiling characterisation pinned the ceiling between binding loads
3 and 4 -- about a 30x capacity reduction from the pure FHRR algebra
at the same phasor dimension. Candidate 2 of the NEGATIVE branch was
to test whether the spiking-symbol noise was the cause: replace the
noisy activity-derived symbol with the substrate's clean K-of-N
pattern-derived symbol; does the ceiling rise?

## What was run

`research/findings/raw/vocabulary_scaling_run_pattern_grounded.py`.
A focused byte-reuse extension of the trained-substrate decisive
runner, adversarially reviewed CLEAR before launch on ten
exploit-class checks. Recognition front-end unchanged; FHRR pipeline
unchanged; attractor clean-up + familiarity gate unchanged; deriver
identical (same fixed seed 90909, same dimensionalities); frozen 0.80
bar unchanged; multi-seed {42, 43, 44}; loads {2, 3, 5}. The ONLY
change is the symbol-derivation step: instead of
`mean_centered_activity -> deriver -> phasor`, the path is
`pattern_indicator -> deriver -> phasor`. The pattern indicator is
the binary 0/1 vector over the 2000-neuron shared pool with 1s at
the concept's K=100 stored pattern neurons.

CPU; no GPU, no re-train. The trained activity cache (populated by
the previous decisive run) is reused for recognition.

## Result (pre-registered; multi-seed; frozen 0.80 bar)

```
            integrated mean    composition-only mean    per-seed
L=2         0.038              0.038                    0.040 / 0.043 / 0.033
L=3         0.033              0.033                    0.037 / 0.038 / 0.023
L=5         0.029              0.029                    0.030 / 0.035 / 0.023

recognition (reported separately):
  per-observation mean ~0.77
  temporally-averaged mean 1.000

VERDICT -> NEGATIVE (essentially chance for 32-filler argmax = 0.031)
```

Composition-only equals integrated at every load because temporally-
averaged recognition is perfect. The failure is purely at the
composition stage; recognition is not the bound.

For reference, activity-grounded on the same trained substrate,
multi-seed, same loads: 0.842 / 0.814 / 0.756. Pattern-grounded is
about TWENTY TIMES WORSE than activity-grounded.

## The diagnostic: symbol geometry is the cause

A direct measurement of the symbol input's pairwise cosine similarity
across all 2016 concept pairs at seed 42:

```
activity-grounded (mean-centered consolidated activity):
  mean cosine -0.016  std 0.053  min -0.20  max +0.18
pattern-grounded (binary K-of-N indicator):
  mean cosine +0.050  std 0.021  min  0.00  max +0.15
```

The activity-grounded symbol inputs are near-orthogonal with both
positive AND negative correlations (mean centred at zero). The
pattern-grounded symbol inputs are UNIFORMLY POSITIVE (every pair
has a non-negative cosine, with mean exactly equal to the pattern
overlap fraction K/N = 100/2000 = 0.050, as predicted by the
birthday calculation: two random K-of-N patterns share K^2/N pattern
neurons on average).

The phase-coded compositional algebra requires near-orthogonal
symbols. A bundle of two phasors with cosine +0.05 each carries
double-counted shared-direction signal; the attractor clean-up over
32 such phasors degenerates -- one dominant basin captures most
queries.

To confirm the geometric mechanism, a mean-centered pattern variant
was tested as a diagnostic: subtract the across-concept mean
indicator (~K/N at every position, with small variation) from each
pattern indicator before the deriver. The mean-centered pattern
inputs have mean cosine -0.016, std 0.022 -- the activity-grounded
geometry exactly. Running the same pipeline on these mean-centered
pattern symbols (CPU, multi-seed {42, 43, 44}, loads {2, 3, 5}):
1.000 / 1.000 / 0.999. The geometric mechanism is the precise cause:
mean-centring is the load-bearing operation.

This diagnostic is reported here ONLY to pinpoint the mechanism. It
is NOT a capability claim. It edges further toward the oracle-lookup
shortcut than activity-grounded does -- the mean-centered pattern is
a deterministic function of the substrate's stored code and the
across-concept mean, with no per-observation noise. A capability
claim from mean-centered patterns would need its own pre-registered
arc and would carry an even sharper oracle-adjacency caveat than the
original pattern-grounded test.

## What this means

The compositional algebra has a precise geometric requirement: the
symbol inputs must be approximately orthogonal with both positive
and negative pairwise correlations. The activity-grounded path
satisfies this requirement naturally, via mean-centring (subtractive
normalisation -- a real cortical computation implemented by pooled
inhibition). The spiking-symbol noise sits on TOP of the right
geometry and costs an additional ~0.15-0.25 in accuracy at higher
loads (the 30x capacity-reduction gap to the pure algebra). Removing
the noise without removing the geometry violation (the
pattern-grounded path) collapses composition to chance: the geometry
is necessary; the noise on top of good geometry is the second-order
cost.

The biology-translatable refinement: for a brain that uses the
project's class of phase-coded compositional algebra, the
representation a concept maps to for the composition step CANNOT be
just the stable identity-defining ensemble (the engram cells; the
K-of-N pattern). It must ALSO be common-mode-removed -- the concept-
specific part, after subtracting what is shared across concepts. A
real cortical mechanism delivers exactly this: pooled inhibition
across the cortical pool produces a subtractive-normalised activity
vector that is the concept-specific signature, not the raw firing.
This is what the activity-grounded path does, and it is what the
pattern-grounded path fails to do.

## What this is, and what it is not

This is the pre-registered NEGATIVE for the pattern-grounded
hypothesis, with a clean diagnostic that pinpoints the actual cause
to symbol geometry. It is NOT a claim that mean-centered patterns
are a biologically faithful compositional substrate -- that result
is reported only as a confirmatory diagnostic, with the
oracle-adjacency caveat sharpened. It is NOT a claim that
composition at 64 concepts is solved -- the validated capability
remains the activity-grounded one at loads up to 3 multi-seed, with
the L=3-4 ceiling.

## Next step (the discipline's "iterate following biology")

The diagnostic narrows the path forward. The biology-translatable
hypothesis: if the activity-grounded ceiling is geometry-clean (mean-
centring already in place) and noise-bounded, then reducing the
per-observation noise of the activity-derived symbol -- by averaging
more observations per concept before deriving the symbol -- should
raise the activity-grounded ceiling toward the geometry-clean
mean-centered-pattern variant's ~1.0 ceiling. The activity cache has
M_OBS=16 observations per concept; the current pipeline uses
K_VOCAB=8 for the symbol consolidation. A cheap CPU sweep over
K_VOCAB in {1, 2, 4, 8, 16} on the existing cache directly tests
whether more observations narrows the gap.

This is a cheap, biologically faithful next probe (matches the
project's principle that longer temporal integration reduces spiking
noise, as the earlier recognition-bound probe showed for recognition
itself). It is one further test within the activity-grounded
biologization line, no oracle-adjacency added. A new pre-registered
step.

(Broader horizon, surfaced for the owner, not auto-launched: the
owner's standing conversational-path directives -- SPEAR, theta-gamma
mode-unification, generative replay -- and the integrated closed loop
are the larger arcs.)

## Honest scope

A clean multi-seed decisive run with a clear pre-registered NEGATIVE
verdict. The frozen 0.80 bar was not moved. The runner was
adversarially reviewed CLEAR before launch on ten exploit-class
checks. The mean-centered pattern diagnostic is reported as a
mechanism-pinpointing measurement only, not a capability claim, with
the oracle-adjacency caveat sharpened (deterministic function of
stored patterns, no per-observation noise). The completed twice-
reviewed 16-concept activity-grounded biologization arc (multi-seed
0.98) stands. The trained-substrate 64-concept BOUNDARY result
(multi-seed 0.84 / 0.81 at loads 2-3, ceiling between 3-4) stands.
No protected, frozen, or moat module modified; no automatic
differentiation; no-confab moat 7/7 green.

## Files / evidence

- Runner: `research/findings/raw/vocabulary_scaling_run_pattern_grounded.py`
- Helper: `research/findings/raw/vocabulary_scaling_pattern_helpers.py`
- Soundness tests (10/10 PASS):
  `tests/test_pattern_grounded_pin.py`,
  `tests/test_vocabulary_scaling_pattern_helpers.py`,
  `tests/test_vocabulary_scaling_pattern_grounded.py`
- Result: `research/findings/raw/vocabulary_scaling_run_pattern_grounded.json`
- The trained-substrate BOUNDARY this builds on:
  `research/findings/2026-05-22-vocabulary-scaling-trained-substrate-BELOW-BAR-with-loads-2-3-PASS-and-load-5-ceiling.md`
- The load-ceiling map (activity-grounded reference curve):
  `research/findings/2026-05-22-vocabulary-scaling-load-ceiling-map-ceiling-sits-between-loads-3-and-4.md`
- Design + plan:
  `docs/plans/2026-05-22-pattern-grounded-symbol-design.md`,
  `docs/plans/2026-05-22-pattern-grounded-symbol-implementation.md`
- Adversarial review verdict: VERDICT CLEAR on ten exploit-class
  checks (no answer leak, recognition load-bearing, deriver
  identical, frozen bar immovable, byte-unchanged reuse, no autograd,
  pipeline body identical, pattern store is substrate-stored, tests
  pin the load-bearing properties, legitimate biologized refinement
  not oracle shortcut).
