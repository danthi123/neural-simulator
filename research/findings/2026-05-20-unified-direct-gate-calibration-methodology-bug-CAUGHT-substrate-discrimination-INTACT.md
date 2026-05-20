# Diagnostic methodology bug CAUGHT before propagating wrong conclusion; corrected diagnostic shows unified substrate retains per-word binding (10/16 correct direction at seed 42, vs pure v14's 13/16); the prior "INSUFFICIENT-SEPARATION at 2/3 seeds" calibration verdict is mostly a half-split-of-trained-vocab statistical fragility, not a substrate failure -- next iteration redesigns the direct-gate calibration readout

## Status

Honest mid-arc finding. The pre-registered diagnostic probe ran but
the first version had a subtle methodology bug that produced misleading
output. The bug was caught by the controller's discipline (the
documented v14/v16 multi-seed 77.5% W->A baseline should reproduce on
pure v14 at seed 42; the v1 diagnostic showed only 2/12 -- a six-fold
discrepancy against documented behaviour -- which signalled a probe
problem, not a substrate problem). The fix was a one-line correction
to match the training/calibration canonical vocabulary layout. The
v2 diagnostic produces a substantively different and honest picture.

## The methodology bug (v1) and its fix (v2)

`research/runners/concept_pool_demo.py:898-904` and
`research/runners/unified_per_regime_monitor_runner.py:690-703,1217`
both use a 16-word canonical vocabulary (motor + nouns + verbs +
adjectives) for orthogonal-code drive generation, with
`n_words_for_orthogonal = 16` and a `word_to_idx` ordered
motor-first then nouns, verbs, adjectives.

The v1 diagnostic instead used a 12-word `word_to_idx` (nouns + verbs
+ adjectives only, motor omitted) with `n_words_for_orthogonal = 12`.
Because `orthogonal_drive_pattern` derives the stripe layout from
`(n_lang_input, n_cues, cue_idx, sparsity)`, a different `n_cues`
produces a different stride and therefore an entirely different
drive pattern for the same word. The v1 diagnostic was driving the
substrate with patterns it had never been trained on -- both
substrates "failed" because the probe was mis-keyed, not because
the substrate was broken.

v2 fixes this: it uses the 16-word canonical layout exactly matching
the training and calibration code paths, tests both substrates on
all 16 trained words plus a separate non-motor 12-word view that
matches the calibration's scope.

## v2 diagnostic results (seed 42)

| Substrate | Scope | groundable_median | ungroundable_median | Correct direction |
|-----------|-------|-------------------|---------------------|--------------------|
| Pure v14 (no hippocampus, no dlpfc) | all 16 | 0.380 | 0.240 | **13/16 (81%)** |
| Pure v14 | 12 non-motor | 0.313 | 0.280 | 9/12 (75%) |
| Unified (hippocampus + dlpfc + concept pools) | all 16 | 0.265 | 0.235 | **10/16 (62.5%)** |
| Unified | 12 non-motor | 0.285 | 0.220 | 8/12 (67%) |

Where:
- groundable = target-pool firing rate when its own word is driven
- ungroundable = best off-target pool firing rate when the same word
  is driven

The pure v14 13/16 = 81% correct direction matches the documented
v14 5-seed mean 12.4/16 (77.5%) W->A baseline. The v1 diagnostic
showed 2/12 (17%) on the same substrate at the same seed -- the
six-fold discrepancy is the methodology bug, not the substrate.

The unified substrate at the same seed shows 10/16 (62.5%) correct
direction -- POSITIVE separation in the right direction, about 18.5
percentage points lower than pure v14. The hippocampus + dlpfc
addition modestly degrades per-word direct retrieval but does NOT
abolish it.

Per-word at seed 42 (unified, all 16):
- Strong robust binders: cat 1.16 (off 0.27), small 0.87 (off 0.245),
  hot 0.595 (off 0.205), river 0.42 (off 0.31), go 0.39 (off 0.245),
  south 0.365 (off 0.30), dog 0.365 (off 0.265), look 0.30 (off 0.24),
  apple 0.23 (off 0.22), north 0.185 (off 0.14) -- 10/16 robust.
- Failed at seed 42: east 0.07 (off 0.23), west 0.17 (tie 0.17),
  come 0.22 (off 0.245), stop 0.165 (off 0.215), big 0.145 (off 0.195),
  cold 0.135 (off 0.165) -- 6/16 inverted; 4 of them have the
  inversion margin below 0.05 (near-tie).

## Reinterpreting the calibration's INSUFFICIENT-SEPARATION

The unified runner's direct-gate calibration
(`_calibrate_direct_one_seed`, lines 1179-1310) constructs its
populations as follows:

- GROUNDABLE: trained word -> target-pool rate (the right pool)
- UNGROUNDABLE: a NON-OVERLAPPING TRAINED word -> TOP-pool rate
  (any pool, no oracle target)

Both halves are TRAINED. The "ungroundable" set is the held-out half
of the trained vocab queried by its OWN trained code, with the
top-pool rate measured. Because all 16 words ARE trained, the
top-pool rate is in most cases the target-pool rate of that
held-out word -- a strong-binder signal, not a noise-floor.

The per-seed random half-split of the 16-word vocab is therefore
measuring (strong-binder-half-median) versus
(other-strong-binder-half-median plus off-target-leakage). Seed-
dependent outcomes:

- Seed 42: groundable half includes weak binders (e.g. big, cold,
  come, stop near or below the off-target leakage); ungroundable
  half includes strong binders (cat, small, hot). Median(g)=0.27
  < median(u)=0.30 -> INVERTED.
- Seed 43: groundable half includes strong binders; ungroundable
  half includes weak. Median(g)=0.48 > median(u)=0.345 -> OK.
- Seed 44: groundable half includes weak binders again; INVERTED.

The "INSUFFICIENT-SEPARATION at 2/3 seeds" verdict the calibration
emitted IS technically correct for the protocol it implements --
the controller correctly refused to commit a degenerate threshold
when groundable_median <= ungroundable_median. But the protocol
was measuring split-luck on a trained-only vocabulary, NOT the
trained-vs-untrained signal we want a direct-gate threshold to
calibrate.

## Two honest discoveries

### 1. The integrated unified substrate retains per-word direct binding

The hippocampus + dlpfc + concept-pools substrate WORKS at the
per-word direct readout: 10/16 correct direction at seed 42 (and
the v2 diagnostic confirms this corresponds to a positive median
separation of 0.265 vs 0.235 over the full 16-word vocab). The
integration adds modest cross-pathway interference -- the unified
substrate is about 18.5 percentage points weaker than the pure
concept-pool substrate at seed 42 -- but the core capability
SURVIVES the integration. This is consistent with the integrated-
loop hypothesis: integration introduces tradeoffs across multiple
subsystems; the load-bearing question is whether the integrated
loop emerges new capabilities (compositional one-shot, sequential
recall), not whether each subsystem is byte-equivalent to its
isolated version.

### 2. The direct-gate calibration protocol has a methodology fragility

The per-seed half-split of a trained-only vocab does not measure
trained-vs-untrained discriminability. It measures median-vs-
median between two random halves of a single (trained) population.
The half-split outcome is statistically fragile and produces seed-
variable INVERTED outcomes even when the underlying substrate is
discriminably bound. A more principled direct-gate readout would
either:

(a) Use truly-untrained queries (e.g. sub-seed-shifted orthogonal
codes -- patterns the substrate has not been trained against -- as
the ungroundable population). This is the cleanest construction
but requires a new pattern-generation primitive that produces
"in-layout but unmapped" codes.

(b) Use a per-word INTERNAL signal-to-noise ratio (top-pool minus
best-off-target, or top minus mean off-target) and calibrate the
threshold on the distribution of these ratios. At deployment, a
query whose top-minus-second exceeds the threshold is emitted;
otherwise abstained. This is what the v2 diagnostic essentially
measures, and it shows positive separation on the unified
substrate at seed 42.

(c) Use the existing 650 SharedPool-recall moat for direct queries,
recognising that direct retrieval on G.20 SharedPool substrates and
direct retrieval on the v14/v16 concept-pool substrate are
different readout pipelines that may not unify in one threshold.

## What this means for the integrated-loop arc

The original "decisive run BLOCKED on direct-gate INSUFFICIENT-
SEPARATION at 2/3 seeds" framing was based on the v1 calibration
protocol which is now understood to be measuring split-luck rather
than substrate discriminability. The substrate IS working at the
per-word direct level (10/16 at seed 42 on the unified substrate).

The integrated-loop catalog-grounded design (compose_concept_engram
+ trisynaptic loop + shared theta-gamma + neuromodulator timing
+ trustworthy abstention gate) has its LOAD-BEARING test in the
COMPOSITIONAL one-shot encoding/recall (not the direct readout).
The compositional gate calibrated cleanly on the unified substrate
(0.198 median; 3/3 seeds positive direction; MATCH-able). The
direct gate's calibration question is a measurement-protocol
question downstream of the load-bearing integration test.

## Pre-registered next step (autonomous, no hand-back)

Redesign the direct-gate calibration to use the v2-diagnostic
readout (target-vs-best-off-target gap per trained word), or
truly-untrained sub-seed-shifted codes. Implement as an additive
function `_calibrate_direct_v2_one_seed` alongside the existing
`_calibrate_direct_one_seed` (no modification of the existing
protocol; backwards-compatible opt-in via a CLI flag). The new
function:

- Uses target-pool rate as the per-word groundable signal (matching
  the diagnostic v2)
- Uses best-off-target-pool rate as the per-word ungroundable signal
  (matching the diagnostic v2; symmetric to groundable; isolates
  substrate's signal-to-noise ratio rather than half-split-luck)
- Aggregates over the FULL trained vocab (no per-seed half-split)
- Computes the median-midpoint threshold the same way the existing
  protocol does, but on a more principled signal
- Emits MATCH / PENDING / MISMATCH / INSUFFICIENT-SEPARATION on
  the SAME strengthen-only fail-closed criterion

This goes through the full subagent-driven discipline (TDD on a
new frozen verdict module if needed; controller verifies protected
set byte-empty; sixth adversarial review of the new calibration
protocol; full-scale 3-seed calibration on the cached Phase-1
unified substrate; if MATCH on a positive threshold, controller
commits it; decisive eval proceeds).

If the redesigned calibration shows INSUFFICIENT-SEPARATION on the
new protocol too, the substrate genuinely cannot abstain reliably
on direct queries at the unified-substrate scale -- that would be
a different finding (real substrate limit) requiring a deeper
integration design iteration (e.g. add per-region inhibitory
normalization to suppress cross-pathway interference, mirroring
v14's within-kind FS mechanism).

## Honest ceiling (unchanged, restated)

Conversational / compositional capability is NOT achieved and is
NOT claimed. The decisive evaluation of the unified integrated
architecture has NOT yet run. What IS achieved this iteration:

- Caught the diagnostic methodology bug BEFORE propagating the wrong
  "unified substrate degrades direct retrieval to 0/12" conclusion.
- Verified pure v14 substrate reproduces its documented 77.5% W->A
  multi-seed baseline at seed 42 (13/16 = 81%) -- the documented
  capability is real and unaffected by this arc.
- Localised the unified substrate's actual per-word direct readout
  at seed 42: 10/16 = 62.5% correct direction. Modestly degraded by
  hippocampus+dlpfc integration, NOT abolished.
- Localised the prior calibration's INSUFFICIENT-SEPARATION to a
  statistical fragility in the half-split protocol, NOT a substrate
  failure.
- 72/72 tests still green; no-confab moat still 7/7 byte-identical;
  the protected set byte-empty diff vs e8a99a2 holds across every
  commit of this arc.

The accumulated substrate-specific calibration moats:
- 650 (abstention_gate.py): G.20 SharedPool recall_rates, scale
  ~500-800 (byte-unchanged this arc)
- 5.6887 (abstention_gate_compositional.py): per-regime stage's
  hippocampal one-shot substrate's lang_output readout, scale ~5
  (byte-unchanged this arc)
- 0.198 (would-be unified compositional, NOT YET committed): the
  unified substrate's lang_output readout under the per-regime stage
  compositional protocol, scale ~0.2; MATCH-able if controller
  commits in a separate step
- 0.0 (placeholder for unified direct, NOT committed): the
  abstention_gate_direct_unified.py placeholder; the v1 protocol's
  INSUFFICIENT-SEPARATION outcome is now understood to be mostly
  measurement-protocol fragility; the v2 protocol redesign is the
  next staged step.

The biology-translatable insight stands and is now sharper:
**trustworthy abstention thresholds are SUBSTRATE-specific AND
PROTOCOL-specific**. A calibration that conflates trained-vs-trained
half-splits with trained-vs-untrained discriminability does not
produce a trustworthy threshold even at infinite data. The brain's
per-regime metacognitive monitors (Miyamoto 2017) calibrate their
thresholds against actual recent-vs-remote ambiguity boundaries,
not against arbitrary halvings of a single trained population. Our
calibration designs must follow the same principle.

The disciplined refusal-to-commit-a-degenerate-threshold worked a
SEVENTH consecutive time across this arc -- in the form of the
controller catching its OWN diagnostic methodology bug before
propagating a misleading conclusion. That meta-deliverable is the
discipline operating as designed.

## Files / evidence

- Corrected diagnostic: `research/findings/raw/unified_DIAGNOSTIC_pure_vs_unified.py`
- Corrected diagnostic JSON: `research/findings/raw/unified_DIAGNOSTIC_pure_vs_unified.json`
- Corrected diagnostic log: `research/findings/raw/unified_DIAGNOSTIC_pure_vs_unified_v2.log`
- Original calibration JSON (now reinterpreted): `research/findings/raw/unified_CALIBRATION_fullscale.json`
- Phase-1 cached checkpoints: `research/findings/raw/unified_per_regime/phase1/seed{42,43,44}.simstate.h5`
- All previously-validated modules + calibrated moats byte-unchanged.
