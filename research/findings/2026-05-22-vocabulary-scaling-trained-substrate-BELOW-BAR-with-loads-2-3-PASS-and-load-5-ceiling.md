# Vocabulary-scaling trained-substrate re-run: BELOW the strict bar at load 5; the substrate fix worked, composition clears 0.80 at loads 2-3 and ceilings at load 5 on a 64-concept vocabulary

## Status

The corrected, multi-seed, GPU re-run of the 64-concept
vocabulary-scaling test on a properly-trained sparse-distributed
substrate. Per the pre-registered bar (PASS = integrated multi-seed
mean >= 0.80 at every load in {2, 3, 5}), the verdict is BELOW BAR --
load 5 misses by 0.044. But the result is a substantive, refined
improvement, not a repeat of the original NEGATIVE. The corrective
intervention worked exactly as predicted (the substrate is no longer
near-silent, recognition is perfect), and the biologized pipeline
cleanly clears the bar at loads 2 and 3; the failure mode is a clean
load ceiling at load 5 on a 64-concept vocabulary. This is a refined,
biology-translatable load-capacity finding, not a substrate or
recognition failure.

## Background

The original 64-concept vocabulary-scaling decisive run was a NEGATIVE
(integrated multi-seed ~0.11 at all loads). A capture-drive probe arc
diagnosed the cause: the run had captured activity from a freshly-built,
UNTRAINED sparse-distributed substrate; the language-input pathway was
random and non-selective, so a concept's drive did not preferentially
evoke its pattern, and the strong fast-spiking winner-take-all loop
collapsed the result to near-silence (about 0.008 of the pool active vs
about 0.075 on the validated 16-concept substrate the pipeline was
proven on). The probes retracted a flawed scale-artifact intermediate
verdict, ruled out the two cheap fixes (a stronger teacher current was
oracle-adjacent; the topographic prior alone was insufficient), and
identified the live fix as the original NEGATIVE's candidate 1: capture
from a fully TRAINED substrate (the prior plus the validated
spike-timing encoding).

This re-run is the corrected test on a trained substrate.

## What was run

`research/findings/raw/vocabulary_scaling_run_trained.py`. A focused
byte-reuse extension of the adversarially-reviewed decisive runner: it
inserts the validated encoding (topographic prior plus 400 interleaved
per-concept spike-timing training events) before the activity capture,
then runs the same biologized grounded-composition pipeline (imported
unchanged) against the same frozen 0.80 compositional bar. Seeds
42/43/44; the full 18,684-neuron, 10.4M-synapse substrate. Soundness
tests 2/2 (pinning the validated-encoding constants and that the
training stage genuinely reshapes the substrate); a dedicated
adversarial reviewer ran all ten exploit-class checks and returned
CLEAR before launch. About 58 minutes per seed on the RTX 3090.

## Result (pre-registered; multi-seed; frozen 0.80 bar)

```
            integrated mean    composition-only mean    per-seed
L=2         0.842              0.842                    0.875 / 0.850 / 0.800
L=3         0.814              0.814                    0.860 / 0.802 / 0.780
L=5         0.756              0.756                    0.769 / 0.803 / 0.696

recognition (reported separately):
  per-observation mean 0.774
  temporally-averaged mean 1.000

VERDICT (per the pre-registered bar) -> BELOW BAR at L=5 (0.756 < 0.80)
```

Composition-only equals integrated at every load because temporally-
averaged recognition is perfect (1.000): every fact's words are
correctly recognised, so every fact contributes to composition-only.

The mandatory anti-cheat smell-test recomputed the per-load means
directly from the per-seed records (matches the recorded aggregate
exactly), recomputed the captured pool-activity density from the
trained activity cache (0.097, 0.107, 0.104 across seeds -- decisively
above the untrained 0.0077 the original NEGATIVE recorded, and even
above the validated substrate's 0.075), re-derived the verdict against
the frozen bar (matches), and ran consistency checks (14 of 14 pass).
The recorded result is internally consistent and recomputes cleanly.

## What the corrective intervention fixed, and what it did not

The training stage worked exactly as predicted by the probe arc.
Captured pool density rose from the untrained 0.0077 to 0.097-0.107
(decisively above the validated substrate's 0.075 -- the substrate is
no longer near-silent, in fact slightly denser than the substrate the
pipeline was proven on). Recognition rose from the untrained 0.84
temporally-averaged to a clean 1.000 -- the trained substrate's
concept-specific activity is genuinely groundable.

The biologized composition pipeline then clears the 0.80 bar at
compositional loads 2 and 3 (means 0.842 and 0.814) -- the first time
the activity-grounded compositional capability has been demonstrated at
a 64-concept vocabulary on the project's sparse-distributed substrate.

At load 5 (binding five facts into one composite) the multi-seed mean
is 0.756, missing the bar by 0.044. Per-seed values (0.769, 0.803,
0.696) span a range of 0.107 -- borderline, seed-variable. This is a
LOAD CEILING, not a substrate or recognition failure. The composition-
only accuracy equals integrated at every load, so the bound is the
composition itself, not the recognition front-end.

## What this is, and what it is not

This is BELOW the strict pre-registered bar (the bar requires all
loads). It is NOT a repeat of the original NEGATIVE: that one was
near-chance everywhere because the substrate was near-silent; this one
clears the bar at loads 2-3 and ceilings at 5. The failure mode
contradicts the routing's a-priori premise (that a NEGATIVE here would
mean the substrate was still too sparse): the trained substrate is
denser than the validated benchmark and recognition is perfect; the
ceiling is in the composition itself at higher binding loads. That is
a sharper, finer-grained finding.

It is NOT a claim that the biologized grounded compositional capability
scales to 64 concepts in general -- the strict bar required clearing
all loads, and load 5 missed. The honest claim: on a properly-trained
64-concept sparse-distributed substrate the biologized pipeline
demonstrates compositional capability at small loads (2-3) at
multi-seed multi-seed accuracy above 0.80, with a clear load ceiling
emerging at 5.

For context, the pure FHRR algebra's capacity curve (numpy probe,
phasor dim 512) is well above load 96 at this dimension. The spiking-
grounded pipeline ceilings much earlier than the algebra -- the
ceiling appears to be the spiking-implementation noise floor of the
grounded symbols, not an algebraic limit.

## Next step (per the pre-registered routing)

The pre-registered NEGATIVE-branch routing is candidate 2: grounding
the symbol in the sparse K-of-N PATTERN itself (the concept's clean
code on the trained substrate), weighed honestly against whether that
is still substrate-grounded or closer to an oracle lookup. The actual
failure mode (load ceiling from spiking-symbol noise, not from a
too-sparse substrate) sharpens the motivation: replacing the noisy
activity-derived symbol with the clean K-of-N pattern would test
exactly whether the symbol noise floor is the limit.

A cheap pre-built step before candidate 2: a load-ceiling
characterisation that re-runs the pipeline on the existing trained
activity cache at loads {2, 3, 4, 5, 6, 7} (no new GPU run -- the
activity is cached) to map the ceiling precisely.

Either is a new pre-registered step. This BELOW-BAR result is
propagated first.

## Honest scope

A multi-seed decisive run with a clear pre-registered verdict and a
fully passing smell-test. The 0.80 bar was not moved; the run was
GPU-faithful on the validated full-scale substrate; the runner was
adversarially reviewed CLEAR before launch and the recording was
recomputed end-to-end after. No protected, frozen, or moat module was
modified; the substrate builder, the validated encoding, and the
biologized pipeline were reused by import. No automatic
differentiation. The completed twice-reviewed 16-concept
FHRR-biologization arc (multi-seed 0.98) stands, unaffected.

## Files / evidence

- Runner: `research/findings/raw/vocabulary_scaling_run_trained.py`
- Soundness tests: `tests/test_vocabulary_scaling_trained.py`
- Result: `research/findings/raw/vocabulary_scaling_run_trained_full.json`
- Activity cache:
  `research/findings/raw/vocabulary_scaling_trained_cache/trained_full_seed{42,43,44}.npz`
- Run log: `research/findings/raw/vocabulary_scaling_run_trained_full.log`
- Smell-test tool: `research/findings/raw/vocabulary_scaling_smell_test.py`
- The diagnostic arc that motivated this corrected run:
  `research/findings/2026-05-22-vocabulary-scaling-capture-drive-probe-near-silence-diagnosed-to-untrained-substrate.md`
- The original NEGATIVE this corrected run supersedes the cause of:
  `research/findings/2026-05-22-vocabulary-scaling-64concept-NEGATIVE-G20-sparse-activity-too-sparse-for-the-activity-grounded-pipeline.md`
- Design + plan:
  `docs/plans/2026-05-22-vocabulary-scaling-design.md`,
  `docs/plans/2026-05-22-vocabulary-scaling-implementation.md`
