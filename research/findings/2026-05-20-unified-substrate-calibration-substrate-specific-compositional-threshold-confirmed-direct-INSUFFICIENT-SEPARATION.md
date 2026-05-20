# Unified-substrate calibration confirms substrate-specific compositional threshold (5.69 doesn't transfer; new substrate calibrates to 0.198) AND reveals direct-retrieval INSUFFICIENT-SEPARATION at 2/3 seeds on the hippocampus-equipped substrate -- a multi-seed direct-retrieval variability finding the decisive run cannot proceed past

## Status

Honest mid-arc finding from the controller-only calibration step
of the unified per-regime-monitor + per-regime-encoding stage,
propagated without spin. The fifth adversarial review BLOCKED the
prior runner on two critical defects; both were fixed (substrate
rebuild on `build_biological_brain_regions` so engram region_filter
has real hippocampal regions; new substrate-specific direct gate
`abstention_gate_direct_unified.py` placeholder added alongside the
byte-unchanged 650 moat). The disciplined calibration step on the
unified substrate has now run at full biological scale (3 seeds;
Phase-1 multi-event direct training; calibration of BOTH gates on
held-out groundable vs ungroundable queries). The runner emits
PENDING/MISMATCH/INSUFFICIENT-SEPARATION status correctly; the
controller refuses to commit degenerate threshold values; the
decisive evaluation cannot proceed until the calibration findings
are interpreted.

## Calibration result (full biological scale; seeds 42/43/44; held-out per gate)

### Compositional gate (re-calibrated on the unified substrate)

| seed | groundable_median | ungroundable_median | calibrated_threshold |
|------|-------------------|---------------------|---------------------|
| 42   | 0.250             | 0.186               | 0.218               |
| 43   | 0.265             | 0.147               | 0.206               |
| 44   | 0.201             | 0.137               | 0.169               |
| **median** | -           | -                   | **0.198**           |

- Status: **MISMATCH** (committed `COMPOSITIONAL_THRESHOLD = 5.6887`
  from per-regime stage; calibrated on unified substrate is 0.198).
- Direction at every seed: groundable > ungroundable. Separation
  consistent.

### Direct gate (new substrate-specific calibration)

| seed | groundable_median | ungroundable_median | calibrated_threshold |
|------|-------------------|---------------------|---------------------|
| 42   | 0.270             | 0.303               | 0.286               |
| 43   | 0.480             | 0.345               | 0.413               |
| 44   | 0.333             | 0.410               | 0.371               |
| **median** | -           | -                   | **0.357**           |

- Status: **INSUFFICIENT-SEPARATION** (the reviewer's strengthen-only
  fix correctly fires: ANY per-seed cell with groundable_median <=
  ungroundable_median triggers the status).
- Seed 43: groundable (0.48) > ungroundable (0.345) -- correct
  direction, margin ~0.135.
- Seed 42: groundable (0.27) < ungroundable (0.303) -- INVERTED.
- Seed 44: groundable (0.333) < ungroundable (0.410) -- INVERTED.

## The two findings

### Finding 1: Substrate-specific compositional threshold EMPIRICALLY CONFIRMED

The compositional readout (`pat[active].sum() / n_active` at
`lang_output_pattern_during_stim` via the validated path) produces
**confidence at scale ~0.20 on the unified substrate (with hippocampus
+ dlpfc + concept pools)** versus **scale ~5.69 on the per-regime
stage's substrate (hippocampus + dlpfc, no concept pools)**. The same
quantity, the same readout function, calibrated on different
substrates yields ~28x different thresholds.

This is the adversarial reviewer's substrate-specificity insight
empirically confirmed at full biological scale. Trustworthy
abstention thresholds are SUBSTRATE-specific, not regime-specific.
The 5.6887 committed in `abstention_gate_compositional.py` is
HISTORICAL CALIBRATION for the per-regime stage's substrate and
does NOT transfer to the unified substrate. The unified substrate's
correct compositional threshold would be ~0.198 if it were committed
(but the runner emits MISMATCH and refuses to silently use 5.6887).

### Finding 2: Direct retrieval shows INSUFFICIENT-SEPARATION at 2/3 seeds on the unified substrate

`measure_pool_firing` on the Phase-1-trained unified substrate
produces:
- Consistent separation only at seed 43 (groundable 0.48 vs
  ungroundable 0.345).
- INVERTED separation at seeds 42 and 44 (groundable below
  ungroundable). For these seeds, the "trained" target pool produces
  LOWER firing than the "untrained" control pool when queried.

This is unexpected against the project's documented v14 multi-seed
direct retrieval (mean W->A 11.8/16 = 74% at v16; A->W 100%
unanimous; BIDIRECTIONAL average ~88.75% multi-seed). Possible
explanations the project should investigate:

(a) The v14/v16 88.75% multi-seed measurement uses the project's
`concept_pool_demo.build_concept_bridge` substrate which has concept
pools but NO hippocampus + dlpfc. The unified substrate adds
hippocampal regions + the prefrontal `dlpfc_verb` region; these
additions may DEGRADE direct retrieval relative to the pure
concept-pool substrate (e.g. via cross-pathway competition or
homeostasis-mediated suppression of trained pools by the
untrained-but-active hippocampal/dlpfc pathways).

(b) The v14/v16 88.75% uses a different measurement / decoding
methodology than `measure_pool_firing` -- e.g. A->W (driving the
pool and reading lang_output) rather than W->A (driving lang_input
and reading the pool, which is what `measure_pool_firing` does).
The 100% A->W documented in v14/v16 may not translate to W->A
consistency at the measurement granularity the calibration uses.

(c) The held-out calibration query design (8 groundable + 8
ungroundable per seed; median-midpoint method) is statistically
noisy enough that multi-seed direct retrieval can appear
INVERTED on a substantial fraction of seeds even when the
underlying signal exists. The per-regime stage's compositional
calibration used different query counts and did not hit this
status; the direct-gate calibration's noise sensitivity may be
higher.

## What this means for the unified architecture

The unified architecture's decisive evaluation CANNOT proceed in
its current form. Two gating issues:

1. The compositional gate's MISMATCH means the runner would either
   (a) use the wrong 5.6887 threshold (substrate-mismatched; would
   over-abstain at scale ~0.2) or (b) be re-committed to 0.198 in
   a separate controller commit. The disciplined fix is (b) IF
   we are confident the unified substrate is the right basis for
   future stages.

2. The direct gate's INSUFFICIENT-SEPARATION means there is no
   sensible threshold to commit. Committing any value would mean
   abstaining on some "trained" queries while accepting some
   "untrained" ones -- exactly the false-PASS / false-FAIL pattern
   the trustworthy-abstention discipline forbids. The runner
   correctly BLOCKS this commit by emitting INSUFFICIENT-SEPARATION.

The honest reading: the architectural hypothesis the unified stage
was designed to test (Phase-1 multi-event training + compositional
one-shot encoding + per-regime routing through both gates) is
PARTIALLY validated (the compositional calibration is now
substrate-specific and well-separated at every seed) but BLOCKED
on the direct gate's substrate-specific calibration. Before the
decisive evaluation can run, the project must understand WHY direct
retrieval on the unified substrate doesn't separate reliably across
seeds -- either via a diagnostic probe (compare unified vs pure
v14/v16 substrate's direct retrieval) or via a measurement-
methodology recheck (W->A vs A->W; statistical-power audit of the
held-out query design).

## The deeper biology-translatable insight (now empirically validated multiple times)

Trustworthy abstention thresholds are SUBSTRATE-specific. They
encode the calibrated confidence distribution OF A PARTICULAR
SUBSTRATE'S READOUT, not a universal property of "this regime".
The project's accumulated calibrated values:
- 650 (abstention_gate.py): G.20 SharedPool recall_rates, scale
  ~500-800.
- 5.6887 (abstention_gate_compositional.py): per-regime stage's
  hippocampal one-shot substrate's lang_output readout, scale ~5.
- 0.198 (would-be unified compositional, NOT YET committed): the
  unified substrate's lang_output readout, scale ~0.2.
- 0.357 (would-be unified direct, NOT committed -- INSUFFICIENT-
  SEPARATION blocks).

These do not interconvert. The brain's per-regime metacognitive
monitors (Miyamoto 2017) likely calibrate their thresholds in-situ
on the specific substrate + readout they monitor. Future stages
that change the substrate or the readout MUST recalibrate before
declaring decisive evaluation.

## Pre-registered next step (autonomous, no hand-back)

Diagnostic probe: compare `measure_pool_firing`'s
groundable-vs-ungroundable separation on (a) the unified substrate
(this stage's, with hippocampus + dlpfc) vs (b) the pure
v14/v16 substrate (`concept_pool_demo.build_concept_bridge`,
concept pools only). The probe answers: does the hippocampus +
dlpfc addition DEGRADE direct retrieval at biological scale, or
is the INSUFFICIENT-SEPARATION an artifact of the calibration's
query design? The probe reuses the Phase-1 cached checkpoints
(no expensive retraining); it just loads each checkpoint and
runs `measure_pool_firing` on a larger, statistically-better
query set (e.g. 16 trained + 16 untrained per seed instead of
8 + 8) and reports the multi-seed separation distribution.

If the probe finds:
- Hippocampus+dlpfc degrades direct retrieval -> the unified
  architecture has a substrate-design issue; the next iteration
  must either remove the hippocampus/dlpfc components for direct
  retrieval (a partition that defeats the point of "unified") OR
  find a way to suppress the cross-pathway interference (a real
  neuroscience-grounded refinement).
- Calibration query design is too noisy -> design a more
  statistically-powerful calibration query set (more queries; the
  median-midpoint method then has more signal).
- Some combination -> the next investigation is the next staged
  step.

Either way, this is a substantive scientific finding that deserves
its own propagation cycle BEFORE proceeding to the decisive
evaluation.

## Honest ceiling (unchanged, restated)

Conversational / compositional capability is NOT achieved and is
NOT claimed. The decisive evaluation of the unified architecture
has NOT yet run. What IS achieved this iteration: empirical
confirmation of the adversarial reviewer's substrate-specific-
threshold insight; localization of the unified substrate's direct-
retrieval issue (INSUFFICIENT-SEPARATION at 2/3 seeds; multi-seed
variability finding that the project may need to investigate at
the substrate level). No fixed bar moved; no protected/frozen
module touched; the existing 650 + 5.6887 calibrated moats stay
byte-unchanged; the new `abstention_gate_direct_unified.py` placeholder
0.0 stays unmodified (the runner correctly refuses to commit
INSUFFICIENT-SEPARATION). 72/72 tests still green across all
relevant suites. The discipline working a SIXTH consecutive time
(the runner's INSUFFICIENT-SEPARATION status correctly blocks a
degenerate commit) is itself a meta-deliverable.

## Files / evidence

- Durable calibration JSON: `research/findings/raw/unified_CALIBRATION_fullscale.json`
- Durable calibration log: `research/findings/raw/unified_CALIBRATION_fullscale.log`
- Phase-1 cached checkpoints (one per seed):
  `research/findings/raw/unified_per_regime/phase1/seed42.simstate.h5`,
  `.../seed43.simstate.h5`, `.../seed44.simstate.h5`
- All previously-validated modules + calibrated moats byte-unchanged.
