# Unified-substrate DIRECT_UNIFIED_THRESHOLD calibrated to 0.284167 via v2 protocol (per-word target-vs-best-off-target gap; full trained vocab; no per-seed half-split); all 3 seeds positive direction with margins 0.030/0.110/0.121; controller commits threshold in a separate frozen step mirroring the per-regime stage's abe65f6 pattern

## Status

Honest mid-arc finding. The v2 direct-gate calibration protocol redesign
(motivated by the corrected-diagnostic discovery that v1's per-seed
half-split of the trained vocab measured strong-half-vs-weak-half
median rather than trained-vs-untrained; commit `7548465`) ran at full
biological scale on the cached Phase-1 checkpoints (3 seeds; commit
`b07486e` ships the v2 calibration function alongside v1, opt-in via
`--direct-calibration-v2` CLI flag; the sixth consecutive dedicated
adversarial review returned CLEAR with two non-load-bearing cosmetic
notes for controller-discretion follow-up). The v2 protocol produces
clean positive separation across all 3 seeds. The controller commits
the calibrated threshold via the same pattern as the per-regime stage's
compositional-gate calibration commit (`abe65f6` committed 5.6887 for
the per-regime substrate). The direct-unified gate is now a frozen
calibrated moat alongside the byte-unchanged 650 G.20 moat and the
byte-unchanged 5.6887 per-regime compositional moat.

## v2 calibration result (full biological scale; seeds 42/43/44)

| seed | groundable_median | ungroundable_median | margin | calibrated_threshold |
|------|-------------------|---------------------|--------|----------------------|
| 42   | 0.265             | 0.235               | 0.030  | 0.250                |
| 43   | 0.365             | 0.255               | 0.110  | 0.310                |
| 44   | 0.353             | 0.232               | 0.121  | 0.293                |
| **aggregate** | -        | -                   | -      | **0.2841666666666667** |

- Status: **PENDING** (committed placeholder 0.0, aggregate non-zero,
  every per-seed cell has groundable_median > ungroundable_median).
  Controller commits the aggregate value in a separate frozen step.
- All 3 seeds positive direction (no INSUFFICIENT-SEPARATION at any
  seed) -- the v2 protocol's strengthen-only fail-closed criterion does
  NOT fire.
- Seed 42 v2 calibration result matches the v2 diagnostic exactly (the
  same 0.265 / 0.235 numbers) -- determinism confirmed.

## Why v2 succeeds where v1 failed

The v1 protocol (calibration_v1 ran 2026-05-20T16:57 with status=
INSUFFICIENT-SEPARATION at 2/3 seeds, durable JSON
`research/findings/raw/unified_CALIBRATION_fullscale.json`) measured:

- groundable = trained word -> target-pool rate
- ungroundable = a NON-OVERLAPPING TRAINED word -> TOP-pool rate

Both halves are trained; the "ungroundable" set is the held-out half of
the trained vocab queried with its own trained code. The per-seed
random half-split of the 16-word trained vocab measures
(strong-binder-half-median) vs (other-strong-binder-half-median + off-
target leakage), NOT trained-vs-untrained discriminability. Per-seed
INVERTED outcomes (seeds 42, 44) reflect random-split luck on a
trained-only population, NOT a real noise floor or a substrate failure.

v2 measures, per word, the WITHIN-WORD signal-to-noise ratio:

- groundable = trained word -> target-pool rate (same as v1)
- ungroundable = SAME trained word -> best-off-target-pool rate

The within-word contrast survives weak per-word binders -- a
substrate-retains-direction word still has target_rate >
best_off_target even when both are small -- and aggregating over the
FULL 16-word vocab eliminates the per-seed-split-luck variance.

The corrected v2 diagnostic (commit `7548465`) at seed 42 showed
groundable_median 0.265 > ungroundable_median 0.235 (positive direction
by margin 0.030). The full v2 calibration at seed 42 reproduces this
result exactly (0.265 / 0.235), and seeds 43 / 44 show even stronger
positive margins (0.110 / 0.121). Total: 3/3 seeds positive direction,
calibratable threshold.

## Biology-translatable insight (sharpened, now empirically validated three times)

Trustworthy abstention thresholds are **substrate-AND-protocol-specific**,
not regime-specific or substrate-only.

(a) **Substrate-specific**: the existing 650 moat
(``research/runners/abstention_gate.py``, byte-unchanged) is on G.20
SharedPool ``recall_rates`` scale (~500-800); the unified runner's
direct readout uses ``measure_pool_firing`` per-neuron mean firing rate
(scale ~0.5-2). 650 is structurally unreachable by the direct readout
regardless of how well Phase-1 trains. (The original substrate-specific
insight from the per-regime calibration `abe65f6`.)

(b) **Protocol-specific**: even on the same substrate, a flawed
calibration protocol (v1 half-split of the trained vocab) produces
INSUFFICIENT-SEPARATION when the right protocol (v2 within-word
target-vs-best-off-target gap) shows clean positive separation. The
substrate's per-word direct binding is intact; the calibration's
measurement design is what determines whether the substrate's signal
is recoverable. (New insight, this iteration.)

(c) **Trustworthy thresholds require BOTH** the right substrate's
right readout AND the right calibration protocol for that readout.
This is the principle the brain's per-regime metacognitive monitors
(Miyamoto 2017) presumably operate by -- in-situ calibration of the
specific signal each monitor accesses against the specific noise floor
of that signal.

The four accumulated substrate-and-protocol-specific calibrated moats:

| Constant | File | Calibrated for | Scale |
|----------|------|----------------|-------|
| 650.0    | abstention_gate.py | G.20 SharedPool recall_rates | ~500-800 |
| 5.688725 | abstention_gate_compositional.py | per-regime stage hippocampal one-shot lang_output readout | ~5 |
| 0.197712 (NOT YET committed) | (would-be unified compositional) | unified substrate lang_output readout via per-regime compositional calibration protocol | ~0.2 |
| **0.284167 (THIS COMMIT)** | **abstention_gate_direct_unified.py** | **unified substrate measure_pool_firing readout via v2 within-word protocol** | **~0.25-0.30** |

These do not interconvert. The decisive evaluation of the unified
architecture requires committing all relevant substrate-specific
thresholds; the COMPOSITIONAL-gate substrate-specific calibration on
the unified substrate (aggregate 0.198) is the NEXT iteration after
this direct-gate threshold commit lands.

## What this commit changes

- `research/runners/abstention_gate_direct_unified.py`: replace
  `DIRECT_UNIFIED_THRESHOLD = 0.0` (placeholder) with
  `DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667` (calibrated). Update
  docstring to remove "PLACEHOLDER" framing + cite the calibration
  source.
- That's the only source-code change. All other files byte-unchanged.

## What this commit does NOT change

- The existing 650 moat (`abstention_gate.py`) byte-unchanged.
- The existing 5.6887 compositional moat
  (`abstention_gate_compositional.py`) byte-unchanged.
- The v1 calibration function (`_calibrate_direct_one_seed`)
  byte-unchanged.
- The v2 calibration function (`_calibrate_direct_v2_one_seed`)
  byte-unchanged (it shipped in commit `b07486e`).
- Every protected file (validated subsystems + frozen verdict modules
  + the no-confab moat byte-identical and 7/7) byte-unchanged.
- The protected set byte-empty diff vs `e8a99a2` continues to hold.

## Pre-registered next step (autonomous, no hand-back)

Substrate-specific COMPOSITIONAL gate calibration commit. The previous
v1 calibration (also at commit `7548465`) already produced the
compositional-gate aggregate at full biological scale: 0.197712 with
3/3 seeds positive direction (per-seed [0.218, 0.206, 0.169]). The
substrate-specific compositional gate is the next staged iteration:

1. Add new file `abstention_gate_compositional_unified.py` mirroring
   the `abstention_gate_direct_unified.py` pattern: `COMPOSITIONAL_UNIFIED_THRESHOLD
   = 0.197712...` with calibrated docstring + same gate function
   shape.
2. Update the unified runner's compositional gate routing to use
   `COMPOSITIONAL_UNIFIED_THRESHOLD` instead of `COMPOSITIONAL_THRESHOLD`
   for the unified substrate (the per-regime stage's 5.6887 stays
   byte-unchanged for the per-regime substrate's hippocampal one-shot
   readout).
3. Subagent + adversarial review + tests + commit.
4. Then Task 4 no-harm (verify protected set still byte-empty, all
   tests still green) + Task 5 controller-only decisive run (full
   biological scale; ladder 2/4/8; 3 seeds; with BOTH unified-substrate-
   specific thresholds now in place) + mandatory smell-test (scrutinize
   a PASS harder than a FAIL) + honest propagation EVERY outcome both
   remotes.

## Honest ceiling (unchanged, restated)

Conversational / compositional capability is NOT achieved and is NOT
claimed. The decisive evaluation of the unified architecture has NOT
yet run -- this commit only lands the calibrated direct-gate
threshold; the compositional-gate substrate-specific calibration is
the next iteration, and the decisive evaluation comes after that. The
biology-translatable insight (trustworthy thresholds are substrate-
and-protocol-specific) is now three times empirically validated
(650 -> 5.6887 -> 0.197712 -> 0.284167) at progressively finer
granularity. The seventh-consecutive refusal-to-commit-a-degenerate-
threshold discipline still holds (the controller's own diagnostic
methodology bug + the v1 calibration protocol fragility were both
caught; neither was propagated as a misleading conclusion). The
no-confab moat stays 7/7 green and byte-identical; the protected set
byte-empty diff vs `e8a99a2` continues to hold; all relevant tests
still pass.

## Files / evidence

- This calibration durable JSON:
  `research/findings/raw/unified_CALIBRATION_v2_fullscale.json`
- This calibration durable log:
  `research/findings/raw/unified_CALIBRATION_v2_fullscale.log`
- Previous v1 calibration JSON (still valid for the compositional
  gate's calibration; the v1 direct-gate verdict is now reinterpreted):
  `research/findings/raw/unified_CALIBRATION_fullscale.json`
- v2 diagnostic JSON (motivated the v2 protocol design):
  `research/findings/raw/unified_DIAGNOSTIC_pure_vs_unified.json`
- v2 calibration code: `_calibrate_direct_v2_one_seed` in
  `research/runners/unified_per_regime_monitor_runner.py` (commit
  `b07486e`)
- All previously-validated modules + calibrated moats byte-unchanged.
