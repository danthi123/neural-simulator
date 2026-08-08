---
type: finding
status: contributing
date: 2026-08-07
mechanism: stageA-conversation-integration-foundation-honesty-floor
lane: E-language
runner: research/runners/_stageA_foundation_honesty_arbiter_derisk.py
artifacts:
  - research/findings/raw/lanes/stageA/stageA_foundation_honesty_arbiter_s42.json
---

# Stage-A FOUNDATION (STEP 0 + STEP 1) — the honesty floor routes the CALIBRATED monitor, single-seed smoke

Product of the revised Stage-A first build (`2026-08-07-stageA-conversation-integration-DESIGN.md`): lead with the
HONESTY FLOOR, not the affect win. Single-seed SMOKE, backend numpy, seed 42; the 6-seed sweep is the parent's job.
Artifact: `research/findings/raw/lanes/stageA/stageA_foundation_honesty_arbiter_s42.json`.

## What was built + measured (all anti-cheats live in one process)

- **STEP 0 substrate unification.** `MergedNavConvAgent` drives a `CoResidentOneBrainComposer` on ONE merged
  `SimulationBridge` (`substrate_unified=true`, composer class confirmed). The real no-confab moat holds under a
  hard-moat abstain battery: 475/475 unknown cues abstain with 0 added false-accepts, and the honesty-floor wrapper
  never flips a moat abstain into an answer.
- **STEP 0 default-OFF byte-identity.** An inert null co-resident slice (`hon_null`, `internal_density=0`, appended
  LAST, 350→414 neurons) leaves the baseline neurons' `cp_neuron_firing_thresholds` byte-identical (sha256 match).
- **STEP 0 RNG isolation.** A read-only measurement forward wrapped in a snapshot/restore guard (dynamical state +
  global numpy RNG) leaves the faculty trajectory byte-identical to a no-measurement run, while the UNGUARDED
  measurement shifts it (`isolation_proven=true`).
- **STEP 1 the crux — the honesty floor routes the CALIBRATED monitor.** The confidence source is the calibrated
  `LearnedAccApfcMonitor.confidence_from_features` (dynamic feature mode) → `current_from_confidence` → the spiking
  `meta_schema→self_schema` relay → the certainty band `{assert, hedge, soft_abstain, MOAT}`. It is NOT the
  recall/trace/margin score (the PARTIAL wire-in's flaw). The monitor separates correct-from-error at type2-AUC
  0.9124 vs the raw recall score's 0.7346, and the two signals are distinct (learned-vs-recall correlation 0.3009).
- **STEP 1 the g_eff law + FM4.** The fixed composition law `cue_match_moat (HARD) < honesty_floor < affect/DA`
  holds (`ordering_ok`), and a yoked high-arousal affect term can never flip an abstain into an assert.
- **STEP 1b the 3-way arbiter.** A genuine competitive-queuing build — three self-exciting pools
  {volunteer | ask | stay-silent} + one shared inhibitory pool — arbitrates 3-way: each faculty wins its regime
  (distinct correct winners) and the winner suppresses the runner-up via the shared inhibition (intact margin 0.589 min <!--derived-->
  ), which the mutual-inhibition lesion collapses (0.064 max <!--derived-->). NOT a repurposed 2-pool standing-state WTA.

## The honesty behavior — reported HONESTLY (single seed, a lift, not a solve)

At a MATCHED assert count (40 asserts), routing the calibrated monitor makes 1 confident-wrong assertion vs the
recall score's 5, while keeping MORE correct asserts (39 vs 35) — a single-seed reduction of confident-wrong
assertions on 42 error trials.

<!--derived-->
This is a single-seed LIFT of the honesty behavior, driven by the calibrated monitor's better correct/error
discrimination (AUC 0.9124 vs 0.7346, a 0.18 gap). It is NOT the monitor's 6/6 discrimination label imported onto
the behavior (the premortem's exact overclaim), NOT a 6-seed result, and NOT a solved honesty mechanism — the
isolated honesty behavior was 3/6 PARTIAL and the co-resident 6-seed sweep is pending.

## Honest scope / residuals

- SINGLE SEED smoke; the 6-seed confirmation is the parent run.
- The familiar-but-wrong battery is operationalized as genuine first-order 2AFC errors (a familiar item decoded
  wrongly), NOT composer recall confusions; the store↔spiking-readout reconciliation (design FM6) is untouched here.
- RNG isolation: in this OU-off config the raw measurement does NOT advance the global numpy RNG
  (`rng_advanced_by_raw_measurement=false`), so the contamination the guard eliminates here is via dynamical STATE;
  the global-RNG restore is the belt-and-suspenders that covers the seed-46 leak for configs that DO draw per step.
- The affect term in the g_eff law is a STUB (Step 2 builds the real spiking affect coloring), and the 3-way
  arbiter is validated as a mechanism but is not yet FED by the live faculties (Steps 2/3 wire answer-readiness,
  curiosity ask-drive, and affect-arousal into it).
- Reuse-by-import, additive/default-OFF, no `sim/` edit; `cfg.seed` set (not `actual_seed_used`).

Parent 6-seed:
`PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._stageA_foundation_honesty_arbiter_derisk --seed <S> --n-trials 120 --moat-battery 475 --out research/findings/raw/lanes/stageA/stageA_foundation_honesty_arbiter_s<S>.json`
for S in 42 43 44 100 101 102.
