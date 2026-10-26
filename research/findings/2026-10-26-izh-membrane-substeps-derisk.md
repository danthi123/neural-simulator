---
type: finding
status: live
claim_check: measured (T0/T1 = pytest tests/test_izh_membrane_substeps.py on research/izh-substeps; T2 = python -m
  research.runners.izh_substep_organ_probe, 6 arms x dev seeds 71-78, numpy CPU, SIM_NO_PROVENANCE=1, outputs in the
  workstation scratch dir, bars written to BARS_PRESTATED.md before any T2 run; re-run 2026-10-26 ~15:40 as a module WITH
  provenance into research/findings/raw/_izh_substeps/t2/, every field identical, aggregated by
  research/runners/izh_substep_t2_aggregate.py)
date: 2026-10-26
lane: BG commit seam, step-1 STOP rule 4 follow-up (substrate correctness, Izhikevich integrator)
mechanism: cfg.izh_membrane_substeps (default 1) + cfg.izh_substep_refractory_rule (default model_reset); organ
  switches BRAIN_BG_IZH_SUBSTEPS / BRAIN_BG_IZH_SUBSTEP_RULE (default unset)
flip: no - default OFF; the step-1 grid re-run under the option is the next ruling's job
seeds: [71, 72, 73, 74, 75, 76, 77, 78]
verdict: de-risk passes its three pre-stated bars on 8/8 dev seeds; default OFF; no gate seed was run
runner: research/runners/izh_substep_organ_probe.py
artifacts:
  - research/findings/raw/_izh_substeps/t2/T2_summary.json
  - research/findings/raw/_izh_substeps/t2/fix_a/s72.json
  - tests/fixtures/izh_substep_cells.npz
---

# Izhikevich membrane sub-steps remove the BG membrane breach (default OFF, de-risk)

Branch `research/izh-substeps`, based on `research/bg-tonic-loop` @ 3f1249c79, not on main: the organ switches
`BRAIN_BG_PRESET_CELLS` / `BRAIN_BG_BOUNDED_COND` and `tests/test_bg_tonic_loop_identity.py` exist only on that branch.
`sim/` at 3f1249c79 equals `sim/` on main @ 011fa0036 (empty `git diff -- sim/`), so the engine change applies to main
unchanged. Scratch outputs (not committed; folder izhsub_scratch under sim-rev) are on the
workstation.

## 1. Mechanism

The bridge advances dv/dt = (k(v-vr)(v-vt) - u + I)/C by one explicit-Euler step of dt = 1 ms. The step's local
multiplier is 1 + dt k(2v-vr-vt)/C. For the striatal MSN presets (k 1, vr -80 mV, vt -25 mV) it is 1 - 55/C at rest,
so a cell is unstable at rest below C = 27.5 pF. Preset-centred jitter draws C from N(1, 0.15) x 50 pF with the ratio
clipped only to [0.1, 3.0]. Dev seeds 72/75/77 drew MSNs at 24.42 / 18.43 / 24.12 pF (str_d2_1 idx 439, str_d1_0 idx
145, str_d1_1 idx 418). The selector_reset GABA_A volley in `bg_commit_settle` kicks each one into a growing period-2
oscillation. The cell then locks into an absorbing 2-step cycle: spike, reset to c = -55 mV, a deep dip, spike. With
b < 0, u runs away and the dip reaches -1918 / -1633 / -460 mV. Bounded conductance integration was on in that organ
and is needed, but it bounds only the conductance terms, not the -u/C or quadratic term.

## 2. What was built (all default OFF)

- `CoreSimConfig.izh_membrane_substeps: int = 1` and `izh_substep_refractory_rule: str = "model_reset"`, validated in
  `__post_init__` and again at bridge build. n > 1 with a non-Izhikevich model fails the build.
- `sim/kernels.py izhikevich2007_substep_update`: n Euler sub-steps of dt/n of the same `izhikevich2007_dynamics_update`,
  with the step's assembled input current held constant. The bounded conductance currents are constant effective
  currents, so their displacement per step and their bound are unchanged. Spike detection runs at sub-step resolution
  against the bridge's own `current_spike_thresholds` array (homeostatic or vpeak), not `cp_izh_vpeak`. A
  non-refractory neuron that reaches threshold stops advancing, and the existing threshold/reset fires it.
- Refractory rules: (a) `model_reset`: a neuron held by the engine timer that reaches threshold inside a sub-step gets
  the model reset (v <- c, u <- u + d), emits no spike and keeps integrating. (b) `no_engine_timer`: the engine timer
  is never set for Izhikevich cells (effective refractory_period_steps 1).
- Bridge counters `substep_calls` (steps that ran the path) and `refractory_model_resets` (rule a; kept on the device,
  so no host sync). Both are plain Python values when the option is off, so a flag-off bridge carries no new array.
- `_step_megakernel_can_dispatch` declines when n != 1. The threshold selection moved above the update; it only
  reorders reads that the update does not write.
- `research/runners/_vocal_action_selector_gate.py`: `BRAIN_BG_IZH_SUBSTEPS` / `BRAIN_BG_IZH_SUBSTEP_RULE` ride the
  organ's existing `core_config_updates` path. When unset, no update is added.

## 3. Evidence

**T0, the three breaching cells replayed open-loop through a one-neuron SimulationBridge**
(`tests/test_izh_membrane_substeps.py`; fixture `tests/fixtures/izh_substep_cells.npz`, 48 KB, built from the
diagnosis probe's traces by `tests/fixtures/build_izh_substep_cells_fixture.py`). Each step writes the cell's recorded
g_e, g_i and other currents into the bridge, with conductance decay set to 1. The bridge's own bounded kernel,
Izhikevich update, threshold, reset and 2-step timer do the rest. Values: v_min mV / spikes over 2755 steps.

| cell | flag off (= organ trace) | C 50 pF, flag off | n 2 | n 4 rule a | n 4 rule b |
|---|---|---|---|---|---|
| s72 str_d2_1 (C 24.42) | -1917.787 / 1332 | -80.00 / 37 | -80.00 / 23 | -80.00 / 20 | -80.00 / 20 |
| s75 str_d1_0 (C 18.43) | -1632.923 / 1334 | -83.82 / 15 | -81.71 / 13 | -84.18 / 11 | -84.18 / 11 |
| s77 str_d1_1 (C 24.12) | -460.418 / 1329 | -80.39 / 14 | -80.73 / 13 | -81.43 / 12 | -81.43 / 12 |

With the flag off, the replay equals the organ exactly: same v_min, spike count and first step below -150 mV
(107 / 95 / 110). It also holds the lock-in to step 2755. Every n = 4 arm keeps v in [-150, vpeak] on every step. In
the replay these MSNs never cross threshold while refractory, so rules a and b coincide (0 model resets).

**T1, default path unchanged.**
- With the option set to 1 explicitly (and rule b set, which is ignored at 1), the bounded-conductance reference
  network (AMPA/GABA_A, GABA_B, NMDA, OU; 60 steps) gives a hash equal to the default build's.
- Both still equal the pre-fix engine hash captured on main at 3c2907b85.
- Scratch pre/post capture (`identity_pre.json` before the edit, `identity_post.json` after; files compare equal):
  the four BG golden build cases plus a dev-seed-71 organ run under PRESET+BOUNDED+TONIC. That run covers a per-step
  sha256 of v/u/fired over 2755 steps and every bridge array after the 4 hsb events.
- Bridge-level checks of the option itself: the legacy reset path (fast_spike_reset off) gives the same trace as the
  fast path under sub-steps for both rules. A 1200 pA FS cell (the selector_reset drive) gives, over 50 steps:
  off, 25 spikes with none on consecutive steps; rule a, 25 spikes plus 30 counted model resets; rule b, 50 spikes,
  49 of them one step after the last.
- `test_bounded_conductance_integration`, `test_bg_neural_seam`, `test_read_only_fast_step`,
  `test_preset_centred_heterogeneity`, `test_backend_neutral_izh_*`, `test_kernels_cpu`, `test_step_megakernel` (GPU
  cases skip) and `test_adex_fast_spike_reset` pass.
- I ran the 76 existing test files that step the bridge on the unedited base 3f1249c79 and on this branch, numpy CPU.
  Both give 625 passed, 49 failed and 1 collection error (`test_determinism`, cupy), and the failure lists are
  identical (`step_tests_base.fails` = `step_tests_branch.fails` in scratch).

**T2, organ on dev seeds 71-78** (GPe/STN tonic 100/100 pA = grid set stn100_gpe100; build 215 steps + 4 hsb events
= 2755 steps; breach = any neuron outside [-150, +100] mV on any step). Bars were pre-stated in
`BARS_PRESTATED.md` (13:55:39), before any T2 run. Disclosure: one s72 code-path smoke under rule a ran before that
file was written. It reproduced the diagnosis prototype's substep4r numbers, which were already known.

| arm | breach cells, seeds 71..78 | v_min range mV | events matching [1,0,1,0] | abstentions | model resets (8 seeds) |
|---|---|---|---|---|---|
| fix_off | 0,1,0,0,1,0,1,0 | -1917.8 .. -96.5 | 29/32 | 2 | 0 |
| fix_a (n 4, rule a) | 0 x 8 | -90.1 .. -86.3 | 31/32 | 1 | 15730 |
| fix_b (n 4, rule b) | 0 x 8 | -90.4 .. -86.8 | 31/32 | 0 | 0 |
| fix_a + bounded off | 217,205,206,231,235,217,228,233 | -279.7 .. -268.6 | 0/32 | 32 | 1247797 |
| fix_a + preset cells off | 0 x 8 | -83.7 .. -81.2 | 16/32 | 4 | 0 |
| fix_a n 2 (report only) | 0 x 8 | -88.4 .. -86.2 | 30/32 | 1 | 9300 |

- B1 (fix-on arms keep every neuron in [-150, +100] mV on every step, 8/8 seeds): fix_a 8/8, fix_b 8/8,
  fix_a_nopreset 8/8. PASS. The maximum v in every fix-on arm is at most 39.998 mV.
- B2 (fix off breaches on exactly {72, 75, 77} and its hsb_empty1/hsb_q v_range equals the step-1 grid's stn100_gpe100
  seam_b v_range): breach set {72, 75, 77}, one cell each (the three cells above, first breach step 107 / 95 / 110),
  1323-1330 out-of-range steps. The v_range matches exactly on 8/8 seeds (e.g. s72 [-1425.808, 39.976] and
  [-1755.478, 39.972]). PASS.
- B3 (fix on + bounded off still breaches, >= 100 cells per seed): 205-235 cells on 8/8 seeds, all 32 events
  abstain. PASS: the bound is still needed, and the sub-steps do not replace it.
- Instrument validity, 6/6 arms: `substep_calls` = 2755 in every fix-on arm and 0 in fix_off.
  `refractory_model_resets` = 0 in fix_off and fix_b. The cfg read-back (substeps, rule, bounded, preset) matches each
  arm. A rerun of all 48 cells reproduced every membrane, breach, event, counter and cfg field.

## 4. Refractory rule: recommend (a) `model_reset`, with n = 4

Both rules pass B1 with identical event matches (31/32). They differ in what the rest of the organ sees (pooled over
8 seeds):

| region | rule a: silent model resets | rule b: spikes on consecutive 1 ms steps | transmitted spikes off / a / b |
|---|---|---|---|
| selector_reset | 12975 | 25675 | 14540 / 14145 / 27120 |
| latch_fs_0 / latch_fs_1 | 1286 / 1118 | 2150 / 1877 | 36936 / 48104 / 49186 (fs_0) |
| hold_inh | 176 | 259 | 29265 / 34267 / 34440 |
| commit_fs, bgc_gate_inh | 7-97 | 4-144 | |
| MSN, GPe, GPi, STN, thalamus, RS pools | 0 | 0 | |

- **Rule (b)** doubles the selector_reset volley: the cell fires on 25675 of its 27120 spikes one step after the last,
  so it sits at the 1-spike-per-step cap of 1000 Hz. It also changes the rate ceiling of every Izhikevich cell
  brain-wide, from 500 Hz (refractory_period_steps 2) to 1000 Hz. A spike event cannot be emitted at sub-step
  resolution, so rule (b) is not Izhikevich's formulation either. It is a different ceiling, and adopting it would
  recalibrate every faculty that uses Izhikevich cells.
- **Rule (a)** keeps the engine's refractory semantics everywhere; transmitted selector_reset spikes stay at 14145 vs
  14540 off. The cost is a silent u += d with no emitted spike, 1773-2180 times per seed. 82% of these are in
  selector_reset, which is driven at the held 1200 pA reset current ("no source found", STEP1-SPEC 2b); the rest are
  in latch_fs. Only cells driven past the 500 Hz engine ceiling are touched, and `refractory_model_resets` counts every
  case, so the effect stays auditable. The companion default is already `model_reset`.
- **n = 4, not 2:** n = 2 also clears every breach. At n = 2, though, an FS cell hyperpolarized toward E_i with no
  conductance still has a step multiplier above 1 (section 5). At n = 4 every region's worst-cell multiplier is at
  most 0.946. <!--derived-->

## 5. A-priori stability (worst cell per region over the 8 dev seeds; E_i = -75 mV everywhere)

The table gives the linearised full-step multiplier, u held, rounded from `minC_mult` in research/findings/raw/_izh_substeps/t2/T2_summary.json (derived). At v = vr: |r^n|. At v = E_i with strong conductance:
|r^n - (1/n) sum r^j|, where r = 1 + (dt/n) k(2v-vr-vt)/C and the bounded current is held over the sub-steps. Below 1
is stable. The per-sub-step factor |r| (the literal |1 + h Q'(v)/C|) is in each probe JSON (`substep_rest`,
`substep_Ei`).

| region | min C pF | n 1: rest | n 1: E_i, g 0 | n 1: E_i, strong g | n 4: rest | n 4: E_i, strong g |
|---|---|---|---|---|---|---|
| str_d1_0 | 18.43 | 1.984 | 1.441 | 2.441 | 0.420 | 0.412 | <!--derived-->
| str_d1_1 | 24.12 | 1.280 | 0.866 | 1.865 | 0.397 | 0.411 | <!--derived-->
| str_d2_1 | 24.42 | 1.252 | 0.843 | 1.843 | 0.427 | 0.414 | <!--derived-->
| str_d2_0 | 27.68 | 0.987 | 0.626 | 1.626 | 0.451 | 0.415 | <!--derived-->
| FS regions (selector_reset, latch_fs, hold_inh, commit_fs, bgc_gate_inh) | 9.79-12.98 | 0.43-0.53 | 3.24-4.62 | 4.24-5.62 | 0.54-0.60 | 0.40-0.41 | <!--derived-->
| selector_stn | 52.91 | 0.869 | 0.475 | 1.134 | 0.875 | 0.386 | <!--derived-->
| GPe / GPi | 39.0 / 36.5 | 0.81-0.82 | 0.56-0.59 | 0.90 / 0.96 | 0.82-0.84 | 0.35-0.36 | <!--derived-->
| RS pools, thalamus | 50.6-125.0 | 0.90-0.95 | 0.74-0.78 | 0.51-0.69 | 0.90-0.95 | 0.24-0.30 | <!--derived-->

FS cells are stable at their own rest (vr -55 mV) but strongly unstable at dt = 1 ms if inhibition pulls them toward
E_i. The defect is latent beyond the MSNs.

## 6. Caveats

- **Backend.** The seed-to-cell map is numpy-CPU specific: heterogeneity draws through `cp.random`, so CuPy draws other
  cells. Which cells breach on GPU is unmeasured, and a CuPy membrane check is owed before any GPU claim. The
  megakernel declines with the option on, so a GPU run falls back to the Python step.
- **Gate seeds not run.** Per the diagnosis, build-only reads put most gate seeds' minimum MSN C at 25.8-29.9 pF.
  Without the volley kick, the breach may be mostly a dev-block artifact, but the defect is latent substrate-wide
  (section 5).
- **Decisions move on clean seeds too.** fix_a changed 3 of the 20 events on the clean seeds 71/73/74/76/78 (s71
  hsb_empty1 1 -> abstain, s73 hsb_hi abstain -> 0, s78 hsb_empty2 abstain -> 1). The option also shifts firing rates (fix_a vs off,
  pooled): GPi +18.6%, GPe +8.1%, latch_fs_0 +30%, hold_inh +17%, selector_stn -4%. The dt = 1 ms Euler f-I is not the
  model's f-I. So a step-1 grid under the option is a new measurement, and the Part C f-I pre-screen and the tonic
  calibration (computed at n = 1) need re-reading under the same integrator before any calibration claim.
- **Open-loop T0.** The T0 arms other than flag off replay the flag-off organ's recorded input; closed-loop behaviour
  is T2's job. Within a step, a neuron that crosses threshold freezes (u included) until the step's reset. Spike
  times are still quantized to the step, with at most one spike per step.
- **Timing.** Wall time is not measured cleanly: the probe's per-neuron attribution loop runs only in rule-a arms.
  The medians per seed (build + 4 events, 872 neurons, numpy) are 1.00 s off, 1.14 s rule b and 1.39 s rule a.
- **Pre-existing test failures, identical on the unedited base 3f1249c79:**
  - `test_bg_tonic_loop_identity::test_flag_off_identical_to_golden`: the golden's array set predates
    `cp_synapse_pulse_progress` / `cp_synapse_pulse_timers`, added by a main merge. Every common array hash still
    matches. Not re-captured, since that needs a ruling.
  - `test_bridge_lineage_cli::test_rollback_to_snapshot`.
  - `test_determinism.py` and `test_fast_spike_reset.py` import cupy and need a GPU, so they were not run in this
    CPU-only lane.
  - Of the bridge-stepping files, 49 tests fail on the base exactly as on the branch (e.g. `test_regions`,
    `test_neuromodulators`, `test_d1_d2_asymmetry`). Their causes were not investigated here.

## 7. Next (for the ruling, not done here)

1. Amend STEP1-SPEC 2c with `BRAIN_BG_IZH_SUBSTEPS=4` (`model_reset`) as a fixed item, then re-run the step-1 grid on
   71-78. The old grid does not carry over (section 6).
2. Run a CuPy membrane check of the same organ before any GPU or gate-seed claim.
3. Re-read the Part C isolated-cell f-I under the same integrator.
4. Examine the 1200 pA selector_reset drive, the source of most model resets and of rule (b)'s 1 kHz volley.

Functional read-outs only (membrane range, spike counts, event winners); nothing here bears on experience.
