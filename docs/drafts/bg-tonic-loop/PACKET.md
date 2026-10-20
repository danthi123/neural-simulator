# bg-tonic-loop: put the BG organ on its real cell types and measure it before choosing a mechanism

**Owner:** Claude (2026-10-20, v3; v1 and v2 each came back FIX from a two-lens adversarial review, and every issue
is addressed here or moved to section 4). **Driver:** Hermes. **Answers:** RULINGS bg-tonic-stn-cliff-1020 to -1020g and
preset-het-1020/-1020b (`research/coordination/LOCAL_LLM_RUNBOOK.md` section 6). **Scope:** dev seeds 9-16 only; no gate
seed and no bar of `research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md` is read or reported. Every
new switch is DEFAULT OFF; with every switch unset the organ is byte-identical to `research/bg-tonic-drive` at
`8177576ae` after it is merged with S main at or after `42747441e` (entry assert: `git merge-base --is-ancestor 42747441e
HEAD`).

## 1. Why this packet is measurement-first

The tonic pallidal drive is the BG commit seam's second method (RULING bg-commit-next-1014: real BG holds by default
through tonic GPi/SNr inhibition fed by tonically active GPe and STN). Its STN rate swung 7.3-32.7 Hz across dev seeds at
40 pA (`research/coordination/bg-tonic-stn-1019-RESULT.md`). Two reviews of this packet's earlier drafts then showed:

- **The organ never ran its own cell types.** Global parameter heterogeneity (on by default) overwrote every neuron's
  Izhikevich a, b, d and C with jitter centred on the RS defaults, so STN, GPe, GPi, the MSNs and the thalamic relay all
  ran as RS hybrids (dump at seeds 9/12/14; `docs/drafts/preset-centred-het/CENSUS.md`). The fix is the default-OFF
  `preset_centred_parameter_heterogeneity` (S `42747441e`). Every constant in this organ (GPe 100 pA, STN 40-50 pA, GPi
  1000 pA, thalamus 900 pA, practice 1000 pA, reset 1200 pA, and the weights) was tuned on hybrids.
- **The presets have their own behaviour, not yet checked against biology** (numpy, single cells, kernel update order,
  dt 1 ms, refractory 2, from the reviewers' runs; to be reproduced in step 0):
  - the STN preset BURSTS throughout 10-30 Hz (ISI-CV 1.2-1.9, burst fraction 0.5-0.7) and fires regularly only above
    about 56 Hz;
  - the GPe preset jumps from silent to about 62 Hz at rheobase, and is at about 88 Hz at 100 pA;
  - at STN 40 pA a preset STN cell fires about 4 Hz alone.
- **Synapses are conductance-based** (E_inh -75 mV), so the dev table's -117.2 mV (a whole-bridge minimum) is the
  documented forward-Euler overshoot (`enable_bounded_conductance_integration`, default OFF, bounds synaptic terms one at
  a time; it does not touch the Izhikevich quadratic term).

So any grid of drive currents, adaptation or loop weights designed now would be fitted to cells nobody has measured in the
organ. Step 0 measures; Claude writes the recalibration (step 1 onward) from its RESULT.

## 2. Step 0a: the switches (a small build, default OFF)

- `BRAIN_BG_PRESET_CELLS` (env, default OFF): when on, `build_selector_bridge` passes
  `preset_centred_parameter_heterogeneity=True` through `core_config_updates`, including when the bridge is built by
  `NeuralSeamSelector` (whose `core_config_updates` today come only from `neural_seam_parts`).
- `BRAIN_BG_BOUNDED_COND` (env, default OFF): the same path for `enable_bounded_conductance_integration=True`.
- A flag-off identity test for both: every bridge array and the RNG state after build are byte-identical to today with
  either switch unset, through both `build_selector_bridge` and `NeuralSeamSelector`.

## 3. Step 0b: the measurement (nothing else changes)

**Part A, the tonic epoch, organ only (numpy, minutes per run).** A new probe modelled line for line on
`research/runners/_bg_tonic_calibration.py`: `selector_config("v2")`, `build_selector_bridge(seed, cfg,
core_config_updates=...)`, `_set_equal_tonic_current`, BRAIN_BG_TONIC_DRIVE=1, no arousal and no stimulus, 200 steps of
equilibration discarded, then 1000 measured steps (dt 1 ms). One fresh bridge per run (the calibration probe reused one
bridge across grid points; this probe does not). Seeds 9-16. Currents: GPe 100 pA with STN 50 pA (the registered pick
the dev refile ran) and with STN 40 pA (the sweep point). Arms: A0 (hybrid, today), A0p (BRAIN_BG_PRESET_CELLS=1), A5
(A0 + BRAIN_BG_BOUNDED_COND=1), A5p (A0p + BRAIN_BG_BOUNDED_COND=1). Assert per run that A0p differs from A0 only in the
four parameter arrays (a, b, d, C). That is 8 seeds x 2 currents x 4 arms = 64 runs.

Record for every region the bridge builds (practice_arousal, selector_stn, selector_reset, and for ch in CHANNELS =
(0, 1): proposal_{ch}, str_fsi_{ch}, str_d1_{ch}, str_d2_{ch}, gpe_{ch}, gpi_{ch}, thal_{ch}, commit_{ch},
commit_fs_{ch}, motor_{ch}):
- every neuron's rate, ISI-CV and burst fraction (share of ISIs at or below 10 ms), and the count of silent neurons;
- the count of steps with v below E_inh - 5 mV, and v_min.

**Part B, the decision check in the chat (the dev refile's exact shape).** `host_share_bg_seam` arms seam_b,
seam_readout and seam_lesion_value with BRAIN_BG_COMMIT=1, BRAIN_BG_TONIC_DRIVE=1, GPe 100 pA and STN 50 pA, seeds 9-16,
on the same lane as `research/coordination/bg-tonic-dev-1019.md` (96 bg_neural commit records per arm). Arms A0 and A0p;
A5p is added if Part A shows bounding removes sub-reversal steps.
- **Entry gate:** A0 reproduces the refile's persisted `tonic_rates_hz` on all 96 records exactly (the step-0a identity
  test predicts equality), or STOP before anything else is read.
- Record per record: the committed winner, `read_invariant`, latch onset and latch rates, gate and motor counts,
  `tonic_rates_hz`, and per turn the reply and whether it abstained.

**Part C, isolated cells (numpy, minutes; kernel-faithful: simultaneous v/u update from the old v, refractory 2, dt 1).**
For each preset in the organ (IZH2007_STN_BURST, IZH2007_GPE_PACEMAKER, IZH2007_GPI_OUTPUT, IZH2007_THALAMIC_RELAY,
IZH2007_STRIATAL_MSN_D1, IZH2007_STRIATAL_MSN_D2, IZH2007_FS_CORTICAL_INTERNEURON), with jitter centred on the preset:
rate, ISI-CV and burst fraction against injected current (0 to 1200 pA), and the rebound response to a 200 ms
hyperpolarising step. Report next to each the published range it should match, with a citation, or mark "no source
found".

**What step 0 decides (fixed now; nothing else).**
- If A0p's committed winner, `read_invariant`, or any turn's reply or abstention differs from A0's on any seed, the seam
  does not survive the move to real cells at today's constants. That is reported as the main result, and the
  recalibration becomes the method.
- If A5 or A5p removes the sub-reversal steps in Part A, every later step runs with BRAIN_BG_BOUNDED_COND=1. If steps
  below E_inh - 5 mV remain under A5p, report which regions, and Claude rules before any bar uses v.
- Nothing in step 0 picks a current, a weight or a mechanism.

Hermes files `[bg-tonic-loop-0]` RESULT with the tables and STOPs.

## 4. What comes after step 0 (named so the record is read in advance; specified only after the RESULT)

- **Recalibration on real cells:** each region's tonic constant re-derived against its isolated-cell f-I and the
  published band, with per-cell criteria (rate band, ISI-CV, no silent cells) and the PRE window as the tonic epoch.
- **STN pacing (P):** the preset bursts in band, so in-band single-spike pacing needs a cell-level method: the reset
  voltage (c) or a second slow variable standing for slow Na+ inactivation (Do & Bean 2003 Neuron 39:109: persistent and
  resurgent Na+ drive STN pacing, and their slow inactivation keeps the rate constant). Template: the HH-only NaP/SK/Ca/h
  bundle (`BrainRegion.snr_g_*`, `kernels.fused_snr_conductance_update`), which does not run in this Izhikevich bridge.
  The SK medium AHP sets period and precision and terminates rebound bursts, but does not cause spike-frequency
  adaptation (Hallworth et al. 2003 J Neurosci 23:7525); the Izhikevich (d, a) is a linear adaptation scaffold, not SK.
- **The STN-GPe loop (L):** STN -> GPe is missing. The reciprocal circuit can regulate or oscillate depending on coupling
  (Plenz & Kitai 1999 Nature 400:677; Terman et al. 2002 J Neurosci 22:2963), so any loop bar measures synchrony against a
  shuffled surrogate with a stated bin width, not ISI-CV floors or a raw spectral peak.
- **Hyperdirect input (proposal -> STN):** absent today, so STN cannot carry proposal information; its own packet later.
- **Bands** to be cited in that packet: STN, GPe, GPi/SNr and thalamus, per state; this draft states none as a bar.

## 5. Cost and calendar

Step 0a: one small build with its identity test (Hermes). Step 0b: Part A is 64 organ-only numpy runs of 1200 steps;
Part B is two (or three) repeats of the dev refile's seam runs; Part C is minutes. Target: `[bg-tonic-loop-0]` by 10-23. The BG
commit seam's gate (Stage-2 window 11-03..11-10) now depends on the recalibration; if step 0 shows it cannot land by 11-01,
Claude rules on the calendar in the RESULT's reading rather than run gate seeds on hybrid cells.
