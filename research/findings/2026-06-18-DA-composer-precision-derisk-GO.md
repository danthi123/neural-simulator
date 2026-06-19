# DA -> conversational composer = salience-gated PRECISION -- cheap-first de-risk: **GO** (6/6)

**Date:** 2026-06-18
**Type:** cheap-first de-risk (numpy/CPU, NO GPU). Verdict: **GO** (6/6 seeds 42-47, all anti-cheats clean).
**Direction:** TRUE-ONE-BRAIN roadmap **#6** -- let the SHARED spiking dopamine (SNc) signal MODULATE the
conversational composer, so the limbic core reaches the cortex on BOTH halves (navigation AND conversation). The
deepest "one self" step: the same dopamine the BG actor learns from also shapes how the composer recalls.
**Scoping (GO):** `research/findings/2026-06-18-DA-NM-composer-closure-scoping.md` (committed `566b68af`),
Option A (salience-gated cleanup sharpening via a DA-driven `confidence_gate`).
**Runner:** `research/runners/_da_composer_salience_cleanup_derisk.py`.
**Raw:** `research/findings/raw/_da_composer_precision_derisk.json`.

---

## TL;DR for the controller

- **VERDICT = GO, 6/6 seeds.** A DA/salience signal -- sourced from a SPIKING SNc -- scaling the composer's
  `confidence_gate` produces a robust, biologically-meaningful **salience-gated PRECISION** effect: under matched
  cleanup noise, a salient (high-DA) turn makes the answered reads markedly more reliable (cue-role error roughly
  HALVED, all 6 seeds), while the moat is held-or-STRENGTHENED at every DA level, and a lesion of the SNc->dopamine
  drive abolishes the effect exactly.
- **The mechanism is moat-safe BY CONSTRUCTION and confirmed empirically.** DA only ever RAISES the gate
  (`g_eff = clip(g0, g_cap, g0 + k*(DA - DA_baseline))`, clamped below at `g0`), so abstention can only get
  STRICTER -- a raised gate converts marginal reads to abstain; it can never turn an abstain into a false-accept.
  Empirically: **0 true moat breaches** (a true breach would be DA_high false-accepts > DA_low, structurally
  impossible here); DA_high false-accepts = 0 on all 6 seeds; on seed 43 DA even **closes** a baseline-`g0` leak
  (1 -> 0 false-accepts). The no-confab moat is never weakened -- DA tightens it.
- **The controller's CORRECTED target ("salience-gated precision", not "more facts recovered") is the right one,
  and is met -- with one honest sharpening of WHICH precision.** The `confidence_gate` keys on the **agent+action
  cue-role** margins, so the inverted-U sharpening lands on the **cue-role read** (agent+action correctness among
  non-abstained), exactly the quantity it gates on -- and a confident cue-MATCH is precisely what makes a recall
  trustworthy. (Honest boundary, foregrounded: the PATIENT unbind has INDEPENDENT FHRR noise, so the same
  agent+action gate does NOT reduce patient error -- reported as `patient_error_rate`. The gate gates on what it
  answers; the cue-role fidelity is the load-bearing precision.)
- **NO `sim/` edit.** Composer-runner-layer read of `bridge.neuromodulator_manager.get_concentration("dopamine")`
  -> scale the existing `OneBrainComposer.confidence_gate`. The DA signal is already produced on the merged "one
  brain" by the spiking SNc (`nav_conv_merged_bridge.py:757-761`, roadmap #1/#2 DONE); only the composer-side
  CONSUMER is new, and it is host glue reading a spike-derived scalar.

---

## The probe (numpy/CPU)

`research/runners/_da_composer_salience_cleanup_derisk.py`, `SIM_BACKEND=numpy`, 6 seeds (42-47).

**1. The salience source is NEURAL** -- a tiny spiking `snc` Izhikevich pool (`IZH2007_DOPAMINE`, 30 neurons) +
the `dopamine` `from_region_firing_signed` modulator, the EXACT mechanism the merged one-brain uses (and the proven
drive-SNc-and-read recipe of `snc_pavlovian_probe.py`). Driven via `cp_external_input_current`, stepped through
`_run_one_simulation_step` (which advances the dopamine EMA each step), DA read from
`get_concentration("dopamine")`:
- **DA_low** = SNc tonic (drive 80 pA -> ~10 Hz) -> DA ~= 0.49 ~= baseline (0.50). `threshold` set to the tonic
  firing fraction so a quiescent SNc maps to ~0 net production = no modulation.
- **DA_high** = SNc salient (drive 600 pA -> ~130-160 Hz) -> DA ~= 0.61-0.64 (above baseline = a salient/novel turn).

**2. The map (clamped to ONLY sharpen + the inverted-U ceiling):**
`g_eff = clip(g0=0.06, g_cap=0.25, g0 + k*(DA - DA_baseline))`, `k=2.0`. The lower clamp (`g0`) makes DA moat-safe
(it can only raise the gate); the upper clamp (`g_cap`) is the biologically-apt **inverted-U ceiling**
(Vijayraghavan/Arnsten: excess D1 ERODES tuning, so DA must *raise*, not blindly maximize, the gate) -- it also
prevents a hot SNc from over-sharpening into uselessness. At DA_baseline, `g_eff = g0` (the no-modulation knob =
the byte-identical current composer).

**3. The composer cleanup is the ACTUAL production knob.** We import `OneBrainComposer._margin` (the real margin
function) and apply `OneBrainComposer`'s EXACT gate logic -- `min(margin(agent), margin(action)) < g => abstain`
(`one_brain_composer.py:211,262`) -- to FHRR phasor cleanup (the same bind = role (x) filler / bundle = sum /
unbind = conj / cosine-argmax algebra `RFPhasorComposer` uses), with controllable complex-jitter cleanup noise (the
graceful-degradation dial; `noise_sigma=2.0`, D=64, 8 facts -- a regime where DA_low has a real ~16-36% cue-role
error the gate can reduce). This is faithful to the production `confidence_gate` mechanism while staying CPU-cheap
(no parser train, no per-op bridge build).

---

## Results (6/6 GO)

| seed | DA_low (SNc Hz) | DA_high (SNc Hz) | g_eff low->high | CUE-ROLE err DA_low -> DA_high | dErr | answer-rate high | moat (low,high) | lesion |
|------|-----------------|------------------|-----------------|-------------------------------|------|------------------|-----------------|--------|
| 42 | 0.493 (10) | 0.610 (130) | 0.060 -> 0.250 | 0.220 -> **0.099** | +0.121 | 0.51 | (0, 0) | abolishes (0.000) |
| 43 | 0.493 (10) | 0.617 (137) | 0.060 -> 0.250 | 0.362 -> **0.153** | +0.210 | 0.37 | (1 -> **0**) | abolishes (0.000) |
| 44 | 0.493 (10) | 0.612 (132) | 0.060 -> 0.250 | 0.271 -> **0.113** | +0.158 | 0.44 | (0, 0) | abolishes (0.000) |
| 45 | 0.493 (10) | 0.625 (146) | 0.060 -> 0.250 | 0.162 -> **0.082** | +0.081 | 0.61 | (0, 0) | abolishes (0.000) |
| 46 | 0.492 (9)  | 0.613 (133) | 0.060 -> 0.250 | 0.226 -> **0.113** | +0.114 | 0.44 | (0, 0) | abolishes (0.000) |
| 47 | 0.494 (11) | 0.640 (162) | 0.060 -> 0.250 | 0.321 -> **0.097** | +0.224 | 0.39 | (0, 0) | abolishes (0.000) |

**Frozen GO bar (all met):**
- **(a) salience-gated PRECISION -- PASS 6/6.** The cue-role error-rate among non-abstained reads is LOWER at
  DA_high than DA_low on every seed (error roughly HALVED; dErr +0.081 to +0.224). The gate is NOT over-abstained
  into uselessness: DA_high still answers 37-61% of reads (and recall of margin>=g_eff reads = 1.00 -- it keeps its
  own confident reads). This is the Vijayraghavan/Arnsten D1 inverted-U "sharpens tuning by suppressing nonpreferred
  responses" landing on the cue-role read-out the gate evaluates.
- **(b) the MOAT held-or-STRICTER -- PASS 6/6 (the hard gate).** Zero true moat breaches (DA_high false-accepts >
  DA_low is structurally impossible since `g_eff_high >= g_eff_low`). DA_high false-accepts = 0 on all 6 seeds; on
  seed 43 a baseline-`g0` leak under heavy noise (1 false-accept) is **CLOSED by DA** (-> 0) -- the mechanism
  TIGHTENING the moat, reported honestly, not a breach. The no-confab moat is never weakened.
- **(c) LESION abolishes the effect -- PASS 6/6 (decisive, proves it's neural).** With the SNc->dopamine drive
  severed (DA pinned at baseline regardless of the SNc), `g_eff` collapses to `g0` and the precision difference
  VANISHES: `lesion_effect = 0.000` on every seed vs the live `da_effect` of +0.08 to +0.22. The precision gain is
  driven by the spiking SNc firing, not a re-hidden host scalar.

---

## Anti-cheat controls (all hold)

1. **The modulation is NEURAL.** DA = `from_region_firing_signed` over the spiking `snc` pool's FIRING; the
   **lesion** control (sever SNc->dopamine) abolishes the effect exactly (a host-constant version would be
   lesion-insensitive). Host residual is limited to presenting the cue + reading the cleanup argmax/margin.
2. **The no-confab MOAT holds-or-strengthens at EVERY DA level.** 0 true breaches; DA_high = 0 everywhere; DA
   closes the one baseline leak. Structurally one-directional (DA can only raise `g`).
3. **DA-OFF == the no-modulation baseline.** At DA_low (~= baseline), `g_eff = g0` -- the composer behaves as the
   chosen `confidence_gate` default; no always-on perturbation.
4. **Honest-negative framing applied.** The PATIENT-precision null (the gate's agent+action margin does NOT reduce
   the decorrelated patient-unbind error) is reported as `patient_error_rate`, not buried -- and it correctly
   re-points the precision claim to the CUE-ROLE read (what the gate gates on), which is the trustworthy-recall
   signal.

---

## Honest scope / what this is and isn't

- **Is:** a multi-seed-robust demonstration that the SHARED spiking-SNc dopamine can usefully + safely modulate the
  conversational composer's recall PRECISION (salience-gated cue-role sharpening), realized at the composer-runner
  layer with NO `sim/` edit, the moat tightened not weakened. The read-side "one self" hook (Option A) is
  de-risked GO.
- **Isn't:** (i) patient-read sharpening -- the gate keys on the cue role, and FHRR patient noise is independent
  (a real, reported boundary). (ii) The ENCODING hook (Option B, the Lisman-Grace novelty-gated write /
  reconsolidation labilization) -- a ranked FOLLOW-ON with a two-directional (write-fidelity) moat risk, not tested
  here. (iii) The deep RF-dynamics `sim/` edit (a continuous DA-modulated resonate gain, scoping doc Sec.7) --
  deferred; the discrete cleanup-margin gate suffices for this hook.
- **Scale:** validated at D=64, 8 facts, the FHRR-cleanup-with-noise harness (faithful to the production
  `confidence_gate` mechanism). The next step is to wire the composer-runner DA read into the merged-bridge
  conversational path (Option A production), reuse-by-import, NO `sim/` edit -- and optionally add the zero-cost
  Option-C dlPFC `excitability_drive` `ModulatorTarget` (which already rides the existing NM path).

---

## EXACT NEXT

Option A is GO -> wire the composer-runner read (`get_concentration("dopamine")` -> `confidence_gate`, clamped to
sharpen + the inverted-U `g_cap` ceiling) into the merged-bridge conversational path (reuse-by-import, NO `sim/`
edit). Carry the zero-cost Option-C dlPFC gain `ModulatorTarget` as a free add-on. Defer Option B
(encoding/reconsolidation gating) and the deep RF-dynamics `sim/` edit (scoping Sec.7) as ranked follow-ons.
