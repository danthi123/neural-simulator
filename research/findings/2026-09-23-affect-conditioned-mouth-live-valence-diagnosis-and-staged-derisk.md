---
type: finding
status: partial
claim_check: measured
date: 2026-09-23
lane: A · Affect — D5 "feel" over the OPEN reply (roadmap §8)
seeds: [42, 43, 44, 100, 101, 102]
mechanism: webapp/affect_conditioned_mouth.py (BRAIN_OPEN_ENDED_AFFECT_CONDITIONED=prompt|resid, default-OFF) conditions
  the Qwen articulation mouth's GENERATION on the spiking affect organ's held valence. 'prompt' = graded,
  dead-zone-free MOOD line in the existing build_prompt channel; 'resid' = c*K*u added to Qwen's layer-12 residual
  stream (contrastive activation addition), u = the mouth's own WARRINER-contrast axis.
---

# Affect-conditioned Qwen mouth — why the brain's valence never reached the mouth, and the staged 6-seed de-risk (2026-09-23)

Status: **partial**. The diagnosis below is measured on all 6 seeds. The tone verdict for the new method is
**not measured yet**: the two 36-arm runs are staged (see "Staged runs"). This doc preregisters their gate.

## What was measured: the live-valence magnitude problem has two parts

<!--derived from research/findings/raw/_affect_conditioned_mouth/magnitude_audit.json + research/findings/raw/_affect_conditioned_mouth/calibration/calibration_summary.json -->
**1. The mouth's own conditioning channel had a dead zone that swallowed every live read.** The production
`_mood_phrase` renders "even and steady" for |valence| < 0.25. The live valence is `clip(4 * differential)`.
It read +0.141..+0.162 after positive priming and -0.180..-0.073 after negative priming. So on **12/12** pos/neg  <!--derived-->
arms of the committed NO-GO data, Qwen's MOOD line read "even and steady". The brain's affect reached the Qwen
prompt only as a two-decimal number. Nothing in the prompt's words carried the mood.

<!--derived from research/findings/raw/_affect_conditioned_mouth/calibration/calibration_summary.json (per-seed files organ_calib_s<seed>.json alongside) -->
**2. The organ is not weak. It runs at about half its range after one priming message.** The affect organ is
690 neurons. It was swept over appraisal -1..+1 on each seed (`--calibrate-organ`, numpy). Results:
- Full scale is symmetric. The held differential at |appraisal| = 1 is +0.075..+0.086 and -0.076..-0.093.  <!--derived-->
  VALENCE_FS = 4 x the mean full-scale differential = 0.309..0.341 across seeds.
- The ladder has a threshold dead zone. At |appraisal| <= 0.25 no rung ignites and the differential is 0.0.
- One strongly affective message moves the session-mood EMA to appraisal ±0.475. That is where the live read
  sits: 0.449..0.525 of full scale for positive priming, 0.237..0.528 for negative priming.
- The negative-side weakness is a MID-RANGE effect, not a ceiling. At appraisal -0.475, 4/6 seeds  <!--derived-->
  (43, 100, 101, 102) hold only -0.018..-0.024. That is below the organ's own neutral tolerance of 0.03.  <!--derived-->
  At -0.75 and -1.0 the same seeds read -0.058..-0.082.  <!--derived-->

The calibration reproduces the production read. The sweep point at ±0.475 matches the NO-GO arms' priming
differential to 4 decimals on the positive side for all 6 seeds. On the negative side it matches to within the
organ's own read noise.

So the earlier description "~0.16 clipped" was wrong on both words. Nothing was clipped, and 0.16 is about half
the organ's range, not a tiny signal. What held affect back was the MOUTH's dead zone, plus a single-message
EMA that holds the organ in its mid-range. The companion process the NO-GOs lacked is **readout gain
normalization**: a reader scaled to the upstream population's operating range. In cortex that is divisive
normalization (Carandini & Heeger 2012). The production path used a fixed x4 constant, and then threshold 0.25
on top of it.

## The method (built, default-OFF)

The conditioning signal is `c = clip(valence / VALENCE_FS, -1, 1)`. `valence` is exactly what brain_chat
already hands the mouth, the organ's held differential. VALENCE_FS is that seed's measured full-scale value.
One constant is used for both signs, so the organ's mid-range sign asymmetry is kept, not normalized away. Live
c is about +0.45..+0.53 (pos) and -0.24..-0.53 (neg).

- **prompt**: the MOOD line is graded monotonically in |c| (faintly / slightly / moderately / clearly /
  intensely) and has no dead zone. c == 0 gives the production neutral wording. No descriptor word is in the
  scoring lexicon ('upbeat', a scored word, became 'buoyant'), so the ruler cannot score an echo of the
  conditioning text.
- **resid**: the prompt stays exactly as production builds it. During generation a forward hook adds `c*K*u` to
  the output of decoder layer 12. u is the difference of mean layer-12 activations of Qwen on 6 positive vs 6
  negative sentences built only from WARRINER words; the selftest enforces zero overlap with the scoring
  lexicon. K = 4.0 is fixed. When c == 0 no hook is registered.

Brain-based boundary: the conditioning SIGNAL comes from the spiking organ. The division by VALENCE_FS is host
code, a **named shortcut** on the scaffold boundary. The Qwen mouth, its prompt and its internal axis u are the
owner-ratified articulation scaffold (2026-09-19), not brain computation.

Off path, checked in data: `tests/test_affect_conditioned_mouth.py` uses a fake generator. With the flag unset,
answer_turn hands the generator exactly build_prompt's (system, user) and never imports the module.

## PREREGISTERED GO gate for the staged runs (written before any conditioned-mouth tone result existed)

This is the NO-GOs' gate verbatim, via `score_and_gate(mouth="qwen")`:
(1) DIRECTIONAL. On each seed pos_gap > +delta AND neg_gap < -delta, on 6/6 seeds, or 5/6 with the 6th null.
delta is the pooled std of the lesion arm's per-prompt tone, with the independent 356-word lexicon.
(2) ATTRIBUTION. |ctrl_pos - ctrl_neg| < delta on every seed. The shuffled-valence control injects ±0.16
random-sign valence.
(3) Content identity holds and the moat holds.
(4) Fluency: salad <= 0.16.
(5) lesion == lesion_rep, byte-identical.

Plus two instrument preconditions. Either one unmet makes the verdict UNDEFINED, not a negative:
(L) the conditioning lever moved: c != 0 on pos/neg/ctrl rows, the hook fired in resid mode, and c == 0 on lesion
rows;
(O) each arm's priming differential equals the committed NO-GO arm's value, so all three verdicts see the same
brain state.

Literal scoring command:
`.venv/bin/python -m research.runners._lbf_affect_conditioned_mouth_derisk --score-only --mode <prompt|resid>`

## Staged runs

One gpu_queue line per (mode, seed). Qwen runs on CUDA; the brain runs numpy, the same organ numerics as the
NO-GOs. Arms run one after another inside each line, so only one brain is live at a time. Results go to
`research/findings/raw/_affect_conditioned_mouth/{prompt,resid}/arm_s<seed>_<arm>.json`, and the verdict to
`affect_conditioned_mouth_<mode>_verdict.json`.

## Honest residuals
- The tone result does not exist yet. This doc makes no claim that affect is load-bearing over the open reply.
- The normalization is host arithmetic, a named shortcut. The brain-based version would be a gain-adapting
  reader population.
- Qwen generation uses the server's fixed seed 42 in every arm. Seeds vary only the organ, which is the
  conditioning signal. The mouth itself is seed-invariant, as in the linattn NO-GOs.
- The salad<=0.16 precondition was designed for WKV salad. First-person Qwen prose repeats "I", so short
  replies could trip it. If it does, the verdict is UNDEFINED and the instrument, not the method, needs a fix.
- The organ's mid-range V- weakness (4/6 seeds below tolerance at -0.475) may still cap neg_gap. If so, the  <!--derived-->
  surpass is on the organ/appraisal side (the session-mood EMA protocol or the V- rung-2 ignition), not the
  mouth.

Honesty boundary: this is a functional read-out only. Reply tone tracking the organ's valence is a coupling.
Nothing here claims felt experience.
