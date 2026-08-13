---
type: finding
status: live
lane: A (affect / emotion keystone)
date: 2026-08-13
mechanism: spiking-appraisal-discrete-emotion-reappraisal
---

# Spiking appraisal → discrete emotion attractors + a vmPFC reappraisal gate — de-risk (faculty-map T1-5)

**Status:** 6-seed GO (gate's own verdict positive; all 8 checks pass on seeds 42/43/44/100/101/102, GPU/cupy, 38s).
**Scope:** runner-only de-risk (NOT wired to production; NOT "closed"). NO `sim/` edit; additive; reads off `cp_firing_states`.
**Runner:** `research/runners/_affect_appraisal_emotion_reappraisal_derisk.py`
**Raw:** `research/findings/raw/_affect_appraisal_emotion_reappraisal.json` (smoke) · `..._6seed.json` (6-seed).

## 6-seed result (means; per-seed spread)

<!--derived-->
<!-- numbers below are rounded aggregate MEANS + per-seed spans computed over the 6 seeds; the full-precision values
     live in the cited ..._6seed.json "means"/"per_seed" (means over seeds live in no single per-seed file). -->
RUNG (a): held-out opponent **r=+0.745** (per-seed 0.686–0.816); `|differential|`~valence-strength **r=+0.270**;
shuffle code↔word collapses to **−0.050**; input-lesion collapses the differential to **0.000** (vs intact 0.019).
RUNG (b): emotion discrimination **0.833** (per-seed 0.75–1.00, every seed ≥ the 0.75 bar); MISMATCHED-appraisal
control **0.333** (~chance) vs intact 0.833; reappraisal down-regulates the amygdala **83%** (reap-lesioned **−0.2%**);
WTA-lesion collapses the categorical margin **0.230 → 0.135**. The one condition most often missed is RAGE (the
smallest margin) — every seed still clears ≥ 3/4.

## Why

Today's wired affect (`research/runners/affect_production_organ.py`) is a signed SCALAR: a bistable valence ladder
whose per-word VALUE is DR-2-learned but whose SALIENCE GATE (which words move the mood) + seed norms are still the
HOST Warriner lexicon (`appraise_text`: `w in WARRINER and abs(v9-5) >= 2.0`), with NO discrete emotions, NO appraisal
structure, NO reappraisal. This de-risks the two named next rungs on ONE co-resident spiking bridge.

## Rung (a) — spiking opponent that reads valence FROM THE SUBSTRATE, retiring the host salience GATE

<!--derived-->
<!-- the "Smoke (seed 42…)" numbers below are the rounded per-seed values in the cited smoke JSON per_seed[0]. -->

A concept's DR-2 LEARNED co-occurrence CODE (PPMI over hubs, the substrate's own word representation; reuse
`_affect_distributional_tag_derisk.build_cooccurrence` / `codes_from_cooccurrence`) is presented as sensory drive to a
`code_in` relay. Its firing is carried by SYNAPSES — a learned rectified-opponent feedforward (Namburi-Tye V+/V-,
ridge-fit `w`, split `W+ = g·max(w,0)`, `W- = g·max(-w,0)`) — to two opponent pools `appr_vplus`/`appr_vminus` that
cross-inhibit. The appraisal is the SPIKING differential `rate(vplus) − rate(vminus)` read off `cp_firing_states`. The
salience gate is now EMERGENT: a word moves the mood iff its opponent differential clears a magnitude — the
`|differential|` tracks valence STRENGTH — NOT a lexicon-membership + `|v-5|>=2` test.

Smoke (seed 42, 8k-story corpus): held-out concepts (never in the ridge train split) appraise to a spiking differential
correlating **r=+0.760** with true signed valence; `|differential|` tracks valence strength **r=+0.257**; the input-
lesion collapses the differential to **0.000** (vs intact 0.015); shuffling which code belongs to which word collapses
the read to **−0.253**.

## Rung (b) — multi-dimensional appraisal → 4 discrete emotion categories + a vmPFC→amygdala reappraisal gate

<!--derived-->
<!-- the "Smoke (seed 42)" numbers below are the rounded per-seed values in the cited smoke JSON per_seed[0]. -->

Appraisal DIMENSION pools — valence (the rung-a opponent, load-bearing), agency (`agency_self`/`agency_other`),
certainty (`certainty`/`uncertainty`) — converge via WIRED excitatory + inhibitory projections (the Scherer/OCC/Barrett
appraisal STRUCTURE) onto FOUR Panksepp primary-process EMOTION categories {SEEKING, CARE, FEAR, RAGE} that compete in
a shared-FS Wong-Wang WTA (the project's validated concept-pool WTA biology). Each emotion has a symmetric 2-excitatory
signature (valence + one distinguishing dim); the incongruent dim actively INHIBITS the same-valence rival via a per-dim
inhibitory relay (FS-sourced, so an excitatory dim pool can inhibit); the opponent cross-FS enforces the valence
opposition. A `vmpfc_reap` pool drives an inhibitory relay `reap_fs` onto `appr_vminus` (the "amygdala") — the
Ochsner-Gross cognitive-reappraisal down-regulation. The winner (argmax pool rate off `cp_firing_states`) is the
discrete emotion.

Smoke (seed 42): the 4 appraisal conditions select their intended emotion at **accuracy 1.00** (4 distinct winners); a
MISMATCHED-appraisal control (a condition's valence with a different condition's dims) collapses accuracy to **0.25**
(chance) — the winner is determined by the appraisal STRUCTURE, not valence alone or a fixed pool. Engaging `vmpfc_reap`
on a NEGATIVE condition down-regulates `appr_vminus` by **82%** (reap-lesioned **−4%** ≈ 0, so the down-regulation is
carried by the reappraisal projection). Lesioning the WTA cross-inhibition collapses the categorical margin
**0.250 → 0.130**.

## Pre-registered GO gate (all 8 pass on the smoke)

<!--derived-->
<!-- thresholds/decision-rule constants (in the runner's aggregate()), not measurements. -->

A1 held-out opponent r ≥ 0.45 · A2 `|differential|` tracks valence-strength (r>0.2) AND input-lesion collapses ·
A3 shuffle code↔word collapses · B1 emotion discrimination ≥ 0.75 AND ≥3 distinct winners · B2 reappraisal
down-regulates the amygdala ≥ 25% · B3 WTA-lesion collapses the margin ≥ 35% · B4 reap-lesion abolishes the
down-regulation · B5 mismatched appraisal collapses discrimination to ~chance.

## Brain-based statement

The appraisal READ (opponent differential) and the emotion SELECTION (shared-FS WTA winner) are spike-rate reads off
`cp_firing_states`; the salience gate is emergent; reappraisal is a spiking inhibitory projection. This is NOT claimed
"fully spiking": the residuals below are host.

## HONEST RESIDUALS (declared)

1. The opponent weights are ridge-fit in numpy (a host readout of the DR-2 learned code) and SEEDED from Warriner norms
   — the seed supervision is NOT retired; only the salience GATE + the READ move to the substrate.
2. The appraisal DIMENSION conditions (agency, certainty) are set as sensory drive by the environment/teacher (the
   situation the brain is in); the appraisal COMPUTATION dims→emotion is spiking.
3. Emotion pools are RELAY (shared-FS competitive WTA), not self-sustaining attractors: NMDA self-recurrence
   self-ignites from noise at every bias tried. With parameter heterogeneity ON there is a resting spontaneous "default"
   activation that appraisal OVERRIDES (4/4 correct) but does not eliminate — the named next rung is a latching
   attractor with an anti-self-ignition homeostatic quiescence set-point (the missing companion process: a
   spike-frequency-adaptation / homeostatic set-point the animal runs alongside the WTA that we proxied with a static
   bias).
4. Runner-only: folding the slice into `build_one_brain` (like the ladder) and reaching it from `/api/brain-chat` is
   the production-integration step; this de-risk does not wire it.

## Operating point (grid-searched, generalizes seeds 42+43, not overfit)

<!--derived-->
<!-- circuit constants (in the runner), not measurements. -->

The decisive tuning was: the valence weight is kept BELOW the distinguishing-dim weight (VAL_TO_EMO_W=20 < W_MAP=34),
so the agency/certainty dim — not the shared valence — TIPS the winner within a same-valence pair; a graded rung-a read
scale (OPP_READ_SCALE=800) so the opponent magnitude tracks valence strength; read window [30:110] after the initial
burst/adaptation transient.
