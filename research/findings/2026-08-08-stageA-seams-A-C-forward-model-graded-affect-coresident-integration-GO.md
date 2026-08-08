---
type: finding
status: qualified
date: 2026-08-08
mechanism: stageA-seam-integration-forward-model-reservoir-and-graded-affect-ladder
lane: A (affect / world-model keystone) + world-model
artifacts:
  - research/findings/raw/lanes/stageA/seam_A_probe_s42.json
  - research/findings/raw/lanes/stageA/seam_A_probe_s43.json
  - research/findings/raw/lanes/stageA/seam_C_probe_s42.json
  - research/findings/raw/lanes/stageA/seam_C_probe_s43.json
  - research/findings/raw/_consolidation_probe1_byteident.json
  - research/findings/raw/_affect_graded_ladder_6seed_smoke.json
---

# Stage-A seams A + C integrated onto the ONE merged bridge — forward-model reservoir + graded-affect ladder, co-resident, default-off byte-identical, regression stays GO

Two adversarially-verified faculties are wired onto the TRUE-ONE-BRAIN conversation composer
(`_stageA_full_integration_derisk.build_one_brain`) as APPENDED-LAST co-resident region slices, ONE seam at a time,
each **default-off + byte-identical-when-off**, each proven to ENGAGE when on, and each moat/FM4-safe by construction.
The banked 6-seed TRUE-ONE-BRAIN regression stays GO after both seams (default path byte-unchanged).

## SEAM-A — forward-model reservoir (OnBridgeLSM) → world-model → certainty-band + content seam

`co_resident_forward_model=False` default. When on, ONE recurrent Izhikevich `fm_reservoir` slice (300 neurons,
`internal_density=0` in cfg so the shared wiring plan draws no fm-internal rng; the fixed-random LSM recurrence
density 0.1 is injected as a SEPARATE union entry with an independent rng) is appended LAST with NO out-edges
(nav/conv-inert). Per `(s,a)` token the agent writes `W_in @ U[t] + BIAS` into `cp_external_input_current[fm_idx]`,
runs the bridge's real step loop, and accumulates `cp_firing_states[fm_idx]` → per-neuron spike-COUNT (the read-out
feature); a masked snapshot restore washes the fm slice between reads.

Routes (moat-safe): (i) the read-out MARGIN feeds `g_eff = max(g_eff, g0 + k_fm*(1 - margin))` — a LOW forward-model
margin can only RAISE g_eff (tighten abstention), NEVER lower it (the `da_to_gate` clamp discipline); (ii) the decoded
s' enters render ONLY as a certainty-TAGGED "predicted, not observed" channel — NOT written to `cp_rf_w_*`, NOT added
to the cue-match candidate set → the no-confab moat's HARD structural floor is untouched.

Declared host shortcut: the ridge `argmax(spikecounts @ Ws)` decode is the read-out (identical in status to the
composer's numpy render + OnBridgeLSM's own `_fit_slots`); the brain-based content is the reservoir SPIKES.

<!--derived-->

| seam-A check (seeds 42, 43) | result |
|---|---|
| byte-identity (thresholds + conn prefix, ladder off vs on) | True / True (n 24961 → 25261, +300) |
| at-rest neutrality (silent reservoir → g_eff = g0 floor 0.06) | True / True |
| engages (reservoir active; ridge decode test acc vs chance 0.25) | 1.00 / 1.00 |
| certainty band tightening-only (g_eff low-margin 0.36 ≥ high-margin 0.06 = floor) | True / True |
| masked-reset isolation (nav/conv v & u byte-untouched) | True / True |

Faculty-A co-resident smoke (`_emerge82_onbridge_lsm_derisk --derisk`, 6-seed): GO (reservoir learns form→role
train 1.000, resolves the non-local relative-clause head 1.000 vs chance 0.500; region-silence lesion collapses to
0.500). Byte-identity oracle `_consolidation_probe1_byteident`: GO (matrix 30/30, abstain 8/8, moat preserved,
nav-slice byte-identical, max|dmem|=0).

## SEAM-C — staggered bistable ladder graded affect → affect-coloring seam

`co_resident_affect_ladder=False` default, `aff_n_rungs=8`. When on, the P0.3 single-pool good/bad latch is
augmented by a Koulakov-2002 robust-discrete integrator: N self-recurrent slow-NMDA sub-pools per valence sign
(`aff_vplus_L1..LN` / `aff_vminus_L1..LN`), each latched by its OWN within-pool NMDA recurrence (density 0.8 injected
as independent union entries), recruited at STAGGERED intrinsic-excitability offsets by a UNIFORM diffuse appraisal
broadcast (`excitability_drive`, group-scoped per sign). Opponent cross-inhibition ONLY at the AGGREGATE
(`aff_agg_plus`/`aff_agg_minus`); NO intra-sign lateral inhibition (the load-bearing rule — else the ladder collapses
to the 2-level latch). Held value = number of latched rungs, read NEURALLY as
`rate(aff_pos_readout) - rate(aff_neg_readout)` through the SAME `affect_out` transmission gate the P0.3 organ uses.

FM4 safety is STRUCTURAL: `affect_out` is array-disjoint from `g_eff` (`_da_confidence_gate`) and from the cue-match
moat gate → graded affect colors tone WITHIN the already-decided band and can NEVER flip abstain→assert. Lesion =
`set_transmission_gate("affect_out", 0.0)` collapses the read.

Byte-identity is preserved the SAME way as seam-A: ladder sub-pools carry `internal_density=0` (recurrence injected
independently); the ladder regions + ladder pathways are appended LAST so every pre-existing rng draw is unchanged;
per-region homeostasis stays OFF (enabling it would draw init-time RNG before the threshold draw and shift every
pre-existing threshold — the seam-A measurement, unchanged here).

<!--derived-->

| seam-C check (seeds 42, 43) | result |
|---|---|
| byte-identity (thresholds + conn prefix, ladder off vs on) | True / True (n 24961 → 25531, +570) |
| at-rest neutrality (neutral appraisal → differential ≈ 0) | 0.0000 / 0.0000 |
| graded staircase Spearman ρ (appraisal → held differential) | 1.00 / 1.00 |
| held-differential range (bar ≥ 0.05) | 0.0703 / 0.0675 |
| affect_out lesion collapses the read (FM4 gate is the read path) | True / True |

Faculty-C standalone smoke (`_affect_graded_ladder_derisk --smoke`): pass — staircase ρ=1.00 range 0.068
counts `[0,1,2,3,4,5]`; the single lumped pool AND the unstaggered ladder both collapse to 2 levels; persistence
retention 1.04 (NMDA-on) vs 0.00 (NMDA-off); FM4 abstain-protected True, arousal-gates-vigor True.

The het-ON operating-point risk the design flagged (the merged bridge runs `enable_parameter_heterogeneity=True`,
the faculty calibrated het-OFF) did NOT materialize: the co-resident staircase is perfectly monotone (ρ=1.00) with
range 0.067–0.070 > the 0.05 bar, WITHOUT per-region homeostasis (so byte-identity is preserved). `off_step` is
computed to keep the deepest of 8 rungs above the holding floor for any rung count.

## Regression — the banked TRUE-ONE-BRAIN 6-seed GO stays GO

`_stageA_full_integration_derisk` builds `build_one_brain(seed, with_faculties=True)` WITHOUT the seam flags, so the
default (both seams off) build is byte-identical to the pre-integration file → the regression is unaffected by
construction. Confirmed empirically across the seam-C edit boundary:

<!--derived-->

| 6-seed regression (42 43 44 100 101 102), SIM_BACKEND=numpy | after seam A | after seam C |
|---|---|---|
| seeds GO | 6/6 | 6/6 |
| moat live | 475/475 every seed | 475/475 every seed |
| FM4 g_eff-law abstain→assert flips | 0 every seed | 0 every seed |
| default-off byte-identity | True every seed | True every seed |

Seam-engagement artifacts: `research/findings/raw/lanes/stageA/seam_A_probe_s42.json`,
`research/findings/raw/lanes/stageA/seam_A_probe_s43.json`,
`research/findings/raw/lanes/stageA/seam_C_probe_s42.json`,
`research/findings/raw/lanes/stageA/seam_C_probe_s43.json`; byte-identity oracle
`research/findings/raw/_consolidation_probe1_byteident.json`; faculty-C smoke
`research/findings/raw/_affect_graded_ladder_6seed_smoke.json`.

## Honest boundaries (declared)

- **Host read-outs, not host decisions.** Seam-A's ridge `argmax` decode and seam-C's downstream differential are
  read-outs of brain spikes (the reservoir's `cp_firing_states`, the ladder's `aff_pos_readout`/`aff_neg_readout`
  population rate through `affect_out`), NOT host arbiters of content or the moat — declared exactly as the
  composer's numpy render is. The spiking synaptic read-out is the biologize-later step
  (`_rungB1c_spiking_reservoir_synaptic_readout_derisk.py`).
- **`rf_w` byte-identity sub-check was vacuous:** the composer stores its complex VSA phasor weights under attribute
  names the probe's `_rf_w_sha` did not find (returned None → treated as pass). The MOAT byte-identity is instead
  covered by the regression's live `moat_live` 475/475 (unchanged) and the `_consolidation_probe1_byteident` oracle
  (moat preserved, byte-identical); the threshold + `cp_connections` prefix hashes ARE bit-identical.
- **Seams are AVAILABLE but not yet EXERCISED inside the multi-turn loop.** The regression proves the seams are
  additive/default-off/byte-identical and the loop stays GO; wiring the forward-model certainty band and the graded
  ladder INTO the live turn composition (so a novel-(s,a) prediction and a graded mood actually color a rendered
  turn) is the next integration step. The seams' engagement is proven by `_stageA_seam_integration_probe`
  (seeds 42, 43) and the two faculty smokes.
- Inherited STEP-2/3 boundaries unchanged (host-fed appraisal; host wh/tone render; scaffold burn-down pending).

numpy/CPU (the literal byte claim; on GPU the claim is FUNCTIONAL neutrality per the Closure-1 FP32 note). NO
`sim/` edit — reuse-by-import of BrainRegion/RegionPathway + OnBridgeLSM + the graded-ladder topology; additive
default-off config only. `cfg.seed` seeds the substrate.
