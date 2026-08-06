---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency-v8
runner: research/runners/_laneC_source_monitor_coresidency_gate_v8.py
aggregator: research/runners/aggregate_source_monitor_v8_seeds.py
artifacts:
  - research/findings/raw/source_monitor_v8_generalization/development_verdict.json
  - research/findings/raw/source_monitor_v8_generalization/development_652.json
  - research/findings/raw/source_monitor_v8_generalization/development_653.json
  - research/findings/raw/source_monitor_v8_generalization/development_654.json
  - research/findings/raw/source_monitor_v8_generalization/calibration_characterization.json
---

# v8 NO-GO: Turrigiano synaptic scaling toward an activity set-point COMPRESSES the discrimination margins

<!--derived-->
**Verdict: NO-GO at v8 (fails at calibration AND development).** v8 = v6 UNCHANGED (silent-by-construction
settle-to-quiescence recall, fixed local fast-spiking GABA-A competition, all twenty frozen components + thresholds)
PLUS the shipped Turrigiano synaptic scaling (`cfg.enable_synaptic_scaling`; Turrigiano 2008, Cell 135:422) run in a
learning-off settling window to multiplicatively up-regulate the under-active source's `episode→source` recall
synapses toward an activity set-point. Because scaling touches `cp_connections` (real synapse weights) and NOT the
firing thresholds, the source pools stayed at PEAK Izhikevich-spike detection and the v6 competition ran unchanged —
the v7 sub-threshold-masking failure was avoided by construction (verified every seed:
`homeostasis_mask_stays_none=True`, `source_thresholds_unchanged=True`, `non_source_weights_unchanged=True`). But the
mechanism does not deliver the capability: on dev seeds 652/653/654 all three `DEVELOPMENT_FAIL`
(`research/findings/raw/source_monitor_v8_generalization/development_verdict.json`). Held-out seeds 655/656/657 stay
sealed (`validate_phase_seed` opens them only on a dev GO) and were NOT run.

## The failure, quantified

<!--derived-->
`M` = intact margin (competition ON), `L` = matched competition-lesion margin, floor `F = 0.15`. Scaling ran and was
correctly scoped every seed (`weight_delta_l1 ≈ 4.1e4`, non-source weights byte-identical). The margins after scaling:

| seed | min M | min L | strict M>L | floor met | failing components | status |
|---:|---:|---:|:---:|:---:|---|:---:|
| 652 | .1325 | .1325 | no (tie) | no | floor, strict-improvement | DEVELOPMENT_FAIL | <!--derived-->
| 653 | .1358 | .1192 | yes | no | floor | DEVELOPMENT_FAIL | <!--derived-->
| 654 | .1292 | .1292 | no (tie) | no | floor | DEVELOPMENT_FAIL | <!--derived-->

<!--derived-->
The decisive control is the CALIBRATION characterization (`calibration_characterization.json`): v6 passes both
calibration seeds (650 min M = .1767, 651 min M = .1667, both above the .15 floor), but v8 scaling — at EVERY operating
point swept (target set-point 0.02…0.5 × scaling rate 0.02/0.05) — COMPRESSES the margins below the floor. Seed 651 is
the proof: v6 min M = .167 (floor met, strict met); v8 min M saturates at .134 at every set-point, BELOW the floor.
No operating point makes both calibration seeds meet floor + strict + competition-active
(`any op-point passes both = False`). Scaling also starved the competition interneurons on 651 (source firing dropped
enough that `competition_circuit_is_active_and_lesionable` failed).

## Why (mechanism-level)

<!--derived-->
Multiplicative synaptic scaling drives every source-memory neuron's incoming recall weight toward a COMMON firing
set-point, so it EQUALIZES per-source firing rate. But the acceptance criterion measures per-source discrimination
MARGIN — the CONTRAST between the correct source and its rivals during the correct recall. Equalizing firing rates
COMPRESSES that contrast: pulling the strong source down and the weak source up toward one rate reduces how much each
source dominates its rivals when its own pattern is recalled, so ALL margins shrink toward the middle (651: .167 → .134
at every set-point). Two further consequences: (1) the "under-active source" scaling targets (by firing rate) is not
the "weakest-margin source" the criterion certifies — on the dev seeds the weakest-by-firing source's weight SHARE even
fell slightly (`weakest_source_share_gain` ≈ −0.003…−0.012); (2) the 5000 pA episode drive pushes source firing above
any reasonable per-step set-point, so scaling moves NET DOWNWARD, cutting the source drive that activates the
competition interneurons. The margin and the activity set-point are orthogonal quantities; a homeostat that defends a
firing rate cannot, in principle, defend a between-source contrast.

## This maps the wall precisely

<!--derived-->
v6 (fixed symmetric GABA-A competition) was calibration-GO but generalization-NO-GO: competition lifted the
second-strongest source, not the weakest. v7 (intrinsic THRESHOLD homeostasis) was a worse NO-GO: masking the pools for
homeostasis switched them to sub-threshold detection, collapsing the competition. v8 shows the third homeostatic
sibling — synaptic scaling — is also the wrong tool, for a DIFFERENT reason: it operates on the wrong quantity (firing
rate, not contrast) and so compresses the very margin it was meant to protect. All three homeostatic mechanisms defend
an ACTIVITY level; none defends a DISCRIMINATION margin.

## Next mechanism (no-defer; a new method for the SAME frozen criterion)

<!--derived-->
Target the CONTRAST, not the firing rate. The weakest-margin source loses because its self-advocacy against rivals is
weak — so up-regulate the INHIBITORY competition, not the excitatory recall drive. **Vogels–Sprekeler inhibitory
synaptic plasticity (ISP; Vogels et al. 2011, Science 334:1569)** on the `interneuron→rival` GABA-A synapses: the
under-margin source's interneuron scales its rival-inhibition toward a target E/I balance, so when that source's pattern
is recalled it suppresses rivals MORE, raising its margin (a between-source contrast) WITHOUT compressing the excitatory
code or starving the interneurons. This is still a synaptic homeostat on real synapses, but scoped to the competition
(inhibitory) pathway, which is what the margin criterion actually rewards. A second candidate is a BCM sliding-threshold
selectivity rule on the recall synapses (increase selectivity of the low-margin source rather than equalize its rate).
This is a v9 method; the frozen criterion and thresholds are NOT loosened. (Synaptic scaling of the EXCITATORY recall
drive is a wall for THIS margin criterion, not for the capability.)

## What was NOT done, on purpose

<!--derived-->
No frozen criterion was relaxed; the twenty v6 components + thresholds are the byte-for-byte v6 set (the scaling
integrity checks are recorded as preconditions, not new components, so the aggregate still requires exactly twenty).
Held-out seeds remain sealed. The scaling operating point (target 0.02, rate 0.02, 5000-step settle) was characterized
and frozen on the calibration seeds; the dev seeds were run once for the recorded verdict, not tuned. The NO-GO is a
verdict on the METHOD (multiplicative synaptic scaling of the excitatory recall synapses toward an activity set-point),
not on the capability of protecting the weakest source's margin.

## Provenance

All three development seeds and the calibration characterization ran locally on the NumPy backend, deterministic across
re-runs. Runner `research/runners/_laneC_source_monitor_coresidency_gate_v8.py`; aggregator
`research/runners/aggregate_source_monitor_v8_seeds.py`; synaptic scaling scoped via the shipped per-synapse
plasticity-gain gate with NO `sim/` edit. Artifacts and `.prov.json` sidecars are listed in the front matter.
