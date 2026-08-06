---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency-v7
runner: research/runners/_laneC_source_monitor_coresidency_gate_v7.py
aggregator: research/runners/aggregate_source_monitor_v7_seeds.py
artifacts:
  - research/findings/raw/source_monitor_v7_generalization/development_verdict.json
  - research/findings/raw/source_monitor_v7_generalization/development_652.json
  - research/findings/raw/source_monitor_v7_generalization/development_653.json
  - research/findings/raw/source_monitor_v7_generalization/development_654.json
---

# v7 development NO-GO: region-scoped threshold homeostasis breaks the fixed GABA-A competition

<!--derived-->
**Verdict: NO-GO at v7 development.** v7 = v6 UNCHANGED (silent-by-construction recall, fixed local fast-spiking GABA-A
competition, all twenty frozen components + thresholds) PLUS the shipped region-scoped intrinsic threshold homeostasis
(Turrigiano; the same mechanism v3 used) on the source-memory populations, run in a learning-off settling window to
up-regulate the weakest source. On dev seeds 652/653/654 (aggregate
`research/findings/raw/source_monitor_v7_generalization/development_verdict.json`) all three `DEVELOPMENT_FAIL`. The
named surpass does not merely miss the weakest-source criterion — it collapses the whole discrimination: enabling the
competition circuit on homeostasis-masked source pools drives every intact margin far BELOW its competition-lesion
value. Held-out seeds 655/656/657 stay sealed (`validate_phase_seed` opens them only on a dev GO) and were NOT run.

## The failure, quantified

<!--derived-->
`M` = intact margin (competition ON), `L` = matched competition-lesion margin (competition OFF, same trained + settled
network), floor `F = 0.15`. Homeostasis was demonstrably active every seed (threshold L1 shift 44–111 mV across the 36
masked neurons). Silent-by-construction held (`learning_off` source spikes = 0 on all seeds); settle reached quiescence.

| seed | M seen/heard/self | L seen/heard/self | min M | min L | competition gains | status |
|---:|---|---|---:|---:|---|:---:|
| 652 | .120/−.041/.443 | .424/.466/.443 | −.041 | .424 | s −.304, h −.507, sg 0 | DEVELOPMENT_FAIL | <!--derived-->
| 653 | .053/.024/.048 | .442/.398/.448 | .024 | .398 | s −.389, h −.374, sg −.400 | DEVELOPMENT_FAIL | <!--derived-->
| 654 | .068/.015/.449 | .449/.422/.449 | .015 | .422 | s −.382, h −.407, sg 0 | DEVELOPMENT_FAIL | <!--derived-->

<!--derived-->
Failing components (all seeds): `all_source_margins_meet_fixed_floor` (min M ≪ 0.15, even negative on 652),
`bounded_loss_only_spends_surplus` (competition now costs far more margin than the surplus above the floor), and the
target criterion `weakest_source_margin_strictly_improved` (min M < min L). Seeds 652 and 654 also lost
`source_swap_follows_afferent_activity`; 652 lost `heard_source_recalled`. On seed 654 — the exact v6 failure — the
weakest source `heard` goes from a v6 tie (M = L = .1825) to M = .015 vs L = .422: competition on the up-regulated pool
DESTROYS it (gain −.407), the opposite of the intended rescue.

## Why (mechanism-level)

<!--derived-->
The shipped region-scoped homeostasis works by masking the source-memory neurons so their spike is detected at the
adapted sub-threshold voltage (−55..−30 mV) instead of the Izhikevich peak (`sim/bridge.py:8893`; the masked branch
uses `cp_neuron_firing_thresholds`). Switching the competing pools to sub-threshold detection is INCOMPATIBLE with the
v6 fixed GABA-A biased-competition circuit, which was calibrated for peak detection: with competition ON the intact
margins collapse to ~0.03 (uniform firing across pools) while competition OFF stays clean at ~0.41 — competition now
EQUALIZES rather than sharpens (consistent with post-inhibitory rebound of the low-threshold rival pools re-crossing
the detection voltage). A calibration-seed characterization (650/651 × six operating points, adapt_rate 5e-4…2e-2)
showed the collapse is present at EVERY operating point, including v3's near-inert canonical rate, so it is STRUCTURAL —
a property of the masking, not of the homeostatic strength or the settling schedule. The operating point was
characterized and frozen on the calibration seeds; the dev seeds were run once for the recorded verdict, not tuned.

## This re-diagnoses the v3 NO-GO

<!--derived-->
v3 (2026-08-03-source-monitor-coresidency-v3-calibration-NO-GO) reported minimum source margins of 0.0016 and 0.0091
and concluded "local threshold homeostasis produced zero margin gain." Those near-zero margins are the SAME
competition-vs-masking collapse measured here, not an inert homeostasis: v3 already ran the masked pools under the v6
competition and mis-attributed the flat margin to the stabilizer doing nothing. The correct diagnosis is that masking
the competing pools for homeostasis and running the fixed GABA-A competition on them cannot both hold.

## Next mechanism (no-defer; a new method for the SAME frozen criterion)

<!--derived-->
Up-regulate the weakest source with a homeostatic mechanism that does NOT change the spike-detection basis of the
competing pools — **Turrigiano synaptic scaling**: multiplicatively up-regulate the under-active source's
episode→source recall synapses (or its source-afferent gain) toward an activity set-point, leaving the source-memory
neurons at peak detection so the v6 competition keeps functioning. Synaptic scaling is the canonical homeostatic-
plasticity sibling of intrinsic excitability and directly strengthens the weakest source's recall drive, which is what
the `weakest_source_margin_strictly_improved` criterion certifies. This is a v8 method; the frozen criterion and
thresholds are NOT loosened. (Intrinsic-excitability homeostasis is a wall for THIS competition circuit, not the
capability: an excitability homeostat co-designed with a competition circuit that survives sub-threshold detection is a
separate, later method.)

## What was NOT done, on purpose

<!--derived-->
No frozen criterion was relaxed, and the twenty v6 components + thresholds are byte-for-byte the v6 set (the homeostasis
integrity checks are recorded as preconditions, not as new components, so the aggregate still requires exactly twenty).
Held-out seeds remain sealed. The NO-GO is a verdict on the METHOD (merging the shipped region-scoped threshold
homeostasis with the fixed GABA-A competition), not on the capability of protecting the weakest source.

## Provenance

All three development seeds ran locally on the NumPy backend, deterministic across re-runs. Runner
`research/runners/_laneC_source_monitor_coresidency_gate_v7.py`; aggregator
`research/runners/aggregate_source_monitor_v7_seeds.py`; artifacts and `.prov.json` sidecars listed in the front
matter, stamped from git `b89c3edc0a3e6c7a980de38573377479b68187dc`.
