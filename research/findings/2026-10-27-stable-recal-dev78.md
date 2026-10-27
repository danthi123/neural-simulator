---
type: finding
status: qualified
date: 2026-10-27
lane: substrate (integrator validity); follow-up of research/findings/2026-10-26-stable-integration-dev7.md
mechanism: SIM_STABLE_RECAL=1 (research/runners/stable_recal.py), only with SIM_STABLE_INTEGRATION=1, re-calibrates the GNW ignition-bus workspace, the thought-swap workspace and (addendum, 3bc2ae085) the global-stop distributed-overwrite workspace by their own rules on valid membranes; default unset = unchanged
flip: no - default OFF; dev seeds only
seeds: [7, 8]
runner: research/runners/membrane_census.py
artifacts:
  - research/findings/raw/_stable_recal/swap_calibration_record.json
  - research/findings/raw/_stable_recal/swap_ignition_curve_s7.json
  - research/findings/raw/_stable_recal/swap_ignition_inhibition_s7.json
  - research/findings/raw/_stable_recal/swap_ignition_curve_w6p5_nws4_thal25_s7.json
  - research/findings/raw/_stable_recal/swap_ignition_curve_w6p5_nws4_thal25_s8.json
  - research/findings/raw/_stable_recal/swap_sustain_nws/nws4_s7.json
  - research/findings/raw/_stable_recal/swap_sustain_nws/nws4_s8.json
  - research/findings/raw/_stable_recal/swap_sustain_nws/nws5_s7.json
  - research/findings/raw/_stable_recal/swap_sustain_nws/nws6_s7.json
  - research/findings/raw/_stable_recal/swap_sustain_nws/nws8_s7.json
  - research/findings/raw/_stable_recal/swap_sustain_nws/nws16_s7.json
  - research/findings/raw/_stable_recal/swap_thal_support_w6p5_nws4_s7.json
  - research/findings/raw/_stable_recal/swap_thal_support_w6p5_nws4_s8.json
  - research/findings/raw/_stable_recal/swap_primitives/w_gate_ws_s7.json
  - research/findings/raw/_stable_recal/swap_primitives/w_gate_ws_s8.json
  - research/findings/raw/_stable_recal/swap_primitives/w_occ_gate_s7.json
  - research/findings/raw/_stable_recal/swap_primitives/w_occ_gate_s8.json
  - research/findings/raw/_stable_recal/swap_primitives/w_pred_mm_s7.json
  - research/findings/raw/_stable_recal/swap_primitives/w_pred_mm_s8.json
  - research/findings/raw/_stable_recal/swap_collapse/collapse_s7.json
  - research/findings/raw/_stable_recal/swap_collapse/collapse_s8.json
  - research/findings/raw/_stable_recal/swap_organ_gate/production_off_s7.json
  - research/findings/raw/_stable_recal/swap_organ_gate/production_on_s7.json
  - research/findings/raw/_stable_recal/swap_organ_gate/recal_on_s7.json
  - research/findings/raw/_stable_recal/swap_organ_gate_tw36/recal_on_s7.json
  - research/findings/raw/_stable_recal/swap_organ_gate_wgate1/wgate1_on_s7.json
  - research/findings/raw/_stable_recal/swap_organ_gate_wgate1/wgate1_on_s8.json
  - research/findings/raw/_stable_recal/stop_lead_attribution_s7.json
  - research/findings/raw/_stable_recal/dev78/dev_check.json
  - research/findings/raw/_stable_recal/dev78/compare_s7.json
  - research/findings/raw/_stable_recal/dev78/compare_s8.json
  - research/findings/raw/_stable_recal/stop_calibration_record.json
  - research/findings/raw/_stable_recal/stop_ignition_curve_s7.json
  - research/findings/raw/_stable_recal/stop_ignition_curve_s8.json
  - research/findings/raw/_stable_recal/stop_ignition_inhibition_s7.json
  - research/findings/raw/_stable_recal/stop_ignition_curve_w6p5_nws4_thal50_s7.json
  - research/findings/raw/_stable_recal/stop_ignition_curve_w6p5_nws4_thal50_s8.json
  - research/findings/raw/_stable_recal/stop_sustain_nws/nws4_s7.json
  - research/findings/raw/_stable_recal/stop_sustain_nws/nws4_s8.json
  - research/findings/raw/_stable_recal/stop_thal_support/w6p5_nws4_t0_25_50_s7.json
  - research/findings/raw/_stable_recal/stop_thal_support/w6p5_nws4_t75_100_200_s8.json
  - research/findings/raw/_stable_recal/stop_window/recal_on_s7.json
  - research/findings/raw/_stable_recal/stop_window/recal_on_s8.json
  - research/findings/raw/_stable_recal/stop_window/production_off_s7.json
  - research/findings/raw/_stable_recal/stop_margins/recal_on_s7.json
  - research/findings/raw/_stable_recal/stop_margins/recal_on_s8.json
  - research/findings/raw/_stable_recal/stop_margins/production_off_s7.json
  - research/findings/raw/_stable_recal/stop_organ_gate/recal_on_s1.json
  - research/findings/raw/_stable_recal/stop_organ_gate/production_off_s7.json
  - research/findings/raw/_stable_recal/stop_organ_gate/production_on_s7.json
  - research/findings/raw/_stable_recal/stop_organ_gate_literal/literal_on_s1.json
  - research/findings/raw/_stable_recal/stop_organ_held/recal_on_held01_s1.json
  - research/findings/raw/_stable_recal/stop_organ_held/recal_on_held01_s7.json
  - research/findings/raw/_stable_recal/stop_organ_held/recal_on_held01_s8.json
  - research/findings/raw/_stable_recal/dev78_stop/dev_check.json
  - research/findings/raw/_stable_recal/dev78_stop/compare_s7.json
  - research/findings/raw/_stable_recal/dev78_stop/compare_s8.json
  - research/findings/raw/_stable_recal/dev78_stop/compare_vs_busswap_s7.json
  - research/findings/raw/_stable_recal/dev78_stop/compare_vs_busswap_s8.json
  - research/findings/raw/_stable_recal/dev78_stop/off_head/s7/inner__intact_a.json
---

seed-waiver: a labelled dev-seed diagnostic (dev seeds 7 and 8 for the brain; dev seeds 1-15 for the isolated swap and global-stop organs); no gate seed was run, no default was flipped and no capability claim is made.

# The GNW workspaces re-calibrated for valid membranes (bus and swap; addendum: global stop), and the stop lead attributed (dev seeds 7 and 8, 2026-10-27)

## Answer in one paragraph

The thought-swap workspace, measured alone, behaves like the GNW bus: on the old integrator its ignition knee is the
drive at which its cells leave the valid membrane range (111 to 118 of 300 out of range, rate exactly 1/3), and on
valid membranes its production constants give depolarization block and no selective ignition. Re-calibrated by its
own rules on valid membranes (recurrent weight 6.5, normalization 4, thalamic support 25 pA, plateau 0.084 with the
criterion at half of it, establishment 1280 pA, read and veto weights rescaled to the new plateau, admission weight
4, timing window 36; every value and its reason in `swap_calibration_record.json`), it passes its own GO gate on 12
of 15 dev seeds, every non-timing check on 15 of 15. The order matters for that count: the gate ran first with the
production timing window of 9 steps (GO on 4 of 15, every failure the timing check) and the window was re-derived to
36 by the organ's own 3-ignition-period rule after that result (see "Order of events and which seeds were fitted"). By intervention on inner, dev seed 7, the "Setting the held
thread aside" stop lead is decided by the global-stop workspace alone, in both directions (the ACC-BG trigger and
the swap workspace do not move it), and it appears only while that workspace has 250 to 270 cells out of range.
`SIM_STABLE_RECAL=1` (default OFF, refused without `SIM_STABLE_INTEGRATION=1`, byte-identical when unset) applies
the bus and swap re-calibrations in the production organ builders. On dev seeds 7 and 8, all five Stage-2 groups,
STABLE+RECAL against OFF: the answer rate is identical in every cell (no FLIP-HURTS-ANSWERS), every scorer `all`
is identical (C4 and C5 pass, as under OFF), the same recalls are verified, the topic-swap lead is back, and
out-of-range neurons fall from 4647-5381 to 111-820 per group with none left in the three GNW workspaces. The stop
lead is gone (its workspace is not re-calibrated here). The gate seeds were not run; the C5 caveat is now on the
10-26 Stage-2 result. Dev seeds 7 and 8 are the seeds the swap workspace was calibrated on (the bus: seed 7), so
that whole-brain comparison is in-sample: it shows the switch leaves intact what it was fitted on, not that the
replies stay identical on a seed it was not fitted on. ADDENDUM (same day, below): the global-stop workspace is now
re-calibrated by its own rules too and the switch covers it; its own GO gate holds on 5 of 15 dev seeds (the stop
itself on 11 of 15; with two held contents, what production holds, the stop, the normalization lesion, freeze and
scramble on 15 of 15), and on dev seeds 7 and 8 (in-sample)
the stop lead is back on all 38 turns that carry it under OFF, with the answer rate, every scorer, every recall and
the swap lead unchanged and no cell of the stop workspace out of range; against the bus and swap re-calibration
alone those 38 leads are the only change in any reply. On the old integrator that lead rested on 250 to 270 runaway
cells; on valid membranes the three-content conflict the organ gate clears no longer stands by itself, so this
calibration is a measured operating point of the stop, not a GO of the organ.

## What was built (default OFF)

- `SIM_STABLE_RECAL` (commit 99cc7aa8a, `research/runners/stable_recal.py`): unset, "" or "0" change nothing; "1"
  needs `SIM_STABLE_INTEGRATION=1` and raises `RuntimeError` without it (also from inside the organ builders); any
  other value raises `ValueError`. Read at call time in the production organ builders:
  - bus: `build_workspace_bridge` weight 30 -> 5; the ignition threshold 0.0445 (half of the measured plateau 0.089)
    on the N-organ, coincidence, two-distinct-organ, reentrant and P1.2 reads; per-organ drive 1000 -> 500 pA
    (three-organ bus and bus shadow) and 1400 -> 750 pA (two-organ bus). The two-distinct-organ module's own
    threshold copy was missed by the 10-26 probe; it is included here.
  - thought-swap: the constants of the table below in `build` and `run_intention_swap`, which reach the chat organ
    (`webapp/gnw_thought_swap.py`), the ACC-BG stop trigger and the congruence reader. The ignition criterion is
    scoped to the swap organ, so Rung-2's `SOLO_PLATEAU` (shared with the global-stop workspace) is unchanged.
  - an explicit non-production argument (a lesion's weight 0, a research weight) is kept.
- Tests (`tests/test_stable_recal_switch.py`, with `tests/_stable_recal_fingerprints.py`): fingerprints of the
  production swap path and of four production bus hops, captured on the code before the switch, are reproduced with
  the switch unset, "" and "0" on both integrators (byte-identical); the switch raises without
  `SIM_STABLE_INTEGRATION=1` and on bad values; every accessor returns the production value when off and the
  measured value when on; under the switch the swap swaps and the bus keeps its quorum rules (three votes ignite and
  two do not; two coincident votes ignite and one does not). 37 passed together with
  `tests/test_stable_integration_switch.py` (numpy backend); the existing bus tests (`test_bg_neural_seam.py`,
  `test_reasoning_transitive_chat_wirein.py`, `test_wm_referent_focus_bind.py`) 29 passed.
- Whole-brain identity: inner on dev seed 7 with both switches empty, run at this branch's head, gives replies
  byte-identical to the 10-26 OFF census and the same out-of-range count per organ
  (`research/findings/raw/_stable_recal/dev78/off_head/s7/inner__intact_a.json` against
  `research/findings/raw/_stable_integration/census/off/s7/inner__intact_a.json`).
- Probes: `research/runners/stable_integration_swap_ignition_probe.py` (the swap organ alone: curve, inhibition,
  sustain with the 15-step read reliability, thalamic support, gate / mismatch primitives, eviction collapse, the
  organ's GO gate under a calibration; commits 2e21fd7c8, 09adb95f2);
  `research/runners/stable_integration_exempt_probe.py` gains the `stop` and `accbg` organs (b2dace642);
  `research/runners/stable_recal_stop_attribution.py` (1ea629077, 30c722eb9);
  `research/runners/stable_recal_dev_check.py` (the dev check below); the census records `stable_recal_env`
  (3d0653c20); `research/runners/stable_recal_calibration_record.py` writes `swap_calibration_record.json` (with its
  provenance sidecar) by reading the cited artifacts and checking the record's values against `stable_recal.SWAP`.
- Remapping limit (documented in `research/runners/stable_recal.py`, pinned by
  `test_explicit_production_value_is_remapped_under_the_switch`): the value-conditional accessors cannot tell a default
  from an explicit argument, so under the switch an explicit argument equal to the production value (a bus weight of
  30.0, a swap `w_rec` of 34.0, a bus drive of 1000 or 1400 pA) is remapped like the default; only a value that
  differs from the production value is kept.
- Backend: the byte-identity references and the re-calibrated fingerprints were captured on the numpy backend, and
  `test_unset_is_byte_identical` and `test_recal_fingerprints_pinned` skip (they do not pass) on any other backend.

## The thought-swap workspace alone, on both integrators

`research/runners/stable_integration_swap_ignition_probe.py` builds the swap organ exactly as production does (960
neurons: a 320-cell workspace holding three disjoint 100-cell patterns, normalization pool, tonic thalamus, occupancy
pool, per-pattern admission gates, mismatch detectors and prediction interneurons) and reads it the way the organ
does, on the old integrator and on the stable one. Out of range means v below -110 mV, above the spike ceiling plus
5 mV, or not finite.

- **Old integrator, production constants (recurrent weight 34, normalization 16, thalamus 700 pA).** Driven alone at
  500 to 2000 pA the pattern does not ignite and at most 6 of its 300 workspace cells leave the range; from 2500 pA
  on, its rate is exactly 0.3333 and 111 to 118 cells are out of range with late mean membranes of -617.2 to
  -740.93 mV (`swap_ignition_curve_s7.json`). The knee is the runaway onset, as for the bus.
- **Stable integrator, production constants.** No cell leaves the range, but weight 34 sits in depolarization block
  (late mean v -8.56 to +12.07 mV, rate 0.0107 to 0.062), and the 700 pA thalamus fires all three patterns together.
  No weight from 5 to 60 (at normalization 16) and no normalization from 0 to 64 (at weights 5, 10, 20 and 34)
  gives a selective sustained ignition at that thalamic drive: the driven pattern's hold rate never exceeds the undriven ones by more than about 0.011 <!--derived-->
  (`swap_ignition_curve_s7.json`, `swap_ignition_inhibition_s7.json`).

## Re-calibrated by its own rules

Each constant was set on the isolated organ (dev seeds 7 and 8), never on a reply. The full record, with every
source file: `research/findings/raw/_stable_recal/swap_calibration_record.json`.

| constant | production | stable re-calibration | why |
|---|---|---|---|
| normalization (norm -> workspace) | 16 | 4 | the organ reads a 15-step window; at 16 and 8 every weight 4-6.5 reads below half its plateau in 0.53-0.93 of windows, at 6 in 0.23-0.42, at 5 in none for w >= 5 but the worst window is ~0.55 of the plateau, at 4 in none for w >= 4.5 and the worst window is 0.82-0.90 of it (4 over 5 is a margin judgment) |
| recurrent weight | 34 | 6.5 | at normalization 4 the sustain knee lies between 4.0 (0.28-0.38 of windows below half) and 4.5 (none); production is deliberately supra-critical, so 6.5 (about 1.4x the knee; plateau 0.0841 / 0.0844, worst window 0.076 / 0.0693) is a judgment, not a formula |
| ignited plateau (criterion = half of it) | 1/3 | 0.084 (0.042) | the measured plateau; the criterion is scoped to the swap organ (the global-stop workspace shares Rung-2's) |
| thalamic support | 700 pA | 25 pA | largest support with no spontaneous ignition over 1000 undriven steps on both seeds (50 pA: seed 7 ignites by itself; 75 pA: seed 8) |
| establishment drive | 8000 pA | 1280 pA | production ratio to the knee (8000 / 2500 = 3.2) times the stable knee (300 pA does not ignite, 400 pA does) |
| workspace -> occupancy | 8 | 32 | the read's input conserved: 8 x (1/3) / 0.084 (a formula; the sweep 8, 16, 32, 64 keeps the gate and mismatch primitives holding on both seeds, `research/findings/raw/_stable_recal/swap_primitives/w_ws_occ_s7.json` and `w_ws_occ_s8.json`) |
| pattern -> prediction (`w_pat_pred`) | 8 | 32 | **formula-only, not measured**: the same conservation applied to the held-content prediction read; no cited artifact varies it, so the organ gate and the sweeps show that they pass with 32, not that 32 is better than another value |
| occupancy -> gate (vacancy veto) | 20 | 80 | occupied-gate rate 0.2152 / 0.1993 at 20 (no veto), 0.0933 / 0.077 at 40, 0.0441 / 0.0367 at 80 against 0.1948 / 0.1911 empty |
| prediction -> mismatch (match veto) | 20 | 80 | the organ's own match bar (< 0.02): 0.0821 / 0.0843 at 20, 0.0377 / 0.0384 at 40, 0.0094 / 0.0133 at 80 |
| gate -> workspace (admission) | 100 | 4 | the gate primitive holds for 1-8 on both seeds and fails at 16 (seed 7), 32 (seed 8) and at 100; in the full organ, with everything else at the re-calibration, weight 1 fails the swap on seed 7 (the newcomer never ignites, rate 0.037; 5 of 11 checks fail) and passes it on seed 8 (timing fails), and weight 4 swaps on both (`swap_organ_gate_wgate1/`, `swap_organ_gate_tw36/`); 2 and 8 were not run through the organ, so 4 is the middle of the primitive's 2-8 band, not a measured optimum |
| timing window (instrument only) | 9 | 36 | the organ's rule is 3 ignition periods: 3 spikes of an ignited neuron (every 3 steps old, every 11.9 steps at 0.084); **re-derived after the 9-step window had passed GO on 4 of 15 dev seeds** (all 11 failures the timing check), see "Order of events and which seeds were fitted" |
| salient / non-salient drive, boost gain, cap | 5000 / 600 pA, 1, 0.16 | unchanged | the mismatch detector reads 0.1539 / 0.1546 salient and 0.0379 / 0.035 non-salient on stable, so the salient boost reaches the cap and the non-salient one stays near 0.06; the eviction collapse knee on stable is 0.06-0.10 |

On the old integrator the re-calibrated point still runs away under strong drive (1 to 7 cells out of range at 3000
to 4000 pA, 89 to 91 at 8000 pA), so the re-calibration is for valid membranes only, which is why the switch
refuses to run without `SIM_STABLE_INTEGRATION=1`. One property is new on valid membranes and travels as a caveat:
with thalamic support 25 pA, any pulse of 100 pA or more ignites the pattern by the hold read (150 steps), so the
knee is a property of the 45-step read, and selectivity comes from the gate and the competition, not from a high
drive threshold (`swap_ignition_curve_w6p5_nws4_thal25_s7.json`, `_s8.json`).

## The organ's own GO gate at the re-calibration (dev seeds 1-15)

`--organ` runs `evaluate_seed` (the swap's six-seed GO gate: swap, re-ignite, timing, specificity, non-salient holds,
match holds, trigger lesion load-bearing, reversible, no host reset, deterministic) under the calibration.

- Production constants: GO on the old integrator (seeds 7, 8; every ignited read is the 1/3 runaway); on the stable
  integrator 9 of 11 checks fail because nothing reaches the production criterion (`swap_organ_gate/production_*`).
- Re-calibration on the stable integrator, timing window 36: GO on 12 of 15 dev seeds; every check other than timing
  passes on 15 of 15; seeds 1, 6 and 12 fail timing only (the incumbent and the newcomer both read above the
  criterion for 3, 15 and 2 steps while the newcomer ignites). With the old window of 9 steps, which ran first, the
  same runs pass timing on 4 of 15 (`swap_organ_gate/`, `swap_organ_gate_tw36/`); the window was re-derived after
  that result (next section). The swap's new-pattern rate is 0.0847 on seed 7
  with the old pattern at 0.0, and the swap is 100% attributable to the trigger (lesion holds).

## Order of events and which seeds were fitted

**Order of the timing result.** The organ GO gate at the re-calibration was run first with the production timing
window of 9 steps: GO on 4 of 15 dev seeds, and every one of the 11 failures is the timing check (the incumbent and
the newcomer both read above the criterion for 3 to 15 steps, and 9 steps is 3 spikes of the old integrator's period-3
oscillation, not of a neuron firing every 11.9 steps). Only after that result was the window re-derived, by the organ's
own rule (3 ignition periods: 3 / 0.084 = 35.7, so 36), and the gate re-run: GO on 12 of 15. No other constant differs
between the two sets of runs (the generator checks that from their recorded calibrations), and the provenance
sidecars give the run order (window 9: started 12:40:56 to 12:41:12; window 36: 12:44:46 to 12:44:57, 2026-10-27). So
12 of 15 is the count after a correction made once the 4 of 15 was seen, not a pass of a window fixed in advance. The
window is a read-only timing instrument and production replies do not use it, so no reply depends on this choice.

**Which seeds were fitted.** Every swap constant was chosen on the isolated organ on dev seeds 7 and 8; the bus
constants on dev seed 7 (`research/findings/raw/_stable_integration/bus_ignition_curve_s7.json`, `research/findings/raw/_stable_integration/bus_quorum_w5_s7.json`). The whole-brain
OFF-versus-RECAL comparison below was run on those same two seeds, so its identical answer rates, scorer verdicts and
recalled turns are in-sample. The isolated organ's gate on dev seeds 1-6 and 9-15 (13 of the 15) is out-of-sample for
the constants, except for the timing window, which was fixed after those seeds' 9-step results. No whole-brain
comparison has been run on a seed the constants were not fitted on; the written pool battery (dev seeds 1-12, below)
would give seeds 1-6 and 9-12 for that.

## Who decides the stop lead: attribution by intervention (inner, dev seed 7)

The 10-26 finding left the "Setting the held thread aside" lead unattributed. Each candidate organ was put on the
other integrator inside a whole brain (`research/runners/stable_integration_exempt_probe.py`, organs `stop` =
the global-stop distributed-overwrite workspace, `accbg` = the ACC-BG stop trigger). Turns with the lead, out of 10
inner turns, and the out-of-range count of each candidate (`research/findings/raw/_stable_recal/stop_lead_attribution_s7.json`):

| arm | stop lead | stop workspace out of range | trigger out of range | swap workspace out of range |
|---|---|---|---|---|
| old brain (OFF) | 7 | 250 | 0 | 344 |
| stable brain (ON) | 0 | 0 | 0 | 2 |
| ON, stop workspace on the old integrator | 8 | 270 | 0 | 2 |
| OFF, stop workspace on the stable integrator | 0 | 0 | 0 | 344 |
| ON, ACC-BG trigger on the old integrator | 0 | 0 | 0 | 3 |
| OFF, ACC-BG trigger on the stable integrator | 7 (0 of 10 replies change) | 250 | 0 | 344 |
| ON, swap workspace on the old integrator | 0 | 0 | 0 | 344 |
| OFF, swap workspace on the stable integrator | 8 | 270 | 0 | 2 |

The global-stop workspace alone decides the stop lead, in both directions, and the lead appears only while that
workspace has 250 to 270 cells out of range. The trigger is valid on both integrators and moves nothing. The swap
workspace moves the "On X, then" lead and not the stop lead. So the stop lead, like the bus's recall verification
and the swap's topic change, is produced by a workspace whose ignition is a numerical runaway. When this section was
written its re-calibration was not in this branch; the addendum below adds it. It shares Rung-2's ignition
criterion, which is why the swap criterion is scoped to the swap organ (and the stop criterion to the stop organ).

## Dev check: OFF vs STABLE+RECAL (bus and swap only), dev seeds 7 and 8, five groups

This section is the dev check before the global-stop re-calibration (commit 176900ce4); the addendum repeats it with the
global stop re-calibrated.

Workload: the membrane census (`python -m research.runners.membrane_census --shard --seed N --group G --arm
intact_a`) of the five Stage-2 groups on dev seeds 7 and 8, numpy CPU, each in a fresh subprocess under
`tools/memcap.sh 10` after `tools/mem_ok.sh 9`. OFF = both switches unset or empty (seed 7 and seed 8 inner: the
10-26 census records; seed 8 conv, infer, ask and emo: run at this branch, `census/off/s8/`); STABLE+RECAL =
`SIM_STABLE_INTEGRATION=1 SIM_STABLE_RECAL=1` (`research/findings/raw/_stable_recal/dev78/stable_recal/s{7,8}/`).
These two seeds are the calibration seeds of the swap workspace (the bus: seed 7), so every "identical to OFF" below
is in-sample. Comparison by `stable_integration_compare` (`research/findings/raw/_stable_recal/dev78/compare_s7.json`,
`research/findings/raw/_stable_recal/dev78/compare_s8.json`) and `research/runners/stable_recal_dev_check.py`
(`research/findings/raw/_stable_recal/dev78/dev_check.json`); the answer rate is
`tools/answer_rate.py` `classify` / `rate` over every turn reply of the shard, with `report` per group-seed.

| seed, group (criterion) | answered OFF / RECAL | scorer `all` OFF / RECAL | recalls verified OFF / RECAL | out-of-range neurons OFF / RECAL | bus+swap+stop workspace out of range OFF / RECAL | turns with a visible change |
|---|---|---|---|---|---|---|
| 7, conv (C1) | 7/11 / 7/11 | false / false | 7 / 7 (same turns) | 5082 / 522 | 80+342+247 / 0+0+0 | 6 of 11 |
| 7, infer (C2) | 9/10 / 9/10 | true / true | 8 / 8 (same turns) | 4878 / 111 | 80+344+250 / 0+0+0 | 7 of 10 |
| 7, ask (C3) | 7/19 / 7/19 | false / false | 7 / 7 (same turns) | 5381 / 505 | 80+344+247 / 0+0+0 | 8 of 19 |
| 7, emo (C4) | 5/13 / 5/13 | true / true | 0 / 0 (same turns) | 382 / 110 | 0+0+0 / 0+0+0 | 6 of 13 |
| 7, inner (C5) | 9/10 / 9/10 | true / true | 6 / 6 (same turns) | 4984 / 657 | 400+344+250 / 0+0+0 | 8 of 10 |
| 8, conv (C1) | 7/11 / 7/11 | false / false | 7 / 7 (same turns) | 4748 / 526 | 80+347+207 / 0+0+0 | 6 of 11 |
| 8, infer (C2) | 9/10 / 9/10 | true / true | 8 / 8 (same turns) | 5303 / 820 | 80+349+207 / 0+0+0 | 7 of 10 |
| 8, ask (C3) | 7/19 / 7/19 | false / false | 7 / 7 (same turns) | 5229 / 502 | 80+349+207 / 0+0+0 | 6 of 19 |
| 8, emo (C4) | 5/13 / 5/13 | true / true | 0 / 0 (same turns) | 342 / 106 | 0+0+0 / 0+0+0 | 8 of 13 |
| 8, inner (C5) | 9/10 / 9/10 | true / true | 6 / 6 (same turns) | 4647 / 661 | 400+349+218 / 0+0+0 | 7 of 10 |

- **Answer rate: no FLIP-HURTS-ANSWERS.** Identical in all ten cells, over every turn and over the non-teach turns
  (`answer_rate_report_all_turns` and `answer_rate_report_non_teach` status OK).
- **C4 and C5 hold; nothing else moves.** Every scorer `all` is the same as OFF on both seeds (C1 and C3 false in
  both arms, C2, C4 and C5 true in both). Under SIM_STABLE_INTEGRATION=1 alone the 10-26 finding had C5 false on
  both seeds and infer false on seed 7.
- **Recall verification is restored exactly:** in every cell the same turns carry a verified `recalled_svo` as OFF.
  The derived chains of inner come back word for word ("I derived this from: lynx chase vole; vole chase beetle;
  beetle chase mite. the lynx chases the mite").
- **Membranes:** out-of-range neurons fall from 4647-5381 to 111-820 per group (emo 342-382 to 106-110), and the
  three GNW workspaces have none (OFF: bus 80-400, swap 342-349, stop 207-250). What is left is what the 10-26
  finding left under ON: the BridgeParser phasor slice (394-713), the LearnedAssocGraph cells whose own recovery
  variable holds v below -110 mV (106-110) and one biased-competition buffer cell.
- **The topic swap is back:** the "On X, then --" lead is on 38 turns under OFF and on 38 under STABLE+RECAL (under
  SIM_STABLE_INTEGRATION=1 alone it was gone).
- **What still differs from OFF, all visible in the reply text, never in answered / abstained / recalled / derived:**
  - the stop lead "Setting the held thread aside --" is gone on all 38 turns that carry it under OFF, because the
    global-stop workspace that decides it keeps its production constants (above);
  - "-- worth going further here." is appended on 18 turns, as under ON on 10-26;
  - emo's valence differentials grow (joy +0.038 to +0.062 on seed 7), as under ON;
  - a few phrasings: ask's role-binding refusal for "the goat eats the hay" is worded differently, c_dog_3 loses
    "As for it --", and two abstentions gain the curiosity follow-up ("I haven't learned about crab yet").
- Speed: the workload ran 0.866x to 1.147x of OFF's wall time; these ran beside other jobs, so it is not a
  like-for-like cost measurement.

## The global-stop workspace alone, on both integrators (addendum, 2026-10-27 evening)

`research/runners/stable_integration_stop_ignition_probe.py` builds the global-stop distributed-overwrite workspace
exactly as `webapp/gnw_global_stop.py` does (`build_overwrite_bridge`: 440 neurons, a 320-cell workspace holding three
overlapping 100-cell patterns over 270 used cells, a 60-cell normalization pool and a 60-cell tonic thalamus; the
conflict-gated Tsodyks-Markram depression of the shared recurrence) and reads it the way the organ does (35 drive
steps, the late 15 of 45 free steps, a 100-step read after the stop). Dev seeds 7 and 8. Out of range is the census
rule (v below -110 mV, above the spike ceiling plus 5 mV, or not finite).

- **Old integrator, production constants (weight 34, normalization 16, thalamus 700 pA).** One pattern driven alone
  does not ignite up to its knee (seed 8: 2000 pA, seed 7: 3000 pA); from the knee on its rate is exactly 0.3333 and
  103 to 109 of the 270 workspace cells are out of range. All three driven together at the production establishment
  drive (5000 pA) ignite with all 270 cells out of range: the 250 to 270 the attribution saw inside the brain
  (`stop_ignition_curve_s7.json`, `_s8.json`). The sensor reads a conflict margin of exactly 0 and a confident margin
  of 0.324 to 0.331 (`research/findings/raw/_stable_recal/stop_margins/production_off_s*.json`); the stop clears with
  a boost from 0.09 and the normalization lesion from 0.22 to 0.3, the documented [0.09, 0.15] window
  (`research/findings/raw/_stable_recal/stop_window/production_off_s*.json`). The organ's own GO gate passes on both
  seeds with 270 cells out of range in its stop trial.
- **Stable integrator, production constants.** No pattern ignites at any weight from 5 to 60; the 700 pA thalamus
  fires all three patterns together at about 0.06; at most one cell leaves the range (seed 8, weight 5, all three
  at 8000 pA). No normalization from 0 to 64 at weights 5 to 34 gives a selective sustained pattern (the driven
  pattern's hold rate never exceeds the others' by more than 0.013 <!--derived-->, `stop_ignition_inhibition_s7.json`). The organ's
  gate is UNDEFINED on both seeds (nothing to clear), and inside the brain every stop the trigger asks for reads
  `n_ignited_pre` 0, so it never clears and the lead never appears.

So the stop lead, like the bus's recall verification and the swap's topic change, rests on a numerical runaway.

### Re-calibrated by its own rules (scoped to the stop organ)

Each constant was set on the isolated organ on dev seeds 7 and 8, never on a reply, before the organ gate ran. The full
record, with every source file and a check of every typed claim against the artifacts:
`research/findings/raw/_stable_recal/stop_calibration_record.json` (written by
`research/runners/stable_recal_stop_calibration_record.py` as a module, provenance sidecar; it also checks the values
against `stable_recal.STOP`).

| constant | production | stable re-calibration | why |
|---|---|---|---|
| normalization (norm -> workspace) | 16 | 4 | the swap organ's rule: the largest normalization whose sustained pattern reads reliably in the organ's 15-step window; at 16, 8 and 6 every weight 4-10 has windows below half its plateau, at 5 only weight 5 has none (worst window ~0.6 of the plateau), at 4 weights 4.5-8 have none (at 6.5 the worst window is 0.80-0.88 of it); at 2 the read is reliable but the overlapped neighbour is recruited from weight 6.5 (4 over 5 is a margin judgment) |
| recurrent weight | 34 | 6.5 | at normalization 4 the sustain knee lies between 4.0 (0.37 / 0.38 of windows below half) and 4.5 (none) and the neighbour is recruited at 10, not at 8; 6.5 is ~1.44x the knee inside that band (the swap organ's judgment), plateau 0.0846 / 0.0830 |
| ignited plateau (criterion = half of it) | 1/3 | 0.084 (0.042) | the measured plateau; the criterion is scoped to the stop organ (`_st_ignited`), Rung-2's `SOLO_PLATEAU` is untouched |
| thalamic support | 700 pA | 50 pA | the swap organ's rule: the largest support with no spontaneous ignition over 1000 undriven steps on both seeds (0, 25, 50 pA silent; 75 pA ignites patterns by themselves on both) |
| establishment drive (`STRONG_PA`) | 5000 pA | 800 pA | the module's rule `STRONG_PA = 2 x IGNITE_PA` (the knee); the stable knee by the organ's 45-step read is 400 pA (300 pA reads 0.037 on seed 7 and 350 pA 0.028 on seed 8, below 0.042; 400 pA ignites on both) |
| conflict reference (`MARGIN_REF`) | 1/6 | 0.057 | the module's stated rule, "halfway between single-content and co-ignition": the measured confident margin (0.0825 / 0.0833) and three-content conflict margin (0.0267 / 0.0351, max 0.0468) give a midpoint of 0.057. The formula `SOLO_PLATEAU x 0.5` (0.042) is that rule at a conflict margin of 0, true only in the lockstep runaway; it was run through the gate beside the midpoint as a comparison arm, both fixed before the gate ran |
| boost scale | 0.85 | 31 | the module's rule: the full-conflict boost lies in the window where the intact workspace clears and the normalization-lesioned one does not. On valid membranes (fixed boosts, 8 trials each, both seeds, three and two held contents) the intact workspace clears on every trial at 0.6, 0.8 and 1.0, with a non-monotone dip at 0.25-0.5; the lesion never clears up to 1.0 (U >= 1 empties x every spike), so normalization owns the clearing there (`attributable_to` 1.0). Target 0.8, the middle of that band (a judgment): 0.8 / (0.057 - 0.0309) = 30.7, rounded to 31 |
| pulse 110 steps, boost gain 1, tau_D 250 ms, overlap 15, ws -> norm 5, thal -> ws 5, OU 30 pA, read windows | | unchanged | not constants of the runaway; nothing measured here asked for them to move |

On the old integrator the re-calibrated point still runs away under strong drive (seeds 7 / 8: 19 / 10 cells out of range
for one pattern at 5000 pA, 88 / 87 at 8000 pA; 254 / 260 for all three at 8000 pA), so like the other two re-calibrations it
is for valid membranes only. As for the swap, the hold read (145 steps) of a pattern driven at 100 pA or more already
reads ignited, so the knee is a property of the organ's 45-step read.

### The organ's own GO gate at the re-calibration (dev seeds 1-15)

`--organ` runs `evaluate_seed` (global stop, selectivity, normalization and STD load-bearing, scramble, conflict-off,
continuous stop without restore, byte-identical substrate, determinism) under the calibration. No constant was changed
after the gate ran; it was re-run twice on unchanged code only to re-stamp provenance from clean commits (the first
run's sidecars read dirty because a test helper was being edited meanwhile), and the results are byte-identical.

| | GO | global stop | selectivity | normalization load-bearing | freeze / scramble / conflict-off hold | conflict-gated boost | continuous stop | out of range in the stop trial |
|---|---|---|---|---|---|---|---|---|
| production, old integrator (7, 8) | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 270 |
| production, stable (7, 8) | 0/2 (UNDEFINED) | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0 |
| re-calibration, stable, margin_ref 0.057 (1-15) | 5/15 | 11/15 | 15/15 | 15/15 | 7/15 | 14/15 | 11/15 | 0 |
| comparison, margin_ref 0.042 (formula) (1-15) | 4/15 | 6/15 | 15/15 | 15/15 | 7/15 | 8/15 | 6/15 | 0 |
| supplementary: two held contents (1-15) | | 15/15 | | 15/15 | 15/15 (freeze, scramble) | | | |

GO seeds at the re-calibration: 1, 3, 5, 7 and 10. The dominant failure is not the stop: the anti-cheats that need the
three-content conflict to stand WITHOUT the stop (freeze, scramble, conflict-off) pass on 7 of 15, because on valid
membranes the divisively normalized workspace resolves a three-way co-ignition to one pattern by itself within about
250 steps (seed 8 at a fixed zero boost: 5 of 8 trials end with one pattern). On the old integrator every co-ignited
pattern was a self-sufficient lockstep runaway, so the conflict stood until the stop removed it. The other failures
are the sensor: on valid membranes the conflict margin is noisy (0.018 to 0.047 on the calibration seeds), so a
conflict that reads near or above the reference gets a boost in the 0.25-0.5 dip or none (seeds 2, 9, 12; seed 6
starts with one pattern). With the two contents production holds (`max(2, n_ignited)` in
`webapp/gnw_global_stop.py`; every stop in the dev workload holds two), the stop, the normalization lesion, freeze
and scramble pass on 15 of 15 (`stop_organ_held/`, supplementary, not the gate; conflict-off, the conflict-gated boost
and the continuous stop were not run with two). Seeds 7 and 8 are the calibration seeds; 1-6 and 9-15 are
out-of-sample for every constant.

So this calibration is a measured operating point of the stop on valid membranes, not a GO of the organ: the stop
itself works on 11 of 15 seeds and on every seed with two held contents, but the three-content conflict the gate
clears is not self-sustaining any more. Named next method (not run): the conflict sensor and the co-ignited state
are the residual. A three-way co-ignition that resolves by itself raises the question the 2026-07-31 wall finding
asks: what kept the real conflict standing that the runaway replaced (here, the incumbent's own persistence, e.g. the
slow NMDA / plateau support the swap organ's incumbent has), and the sensor reads a 15-step window of a ~0.08 rate on
70-85 cells, so its margin noise is sampling noise a longer or population-pooled read would cut.

### In the switch (default OFF)

`SIM_STABLE_RECAL=1` (commit 3bc2ae085) now also applies the stop re-calibration: `stable_recal.STOP` with
`stop_value` / `stop_ignite_thresh`, read at call time in `_gnw_distributed_overwrite_workspace_derisk`
(`build_overwrite_bridge`: weight, normalization, thalamus; the establishment drive in every trial; `margin_ref` and
`boost_scale` in `run_conflict_stop`, `run_confident` and `evaluate_seed`, which `webapp/gnw_global_stop.py` reaches
with the explicit production values, remapped like defaults; the ignition criterion scoped to the stop organ). No
`webapp/` or `sim/` edit. Rung-2's `SOLO_PLATEAU` is untouched (tested). The same rules as before: unset, "" or "0"
change nothing; "1" needs `SIM_STABLE_INTEGRATION=1`, also from inside the stop organ's builder.

- Byte-identity: fingerprints of the production `_StopWorkspace` path (two and three held contents, the upstream-scaled
  boost and the lesion), captured at 797ae26de before the module read the switch, are reproduced with the switch
  unset, "" and "0" on both integrators.
- Under the switch on the stable integrator the production path clears a two- and a three-content conflict to
  n_ignited 0 (boosts 0.99 and 1.43), the boost scaled by 0.5 still clears, the lesion leaves the conflict standing; on
  the stable integrator with the switch unset nothing ignites and no stop is clean (both pinned).
- `tests/test_stable_recal_switch.py` 39 passed together with `tests/test_stable_integration_switch.py` (numpy); the
  existing bus tests 29 passed.

### Dev check with the global stop re-calibrated: OFF vs STABLE+RECAL, dev seeds 7 and 8 (in-sample)

The workload of the dev check above (the membrane census of the five Stage-2 groups, arm `intact_a`, numpy CPU, each in
a fresh subprocess under `tools/memcap.sh 10` after `tools/mem_ok.sh 9`), with STABLE+RECAL now also re-calibrating the
global stop (code at 8e25c8e04; `research/findings/raw/_stable_recal/dev78_stop/stable_recal/s{7,8}/`), against the
same OFF records. Comparisons: `research/findings/raw/_stable_recal/dev78_stop/compare_s7.json`,
`research/findings/raw/_stable_recal/dev78_stop/compare_s8.json` and
`research/findings/raw/_stable_recal/dev78_stop/dev_check.json` (VALID: no turn in either arm carries an organ
error); against the bus and swap re-calibration alone (the census of the dev check above):
`research/findings/raw/_stable_recal/dev78_stop/compare_vs_busswap_s7.json` and `_s8.json`. Seeds 7 and 8 are the
calibration seeds of the global stop (as of the swap; the bus: seed 7), so every "same as OFF" here is in-sample.
OFF at this head (both switches unset, seed 7 inner, `dev78_stop/off_head/`) reproduces the OFF census record's
replies and per-organ membrane census exactly.

A first run of this census is not used: the worktree had no `data/` link, so the comprehension organ could not open
its corpus on 16 turns of the seed-7 groups. The dev check now refuses to call a comparison VALID when any turn
carries an organ error (`research/FAILURE_LOG.md`, 2026-10-27).

| seed, group (criterion) | answered OFF / RECAL | scorer `all` OFF / RECAL | turns with `recalled_svo` OFF / RECAL | stop lead OFF / RECAL | swap lead OFF / RECAL | out-of-range neurons OFF / RECAL | bus+swap+stop workspace out of range OFF / RECAL | turns with a visible change |
|---|---|---|---|---|---|---|---|---|
| 7, conv (C1) | 7/11 / 7/11 | false / false | 7 / 7 (same turns) | 4 / 4 | 4 / 4 | 5082 / 522 | 80+342+247 / 0+0+0 | 2 of 11 |
| 7, infer (C2) | 9/10 / 9/10 | true / true | 8 / 8 (same turns) | 5 / 5 | 5 / 5 | 4878 / 111 | 80+344+250 / 0+0+0 | 2 of 10 |
| 7, ask (C3) | 7/19 / 7/19 | false / false | 7 / 7 (same turns) | 3 / 3 | 3 / 3 | 5381 / 505 | 80+344+247 / 0+0+0 | 5 of 19 |
| 7, emo (C4) | 5/13 / 5/13 | true / true | 0 / 0 | 0 / 0 | 0 / 0 | 382 / 110 | 0+0+0 / 0+0+0 | 6 of 13 |
| 7, inner (C5) | 9/10 / 9/10 | true / true | 6 / 6 (same turns) | 7 / 7 | 7 / 7 | 4984 / 657 | 400+344+250 / 0+0+0 | 2 of 10 |
| 8, conv (C1) | 7/11 / 7/11 | false / false | 7 / 7 (same turns) | 4 / 4 | 4 / 4 | 4748 / 526 | 80+347+207 / 0+0+0 | 2 of 11 |
| 8, infer (C2) | 9/10 / 9/10 | true / true | 8 / 8 (same turns) | 5 / 5 | 5 / 5 | 5303 / 820 | 80+349+207 / 0+0+0 | 2 of 10 |
| 8, ask (C3) | 7/19 / 7/19 | false / false | 7 / 7 (same turns) | 3 / 3 | 3 / 3 | 5229 / 502 | 80+349+207 / 0+0+0 | 3 of 19 |
| 8, emo (C4) | 5/13 / 5/13 | true / true | 0 / 0 | 0 / 0 | 0 / 0 | 342 / 106 | 0+0+0 / 0+0+0 | 8 of 13 |
| 8, inner (C5) | 9/10 / 9/10 | true / true | 6 / 6 (same turns) | 7 / 7 | 7 / 7 | 4647 / 661 | 400+349+218 / 0+0+0 | 0 of 10 |

- **The stop lead is back on every turn that carries it under OFF:** 38 of 38, on the same turns. On each of those
  turns the stop organ held two contents, read both ignited before the stop and none after (`n_ignited_pre` 2,
  `n_ignited_post` 0, cleared), as under OFF. Its boost there is 1.01 to 1.46 on seed 7 and 1.34 to 1.77 on seed 8
  (OFF: 0.1417 on every turn, 0.85 x 1/6 at a conflict margin of exactly 0). The two held contents co-ignite with a
  margin near 0, so the boost sits above the robust band measured with fixed boosts (0.6 to 1.0) and inside the range
  of the supplementary two-content checks, where the normalization-lesioned workspace did not clear at 1.67 and 1.74
  (`research/findings/raw/_stable_recal/stop_organ_held/recal_on_held01_s7.json`, `_s8.json`). The turn's words do
  not reach the stop organ: on these turns it is given the number of held contents, and the same boost sequence
  repeats in every group of a seed, so the boost's spread is the workspace's own noise.
- **Nothing else that the dev check reads moves:** the answer rate is identical in every cell (all turns and
  non-teach turns, status OK, no FLIP-HURTS-ANSWERS); every scorer `all` is the same as OFF (C2, C4 and C5 true, C1 and
  C3 false, in both arms); the same turns carry `recalled_svo` and the same number carry `verified`; the swap lead is
  on 38 of 38 turns in both arms.
- **Membranes:** out-of-range neurons per group are those of the bus and swap re-calibration (111-820, OFF
  4647-5381); the global-stop workspace has none (OFF 207-250), like the bus and swap workspaces.
- **Against the bus and swap re-calibration alone** (`compare_vs_busswap_s*.json`): the replies differ on 38 turns,
  each by the stop lead alone ("Setting the held thread aside --" put back in front of the same text); scorers and
  out-of-range counts are identical.
- **What still differs from OFF** (32 visible changes over the ten cells, the same as in the dev check above):
  "-- worth going further here." appended on 18 turns; emo's valence differentials grow on the four `feel_*` turns of
  each seed (feel_fear reads "negative" where OFF reads "neutral"); ask's role-binding refusal for "the goat eats the
  hay" is worded differently on both seeds; c_dog_3 loses "As for it --" on both seeds; two abstentions on seed 8
  gain the curiosity follow-up. Never in answered / abstained / recalled / derived.
- This is in-sample: seeds 7 and 8 fixed every global-stop constant. The out-of-sample read is the pool battery
  (seeds 1-6 and 9-12, written, not queued).

## What this does not show

- The gate seeds were not run. Nothing here says whether C1-C5 hold on valid membranes at seeds 42, 43, 44, 100,
  101 and 102.
- (Before the addendum) the global-stop workspace was not re-calibrated and the stop lead was absent under
  STABLE+RECAL. The addendum re-calibrates it, but its own GO gate holds on 5 of 15 dev seeds: the three-content
  conflict no longer stands without the stop on valid membranes (the checks that need it hold on 7 of 15), and the
  noisy conflict sensor misses the stop on 4 of 15. The lead's return on dev seeds 7 and 8 is in-sample (they are its calibration seeds), and
  every stop in that workload holds two contents, the case the organ gate does not test (the supplementary
  two-content checks pass on 15 of 15).
- Two global-stop constants are judgments recorded as such: weight 6.5 (~1.44x the knee) and the boost target 0.8
  (the middle of the robust clearing band); margin_ref follows the module's stated rule (the midpoint of the two
  measured margin levels), not its formula (plateau / 2), a choice made before the gate ran and run beside the
  formula (GO 5 of 15 against 4 of 15).
- The swap organ's timing check fails on 3 of 15 dev seeds at the re-calibration; two of its constants (weight 6.5,
  normalization 4 over 5) are judgments recorded as such.
- The whole-brain OFF-versus-RECAL result (identical answer rate, scorer verdicts and recalled turns) is in-sample:
  dev seeds 7 and 8 are the calibration seeds (swap 7 and 8, bus 7). It has not been repeated on a seed the constants
  were not fitted on.
- The 12 of 15 organ GO is the count after the timing window was re-derived from 9 to 36 once the 9-step window had
  given 4 of 15 (every failure the timing check); the 36 follows the organ's own rule but was applied after the
  result was seen.
- `w_pat_pred` = 32 is formula-only: no artifact varies it. `w_gate_ws` = 4 was run through the full organ only
  against weight 1 (which fails the swap on seed 7); 2 and 8 were not run through the organ.
- One brain workload per group and seed; the 10-26 finding measured the run-to-run floor of this workload as zero
  (replicate pairs identical), not repeated here.

## Pool battery (written, not queued)

`research/coordination/stable-recal-pool-lines.txt`: dev seeds 1-12 x the five groups x {off, stable, stable_recal},
180 lines in the membrane-census format (generator `research/runners/stable_recal_pool_lines.py`); outputs under
`research/findings/raw/_stable_recal/dev/<arm>/s<N>/`. Seeds 7 and 8 in it repeat the calibration seeds; seeds 1-6
and 9-12 are the ones a result there would count as out-of-sample. Regenerated with the addendum (f3a4d818d): the
stable_recal arm now also re-calibrates the global stop, and every line's reason states its seed's role (seeds 7
and 8: calibration replicate, in-sample; 1-6 and 9-12: held-out, out-of-sample).

## Derived

Ratios, differences, counts of turns and the "about 1.4x" / "~0.55" / "0.82-0.90" fractions are derived from the
cited records; 3.2 x 400 = 1280; 8 x (1/3) / 0.084 = 31.7; 3 / 0.084 = 35.7. Global stop: 2 x 400 = 800;
(0.0829 + 0.0309) / 2 = 0.0569; 0.8 / (0.057 - 0.0309) = 30.7; 0.8 / (0.042 - 0.0309) = 72; 6.5 / 4.5 = 1.44; the
pooled means 0.0829 and 0.0309 are the means of the two seeds' figures; the turn and lead counts are counted from the
cited census records.
