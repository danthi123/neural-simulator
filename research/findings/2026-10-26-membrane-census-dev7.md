---
type: finding
status: qualified
date: 2026-10-26
lane: substrate (integrator validity); the census owed by research/findings/2026-10-26-stage2-fail-diagnoses.md ("Substrate")
mechanism: the 1 ms explicit-Euler membrane step leaving the physiological range, counted per organ over the production brain
seeds: [7]
---

seed-waiver: a labelled dev-seed diagnostic census of ONE heterogeneity draw (dev seed 7); no gate seed was run and no capability claim is made.

# Membrane-validity census of the Stage-2 production brain, dev seed 7 (2026-10-26)

## Answer in one paragraph

The defect class behind the emotion organ's s101 joy miss and the BG step-1 breach is not confined to those two organs. On dev seed 7, with the five Stage-2 groups run as the live `brain_chat` handler at production defaults (`intact_a` arms), the census found 12 stepper labels with membrane excursions plus one resonate-and-fire phasor slice (BridgeParser's `onebrain_composer` slice, 721 rows, not membranes). The labels are the 31 distinct stepper labels over 189 bridge instances in the five fresh-brain arms (all Izhikevich, all with bounded conductance integration OFF) [1]; 18 labels were clean. The phasor slice is a resonate-and-fire slice that the Izhikevich step also integrates, so its v and u are not membranes. One of the 12 labels, the metacog pool bridge, owes its excursions to the role-competition circuit stepping that bridge (see its row).
Low drawn capacitance is neither necessary nor sufficient: the offenders' drawn C spans 47 to 151 pF in the heterogeneity-drawn regions (only 20 pF fast-spiking preset cells in three organs are lower); the common factor is a large end-of-step conductance relative to C (the explicit-step hazard (g_i+g_e)*dt/C at or above about 2) or a spike crossing withheld by the refractory timer. Whether any of this changes a reply is not measured here.

[1] The organ label is the majority stepper of a bridge: a bridge stepped by more than one caller carries the label of the caller with the most steps, so its offenders can belong to a minority stepper.

## What was run

- Probe: `research/runners/membrane_census.py` (unit tests `tests/test_membrane_census.py`, 19 pass). It wraps `SimulationBridge._run_one_simulation_step` at class level (restored on exit; checked after every run), reads each bridge's membrane array v after every step and never writes, and runs one Stage-2 gate arm in-process through `stage2_exit_gate.run_worker` as the workload. No `sim/` or production edit.
- It records per bridge and region the minimum and maximum v, the minimum drawn C, the neurons below -110 mV or above their spike ceiling (vpeak, or a homeostatic threshold if higher) plus 5 mV or non-finite, and for each such neuron its parameters, first out-of-range step, extremes, and the conductances, external current and refractory state at that first step. It also reads `enable_bounded_conductance_integration` back from each bridge's own core config and counts spikes emitted by cells after their first excursion (an upper bound on spurious spikes: it includes ordinary firing).
- Workload: dev seed 7, `intact_a` of conv, infer, ask, emo and inner, at main 3065c5614 plus the probe (commit 51e1b39bd; the `--aggregate` mode added afterwards only merges the five JSONs), numpy CPU, the gate's own frozen environment, serial under a memory cap. Observer cost 1 to 2 percent of wall time; peak RSS 6.5 to 8.2 GB.
- Artifact: `research/findings/raw/_membrane_census/dev/s7_intact_a_organs.json` (with its `.prov.json` sidecar) is the committed per-organ summary of the five arms, produced by `python -m research.runners.membrane_census --aggregate` over the five per-group outputs: per organ the neuron count, lowest v, offender counts per group, offender regions with their drawn-C range, minimum drawn C, maximum hazard index, first-reading classes and the bounded-conductance flag. The tables below are read from it. The five per-group raw JSONs (per-neuron rows with the conductances, external current and refractory state at the first reading, from which the medians and maxima quoted under Reading the table and What the offenders have in common come) are diagnostic and stay uncommitted on the workstation in the census_scratch folder under sim-rev (s7/<group>__intact_a.json), as with the diagnoses folder.

## Result: organs with out-of-range membranes (dev seed 7)

Organ is the code that steps the bridge most (the builder is kept in the JSON as `built_by`). Counts are neurons that left the range at least once, the maximum over the groups in which the organ appears. Steps are bridge steps in that group: an absence below is only as strong as the exposure.

| organ (stepper) | neurons | offenders (low/high) | lowest v mV (any group) | offending regions: n (drawn C pF) | min drawn C pF (region) | max (g_i+g_e)dt/C | first reading of each offender |
|---|---|---|---|---|---|---|---|
| content_selection_spiking SpikingLoopContextBuffer (loop-WM bridges) | 9360 | 3364 (3364/1911) | -89600 | dlpfc_wm 1882 (51-151), cortex_ctx 1384 (47-140) | 46.98 (cortex_ctx) | 53.78 | low with g>=2: 2219; high refractory: 1145 |
| learned_assoc_graph LearnedAssocGraph (store_fact pool) | 1800 | 307 (222/112) | -1836 | pool 202 (58-137), fs 105 (71-132) | 51.45 (pool) | 23.36 | low g>=2: 195; low other: 14; high refractory: 98 |
| common_ground_ledger_production_organ CommonGroundLedgerOrgan | 750 | 400 (400/400) | -24780 | led0-led4, 80 each (52-151) | 51.91 (led4) | 9.34 | high refractory: 400 |
| _gnw_active_overwrite_derisk workspace | 960 | 344 (344/301) | -16042 | workspace 300 (54-140), mm 16 (47-85), gate 28 (51-83) | 46.98 (mm) | 11.89 | high refractory: 244; low g>=2: 89; low other: 11 |
| _gnw_reentrant_metacog_gated_deliberation (inner only) | 390 | 320 (320/320) | -1339 | workspace 320 (100) | 100.00 | 3.67 | high refractory: 320 |
| _gnw_distributed_overwrite_workspace | 440 | 250 (250/185) | -11479 | workspace 250 (56-145) | 56.41 | 10.49 | high refractory: 185; low g>=2: 65 |
| biased_competition_buffer BiasedCompetitionContextBuffer | 1530 | 72 (72/38) | -1351 | cortex_ctx 40 (72-136), dlpfc_wm 32 (60-110) | 51.45 (sel_brain) | 6.27 | low g>=2: 68; high refractory: 2; low other: 2 |
| _phaseB_multicue_competition SpikingRoleCompetition | 1752 | 72 (48/72) | -4436 | sel_agent 24 and sel_patient 24 (100), sel_FS_agent 12 and sel_FS_patient 12 (20) | 20.00 (sel_FS_agent) | 5.54 | high refractory: 71; low g>=2: 1 |
| metacog pool bridge (majority stepper metacog_production_organ; the 72 offenders are role-competition cells, see the note below the table; ask only) | 7002 | 72 (48/72) | -8104 | the same four sel_* regions as the role-competition row (24 and 24 at 100, 12 and 12 at 20) | 20.00 (patient_expected) | 5.55 | high refractory: 72 |
| _p1_2_workspace_deliberation_loop | 390 | 80 (80/80) | -1356 | workspace 80 (100) | 100.00 | 3.44 | high refractory: 80 |
| emotion_category_organ EmotionCategoryOrgan (emo only) | 920 | 2 (2/0) | -123 | one bridge, no regions: 2 (48.8 and 51.1) | 46.98 | 2.76 | low g>=2: 2 |
| _curiosity_seek_learn_onbridge (merged pool) | 7002 | 1 (0/1) | high 105 | sel_FS_agent 1 (20) | 20.00 (patient_expected) | 0.48 | high refractory: 1 |
| brain_conversational_agent BridgeParser (shared one-brain substrate) | 54320 | 721 (454/440) | -20918 | onebrain_composer 721 (100) | 20.00 (patient_expected) | 0.46 | all 721 in the RF-masked slice (low other 364, high refractory 357) |

Metacog row: in `ask`, the 72 sel_* offenders on the metacog pool bridge coincide with `SpikingRoleCompetition._noun_role_rates` stepping that bridge (3120 steps, ask only). They are the same cells as bridge 13 (the role-competition bridge), with first excursions offset by about 19400 steps. They are the role-competition circuit on a second bridge, not the metacog organ. Metacog's own steps show 0 offenders in conv, infer and inner.

Reading the table:

- Groups (offenders / bridge steps): the learned-assoc graph and the role-competition buffer show the same counts in all five groups (307 and 72); the loop-WM, common-ground, GNW and biased-competition buffers appear in conv, infer, ask and inner with group-dependent counts (loop-WM 2429 to 3364); the emotion organ carries its two excursions only in `emo` (15580 steps; 0 in the other four at 13500 steps); the metacog pool bridge carries 72 offenders only in `ask` (28416 steps), and they are role-competition cells (see the metacog row note), while metacog's own steps show 0 offenders in conv, infer and inner (24880, 40920 and 25960 bridge steps); BridgeParser in conv, ask and inner (710, 677, 721) and not in infer or emo.
- The emotion organ's two excursions have the mechanism the diagnoses found on s101: two cells of the organ's one bridge at 48.8 and 51.1 pF with g_i about 103 and 113 nS, (g_i+g_e)*dt/C at the first reading 2.0 and 2.3, lowest v -123 mV, at steps 15155 and 15156, one spike from those cells afterwards. The census does not say whether the organ's read-out changed on this seed.
- Spikes by cells after their first excursion, as a share of the bridge's spikes (upper bound, includes ordinary firing): common-ground ledger 90 percent, loop-WM 73 percent, GNW active overwrite 50 percent, learned-assoc 36 percent, role competition 18 percent, the metacog pool bridge (role-competition cells) 8 percent, emotion organ 0 percent. These bridges are mostly short-lived or auxiliary, so the share says how much of their own output is affected, not how much of any reply.
- BridgeParser: the 721 offenders are all inside the `onebrain_composer` region, whose cells the composer drives as resonate-and-fire phasors through the RF mask (6880 of the region's 45856 cells are masked at the end; onebrain_merge_production.py lines 296 to 313 and 555 to 570 describe the slice as idle between composer ops and without synapses). Their |u| before the first v excursion has median about 6e4 and reaches 2.1e6, with g_i, g_e and the external current 0 (implied input current median 0). These are phasor state integrated by the Izhikevich step, an interaction to be aware of, not an organ membrane. The other 11 regions of that 47440-neuron pool stay between -60 and -55 mV over 7200 to 9600 steps.

## Result: organs with none (dev seed 7)

| organ (stepper) | neurons | steps (max over groups) | lowest v mV | min drawn C pF (region) | max (g_i+g_e)dt/C |
|---|---|---|---|---|---|
| attributed_parser AttributedBridgeParser | 224 | 36960 | -93.96 | 54.99 | 0.57 |
| _stageA_full_integration build_one_brain | 25531 | 40 | -68.44 | 39.02 (rf) | 0 |
| _self_schema_region | 1380 | 80 | -60.00 | 51.91 | 0 |
| _appraisal_interoceptive_ladder | 690 | 16340 | -70.57 | 51.91 | 0.29 |
| _graded_affect_attractor | 1044 | 4940 | -61.42 | 20.00 (inh_plus) | 0.46 |
| spiking_novelty_habituation_organ | 3520 | 1204 | -80.00 | 48.63 | 0.40 |
| _neuromod_spiking_da_mode (striatal) | 758 | 6080 | -88.59 | 20.00 (str_PV_FSI_N) | 2.07 |
| _da_write_gain_spiking | 80 | 22400 | -72.55 | 51.38 | 0 |
| category_inherit_organ | 960 | 1000 | -60.00 | 46.98 | 1.22 |
| one_brain_composer OneBrainComposer | 146 | 46200 | -60.00 | 57.64 | 0 |
| _emerge59 broca frame slots | 184 | 440 | -80.00 | 54.99 | 0 |
| _gnw_acc_bg_stop_trigger | 480 | 750 | -60.86 | 56.41 | 1.24 |
| _comprehension_learned_verbselects_cue (conv) | 80 | 75 | -59.83 | 51.38 | 0 |
| _comprehension_learned_animacy_spiking (conv, ask) | 80 | 375 | -59.83 | 51.38 | 0 |
| _affect_marker_wta (valence/arousal markers) | 2160 | 3520 | -89.12 | 54.99 | 2.32 |
| _spiking_expectation_rpe | 2184 | 39520 | -74.99 | 20.00 (patient_expected) | 1.91 |
| _affective_world_model (emo only) | 7002 | 7520 | -80.00 | 20.00 (patient_expected) | 0.62 |
| idle_replay_organ ReplayCircuit (inner only) | 504 | 660 | -79.35 | 55.13 | 1.91 |

Four of these sit at or just under the hazard line with no excursion on this draw (marker WTA 2.32, striatal DA mode 2.07, RPE and idle replay 1.91; lowest v about -89, -89, -75 and -79 mV): they are the ones to expect on another draw. Several have very few steps (40 to 440), so their clean result carries little weight.

## What the offenders have in common

First reading of every offender, counted exactly (the JSON rows are capped, these counts are not):

- Low with (g_i+g_e)*dt/C at or above 2 is the explicit-step conductance overshoot of the diagnoses (7 organs: loop-WM, learned-assoc, GNW active and distributed, biased competition, role competition, and the emotion organ). The conductance is excitatory-dominated in the loop-WM and role-competition buffers, inhibitory-dominated in the learned-assoc graph, the GNW distributed workspace and the emotion organ, and mixed in the GNW active and biased-competition buffers; over these 9977 rows (summed over the groups) the median g_e is about 330 nS and the median g_i about 300 nS (maxima 1086 and 1870 nS). NMDA conductance is non-zero at the first reading in 8879 of them (maximum 1686 nS) and is not in the hazard index, so the index understates the load.
- High with the cell inside its refractory window (11 organs) is a spike crossing withheld by `fired = v_new >= threshold & not_in_refractory`: nothing resets the cell, the quadratic term runs it far above vpeak, and on later steps it often goes far below -110 mV. It is the same strong-drive regime, not a separate one: over its 8383 rows (summed over the groups) the median (g_i+g_e)*dt/C at the first reading is 4.3 (only 466 rows are below 2) and the median implied input current is about 16 nA. The curiosity organ's single cell is the exception (about 3 nA, no conductance).
- Low with small conductance is a minority outside the RF slice (14 learned-assoc cells, 11 GNW-active cells, 2 biased-competition cells).

C is not the discriminator: the offending cells of the heterogeneity-drawn regions span 47 to 151 pF; only the 20 pF fast-spiking preset cells (sel_FS_*) are lower, and those three organs' sel_FS offenders are the withheld-crossing kind. The organ's maximum hazard index over the run is 2.76 or more for 11 of the 12 membrane offenders (the curiosity organ is 0.48), against at most 1.24 for 14 of the 18 clean organs and 1.91 to 2.32 for the other four.

All 189 bridge instances report bounded conductance integration OFF and the Izhikevich model; the BG neural-seam organ (where the diagnoses report it ON) does not appear among the stepped organs, so no bridge here was protected by that guard.

## Controls and validity

- Known positive: the emotion organ alone (organ-level, not a gate run) on the pinned failing non-gate seeds 202 and 242 is flagged, and with the repo's bounded-conductance guard set at construction the seed-202 count is 0. Organ-level dev seed 7 is clean unbounded (lowest v -102 mV) and bounded. This reproduces the diagnoses finding's mechanism with an independent instrument. These control runs are uncommitted scratch, run with the probe at an earlier commit (the files `control_emo_*.json` in the folder /home/dant123/Projects/sim-rev/census_scratch, workstation only), and the numbers are quoted as reported: seed 202 unbounded 1 offender (idx 363, the joy-appraisal cell, C 39.65 pF, lowest v -160.21 mV, g_i 93 nS); seed 202 bounded 0 offenders, lowest v -81.62 mV; seed 242 unbounded lowest v -149.21 mV with C 47.79 pF (the sadness-appraisal cell).
- Observer purity, what was shown: a unit test hashes final v and u of a tiny bridge with and without the census (identical), plus a reviewer's one-turn check on emo/intact_a at dev seed 7. Census-versus-census rerun identity (the next bullet) is determinism, not purity.
- Reproducibility: every group was run at least twice on the same seed with the probe at different commits (later commits only added fields; emo and inner were run four times); the replies (sha256 over all turn responses) and all per-organ counts, extremes and spike totals were identical across runs for all five groups. The positive and negative controls above ran with the probe at an earlier commit.
- Wrapper removal: after each run the class method and `__init__` were the originals again (recorded in the JSON).

## What this does not show

- Whether any excursion changes a reply. No lesion or integrator-swap arm was run; the census only says where the integrator leaves the range. Per the load-bearing rule, the next step for an offender that matters to a reply is a bounded-conductance or sub-step arm compared at the reply level.
- Anything about seeds other than 7. One dev seed is a census of one draw, and the seed-to-cell map is specific to the numpy CPU random stream; cupy would draw a different network. The 60 pool lines named under Next are the replication.
- Coverage beyond `_run_one_simulation_step`: `rf_resonate_steps` (phasor windows) is not observed.
- A "high" reading is a crossing the refractory timer withheld, not a spike the brain emitted; the spurious spikes the diagnoses describe follow the low excursions, and the spike shares above are an upper bound on them.
- Which of the 31 organs are on a given reply's path. All were stepped by the live handler; their role in the answer was not traced.

## Next (not done here)

- Replication: `research/coordination/membrane-census-pool-lines.txt` holds 60 pool lines in the Stage-2 shape (TSV: command, then a checked-reason carrying `mem_gb=9 max_age_h=240`; dev seeds 1 to 12 times the five groups, `intact_a`, `<sha>` left as the literal placeholder, mem_ok 9, memcap 10, outputs under research/findings/raw/_membrane_census/dev/s<seed>/). At dev seed 7 the five arms took 8 to 14 minutes each, so the set is about 11 lane-hours. Gate seeds are refused by the probe unless a flag is given; none was.
- For the offenders on the production reply path (common-ground ledger, role-competition cells on the metacog pool bridge, loop-WM and learned-assoc buffers first), a paired bounded-conductance or sub-step arm with reply-level comparison, in the order the stage-exit calendar allows.
- The probe's JSON also supports the BG step-1 grid and the emotion-organ fix checks as a before/after instrument (zero offenders is the pass condition for those organs).
