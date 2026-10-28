# STEP1-SPEC — BG seam recalibration (step 1) — DRAFT for Claude review

host-decision: none — the spec only pins runner env (tonic currents, preset cells, sub-step rule) for the registered scorer; no host code computes weights, spikes, rates or readouts — all of that stays in the spiking substrate

**Status: PROVISIONAL.** Design draft only (RULING batch-1024 item 8, answering NEEDS [bg-tonic-loop-0]). **Run nothing until Claude rules.** **The spec must be reviewed by 2026-10-27 or the 11-03 window is lost** (the BG commit seam's stage-2 gate window 11-03..11-10 holds only then: PACKET.md sec 5; bg-tonic-loop-0-RESULT.md sec 5). No default, prereg or registered bar is changed by this document.

## 1. Target and non-target

**Target:** the BG seam's own REGISTERED functional bars — gates G1-G5 and every precondition of research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md as amended by research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md — scored by the registered scorer (`research/runners/host_share_bg_seam.py`, `--score --commit`, verdicts read from score.json) on **preset-centred cells** (`BRAIN_BG_PRESET_CELLS=1`: the 7 organ presets with Part C preset-centred jitter, commit 71efc4a5c) with **`BRAIN_BG_BOUNDED_COND=1`** on every run (ruling batch-1024 item 8; bounded-conductance fixed ON per ruling -1020a rule 2 after the step-0 rule FIRED: A0p sub-reversal steps 8319, all bounded arms 0 — bg-tonic-loop-0-RESULT.md sec 1).

The registered bars, verbatim and unmodified (PREREGISTRATION.md 'Gates'; AMENDMENT.md items 4-5):

| bar | measure (registered scorer) | registered bar |
|---|---|---|
| G1 production host owns the reply | F: the reply follows the flipped host decision (prod_intervene vs prod_a) | F >= 0.8 |
| G2a neural seam holds empty turns | fraction of hsb_empty1/hsb_empty2 events held in seam_b | >= 0.75 |
| G2b neural seam never silences a question | fraction of hsb_q events held in seam_b | = 0 |
| G3 the neural route causes the reply | fraction of target events whose reply changes with BRAIN_BG_NEURAL_SEAM_LESION=value | >= 0.3 |
| G4 commit read threshold-invariant (amended) | fraction of seam_b neural-seam events whose committed action under the commit read changes at READ_FRAC x0.5 or x2, or that carry no commit record | = 0 (no event changes) |
| G5 the bus argmax does no work | fraction of norgan_hop calls (prod_b + seam_b) where the unique-ignition read equals the argmax read | >= 0.95 |

Preconditions — a failure makes the verdict UNDEFINED, never NO-GO (PREREGISTRATION.md 'Preconditions'; AMENDMENT.md item 5): the sham null (prod_a == prod_b on the decision fields and the answer) on every target event; membrane in [-150, +100] mV on every neural-seam trial; at least 6 events per gated measure; every neural-seam event carries a commit record (commit_engaged = 1.0); exactly one neural-seam event per seam-arm turn (12 per seed: 12 x 6 = 72 on the gate block; on the 8-seed calibration block the ported count is 12 x 8 = 96 — open question 7); commit.op == 4 on every event; commit.organ_seed equals the shard's chat seed.

**Explicitly NOT the target: agreement with the A0 hybrid baseline.** The hybrid cells were the defect (PACKET.md sec 1: global RS-centred heterogeneity overwrote every preset, and every constant in the organ — GPe 100 pA, STN 40-50 pA, GPi 1000 pA, thalamus 900 pA, practice 1000 pA, reset 1200 pA, and the weights — was tuned on hybrids). Step 0's A0p-vs-A0 and A5p-vs-A0 decision-field diffs (bg-tonic-loop-0-RESULT.md secs 1-2) are the diagnosis that made recalibration the method; no step-1 score, selection key or STOP compares any set to A0, and no hybrid arm runs in step 1.

Reported context, never bars: the registered tonic-rate bands GPe 20-60 Hz and STN 10-30 Hz (bg-tonic-dev-1019.md, I_tonic rule), used only as tie-break input (section 4 key 2); the PACKET.md sec 4 per-cell criteria (rate band, ISI-CV, no silent cells), whose thresholds no source quantifies — reported, not gated (open question 4); the published isolated-cell references STN 5-15 Hz, GPe 30-80 Hz, GPi 40-80 Hz with citations (bg-tonic-loop-0-RESULT.md sec 4; bg_loop_part_c.py REFERENCES) as context only.

## 2. Free parameters

Every knob the seam/recalibration can touch, with the SOURCE of its range. Where the record gives no range the row says 'no source found — Claude to set' and the parameter is HELD at its current value in step 1; no range is invented (ruling batch-1024 item 8).

### 2a. Varied in the step-1 grid (sourced ranges)

| parameter | current value | proposed range | SOURCE of the range |
|---|---|---|---|
| `BRAIN_BG_TONIC_STN_PA` / `stn_tonic_pA`, tonic current on `selector_stn` | 50 pA | {50, 100, 200, 400} pA | levels drawn from the declared I_tonic grid {25, 50, 100, 200, 400, 700, 1000} pA (bg-tonic-dev-1019.md preconditions); preset-centred STN f-I reads 10.6/19.1/37.0 Hz at 100/200/400 pA against the 10-30 Hz band (bg-tonic-loop-0-RESULT.md sec 4); preset-centred STN is silent at 40-50 pA in-organ (bg-tonic-loop-partA-INFO.md); the hybrid cliff ramp 30-40 pA (bg-tonic-stn-1020-RESULT.md) shows lower levels waste cells |
| `BRAIN_BG_TONIC_GPE_PA` / `gpe_tonic_pA`, tonic current on `gpe_{ch}` | 100 pA | {50, 100} pA | same declared grid (bg-tonic-dev-1019.md preconditions); 100 pA is the registered pick (30.4-31.9 Hz on the calibration bridge; GPe in band on 8/8 dev seeds — bg-tonic-dev-1019.md); preset-centred GPe f-I reads 86.1 Hz at 100 pA, ABOVE the 20-60 Hz band (bg-tonic-loop-0-RESULT.md sec 4), so one lower declared-grid level brackets it |

### 2b. Free but HELD at current value in step 1 (no sourced range)

| parameter | current value | proposed range | SOURCE of the current value |
|---|---|---|---|
| `gpi_tonic_pA`, tonic current on `gpi_{ch}` | 1000.0 pA | no source found — Claude to set; held | _vocal_action_selector_gate.py:50 (branch research/bg-tonic-loop); 'GPi 1000 pA ... tuned on hybrids' (PACKET.md sec 1); its output is what the commit records tabulate as gate/motor counts (bg-tonic-dev-1019.md condition 3) |
| `thalamus_tonic_pA`, tonic current on `thal_{ch}` | 900.0 pA | no source found — Claude to set; held | _vocal_action_selector_gate.py:51; PACKET.md sec 1 |
| practice_arousal tonic current | 1000 pA | no source found — Claude to set; held | PACKET.md sec 1 |
| selector_reset tonic current | 1200 pA | no source found — Claude to set; held | PACKET.md sec 1 |
| hold tonic current (neural-seam hold region) | 350 pA | no source found — Claude to set; held | PREREGISTRATION.md 'Fixed build' |
| heard word-rendering current | 500 pA (12 neurons per word) | no source found — Claude to set; held | PREREGISTRATION.md 'Fixed build' |
| heard -> str_d1 weight (density 0.5, slow NMDA, gate bg_value_afferent) | 20 | no source found — Claude to set; held | PREREGISTRATION.md 'Fixed build'; the weights were tuned on hybrids (PACKET.md sec 1) |
| motor -> latch weight | 40 | no source found — Claude to set; held | PREREGISTRATION.md 'Fixed build' |
| latch internal weight (30 RS, density 0.5, NMDA) | 64 | no source found — Claude to set; held | PREREGISTRATION.md 'Fixed build' |
| latch FS cross weight | 10 | no source found — Claude to set; held | PREREGISTRATION.md 'Fixed build' |
| `arousal_to_proposal_weight` | 1.0 | no source found — Claude to set; held | _vocal_action_selector_gate.py:55 |
| `proposal_to_fsi_weight` | 100.0 | no source found — Claude to set; held | _vocal_action_selector_gate.py:56 |
| `fsi_to_msn_weight` | 10.0 | no source found — Claude to set; held | _vocal_action_selector_gate.py:57 |
| preset-centred jitter width: a lognormal sigma_log 0.3 (gaussian std 0.3 abs(a) when a is not positive), b gaussian std 0.25 abs(b), d gaussian std 0.25 abs(d) (10.0 when d = 0), C gaussian std 0.15 C, ratio clamp [0.1, 3.0] | the Part C defaults | multiplier 1.0 held in step 1 (the population identity must match step 0's cells); any other width: no source found — Claude to set | bg_loop_part_c.py docstring (draw specs from sim/bridge.py:4094-4100; ratio clamp per docs/drafts/preset-centred-het/CENSUS.md line 166 and tests/test_preset_centred_heterogeneity.py:209) |

### 2c. Fixed for step 1 (not free: changing any row voids a ruling, a prereg or the population definition)

| item | fixed value | SOURCE |
|---|---|---|
| `BRAIN_BG_BOUNDED_COND` | 1 on every step 1 run | the step-0 rule FIRED (sub-reversal steps A0p = 8319 vs 0 in every bounded arm — bg-tonic-loop-0-RESULT.md sec 1); ruling -1020a rule 2 |
| `BRAIN_BG_PRESET_CELLS` | 1 (preset-centred cells are the population) | ruling batch-1024 item 8; the hybrid cells were the defect (PACKET.md sec 1) |
| `BRAIN_BG_IZH_SUBSTEPS` | 4 on every step-1 run | ruling emo-bg-merge-1026 (3)(c); finding research/findings/2026-10-26-izh-membrane-substeps-derisk.md (0 breach cells on 8/8 dev seeds at n=4) |
| `BRAIN_BG_IZH_SUBSTEP_RULE` | model_reset on every step-1 run | ruling emo-bg-merge-1026 (3)(c); the sub-step option's registered rule |
| E_inh, inhibitory synaptic reversal | -75 mV | PACKET.md sec 1 |
| sub-reversal bound | steps with v below E_inh - 5 mV, counted per region | PACKET.md sec 3 Part A; bg-tonic-loop-0-RESULT.md sec 1 |
| tonic rate targets (tie-break bands only) | GPe 20-60 Hz, STN 10-30 Hz | bg-tonic-dev-1019.md I_tonic rule (registered; not to be moved) |
| spiking commit stage | BG_COMMIT_OP = 4 (row 4): NG 12, NGI 8, NS 20, W_XG 30, W_GG 20, W_GGI 60, W_GIG 60, W_GS 12, W_SS 60, W_GIS 60, W_RESET 16, READ_STEPS 30, READ_FRAC 0.03 | AMENDMENT.md item 1 — frozen; any change to the row, to BG_COMMIT_OP or to the trial schedule voids the amendment |
| trial schedule and reads | drive 300 ms, hold 200 ms, latch read last 100 ms at 0.02 spikes/neuron/ms; commit read last 30 ms at READ_FRAC 0.03 (18-spike threshold) | PREREGISTRATION.md 'Fixed build'; AMENDMENT.md items 1-2 |
| the 7 organ presets (IZH2007_STN_BURST, IZH2007_GPE_PACEMAKER, IZH2007_GPI_OUTPUT, IZH2007_THALAMIC_RELAY, IZH2007_STRIATAL_MSN_D1, IZH2007_STRIATAL_MSN_D2, IZH2007_FS_CORTICAL_INTERNEURON) | the preset parameter values themselves | presets are the jitter centres (bg_loop_part_c.py); partC-1021: mismatch is reported, never tuned (bg-tonic-loop-0-RESULT.md sec 4) |

## 3. Calibration grid (one grid)

**Population:** every run uses preset-centred cells with bounded conductance — `BRAIN_BG_PRESET_CELLS=1`, `BRAIN_BG_BOUNDED_COND=1`, `BRAIN_BG_TONIC_DRIVE=1` — i.e. the step-0 A5p cell definition (bg-tonic-loop-partA-INFO.md): the 7 organ presets with Part C preset-centred jitter (bg_loop_part_c.py).

**Seeds: 71-78.** Disjoint from dev (seeds 1-15 and 7; treated as 1-16 so step-0's seed 16 is excluded too) and from the gate six 42/43/44/100/101/102. Justification: ruling batch-1024 item 8 requires a block outside both; a grep of research/ on 2026-10-24 shows 71-78 unused (adjacent blocks are taken: 19-24 lane dev restocks, 25-40 cs-race CONFIRM, 45, 49, 61-70 deixis calibration, 200-205 stageA/vision). Eight seeds mirror step 0's shape, where 8 seeds exposed the defect on all 8 (bg-tonic-loop-0-RESULT.md sec 2). Gate seeds are never run in step 1.

**Axes and levels (the full product; every level is drawn from the declared I_tonic grid {25, 50, 100, 200, 400, 700, 1000} pA — bg-tonic-dev-1019.md preconditions):**

- `stn_pA` in {50, 100, 200, 400}: 50 pA is today's anchor; 100/200/400 pA bracket the preset-centred STN f-I crossing of the 10-30 Hz band (10.6/19.1/37.0 Hz — bg-tonic-loop-0-RESULT.md sec 4).
- `gpe_pA` in {50, 100}: 100 pA is today's anchor (the registered pick, bg-tonic-dev-1019.md); 50 pA is the next lower declared-grid level (preset-centred GPe reads 86.1 Hz at 100 pA, above the 20-60 Hz band).

**The grid = the 8 sets** of the 4 x 2 product, enumerated lexicographically — stn_pA ascending outer, gpe_pA ascending inner: (50,50), (50,100), (100,50), (100,100), (200,50), (200,100), (400,50), (400,100). Each set is applied per run through `BRAIN_BG_TONIC_STN_PA` / `BRAIN_BG_TONIC_GPE_PA` with `BRAIN_BG_TONIC_DRIVE=1` (env overrides in _vocal_action_selector_gate.py, branch research/bg-tonic-loop; recorded in every artifact's .prov.json as in bg-tonic-dev-1019.md).

**Sub-step re-run (RULING emo-bg-merge-1026 (3)(c)).** The grid re-runs with `BRAIN_BG_IZH_SUBSTEPS=4` and `BRAIN_BG_IZH_SUBSTEP_RULE=model_reset` on every run. The option shifts rates (GPi +18.6%) and changes 3 of 20 clean-seed events, so NOTHING carries over from the n=1 grid: it is a new measurement, scored by the registered scorer with the membrane precondition read first. Gate seeds are never run in step 1.

**Cells and cost.** Per set: one isolated-cell pre-screen in the bg_loop_part_c.py shape (7 presets x 13 currents 0-1200 pA x 20 preset-centred jittered cells; numpy, minutes per set — the Part C budget, PACKET.md sec 5 and bg-tonic-loop-0-RESULT.md sec 5), reported beside the set's rates; and one seam cell per seed — one `host_share_bg_seam --run-all --seed <s>` invocation = 7 artifacts at about 4400 s wall (recorded canaries: bg-tonic-dev-1019.md preconditions 4470 s; 2026-10-14-bg-live-seam-commit-RESULT.md evidence 4409 s). Total 8 sets x 8 seeds = 64 seam cells + 8 pre-screen cells. The 64-cell cap is twice step 0's Part B volume (224 artifacts = 32 cells — bg-tonic-loop-0-RESULT.md sec 5) and is the budget STOP in section 5.

**Scoring.** Per set, the registered scorer over all 8 seeds: `host_share_bg_seam --score --seeds 71 72 73 74 75 76 77 78 --commit`, verdict read from score.json (AMENDMENT.md item 6). Run and score commands are copied verbatim from PREREGISTRATION.md 'Run and score' and AMENDMENT.md 'Run and score' with only the seed block and the two env currents substituted; no bar, arm or schedule changes with them.

### 3.9. Addition A (RULING batch-1024b): region table at in-organ tonic, report only

Built from the EXISTING Part C f-I data (bg-tonic-loop-0-RESULT.md section 4; canary run bg_loop_part_c_71efc4a5c_pool41 20261021T022651, rc=0) with no new runs. In-organ tonic = what `_set_equal_tonic_current` injects (`research/runners/_vocal_action_selector_gate.py:408-431`): gpi/thal always; gpe/stn under `BRAIN_BG_TONIC_DRIVE` at the registered 100/50 pA; the MSN D1/D2 and cortical FS interneuron presets receive no injected tonic term (0 pA).

| preset | in-organ tonic | preset-centred isolated rate | registered band | published context band | check |
| --- | --- | --- | --- | --- | --- |
| IZH2007_GPI_OUTPUT | 1000 pA | 279.0 Hz | - | 40-80 | 2.5x beyond the 80 ceiling (> 2x): **FIRST next candidate if STOP 1 fires** |
| IZH2007_THALAMIC_RELAY | 900 pA | 70.4 Hz | - | no source found | n/a |
| IZH2007_STN_BURST | 50 pA (registered pick) | in-organ SILENT at 40-50 pA (bg-tonic-loop-partA-INFO.md); isolated grid 0.0 @ 0 pA, 10.6 @ 100 pA (50 pA is off the 13-point grid) | 10-30 | 5-15 | below both, at most 1x beyond the 5 floor: reported, not named |
| IZH2007_GPE_PACEMAKER | 100 pA (registered pick) | 86.1 Hz | 20-60 (1.4x above) | 30-80 (1.1x above) | within 2x |
| IZH2007_STRIATAL_MSN_D1 | 0 pA (no injected tonic) | 0.0 Hz | - | no source found | n/a |
| IZH2007_STRIATAL_MSN_D2 | 0 pA (no injected tonic) | 0.0 Hz | - | no source found | n/a |
| IZH2007_FS_CORTICAL_INTERNEURON | 0 pA (no injected tonic) | 0.0 Hz | - | no source found | n/a |

Report only; no gate, bar or selection rule reads this table. Named as the first next candidate if STOP 1 fires: **GPI at 279.0 Hz, 3.5x the published 40-80 ceiling** (the only region more than 2x outside its published band).

## 4. Frozen selection rule (stated before any run; deterministic)

The winning set is chosen from the sets that completed all 8 seam cells with every precondition holding (a set with a precondition failure is UNDEFINED and cannot be selected — the prereg verdict logic). Keys, applied lexicographically:

1. **Most registered bars passed.** The count of {G1, G2a, G2b, G3, G4-amended, G5} true in the set's pooled 8-seed score.json, each judged at its registered threshold (section 1). Higher is better. Verdicts are read from the score artifact, never recomputed (2026-10-14-bg-live-seam-commit-RESULT.md).
2. **Closest tonic-rate match.** Smallest d = the sum over all recorded trials of the distance in Hz outside the registered bands (below 20 or above 60 for GPe; below 10 or above 30 for STN; 0 inside), read from the persisted `tonic_rates_hz` records (bg-tonic-dev-1019.md). Lower is better.
3. **Lowest parameter norm.** Smallest sum over the varied parameters of ((v - v0) / v0)^2 with v0 = 50 pA (stn_pA) and 100 pA (gpe_pA) — among equals, prefer the set closest to today's constants.
4. **Final tie-break.** The lexicographically smallest (stn_pA, gpe_pA) tuple in the section 3 enumeration order.

Keys 1-4 form a total order, so no further tie-breaking can arise. All keys are computed in one deterministic pass over the score artifacts and the persisted rate records. Declared before the data (the lesson of the amendment's declared deviation, AMENDMENT.md 'Operating point'): if this rule selects nothing, that is a STOP (section 5), never a re-pick under a new rule.

## 5. STOP rules

1. **No set passes all six registered bars.** If the grid completes (8 sets x 8 seeds) and no set passes G1-G5 with all preconditions holding, step 1 STOPS with a documented negative plus a NEEDS line to Claude (ruling batch-1024 item 8). The negative is a verdict on THIS METHOD — re-deriving the tonic drives on preset-centred cells with BRAIN_BG_BOUNDED_COND=1 over this grid — never a licence to drop the capability (CLAUDE.md non-negotiable 3, no defer). The NEEDS line names the next candidates already in the record: STN pacing (P), the STN-GPe loop (L), hyperdirect input (PACKET.md sec 4), and the held parameters of section 2b.
2. **Budget cap.** 64 seam cells (8 sets x 8 seeds) and 8 isolated pre-screen cells. At the cap, stop and file; no extension without a new Claude ruling.
3. **Cell death.** A cell that dies without output is requeued VERBATIM exactly once with a canary on record (the step-0 rule, bg-tonic-loop-0-RESULT.md sec 0). A second death on the same cell STOPs that set, which is reported as incomplete — never silently dropped.
4. **Precondition failure.** A run whose membrane leaves [-150, +100] mV, whose event counts miss the amendment's preconditions (section 1), or which again shows sub-reversal steps below E_inh - 5 mV makes its set UNDEFINED. If every set is UNDEFINED, STOP and report which regions before any bar uses v (PACKET.md sec 3 decision clause 2).
5. **Identity regression.** The flag-off byte-identity and construction-guard tests (tests/test_bg_tonic_loop_identity.py; tests/test_bg_neural_seam.py per PREREGISTRATION.md 'Construction guards') must pass on the run revision before any cell dispatches; a failure is a bug to fix before scoring and STOPs dispatch.
6. **Calendar.** If the grid cannot complete before the stage-2 gate window opens (11-03), STOP and hand the calendar reading to Claude (PACKET.md sec 5) rather than run a gate seed on a recalibrating cell.

## 6. Provenance & honesty

- **Run nothing until Claude rules** (ruling batch-1024 item 8). This document is a design draft; no experiment or side-effecting command was run to produce it.
- **Status: PROVISIONAL** until verify-go and Claude review. **The spec must be reviewed by 2026-10-27 or the 11-03 window is lost** (stage-2 gate window 11-03..11-10: PACKET.md sec 5; bg-tonic-loop-0-RESULT.md sec 5).
- **Functional bars only.** Every measure — rates, latch and gate counts, win/hold counts, membrane range, abstention — is a functional read-out of the seam (CLAUDE.md non-negotiable 5); nothing here supports a claim of phenomenal experience.
- **Nothing registered is changed.** No default, no prereg, no bar. The bars are cited as found (PREREGISTRATION.md gates; AMENDMENT.md items 4-6); step 1's only degrees of freedom are the grid parameters of section 3 and the calibration seed block. Gate seeds 42/43/44/100/101/102 and dev seeds are untouched.
- Sources read and cited: research/coordination/bg-tonic-loop-0-RESULT.md (secs 0-5); docs/drafts/bg-tonic-loop/PACKET.md (secs 1-5); research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md (Fixed build, Gates, Preconditions, Run and score); research/findings/2026-10-02-bg-live-seam-retirement-RESULT.md (six-gate table); research/findings/2026-10-14-bg-live-seam-commit-RESULT.md (gates, evidence, canary walls); research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md (items 1-6, Operating point); research/coordination/bg-tonic-loop-partA-INFO.md (arm definitions, STN reads); research/coordination/bg-tonic-stn-1020-RESULT.md (cliff ramp); research/coordination/bg-tonic-stn-1019-RESULT.md (sweep STOP);
  research/coordination/bg-tonic-dev-1019.md (I_tonic rule and grid, pick, rate bands, preconditions); research/runners/bg_loop_part_c.py and research/runners/_vocal_action_selector_gate.py (branch research/bg-tonic-loop); sim/bridge.py (preset-centred heterogeneity, lines 3620-3699).
