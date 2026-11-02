---
type: preregistration
status: draft
date: 2026-11-01
mechanism: bg-live-seam-retirement, amendment 2 (the gate run on preset-centred, bounded, sub-stepped cells at stn200_gpe100)
amends: research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md (as amended by research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION AMENDMENT DRAFT only. Not in force until Claude signs it (section 9). No gate seed of this build has run.
runner: research/runners/host_share_bg_seam.py
---

# Amendment 2 (DRAFT) to the BG live-seam retirement: the gate on stn200_gpe100

host-decision: none — this amendment pins runner env and adds report-only instrument fields; every decision stays in the spiking organ (heard/hold -> D1 race, GPi/thalamus gate, spiking commit stage). The residual host steps are the ones the 10-14 amendment declared (word rendering, the trial schedule, the commit read threshold, the env flags).

bar-sanity: N/A no new bar is registered here; every bar is the 09-29 / 10-14 registered bar, verbatim (the 10-14 G4 record is docs/drafts/bg-commit/bar_sanity_record_bg_commit_g4.json)

**Status: DRAFT for Claude to sign.** Written by a build agent (RULING bg-gate-1101). It changes no bar, no arm, no
turn, no schedule and no frozen row. It is not in force until section 9 carries Claude's signature, and no gate line
may be queued before that.

## 1. What this amendment does, in one paragraph

The step-1 recalibration grid (`docs/drafts/bg-tonic-loop/STEP1-SPEC.md`; artifacts
`research/findings/raw/_bg_live_seam/step1/507b9c789c/`, 8 sets x seeds 71-78, every run with `--commit`) passes
every registered bar on every set (DEV, not a verdict), and the frozen selection rule (STEP1-SPEC section 4) picks
`stn200_gpe100`. This amendment pins that operating point for the six gate seeds, keeps every registered bar verbatim,
says up front what a GO would and would not credit, and adds report-only same-seed ablation arms that test the
credit claim directly.

## 2. The pin

- **Revision (the pin):** `218a3378c873ea38415a6aa7f62effc21b410ff1` on branch `research/bg-gate-amend-1101` (based on `research/bg-tonic-loop` head
  `507b9c7890e680ae5d607f9dff4d49cad33d3944`, the step-1 grid's revision, plus the report-only instrument of section 6
  at `1b53d62340ad5a54291ba758cbe03524a354ea1b` and Claude's STOP-4 / STOP-5 rulings applied at `218a3378c873ea38415a6aa7f62effc21b410ff1`; the earlier draft pin `1b53d62340ad5a54291ba758cbe03524a354ea1b` is superseded).
  The branch head after the pin adds only this document and the pool-line TSV; neither is read at run time.
- **Environment, every gate line (exactly the step-1 grid's, with the selected set's two currents):**

```
BRAIN_BG_PRESET_CELLS=1 BRAIN_BG_BOUNDED_COND=1 BRAIN_BG_TONIC_DRIVE=1 BRAIN_BG_IZH_SUBSTEPS=4
BRAIN_BG_IZH_SUBSTEP_RULE=model_reset BRAIN_BG_TONIC_STN_PA=200 BRAIN_BG_TONIC_GPE_PA=100
```

  plus `PYTHONHASHSEED=0 SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES=` and the runner flags `--commit --allow-gate-seeds`.
- **Lines and score (the 10-14 amendment's 'Run and score' with the env and out root substituted):** one pool line
  per gate seed, `research/coordination/bg-gate-1101-pool-lines.tsv` (arm `gate`); out root (`<OUT>` below) is
  `_bg_live_seam/commit_gate/218a337_preset` under `research/findings/raw`; the score file is written after the run,
  so it is named only inside the command:

```
.venv/bin/python -m research.runners.host_share_bg_seam --score <OUT> --seeds 42 43 44 100 101 102 --commit \
  --out <OUT>/score.json
```

  The verdict is read from that score file.
- **Unchanged:** the four turns, the seven arms, `BG_COMMIT_OP = 4` and the frozen row of the 10-14 amendment item 1,
  the trial schedule (drive 300 ms, hold 200 ms, latch read last 100 ms, commit read last 30 ms at READ_FRAC 0.03),
  the scorer's bars (`BARS`, `BARS_AMENDED`) and `verdict()`.

## 3. Declared up front: what a GO credits, and what it does not

**A GO credits the spiking commit stage and the value route (heard/hold -> D1) on preset-centred, bounded-conductance,
sub-stepped cells. It does NOT credit the tonic STN/GPe drive.** The step-1 record shows the tonic lever is inert on
this organ (every number in this section is derived from the step-1 grid's artifacts, which live on S main from
commit e54ea844b onward and not on this branch: the eight `step1/507b9c789c/<set>/score.json` files and their
`s71`-`s78` seam_b shards under `research/findings/raw/_bg_live_seam/`):

<!--derived-->
- seam_b committed winners are identical across all 8 sets on 31 of 32 events (seeds 71-78 x 4 turns); the one
  varying event (s71 `hsb_empty1`) alternates between no commit and SPEAK with no order in the currents;
- motor -> commit gate counts barely move with the currents (relative spread of `gate_counts_drive` across sets:
  median 3.8% per event);
- GPe reads 69.1 Hz mean at GPe 100 pA whatever the STN current (50, 100, 200 or 400 pA) and 0.0 Hz at GPe 50 pA,
  while STN climbs 3.6 -> 34.2 Hz: STN does not reach GPe (the STN -> GPe pathway is absent; PACKET.md section 4 (L));
- the registered bars agree to the event across sets: F 1.0, empty_held 0.9375, content_held 0.0,
  commit_thr_change 0.0, bus_unique_agree 1.0 on all 8 sets; ns 0.4062 on six sets and 0.375 on the two STN 400 sets.

So the selection rule's key 1 (bars passed) tied all 8 sets, and `stn200_gpe100` was chosen by a later key (key 2,
the tonic-rate match, onward), not by any bar. The pallidal capability (tonic GPe/STN/GPi shaping the hold) is therefore NOT tested by this gate; it gets its
own method ladder (section 8). Recorded for the reader of G3: on the selected set G3's numerator is 13 of 32 events,
of which 9 are empty-turn hold losses, 2 are content-turn wrong holds and 2 are one-word turns spoken intact and held
under the lesion (scorer diagnostic of section 6c, recomputed on the step-1 shards; dev seeds, not a verdict).

## 4. The registered bars, verbatim (unchanged)

From research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md, 'Gates':

| gate | measure | bar |
|---|---|---|
| G1 production host owns the reply | F: the reply follows the flipped host decision (prod_intervene vs prod_a), lane C's definition | F >= 0.8 |
| G2a neural seam holds empty turns | fraction of `hsb_empty1/2` events held in `seam_b` | >= 0.75 |
| G2b neural seam never silences a question | fraction of `hsb_q` events held in `seam_b` | = 0 |
| G3 the neural route causes the reply | fraction of target events whose reply changes with `BRAIN_BG_NEURAL_SEAM_LESION=value` | >= 0.3 |
| G4 the body's latch read is a transducer | fraction of `seam_b` neural trials whose committed action changes at latch threshold x0.5 or x2 | <= 0.10 |
| G5 the bus argmax does no work | fraction of `norgan_hop` calls (prod_b + seam_b) where the unique-ignition read equals the argmax read | >= 0.95 |

"Preconditions (a failure makes the verdict UNDEFINED, never NO-GO): the sham null (`prod_a` == `prod_b` on the
decision fields and the answer) holds on every target event; every neural-seam trial keeps the membrane in [-150,
+100] mV (checked every step, including reset and washout); every gated measure has at least 6 events."

From research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md, item 4 (G4 as amended, which replaces the 09-29 G4
as the gated read):

"**G4, amended.** Measure: the fraction of `seam_b` neural-seam events whose committed action under the commit read
changes at READ_FRAC x0.5 or x2, or that carry no commit record. **Bar: 0 (no event changes).** The 09-29 G4 measure
of the old latch read (now against the old read's own choice) is reported, not gated."

And item 5 (new preconditions, UNDEFINED never NO-GO): "(a) Every neural-seam event of the three seam arms carries a
commit record, in shards run with `BRAIN_BG_COMMIT=1` (`commit_engaged` = 1.0). ... (b) Exactly one neural-seam event
per seam-arm turn: 12 x 6 = 72 (`commit_event_count_ok`). ... (c) `commit.op == 4` on every event. (d)
`commit.organ_seed` equals the shard's chat seed on every event. (e) G4 has at least 6 events."

The verdict rule is the 10-14 amendment's 'Outcome -> verdict' table, unchanged.

## 5. Pre-queue requirements (no gate line dispatches until all hold)

1. **STOP-5 identity tests on the pin.** `tests/test_bg_neural_seam.py` (the prereg's construction guards),
   `tests/test_bg_tonic_loop_identity.py` (STEP1-SPEC STOP 5), `tests/test_bg_commit.py` and
   `tests/test_bg_gate_amend_1101_instrument.py` all pass at `218a3378c873ea38415a6aa7f62effc21b410ff1` (CPU, numpy), together with
   `tests/test_host_share_bg_seam_score.py` and `tests/test_izh_membrane_substeps.py` (79 passed on the workstation,
   where PyOpenGL imports; the identity test also passes with OpenGL blocked).
   **STOP-5, ruled (Claude, 2026-11-01) and applied at the pin.** `test_flag_off_identical_to_golden` failed on the
   workstation because the bridge carried two arrays the golden lacks (`cp_synapse_pulse_progress`,
   `cp_synapse_pulse_timers`); every array the golden has was byte-identical (seeds 9 and 16, both paths). Measured
   cause (the draft's "the merged engine builds two arrays the golden predates" was wrong): they are visualization
   buffers `sim/bridge.py` allocates only when PyOpenGL imports (`if OPENGL_AVAILABLE and num_synapses > 0:`, present
   since the SimulationBridge extraction 6be22526c); the golden was captured without OpenGL, and with OpenGL blocked
   the unchanged test passes. Ruling as applied: the comparison is narrowed to the golden's own array set, and the set
   difference must be exactly those two arrays when `sim.bridge.OPENGL_AVAILABLE` (empty otherwise), so any other new
   or missing array still fails. The golden is not re-captured; the reason is recorded in the test docstring.
2. **canary_preconditions.** Every line passes `tools/canary_preconditions.py check-line` (S main; the registry
   requires `--commit` for any line writing under `_bg_live_seam/(step1|commit_gate|commit_dev)/`), and the canary
   (`tools/pool_canary.sh`, which runs the registered scorer on the canary's own output) PASSES on the first gate line
   before the other lines are released.
3. **This amendment is signed** (section 9) and committed on S main before any line is queued.
4. **Calendar.** The stage-2 gate window is 11-03..11-10; at about 4400 s per seed cell, the six gate cells plus the
   18 ablation cells are about 29 CPU-hours on the pool.

## 6. The instrument added at the pin (report-only; never a bar)

All three are additive fields; no registered measure, pooled value or event count changes (pinned by
`tests/test_bg_gate_amend_1101_instrument.py`: the 10-02 gate shards still score to their committed `pooled` and `n`,
and the neural-seam trial record still hashes to the cc9b4bf3e golden).

- **(a) Sub-reversal, measured.** Every `bg_neural` event carries `sub_reversal`: for every region the organ bridge
  builds, the count of steps whose lowest end-of-step v is below E_inh - 5 mV = -80 mV (`steps_below`), `v_min`, the
  regions that crossed, and `steps_checked` (trial + reset + washout, the membrane precondition's scope). STEP1-SPEC
  STOP 4 is therefore read from data. With `BRAIN_BG_IZH_SUBSTEPS=4` only end-of-step v is visible, so the count is in
  bridge steps, as in step 0 (`research/runners/bg_tonic_loop_probe.py`).
  **STOP-4, ruled (Claude, 2026-11-01) and applied at the pin:** beside the raw count (kept unchanged), each event also
  carries, per region, `steps_below_rest_bound`: steps whose lowest end-of-step v is below min(E_inh, vr_region) - 5 mV,
  with vr_region the region's Izhikevich rest/reset potential as configured on the built bridge (`cp_izh_vr`, the
  region's minimum), plus `rest_bound_mv`, `vr_mv` and `regions_below_rest_bound`; the scorer sums both counts per
  region in `diagnostics["STOP 4 sub-reversal (report-only; never gated)"]`. **STOP 4 is REPORT-ONLY in this gate: it
  never gated, no bar reads either count, and no outcome of it changes the verdict.** On organ seed 1 (`...`,
  default flags) the four MSN regions (vr -80 mV, bound -85 mV) leave the rest-bound count while motor_1 and
  practice_arousal (vr -60 mV) stay in it. Note that the step-1 record already shows
  sub-reversal on the selected set (e.g. s71 `hsb_q` v_min -86.5 mV): STOP 4's "again shows sub-reversal steps" was not
  measured there, and is reported, not gated, here.
  **1-event smoke (dev seed 79, seam_b, `--commit`, the pinned env, turn `hsb_empty1` only; 163 s CPU; scratch
  output, not committed):** the event carries both new fields; 3 regions cross -80 mV over 635 checked steps:
  `str_d1_1` 294 steps (v_min -84.2), `str_d2_0` 206 (-80.8), `practice_arousal` 13 (-80.9). **Reading for Claude:**
  the MSN presets rest at vr = -80 mV (`sim/enums.py`, IZH2007_STRIATAL_MSN_D1/D2), exactly the STOP-4 bound, so the
  bound as written counts preset MSN rest, not only the forward-Euler overshoot it was built to catch (bounded
  conductance is on); hence the per-region rest-bound count above (that smoke ran at the earlier draft pin, before the
  rest-bound count existed).
- **(b) Pallidal and output rates.** Every `bg_neural` event carries `bg_rates_hz`: GPi_k, thal_k, motor_k, GPe_k and
  selector_stn over the drive (300 ms) and hold (200 ms) windows, beside `tonic_rates_hz` (present in every arm,
  including the TONIC_DRIVE=0 ablation, where `tonic_rates_hz` is absent).
- **(c) Scorer diagnostics.** `diagnostics["G3 numerator split (report-only; bar unchanged)"]`: G3's numerator split
  into `empty_turn_hold_loss`, `content_turn_wrong_hold` and `other` (keyed by class and intact->lesion hold state),
  with per-class denominators; it reproduces ns. The former `diagnostics.seam_held_attribution` is relabelled
  `"value-route lesion attribution (held rate), pooled"` (same number) with
  `"value-route lesion attribution (held rate), by turn class"` beside it.

## 7. Report-only ablation arms (same six seeds; never gated, never part of the verdict)

Each is a full seven-arm `--run-all` on the same gate seeds with ONE change from the pinned environment; each is
scored by the registered scorer and reported next to the gate verdict. None of them can turn a GO into a NO-GO or the
reverse. They test the credit claim of section 3.

| arm | the one change | knob | out root | prediction (stated before any run) |
|---|---|---|---|---|
| A-tonic0 | no tonic STN/GPe drive | `BRAIN_BG_TONIC_DRIVE=0` (the two currents are then ignored) | `commit_gate/218a337_tonic0/` | seam_b committed winners equal the gate arm's on at least 22 of the 24 seam_b events (6 seeds x 4 turns), and every registered measure within one event of the gate arm: the tonic drive is not load-bearing. GPe drive-window rate near 0 Hz (GPe silent at 50 pA in step 1). A larger change says the drive is load-bearing, contradicting step 1, and is reported as such. |
| A-hybrid | RS-centred hybrid cells, bounded + sub-steps kept | `BRAIN_BG_PRESET_CELLS=0` | `commit_gate/218a337_hybrid/` | the committed winner differs from the gate arm on at least one seam_b event (step 0: moving to preset cells changed decision fields at today's constants). Equality on all events means a GO does not credit the preset cells either, and is reported as such. |
| A-commit-off | the 09-29 fixed-rate latch read decides (no commit stage) | no `--commit` (BRAIN_BG_COMMIT unset; `arm_env` blanks every commit variable) | `commit_off_ablation/218a337_preset/` (outside `commit_gate/`: the canary registry requires `--commit` there) | scored WITHOUT `--commit` (the 09-29 scorer): the latch read's threshold change `latch_thr_change` is above 0.10 (it was 0.25 on the 10-02 gate run), while G2a/G2b/G3 stay near the gate arm: the commit stage, not the race, is what makes the read invariant. |

The value-route lesion is already a registered arm (`seam_lesion_value`, G3) and is not duplicated.

## 8. The new-method ladder for the pallidal capability (not tested by this gate; no defer)

The tonic lever's inertness is a verdict on the METHOD (independent tonic currents on unconnected STN and GPe), not on
the capability (a basal-ganglia hold shaped by tonic pallidal activity). Next methods, in order:

1. **STN <-> GPe loop, with GPi / thalamus / motor observed.** Add STN -> GPe (excitatory) beside the existing GPe -|
   STN (GABA_A, density 0.30, weight 1.5), default OFF; the organ also has no GPe -| GPi pathway today (its indirect
   path reaches GPi only through STN -> GPi), so that edge is part of the same method; measure synchrony against a shuffled surrogate with a stated bin width (PACKET.md section 4 (L);
   Plenz & Kitai 1999; Terman et al. 2002), and read GPi_k / thal_k / motor_k from the `bg_rates_hz` field added here.
   Done when an STN current change moves GPe and GPi in-organ.
2. **GPi tonic recalibration.** GPi at 1000 pA reads 279.0 Hz on preset-centred cells, 3.5x the published 40-80 Hz
   ceiling (STEP1-SPEC section 3.9, the first named candidate); re-derive `gpi_tonic_pA` against the preset f-I and
   the band, then re-run the seam's bars on dev seeds.
3. **Hyperdirect proposal -> STN.** STN today carries no proposal information (PACKET.md section 4); a cortical
   proposal -> STN pathway (the hyperdirect route) is its own packet.

## 9. Signature

Not signed. Claude signs here (date, revision checked, the STOP-5 blocker's ruling) before any line of
`research/coordination/bg-gate-1101-pool-lines.tsv` is queued. Until then this document is a DRAFT and registers
nothing.

## 10. Honesty boundary

Every quantity named here is a functional read-out of the organ and the registered scorer (rates, spike counts,
membrane steps, held fractions). Nothing here asserts phenomenal experience. The DEV numbers of section 3 come from
seeds 71-78 and are context, not a verdict.
