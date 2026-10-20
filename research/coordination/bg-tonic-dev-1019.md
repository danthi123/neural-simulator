# [bg-tonic-dev] RESULT — bg-tonic amendment-2, flag-ON dev run, seeds 9-16

**REFILE (2026-10-19 18:42 EDT, revision `8177576ae802f52be5bfa477b134fbe3b67871b6`)**: full replacement of the 12:46 agent-filed PROVISIONAL (first filed at `d794c6953`), per RULING bg-tonic-persist-1019 (`research/coordination/LOCAL_LLM_RUNBOOK.md` line ~1517): the mandated one-line persistence patch (`_rec` kind=bg_neural now persists the organ's `tonic_rates_hz`, flag-ON-only key) landed as branch commit 8177576ae, dev seeds 9-16 were re-run flag-ON, and condition (2) of bg-tonic-ok-1019b is now **READ** from persisted per-trial rates instead of inferred from the calibration.

**PROVISIONAL**: written by the local model from the registered records (rerun dev digest, packet WORKLOG, artifacts); not yet inspected by Claude. Dev results only — bar flips, prereg changes and any re-pick/re-design call are Claude's (claude-quiet-1018 holds gate-seed prereg until 10-20). Functional read-outs only (membrane in band, commit read invariant, rate bands); no claim of phenomenal experience.

Run: `host_share_bg_seam`, branch `research/bg-tonic-drive`, dev revision `8177576ae802f52be5bfa477b134fbe3b67871b6` (flag built at `0220f08ca`; prior dev revision `7438d4a1aacf569a631397c6a8f955aa3275d5ee`); flag `BRAIN_BG_TONIC_DRIVE` with `BRAIN_BG_TONIC_GPE_PA=100`, `BRAIN_BG_TONIC_STN_PA=50` recorded in every artifact's `.prov.json`. Flag-off proven byte-identical at the persist revision (43 tests incl `test_flag_off_is_byte_identical`; the added key exists only on the flag-ON path — record-level check OK). Canary: host_share_bg_seam 8177576ae on pool40, rc=0, wall=4470s, 7 json, 0 errors (seed 9). All 8 dev seeds re-run and complete at 8177576ae: 7 artifacts + prov sidecars each; 12 bg_neural events + 12 commit records per seed; `tonic_rates_hz` present in all 96 commit records (verified by record count: 84 across s10-s16 + 12 on s9).

## Ruling and conditions status (bg-tonic-ok-1019 / bg-tonic-ok-1019b)

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | Physiological I_tonic rule written BEFORE the first dev run | MET | Rule, declared grid and smallest-in-band pick recorded in `docs/drafts/bg-commit/WORKLOG.md` at `[rule-declared 2026-10-19 06:5x]`, plus the 200-step equilibration addendum at `[pre-run addendum-declared 2026-10-19 06:5x]`, both before the calibration probe and before any dev cell; no bar read in the pick (a bar-driven pick would void the amendment, R4) |
| 2 | Membrane in [-150,+100] mV and rates in band on EVERY flag-ON dev trial | **NOT MET** | Membrane: PASS (worst case s14 at -117.2/+40.0 mV; all 8 seeds inside the band). Rates, now READ per trial from persisted `tonic_rates_hz`: GPe in band 20-60 Hz on 8/8 seeds, **STN in band 10-30 Hz on only 2/8 (s12, s16), above band on 6/8 (max 42.83 Hz)** — the STN half fails, see the rates table and condition (2) verdict below |
| 3 | Report whether the organ HAS a GPi/SNr-equivalent tonic output onto the commit/motor read | ANSWERED: YES (bg organ) | `gpi_tonic_pA=1000.0` pA injected current on `gpi_{ch}` inside `_set_equal_tonic_current` (`research/runners/_vocal_action_selector_gate.py` lines 49, 393-400; pre-existing, flag-independent) drives the GPi-ARB commit-gate stage; its functional output is what the commit records tabulate as gate_counts_drive / gate_counts_hold / motor_counts_hold. Contrast: the LDC-side synzero work found no GPi/SNr tonic |
| 4 | Post `[bg-tonic-dev]` RESULT with the per-seed dev table and diagnostics (a)-(e) | MET (this document) | Table and diagnostics below |

## Calibration pick (condition 1; from the packet WORKLOG, rule applied, no bar read)

Probe `_bg_tonic_calibration.py`, flag-ON, seed 9, 200-step equilibration + 1000 ms measured per grid point, 14 points, rc=0 (raw `/opt/data/cache/scratch/bg_tonic_calibration.json`). GPe scan (STN 0): 25 pA 0.0 Hz | 50 pA 10.7 | **100 pA 30.4/31.9 IN BAND** | 200 pA 56.4/58.9 (also in) | 400 pA 89.1/93.6 | 700 pA 127.8/132.6 | 1000 pA 159.4/164.4 (all out). STN scan (GPe 100): 25 pA 0.0 | **50 pA 18.2 IN BAND** | 100 pA 43.2 | 200 pA 75.5 | 400/700/1000 pA out. **PICK = gpe 100 pA, stn 50 pA** — smallest in band for both (bands GPe 20-60 Hz, STN 10-30 Hz; no extension needed; GPe held 30-32 Hz across the STN scan). Pinned for dev seeds 9-16 via `BRAIN_BG_TONIC_GPE_PA=100 BRAIN_BG_TONIC_STN_PA=50`. **Transfer caveat now measurable:** the STN scan's 18.2 Hz at 50 pA was measured on the bare seed-9 calibration bridge; the dev-trial network state runs STN roughly 2x higher at the same current (see rates table), so the pick does NOT transfer for STN, while GPe transfers (in band 8/8).

## Per-seed dev table (seeds 9-16, flag-ON, rerun at 8177576ae)

Columns: v_min/v_max = per-seed min/max membrane (mV) over all 12 flag-ON bg_neural events; gate_counts_drive, gate_counts_hold, motor_counts_hold, winner, latch_onset_ms, latch_rates = the seed's primary dev-trial row (first bg_neural commit record) from the dev digest; the full 12 per-seed commit records are in the artifacts. read_invariant=True on all 96 bg_neural commit records (x0.5/x2 read variants agree on every one; verified: zero `read_invariant: false` across the rerun artifacts). Threshold 18.0 on every commit record. Committed action = the bus commit label, present on every seed.

| seed | v_min | v_max | gate_counts_drive | gate_counts_hold | motor_counts_hold | winner | read_invariant | latch_onset_ms | latch_rates | committed action |
|---|---|---|---|---|---|---|---|---|---|---|
| s9 | -81.8 | 40.0 | [0,200] | [0,104] | [0,173] | 1 | True | [null,154] | [0.0,0.134] | cat (5 arms) |
| s10 | -82.1 | 40.0 | [0,150] | [83,2] | [195,0] | 0 | True | [340,194] | [0.1217,0.0] | cat (5 arms) |
| s11 | -82.8 | 40.0 | [72,155] | [12,56] | [5,193] | 1 | True | [284,157] | [0.0,0.0] | cat (4 arms) |
| s12 | -82.3 | 39.9 | [0,77] | [0,84] | [0,171] | 1 | True | [null,268] | [0.0,0.114] | cat (5 arms) |
| s13 | -82.4 | 40.0 | [0,303] | [0,134] | [0,312] | 1 | True | [null,114] | [0.0,0.106] | cat (5 arms) |
| s14 | -117.2 | 40.0 | [0,207] | [0,80] | [0,181] | 1 | True | [null,177] | [0.0,0.091] | cat (4 arms) |
| s15 | -81.4 | 40.0 | [0,173] | [0,77] | [0,137] | 1 | True | [null,212] | [0.0,0.0983] | cat (5 arms) |
| s16 | -82.3 | 40.0 | [115,81] | [0,35] | [0,41] | null | True | [228,146] | [0.1063,0.0] | cat (3 arms) |

Response labels per seed: hsb_empty1 / hsb_q / hsb_empty2 / hsb_hi. Arms carrying a bus commit of 'cat': s9/s10/s12/s13/s15 = prod_b, prod_substitute, seam_b, seam_lesion_value, seam_readout; s11/s14 = prod_b, prod_substitute, seam_b, seam_readout; s16 = prod_b, prod_substitute, seam_b (its hsb_q turn abstained). Every seed committed 'cat' at the question turn (prod_b). The rerun reproduces the pre-persistence run's functional read exactly (same table values as the 7438d4a1a pass), as expected for an additive record key.

## Persisted tonic rates (the new READ; per seed, min-max over 12 trials)

Record key `tonic_rates_hz` = {gpe_0, gpe_1, selector_stn, window: "drive"} per bg_neural commit record (window=drive, drive_steps=300); `stn` below is the record's `selector_stn`. Bands: GPe 20-60 Hz, STN 10-30 Hz. A seed passes a band only if EVERY trial's value is inside it.

| seed | gpe_0 (Hz) | gpe_1 (Hz) | stn (Hz) | GPe 20-60 | STN 10-30 |
|---|---|---|---|---|---|
| s9 | 33.54-38.12 | 35.21-38.96 | 32.17-34.50 | PASS | ABOVE (FAIL) |
| s10 | 23.96-27.08 | 30.00-36.67 | 36.17-38.67 | PASS | ABOVE (FAIL) |
| s11 | 28.75-37.50 | 22.08-30.42 | 29.83-33.83 | PASS | ABOVE (FAIL) |
| s12 | 28.96-31.25 | 27.50-31.04 | 14.00-17.00 | PASS | PASS |
| s13 | 25.83-30.21 | 27.08-30.42 | 40.00-42.83 | PASS | ABOVE (FAIL) |
| s14 | 27.92-31.88 | 25.00-28.96 | 39.00-42.33 | PASS | ABOVE (FAIL) |
| s15 | 27.50-29.79 | 26.67-30.21 | 35.17-37.50 | PASS | ABOVE (FAIL) |
| s16 | 34.58-38.12 | 29.58-34.79 | 23.67-27.33 | PASS | PASS |

Totals: GPe in band 8/8 seeds (all 96 trials within 22.08-38.96 Hz); STN in band 2/8 (s12, s16), above band 6/8 — s11's minimum 29.83 grazes the band but its max 33.83 is above, so not every trial is in band; the worst case is s13 at 42.83 Hz.

## Diagnostics

### (a) Baseline firing rates for the same flag-ON dev trial

**PERSISTENCE GAP CLOSED.** RULING bg-tonic-persist-1019's one additive line (`_rec` kind=bg_neural, flag-ON-only key) persists the organ's `tonic_rates_hz` per trial; the rerun at 8177576ae carries it in all 96 bg_neural commit records (record counts verified against the artifacts). Per-trial GPe/STN rates are now READ directly — table above. The tonic-hold continuation half of (a) is answered under (c).

### (b) Whether the flag-ON dev trials show any expected rate/band changes

Now answerable with real rates. **GPe is stably in band:** every one of the 96 trials sits inside 20-60 Hz (22.08-38.96 Hz across gpe_0/gpe_1), 8/8 seeds. **STN is above band on 6/8 seeds:** 32.17-34.50 (s9), 36.17-38.67 (s10), up to 33.83 (s11), 40.00-42.83 (s13), 39.00-42.33 (s14), 35.17-37.50 (s15) — only s12 (14.00-17.00) and s16 (23.67-27.33) hold the 10-30 Hz band on every trial. Reading: tonic drive is present and consistent with the GPe band, but the stn 50 pA pick produces dev-trial STN rates roughly 2x the calibration's 18.2 Hz on most seeds. The count/read-level observations from the earlier pass (nonzero gate/motor counts, threshold-18.0 commits, read-invariant winners on all 96 records) stand unchanged and show the tonic drive does not disrupt the commit read.

### (c) Tonic-hold continuation on the bg side

Present in the recorded reads. gate_counts_hold and motor_counts_hold are nonzero through the hold windows on 10-12 of 12 trials per seed (per-seed table; e.g. s9 [0,104] and [0,173]), and the latch read settles on one side, flips across the intervening content turn (hsb_q), and re-settles by the next empty read on all 8 seeds (hsb_empty1 and hsb_empty2 abstain on 8/8 seeds). The tonic current itself is constant by construction (`_set_equal_tonic_current` applies it every step). Direct GPe/STN rate continuation through the reward window is now measurable per trial (the persisted window is the drive window; window=drive), and the drive-window rates above are the read.

### (d) Whether the LDC mechanism can see the bg drive (the ldc probe)

**STILL OPEN.** The ldc dev-7 GPe probe script `ab_gpe_probe.py` is absent from both LDC worktrees and main, so diagnostic (d) could not be run in this dev pass. [bg-tonic-ldc-probe] (filed 2026-10-19 07:14) already covers locating it or naming the port source — not duplicated here. No claim is made either way.

### (e) Carry-over to the step-1 reads

Carries over in shape. The step-1 read-only analysis (packet WORKLOG, 'bg-commit-next step 1: what carries over from hsb_empty1 to hsb_empty2') named the carry-over as: the intervening turn's G-side drive and slot counts survive the inter-turn reset window and co-activate with the re-settling S latch at the next empty read. The flag-ON dev run's hsb_empty1/hsb_empty2 diff reads show the same pattern — the latch flips across hsb_q and re-settles by hsb_empty2 on every seed (per-event latch_rates in the artifacts), and every empty turn abstains (8/8 seeds at both empties). The per-seed empty1/empty2 gate asymmetry is the step-1 analysis's subject and remains as recorded there.

## Condition (2) verdict (explicit, now READ not inferred)

- Membrane in [-150,+100] mV on every flag-ON dev trial: **YES** — worst case s14 at -117.2/+40.0 mV; every other seed spans about -82.8..-81.4 / 39.9..40.0 mV. All 8 seeds pass the membrane half of the precondition.
- Rates in band (GPe 20-60 Hz, STN 10-30 Hz) on every flag-ON dev trial, READ from persisted `tonic_rates_hz`: **GPe PASS 8/8; STN FAIL 6/8 (above band, up to 42.83 Hz; in band only on s12 and s16).** Condition (2) therefore does **NOT** fully hold: the physiological pick's STN half fails in the dev-trial context.
- Stated plainly: this is a **partial failure of the physiological pick in the dev-trial context** — the smallest-in-band rule was applied honestly, but it was calibrated on the bare seed-9 bridge, and that calibration does not transfer to the dev-trial network state for STN (~2x higher there), while it does transfer for GPe. This is a finding on the METHOD (bare-bridge calibration does not predict dev-trial STN rates), NOT a license to drop the tonic-drive capability — no defer. The re-pick or re-design call (e.g. re-derive the stn current against dev-trial state, or recalibrate inside the dev network) is Claude's.

## Verdict

**PROVISIONAL — condition (2) NOT MET (STN half); conditions 1, 3, 4 stand.** Condition 1: MET — physiological I_tonic rule declared before any run, smallest-in-band pick (gpe 100 pA / stn 50 pA), no bar read. Condition 2: membrane half PASS on all 8 seeds' trials; rates half now READ — GPe in band on every trial of every seed (8/8), STN above band on 6/8 (14.0-42.83 Hz overall), so the precondition as written (rates in band on EVERY trial) fails on its STN side. Condition 3: ANSWERED — YES, the bg organ has a GPi/SNr-equivalent tonic output onto the commit/motor read (`gpi_tonic_pA=1000.0` pA on `gpi_{ch}`, its counts recorded as gate/motor hold and drive counts), unlike the LDC-side synzero result where no GPi/SNr tonic was found. Condition 4: satisfied by this document. Open item: diagnostic (d), the LDC GPe probe ([bg-tonic-ldc-probe], already filed 07:14 — not duplicated). The persistence gap that made the prior pass PARTIAL is closed. This is an agent-filed dev result only: bar flips, gate-seed prereg changes and the STN re-pick/re-design decision are Claude's (claude-quiet-1018 holds gate-seed prereg until 10-20).

## Preconditions

- Flag `BRAIN_BG_TONIC_DRIVE` (default OFF) on `research/bg-tonic-drive`, dev revision `8177576ae802f52be5bfa477b134fbe3b67871b6` (persist-patch commit on top of the flag built at `0220f08ca`); flag-off proven byte-identical at this revision (43 tests incl `test_flag_off_is_byte_identical`; the `tonic_rates_hz` key is added on the flag-ON path only, so flag-OFF records gain no key).
- Canary PASS at this revision: host_share_bg_seam 8177576ae on pool40 at 2026-10-19 14:44 (rc=0, wall=4470s, 7 json, 0 errors).
- Flag-ON env recorded in every artifact's `.prov.json`: `BRAIN_BG_TONIC_DRIVE=1`, `BRAIN_BG_TONIC_GPE_PA=100`, `BRAIN_BG_TONIC_STN_PA=50`.
- I_tonic rule (smallest current giving baseline rates in band GPe 20-60 Hz / STN 10-30 Hz, grid {25, 50, 100, 200, 400, 700, 1000} pA, independent scans) and the 200-step equilibration per grid point declared in `docs/drafts/bg-commit/WORKLOG.md` before the first calibration or dev run; pick = gpe 100 pA / stn 50 pA.
- Membrane precondition [-150,+100] mV: verified per trial (above). Rates precondition: READ per trial from persisted `tonic_rates_hz` — GPe 8/8 in band, STN 6/8 above band, precondition not met (condition 2).
- Honesty boundary: functional read-outs only; no phenomenal-experience claims anywhere in this result.
- Dev seeds 9-16 carry no gate-seed status; gate seeds, bar flips and any re-pick design call are Claude's.

## Artifacts

- Dev seeds 10-16 (rerun at 8177576ae): `research/findings/raw/_bg_live_seam/tonic_dev/8177576/<seed>/<seed>/` (7 json + `.prov.json` sidecars each: prod_a, prod_b, prod_intervene, prod_substitute, seam_b, seam_lesion_value, seam_readout).
- Seed 9 (canary, rerun at 8177576ae): `research/findings/raw/_pool_canary/host_share_bg_seam_8177576ae_pool40_20261019T133006/_bg_live_seam/tonic_dev/8177576/s9/s9/` (7 json).
- Persisted rates: `tonic_rates_hz` in all 96 bg_neural commit records (the 12 per seed live in the seam_b / seam_readout / seam_lesion_value json); dev digest `/opt/data/cache/scratch/tonic_dev_digest817.json` (per-seed v_min/v_max, n_bg/n_commits/n_tonic, first_commit, committed_in, tonic_ranges).
- Calibration raw: `/opt/data/cache/scratch/bg_tonic_calibration.json`; rule, pick and per-point Hz in `docs/drafts/bg-commit/WORKLOG.md` on `research/bg-tonic-drive` (the bare-bridge STN 18.2 Hz reading that does not transfer to dev-trial state).