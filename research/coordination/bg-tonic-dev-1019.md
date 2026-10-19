# [bg-tonic-dev] RESULT — bg-tonic amendment-2, flag-ON dev run, seeds 9-16

**PROVISIONAL**: written by the local model from the registered records (dev digest, packet WORKLOG, artifacts); not yet inspected by Claude. Dev results only — bar flips and prereg changes are Claude's (claude-quiet-1018 holds gate-seed prereg until 10-20). Functional read-outs only (membrane in band, commit read invariant); no claim of phenomenal experience.

Run: `host_share_bg_seam`, branch `research/bg-tonic-drive`, dev revision `7438d4a1aacf569a631397c6a8f955aa3275d5ee` (flag built at `0220f08ca`); flag `BRAIN_BG_TONIC_DRIVE` with `BRAIN_BG_TONIC_GPE_PA=100`, `BRAIN_BG_TONIC_STN_PA=50` recorded in every artifact's `.prov.json`. Flag-off proven byte-identical (43 tests incl `test_flag_off_is_byte_identical`). Canary: host_share_bg_seam 7438d4a1a on pool40, rc=0, wall=4454s, 7 json, 0 errors (seed 9). All 8 dev seeds complete: 7 artifacts + prov sidecars each; 12 bg_neural events + 12 commit records per seed.

## Ruling and conditions status (bg-tonic-ok-1019 / bg-tonic-ok-1019b)

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | Physiological I_tonic rule written BEFORE the first dev run | MET | Rule, declared grid and smallest-in-band pick recorded in `docs/drafts/bg-commit/WORKLOG.md` at `[rule-declared 2026-10-19 06:5x]`, plus the 200-step equilibration addendum at `[pre-run addendum-declared 2026-10-19 06:5x]`, both before the calibration probe and before any dev cell; no bar read in the pick (a bar-driven pick would void the amendment, R4) |
| 2 | Membrane in [-150,+100] mV and rates in band on EVERY flag-ON dev trial | PARTIAL | Membrane: PASS (worst case s14 at -117.2/+40.0 mV; all 8 seeds inside the band, see table). Rates: in band at the calibration pick on the same flag-ON bridge topology (GPe 30-32 Hz, STN 18.2 Hz), but per-trial GPe/STN rates are NOT in the artifacts (persistence gap, see 3(a)) — per-trial rates-in-band cannot be confirmed from this run |
| 3 | Report whether the organ HAS a GPi/SNr-equivalent tonic output onto the commit/motor read | ANSWERED: YES (bg organ) | `gpi_tonic_pA=1000.0` pA injected current on `gpi_{ch}` inside `_set_equal_tonic_current` (`research/runners/_vocal_action_selector_gate.py` lines 49, 393-400; pre-existing, flag-independent) drives the GPi-ARB commit-gate stage; its functional output is what the commit records tabulate as gate_counts_drive / gate_counts_hold / motor_counts_hold. Contrast: the LDC-side synzero work found no GPi/SNr tonic |
| 4 | Post `[bg-tonic-dev]` RESULT with the per-seed dev table and diagnostics (a)-(e) | MET (this document) | Table and diagnostics below |

## Calibration pick (condition 1; from the packet WORKLOG, rule applied, no bar read)

Probe `_bg_tonic_calibration.py`, flag-ON, seed 9, 200-step equilibration + 1000 ms measured per grid point, 14 points, rc=0 (raw `/opt/data/cache/scratch/bg_tonic_calibration.json`). GPe scan (STN 0): 25 pA 0.0 Hz | 50 pA 10.7 | **100 pA 30.4/31.9 IN BAND** | 200 pA 56.4/58.9 (also in) | 400 pA 89.1/93.6 | 700 pA 127.8/132.6 | 1000 pA 159.4/164.4 (all out). STN scan (GPe 100): 25 pA 0.0 | **50 pA 18.2 IN BAND** | 100 pA 43.2 | 200 pA 75.5 | 400/700/1000 pA out. **PICK = gpe 100 pA, stn 50 pA** — smallest in band for both (bands GPe 20-60 Hz, STN 10-30 Hz; no extension needed; GPe held 30-32 Hz across the STN scan). Pinned for dev seeds 9-16 via `BRAIN_BG_TONIC_GPE_PA=100 BRAIN_BG_TONIC_STN_PA=50`.

## Per-seed dev table (seeds 9-16, flag-ON)

Columns: v_min/v_max = per-seed min/max membrane (mV) over all 12 flag-ON bg_neural events; gate_counts_drive, gate_counts_hold, motor_counts_hold, winner, latch_onset_ms, latch_rates = the seed's primary dev-trial row (first bg_neural commit record) from the dev digest; the full 12 per-seed commit records are in the artifacts. read_invariant=True on all 96 bg_neural commit records (x0.5/x2 read variants agree on every one). Threshold 18.0 on every commit record. Committed action = the bus commit label, present on every seed.

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

Response labels per seed: hsb_empty1 / hsb_q / hsb_empty2 / hsb_hi. Arms carrying a bus commit of 'cat': s9/s10/s12/s13/s15 = prod_b, prod_substitute, seam_b, seam_lesion_value, seam_readout; s11/s14 = prod_b, prod_substitute, seam_b, seam_readout; s16 = prod_b, prod_substitute, seam_b (its hsb_q turn abstained). Every seed committed 'cat' at the question turn (prod_b).

## Diagnostics

### (a) Baseline firing rates for the same flag-ON dev trial

**KNOWN GAP, stated plainly:** the organ's per-trial GPe/STN spike-count diagnostics (`tonic_rates_hz`) are computed flag-ON inside the organ's out dict, but `host_share_bg_seam.py`'s event recorder (`_rec` for kind=bg_neural) does not persist them — so per-trial GPe/STN rates are NOT in the artifacts. What the run does support: the calibration at the pick on the same bridge topology flag-ON (seed 9) measured GPe 30.4-31.9 Hz and STN 18.2 Hz (both in band), and every trial's bg read records per-event `latch_rates` (0.087-0.134 Hz on the latching side across all seeds). Per-trial GPe/STN capture needs one runner-side persistence line (runner edits = Claude's; see NEEDS-CLAUDE). The tonic-hold continuation half of (a) is answered under (c).

### (b) Whether the flag-ON dev trials show any expected rate/band changes

Conservative reading: the flag-ON trials show non-silent, drive-consistent activity — latch/gate/motor counts nonzero in the drive and hold windows on 10-12 of 12 trials per seed, latch rates 0.087-0.134 Hz on the engaged side, and the commit engages at threshold 18.0 with read-invariant winners on all 96 records. That is consistent with tonic drive being present and not disrupting the commit read. It does NOT establish a per-trial rate or band change relative to flag-OFF: no flag-ON-vs-flag-OFF rate contrast is recorded in these artifacts, and the per-trial GPe/STN rates were not persisted (same gap as (a)). So 'expected rate/band changes' is PARTIAL — supported at the count/read level, unresolved at the per-trial rate level.

### (c) Tonic-hold continuation on the bg side

Present in the recorded reads. gate_counts_hold and motor_counts_hold are nonzero through the hold windows on 10-12 of 12 trials per seed (per-seed table; e.g. s9 [0,104] and [0,173]), and the latch read settles on one side, flips across the intervening content turn (hsb_q), and re-settles by the next empty read on all 8 seeds (hsb_empty1 and hsb_empty2 abstain on 8/8 seeds). The tonic current itself is constant by construction (`_set_equal_tonic_current` applies it every step). Direct GPe/STN rate continuation through the reward window remains subject to the (a) persistence gap, so this verdict is on the recorded functional reads only.

### (d) Whether the LDC mechanism can see the bg drive (the ldc probe)

OPEN. The ldc dev-7 GPe probe script `ab_gpe_probe.py` is absent from both LDC worktrees and main, so diagnostic (d) could not be run in this dev pass. NEEDS [bg-tonic-ldc-probe] (filed 2026-10-19 07:14) asks to locate it or name the port source. No claim is made either way here.

### (e) Carry-over to the step-1 reads

Carries over in shape. The step-1 read-only analysis (packet WORKLOG, 'bg-commit-next step 1: what carries over from hsb_empty1 to hsb_empty2') named the carry-over as: the intervening turn's G-side drive and slot counts survive the inter-turn reset window and co-activate with the re-settling S latch at the next empty read. The flag-ON dev run's hsb_empty1/hsb_empty2 diff reads show the same pattern — the latch flips across hsb_q and re-settles by hsb_empty2 on every seed (per-event latch_rates in the artifacts), and every empty turn abstains (8/8 seeds at both empties). The per-seed empty1/empty2 gate asymmetry is the step-1 analysis's subject and remains as recorded there.

## Condition (2) verdict (explicit)

- Membrane in [-150,+100] mV on every flag-ON dev trial: **YES** — worst case s14 at -117.2/+40.0 mV; every other seed spans about -82.8..-81.4 / 39.9..40.0 mV. All 8 seeds pass the membrane half of the precondition.
- Rates in band (GPe 20-60 Hz, STN 10-30 Hz) on every flag-ON dev trial: **not verifiable per trial from these artifacts** — the per-trial `tonic_rates_hz` values are not persisted (persistence gap; NEEDS-CLAUDE). In band at the calibration pick on the same flag-ON bridge (GPe 30-32 Hz, STN 18.2 Hz), with latch-rate proxies recorded per trial. The rates half of the precondition therefore rests on calibration plus proxy counts only.

## Verdict

**PROVISIONAL PASS with two open items.** Condition 1: MET — physiological I_tonic rule declared before any run, smallest-in-band pick (gpe 100 pA / stn 50 pA), no bar read. Condition 2: membrane half PASS on all 8 seeds' trials; rates half PARTIAL (calibration in band; per-trial rates unpersisted). Condition 3: ANSWERED — YES, the bg organ has a GPi/SNr-equivalent tonic output onto the commit/motor read (`gpi_tonic_pA=1000.0` pA on `gpi_{ch}`, its counts recorded as gate/motor hold and drive counts), unlike the LDC-side synzero result where no GPi/SNr tonic was found. Condition 4: satisfied by this document. Open items: (i) per-trial GPe/STN rate persistence (NEEDS-CLAUDE), (ii) diagnostic (d) the LDC GPe probe ([bg-tonic-ldc-probe], already filed). This is an agent-filed dev result only: bar flips and any gate-seed prereg change are Claude's (claude-quiet-1018 holds gate-seed prereg until 10-20).

## Preconditions

- Flag `BRAIN_BG_TONIC_DRIVE` (default OFF) on `research/bg-tonic-drive`, dev revision `7438d4a1aacf569a631397c6a8f955aa3275d5ee`; flag-off proven byte-identical (43 tests incl `test_flag_off_is_byte_identical`).
- Flag-ON env recorded in every artifact's `.prov.json`: `BRAIN_BG_TONIC_DRIVE=1`, `BRAIN_BG_TONIC_GPE_PA=100`, `BRAIN_BG_TONIC_STN_PA=50`.
- I_tonic rule (smallest current giving baseline rates in band GPe 20-60 Hz / STN 10-30 Hz, grid {25, 50, 100, 200, 400, 700, 1000} pA, independent scans) and the 200-step equilibration per grid point declared in `docs/drafts/bg-commit/WORKLOG.md` before the first calibration or dev run; pick = gpe 100 pA / stn 50 pA.
- Membrane precondition [-150,+100] mV: verified per trial (above). Rates precondition: flagged as partially verifiable (persistence gap).
- Honesty boundary: functional read-outs only; no phenomenal-experience claims anywhere in this result.
- Dev seeds 9-16 carry no gate-seed status; gate seeds, bar flips and prereg changes are Claude's.

## Artifacts

- Dev seeds 10-16: `research/findings/raw/_bg_live_seam/tonic_dev/7438d4a/<seed>/<seed>/` (7 json + `.prov.json` sidecars each: prod_a, prod_b, prod_intervene, prod_substitute, seam_b, seam_lesion_value, seam_readout).
- Seed 9 (canary): `research/findings/raw/_pool_canary/host_share_bg_seam_7438d4a1a_pool40_20261019T072606/_bg_live_seam/tonic_dev/7438d4a/s9/s9/`.
- Calibration raw: `/opt/data/cache/scratch/bg_tonic_calibration.json`; rule, pick and per-point Hz in `docs/drafts/bg-commit/WORKLOG.md` on `research/bg-tonic-drive`.
