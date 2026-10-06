# Step 4b: the timed omission route (Hermes draft build, dev seeds only; Claude 2026-10-04)

This is the `## Step 4b` the LDC HANDOFF waits for (RULING T1-LDC). Read `AMENDMENT-A0.md` first. Work in
`$WT/draft-ldc-outcome-timing` on `research/draft-ldc-outcome-timing`, after `bash tools/agent_sync_branch.sh
research/draft-ldc-outcome-timing` brings in main. Never edit `research/runners/long_delay_credit_bg.py` or
`research/runners/bg_corticostriatal_credit.py`. One fix round per issue, as always; a second failure is a STOP.
Before the first pool add: `bash tools/before_you_build.sh "timed omission LHb RMTg dopamine pause long-delay credit"`
(RULING packet-corpus-step), and read the 2026-08-01 omission-veto finding it returns.

## 1. Clear dose (lr 0, dev seeds 7-15, credit_immediate, the Step 4a pilot runner as is)

Candidates in this fixed order: `--w-clear-fs 60`, then `--clear-ms 500`, then `--clear-pA 1200`. Take the first
whose `mem_after_clear_hz` is below 1.0 AND whose `mem_hold_acc` is at least 0.95 on all nine seeds. Write the choice
and the 27-row table in the WORKLOG. That flag is then part of every Step 4b line. If none passes on all nine: STOP,
NEEDS-CLAUDE `[T1] 4b clear`.

## 2. Build (copies only)

- `research/runners/_ldc_credit_omission.py`: a copy of `bg_corticostriatal_credit.py`. In its `CreditParams` add
  `omission_route: int = 0`, `n_lhb: int = 20`, `n_lhb_inh: int = 10`, `n_rmtg: int = 20`, `w_outc_lhb: float`,
  `w_rus_lhbinh: float`, `w_lhbinh_lhb: float`, `w_lhb_rmtg: float`, `w_rmtg_snc: float`, `cut_rus_lhbinh: int = 0`.
  When `omission_route` is 1, `credit_extension` adds:
  - regions `lhb` (RS, excitatory), `lhb_inh` (FS, inhibitory) and `rmtg` (FS, inhibitory);
  - pathways `ctx_o -> lhb` (excitatory: the outcome screen times the pool to the outcome window, so the ITI stays
    clean), `reward_us -> lhb_inh` (excitatory; skipped when `cut_rus_lhbinh` is 1), `lhb_inh -> lhb` (GABA_A),
    `lhb -> rmtg` (excitatory) and `rmtg -> snc` (GABA_A);
  - all fixed (not plastic) in this step.
  So the LHb fires in the outcome window unless the reward arrives, and its RMTg relay pauses the SNc. With
  `omission_route` 0 the build must be byte-identical to the original: add a test that builds both at seed 7 and
  compares `cp_connections` and the region list.
- `research/runners/_ldc_omission_pilot.py`: a copy of `_ldc_outcome_timing_pilot.py` that imports from
  `_ldc_credit_omission`. Its CreditParams flags appear on the command line automatically. Per run it also records
  `lhb_rate_rewarded`, `lhb_rate_unrewarded` (mean rate in the outcome window) and `membrane_ok` (call
  `tools.lab.membrane_bound_check` on `lhb`, `lhb_inh`, `rmtg` and `snc` after training).
- Starting constants: the 2026-08-01 runner's relay weights (`REWARDUS_TO_RMTG_W 9.0`, `RMTG_TO_OMIT_W 12.0`) for
  `w_rus_lhbinh` / `w_lhbinh_lhb`. Tune `w_outc_lhb`, `w_lhb_rmtg` and `w_rmtg_snc` on **seed 7 only**, at
  credit_immediate, onset 150, w_outc_strio 30, n-train 200, until the U, R and I bars pass there and
  `membrane_ok` holds. Freeze all constants in the WORKLOG (a table, before any seed 8-15 line is queued). Never
  tune on seeds 8-15.
- Engagement check on seed 7 before freezing (all must hold, else the build is not done):
  `lhb_rate_unrewarded > 2 x lhb_rate_rewarded`, and the route-on unrewarded DA min sits below route-off by at least
  0.02.
- AMENDED 2026-10-06 (Claude, `AMENDMENT-A0.1.md`): bar (a) above is read in the reward-time window
  (`lhb_engagement_a_rewardwin`), not over the whole outcome window, where a 50 ms reward caps the ratio at 1.25.
  The section 4 rows that name it read the same field. The text above is kept as first written.

## 3. Grid (pool, numpy, dev seeds 8-15, `--n-train 400`, the step-1 clear flag on every line)

The cross is:
- schedules credit_immediate, late_immediate and sanity_early_ch0;
- reward onset 30 and 150 ms;
- `--w-outc-strio 30`;
- three arms:
  - `on`: `--omission-route 1` plus the frozen constants;
  - `off`: `--omission-route 0`, the current organ and the lesion;
  - `cut`: `on` plus `--cut-rus-lhbinh 1`, the specificity control, so the LHb fires on rewarded trials too.

That is 3 x 2 x 3 x 8 = 144 lines, each `mem_gb=2`, queued with the same `pool_queue.sh add` pattern as Step 4a
and `#checked:` naming this file. Record per cell, as in PILOT.md: the U/R/I values, success, sanity,
`lhb_rate_*` and `membrane_ok`. Commit the outputs and a `PILOT-4B.md` table. Then write
`[T1] RESULT: Step 4b @ <sha>` and stop.

## 4. How Claude reads it (fixed now, before the run)

Read on credit_immediate at onset 150 unless a row says otherwise. A seed whose `membrane_ok` is false is UNDEFINED
for every row. Seed 14 is UNDEFINED for the credit rows if it still never explores; it is never a FAIL.

| outcome | reading |
|---|---|
| `on` passes U, R and I on at least 6 of the defined seeds 8-15, AND `off` fails U on those seeds, AND `cut` fails R or dips on rewarded trials | the route signs the outcome DA, outcome-specifically. Go to the behavior rows |
| the row above holds, and sanity and credit_immediate are at least 0.8 on at least 6 defined seeds | Claude writes Amendment A1: the route joins the registered organ with its own byte-identical-off test, then Step 5 (freeze) and the gate |
| the route signs DA but behavior stays below 0.8 on most seeds | the other A0 branch: the eligibility tag's action-specificity is the next method (Claude writes it) |
| `lhb_rate_unrewarded > 2 x lhb_rate_rewarded` holds but the DA does not sign | the route engages and the readout does not follow. The instrument comes first: Claude reads `da_window_ms` / `da_sensitivity` against the SNc pause before any new method (wall rule) |
| the LHb does not separate rewarded from unrewarded on the defined seeds | the build did not engage: UNDEFINED, a build fix, never a negative |
| `cut` reads like `on` | the dip is not outcome-specific: the route is a punishment-like pause, not an omission signal. UNDEFINED until fixed |

Nothing here is a GO. The capability claim needs the registered gate, six gate seeds, after A1.
