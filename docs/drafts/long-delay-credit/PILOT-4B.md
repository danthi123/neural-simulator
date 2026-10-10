# PILOT-4B: what the step-4b grid says about the paired behaviour, the bound flags, and the dip's timing
(pilot; dev seeds 8-15 only; MiMo Pro delegate for RULING pilot-4b-next, 2026-10-09)

Scope: method diagnosis on dev-pilot cells only. NOT evidence, no six-seed generalization, no GO/NO-GO for the
credit gate. Units: time ms. Honesty boundary: functional read-outs only (spike rates, DA concentration, weight
changes, choice statistics); no phenomenal-experience claims.

Source: `research/findings/raw/_long_delay_credit_bg/dev_pilot_4b/`, the 96 family cells
`grid_{credit,late}_immediate_{on30,on150}_{on,off,cut}_s{8..15}.json` (each with a `.prov.json` sidecar).
Runner `research/runners/_ldc_omission_pilot.py` (+ `_ldc_credit_omission.py`), `runner_sha256`
`4bd1e097f0ba...`, one uniform hash across all 96 cells; provenance `git_sha 9f625fd1c1f201a871ea1c5c0a2d475e7a3fbe88`
(e.g. `grid_credit_immediate_on30_on_s8.json.prov.json`). The 27 `clear_*` cells and the 48
`grid_sanity_early_ch0_*` cells are excluded per the ruling. Tasking said "48 family cells"; the glob expands to
12 families x 8 seeds = 96 (48 is the sanity count) — all 96 are analyzed here.

Cell facts (verified in every cell): `n_train` 400, `n_test` 100, `elig_tau_ms` 1000.0, `da_window_ms` 30.0,
`da_decay_ms` 100.0, `da_baseline` 0.5, `w_max` 90.0, `membrane_ok` true everywhere (no UNDEFINED rows from
membrane bounds). Arms per STEP4B: `on` = omission_route 1, `cut` = on + cut_rus_lhbinh 1, `off` = omission_route 0.
Row `lhb_rewardwin_ms` is `[30,80]` in all on30 cells and `[150,200]` in all on150 cells: "on30/on150" is the
reward-onset offset into the 250 ms outcome screen (`--reward-on-ms`, `--outcome-ms 250` in the prov argv).

## 0. What "v1" is in the data

There is no `v1` field, key or enum anywhere in the cells (top level: `runner_sha256`, `git_head`, `git_dirty`,
`params`, `schedule`, `rows`, plus `below_chance`/`below_chance_reason` declarations on some cells; row level: 44
keys; nested dicts enumerated). The only bound instrument is the row boolean `void_floor_ceiling` =
`any(floor_or_ceiling.values())`, where `floor_or_ceiling[k]` flags a plastic pathway when its mean final weight is
`<= 0.05 * w_init[k]` (floor) or `>= 0.95 * w_max = 85.5` (ceiling) — `_ldc_omission_pilot.py` line 471 in the
worktree copy. Reading the ruling's "v1" as `void_floor_ceiling = 1 (true)`; every number below is given exactly,
so the mapping can be re-read if "v1" meant a different flag.

## 1. (a) Per-seed paired table: ON minus CUT success (dev seeds)

`success` = test-block accuracy of the schedule's relevant race (early for credit, terminal for late).
Per STEP4B section 4, seed 14 is UNDEFINED for the credit rows: its lr-0 probe never explores
(`p_ch0_given_A = p_ch0_given_B = 1.000` in all 16 credit s14 cells' `probe_lr0`) — never a FAIL. The same
signature holds in the late s14 cells (caveat noted there; the fixed rule names the credit rows).

credit_immediate, onset 30 (files `grid_credit_immediate_on30_{on,cut}_s*.json`):

| seed | ON | CUT | ON-CUT |
|---|---|---|---|
| s8 | 0.49 | 0.52 | -0.03 |
| s9 | 0.32 | 0.41 | -0.09 |
| s10 | 0.43 | 0.36 | +0.07 |
| s11 | 0.52 | 0.47 | +0.05 |
| s12 | 0.61 | 0.55 | +0.06 |
| s13 | 0.36 | 0.47 | -0.11 |
| s14 | 0.52 | 0.51 | +0.01 (UNDEFINED, never explores) |
| s15 | 0.37 | 0.63 | -0.26 |

mean -0.0375 (all 8) / -0.0443 (7 defined); sign 3+/4- defined.

credit_immediate, onset 150 (files `grid_credit_immediate_on150_{on,cut}_s*.json`):

| seed | ON | CUT | ON-CUT |
|---|---|---|---|
| s8 | 0.55 | 0.59 | -0.04 |
| s9 | 0.37 | 0.50 | -0.13 |
| s10 | 0.45 | 0.39 | +0.06 |
| s11 | 0.42 | 0.45 | -0.03 |
| s12 | 0.61 | 0.61 | +0.00 |
| s13 | 0.55 | 0.40 | +0.15 |
| s14 | 0.52 | 0.52 | +0.00 (UNDEFINED, never explores) |
| s15 | 0.59 | 0.49 | +0.10 |

mean +0.0138 (all 8) / +0.0157 (7 defined); sign 3+/1= /3- defined.

late_immediate, onset 30 (files `grid_late_immediate_on30_{on,cut}_s*.json`):

| seed | ON | CUT | ON-CUT |
|---|---|---|---|
| s8 | 0.52 | 0.11 | +0.41 |
| s9 | 0.33 | 0.26 | +0.07 |
| s10 | 0.27 | 0.37 | -0.10 |
| s11 | 0.59 | 0.72 | -0.13 |
| s12 | 0.44 | 0.50 | -0.06 |
| s13 | 0.44 | 0.40 | +0.04 |
| s14 | 0.52 | 0.52 | +0.00 (caveat: same no-exploration probe) |
| s15 | 0.53 | 0.28 | +0.25 |

mean +0.0600.

late_immediate, onset 150 (files `grid_late_immediate_on150_{on,cut}_s*.json`):

| seed | ON | CUT | ON-CUT |
|---|---|---|---|
| s8 | 0.42 | 0.41 | +0.01 |
| s9 | 0.43 | 0.33 | +0.10 |
| s10 | 0.37 | 0.25 | +0.12 |
| s11 | 0.30 | 0.66 | -0.36 |
| s12 | 0.03 | 0.26 | -0.23 |
| s13 | 0.26 | 0.72 | -0.46 |
| s14 | 0.52 | 0.52 | +0.00 (caveat: same no-exploration probe) |
| s15 | 0.64 | 0.44 | +0.20 |

mean -0.0775.

Aggregates (pilot, dev seeds): all 32 pairs mean -0.0103, 15 positive / 4 exactly 0 / 13 negative. Credit, the 14
defined pairs: mean -0.0143, 6+/7-. Late, all 16: mean -0.0087, 8+/2=/6-. Per-seed differences span -0.46 to +0.41,
much larger than any block mean. Reading (pilot): no consistent ON-minus-CUT behavioural effect at either onset in
either family; every block mean sits inside the seed-to-seed spread (verify-go rule: an effect inside the spread is
not an effect). This is a paired method read on dev seeds, not a gate verdict.

## 2. (a) ON void_floor_ceiling rate, and what saturates

Rate (pilot, dev seeds; `void_floor_ceiling` of the ON cells):

| block | voided ON cells | seeds |
|---|---|---|
| credit on30 | 5/8 | s9, s10, s12, s14, s15 |
| credit on150 | 5/8 | s8, s9, s11, s12, s14 |
| late on30 | 3/8 | s12, s14, s15 |
| late on150 | 3/8 | s12, s14, s15 |

16/32 ON cells voided overall (the ruling's "3-6 of 8" bracket holds: observed 3-5 per block). Context: CUT voids
12/32, OFF voids 29/32.

What saturates (every flagged synapse classified from `weights_initial`/`weights_final` in the same rows):
**the floor, exclusively — zero ceiling hits in any ON cell.** All 62 flagged synapse-instances in the 16 voided ON
cells are mean final weights at or below 5% of their initial value (effectively the `hebbian_min_weight = 0.0`
bound): 46/62 target `str_d2_*` (D2 MSNs), 16/62 target `str_d1_*`; by source, 26/62 are cortico-striatal
(`ctx_*->str_*`), 36/62 are memory-to-striatal (`mem_*->str_*`). Named: the plastic D2-projecting corticostriatal
and mem-to-striatal weights collapse to the lower bound (examples, `grid_credit_immediate_on30_on_s9.json`:
`ctx_a->str_d2_0` 20.06 -> 0.705; `grid_credit_immediate_on30_on_s12.json`: `ctx_a->str_d2_1` 19.99 -> 0.017,
`mem_a->str_d2_0` 9.92 -> 0.056; `grid_credit_immediate_on30_on_s14.json`: `ctx_a->str_d2_0` 19.85 -> 0.056).
Route-off cells add 7 ceiling hits (`mem_*->str_d1_1` 4, `ctx_*->str_d1_0` 2, `ctx_*->str_d1_1` 1), so the ceiling
(85.5 = 0.95 x 90) is reachable in this build, just not in the ON arm. Mechanistic reading (pilot): the omission
dip's negative teaching term, applied to D2-signed plasticity, drives the D2 pathways to zero on enough trials to
hit the floor in half the ON cells — the same signal that is supposed to shape the map is clipping it.

## 3. (b) Does the omission dip reach striatal plasticity inside the eligibility window?

Trace span vs onset (arithmetic from committed constants; per-trial timing NOT in the artifacts):

- Eligibility trace: `elig_tau_ms` 1000 (decay d = exp(-1/1000) per ms; `_ldc_credit_omission.py` sets
  `reward_eligibility_tau_ms`), tagged at each event mark and read at the outcome mark.
- The dip window is the reward-time window `[REWARD_ON_MS, REWARD_ON_MS + 50)` = `[30,80)` at onset 30 and
  `[150,200)` at onset 150 (row `lhb_rewardwin_ms`; A0.1), inside the 250 ms outcome screen (`--outcome-ms 250`,
  prov argv), which is the interval the DA trace spans and the credited scalar
  `A = lr * sum_t (DA_t - 0.5) * d^t` sums over.
- Remaining eligibility at the dip window: d^30 = 0.970 to d^79 = 0.924 (onset 30); d^150 = 0.861 to d^199 = 0.819
  (onset 150). Both dip windows lie fully inside the outcome window and inside the eligibility span: geometrically
  the dip lands inside the eligibility window at both onsets, attenuated at most ~18% at onset 150.

What the committed aggregates say (per-block means over the 8 dev seeds; each number from the block's 8 files):
unrewarded-trial outcome-window DA min (`da_outcome_unrewarded_min_mean`, mean of per-trial minima over the last
100 train trials): ON 0.4708 at onset 30 and 0.4708 at onset 150 (both families; per-seed range 0.4664-0.4763),
against `da_baseline` 0.5 — a dip of depth ~0.029, and identical at the two onsets. CUT is identical to ON
(0.4706; depth 0.0294), OFF is shallow (0.4954-0.4959; depth ~0.004). The LHb reward-window rates localize the
route's response to the reward-time window (ON rewardwin: 67.8 Hz unrewarded vs 30.9 Hz rewarded at onset 30;
58.0 vs 24.4 at onset 150; `lhb_engagement_a_rewardwin` passes on all defined seeds, failing only s14 at on150,
ratio 1.92). Functional read (pilot): the route produces a DA dip of the same depth at both onsets, so the dip
does reach the outcome window where striatal plasticity reads DA at onset 30 and 150 alike; but it is not
outcome-specific (CUT = ON), so the striatal plasticity sees a punishment-like pause, not a labelled omission
signal — consistent with the STEP4B section-4 "cut reads like on" row.

Eligibility liveness at the outcome mark: `decomposition_last200` (sum over the last 200 train trials of the
per-trial credited dw, split by window) shows the early-cue tags are live and credited at the outcome onset in the
credit family — `early.cue` positive in all 16 credit ON cells (+0.48 to +39.46; e.g.
`grid_credit_immediate_on30_on_s9.json` +39.455, `grid_credit_immediate_on150_on_s13.json` +24.537) — and tiny or
mixed in the late family (`grid_late_immediate_on150_on_s8.json` -0.728 to `grid_late_immediate_on30_on_s12.json`
+8.945). But these sums mix rewarded (positive A) and unrewarded (negative A) trials: the dip's negative teaching
term is inside the sum and cannot be separated per trial from the committed fields.

Per-trial read: NOT computable from the committed cells. The artifacts contain no per-trial DA trace, no per-trial
decomposition, and no latency of the per-trial DA minimum: `da_outcome` (per-trial min/max over the full 250 ms
window) and the per-trial decomposition dicts exist at run time (`_ldc_omission_pilot.py` keeps them in `train` /
`dec`) but only these are committed: `da_outcome_{unrewarded_min,rewarded_max}_mean` over the last 100 train trials
(2 scalars, unlocalized in time; `..._rewarded_max_mean` is null in
`grid_late_immediate_on30_off_s12.json`, `grid_late_immediate_on150_off_s12.json`,
`grid_late_immediate_on150_off_s15.json`, which have 0 rewarded trials in the last 100), `decomposition_last200`
(sums), and `train_reward_sequence` (400 booleans). What a follow-up must record (no mechanism change):
(1) per-trial `da_outcome` min/max plus the sample index of the min within the 250 ms window (or the full per-step
`das`), (2) the per-trial `dec` decomposition dicts the runner already computes at `keep=True`, (3) the `marks`
timestamps. With those, the per-trial question "does the dip (min below baseline inside `[onset, onset+50)`) multiply
non-zero early-tag eligibility on that trial" is a direct read.

## 4. Verdict lines (pilot, dev seeds; method diagnosis only)

- Paired behaviour: ON-minus-CUT success shows no consistent effect (32 pairs, mean -0.0103; per-block means
  -0.078 to +0.060, inside seed spread) at either onset in either family.
- Void rate: 16/32 ON cells carry `void_floor_ceiling` (5/8, 5/8, 3/8, 3/8); what saturates is the floor of the
  plastic D2-dominant cortico-striatal and mem-to-striatal weights (62 flags, 0 ceiling).
- Dip vs plasticity: at onset 30 and 150 alike the dip (depth ~0.029 vs 0.5 baseline) lies inside the 250 ms
  outcome window and inside the 1000 ms eligibility span (0.82-0.97 of the outcome-onset eligibility alive), so it
  reaches striatal plasticity geometrically at both onsets; it is NOT outcome-specific (CUT = ON), and the
  per-trial temporal read is not in the committed artifacts (section 3 lists what a follow-up must record).

Seed 14 is UNDEFINED for the credit rows (never explores), never a FAIL. No six-seed claim, no gate verdict here.
