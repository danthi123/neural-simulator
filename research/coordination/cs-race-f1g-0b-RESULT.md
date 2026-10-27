# [cs-race-f1g-0b] RESULT — STEP 0b (`--ment-pa 60`): **STOP-C0 + STOP-CAL + STOP-SMOKE**

- Ruling executed: `cs-race-f1g-0-1022` (f8b336f1d) parts A (record
  AMENDMENTs efbea85b2 + FAILURE_LOG row 45c6e6d04), B (`--ment-pa`
  flag, default 40 → identity exact) and C (this re-run). Current set
  to CUE_PA = 60 exactly as ruled; no other constant changed. All
  verdicts below are the frozen rules applied mechanically (booleans
  re-derived from raw fields, 0 mismatches vs the runner).
- Grid: W_MM {10,15,20,30,40,60} x W_IM {20,40,80}, W_MI=20, NIM=8,
  W_GIS=30, encode-words on, seeds 41-48, `--ment-pa 60`. Pool runs @
  b8a1c11ad (code 686fd67a4; gate-seed canary PASS pool41 6 s;
  preflight CLEAN runbook@b8a1c11ad). STOP-ID: PASS 8/8 on the default
  40 (separate identity run, `identity_postmentpa.json`).
- Full reads: `research/findings/raw/_cs_race/f1g/step1g0b_stats.json`
  (+ `.prov.json`), inputs `cal60_wmm*_wim*.json` (18/18, 8 seeds each,
  `ment_pa` = 60 recorded in every file).

## Every cell's items (pass-counts over seeds 41-48)

<!--derived-->

| cell | C0 | C1 2-20 Hz | C2 <1 Hz | C3 | C4 | C1 range Hz | pulse-phase spikes | C5a med [range] pA | C5b med [range] pA |
|---|---|---|---|---|---|---|---|---|---|
| wmm10_wim20 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.5769 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm10_wim40 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.5769 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm10_wim80 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.5769 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm15_wim20 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.5769 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm15_wim40 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.5769 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm15_wim80 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.5769 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm20_wim20 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.7692 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm20_wim40 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.7692 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm20_wim80 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.7692 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm30_wim20 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.9615 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm30_wim40 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.9615 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm30_wim80 | 0/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0..0.9615 | 0..7 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm40_wim20 | 0/8 | 1/8 | 8/8 | 0/8 | 4/8 | 0..75.7692 | 0..7 | 0.0 [0.0, 0.0] | 5.43645 [-1.3544, 165.317] |
| wmm40_wim40 | 0/8 | 1/8 | 8/8 | 0/8 | 4/8 | 0..66.3462 | 0..7 | 0.0 [0.0, 0.0] | 5.43645 [-1.3544, 140.3604] |
| wmm40_wim80 | 0/8 | 1/8 | 8/8 | 0/8 | 4/8 | 0..50.5769 | 0..7 | 0.0 [0.0, 0.0] | 5.43645 [-1.3544, 105.6288] |
| wmm60_wim20 | 0/8 | 0/8 | 8/8 | 0/8 | 2/8 | 0..63.6538 | 0..9 | 0.0 [0.0, 0.0] | 102.33155 [-1.3544, 110.6396] |
| wmm60_wim40 | 0/8 | 0/8 | 8/8 | 0/8 | 2/8 | 0..60.7692 | 0..9 | 0.0 [0.0, 0.0] | 99.249 [-1.3544, 106.5569] |
| wmm60_wim80 | 0/8 | 0/8 | 8/8 | 0/8 | 2/8 | 0..56.5385 | 0..9 | 0.0 [0.0, 0.0] | 89.57325 [-1.3544, 113.0911] |

Runner-vs-scorer mismatches: 0 of 144 seed-items (C0-C4). Readback
flags (nmda, STF, heterogeneity, w_gis 30) present in all files; latch
exclusion 30.0.

## Item C0 detail (the new ruling item)

- Definition as ruled: every pulsed slot fires ≥ 1 spike **per cell**
  during its 150 ms pulse, on 8/8 seeds.
- 144 seed records (18 cells x 8 seeds), 1,008 pulse events (7 per
  record): **all 144 seed records FAIL C0.**
- Per-cell minimum = 0 on every one of the 1,008 pulses (spikes, when
  they come, concentrate on a minority of the 8 slot cells).
- Even the weaker slot-total reading fails: **164 of 1,008 pulses have
  0 spikes in the whole slot**; totals range 0..9 spikes.
- So at 60 pA the mention pulse fires sometimes (phase 0..9 spikes) but
  engagement is neither per-cell nor reliable per-pulse.

## STOP verdicts (frozen rules; ruling C: do NOT raise the current)

- **STOP-C0: FIRED** — C0 fails at 60 pA under both readings (0/8 in
  every cell; 164 zero-spike pulses). Per the ruling the current is
  NOT raised.
- **STOP-CAL: FIRED** — selection cell NONE: C1 passes at most 1/8
  seeds (15 of 18 cells 0/8); C3 fails 0/8 in every cell; C4 degrades
  to 4/8 (W_MM 40) and 2/8 (W_MM 60) where the fired pulse now drives
  burst/latch.
- **STOP-SMOKE: FIRED** — C5 clause-(a) median 0.0 pA in all 18 cells
  (< 2.0).
- STOP-ID: PASS 8/8 (identity at default 40; flag-off identity exact).

## Observation (recorded, not adopted)

At W_MM 40/60 the 60 pA pulse engages the slow-NMDA trace enough to
give the first non-zero clause-(b) medians of the step-0 series (5.4 pA
at W_MM 40; 89.6-102.3 pA at W_MM 60, ranges up to 165 pA), while
clause (a) stays 0.0 everywhere. Those cells fail C1 (rate > 20 Hz on
the firing seeds) and C4 (latch/burst), so none is selectable under the
frozen rules. No value is chosen from this table.

## Next

- STOP per ruling cs-race-f1g-0-1022 (C): lane holds for Claude's next
  ruling; NEEDS line `[cs-race-f1g-0b]` posted with this RESULT.