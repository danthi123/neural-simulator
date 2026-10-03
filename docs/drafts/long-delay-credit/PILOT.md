# PILOT.md — Step 4a outcome-timing pilot (2026-10-03)

Dev seeds only (grid 8/11; diagnosis 9/12/14) · n-train 400 grid / 200 diagnosis · **PILOT, NOT EVIDENCE** — it
informs Claude's Amendment A0; no GO/NO-GO claim is made or implied here.
Runs: 24 grid cells @ 57c0f6ce + 17 diagnosis cells @ 50f52c23e (pilot runner e902550b), data in
`research/findings/raw/_long_delay_credit_bg/dev_pilot/`. Per-cell sanity runs (12, `sanity_early_ch0`) are in
`research/findings/raw/_ldc_outcome_timing/` @ e63b22f6d; their 24 overlapping grid counterparts match the
dev_pilot wave in every row field except `wall_s`, so they are directly comparable.

Bars: **U** = unrewarded in-window DA min ≤ 0.47 · **R** = rewarded max ≥ 0.60 · **I** = |ITI offset at the lr=0
probe| ≤ 0.005. Cell = `min/max/ITI (U R I)` per seed; `ci` = credit_immediate success at the same
on/w/seed; `san` = per-cell sanity_early_ch0 success.

| schedule | onset | w_strio | s8: min/max/ITI (U R I) | s11: min/max/ITI (U R I) | ci s8/s11 | san s8/s11 |
|---|---|---|---|---|---|---|
| credit_immediate | 30 | 30 | 0.498262/0.637974/-0.000003 (✗✓✓) | 0.498179/0.627484/-0.000015 (✗✓✓) | .500/.180 | .115/.750 |
| credit_immediate | 30 | 60 | 0.470600/0.622756/-0.000147 (✗✓✓) | 0.472230/0.613121/-0.000203 (✗✓✓) | .735/.460 | .815/.760 |
| credit_immediate | 150 | 30 | 0.497874/0.638003/+0.000021 (✗✓✓) | 0.497818/0.627415/-0.000050 (✗✓✓) | .530/.170 | .075/.720 |
| credit_immediate | 150 | 60 | 0.471890/0.616770/-0.000218 (✗✓✓) | 0.468728/0.602453/-0.000265 (**✓✓✓**) | .460/.535 | .450/.475 |
| credit_immediate | 300 | 30 | 0.497707/0.637780/+0.000060 (✗✓✓) | 0.497792/0.628046/-0.000048 (✗✓✓) | .485/.180 | .135/.725 |
| credit_immediate | 300 | 60 | 0.480452/0.626113/-0.000339 (✗✓✓) | 0.476677/0.617185/-0.000298 (✗✓✓) | .405/.460 | .415/.505 |
| late_immediate | 30 | 30 | 0.498022/absent/-0.000015 (✗·✓) | 0.498117/0.627755/-0.000029 (✗✓✓) | .500/.180 | .115/.750 |
| late_immediate | 30 | 60 | 0.479725/0.627143/-0.111441 (✗✓✗) | 0.474015/0.613268/-0.122649 (✗✓✗) | .735/.460 | .815/.760 |
| late_immediate | 150 | 30 | 0.497958/0.638011/+0.000033 (✗✓✓) | 0.497948/0.627591/-0.000057 (✗✓✓) | .530/.170 | .075/.720 |
| late_immediate | 150 | 60 | 0.478935/0.623286/-0.131934 (✗✓✗) | 0.461360/0.596271/-0.142120 (✓✗✗) | .460/.535 | .450/.475 |
| late_immediate | 300 | 30 | 0.497754/0.638092/+0.000038 (✗✓✓) | 0.497559/0.627142/-0.000084 (✗✓✓) | .485/.180 | .135/.725 |
| late_immediate | 300 | 60 | 0.482833/0.629009/-0.137647 (✗✓✓) | 0.473273/0.608026/-0.149749 (✗✓✗) | .405/.460 | .415/.505 |

Step 3, lr=0, default constants — rates after the clear (mem_after_clear_hz, hold) one knob at a time vs def:

| knob | s9 (Hz / hold) | s12 (Hz / hold) |
|---|---|---|
| def (credit_immediate) | 32.27 / 0.850 | 28.64 / 0.839 |
| clear_pA 300 | 84.86 / 0.511 | 84.74 / 0.547 |
| clear_pA 1200 | 0.00 / 1.000 | 0.00 / 1.000 |
| CLEAR_MS 125 | 82.06 / 0.495 | 81.75 / 0.551 |
| CLEAR_MS 500 | 0.00 / 1.000 | 0.00 / 1.000 |
| w_clear_fs 15 | 85.24 / 0.544 | 84.61 / 0.466 |
| w_clear_fs 60 | 0.00 / 1.000 | 0.00 / 1.000 |
| def (late_immediate) | 32.40 / 0.854 | 28.78 / 0.785 |

Seed 14 (credit_immediate, def): p(ch0|A) = 1.000, p(ch0|B) = 1.000 (mem_after_clear 0.0493 Hz, hold 1.000,
response 1.0) — equal in both organ contexts, so no A-vs-B channel asymmetry at the readout.

- **DA signs in exactly 1 of 24 cells:** credit_immediate onset 150 / w60 / s11 (min 0.468728,
  |ITI| 0.000265, max 0.602453 — all three bars). Same cell, s8 misses the min bar by 0.0019 (0.471890).
  late_immediate: 0/12; its min-passing cell (0.461360) fails ITI at −0.142120.
- **The ITI leak is w_outc_strio × schedule, not onset:** all six late w60 cells sit −0.111…−0.150
  (bar 0.005), all six late w30 cells and all twelve credit cells sit |x| ≤ 0.00034. Onset 30→300 ms changes
  the ITI offset little within a w group.
- **Missing field (only one in 41 runs):** `da_outcome_rewarded_max_mean` = absent in late_immediate s8 on30
  w30 — that run earned 6/400 rewards (last-100 rate 0.0), so no rewarded maximum exists to record. All 24
  grid + 17 diagnosis cells are present (s12 carries all 7 knobs, including cp300/cp1200).
- **Behavior stays below the A0 0.8 line where DA signs:** credit_immediate success 0.170–0.735 (0/12 ≥ 0.8),
  sanity 0.075–0.815 (1/12 ≥ 0.8), late_immediate success 0.000–0.765; at the signing cell 0.46/0.54 (ci) and
  0.45/0.48 (sanity). mem_hold_acc = 1.000 in all 24 cells — memory intact everywhere.
- **s9/s12 fail to clear on a monotone clear dose, not a seed pathology:** def leaves 32.27/28.64 Hz (matches
  the banked anchors 32.2699/28.6447 — the `--clear-ms` no-op repro held); every stronger-or-longer knob
  reaches 0.00 Hz / hold 1.000 on both seeds; every weaker-or-shorter knob is worse (82–85 Hz). Same on
  late_immediate (32.40/28.78 Hz). Which pool keeps the stale cue is not deliverable (mem_after_clear_hz
  averages mem_a+mem_b).
- **s14 never explores, symmetric across organ channels:** p(ch0|A) = p(ch0|B) = 1.000 at lr=0; the bias is
  ch0-vs-ch1, not A-vs-B; MSN weights initially symmetric (mem_a D1 9.998 vs mem_b 9.849) and lr=0 freezes all
  weights; motor-pool latency is not recorded in these runs.

**A0 read:** one onset ≤300 ms signs the DA (credit_immediate, 150 ms), so the striosome-GABA_B critic is not
banked; sanity/credit_immediate stay below 0.8 on most seeds where it signs, so the evidence points to the
eligibility tag's action-specificity (marginal signature: 1 cell, 1 seed, margin 0.0013 — flagged for
Claude's ruling; a both-seeds reading would select the timed-omission branch instead).
