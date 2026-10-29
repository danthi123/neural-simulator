# cs-race f1g step-0c RESULT — DRIVE chosen 150 pA; grid still STOP

Date 2026-10-22. Runner `efd2fe37a` (item `d0`, `--drive-probe`,
`--ment-pa`). Rulings: `cs-race-f1g-0b-1022` (this step),
`claude-batched-1022` (execution mode). Seeds 41-48 unless noted.

## Verdict (driver stops for a next ruling)

- **STOP-DRIVE = False** — engagement is solved: **150 pA** is the
  smallest value of {80, 100, 120, 150, 200, 300} at which every cell
  of every mention slot fires >= 1 spike within a 50 ms word AND >=
  1 spike in a 150 ms pulse, on 8/8 seeds (trace off, W_MM = 0,
  W_IM = 0; cue report collected at CUE_PA 60, read-only).
- Grid re-run at 150 pA (18 cells, frozen rules): **selection NONE**
  → **STOP-CAL = True**, **STOP-C0 = True**, **STOP-SMOKE = True**,
  **STOP-ID = True (passed 8/8)**. No cell passes C1-C4 on 8/8.
- Runner/scorer agreement: 0 mismatches, design complete 18/18,
  `ment_pa` = 150 in every cell.

## DRIVE sweep (item d0, trace off)

| pa | pass_both (of 8) | zero-spike 50 ms pulses (of 72) | verdict |
|---|---|---|---|
| 80 | 0/8 | 72 | fail |
| 100 | 0/8 | 72 | fail |
| 120 | 2/8 | 12 | partial |
| **150** | **8/8** | **0** | **CHOSEN (min50=1, min150=2)** |
| 200 | 8/8 | 0 | pass |
| 300 | 8/8 | 0 | pass (min50=2, min150=4) |

## Grid at 150 pA — item pass counts (c0,c1,c2,c3,c4), of 8 seeds

```
wmm10_wim20  8,0,8,0,8   rates 0..1.5 Hz
wmm10_wim40  8,0,8,0,8   rates 0..1.5 Hz
wmm10_wim80  8,0,8,0,8   rates 0..1.0 Hz
wmm15_wim20  8,2,8,0,8   rates 0.4..2.3 Hz
wmm15_wim40  8,2,8,0,8   rates 0..2.5 Hz
wmm15_wim80  8,0,8,0,8   rates 0..1.7 Hz
wmm20_wim20  8,5,8,0,5   rates 2.3..32.1 Hz, c5b med 22.9 pA
wmm20_wim40  8,5,8,0,8   rates 1.3..20.6 Hz
wmm20_wim80  8,5,8,0,8   rates 0.2..4.0 Hz
wmm30_wim20  1,0,8,0,0   rates 46..94 Hz,  c5b med 153.6 pA
wmm30_wim40  5,0,8,0,1   rates 26..69 Hz,  c5b med 101.9 pA
wmm30_wim80  8,1,8,0,4   rates 18..36 Hz,  c5b med 39.8 pA
wmm40_wim20  2,0,8,0,0   rates 76..85 Hz
wmm40_wim40  0,0,8,0,0   rates 59..77 Hz
wmm40_wim80  1,0,8,0,0   rates 32..61 Hz
wmm60_wim20  8,0,8,0,0   rates 58..60 Hz,  c5b med 93.1 pA
wmm60_wim40  0,0,8,0,0   rates 56..58 Hz
wmm60_wim80  0,0,8,0,0   rates 51..58 Hz
```

C0 summary (1008 pulses, 144 seed records): 79 pulses with zero
total spikes remain (all in W_MM >= 30 cells), min cell 0, max
total 89; 11/18 cells now pass C0 8/8 (step-0b at 60 pA: 0/18).

## Facts this step establishes

1. **Engagement vs. rate-window conflict.** The same drive that
   makes every cell engage (>= 150 pA) pushes W_MM >= 30 cells to
   25-94 Hz (C1 ceiling 20 Hz) while W_MM <= 15 cells stay at
   0-2.5 Hz (C1 floor 2 Hz); within one cell the seed spread can
   span 0.19-32 Hz (wmm20_wim20). No single global drive fits the
   frozen 2-20 Hz band on 8/8 seeds in any of the 18 cells.
2. **C3 is dead everywhere** (word2-alone rate < 2 Hz with word1
   silent: 0/8 in all 18 cells, unchanged from step-0). Consistent
   with the pre-existing `c5b k0 = 0.0` finding: the association
   path never carries current.
3. **Clause-b fires for the first time** at trace cells (medians
   39-154 pA at wmm20/30/60 — rival side moves), but clause-a
   (favoured) stays ~0 everywhere (max median 0.05 pA), so
   STOP-SMOKE stands and the cells that move violate C1/C4.
4. High-W_MM C0 failures are suppression-shaped: rates are high
   overall yet some slot's cells take 0 spikes during their own
   pulse (winner-slot inhibition), not sub-rheobase (step-0b's
   cause, now fixed by drive).

## Proposal for the next ruling (pick one; bars are yours)

A. **Per-band drive** — sweep drive per W_MM band (high-trace
   cells engage at lower pa when W_MM > 0; the 150 floor was
   measured at W_MM = 0). One sweep, smallest 8/8 per band.
B. **Enable the association path** — investigate/flip `k0 = 0.0`
   (suspected root of C3 0/8); a default flip is reserved (§2).
C. **Restate the C1 window** (bar change, §2) with the measured
   achievable bounds if A cannot fit 2-20 Hz jointly with C0.

## Data / provenance

`drive_paq{80,100,120,150,200,300}.json` (+ `.prov.json`),
`cal150_wmm*_wim*.json` (36 files), `step1g0c_stats.json` (+
`.prov.json`) — all under `research/findings/raw/_cs_race/f1g/`.
Scorer `a_f1g0c_score2.py` (mechanical re-derivation of C0-C4 + C5
medians; inputs pool runs @ `efd2fe37a`).