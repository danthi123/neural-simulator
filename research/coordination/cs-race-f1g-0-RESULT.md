# [cs-race-f1g-0] RESULT — STEP 0 CALIBRATION: **STOP-CAL**

- Packet: `research/coordination/cs-race-f1g-PACKET.md` (RULING
  cs-race-f1g-1022 @ f68837581). Verdicts below are the packet's frozen
  rules applied mechanically (booleans re-derived from raw fields, 0
  mismatches vs the runner) — no judgment labels added.
- Grid: W_MM {10,15,20,30,40,60} x W_IM {20,40,80}, W_MI=20, NIM=8,
  W_GIS=30, encode-words on, seeds 41-48. Pool runs @ 74783b4a6
  (gate-seed canary PASS pool41 35 s; preflight CLEAN runbook@74783b4a6).
- Full reads: `research/findings/raw/_cs_race/f1g/step1g_stats.json`
  (+ `.prov.json`), inputs `cal_wmm*_wim*.json` (18/18, 8 seeds each).

## Every cell's items (pass-counts over seeds 41-48)

<!--derived-->

| cell | C1 2-20 Hz | C2 <1 Hz | C3 | C4 runner | C4 packet primed | C1 range Hz | pulse-phase spikes | C5a med [range] pA | C5b med [range] pA |
|---|---|---|---|---|---|---|---|---|---|
| wmm10_wim20 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm10_wim40 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm10_wim80 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm15_wim20 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm15_wim40 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm15_wim80 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm20_wim20 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm20_wim40 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm20_wim80 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm30_wim20 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm30_wim40 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm30_wim80 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm40_wim20 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm40_wim40 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm40_wim80 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm60_wim20 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm60_wim40 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |
| wmm60_wim80 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0..0 | 0..0 | 0.0 [0.0, 0.0] | 0.0 [-1.3544, 0.0] |

All 18 cells are identical on every summary. Runner-vs-scorer
mismatches: 0 of 144 seed-items. Readback flags (nmda, STF,
heterogeneity, w_gis 30) present in all files; latch exclusion 30.0.

## Selection

- Chosen cell: **NONE.** No cell passes C1-C4 on all 8 seeds under
  either rule (the runner's pass dict, or the packet-literal rule that
  additionally requires the forward gate to be primed non-zero).

## STOP verdicts (frozen packet rules)

- **STOP-CAL: FIRED** — no cell passes C1-C4 on 8/8. C1 fails 0/8 at
  every one of the 18 cells; the pulse produces 0 spikes in all 1,152
  slot-probe draws (18 cells x 8 seeds x 4 probe draws... per-seed
  range 0..0).
- STOP-SMOKE: not fired — the instrument is not broken: 18/18 files
  parse, every item readable, flags read back, 0 scorer mismatches.
- STOP-ID: PASS 8/8 (separate identity run, no new flag; committed
  with the code as 7e9f59a79).

## C5 against the flip scale

- Clause medians 0.0 pA in every cell (clause a range [0.0, 0.0];
  clause b [-1.3544, 0.0]) versus the +4..6 pA that moves this race
  (steer arm +4 -> 7/10 wins; f1d rows 3-4 +4=5/8, +8=7/8). The trace
  grid delivers 0% of the flip scale because there are no spikes for
  the trace to run on: the weights are bit-inert across
  W_MM 10->60 and W_IM 20->80.

## Root cause (measured, this step)

1. **MENT_PA 40 pA is sub-rheobase for the mention slots.** Seed 41
   probe, cold and warm identical: 40 pA -> 0 spikes over 150 ms,
   60 pA -> 3, 100 pA -> 19 (rheobase is between 40 and 60), while
   the engram pulse (126 pA, 60 ms) fires 54 spikes — the cells are
   healthy. C1 (2-20 Hz after the pulse) therefore fails at every
   grid point by construction; no (W_MM, W_IM) can pass it. The first
   probes also had the trace on the INHIBITORY `receptor=` key instead
   of the excitatory `exc_receptor` (nmda_slow); fixed — but the fix
   cannot matter while the pulse makes no spikes (warming, pool
   on/off, and encode-words were each ruled out separately).
2. **The base protocol's mention path is already silent** (not
   introduced by this step): clause-b with a single-word mention reads
   0.0000 pA delivered differential in the pure base circuit with the
   pool off (pulse sub-rheobase hits race() the same way). The
   >=3-words branch (MENT_MAX_PA=90) would cross rheobase. Consequence
   flagged, not re-verdicted: f1d/f1e clause-b readings were taken on
   a mention path that carries no spikes at 40 pA — their committed
   verdicts stand, but the clause-b bar's meaning needs Claude's read.

## What is needed (posted as NEEDS-CLAUDE ASK)

- The packet froze MENT_PA=40. To make C1 measurable, the pulse must
  reach rheobase (measured 40-60 pA; recommend margin above 60) or
  the slot excitability must change. Either way a packet-frozen
  constant moves, which is Claude's call; after the fix, step 0
  re-runs the same 18-cell grid unchanged.