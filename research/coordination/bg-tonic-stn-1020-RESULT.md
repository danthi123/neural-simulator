# bg-tonic-stn-1020 — STN drive sweep with new 32/34/36/38 pA points: PROVISIONAL RESULT

**Status:** PROVISIONAL (agent-filed, needs Claude verification). Executes the decision tree in
RULING bg-tonic-stn-cliff-1020 as corrected by -1020e/-1020f, under bg-seam-calendar-1020
(table only, no re-pick). Raw reads + tree outcome only; Claude rules.

## Preconditions / provenance

- Runner `research/runners/host_share_bg_seam.py` @ `8177576ae`, dev seeds 9–16 (never gate
  seeds), GPe 100 pA, STN drive in {10, 15, 20, 25, 30, 32, 34, 36, 38, 40} pA = 80 cells,
  80/80 with all 3 rate sources present EXCEPT stn38/s15 (6/7: `seam_lesion_value` pending —
  its line is still queued; `seam_b` and `seam_readout` are present and carry identical
  `tonic_rates_hz` values for that cell).
- Rate read = mean of every `tonic_rates_hz["selector_stn"]` entry across
  `seam_b.json` / `seam_readout.json` / `seam_lesion_value.json` (the exact extraction of the
  filed 1019 table: it reproduces that RESULT's 40 pA range 7.30–32.68 Hz and its 30 pA row
  0.1/0.0/0.2/0.0/0.0/0.0/0.0/0.2 after rounding). Band = 10–30 Hz on the 8 seed means.
- Artifacts: `research/findings/raw/_bg_live_seam/stn_sweep/8177576/stn<cur>/s<seed>/s<seed>/`.
- The 32–38 pA cells measure the HYBRID organ (arm A0-equivalent) per bg-seam-calendar-1020.
  Corrections honored: -1020e (RS-centred heterogeneity overwrite; the saddle-node/3.6 Hz
  analyses are VOID), -1020f (withdraws -1020d(1), so no synaptic-input report is filed).

## Rate-by-current table: STN tonic mean Hz per seed (all 8 seed means; band 10–30 Hz)

| current pA | s9 | s10 | s11 | s12 | s13 | s14 | s15 | s16 | all-8 in band |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no |
| 30 | 0.08 | 0.00 | 0.24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | no |
| 32 | 3.62 | 3.79 | 1.92 | 0.33 | 2.96 | 2.46 | 0.25 | 0.75 | no |
| 34 | 8.15 | 6.82 | 1.72 | 0.54 | 6.22 | 10.82 | 8.75 | 0.87 | no |
| 36 | 9.22 | 13.48 | 8.18 | 2.89 | 10.85 | 14.99 | 14.19 | 2.07 | no |
| 38 | 12.15 | 24.68 | 17.03 | 3.57 | 23.03 | 27.50 | 21.17 | 6.62 | no (6/8) |
| 40 | 15.28 | 28.93 | 22.93 | 7.30 | 26.11 | 32.68 | 24.44 | 14.06 | no (6/8) |

## Tree outcome (executed, not self-labeled beyond the branches Claude pre-defined)

- Branch (a) does NOT fire: no current puts all 8 dev-seed STN means at 10–30 Hz, so the
  2-pA-neighbour condition is never reached (best rows are 38 pA and 40 pA at 6/8; ≤30 pA is
  silent or near-silent; 32/34/36 pA are below band for most seeds).
- Branch (b) FIRES: per -1020(b) the drive sweep STOPS and this RESULT is filed; the
  drive-only lever is banked as a method. Per bg-seam-calendar-1020 there is NO re-pick and
  no adoption here: gate seeds are never run on this sweep, and any adoption decision is
  Claude's.

## What the new points add to the 1019 cliff picture (reads, no verdict)

- The 30→40 pA cliff is now sampled: 32 pA 0.25–3.79, 34 pA 0.54–10.82, 36 pA 2.07–14.99,
  38 pA 3.57–27.50 Hz — a monotone ramp with the same per-seed ordering stress at every
  current where STN fires (s12 and s16 stay low, s14/s10 stay high), so lowering drive buys
  neither band compliance nor variance reduction anywhere in 10–40 pA.
- stn38/s15 pending `seam_lesion_value` cannot change any branch: its two present sources
  agree (pool ~21.2 Hz, in band), and 38 pA is already not-all-8 (s12 3.57, s16 6.62) either
  way.
- Cross-ref: `research/coordination/bg-tonic-stn-1019-RESULT.md` (original six-current STOP)
  — unchanged; this table extends it rather than replacing it.
