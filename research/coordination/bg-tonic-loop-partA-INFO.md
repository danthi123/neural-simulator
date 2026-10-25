# bg-tonic-loop Part A — tonic-epoch probe: INFO table (agent-filed)

**Status:** INFO, per ruling bg-tonic-loop-front-1020b ("post the Part A table as INFO when it
lands"). Raw reads only — no verdict; Claude rules. PROVISIONAL agent-filed.

## Provenance

- **Runner:** `research/runners/bg_tonic_loop_probe.py` @ `1b96700be` (branch
  `research/bg-tonic-loop`), step 0b Part A, 4 arms × STN {40, 50} pA × dev seeds 9–16 = 64 cells,
  64/64 outputs present (2 cells re-queued verbatim after start-death; all reads below are from
  complete cells).
- **Arms:** A0 = hybrid reference; A0p = A0 + `BRAIN_BG_PRESET_CELLS` (preset-centred
  heterogeneity, `param_assert` compares only the 4 param arrays); A5 = bounded condition only;
  A5p = both. `BRAIN_BG_BOUNDED_COND` on for A5/A5p. Switches are DEFAULT-OFF on merged main.
- **Artifacts:** `research/findings/raw/_bg_live_seam/tonic_loop/1b96700/stn<cur>/s<seed>_<arm>.json`
  (+ `.prov.json` sidecars). Region fields = `regions.<name>.{rate_mean, silent, v_steps_below}`;
  STN read = `regions.selector_stn`, GPe read = `regions.gpe_0`. GPe drive 100 pA, 200 warmup +
  1000 measure steps, dt 1.0 ms, n = 600 total neurons (20 STN / 16 GPe per region index).
- **Assertion:** `param_assert.asserted` = the A0p-vs-A0 hard assert (snapshot differs in exactly
  the 4 param arrays) — fired 8/8 on A0p cells, 0/8 elsewhere (not applicable to those arms).

## Reads: STN rate_mean Hz (silent/20) per region index

| seed | 40 pA A0 | 40 pA A0p | 40 pA A5 | 40 pA A5p |
|---|---|---|---|---|
| s9 | 10.50 (6/20) | 0.00 (20/20) | 10.50 (6/20) | 0.00 (20/20) |
| s10 | 36.40 (2/20) | 0.00 (20/20) | 36.15 (2/20) | 0.00 (20/20) |
| s11 | 22.05 (2/20) | 0.00 (20/20) | 22.65 (2/20) | 0.00 (20/20) |
| s12 | 9.90 (11/20) | 0.00 (20/20) | 9.70 (11/20) | 0.00 (20/20) |
| s13 | 16.90 (2/20) | 0.00 (20/20) | 15.80 (2/20) | 0.00 (20/20) |
| s14 | 19.65 (1/20) | 0.00 (20/20) | 20.25 (1/20) | 0.00 (20/20) |
| s15 | 31.05 (1/20) | 0.00 (20/20) | 31.10 (1/20) | 0.00 (20/20) |
| s16 | 26.25 (5/20) | 0.20 (19/20) | 26.35 (5/20) | 0.20 (19/20) |

| seed | 50 pA A0 | 50 pA A0p | 50 pA A5 | 50 pA A5p |
|---|---|---|---|---|
| s9 | 18.40 (0/20) | 0.20 (18/20) | 18.30 (0/20) | 0.25 (18/20) |
| s10 | 47.45 (0/20) | 0.10 (19/20) | 48.35 (0/20) | 0.10 (19/20) |
| s11 | 32.60 (0/20) | 0.00 (20/20) | 32.55 (0/20) | 0.00 (20/20) |
| s12 | 24.10 (0/20) | 0.15 (19/20) | 24.25 (0/20) | 0.15 (19/20) |
| s13 | 24.25 (0/20) | 0.60 (15/20) | 23.45 (0/20) | 0.55 (15/20) |
| s14 | 32.50 (0/20) | 0.10 (19/20) | 32.95 (0/20) | 0.10 (19/20) |
| s15 | 37.20 (0/20) | 0.00 (20/20) | 37.50 (0/20) | 0.00 (20/20) |
| s16 | 35.65 (0/20) | 0.80 (15/20) | 35.00 (0/20) | 0.80 (15/20) |

## Summary rows (8 seeds each)

| current | arm | mean Hz | range Hz | silent/160 | param_assert |
|---|---|---|---|---|---|
| 40 | A0 | 21.59 | 9.90–36.40 | 30 | 0/8 (n/a) |
| 40 | A0p | 0.03 | 0.00–0.20 | 159 | 8/8 |
| 40 | A5 | 21.56 | 9.70–36.15 | 30 | 0/8 (n/a) |
| 40 | A5p | 0.03 | 0.00–0.20 | 159 | 0/8 (n/a) |
| 50 | A0 | 31.52 | 18.40–47.45 | 0 | 0/8 (n/a) |
| 50 | A0p | 0.24 | 0.00–0.80 | 145 | 8/8 |
| 50 | A5 | 31.54 | 18.30–48.35 | 0 | 0/8 (n/a) |
| 50 | A5p | 0.24 | 0.00–0.80 | 145 | 0/8 (n/a) |

## What the reads show (no verdict)

- The two preset-centred arms (A0p, A5p) read near-zero STN rate at BOTH currents (0.00–0.80 Hz,
  145–159/160 neurons silent); the two hybrid arms (A0, A5) read 9.70–48.35 Hz with per-seed
  scatter (40 pA A0 range 9.90–36.40; 50 pA A0 range 18.40–47.45).
- A0 vs A5 per-seed differences are ≤1.1 Hz everywhere (bounded condition alone ≈ reference).
- At 40 pA A0, seed means in the 10–30 Hz band: 5/8 (s10 36.40, s15 31.05 above; s12 9.90 below).
  This cell set is revision `1b96700be` with a 200+1000-step window, so it is NOT numerically
  interchangeable with the `8177576` sweep's 40 pA row (6/8 there).
- Per-seed pattern is stable across currents within an arm (e.g. s10 highest at both currents).
