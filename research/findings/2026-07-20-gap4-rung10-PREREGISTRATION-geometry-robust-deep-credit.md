# gap#4 RUNG 10 — PRE-REGISTRATION: deep credit under HONEST geometry, with a GEOMETRY-ROBUST metric

**Filed 2026-07-20 before any rung-10 result exists.** Seeds **2100-2111** (twelve; ~1/6 Poisson draws degenerate).
Per-seed Poisson field placement (Rich 2014). Rule OFF (plain BTSP — the valid, characterized arm).

## Why the metric is new, and why it is not goalpost-moving

Rung 9 was instrument-INVALID: its `read_hit` derived a BACKWARD-shifted expected bin (`tgt_bin - shift_bins`),
but rung 3d MEASURED the L2 read at offset **+0** (the backward shift happens once, upstream). So the old metric
demanded a peak that the mechanism, by rung 3d's own measurement, does not produce. `read_hit = 0` on every seed
was the metric failing, not deep credit.

The geometry-robust metrics, derived per-seed from the actual layout:
- **`read_hit_l2`** = L2 peak within +/-2 bins of the plateau-delivered cell's field (the offset-0 expectation
  rung 3d ESTABLISHED, not a new choice);
- **`c_nn`** = L2 response at the plateau cell vs at its CLOSEST other field (nearest-neighbour contrast, computed
  from the actual per-seed field positions rather than a fixed "adjacent = 4 bins" assumption).

This is not moving the goalpost: the offset-0 expectation is rung 3d's PRE-REGISTERED, lesion-confirmed measurement,
carried forward. The change is making the neighbour set track the actual geometry instead of assuming even spacing.

## PRE-REGISTERED GATE + PREDICTIONS (usable seed = `map_ok = 1`)

1. **P1 — the read is LEARNED (must hold):** `r_plat` (L2 response at the plateau cell) on MAIN is >= 5x the
   `C1_frozen` and `C3_moat` values, on >= 5/6 usable seeds. *(If this fails, retract rung 3d.)*
2. **P2 — read ACCURACY at offset 0:** MAIN `read_hit_l2 = 1` on >= 4/6 usable seeds.
3. **P3 — nearest-neighbour contrast:** on usable seeds whose nearest neighbour is >= 3 bins away (the
   biologically-common non-adjacent case), MAIN `c_nn >= 1.60x` on >= half of them.

**INTERPRETATION FIXED IN ADVANCE:**
- **P1 + P2 pass** ⇒ deep credit across a layer reads the RIGHT cell under honest geometry — the keystone stacking
  half confirmed on the biologically-correct layout, honest scope (read-accuracy).
- **P1 + P2 pass, P3 mixed** ⇒ the read is accurate; its neighbour-contrast is geometry-dependent (consistent with
  rung 8), which is expected, not a failure.
- **P1 fails** ⇒ retract rung 3d.

## Honest scope

- Degenerate draws (`map_ok = 0`) are excluded and COUNTED, never silently dropped.
- `c_nn` is `nan` when only one field is measurable; those seeds do not enter P3.
- This tests deep-credit READ-ACCURACY under honest geometry, not a universal 2x-selectivity (rung 8 showed that
  number is geometry-determined).
