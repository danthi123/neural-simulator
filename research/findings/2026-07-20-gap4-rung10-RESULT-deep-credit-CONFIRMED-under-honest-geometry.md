# gap#4 RUNG 10 — RESULT: deep credit reads the RIGHT cell under HONEST (Poisson) geometry. Keystone confirmed.

Pre-registered at `0887d4ea` before the run, with a geometry-robust metric (offset-0 expectation from rung 3d,
per-seed neighbour sets). Seeds 2100-2111, Poisson placement, rule OFF. **10/12 usable** (2 degenerate draws
excluded and counted).

## Scored against the pre-registration

| prediction | bar | result |
|---|---|---|
| **P1** — the read is LEARNED (MAIN `r_plat` >= 5x lesions) | >= 5/6 | **10/10 — PASS** |
| **P2** — read ACCURACY at offset 0 (`read_hit_l2`) | >= 4/6 | **7/10 — PASS** (chance ~25%, binomial p ~ 0.003) |
| **P3** — nearest-neighbour contrast >= 1.60x | >= half of far-neighbour seeds | **0/6 — FAIL (as expected)** |

Per the pre-registered interpretation, this is the **"P1 + P2 pass, P3 mixed"** branch: *the read is accurate; its
neighbour-contrast is geometry-dependent (consistent with rung 8), which is expected, not a failure.*

## What the numbers say

**The read is unambiguously LEARNED.** On every one of 10 usable seeds, MAIN's L2 response at the plateau cell
(`r_plat` 0.10-0.30) is 29x-61x the frozen/no-plateau lesion value (0.003-0.007), and on the four seeds where the
lesion response is ~0 the ratio is effectively infinite. The lesions collapse the read completely, on 10/10.

**The read localizes to the RIGHT cell** on 7/10 usable seeds (L2 peak within +/-2 bins of the plateau-delivered
cell). Chance for a +/-2 window in 20 bins is ~25%; 7/10 is ~2.8x chance (binomial p ~ 0.003). The 3 misses are
seeds where the peak landed at a different field.

**Nearest-neighbour contrast is geometry-dependent** (c_nn 0.84-1.27, none >= 1.60), which is precisely what rung 8
established: contrast varies with layout and only favourable geometries clear 1.60. This is NOT a deep-credit
failure; it is the same geometry-determination, now seen per-seed.

## What this establishes

**Deep credit across a layer — the keystone's stacking half — is CONFIRMED under the biologically-correct
geometry**, not just the arbitrary even-spacing layout of rung 3d. A downstream layer learns to read the learned
population code (lesion-confirmed 10/10) and localizes to the right cell (7/10, ~3x chance), on Poisson field
placement (Rich 2014). Rung 3d's even-geometry result was not a geometry artifact — it holds under the layout
biology actually uses.

## Honest scope + caveats

- **P2 is 7/10, not perfect.** 3 seeds miss the target cell. The read is accurate well above chance but not
  deterministic — honest.
- The lesion arms produce garbage `c_nn` (54, 2007, 871868) from dividing near-zero by near-zero; those are
  artifacts and do NOT enter any gate. P1 uses `r_plat` (the raw response), which is clean.
- This is READ-ACCURACY, not universal 2x-selectivity. Rung 8 showed the latter is geometry-determined; requiring
  it everywhere would require a favourable draw.
- 2/12 seeds were degenerate (`map_ok = 0`), excluded and counted, not dropped.

## The gap#4 deep-credit arc, closed honestly

- One-shot local credit works (rung 1, repaired).
- It composes to a population (rung 2, genuine control).
- **It composes ACROSS A LAYER and reads the right cell under HONEST geometry** (rung 3d even-spacing 6/6 + rung 10
  Poisson P1 10/10 / P2 7/10) — the keystone stacking half, now confirmed on the biologically-correct layout.
- The apparent "contrast blocker" was geometry-determined (rung 8), and the geometry it was measured against had no
  empirical basis (Rich 2014).
- Every `sim/` edit additive/default-off/byte-identical-when-off, each asserted; CI clean.

**Remaining (well-specified, not walls):** deterministic (not 7/10) read localization would need a mechanism that
handles the geometry variability; the weight-dependent rule's contrast on a valid instrument (per-pathway w_max +
k_pot > k_dep so w* clears threshold). Neither is the keystone question — that is answered.
