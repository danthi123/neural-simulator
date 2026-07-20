# gap#4 RUNG 9 — PRE-REGISTRATION: does DEEP CREDIT (L2 reads the learned map) pass its gate under HONEST geometry?

**Filed 2026-07-20 before any rung-9 result exists.** Seeds **2000-2011** (twelve, because Poisson draws yield
~1/6 degenerate maps; a larger pool keeps the usable count near 6). Per-seed Poisson field placement.

## Why this is the RIGHT next test, and why it is not p-hacking

The gap#4 keystone question is whether the substrate learns DEEP representations by a biological rule — i.e. whether
a downstream layer LEARNS to read a learned population code. **Rung 3d already answered the MECHANISM half YES**
(6/6 fresh seeds: L2's read is plateau-locked, tracks the plateau 1:1, collapses under plasticity-lesion and
plateau-lesion). What rung 3 could not pass was the GATE's selectivity leg — and rung 8 has now shown that the
contrast that leg measures was evaluated at **an arbitrary, artificially-deterministic geometry with no empirical
basis**, where it happened to sit low.

**The legitimate follow-up:** run the SAME gate under the geometry biology actually uses (Poisson; Rich 2014). This
is not moving the goalpost — the gate is unchanged. It is testing an already-mechanistically-validated capability
under a VALID instrument, after discovering the prior instrument used an unmotivated geometry.

**Anti-p-hacking commitments, all pre-registered:**
- The geometry choice was justified by the literature agent BEFORE rung 8's result existed, not selected post-hoc.
- I report EVERY usable seed, and I do NOT cherry-pick the favourable draw (seed 1802's 1.902 does not enter here;
  these are fresh seeds 2000-2011).
- The gate is the rung-3 gate verbatim: `read_acc` (L2 peak within +/-2 of the plateau-derived expected bin) plus
  the adjacent-contrast, both reported.

## PRE-REGISTERED GATE + PREDICTIONS

Deep-credit read is the `MAIN` arm (plain BTSP forms the map; BTSP on ca1->l2 IS the deep credit under test).

1. **P1 — the read is learned (mechanism, must hold):** `C1_frozen` (L2 eta=0) collapses the response >= 5x vs MAIN,
   AND `C3_moat` (no L2 plateau) collapses it, on >= 5/6 usable seeds. *(If this fails the read is not learned and
   nothing else matters.)*
2. **P2 — read accuracy:** MAIN `read_hit = 1` (L2 peak within +/-2 of the plateau-derived expected bin) on
   >= 4/6 usable seeds.
3. **P3 — GEOMETRY-GATED contrast:** on the seeds whose geometry places the target's nearest neighbour at
   >= 3 bins (the biologically-common non-adjacent case), adjacent contrast >= 1.60x on >= half of them.

**INTERPRETATION FIXED IN ADVANCE:**
- **P1 + P2 pass** ⇒ **deep credit across a layer WORKS under honest geometry** — the keystone's stacking half is
  demonstrated, with the honest scope that read-accuracy, not 2x-selectivity-at-every-geometry, is the claim.
- **P1 passes, P2 fails** ⇒ the read is learned but does not reliably localize even under realistic geometry — a
  genuine residual, not an artifact.
- **P1 fails** ⇒ retract rung 3d's mechanism claim; the read was never learned.

## Honest scope, in advance

- "usable seed" = `map_ok = 1` (stage 1 formed a distinct map). Degenerate draws are reported as excluded, counted,
  never silently dropped.
- This tests deep credit's READ-ACCURACY under honest geometry. It does NOT claim the 2x-selectivity gate passes at
  every geometry — rung 8 showed that specific number is geometry-determined, so requiring it everywhere would be
  requiring a favourable draw.
