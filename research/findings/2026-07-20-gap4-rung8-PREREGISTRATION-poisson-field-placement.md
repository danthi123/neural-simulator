# gap#4 RUNG 8 — PRE-REGISTRATION: is the deficit an artifact of EVENLY-SPACED fields? (filed BEFORE the run)

**Filed 2026-07-20 before any rung-8 result exists.** Seeds **1800-1805**, never used.

## Why this test, and why it questions the TASK rather than the rule

Eight mechanism attempts have failed to raise adjacent-field contrast. The literature reframe identified something
I had never questioned: **this task's evenly-spaced field layout has no empirical basis.** Rich, Liaw & Lee 2014
(*Science* 345:814) measured CA1 field locations as a spatial **Poisson process** — uniform locations, exponential
interfield intervals, uncorrelated across cells, with **0/61 cells deviating**. The modal gap between neighbouring
fields is therefore **ZERO**, and there is no characteristic spacing at all.

Evenly-spaced-by-4-bins is a modelling convenience this project inherited from the BTSP theory literature (Front.
Comput. Neurosci. 2021 adopts equal spacing explicitly for tractability). **It may be generating the deficit.**

## The test runs with the rule OFF

This deliberately uses **plain BTSP** — the arm that is valid, well-characterized, and has reproduced
`c_adj = 1.213 / c_far = 2.609` identically on **every seed across five independent runs**. That determinism is
what makes the test sharp: under even spacing the geometry is fixed, so `c_adj` is constant. Under Poisson
placement the geometry varies per seed, so **if geometry drives the deficit, `c_adj` must vary too.**

*(Implementation note: the layout is re-drawn per seed. Drawing once would give every seed the same geometry and
render the test vacuous — a defect I caught and fixed before filing this.)*

## PRE-REGISTERED PREDICTIONS

1. **P1 — geometry-dependence:** under Poisson placement `c_adj` **VARIES across seeds** (std > 0.05), against the
   even-spacing case where it is identical to 3 decimals on every seed.
2. **P2 — some geometries clear the bar:** at least 1/6 seeds gives `c_adj >= 1.60x`.
3. **P3 — measurability retained:** `map_ok = 1` on >= 4/6 (Poisson draws can place fields close together; the
   `min_gap = 2` concession is stated in the code and is for measurability, not biology).

**INTERPRETATION, fixed in advance:**
- **P1 and P2 both pass** ⇒ the deficit is substantially a **geometry artifact** of even spacing, and eight
  mechanisms were chasing a problem the task created.
- **P1 passes, P2 fails** ⇒ geometry modulates the deficit but does not remove it; the deficit is real but its
  magnitude is task-dependent.
- **P1 fails** (c_adj constant despite varying geometry) ⇒ the deficit is **intrinsic to the rule**, independent of
  layout, and the eight failures were against a genuine property rather than an artifact.

All three outcomes are informative. I am recording the mapping now so the result cannot be read selectively.

## Honest scope

- `min_gap = 2` departs from a true Poisson draw (which permits coincident fields). Stated in the code and here.
- This tests the BASELINE deficit's geometry-dependence, not any mechanism's contrast performance.
