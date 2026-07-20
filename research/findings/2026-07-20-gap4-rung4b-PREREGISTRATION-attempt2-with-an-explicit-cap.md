# gap#4 RUNG 4b — PRE-REGISTRATION (attempt 2 of a capped 2), filed BEFORE the run

**Filed 2026-07-20, before any rung-4b result exists.** Attempt 1 (`356223d0`) was pre-registered, FAILED, and the
failure is recorded in full at `3edb9144`. This is the corrected derivation.

## ⚠️ THE CAP — stated first, because it is the part that protects this from becoming tuning

A mechanism with two free numbers, re-derived after seeing it fail, is one step away from being fitted to its own
outcome. So:

**If this second pre-registered band also fails, I do NOT derive a third.** The verdict becomes: *the adjacent-band
depression MECHANISM (not merely a placement) is in question*, and the next step is a research gate on the
mechanism, not another set of thresholds. Two derivations is the cap; I am recording it before the run so it cannot
be quietly extended to three.

## What changed, and why it addresses the DIAGNOSED cause

Attempt 1 failed for a specific measured reason: its lower edge (0.006958) sat **below the median eligibility**
(0.007665), so it depressed the bulk of `pos->ca1`, CA1 never reached threshold, and stage 1 never formed.
The error was that I located the adjacent **lag** correctly but never checked how much synaptic **mass** sat there.

The correction anchors the lower edge to the distribution instead of to the far lag:

| quantity | value | source |
|---|---|---|
| measured median eligibility | 0.007665 | 3995-step measurement |
| measured max | 0.022681 | same |
| adjacent-lag eligibility = `exp(-800/1000) * max` | 0.010191 | rule + geometry |
| **band_lo = sqrt(median * adjacent)** | **0.008838** | **CHANGED — anchored to the mass floor** |
| **band_hi = sqrt(adjacent * max)** | **0.015204** | unchanged rationale (protect the peak) |

Checks attempt 1 failed, now satisfied by construction:
- `band_lo (0.008838) > median (0.007665)` → the bulk is spared;
- `band_hi (0.015204) < max (0.022681)` → the peak is still protected;
- `adjacent (0.010191)` lies inside the band.

Note the enabling fact: the adjacent-lag eligibility (0.010191) **is** above the median (0.007665), so an
adjacent-selective band that spares the bulk **can** exist. Had it not been, the mechanism would have been
falsified outright rather than re-derived — and that check is why this is a correction rather than a fudge.

## PRE-REGISTERED PREDICTIONS (unchanged from attempt 1 except P0)

0. **P0 — stage 1 survives:** `map_ok = 1` on >= 5/6 seeds. *(New: this is the specific thing attempt 1 broke.)*
1. **P1 — adjacent contrast rises:** response contrast vs the ADJACENT field goes from 1.213 to **>= 1.60x** on >= 5/6.
2. **P2 — far contrast not sacrificed:** contrast vs the FAR field stays **>= 2.0x** on >= 5/6.
3. **P3 — the trough moves:** the weight-map minimum deviation occurs at a cell **1 field** from the peak, not 2, on >= 5/6.
4. **P4 — band is load-bearing:** band-OFF reproduces 1.213 / 2.609, 6/6.

**FALSIFIED if** P0 fails again (the band cannot be placed without breaking field formation), or P1 fails (the
adjacent-band hypothesis is wrong), or P2 fails (contrast is only redistributed, not added).

## The bar, restated from the compression measurement

Weight contrast 1.73x yields only 1.09-1.21x response contrast — the transfer eats ~1.5x. So P1's 1.60x response
target implies **>= 2.5x adjacent weight contrast**. Merely doubling weight contrast will not clear this.

## Seeds

**306-311** — never used. (42/43/44, 100-102, 200-205, 300 are all contaminated.)
