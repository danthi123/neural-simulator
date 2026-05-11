# P4.1 positional binding multi-seed result

**Date:** 2026-05-11
**Phase:** P4.1 (item-in-context binding) of realigned plan v3
**Catalog:** D.01 + D.02 + D.11
**Seeds:** [42, 43, 44]
**Verdict:** 3/3 PASS

## Per-seed results

| Seed | apple-pos | alice-pos | word@pos0 | word@pos2 | cross | Overall |
|---|---|---|---|---|---|---|
| 42 | 0.097 | 0.000 | 0.100 | 0.054 | 0.045 | PASS |
| 43 | 0.041 | 0.062 | 0.000 | 0.053 | 0.065 | PASS |
| 44 | 0.134 | 0.045 | 0.000 | 0.045 | 0.100 | PASS |

All cosines should be < 0.4. PASS means architecture distinguishes the (word, position) tuples.

## Multi-seed averages

- Same word, different position: apple=0.091, alice=0.035
- Different word, same position: @pos0=0.033, @pos2=0.051
- Different word, different position (cross): 0.070

## Interpretation

**P4.1 substrate confirmed at multi-seed.** The architecture cleanly distinguishes (word, position) tuples — same word at different positions get distinct CA3 ensembles, and different words at the same position also get distinct ensembles. Word-order-dependent meaning is mechanistically supported.

Downstream impact: P5 ventral semantic stream + P6 Broca's can now learn to distinguish 'alice ate apple' from 'apple ate alice' via their distinct (word, position) CA3 ensemble sequences.
