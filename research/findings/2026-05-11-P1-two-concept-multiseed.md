---
type: finding
status: qualified
date: 2026-05-11
---

# P1 two-concept discrimination multi-seed result
**Date:** 2026-05-11
**Phase:** P1+P2 combined integration test
**Seeds:** [42, 43, 44]
**Verdict:** 3/3 biology-faithful PASS, 0/3 strict PASS

## Per-seed results

| Seed | Tag AB cos (sep) | A: cos_aa / cos_ab / margin | B: cos_bb / cos_ba / margin | Bio | Strict |
|---|---|---|---|---|---|
| 42 | 0.120 | 0.446 / 0.074 / 0.371 | 0.439 / 0.101 / 0.338 | PASS | FAIL |
| 43 | 0.105 | 0.326 / 0.111 / 0.215 | 0.479 / 0.047 / 0.432 | PASS | FAIL |
| 44 | 0.000 | 0.460 / 0.066 / 0.394 | 0.449 / 0.064 / 0.385 | PASS | FAIL |

## Multi-seed averages

- Tag AB overlap (lower better, target < 0.3): **0.075**
- Same-concept cosine A→A: 0.411; B→B: 0.456
- Cross-concept cosine A→B: 0.084; B→A: 0.071
- Discrimination margin A: 0.327; B: 0.385

## Biology-faithful criterion (Marr 1971, catalog D.13)

Test: cross-concept cosine < 0.3 AND margin > 0.2.
Pass means: stored attractor converges to ITS OWN pattern, not a different concept's.

**3/3 seeds PASS biology-faithful**

## Strict criterion (engineering-ideal)

Test: same-concept cosine > 0.5 AND cross < 0.3 AND margin > 0.2.
Pass means: ideal pattern completion (re-activates >50% of original ensemble).

**0/3 seeds PASS strict**

## Interpretation

The P1+P2 substrate reliably **distinguishes concepts** across seeds. The architecture is sufficient for the user's 'concepts as tagged hippocampal ensembles' goal. Downstream consolidation (P3 → semantic_cortex P5) will have clear signal to learn from.

The strict criterion (same > 0.5) is not robustly met. The autoassociator reactivates ~45% of the original ensemble rather than the ideal >50%. This is fine for downstream STDP-based consolidation, but worth noting if future work wants tighter completion (e.g. for Tonegawa-style optogenetic-recall reproduction, where perfect reactivation matters more).
