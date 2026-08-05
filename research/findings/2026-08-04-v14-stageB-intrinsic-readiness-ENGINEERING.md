---
type: finding
status: qualified
date: 2026-08-04
mechanism: neural-vocal-action-credit-v14-stageB-intrinsic-readiness
---

# V14 Stage B intrinsic-lesion readiness transport

## Result

The authenticated experiment path now executes one intact arm and four complete intrinsic-current lesions for a pinned candidate, binds all five raw traces into an automatic scorer, and emits a provenance-complete receipt. Both filed endpoint candidates completed this engineering path.

This is not a physiology result. The scorer returned `UNAVAILABLE` and `scientific_verdict: null` because production analysis and cohort protocols remain intentionally unimplemented.

## Evidence

- Low endpoint: `research/findings/raw/v14_stageB_intrinsic_readiness_20260804/readiness-receipt.json`, source revision `e194fc29c7a702c6d3cdc842dd4eb97962891717`
- High endpoint: `research/findings/raw/v14_stageB_intrinsic_readiness_high_20260804/readiness-receipt.json`, source revision `a1892dee2e22749fcd3cb88e0e12e6ac9112557b`
- Five completed 20-sample traces per endpoint: intact, Nap lesion, Cav2.2 lesion, SK lesion, and baseline HCN lesion
- High-endpoint conductance transitions: Nap `0.25 -> 0`, Cav2.2 `2.0 -> 0`, SK approximately `0.05 -> 0`, and HCN approximately `0.05 -> 0` mS/cm2
- Scientific seed: none
- Scientific scoring performed: false
- Missing provenance sidecars: none

The low endpoint starts with HCN conductance at zero, so its HCN arm records `0 -> 0`; the high endpoint independently demonstrates the nonzero HCN intervention.

## Remaining blocker

The production runner still emits a short transport trace without a sealed analysis protocol. The source-bound next implementation is event-count-driven: Atherton and Bevan calculated firing characteristics from trains of 101 spontaneous action potentials. Medium-AHP measurement, HCN hyperpolarized input resistance, and the 12-cell SK depolarization-block cohort remain under-specified or unimplemented and must fail closed until their protocols are filed.
