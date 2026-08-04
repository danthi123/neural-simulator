---
status: go
lane: gateb-v14-performance
date: 2026-08-04
type: finding
mechanism: neural-vocal-action-credit-v14-explicit-snr-pacemaker-stageA
promotion_value: engineering-stageA-only
seed-waiver: seed-free engineering comparison; no physiology, calibration, replication, or held-out seed opened
---

# V14 Performance V3 Passes The Sealed Engineering Gate

The prospective V3 matrix compared clean candidate `2c88daaac` with clean
historical control `6c9034991`. It completed all twelve observations through
two persistent, source-isolated workers. Every precondition, timing-block
telemetry check, dispersion check, and worker shutdown passed. No scientific
seed was opened. The receipt is
`research/findings/raw/v14_stageA_performance_v3_2c88daaac.json`.

<!--derived-->
The median paired host ratios were `0.9975` for default-off behavior, `0.9407`
for the active path, and `0.7493` for direct output relative to unfused output.
All passed their fixed ceilings of `1.02`, `1.25`, and `0.85`, respectively.
The largest within-observation host range was `0.0202` against the `0.10`
limit, and the largest paired-ratio host range was `0.0115` against the `0.05`
limit. The engineering performance verdict is therefore **GO**.

This result supersedes the V2 infrastructure-invalid receipt for promotion
decisions under the documented environment controls. It does not supersede
the earlier method-specific NO-GO findings: those exact candidates remain
rejected. Physiology was not measured, scientific seeds remain sealed, and
this result has no behavioral or physiology verdict. The next V14 Stage A
decision must come from its preregistered scientific validation, not from this
timing receipt.
