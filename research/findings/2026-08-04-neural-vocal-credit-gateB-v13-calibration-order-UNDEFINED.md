---
type: finding
status: complete
date: 2026-08-04
verdict: CALIBRATION_UNDEFINED_PROCESS_ORDER
mechanism: gateB-v13-tonic-output-substrate
artifacts:
  - research/findings/raw/v13_tonic_output/calibration-numpy.json
  - research/findings/raw/v13_tonic_output/calibration-cupy.json
  - research/findings/raw/v13_tonic_output/calibration-selection.json
supersedes_verdicts:
  - CALIBRATION_GO
  - REPLICATION_NO_GO
---

# V13 Stage-0 calibration order is undefined

**Verdict: CALIBRATION_UNDEFINED_PROCESS_ORDER.** The locked protocol required
calibration seed `1013` to run on NumPy first and CuPy second. Both processes
started at `2026-08-04T02:30:30`. The NumPy artifact was not written until
`02:30:31.326893`, so CuPy began before NumPy had completed and been sealed.
The merged `CALIBRATION_GO` therefore did not earn promotion.

The artifacts remain useful diagnostics: both backends independently found
`100 pA` to be the only common passing ladder point. They must not be deleted,
rerun, or treated as calibration evidence. Seed `1013` is consumed.

Replication seed `1019` was launched from the invalid selection. Its NumPy GO
and CuPy inhibitory-response NO-GO are retained as diagnostic observations,
but neither is a preregistered replication verdict. Seed `1019` is consumed.
The earlier replication finding remains in the record and is superseded only
as a promotion verdict by this process-order finding.

No held-out or performance stage ran. Seed `1021` and Stage-1 seed `1031`
remain sealed. A separate correction preregistration assigns mechanically
derived replacement calibration and replication seeds and requires a
fail-closed execution controller before another registered run.
