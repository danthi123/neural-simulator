---
type: finding
status: partial-calibration-go-replication-aborted
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v4
promotion_value: none
seed-waiver: preregistered single calibration partition; this GO selects a replication current and is not a capability claim
instrument: matched five-point NumPy/CuPy physiology ladder with fresh brains, exact source binding, and sealed per-backend controls
---

# V13 Stage-0 process correction v4 stops after calibration GO

V4 completed and sealed its full calibration chain on NumPy and the RTX 3090.
Both backends found `100 pA` as the only passing point. The corrected seed-free
merge carried `SIM_BACKEND=numpy`, received a matching execution receipt and
manifest, and earned `CALIBRATION_GO` with no undefined reasons.

The sealed selection artifact is
research/findings/raw/v13_tonic_output_stage0_process_correction_v4/calibration-selection.json.
Its artifact SHA-256 is
`fbf2bfb70a24a0839e54a0f800fd9cc0b8c2fb23c48d58dc749ed6d8de75609d`,
and its manifest file SHA-256 is
`16b8cd7cc013468492e6c3519be231bc6a330d8fb561d1e08ec54833c0f73a1d`.

## Replication boundary

The registered NumPy and CuPy replication commands were launched concurrently.
Both runners stopped in the selection loader before constructing a replication
brain or executing seed `568500`. The loader rechecked deterministic
compatibility without passing the already-validated process-correction context,
so a historical-source guard rejected later changes to `sim/bridge.py` and
`sim/kernels.py`. Calibration had correctly supplied that context.

No replication artifact, provenance sidecar, or success receipt exists on
either backend. No held-out, performance, or final-merge command was emitted.

## Disposition

- Calibration seed `384414` is consumed.
- Replication seed `568500` was never run and is retired.
- The sealed calibration observations remain diagnostic evidence only; they
  cannot unlock a changed source revision.
- Held-out seed `1021` and Stage-1 seed `1031` remain sealed.
- V5 must load the process correction before selection validation, pass that
  context into compatibility validation, and test the production selection
  loader before source freeze.

This is a downstream orchestration defect, not a negative result for the
tonic-output mechanism. V4 has no Stage-0 promotion value.
