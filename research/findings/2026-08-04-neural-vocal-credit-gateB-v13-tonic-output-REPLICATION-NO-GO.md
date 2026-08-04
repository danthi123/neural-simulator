---
type: finding
status: complete
date: 2026-08-04
verdict: REPLICATION_NO_GO
mechanism: gateB-v13-tonic-output-substrate
artifacts:
  - research/findings/raw/v13_tonic_output/replication-numpy.json
  - research/findings/raw/v13_tonic_output/replication-cupy.json
---

# V13 Stage-0 replication earns NO-GO

**Verdict: REPLICATION_NO_GO.** Calibration selected the locked `100 pA`
intrinsic current. At replication seed `1019`, NumPy earned `REPLICATION_GO`
and passed every gate.

CuPy failed only `inhibition_at_most_ten_percent`. Its source-on GPi/SNr rate
fell from a `63.15 Hz` baseline to `11.125 Hz` during inhibition, a ratio of
`0.1761678543`, above the preregistered maximum of `0.10`. CuPy passed the
population audit, intact physiology, intrinsic-drive lesion, checkpoint
continuation, and every other inhibitory-response check. The lesion emitted
zero spikes with zero external current, and resumed checkpoint state matched
uninterrupted continuation exactly.

The cross-backend Stage-0 replication gate therefore does not pass. Stop before
held-out seed `1021` and the performance gate. Do not retune the current,
inhibitory pathway, model, thresholds, or protocol without a new preregistration.
