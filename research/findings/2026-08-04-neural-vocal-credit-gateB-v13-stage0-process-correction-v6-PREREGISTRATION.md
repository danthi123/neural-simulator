---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v6
spec: research/specs/v13_tonic_output_stage0_process_correction_v6.json
promotion_value: none
supersedes_process_only: gateB-v13-stage0-process-correction-v5
---

# V13 Stage-0 process correction v6

**Status:** locked before V6 execution; no V6 command has been emitted and no
V6 seed has been run

V5 completed the full calibration and both replication measurements. NumPy
earned a sealed replication GO. CuPy produced a complete scientific no-go, but
the runner used exit code `1` for that valid negative outcome. The receipt
wrapper consequently refused a success receipt, leaving the negative artifact
unsealable and the chain undefined.

V6 changes only this process boundary. A complete artifact with an earned `GO`
or `NO-GO` verdict exits `0` and can receive a success receipt and immutable
manifest. An `UNDEFINED` verdict or execution exception remains nonzero and
cannot receive a success receipt. Regression tests bind all three cases.

V5 calibration seed `216274` and replication seed `401461` are consumed. Fresh
values are derived from namespace V6, commit `0c8d60e55`, which predates every
V5 observation, each role, and its corresponding V5 seed. No result enters the
derivation. The fixed V6 seeds are calibration `754571` and replication
`890220`. Held-out seed `1021` and Stage-1 seed `1031` remain sealed.

The mechanism, five-point ladder, thresholds, topology, lesions, inhibitory
response, checkpoint gate, backend order, manifest rules, and final promotion
criteria remain unchanged. V5 results cannot unlock V6 or prefer `100 pA`.

Execution remains NumPy calibration and seal, CuPy calibration and seal,
seed-free lowest-common-pass merge, and both replication cells. Held-out work
is permitted only if both replication cells produce sealed GO evidence. A
sealed replication no-go stops the mechanism as a formal negative result.

This correction has no promotion value. Only the unchanged final Stage-0 merge
may earn `TONIC_OUTPUT_GO`.
