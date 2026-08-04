---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v5
spec: research/specs/v13_tonic_output_stage0_process_correction_v5.json
promotion_value: none
supersedes_process_only: gateB-v13-stage0-process-correction-v4
---

# V13 Stage-0 process correction v5

**Status:** locked before v5 execution; no v5 command has been emitted and no
v5 seed has been run

V4 earned a fully sealed cross-backend calibration GO at `100 pA`, then both
replication commands aborted before brain construction. The selection loader
revalidated historical compatibility without the process-correction context
that calibration had already validated. This activated a stale-source guard for
later simulator changes.

V5 changes that downstream control path only. Replication now loads the process
correction first and passes it through selection and compatibility validation.
A production-loader test builds a valid cross-backend selection and requires
the real loader to accept it before source freeze. Runner/controller authority,
merge environments, evidence manifests, and receipt checks remain explicit.

Calibration seed `384414` is consumed. Replication seed `568500` never ran and
is retired. Fresh values are derived from namespace V5, commit `129b348db`
which predates the v4 observations, each role, and its v4 partition seed. No
result enters the derivation. The fixed v5 seeds are calibration `216274` and
replication `401461`. Held-out seed `1021` and Stage-1 seed `1031` remain sealed.

The mechanism, five-point ladder, thresholds, topology, lesions, inhibitory
response, checkpoint gate, performance criteria, and final promotion criteria
remain unchanged. Prior calibration evidence cannot unlock v5 or prefer
`100 pA`.

Execution remains NumPy calibration and seal, CuPy calibration and seal,
seed-free lowest-common-pass merge, two replication cells, then held-out CuPy
and NumPy only after both replication GO results. Any authority, source,
selection context, digest, receipt, environment, ordering, or rerun violation
stops progression as `UNDEFINED`.

This correction has no promotion value. Only the unchanged final Stage-0 merge
may earn `TONIC_OUTPUT_GO`.
