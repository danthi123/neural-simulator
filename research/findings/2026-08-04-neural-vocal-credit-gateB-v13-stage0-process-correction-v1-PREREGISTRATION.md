---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v1
spec: research/specs/v13_tonic_output_stage0_process_correction_v1.json
promotion_value: none
---

# V13 Stage-0 process correction v1

**Status:** process correction only; no simulation, scored step, or seed was run
while preparing it

## Reason for correction

The original calibration seed `1013` and replication seed `1019` have already
been consumed. Their archived observations may inform the record, but they
cannot be rerun or carry promotion weight. Held-out seed `1021` remains sealed,
and Stage-1 seed `1031` remains unavailable.

This correction replaces the consumed Stage-0 partitions and hardens their
orchestration. It does not change the V13 tonic-output mechanism, physiology
limits, lesions, checkpoint gate, compatibility gate, performance limits, or
scientific promotion criteria established by
`research/findings/2026-08-04-neural-vocal-credit-gateB-v13-tonic-output-substrate-PREREGISTRATION.md`
and `research/specs/v13_tonic_output_substrate.json`.

## Locked replacement method

The calibration and replication replacements are fixed by the deterministic
SHA-256 derivation recorded in
`research/specs/v13_tonic_output_stage0_process_correction_v1.json`. The
derivation binds a role, its consumed seed, a fixed namespace, and the recorded
source revision before any candidate is tested. The controller and runner must
recompute the result and reject an arbitrary substitution. Replacement values
are intentionally not repeated in this human-readable finding; the
machine-readable correction spec is the sole value authority.

The calibration ladder remains exactly `75`, `100`, `125`, `150`, and `175 pA`.
All five points must be measured from fresh brains, without interpolation,
extra points, threshold changes, or carrying the earlier diagnostic choice
forward.

## Prerequisites and source binding

Stage-0 command emission is forbidden until the strict matched-state arithmetic
replay v2 has a validated `DIAGNOSTIC_PASS`. Its result is recorded in
`research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-izh-arithmetic-replay-v2-DIAGNOSTIC-RESULT.md`,
with the receipted evidence chain rooted at
`research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/evidence-manifest.json`.
The controller must validate that chain and require digest equality for the
replay-sensitive candidate files; citing the outcome alone is insufficient.

Before execution, the correction spec, runner, controller, manifest validator,
tests, and all transitively imported scientific source must be committed and
frozen in one complete candidate source manifest. Its revision, manifest
digest, tree digest, file count, and per-file digests become execution inputs.
Every candidate receipt and artifact manifest must bind that exact source
snapshot. A partial manifest, changed file, mismatched revision, different
source tree, malformed receipt, or pre-existing output blocks the stage.

The process is represented by the pending changes to
`tools/v13_stage0_controller.py`, `tools/v13_stage0_manifest.py`, and
`research/runners/_vocal_action_credit_gate_v13_tonic_output.py`. Those changes
are not execution authority until reviewed, tested, committed, and included in
the frozen candidate source manifest.

## Exact execution order

The controller must enforce this order without bypass or rerun:

1. Run and validate calibration on NumPy.
2. Only after sealing the NumPy artifact and receipt, run and validate the same
   five-point calibration on CuPy.
3. Merge the two validated calibration artifacts and select the lowest common
   passing ladder point. If none exists, stop with `CALIBRATION_NO_GO`.
4. Run replication on NumPy and CuPy at only the selected point. These two
   backend cells may run concurrently after the selection is sealed. Any
   replication failure stops Stage 0.
5. After both replication GO artifacts are sealed, run and validate held-out
   CuPy, then held-out NumPy, preserving the original held-out backend order.

Thus NumPy-before-CuPy is exact for replacement calibration, replication is a
parallel backend pair, and held-out evaluation remains deliberately
CuPy-before-NumPy. Held-out continues to use its still-sealed original seed.

## Promotion boundaries

This preregistration and its process checks have no scientific promotion value.
Calibration may promote only a mechanically selected common ladder point into
replication. Two earned replication GO artifacts may unlock held-out execution.
Held-out results cannot repair a failed calibration or replication, and no
backend result may substitute for its twin.

Only the original Stage-0 final merge may earn `TONIC_OUTPUT_GO`, after all
required calibration, replication, held-out, lesion, inhibitory-response,
checkpoint, compatibility, provenance, topology, and performance gates pass.
Only that earned verdict may open the separately preregistered Stage-1 selector
work. This correction does not claim tonic-output success, action selection,
learning, speech, or conversation.

## Stop rules and current state

Any failed prerequisite, source mismatch, receipt mismatch, consumed-seed use,
order violation, changed ladder, unsealed dependency, or scored rerun makes the
affected evidence invalid and stops progression. The readiness path may inspect
prerequisites but must not emit a command, reveal replacement values, create an
artifact, or execute scientific code.

No Stage-0 correction command has been emitted and no replacement or held-out
seed has been executed under this preregistration.
