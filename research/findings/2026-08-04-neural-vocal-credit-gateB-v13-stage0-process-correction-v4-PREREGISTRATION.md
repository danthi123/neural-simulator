---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v4
spec: research/specs/v13_tonic_output_stage0_process_correction_v4.json
promotion_value: none
supersedes_process_only: gateB-v13-stage0-process-correction-v3
---

# V13 Stage-0 process correction v4

**Status:** locked before v4 execution; no v4 command has been emitted and no
v4 seed has been run

## Narrow reason for v4

V3 aborted before measurement because the scientific runner still required the
v2 process-spec path. The external controller correctly emitted a v3 command,
but the runner rejected it before reading a partition, constructing a brain, or
entering the calibration ladder. No v3 result, sidecar, or success receipt
exists.

V4 makes controller-runner authority agreement explicit and testable. The
runner and external controller must name the same v4 path, schema, derivation
namespace, prior partitions, forbidden seeds, retired seeds, and fresh
partitions. A direct runner test must load the production authority before any
source freeze.

The scientific mechanism and every measurement gate remain unchanged.

## Fresh partitions

Although v3 seeds `577995` and `578403` never ran, v4 retires them to keep the
attempt boundary simple. Earlier consumed and retired partitions remain
forbidden. Held-out seed `1021` and Stage-1 seed `1031` remain sealed.

For each role, form:

`V13_STAGE0_PROCESS_CORRECTION_V4|63da248655ee406e159e762ff8c865d5dd49081c|role=<role>|prior_seed=<v3 partition seed>`

Take SHA-256, read the first 12 hexadecimal digits as an integer, and calculate
`100000 + (integer mod 900000)`. Commit `63da24865` predates the failed v3
attempt, and no measured result is an input. This fixes calibration seed
`384414` and replication seed `568500`.

## Execution and stop boundary

The five-point ladder remains `75`, `100`, `125`, `150`, and `175 pA`, with a
fresh brain for every point. NumPy must execute and seal before CuPy. Only a
sealed lowest-common-pass selection may unlock replication. Only two sealed
replication GO artifacts may unlock held-out CuPy and then held-out NumPy.

Any prior-chain promotion, authority mismatch, forbidden or retired seed,
source drift, changed ladder, digest substitution, receipt mismatch, merge
environment mismatch, order violation, or scored rerun makes affected evidence
`UNDEFINED` and stops progression.

This correction carries no promotion value. Only the unchanged final Stage-0
merge can earn `TONIC_OUTPUT_GO`; nothing here establishes action learning,
speech, language, or conversation.
