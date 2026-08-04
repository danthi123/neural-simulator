---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-process-correction-v3
spec: research/specs/v13_tonic_output_stage0_process_correction_v3.json
promotion_value: none
supersedes_process_only: gateB-v13-stage0-process-correction-v2
---

# V13 Stage-0 process correction v3

**Status:** locked before v3 execution; no v3 command has been emitted and no
v3 seed has been run

## Why v2 is undefined

V2 executed its fresh five-point calibration once on NumPy and once on CuPy.
Both sealed backend artifacts found `100 pA` as the only passing point. The
seed-free selection merge also completed and observed `CALIBRATION_GO`.

The selection command envelope omitted `SIM_BACKEND`, while its execution
receipt and manifest contract required `SIM_BACKEND=numpy`. The successful
merge artifact therefore could not receive a valid selection manifest. This is
an execution-contract defect, not a negative result for the tonic-output
mechanism. V2 stopped at that boundary: no replication, held-out, performance,
or final-merge command followed.

The preserved v2 NumPy artifact is
research/findings/raw/v13_tonic_output_stage0_process_correction_v2/calibration-numpy.json,
and the preserved CuPy artifact is
research/findings/raw/v13_tonic_output_stage0_process_correction_v2/calibration-cupy.json.
They remain diagnostic only. The unsealed selection must not choose a v3
current or unlock any later stage.

## Narrow process repair

V3 changes only the failed process boundary and the partitions invalidated by
that stop:

- calibration-selection and final-merge envelopes must explicitly carry
  `SIM_BACKEND=numpy`;
- both emitters use the same environment policy as manifest validation;
- tests assert the environment in each actual emitted envelope; and
- fresh calibration and replication seeds replace the consumed or retired v2
  partitions.

The mechanism, calibration ladder, physiology thresholds, topology, lesions,
inhibitory response, checkpoint gate, compatibility evidence, performance
limits, held-out criteria, and final promotion criteria do not change.

## Fresh result-independent partitions

Calibration seed `645424` is consumed. Replication seed `638726` was never run
and is retired. Earlier consumed seeds `1013`, `1019`, and `840860`, and earlier
retired seed `687979`, remain forbidden. Held-out seed `1021` and Stage-1 seed
`1031` remain sealed.

For each role, form this exact ASCII string:

`V13_STAGE0_PROCESS_CORRECTION_V3|f41575539536809158736cd62ab42ab2292cf20c|role=<role>|prior_seed=<v2 partition seed>`

Take SHA-256, interpret the first 12 hexadecimal digits as an integer, then
calculate `100000 + (integer mod 900000)`. The source anchor was committed at
`2026-08-04T07:01:25-04:00`, before the v2 observations. No measured current,
verdict, raster, state hash, or tested candidate enters the derivation.

This fixes calibration seed `577995` and replication seed `578403`. The
machine-readable v3 spec is the sole value authority. The controller must
recompute the seeds and reject substitutions or collisions.

## Unchanged execution ladder

The calibration ladder remains exactly `75`, `100`, `125`, `150`, and `175 pA`
in ascending order, with a fresh brain at every point. No result from v1 or v2
may skip a point, prefer `100 pA`, interpolate, stop early, or carry state into
v3.

Execution remains fail-closed and ordered:

1. Run and seal all five NumPy calibration points.
2. Only then run and seal the same five CuPy points on the RTX 3090.
3. Merge the two sealed artifacts and select the lowest common passing point.
4. Only after a sealed selection, run NumPy and CuPy replication at that point.
5. Only after both sealed replication GO results, run held-out CuPy and then
   held-out NumPy.

Old-source performance and candidate performance remain serial and retain
their separate frozen source and environment packages.

## Stop and promotion boundary

Any prior-chain promotion attempt, forbidden or retired seed, changed ladder,
source drift, digest substitution, sidecar mismatch, receipt mismatch, merge
environment mismatch, order violation, or scored rerun makes the affected
evidence `UNDEFINED` and stops progression.

This process correction has no promotion value. Only a fully sealed v3
calibration pair may select a current; only sealed v3 replication GO artifacts
may unlock held-out; and only the unchanged final merge may earn
`TONIC_OUTPUT_GO`. Nothing here establishes action learning, language, speech,
or conversation.
