---
type: finding
status: preregistered
date: 2026-08-04
mechanism: gateB-v13-stage0-performance-diagnostic-v9
promotion_value: none
seed-waiver: process-only; no physiology, learning, or Stage-1 seed may be consumed
instrument: controlled RTX 3090 CuPy comparison with V7 source control
---

# V13 Stage-0 v9 performance diagnostic preregistration

The sealed V8 process result is an engineering `NO_GO`: its normal-default
median was `6.130905688 s`, or `1.059092` times the immutable historical
baseline `5.788830481 s`, as recorded in
`research/findings/raw/v13_tonic_output_stage0_performance_confirmation_v8/candidate-performance/performance-candidate.json`.
V8 also passed its targeted v2 active-path correction.
This diagnostic investigates why the ordinary path differs from the historical
comparison. It does not rerun the six-cell V8 matrix, alter the `1.02` limit,
or replace the V8 result with a more favorable measurement.

## Question

Is the V8 normal-default difference better explained by environment/runtime
state, process ordering and CUDA-cache state, or an indirect difference between
the V7 and V8 source packages?

## Locked comparison

The candidate is the current source lineage descended from
`1ecc85cd698539a6ef92e112d2c49092cfa21f1e`; the control is the frozen V7
candidate at `1bec3c22ad7c535a2cbb27860e5bf4cfd51d6d6f`. The same tonic-output
runner is used in both packages. Only source and cache condition vary.

Each of the four cells is run three times, one process per cell repetition, in
a deterministic shuffled order derived from `20260804`:

| Cell | Source | Process condition |
| --- | --- | --- |
| A | V8 candidate | ordinary path in a fresh process |
| B | V7 control | ordinary path in a fresh process |
| C | V8 candidate | compile/warm the v2 path, then ordinary path |
| D | V7 control | compile/warm the v2 path, then ordinary path |

Every worker performs 500 untimed warm-up steps and 20,000 timed ordinary-path
steps. The timed steps are divided into four 5,000-step chunks for CUDA-event
sampling. There is no concurrent GPU work and the local model service must be
stopped. The shared GPU lease and the nonempty GPU queue are hard refusal
conditions.

## Structural boundaries

The worker must prove that it is using the ordinary path: v1 and v2 flags are
off, the default intrinsic-current array is `None`, external current is exactly
zero, and no learning, reward, physiology, or Stage-1 configuration is loaded.
No state reset is permitted inside the timed interval. A separate short trace
counts normal-step calls and megakernel dispatch decisions; its timing is not
used as the performance measurement.

## Recorded evidence

For each repetition, record the source file digests, source revision, Python,
CuPy, CUDA runtime and driver metadata, GPU identity, host wall time, CUDA event
time, host-minus-device time, CuPy pool usage, and GPU temperature, clock,
power, utilization, and memory before and after timing. Record individual times,
not only medians.

## Interpretation boundary

The result can identify a likely performance cause or show that the comparison
remains unresolved. It cannot turn the sealed V8 `NO_GO` into a `GO`, open V13
Stage 1, consume seed `1031`, or change any biological mechanism. Any source
correction requires a new implementation review and a new preregistration.

The executable contract is
`research/specs/v13_stage0_performance_diagnostic_v9.json`, and the runner is
`tools/v13_stage0_performance_diagnostic_v9.py`.
