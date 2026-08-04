---
type: finding
status: no-go
date: 2026-08-04
mechanism: gateB-v13-stage0-performance-confirmation-v8
promotion_value: none
seed-waiver: process-only confirmation; no physiology or learning seed was run
instrument: RTX 3090 CuPy six-cell performance matrix with complete receipt
---

# V13 Stage-0 v8 fixes v2 overhead but remains overall NO-GO

V8 reused the five sealed V6 physiology inputs and the byte-identical V7
historical baseline. It did not rerun a physiology seed and did not read or
execute Stage-1 seed `1031`. The process-only correction passes the static
intrinsic-current array directly into the existing v2 RawKernel and adds it in
the kernel, removing the per-step host-side CuPy addition allocation.

The candidate ran once from the exact V10 source revision on the RTX 3090. All
six cells completed three repetitions, the receipt is complete, and the
provenance sidecar is present. The targeted v2 active/default ratio fell from
the V7 value `1.150378` in
`research/findings/raw/v13_tonic_output_stage0_performance_continuation_v7/candidate-performance/performance-candidate.json`
to `1.006836`, passing the unchanged `1.10` limit.
Normal active overhead (`1.003357`), v1 active overhead (`1.022437`), feature
storage, default allocation, and dispatch checks also passed.

The overall Stage-0 performance gate still fails because the normal default
path measured `6.130906 s`, or `1.059092` times the sealed old baseline
`5.788830 s`; the fixed limit is `1.02`. This is an engineering performance
NO-GO, not a reversal of the sealed V6 physiology result. The formal V8
outcome is `TONIC_OUTPUT_NO_GO`, with `candidate_receipt_complete=true` and
`promotion_eligible=false`.

The candidate artifact is
`research/findings/raw/v13_tonic_output_stage0_performance_confirmation_v8/candidate-performance/performance-candidate.json`
with SHA-256
`5b79f241e1ded1c241f2a66269944f46be012dbd449ee8baaa48052bab1c6c3d`.
The final artifact is
`research/findings/raw/v13_tonic_output_stage0_performance_confirmation_v8/final-stage0-v8.json`
with canonical SHA-256
`4eaa2a8a597f11d3d8c7c3ca74cc5829fdbac5be13d324b0dbd07f1e0e2e6e82`.

## Boundary and next action

The v2 correction is retained because its local performance check passed, but
it does not promote Stage 0. Do not rerun the same V8 measurement. The next
performance action must be a new preregistered investigation of the normal
default-versus-historical regression, with a declared explanation or a
revised comparison boundary. Stage-1 remains sealed.
