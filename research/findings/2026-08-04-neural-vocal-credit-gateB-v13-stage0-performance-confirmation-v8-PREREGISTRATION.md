---
type: preregistration
status: preregistered-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-performance-confirmation-v8
promotion_value: process-only
---

# V13 Stage-0 performance confirmation v8

V8 confirms one process-only performance correction after V7 measured that the
general-step v2 path was too slow when a region-scoped intrinsic current was
present. The correction passes the static intrinsic-current buffer directly to
the existing RawKernel and performs the same current addition in-kernel. It
does not change the neuron model, wiring, plasticity, inputs, seeds, or gate
thresholds.

The sealed V7 legacy baseline is reused byte-for-byte. V8 does not rerun the
historical source and does not read or execute Stage-1 seed `1031`. The
candidate must run from the exact V8 candidate revision under
`v13_stage0_candidate_source_v10.sha256` on the RTX 3090 with the existing
V6 inputs. The candidate artifact is deliberately placed inside the raw
evidence tree so the execution receipt and provenance sidecar are captured;
finalization refuses any incomplete receipt transfer.

The thresholds remain unchanged: default-off versus the old baseline must be
at most `1.02`; normal, v1, and v2 active/default ratios must each be at most
`1.10`; storage, default allocation, and v1/v2 dispatch checks must pass. A
complete measured `PERFORMANCE_GO` earns `TONIC_OUTPUT_GO`; a complete measured
`PERFORMANCE_NO_GO` is retained as honest negative evidence. No physiology
promotion follows from this process-only confirmation by itself.
