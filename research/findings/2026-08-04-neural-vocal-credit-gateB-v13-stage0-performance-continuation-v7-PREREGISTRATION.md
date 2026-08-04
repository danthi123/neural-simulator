---
type: preregistration
status: preregistered-not-executed
date: 2026-08-04
mechanism: gateB-v13-stage0-performance-continuation-v7
promotion_value: none
---

# V13 Stage-0 performance continuation v7

V7 is a process-only continuation of the sealed V6 physiology result. It does
not select or run a scientific seed and does not change the neural mechanism.
It exists because the V6 configuration paired a historical Git revision with a
runner that is absent from that revision, so the old performance command could
not be emitted.

The historical baseline must use the audited legacy package: scientific imports
come from revision `8994b5102b39a8a6aa6abdeb9fde02579b7db6a8`, with only the
accepted V13 measurement runner overlaid from revision
`d091fa6692bdf8115c8073af6fd31fc9626921a8`. A package-owned receipt must bind
the exact command, environment controls, source package, RTX 3090 identity,
timing, and output bytes.

The candidate measurement must run from exact revision
`1bec3c22ad7c535a2cbb27860e5bf4cfd51d6d6f` under source manifest
`v13_stage0_candidate_source_v9.sha256`. Its receipt and provenance sidecar must
be verified in that checkout before create-only transfer.

The thresholds remain unchanged: default-off versus old must be at most `1.02`;
normal, v1, and v2 active/default ratios must each be at most `1.10`; storage,
default allocation, and v1/v2 megakernel checks must all pass. A measured
`PERFORMANCE_NO_GO` is valid evidence and must be sealed as `TONIC_OUTPUT_NO_GO`,
not converted into a process failure. Stage-0 earns GO only when all five sealed
V6 physiology inputs and performance earn GO. Stage-1 seed `1031` remains sealed.
