---
type: finding
status: undefined-receipt-failed
date: 2026-08-04
mechanism: gateB-v13-stage0-performance-continuation-v7
promotion_value: none
seed-waiver: process-only continuation; no physiology or learning seed was run
instrument: RTX 3090 CuPy historical baseline and six-cell candidate performance matrix
---

# V13 Stage-0 v7 misses two performance limits

V7 reused the five sealed V6 physiology inputs without rerunning a seed. The
audited historical package completed its RTX 3090 baseline at a median `5.7888 s`
for 20,000 normal steps. Its package-specific receipt binds the exact old source,
accepted measurement overlay, command, machine, timing, and artifact.

The exact V6 candidate then completed all 18 registered measurements. It passed
feature storage, default allocation, normal active overhead, v1 active overhead,
and v1/v2 megakernel dispatch. It missed two limits:

- Normal default took `5.9163 s`, or `1.022016` times the old baseline. The limit
  was `1.02`.
- V2 active took `0.9409 s` versus `0.8179 s` default, a ratio of `1.150378`.
  The limit was `1.10`.

The measured artifact reports `PERFORMANCE_NO_GO`. This is an engineering
failure, not a reversal of the V6 physiology result. Stage 0 has no promotion.

## Receipt boundary

The candidate receipt failed after the command completed because V7 placed the
artifact in an isolated directory outside the runner's provenance scanner. No
sidecar was written. The preserved run-start record still binds the exact Git
revision, source-manifest digest, backend, run ID, start time, and artifact path.
A frozen one-way recovery accepted only the already measured negative artifact,
recomputed its ratios and checks, and prohibited a rerun or positive verdict.

The formal final boundary is therefore `TONIC_OUTPUT_UNDEFINED`, while the
preserved measurement remains `PERFORMANCE_NO_GO`. It records
`candidate_receipt_complete=false` and `promotion_eligible=false`. This is not a
fully receipted positive gate and cannot unlock Stage 1. Seed `1031` remains
sealed.

The historical artifact is
`research/findings/raw/v13_tonic_output_stage0_performance_continuation_v7/legacy-baseline/legacy_performance_baseline.json`
with SHA-256 `869cde4dcb978fc9eafd2799358caf94654405506b6ad5d3fd840ce9e1b602d1`.
The candidate artifact is
`research/findings/raw/v13_tonic_output_stage0_performance_continuation_v7/candidate-performance/performance-candidate.json`
with SHA-256 `5bfb761815a09ee88f19072e86c871cd369b99e1f313a6813dda6f18ac16f292`.
The final boundary artifact is
`research/findings/raw/v13_tonic_output_stage0_performance_continuation_v7/final-stage0-v7.json`
with canonical SHA-256
`f81731143c6f287fe7d0a71e2b233521a695d34389a0bc2939856ce7c828848f`.

## Next action

Profile the v2 active path against v2 default, remove enough per-step overhead to
meet the fixed `1.10` limit, and correct candidate output placement before any
new performance confirmation is preregistered. The first measurement remains the
controlling negative result; a later run must be a declared confirmation of a
code change, not a retry of unchanged code.
