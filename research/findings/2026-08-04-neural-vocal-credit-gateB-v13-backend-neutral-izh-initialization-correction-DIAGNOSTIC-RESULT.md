---
type: finding
status: complete
date: 2026-08-04
verdict: DIAGNOSTIC_PASS
mechanism: gateB-v13-backend-neutral-izh-initialization-correction
artifacts:
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/source.sha256
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/capture-numpy.command.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/capture-cupy.command.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/compare.command.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/cell-numpy.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/cell-numpy.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/cell-cupy.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/cell-cupy.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/comparison.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/comparison.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/evidence-manifest.json
---

# V13 backend-neutral initialization is exact at step zero

**Verdict: DIAGNOSTIC_PASS.** With the correction enabled, NumPy and CuPy
produced byte-identical values for all `15` preregistered initialization arrays.
This diagnostic has no promotion value and makes no claim about runtime
arithmetic, firing behavior, calibration, replication, held-out, or Stage 1.
Diagnostic seed `6556023` is consumed.

## Evidence integrity

Canonical evidence is
`research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/evidence-manifest.json`;
the exact array result is
`research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic/comparison.json`.

The run is bound to source revision
`58f5b1e4f1e46fdd96c5fa6bb77a5724d82ef25c`, spec SHA-256
`a3be9d0d0c0b99bce7e6f14f54a8e6b2ce937e62d72ac05cc680a9a8a7ed3ef4`,
and a `55`-file source manifest with SHA-256
`775a22dc2d914d229d124311a3fc60b99f1a80a75be510892dabb4ec2fde1342`.
The final evidence manifest has embedded SHA-256
`36184eea6764b000511b18bd5f7f544f4749842457f324dfed2b678d1f696929`
and file SHA-256
`04debc09033da63c4175602efe2e937db69a06fca376479c69a082002879b42b`.

A read-only audit recomputed `206` checks without constructing a bridge or
executing a seed. All three command-envelope self-digests are valid, and each
envelope's inner command exactly matches its success receipt. Every receipt
matches the present artifact hash and size, source identity, environment,
host, device, and timestamps. The NumPy capture and comparison ran on
`dant123-wk` with `CPU: 12th Gen Intel(R) Core(TM) i7-12700`; the CuPy capture
ran on the same host with `NVIDIA GeForce RTX 3090`.

## Result

Both cell artifacts carry the same complete configuration identity,
`1090d2b785a7981562da7437ee51181bbcb4fe1ee411430ddadaa8d379f80873`,
with `backend_neutral_izh_initialization=true`, seed and heterogeneity seed
`6556023`, and the locked `20`-cell inhibitory-source plus `40`-cell GPi/SNr
population. Before and after capture, time was `0.0 ms` and the time-step index
was `0` on both backends. The comparison therefore records zero simulation
steps, not a trajectory result.

Independent base64 decoding and SHA-256 checks confirmed matching dtype,
shape, and bytes for every required array:

- traits, neuron-type IDs, firing thresholds, and 3D positions;
- `C`, `k`, `vr`, `vt`, `vpeak`, `a`, `b`, `c_reset`, and `d_increment`;
- initial membrane potential `v` and recovery state `u`.

The NumPy and CuPy cell file SHA-256 values are respectively
`bcdc48d0af130d9b554b7bb6ca3e0869cc231176d3ee16886f74ff908a8f766d`
and `9f354fd1539f0ebac8d437dee197e8a556f26f00fcd6d8cc588141801ed8866d`.
The comparison artifact reports `DIAGNOSTIC_PASS`; its embedded digest is
`5d91a6c7a336f4ebcb6f065339b3526156b4ac2a8e1b70aa3d0374d95d567bfc`
and its file SHA-256 is
`4715ac91a2d2a0a513f424c30ddd8be9c0d2201e146f5839eadeb47be4fa92ff`.

## Boundary and closure

This closes the initialization half of the backend diagnosis only. The prior
state-transplant result independently found neuron-update arithmetic divergence
after stepping; that correction still requires its own matched-state evidence
before corrected V13 calibration can be released.

The preregistration and spec remain locked at their executed bytes. Completed
diagnostics in this repository record closure in a separate result finding
rather than rewriting source-bound protocol inputs after execution.
