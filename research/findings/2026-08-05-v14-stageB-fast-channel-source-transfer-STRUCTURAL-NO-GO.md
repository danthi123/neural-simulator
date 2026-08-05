---
type: finding
status: no-go
date: 2026-08-05
mechanism: v14-snr-stageB-v2-fast-na-kv3-source-transfer
lane: executed-action-credit
promotion_value: none
instrument: NumPy CPU authority with RTX 3090 CuPy descriptive replay
artifacts:
  - research/specs/v14_snr_stageB_structural_successor_v2.json
  - research/specs/v14_snr_stageB_fast_channel_clamp_execution_v1.json
  - research/findings/raw/v14_snr_stageB_fast_channel_clamp_numpy_v1.json
  - research/findings/raw/v14_snr_stageB_fast_channel_clamp_cupy_v1.json
  - research/findings/raw/v14_snr_stageB_fast_channel_clamp_analysis_v1.json
---

# Stage B fast-channel source transfer: structural NO-GO

## Decision

Stop the V2 Stage 1 fast-channel transfer before compartment integration or
conductance calibration. Seven of 18 independent source gates failed. Revise
the fast-sodium activation/deactivation and Kv3 deactivation state equations,
then file a new prospective Stage 1 protocol. Do not compensate with channel
conductance, reopen the old parameter search, or begin Stage 2.

This is a structural result for the filed channel model. It is not evidence
against biological SNr pacemaking or the later action-credit mechanism.

## Authenticated execution

Authoritative analysis artifact:
`research/findings/raw/v14_snr_stageB_fast_channel_clamp_analysis_v1.json`.
Its authenticated inputs are the adjacent NumPy and CuPy clamp artifacts
listed in the front matter.

The sealed runner executed every source command on both NumPy/CPU and
CuPy/CUDA using float32 Rush-Larsen updates. Each backend used 26 vectorized
segment launches, with no Python loop over time steps and no host transfer
before final serialization. The CPU and GPU observations are create-only,
self-digested, and carry automatic provenance sidecars.

The separate analyzer authenticated the execution contract, structural
protocol, implementation hashes, complete command ladders, both observations,
and both provenance records before independently fitting every endpoint. Its
self-digest is
`5044e632533f36bc829896ac02ca03a5950fa4e991341fee186bb2019c901d40`.

## Result

Eleven source gates passed: fast-sodium inactivation midpoint, slope, and
decay; both recovery time constants and fast fraction; Kv3 activation and
inactivation midpoints and slopes; and Kv3 rise time.

Seven gates failed:

| Endpoint | Observed | Allowed mean +/- 2 SEM |
|---|---:|---:|
| fast-Na activation midpoint | -21.559 mV | -31.4 to -29.0 mV |
| fast-Na activation slope | 8.000 mV | 5.8 to 6.6 mV |
| fast-Na 10-90% rise at 0 mV | 0.0377 ms | 0.075 to 0.095 ms |
| fast-Na deactivation at -40 mV | 0.1919 ms | 0.0812 to 0.1168 ms |
| Kv3 deactivation at -60 mV | 1.1896 ms | 0.70 to 0.94 ms |
| Kv3 deactivation at -50 mV | 2.2158 ms | 1.11 to 1.59 ms |
| Kv3 deactivation at -40 mV | 3.5778 ms | 1.55 to 2.19 ms |

The equation-derived gate roots preserve equilibrium powered-gate curves, but
the source assay measures peak composite current while activation,
inactivation, and powered gates are evolving together. That interaction shifts
the observed sodium activation curve and produces incompatible timing. Kv3
shows the same distinction in deactivation: current-tail decay is not the same
quantity as the single activation-gate time constant. These are state-family
and estimator-transfer problems, not missing conductance scales.

## Parity boundary

The original execution contract required CPU/GPU parity but omitted a numeric
tolerance. The analysis therefore reports descriptive differences and
`NOT_ESTABLISHED_NO_PREREGISTERED_TOLERANCE`; it does not assign a post-hoc
parity verdict. A separate prospective comparison contract now binds a
repository-standard float32 tolerance without using observed differences.
Parity cannot rescue the biological NO-GO.

## Next experiment

Use focused source escalation to identify fast-sodium and Kv3 kinetic/state
models that reproduce the complete voltage-clamp currents under the reported
protocols, including powered-gate and recovery interactions. Prefer directly
reported current traces or established Markov/state models over converting
current-level time constants into independent gate constants. The next
contract must test the complete current waveform on cheap CPU clamps before
any GPU population search or compartment integration.
