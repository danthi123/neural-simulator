---
type: finding
status: complete
date: 2026-08-04
verdict: DIAGNOSTIC_PASS
mechanism: gateB-v13-backend-neutral-izh-arithmetic-replay-v2
artifacts:
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/source.sha256
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/run-numpy.command.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/run-cupy.command.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/compare.command.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/cell-numpy.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/cell-numpy.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/cell-cupy.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/cell-cupy.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/comparison.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/comparison.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/evidence-manifest.json
---

# V13 strict-arithmetic replay v2 is byte-exact across NumPy and CuPy

**Verdict: DIAGNOSTIC_PASS.** The locked v2 matched-state replay produced
byte-identical NumPy and CuPy trajectories for membrane voltage `v`, recovery
state `u`, and spikes across all `1,200` steps and `60` cells. This closes only
the backend-arithmetic diagnostic gate represented by this replay. It has no
promotion value and supplies no scientific verdict about V13 calibration,
replication, held-out evaluation, or Stage 1.

## Evidence integrity

Primary comparison artifact:
`research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2/comparison.json`.

The evidence is bound to source revision
`c2db1becbe0948424958faec0f706f492de447c9`, locked spec SHA-256
`94d96fb6a67e0d7df3d151e7b1dbbb85a2a9315115e1aa165f6f9355df6f0992`,
and a `67`-file source manifest with SHA-256
`e3d90dab8a34d2541ace6aa331807bcac312f9eaac6896c89f82715b672b2b52`.
An independent read-only audit rehashed every manifest entry against its Git
blob at the frozen revision and confirmed that the set contains all `43`
simulator Python files and all `24` declared v2 authority inputs.

The audit verified the three command-envelope self-digests and their canonical
receipt commands, prerequisites, source identity, environment, host, and
device bindings. The successful receipts bind NumPy to `SIM_BACKEND=numpy` on
`cpu:numpy`, CuPy to `SIM_BACKEND=cupy` on `gpu:rtx-3090`, and comparison to
`SIM_BACKEND=numpy` on `cpu:numpy`. Artifact sizes and file hashes match their
receipts and final manifest. The NumPy cell, CuPy cell, comparison, comparison
receipt, and final evidence-manifest file SHA-256 values are respectively:

- `e68ebc82ab4d7cbb2f4dcbfea77bdfaabd150eb2da87aa22d0c8d55e238e9055`;
- `4f8ff1e54773710ac4088b0c3485c6d91e1f2bc5a62cd93b67c78258f976de15`;
- `5710dcdc643a6e0221ba28832501ec5e89e1a854278273c0bae43226de7d755c`;
- `a35ab9a892641028c2bbbfcc6db8cfc64740d63da23da06a15a2d9f11f356292`;
- `c9ceda9eaa263c542c88240f9df6661d808352c800abc8e3b79431f8e1578ecf`.

Both cells report exact restoration of all `32` allocated simulator arrays and
all three CSR arrays before the measured boundary. The strict arithmetic flag
was enabled, its declared default remained disabled, and the incompatible step
megakernel was disabled. The measured replay records `seed: null`, disallows
RNG use, and observed zero calls across all guarded host and GPU RNG APIs.
Weights and intrinsic current remained unchanged, timing and step counts were
exact, and the instrument-valid fields are internally consistent.

**Instrument:** receipt-bound matched-state replay with independent source,
restoration, no-RNG, artifact, aggregate-hash, per-row-hash, and direct-byte
verification.

## Exact result

The audit completed `11,061` checks without executing a simulation or using a
seed. It decoded all six trajectory arrays (`432,000` stored values),
recomputed `7,200` per-row SHA-256 hashes and all six aggregate hashes, and
performed `3,600` direct cross-backend row comparisons. Every decoded array
had the required `1,200 x 60` shape and dtype; both float trajectories were
finite.

| trajectory | exact rows | first difference |
|---|---:|---|
| `v` | `1,200 / 1,200` | none |
| `u` | `1,200 / 1,200` | none |
| spikes | `1,200 / 1,200` | none |

The comparison and final manifest therefore correctly record
`DIAGNOSTIC_PASS`, `promotion_value: none`, and `scientific_verdict: null`.
This establishes only that the locked restored V13 state follows the same
strict arithmetic trajectory on NumPy and CuPy. Fresh V13 scientific
calibration remains a separate next step governed by the experiment
controller.
