---
type: finding
status: complete
date: 2026-08-04
verdict: DIAGNOSTIC_FAIL
mechanism: gateB-v13-backend-neutral-izh-arithmetic-replay-v1
artifacts:
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/source.sha256
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/run-numpy.command.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/run-cupy.command.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/compare.command.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/cell-numpy.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/cell-numpy.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/cell-cupy.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/cell-cupy.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/comparison.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/comparison.receipt.json
  - research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/evidence-manifest.json
---

# V13 strict-arithmetic replay v1 reaches step 1,114 before a recovery-state difference

**Verdict: DIAGNOSTIC_FAIL.** The v1 strict-arithmetic replay kept membrane
voltage `v` and spikes byte-exact for all `1,200` rows, but recovery state `u`
first differed at step `1,114`, cell `0`. This is a diagnostic failure only. It
has no promotion value, consumes no replay or scientific seed, and supplies no
scientific verdict about V13 calibration, replication, held-out, or Stage 1.

## Evidence integrity

Canonical evidence is
`research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/evidence-manifest.json`;
the exact cross-backend result is
`research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/comparison.json`.

The run is bound to source revision
`882b17fea0ed2752fab8dcae3c13f526e921a33e`, locked spec SHA-256
`c8f42fee5d8d2dc044cf05ed7676c06ef7c20e093613a06facce311512686d2d`,
and a `62`-file source manifest with SHA-256
`25dd5a4ab9e7f52b30aeeefc9cb3d364ee4207aa47560cb0834a09c73b09e443`.
Every manifest entry was rehashed against the corresponding blob at the frozen
Git revision, and the path set exactly matched all simulator Python files plus
the replay's declared authority inputs.

A read-only audit independently checked the command-envelope self-digests and
confirmed that each envelope's inner command matches its success receipt. The
receipts bind the NumPy cell to `SIM_BACKEND=numpy` and `cpu:numpy`, the CuPy
cell to `SIM_BACKEND=cupy` and `gpu:rtx-3090`, and the comparison to the same
source identity and input artifacts. Present file hashes and sizes match every
receipt and final-manifest reference. The NumPy cell, CuPy cell, comparison,
and final evidence-manifest file SHA-256 values are respectively:

- `ad69c842077485614b5ba3319cccfb39cdcfd20011965774f94634906e437303`;
- `8fd6dc46ca67c92141e663245b47afe5b3a4ec8c75a22f4d27cf9b07534e0db9`;
- `eff8c25b57017ff994dee0dbce018275862c531ed7b901f6a908934ace51983a`;
- `138478f0f8ded7cb9ee55c3df5a75dda35358de3c8942470661cad4bcc1704c7`.

The audit also revalidated the immutable NumPy-origin transplant bundle, its
completed aggregate and receipts, and their exact hashes from the locked spec.
Both replay cells report byte-exact restoration of all allocated simulator
arrays and CSR state before measurement. They also report strict arithmetic
enabled, its declared default still disabled, the incompatible step
megakernel disabled, immutable weights and intrinsic current, exact time and
step counts, finite `v/u`, and zero calls to guarded RNG APIs across the
measured boundary. The frozen runner source contains the restoration checks,
RNG guard, and fail-closed instrument validation that produced those fields.

**Instrument:** receipt-bound matched-state replay with independent decoding,
aggregate SHA-256 checks, and per-row SHA-256 checks for all three trajectories.

## Exact result

Independent base64 decoding reconstructed all six arrays as exactly
`1,200 x 60`, with float32 `v/u` and boolean spikes. Recomputed aggregate and
per-row hashes matched both cell artifacts. Direct byte comparison found:

| trajectory | exact rows | first difference |
|---|---:|---|
| `v` | `1,200 / 1,200` | none |
| spikes | `1,200 / 1,200` | none |
| `u` | not all exact | step `1,114`, cell `0` |

At row index `1,113` (step `1,114`), NumPy recorded
`-4.504769307288404e-38` (`0x81754352`) while CuPy recorded
`-5.630961493980659e-38` (`0x81994a13`). The preceding `u` value was exact at
`0x81994a13`. The state difference is approximately
`1.1261921866922546e-38` (`0x007aa1a8`), below the smallest normal float32
magnitude. In other words, the first failure is at subnormal scale; the two
stored negative `u` states themselves are not subnormal.

The comparison artifact therefore correctly records `DIAGNOSTIC_FAIL`,
`promotion_value: none`, and `scientific_verdict: null`. No tolerance fallback
is warranted because the locked acceptance criterion requires byte identity.

## Localized cause and required v2

The frozen v1 source used CUDA `__f*` intrinsics in the strict CuPy kernel.
After the replay was frozen, commit
`4c685269c73f1c3a714c212f2c76d5ddb25aaa7b` replaced those operations with
explicit round-to-nearest PTX instructions without `.ftz` and added a
regression for the exact `0x81994a13` to `0x81754352` boundary. That source and
test evidence supports the localized cause: the v1 GPU path flushed the
subnormal recovery update, while NumPy retained it.

The correction was implemented after source revision `882b17fea`; it is not
part of this frozen run and cannot change this result. A fresh, independently
receipted v2 replay of all `1,200` rows is required before strict arithmetic is
considered closed or corrected V13 calibration is released.
