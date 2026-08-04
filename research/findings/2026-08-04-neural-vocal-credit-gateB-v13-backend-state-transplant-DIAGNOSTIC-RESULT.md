---
type: finding
status: complete
date: 2026-08-04
verdict: DIAGNOSTIC_ONLY
mechanism: gateB-v13-backend-state-transplant-diagnostic
artifacts:
  - research/findings/raw/v13_backend_state_transplant/aggregate.json
  - research/findings/raw/v13_backend_state_transplant/bundle-numpy.json
  - research/findings/raw/v13_backend_state_transplant/bundle-cupy.json
  - research/findings/raw/v13_backend_state_transplant/run-numpy-on-numpy-default.json
  - research/findings/raw/v13_backend_state_transplant/run-numpy-on-numpy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/run-numpy-on-cupy-default.json
  - research/findings/raw/v13_backend_state_transplant/run-numpy-on-cupy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/run-cupy-on-numpy-default.json
  - research/findings/raw/v13_backend_state_transplant/run-cupy-on-numpy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/run-cupy-on-cupy-default.json
  - research/findings/raw/v13_backend_state_transplant/run-cupy-on-cupy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/receipt-bundle-numpy.json
  - research/findings/raw/v13_backend_state_transplant/receipt-bundle-cupy.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-numpy-on-numpy-default.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-numpy-on-numpy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-numpy-on-cupy-default.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-numpy-on-cupy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-cupy-on-numpy-default.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-cupy-on-numpy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-cupy-on-cupy-default.json
  - research/findings/raw/v13_backend_state_transplant/receipt-run-cupy-on-cupy-deterministic_transpose_matvec.json
  - research/findings/raw/v13_backend_state_transplant/receipt-aggregate.json
---

# V13 backend state transplant: initialization dominates, execution also diverges

**Verdict: DIAGNOSTIC_ONLY.** This result has no promotion value and does not
validate calibration, replication, held-out, or Stage 1. Diagnostic seed
`7606856` is consumed. Seeds `1013` and `1019` remain consumed, while `1021`
and `1031` remain sealed.

## Evidence integrity

The sealed result is bound to spec SHA-256
`cd29d694d5e4d413167d83f0817532c1e31695ccb0d9a72490e81671c8ff920f`,
protocol source anchor `b3d57494b7dd7d99d5e91088489da44d89a85bf3`, and execution revision
`74f3dcd60ba1e5f866a7639986e1fb96fcbd541b`. Its six-file source manifest is
`4a91ec11cc7023129fb12f75369e9c0e84c7bef1a00322deb11d780d087de7ca`
with tree digest
`5a91edf380a4125d1b99222d7cc722e2b98e2f527af32b7a7f0185cf6648e6a0`.

An independent read-only audit recomputed every receipt's artifact hash and
size, every artifact self-digest, every encoded-array digest, and every
per-step trajectory digest. All two bundle receipts, eight run receipts, and
the aggregate receipt report success and match the present bytes. The matrix
contains exactly the preregistered `2 origins x 2 execution backends x 2
modes = 8` unique cells. Every run reports exact pre-step state restoration,
zero target external current, immutable weights and intrinsic current, the
locked source schedule, and valid instrumentation. See
`research/findings/raw/v13_backend_state_transplant/aggregate.json` and the
receipt files listed above.

## Result

The native-origin bundles are not the same initialized population. NumPy and
CuPy produced different `C`, `a`, `b`, and `d` values in all `40` GPi/SNr
cells, and different neuron-type IDs in `51` of `60` cells. CSR topology and
weight data were exact. This directly confirms that a shared integer seed did
not isolate execution backend.

The large physiological difference followed bundle origin, not execution
backend. Rates below are identical in default and deterministic-transpose
modes:

| bundle origin | execution | baseline Hz | inhibition Hz | release Hz | suppression ratio |
|---|---|---:|---:|---:|---:|
| NumPy | NumPy | `63.800` | `1.125` | `57.950` | `0.017633` |
| NumPy | CuPy | `63.700` | `1.250` | `57.850` | `0.019623` |
| CuPy | NumPy | `63.750` | `11.625` | `57.400` | `0.182353` |
| CuPy | CuPy | `63.650` | `11.375` | `57.500` | `0.178712` |

Initialization is therefore the leading cause of the earlier suppression
split. It is not the only cause: replaying either byte-identical bundle on the
two execution backends still diverged. Conductance trajectories `g_e` and
`g_i` stayed exact, but neuron state and spikes did not:

| origin | first byte difference | first tolerance difference | first spike difference |
|---|---|---|---|
| NumPy | `u`, step `2`, cells `30/33/41/57`; `v`, step `10`, cell `40` | `v`, step `18`, cell `31`; `u`, step `26`, cell `40` | step `184`, cell `55` |
| CuPy | `u`, step `2`, cells `21/24/32/34`; `v`, step `4`, cell `32` | `v`, step `11`, cell `32`; `u`, step `13`, cell `32` | step `183`, cell `51` |

The deterministic transpose-matvec path reproduced the default-mode rates and
the same first-divergence records for both origins. It did not remove or delay
the execution-backend difference in this diagnostic. Because conductances
remain exact and `u` diverges first, sparse transpose accumulation is not the
remaining localizer; the neuron-state update arithmetic is.

## Process note

The first NumPy bundle command was sent to `pool40`, whose provisioned source
was a verified Git archive without repository ancestry. The runner rejected
the source-anchor check before bridge construction with
`source_anchor_present: false`. It produced no artifact or success receipt and
did not advance seed `7606856`. The sealed NumPy bundle was subsequently made
locally from the same receipt-bound revision. Future scientific pool work must
fix this provisioning/ancestry contract before launch rather than weaken a
locked runner during execution.

## Required correction before calibration

1. Replace backend-native initialization with one version-pinned canonical
   generator and a sealed little-endian, C-contiguous population bundle. Bind
   and replay every randomized or assigned array, including neuron-type IDs,
   all Izhikevich parameters, initial `v/u`, intrinsic current, region mapping,
   CSR state, receptor state, and stimulus schedule. Device-load and round-trip
   hashes must match before step zero.
2. Separately localize and standardize the `u/v` neuron-update arithmetic that
   begins diverging at step `2`. The existing deterministic sparse path is not
   a correction. A fresh diagnostic-only, matched-state test must show the
   intended cross-backend tolerance and spike contract before any verdict seed
   is released.
3. Preregister, test, freeze, and commit both corrections and their source
   manifest. Do not reuse diagnostic seed `7606856` or carry forward the
   observed `100 pA` operating point.
4. Only then run corrected calibration seed `840860` in the required serial
   order: NumPy artifact validated and sealed before CuPy starts. Replication
   seed `687979`, held-out seed `1021`, and Stage-1 seed `1031` remain gated by
   the corrected Stage-0 controller.
