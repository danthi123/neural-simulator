---
type: preregistration
status: locked
date: 2026-08-04
mechanism: gateB-v13-backend-state-transplant-diagnostic
spec: research/specs/v13_backend_state_transplant.json
promotion_value: none
---

# V13 backend state-transplant diagnostic

The invalid seed-`1019` run suggests that NumPy and CuPy created different
GPi/SNr neuron populations from the same integer seed. It cannot distinguish
that initialization difference from backend arithmetic near spike threshold
because the artifacts retain means rather than complete cell parameters and
trajectories.

This diagnostic resolves that ambiguity before corrected calibration. It has
no promotion value, cannot rescue the invalid calibration or replication, and
must not execute seeds `1013`, `1019`, `1021`, or `1031`.

## Locked diagnostic seed

Use seed `7606856`. It is mechanically derived by taking SHA-256 of:

`V13_BACKEND_STATE_TRANSPLANT_V1|b3d57494b7dd7d99d5e91088489da44d89a85bf3|role=paired_origin`

The first 12 hexadecimal digits are `b6388351e7c8`; the seed is
`2000000 + (integer mod 7000000) = 7606856`. No alternative was tested.

## Sealed bundles and matrix

Construct one full source-plus-GPi/SNr brain with NumPy-native sampling and one
with CuPy-native sampling, both from seed `7606856` and the frozen V13
inhibitory-response configuration. Before a scored step, archive each bundle's:

- exact `C`, `a`, `b`, `d`, `k`, `vr`, `vt`, and `vpeak` arrays;
- every dynamic neuron, conductance, intrinsic-current, and external-current
  array;
- CSR `data`, `indices`, and `indptr` arrays plus shape;
- region boundaries, neuron types, stimulus schedule, timestep, and all enabled
  feature flags; and
- canonical artifact digest, source manifest, environment, and origin backend.

Load each byte-identical bundle on both NumPy and CuPy. Run two modes per
origin/backend pair: the original default sparse path and the already accepted
deterministic transpose-matvec path. This is an exact `2 origins x 2 execution
backends x 2 modes` matrix. No cell may resample or reconstruct parameters.

Each cell runs the original 1,200-step source-on schedule: 500 ms baseline,
200 ms inhibition with the 20 source cells receiving `1000 pA`, and 500 ms
release. GPi/SNr external current remains exactly zero.

## Measurements

Archive full per-neuron `v`, `u`, `g_e`, `g_i`, and spike state after every
step, with canonical hashes for each trajectory and each individual step.
Record the first exact and first tolerance-exceeding divergence for every pair
sharing a bundle and mode. Also compare complete spike rasters, phase rates,
suppression ratios, source spike schedule, immutable weights/intrinsic current,
and the two origin bundles' exact parameter arrays and topology.

The runner must fail `UNDEFINED` on a missing/mismatched digest, dirty execution
input, resampling, incomplete matrix, source mismatch, nonzero GPi/SNr external
current, changed weight, or changed intrinsic current. It must not label any
scientific outcome GO or NO-GO.

## Interpretation fixed before execution

- If paired execution follows the bundle and matched-bundle spike behavior is
  backend invariant, backend-native initialization is the leading cause.
- If matched bundles diverge by execution backend, backend arithmetic also
  contributes; the first divergent state and mode localize it.
- If deterministic mode removes an execution-backend divergence, retain the
  accepted deterministic sparse path for the corrected cross-backend protocol.
- If the two native-origin bundles differ but each produces the same outcome on
  both execution backends, replace backend-native developmental sampling with a
  canonical, backend-neutral initialization contract before calibration.
- If both initialization and execution contribute, correct and separately test
  both before assigning any new verdict seed.

No corrected calibration may execute until this diagnostic has a committed
comparison finding and any required initialization/arithmetic correction is
preregistered, tested, and frozen.
