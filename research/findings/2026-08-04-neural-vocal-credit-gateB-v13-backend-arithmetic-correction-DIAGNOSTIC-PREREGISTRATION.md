---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-backend-arithmetic-correction
spec: research/specs/v13_backend_arithmetic_localizer.json
promotion_value: none
---

# V13 backend arithmetic correction: diagnostic design

**Status:** implementation-ready, no scientific seed executed
**Scope:** Izhikevich-2007 `v/u` Euler arithmetic only
**Contract:** `research/specs/v13_backend_arithmetic_localizer.json`

## Why this lane exists

The completed state-transplant diagnostic replayed identical population state
on NumPy and CuPy. Conductance trajectories stayed exact, but `u` first differed
at stored trajectory row 2 and `v` at row 10. This design localizes those first
differences without constructing a bridge, drawing randomness, or changing any
completed V13 artifact.

The local RAG was queried for prior project conclusions and scientific sources
using both Izhikevich/numerical-integration and GPU-reproducibility wording. It
returned the completed V13 findings and prior performance tolerances, but no
primary source that specifies CPU/GPU floating-point execution. The biology
catalog describes the Izhikevich mechanism but does not define a cross-device
rounding contract. This is therefore an implementation-reproducibility defect,
not a missing biological parameter.

## Exact localizer result

The read-only probe uses the sealed NumPy-origin bundle and the already-recorded
NumPy/CuPy trajectories:

| equation | fixed probe input | first differing output | cells | residual |
|---|---:|---:|---|---:|
| `u` | row 1 | row 2 | `30, 33, 41, 57` | 1 ULP (`2.98e-8` or `5.96e-8`) |
| `v` | row 9 | row 10 | `40` | 1 ULP (`3.8146973e-6`) |

The row-9 `u` vectors already differ at other cells because `u` diverged first,
but cell 40's `v` and `u` inputs are still byte-exact. The probe feeds the same
sealed NumPy row to both backends and isolates cell 40's first `v` difference.

Every input and every observed intermediate is `float32`. A staged CuPy
evaluation that materializes each primitive operation matches NumPy byte for
byte. The production CuPy fused expression matches an explicit CUDA `fmaf` at
the inner subtraction:

```text
b * (v - vr) - u
(k * (v - vr)) * (v - vt) - u
```

In both equations, NVRTC contracts the final multiply and `- u` into one fused
multiply-add/subtract. NumPy rounds the multiply to `float32` before subtracting
`u`. An NVRTC elementwise kernel compiled with `--fmad=false`, or the same
kernel written with explicit `__fmul_rn` and `__fsub_rn` operations, reproduces
the sealed NumPy bytes. Outer integration contraction is not the cause in these
cases.

This matches the toolchain contract: CuPy fusion creates one CUDA kernel, and
NVRTC enables multiply/add contraction by default. NVIDIA documents that FMA
uses one rounding step, while separate multiply and add/subtract operations use
two.

## Correction design

Add one default-off configuration flag after the initialization correction is
integrated. When false, continue calling the existing
`fused_izhikevich2007_dynamics_update` unchanged. When true:

1. On CuPy, call one device-resident `ElementwiseKernel`. Use explicit
   round-to-nearest `float32` primitives in this exact order:
   `v-vr`, `v-vt`, `k*(v-vr)`, previous result times `(v-vt)`, subtract `u`,
   add current, divide by safe `C`, multiply by `dt`, add `v`; then
   `b*(v-vr)`, subtract `u`, multiply by `a`, multiply by `dt`, add `u`.
2. On NumPy, use explicit `float32` ufunc stages in that same order.
3. Cast `dt` once to `float32`. Keep arrays C-contiguous `float32`; fail closed
   on another dtype rather than silently promoting it.
4. Keep all arithmetic on the selected backend. No per-step host transfer,
   scalar extraction, state quantization, or host-side correction is allowed.

Explicit intrinsics are preferred over relying only on `--fmad=false`: they
make the intended rounding points reviewable and reduce dependence on compiler
defaults. `--fmad=false` remains a diagnostic control.

## Gates before any calibration

1. Unit-localizer: production CuPy must reproduce the sealed CuPy cells;
   strict CuPy must reproduce the sealed NumPy cells exactly.
2. Default-off regression: existing NumPy and CuPy kernel behavior must remain
   byte-identical to current behavior.
3. Opt-in matched-state replay: using sealed state with no RNG, NumPy and CuPy
   must have exact `v`, `u`, and spike trajectories for all 1,200 steps. If a
   later difference appears, localize that operation before widening tolerance.
4. Performance: measure the single strict device kernel against the existing
   fused kernel. A result that stages GPU arrays through multiple launches is a
   diagnostic reference only and cannot be promoted.
5. Only after these gates and the separate backend-neutral initialization gate
   pass may corrected V13 calibration seed `840860` be released.

## Non-claims

- This does not change neuron equations, parameters, time step, or biology.
- This does not prove all simulator kernels are cross-backend byte-identical.
- This does not yet prove the strict kernel removes every later V13 difference;
  the full matched-state replay is deliberately still required.
- No scientific or preregistered seed was executed.

## Sources

- Completed evidence: `research/findings/raw/v13_backend_state_transplant/aggregate.json`
- Local result: `research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-state-transplant-DIAGNOSTIC-RESULT.md`
- CuPy kernel fusion: <https://docs.cupy.dev/en/stable/user_guide/kernel.html#kernel-fusion>
- NVRTC `--fmad` contract: <https://docs.nvidia.com/cuda/nvrtc/index.html#supported-compile-options>
- NVIDIA floating-point/FMA behavior: <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#floating-point-computation>
