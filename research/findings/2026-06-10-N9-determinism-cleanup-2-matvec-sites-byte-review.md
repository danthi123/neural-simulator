# N9 determinism cleanup — the two restricted matvec sites (FOR OWNER BYTE-REVIEW)

> **Protected `sim/` edit, committed DEFAULT-OFF (byte-identical for every current run). Presented for byte-review
> before being relied upon — I will NOT use it in the N9 multi-seed determinism validation until you approve the
> diff.** This completes the "airtight N9 closeout" residual #3 (clean multi-seed determinism).

## What and why

The navigation A/B and the dopamine prediction-error battery want clean multi-seed reproducibility. The project
already has a proven, **owner-approved** fix for the one non-reproducible operation in the step — the transpose
sparse matrix-vector product (SpMV). On the GPU, `csr.T @ v` routes to a transpose SpMV whose atomic scatter sums
in a run-to-run-variable order (floating-point summation-order variance; `CUBLAS_WORKSPACE_CONFIG` pins cuBLAS
only, not this). The fix (`cfg.deterministic_transpose_matvec`): when set, materialize the transpose as a CSR
(`.tocsr()`) so the op becomes a reproducible one-thread-per-output-row non-transpose SpMV; default off leaves the
expression unchanged.

That fix was applied to the conductance matvec (`bridge.py:5648-5651`, `:5660-5662`) and approved. But **two more
restricted transpose SpMVs** — both on the spiking-critic path — were still using the non-reproducible form:

1. the **dendritic-coincidence plateau** drive (`bridge.py:5814`): `c_drive = _co_mat.T @ prev_firing`
2. the **GABA_B / GIRK** value-subtraction increment (`bridge.py:5858`): `gabab_increase = (_gb_mat.T @ prev_firing) * ...`

This edit applies the **identical** proven fix to those two sites.

## The diff (16 insertions / 2 deletions, `sim/bridge.py` only)

Both sites change from:

```python
c_drive = _co_mat.T @ self.cp_prev_firing_states.astype(cp.float32)
```

to:

```python
_co_matT = _co_mat.T
if getattr(cfg, "deterministic_transpose_matvec", False):
    _co_matT = _co_matT.tocsr()
c_drive = _co_matT @ self.cp_prev_firing_states.astype(cp.float32)
```

(and the analogous change at the GABA_B site).

## Byte-identity argument

- **Flag OFF (default — every current run):** the change is a pure **extract-to-variable refactor**. `_co_matT =
  _co_mat.T` then `_co_matT @ v` performs the exact same operations as `_co_mat.T @ v` — there is no
  floating-point reordering, no new array, no behavioural change. Byte-identical by construction (and `py_compile`
  passes).
- **Flag ON:** the only added operation is `.tocsr()` on a `csr_matrix.T` (a `csc_matrix`), which is the **exact
  same operation already approved** at the conductance matvec (`_eff_cT.tocsr()`, `bridge.py:5650`). It yields a
  result `allclose` to the csc product but with a deterministic summation order.
- **Both sites are additionally guarded** by their feature flags (`enable_coincidence_detection` /
  `enable_gabab`): the blocks are skipped entirely unless the spiking critic is enabled, so even the OFF-vs-ON
  distinction only exists on the N9 critic path.

Izhikevich / Hodgkin-Huxley / AdEx dynamics and the global step dispatch are untouched.

## What it unblocks (post-review)

With these two sites covered, `--deterministic-selforg` (which sets `deterministic_transpose_matvec`) makes the
**entire** step reproducible, enabling a clean multi-seed N9 determinism validation (the navigation A/B and the
prediction-error battery across seeds without run-to-run float noise). That validation is the next step, **after**
this diff is byte-reviewed.

## Status

- Committed default-off (no behavioural change for any current run).
- `py_compile sim/bridge.py` passes; `git diff --stat sim/` = `1 file changed, 16 insertions(+), 2 deletions(-)`.
- Awaiting owner byte-review before use in the N9 validation.
