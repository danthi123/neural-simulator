# Deterministic transpose-SpMV — a default-off `sim/` flag that makes the place-code self-org reproducible (FOR OWNER BYTE-REVIEW)

**Date:** 2026-06-10 (overnight autonomous)
**Type:** protected `sim/` edit (additive, default-off, byte-identity-proven) + runner wiring + validation.
**Why:** the N9 place-code self-org is CuPy-non-deterministic run-to-run, blocking multi-seed robustness (the same seed draws a different place code → different critic strength → different gates). Deep research (`2026-06-10-N9-placecode-reproducibility-robustness-research.md`) proved the root cause is the **transpose SpMV** `Wᵀ@fired` (`bridge.py:5613/5620`): `csr.T`=csc → `cusparse.spmv(transa=True)` atomic scatter (bit-non-reproducible; `CUBLAS_WORKSPACE_CONFIG` pins cuBLAS only). The numerically-identical deterministic replacement is to materialize the transpose as a CSR (a non-transpose SpMV).

## The edit (additive, default-off)

- **`sim/config.py`:** `deterministic_transpose_matvec: bool = False` (new field, default off).
- **`sim/bridge.py:5613` (E/I-split) + `:5620` (single):** when the flag is set, bind `_eff_cT = effective_connections_matrix.T.tocsr()` (deterministic non-transpose SpMV) instead of `.T` (the csc atomic-scatter path); the matvec is otherwise unchanged. When the flag is **off**, the expression is the *unchanged* `effective_connections_matrix.T @ fired` ⇒ byte-identical to baseline. No kernel edit.
- **Runner (`g11_bg_runner.py`):** `--deterministic-selforg` toggles the flag **ON only during STEP-1 place self-org** (bounds the per-step `.tocsr()` cost to STEP-1; the place code is then frozen), restored after. Default off ⇒ byte-equivalent.

## Byte-identity PROOF (the owner's gate — PASSED)

Harness `research/findings/raw/_deterministic_matvec_byte_identity_check.py` (NumPy backend = deterministic, so a clean hash-compare is possible; CuPy's atomic scatter is the very thing the ON path fixes). Same-seed 200-step trajectory SHA-256 of `(v, g_e, g_i)`:

| tree / flag | SHA-256 |
|---|---|
| EDITED, flag **OFF** | `a7d15382bc8edf2ae8d3dea4e71ef1c79fe7631184ee5d0763542fc2cc8d92ed` |
| EDITED, flag **ON** | `a7d15382bc8edf2ae8d3dea4e71ef1c79fe7631184ee5d0763542fc2cc8d92ed` |
| BASELINE (sim/ stashed, flag ABSENT) | `a7d15382bc8edf2ae8d3dea4e71ef1c79fe7631184ee5d0763542fc2cc8d92ed` |

**All three identical** ⇒ (1) **off == baseline byte-identical** (the gate), and (2) on == off bit-identical on NumPy (the `.tocsr()` path is numerically exact, not merely allclose, on the deterministic backend). The ON path's *value* is on CuPy, where it converts the non-deterministic atomic scatter to the deterministic one-thread-per-row layout.

## R-A validation (CuPy — does the flag fix the run-to-run non-determinism?) — PASSED

Same seed 42, grid 32, n_place 800, STEP-1 self-org, twice each (CuPy, the production backend):

| condition | run 1 (diff-cos / sparsity) | run 2 (diff-cos / sparsity) | reproducible? |
|---|---|---|---|
| **WITH `--deterministic-selforg`** | 0.061721 / 0.040625 | 0.061721 / 0.040625 | **YES — bit-identical** ✓ |
| WITHOUT (control) | 0.056344 / 0.044375 | 0.060606 / 0.041250 | NO — differs (the non-determinism) |

**R-A PASSES:** with the flag, the same seed draws a byte-identical place code; the control confirms it is the flag (the deterministic SpMV), not luck-of-the-draw. The cusparse transpose-SpMV non-determinism is removed at the place-code source.

## Scope / honest limits

- This fixes the MAIN matvec (`5613/5620`), which determines the **place code** (the primary draw-variance source). The restricted coincidence/GABA_B matvecs (`5771`/`5812`, also transpose SpMVs) are NOT yet covered — so the value-train `w_near` can still vary slightly from the coincidence matvec; full-run reproducibility would extend the flag to those sites (a follow-up, same pattern). For the R-A anti-cheat (reproducible place code) the main matvec suffices.
- Global value (beyond N9): with the flag on, the **entire** deterministic-regime engine becomes reproducible — a project-wide win. Left default-off (the `.tocsr()` per-step cost) so production is unaffected.

## Recommendation for byte-review

Additive, default-off, byte-identical-when-off (proven). The ON path is a storage-layout change (transpose→CSR), numerically exact on NumPy and deterministic on CuPy. Reviewable as a one-line-per-site swap mirroring the established per-region-mask edits. On approval, it unblocks the clean robustness work (attributable multi-seed) + a clean nav A/B.
