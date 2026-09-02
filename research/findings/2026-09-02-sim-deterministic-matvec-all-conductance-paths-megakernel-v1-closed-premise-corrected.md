---
status: live
type: finding
lane: integration
date: 2026-09-02
---

# Deterministic matvec across ALL conductance paths — the last residual (megakernel-v1) is closed; the "matvec is the 7-organ merge blocker" premise was already refuted + closed

Status: GO for the code contract (every sparse conductance transpose-matvec now honors
`cfg.deterministic_transpose_matvec`), with a **verify-first CORRECTION** to the Phase-2 crux framing.

## The one-line result

The one remaining conductance matvec not honoring the deterministic flag — the GPU-only opt-in
read-only **megakernel-v1** fast path (`_run_one_step_megakernel`) — is now routed through the same
`add.reduceat` segmented reduction the main step uses. It was the exact path commit `06ce99c76`
(2026-08-27) explicitly left as "GPU-only megakernel matvec unchanged". With it closed, **every**
sparse conductance transpose-SpMV is deterministic under the flag.

## Verify-first correction to the Phase-2 crux

The integration-program plan (`docs/plans/2026-09-02-onebrain-integration-program.md`) frames the crux
as: the strict 7-organ byte-identity batch was NO-GO because the non-flagged conductance matvecs
(esp. slow-NMDA-recurrent) carried FP summation-order variance past N≈4968. **That hypothesis was
already tested and REFUTED on 2026-08-27** (commit `06ce99c76`): on numpy the matvecs are already
co-residence-invariant, so hardening them changed `read_maxerr` by exactly zero. The real 7-organ wall
was a co-residence-DEPENDENT nmda_slow WIRING/RNG seam (d6: 5148/10819 receptor tags differed
merged-vs-core), since CLOSED by `dedup_synapse_masks` (`2898185dd`) + `per_region_inhibitory_seed`
(`cb8bc175b`) → 7/7 organs GO. So this `sim/` edit is NOT what unblocks the merge (it was already
unblocked); it COMPLETES the flag's determinism contract for the last path prior work deferred.

Ground truth of the conductance matvecs, verified in code + data this session:

- **Already deterministic under the flag (main `_run_one_simulation_step`):** main E/I split
  (`_deterministic_ei_transpose_spmv`), main non-inhibitory, graded transmission, slow-NMDA-recurrent,
  coincidence, graded-dendritic-plateau, GABA_B. (fa0168a20 2026-08-25 + 06ce99c76 2026-08-27.)
- **Already deterministic by construction (default-on megakernel-v2):** its RawKernel sums each
  post-neuron's incoming edges sequentially in CSR order in double — order-stable, no atomics, no flag
  branch needed.
- **Was NOT (closed by THIS edit): megakernel-v1** — its flag branch only `.tocsr()`-ed the transpose,
  then still ran the cuSPARSE two-column csrmv (atomic FP accumulation → run-to-run non-reproducible).

External anchor for the mechanism: the NVIDIA cuSPARSE Library documentation
(docs.nvidia.com/cuda/cusparse/index.html) states operations using **transpose / conjugate-transpose
have NO reproducibility guarantee** — precisely our `.T @ v` case — while the deterministic route
orders accumulations explicitly. The floating-point-non-associativity root cause (atomic-reduction
ordering is runtime-dependent → non-reproducible) is characterized in Chan et al., *Impacts of
floating-point non-associativity on reproducibility for HPC and deep-learning applications*,
arXiv:2408.05148 (2024) <!--derived--> (external arXiv id, not a run measurement). `add.reduceat`
fixes it by imposing a fixed per-row summation order.

## The edit (additive / guarded / byte-identical when OFF)

`sim/bridge.py :: _run_one_step_megakernel`. The E/I-split branch now calls
`_deterministic_ei_transpose_spmv(self.cp_connections, exc_fired_prev, inhib_fired_prev)` and the
non-inhibitory branch calls `_deterministic_csr_matvec(_eff_cT1.tocsr(), prev_fired_float)` when
`cfg.deterministic_transpose_matvec` is set — mirroring the main path (`bridge.py` ~8898/~8922). When
the flag is OFF the expressions are the unchanged `.T @ fired_2col` / `.T @ prev_fired_float` (byte-
identical to pre-edit). Default is OFF; the shipped brain is unchanged.

## Evidence (hashes: `research/findings/raw/_deterministic_matvec_all_paths_ceiling.json`)

Ceiling test — megakernel-v1 read, cupy/3090, N=5200 (>4968), seed 4242, IZH read-only:

- **BEFORE (flag OFF = pre-edit cuSPARSE path):** g_e diverges run-to-run — g_e SHA256 `b85c8ca7…`
  (run1) vs `14169b46…` (run2). (v/firing had not yet amplified into a flip at 40 steps — the jitter
  lives first in continuous conductance, matching the "long read amplifies past ~4968" mechanism.)
- **AFTER (flag ON, post-edit):** byte-identical run-to-run — firing `81e4645f…`, g_e `42172fb9…`,
  v `fbd36b9f…` on both runs. Megakernel-v1 dispatched (verified).

Matvec-level micro-demo (cupy): the cuSPARSE two-column transpose csrmv (exactly megakernel-v1's E/I
op) gives 8 distinct SHA256 over 8 identical calls at BOTH N=4096 and N=6000; the reduceat helper gives
1. (The single-column csrmv was stable at fan-in 48 — density-dependent; the 2026-08-25 gap#5 finding
measured 1-col non-determinism at density 0.9.)

Small-N / no-regression (numpy, main step, N=6000): two builds at one seed are byte-identical under the
flag (helper exercised 120×, non-vacuous); flag ON vs OFF give identical firing+v hashes and g_e max
rel diff 4.38e-8 (pure f32). On numpy the divergence cannot be reproduced (order-stable backend) — the
reason the 2026-08-27 all-numpy merge batch could not attribute the wall to the matvec.

## Tests

`tests/test_determinism.py`:
- `TestDeterministicMatvecAtScale::test_main_step_build_twice_byte_identical_past_4968_under_flag`
  (numpy-runnable; non-vacuous via a helper-call spy) — the "build twice past N≈4968" pin.
- `TestMegakernelV1DeterministicMatvec` — GPU-guarded: the cuSPARSE-vs-helper primitive (fast) + the
  megakernel-v1 end-to-end build-twice byte-identity under the flag (slow; queued on `gpu_queue` →
  `research/findings/raw/_deterministic_matvec_gpu_confirm.txt`).
- Numpy suite: `tests/test_determinism.py` 10 passed / 4 skipped (GPU-guarded).

Also repaired a PRE-EXISTING red: `tests/test_deterministic_sparse_matvec.py`
`::test_split_primitive_executes_two_one_dimensional_spmvs` had a stale `FakeCSR` mock that expected a
bare `@` and raised `AttributeError: 'FakeCSR' object has no attribute 'indptr'` (unchanged on HEAD —
confirmed via `git stash`) since the primitive switched to `_deterministic_csr_matvec` on 2026-08-25.
Repaired to exercise the real helper (tocsr-once + two 1-D SpMVs + float64 correctness). This is
directly the function megakernel-v1 now routes to.

## Honest residual (paths NOT touched, and why)

- **Dense SSM readout `cp_ssm_readout_w @ cp_ssm_state`** (bridge.py ~8602): a DENSE gemv, not a sparse
  conductance matvec; cuBLAS is pinned by `CUBLAS_WORKSPACE_CONFIG`. Out of this arc's scope.
- **Speed:** the reduceat path is the same one the main step already uses; megakernel-v1 is opt-in and
  default-off (v2 is the default), so no shipped path changes cost. Speed<faithfulness holds regardless.
- The queued cupy confirmation is a CI-style re-run of the committed GPU tests; the decisive
  before/after divergence was already measured live this session (hashes above).
