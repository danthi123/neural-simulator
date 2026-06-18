# Resonate-loop CUDA-graph refactor — implementation design (the orchestration 11× win)

**Date:** 2026-06-17
**Status:** DESIGN (de-risked, execution-ready). The mechanism is proven (11× prototype); this is the production
architecture + the gotchas, so the protected `sim/` edit goes cleanly. Owner cleared higher-risk *revertable* edits.
**Goal:** cut the conversational composer's per-op latency (~160 ms/op, 97.7% the 208-step resonate loop —
`2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md`) by collapsing the ~3–4k sequential per-op kernel
launches into ~one graph replay. Stacks with the shipped batched scan (CYCLE 152).

## What's already established (don't redo)

- **Profile + diagnosis:** the op is launch-bound — `rf_resonate_steps` loops `_rf_advance_one` (~15–20 CuPy
  kernels/step) 208× in Python. `_phaseB_composer_op_breakdown.py`: 162 ms = 97.7%, 780 µs/step.
- **The 11× is real:** `_phaseB_resonate_cudagraph_prototype.py` — a pure-GPU resonate step (device counter,
  pre-allocated buffers, the bind matvec as an **elementwise gather-scale**) captured as ONE CUDA graph ran
  107 ms → 9.8 ms/op (11×). The naive capture **fails** because CuPy cannot capture cuSPARSE/cuBLAS in a graph
  (a CuPy capture-path limit, not categorical — `2026-06-17-snn-vsa-gpu-optimization-literature.md`).
- **The matvec is structured:** the composer's `_resonate` weights are a fixed map — bind/unbind = a
  **permutation** (post `D+k` ← pre `k` × phase); bundle = a **strided sum** (post `k` ← Σ_l pre `l·D+k`). Both
  are elementwise/reshape ops (graph-able), NOT general SpMV.
- **The period can't shrink** (`2026-06-17-resonate-period-free-speedup.md`): recursive clauses need the full 208
  steps. So per-op cost must drop by cheaper steps, i.e. this refactor.

## Production architecture (additive, gated, default-off → revertable by construction)

Add a graphed fast path to `sim/bridge.py`; **leave `_rf_advance_one` / `rf_resonate_steps` as the untouched
fallback**. Gate on `cfg.enable_rf_cudagraph` (default **False**) so default behavior is byte-identical until
validated; flip the default only after the bit-identity + full-suite gate passes.

1. **A graph-safe step kernel (custom `RawKernel`, not `@cp.fuse`).** `@cp.fuse` returns new arrays (forces a
   copy-back each step — kills the win and risks reference breaks). Instead write a `RawKernel`
   `rf_step(re_in, im_in, re_out, im_out, prev_im, fired, spike_step, counter_dev, gather_idx, w_re, w_im,
   decay, cosw, sinw, floor2, n)` that does, **in-place into pre-allocated `_out` buffers**: the rotate/decay, the
   **structured matvec** (gather-scale for permutation; a second kernel or a reshape-sum for bundle), the Mg-free
   crossing detection, and the `spike_step`/`fired` updates. No cuSPARSE → graph-capturable.
2. **Pre-allocated double-buffer + pointer swap** (no per-step allocation/copy): `re_a/re_b`, `im_a/im_b`; the step
   reads `_a`, writes `_b`; swap each step. `prev_im`, `fired`, `spike_step` accumulate in place. `counter` is a
   **device scalar** (`cp.zeros((), int32)`) incremented on-device (the host counter would break capture).
3. **Capture once, replay per op.** Build + capture the n_steps loop as a CUDA graph the first time a given
   (n_neurons, n_steps, matvec-kind) is seen; cache it. The weights/kick live in fixed pre-allocated arrays the
   graph reads — per op, **write the new weights + kick into those arrays, reset counter/fired/spike_step, replay**
   (no re-capture; the graph reads current data at fixed addresses). Read phases after replay.
4. **Wire into the composer's `_resonate`:** when `enable_rf_cudagraph` and the conns are a recognized structure
   (permutation or strided-sum — the composer knows which op it is; pass a `matvec_kind` hint), route through the
   graphed path; else the existing loop. The composer is where the hint is cheap to supply.

## The correctness gate (load-bearing — do BEFORE flipping the default)

- **Bit-identity test:** a new `tests/test_rf_cudagraph.py` asserting the graphed `rf_resonate_steps` returns
  phases equal (to ~1e-9) to the existing loop for the composer's bind / unbind / bundle ops, across seeds. The
  cleanup is argmax so tiny float diffs don't change answers — but assert the phases directly to catch drift.
- **Full conversational suite green** at `enable_rf_cudagraph=True` (the agent opts in): `test_rf_phasor_composer`,
  `test_reconsolidation_update`, `test_brain_conversational_agent` (incl. `test_embedded_clause`),
  `test_multi_turn_*`, `test_multihop_query_chain`, `test_batched_query_scan`.
- **Speedup measured** on a quiet GPU (not while gaming) — expect the prototype's order (≫ the cuSPARSE-bound
  baseline). Bank the number.

## Gotchas (found during de-risk / analysis)

- cuSPARSE/cuBLAS **cannot** be in the capture → the matvec MUST be the custom structured kernel.
- `@cp.fuse` forces a copy-back (new arrays) → use a `RawKernel` writing in-place to pre-allocated buffers.
- The per-step **host counter** (`self._rf_counter`, used in `cp.where(crossed, counter, spike_step)`) must become
  a **device** scalar for capture.
- **Reference safety:** the existing path writes state **in place** (`cp_membrane_potential_v[:] = ...`); the main
  step loop holds references. The graphed path swaps buffers — keep it ISOLATED to `rf_resonate_steps` (the
  composer's tight loop, no external refs), and on exit copy the final state back into the canonical
  `cp_membrane_potential_v`/`u` so callers see the right object.
- Co-residence mask (`_rf_neuron_mask`): the graphed path is for the **no-mask** composer bridges; masked
  (nav+conv co-resident) bridges use the fallback.
- numpy backend: `@fuse` is a no-op there and there's no CUDA graph — gate the graphed path on `is_gpu_backend()`,
  fall back to the loop on numpy (CI stays green).

## Execution order (one focused session)

1. `RawKernel` structured step (gather-scale + strided-sum variants) + bit-identity unit test vs the loop — GATE.
2. Double-buffer + device counter + the graphed `rf_resonate_steps_graphed` (default-off) — bit-identity GATE.
3. CUDA-graph capture/cache/replay around it — bit-identity + speedup GATE.
4. Wire the composer `_resonate` to the graphed path under `enable_rf_cudagraph`; full suite GATE; bank speedup.
5. Flip the default (agent opts in) only after 1–4 green; commit each step; revert = gate flip or `git revert`.

## Why a design doc, not the code, today

The mechanism is de-risked (11×) and the architecture is settled, but this is the deepest core-engine edit in the
arc (custom CUDA + buffer/capture semantics). It wants one clean focused pass with a quiet GPU for the bit-identity
+ speedup gates — not a rushed tail-of-marathon implementation where a subtle core bug (even revertable) would
balloon debugging. This doc makes that pass fast and safe.
