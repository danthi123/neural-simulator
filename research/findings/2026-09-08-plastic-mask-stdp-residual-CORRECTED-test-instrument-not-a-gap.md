---
type: finding
status: corrected
date: 2026-09-08
mechanism: plastic-mask
---

# Vikunja #203 follow-up: the "STDP-update path is NOT covered" residual was a test-instrument artifact, not a plasticity-mask gap — test made hermetic

**Corrects** `research/FAILURE_LOG.md`'s 2026-09-02 row ("PARTIALLY RESOLVED... RESIDUALS (open): (1) the
STDP-update path is NOT covered — the pre-existing `tests/test_plastic_mask.py::...freezes_fixed_synapses`
failure indicates the same unenforced-mask gap in the STDP kernel"). That residual (1) is **retracted**: the
STDP kernel already applies `cp_synapse_plastic_mask` unconditionally and correctly. The cited test's failure
was a test-process-isolation bug, now fixed. Artifact:
[`research/findings/raw/_plastic_mask_stdp_recheck/verify.json`](research/findings/raw/_plastic_mask_stdp_recheck/verify.json).

## Background

`BRAIN_ENFORCE_PLASTIC_MASK` (merge `509c7137b`, 2026-09-02) fixed a real bug: the runtime Hebbian LTP/decay/clip
path never consulted `cp_synapse_plastic_mask`, only the named `plasticity_gate`. The same FAILURE_LOG entry
logged three residuals; residual (1) claimed the STDP-update path had the same gap, citing
`tests/test_plastic_mask.py::test_plastic_mask_freezes_fixed_synapses` as evidence — that test failed when run
after `tests/test_enforce_plastic_mask.py` in the same pytest session.

## What was actually true

1. **Direct code read** (`sim/bridge.py:10238-10256`, inside the non-branchless STDP update block) shows the
   STDP kernel already gates every weight write on `cp_synapse_plastic_mask`, unconditionally — no flag needed,
   matching the documented BDSP/BTSP behavior. This was true before, during, and after the Hebbian fix; it was
   never the hole the Hebbian bug closed (`verify.json` → `code_read_confirmation`). <!--derived-->
2. **Root cause of the test failure**: `sim/bridge.py:44` resolves numpy-vs-cupy **once**, at the first time
   `sim.bridge` is imported in a process (`cp, _backend_name = get_backend()`), and every later
   `SimulationBridge` in that process reuses that binding regardless of the `SIM_BACKEND` env var. `tests/
   test_enforce_plastic_mask.py` does a module-level `os.environ.setdefault("SIM_BACKEND", "numpy")` (its own,
   legitimate, numpy-determinism scenario) — collected alphabetically before `test_plastic_mask.py`, it is
   typically the first thing in the session to import `sim.bridge`, permanently pinning the whole process to
   numpy before `test_plastic_mask.py`'s own `import cupy as cp` line ever runs.
3. **Under the numpy backend, the exact tiny seeded network this test used (10 Izhikevich neurons, seed=7,
   Poisson drive) never spikes at all** — `fired_any=False` in BOTH the all-plastic and all-fixed configs, 0/10
   neurons ever cross threshold over 500 steps (measured; `verify.json` →
   `same_scenario_under_forced_numpy_same_seed`). This is a cross-backend RNG/heterogeneity-divergence effect at
   tiny scale (same class as the 2026-06-09 N9 CuPy-vs-numpy firing-margin finding, direction reversed), not a
   plasticity bug. With zero spikes, STDP never fires in either arm, so the test's "plastic weights didn't move"
   failure message was **literally true and completely uninformative** about the mask.
4. Under the intended cupy backend, both arms fire correctly and the mask works exactly as designed: the fixed
   arm's weights are frozen (`max_change=0.0` over 500 steps) and the plastic arm's weights move
   (`max_change=0.214`, `verify.json` → `scenario_under_forced_cupy`). <!--derived-->

## The fix

`tests/_plastic_mask_stdp_scenario.py` (new) holds the drive-and-measure logic and asserts
`get_backend()[1] == "cupy"` before building anything. `tests/test_plastic_mask.py`'s
`test_plastic_mask_freezes_fixed_synapses` now invokes it via `subprocess.check_output([...], env={"SIM_BACKEND":
"cupy", ...})` — a **fresh process**, so `sim.bridge`'s first-ever import in that process sees cupy forced,
regardless of what any sibling test did earlier in the parent pytest process. The other two tests in the file
(`test_no_mask_means_all_plastic`, `test_mask_aligned_with_csr_order`) only check the mask array's structure, not
live spiking dynamics, so they were never backend-sensitive and are unchanged.

Verified robust across every ordering that previously reproduced the bug (`verify.json` → `pytest_after_fix`):
natural alphabetical order, reversed order, and reversed order with an explicit `SIM_BACKEND=numpy` poison — all
3×5 tests pass. An in-process `autouse` fixture that only sets the env var was tried first and **rejected**: it
cannot undo `sim.bridge`'s module-level backend binding once any earlier test has imported it, so it "fixed" the
originally-observed ordering but broke `test_enforce_plastic_mask.py` when the file order was reversed — a
moving-target false fix, not a real one. The subprocess approach is immune to import order by construction.

## What remains open (unchanged by this finding)

FAILURE_LOG's other two 2026-09-02 residuals for `BRAIN_ENFORCE_PLASTIC_MASK` still stand: (2) BCM/rate-window/
branchless Hebbian rule variants are structurally guarded but not individually SHA-tested; (3) flip-to-default is
OWNER-GATED. Neither is addressed here. No production code changed; no default flipped.

## Honesty boundary

Functional/mechanistic claim only (a kernel gating check, a test-process property) — no cognitive or
self-model claim.
