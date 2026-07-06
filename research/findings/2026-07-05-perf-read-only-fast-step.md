# Perf — `read_only_fast_step`: remove the two per-step device→host syncs on the inference loop (byte-identical, guarded)

**Date:** 2026-07-05
**Files:** `sim/config.py` (`read_only_fast_step` field), `sim/bridge.py` (`_run_one_simulation_step` guard), `tests/test_read_only_fast_step.py`.
**Research gate:** the conversational-latency perf gate — root cause = the step loop is launch-bound on per-step device→host synchronizations, NOT a megakernel-rewrite need.

## The bottleneck (root cause, confirmed)

`_run_one_simulation_step` (`sim/bridge.py`) issues TWO device→host synchronizations every step:
- `:5904` `_prev_any = bool(self.cp_prev_firing_states.any())` — cached once, gates STP / synaptic-propagation / Hebbian skip-fast-paths.
- `:6845` `_fired_any = bool(spike_count_gpu > 0)` — gates pulse-timer / STDP / stats skip-paths.

Each `bool(cupy_reduction)` blocks the GPU pipeline (a full device→host transfer + host branch) every step. On the spiking
inference loops (the A→W word read-out, the parser, the WTA, the reservoir — all Izhikevich stepping through this method)
these two syncs are pure overhead: every use of the two flags is a skip-fast-path that produces ZERO contribution on a
genuinely zero-spike step.

## The fix — opt-in, guarded, byte-identical

`read_only_fast_step` (CoreSimConfig, default OFF) forces both flags True, skipping the two syncs. It is **guarded to
activate ONLY when the step is genuinely read-only** (no Hebbian / STP / homeostasis / STDP / structural plasticity /
reward modulation, and no experiment running). With the guard the flag is **byte-identical UNCONDITIONALLY**: it is inert
unless the step is read-only, and on a read-only step forcing the flags True only does redundant zero-work.

**The test caught the boundary (systematic-debugging).** The first version forced the flags True with no guard; the
byte-identity test with plasticity ON FAILED — a plasticity-gated block consumes RNG, so running it on a forced-True
zero-spike step diverges the RNG stream. Root cause found → fixed by the read-only guard (not by weakening the test). The
test now proves: (a) plasticity-off inference is bit-identical with the flag on vs off; (b) with plasticity on the flag is
inert → bit-identical because it never activates; (c) default is off.

## Measured

GPU (RTX 3090), 2000-neuron Izhikevich inference loop, plasticity off: **1.09× (0.153 ms/step saved, 560→612 steps/s)**.
Honest scope: modest — the two syncs are a small fraction of this heavier step; a lighter loop sees a larger relative win.
The gain is free (default-off, byte-identical) and compounds across every inference step in the spiking conversation loops.
It does NOT touch the RF resonate loop (its own path; the `enable_rf_cudagraph` megakernel is the lever there).

## Files
- `sim/config.py` — `read_only_fast_step: bool = False` + doc.
- `sim/bridge.py` — the guarded flag in `_run_one_simulation_step` (default path byte-identical by construction).
- `tests/test_read_only_fast_step.py` — 3 tests (plasticity-off bit-identity; plasticity-on guarded-inert; default off).
