# Route B profile — where the inner-loop time goes

**Date:** 2026-04-24
**Context:** User observed high GPU utilization but almost empty VRAM and asked whether
we could speed up the probes. I estimated Route B (GPU-side readout accumulation,
eliminating per-step CPU↔GPU sync) could give 10-50× speedup. Profile below shows the real
ceiling is much smaller.

---

## Measurement

`research/profile_g9_step.py`, 264-neuron G9 network, 150-step trial.

| Configuration | Time per step | Trial time (150 steps) |
|---|---|---|
| `_run_one_simulation_step` alone (no readout, streamed via CuPy) | **2.41 ms** | 362 ms |
| + per-step CPU sync for motor/hidden firing readout (current g9 behavior) | **4.06 ms** | 609 ms |
| + GPU-side accumulation, single sync at trial end (Route B sketch)         | **3.65 ms** | 547 ms |

## Breakdown

Per step, the 4.06 ms of a current G9 inner-loop iteration splits as:

- **2.41 ms (59%)**: `bridge._run_one_simulation_step()` internals. This is ~50 CuPy
  kernel launches for STDP, plasticity, ion dynamics, synaptic propagation, etc. Most of
  this is kernel-launch **overhead**, not actual GPU compute — the network is so small
  (264 neurons, ~2 KB data) that GPU compute per kernel is microseconds.
- **0.41 ms (10%)**: Per-step CPU↔GPU sync on readout (`cp_firing_states[motor_idx].get()`
  and same for hidden). This is what Route B eliminates.
- **~1.25 ms (31%)**: Python bookkeeping, stimulus-channel setup, runtime-state updates,
  and condition checks. Cannot be directly GPU-batched.

## Why Route B gives only 1.11× speedup

Route B as sketched (keep readout on GPU, sync at trial end) attacks only the 0.41 ms
per-step sync cost. The other 3.65 ms is untouched. Theoretical max speedup: 4.06 / 3.65 =
**1.11×**, matching measured.

## The real bottleneck (and how to fix it, eventually)

**It's CuPy kernel-launch overhead on a tiny network.** Each of the 50 kernels inside
`_run_one_simulation_step` dispatches in ~40-80 µs of Python-side overhead. Actual GPU
work for 264 neurons is microseconds. We're paying launch cost, not compute cost.

Two real fixes, both bigger than tonight's scope:

1. **Kernel fusion inside `bridge._run_one_simulation_step`** — combine the ~50 small
   kernels into a handful of larger ones via `@cp.fuse` or a single custom CUDA kernel.
   Would cut dispatch overhead proportionally.
   *Estimated effort: 1-2 days.* Risk: changes numerical behavior / breaks biological
   benchmarks if the merged kernel has different rounding.

2. **Bigger networks (Route C)** — at 5,000-10,000 neurons, actual GPU compute per kernel
   becomes a few ms, dwarfing the 40-80 µs launch overhead. Dispatch cost becomes a
   rounding error. Silent-motor trap analysis (see `2026-04-24-session-d-part-a.md` §4)
   also recommends larger reservoirs for richer context-dependent activity, so this is
   a double-win when we're ready.
   *Estimated effort: 1 hour to reconfigure + ~1 hour re-tune hyperparameters.*

## Decision for tonight

Route B at 1.11× is not worth the refactor risk (changes RNG dispatch order → could shift
benchmarks bit-for-bit). **Do not integrate.** Keep the profile script on main as
infrastructure for future optimization work.

**Recommendation for Session E or later:** if probe runtime becomes a real blocker, do
Route C (bigger networks, uses VRAM, ~5× speedup by amortizing dispatch) before attempting
kernel fusion. Route C is also mission-aligned via the silent-motor-trap diagnosis.

## What actually worked for tonight

**Route A: parallel seed processes.** The Pavlovian probe had 6 runs (3 seeds × 2
experiments). Running them in pairs of 2 (the parallel driver) vs one-at-a-time (the
serial driver it replaced) gets roughly 2× wall-time savings with zero risk. Infrastructure:
`research/run_pavlovian_parallel.py`.
