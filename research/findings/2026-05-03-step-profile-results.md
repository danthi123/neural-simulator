# Sub-step profile — compute-bound, not Python-bound

**Date:** 2026-05-03 ~14:13 EDT (autonomous overnight)
**Tool:** `python -m research.runners.profile_step` (uses built-in
`GPUConfig.enable_step_profiler` infrastructure that was already
present in the codebase).

---

## Result

Profiling 800 sub-steps on the v2 architecture (5234 neurons, 175k
synapses) **while the fundamentals sweep was running 3 parallel
processes** (4-way GPU contention):

```
[PROFILER] avg/step: 9.38ms |
  t_dyn=2.75ms (29%)    neuron dynamics (Izh)
  t_plast=2.36ms (25%)  STDP + eligibility traces
  t_syn=1.63ms (17%)    synaptic conductance + currents
  t_stp=1.39ms (15%)    short-term plasticity
  t_homeo=0.43ms (5%)   homeostasis
  t_init=0.40ms (4%)    setup
  t_final=0.42ms (4%)   rest (recording, viz, NM)
```

## Correcting for 4-way contention

Single-process v2 runs at ~290 sub-steps/sec = 3.4 ms/step. The 9.38
ms/step under contention represents ~3x slowdown vs uncontested. To
estimate the uncontested per-section breakdown:

Let P = pure-Python time (not slowed by GPU contention), G = GPU
compute time (~3x slowed by 4-way contention).

```
Contended:    P + 3*G = 9.38 ms
Uncontested:  P + G   = 3.40 ms
=>  2G = 5.98  =>  G = 3.0 ms,  P = 0.4 ms
```

So in single-process operation:
- **GPU compute (memory-bandwidth-bound sparse matrix ops):** 3.0 ms = **88%**
- **Python orchestration (kernel launches + attribute access):** 0.4 ms = **12%**

## What this means

**My earlier hypothesis ("Python launch overhead is the bottleneck,
CUDA Graphs would give 2-3x") was WRONG.**

The simulator is dominantly compute-bound at this scale. The sparse
matrix-vector products (synaptic conductance), per-synapse plasticity
updates (STDP + STP on 175k synapses), and per-neuron dynamics
(Izhikevich on 5234 neurons) are all hitting memory bandwidth limits
on the GPU, not waiting for Python.

Reasonable speedup estimates, revised:
| Approach | Old estimate | Actual estimate | Why |
|---|---|---|---|
| CUDA Graphs | 2-3x | **~1.05-1.15x** | Only saves the 12% Python overhead |
| Mega-kernel (hand-tuned CUDA) | 3-5x | ~1.2x | Same — saves orchestration only |
| C++ host loop (same kernels) | 2-3x | ~1.1x | Same — kernels do the work |
| Parallel seeds (3-way) | 1.5-2x | **1.7x ✓** (validated) | Saturates GPU; multiple processes share unused capacity |
| Bigger dt (0.5 → 1.0 ms) | 2x | **~2x** if stable | Halves sub-step count |
| Sparse format optimization | unknown | **1.3-1.5x?** | Reduce memory bandwidth in t_syn + t_stp (32% of compute) |

## Why the simulator is compute-bound

For 175k synapses:
- Each synapse touches 4-8 floats per phase (weight, u, x, eligibility, gain)
- 4 bytes/float × ~6 floats × ~10 phases = ~240 bytes/synapse per sub-step
- 175k synapses × 240 bytes = 42 MB of memory access per sub-step
- 3090 memory bandwidth: 936 GB/s
- Theoretical minimum sub-step time from bandwidth alone: 42 MB / 936 GB/s = **45 μs**

We're at ~3.0 ms compute, so achievable minimum is ~67x faster than
current. But we'd need:
- Coalesced memory access patterns
- ELL or hybrid sparse format (less metadata overhead)
- Fewer redundant kernel passes (mega-kernel design)

These are deep CUDA optimizations, multi-week implementations.

## Where the practical wins are

Given this finding, the right priorities are:

1. **Parallel seeds (already done)** — 1.7x at batch level. Free.

2. **Bigger dt** — halve sub-steps. Need to verify Izhikevich stability
   at dt=1.0ms. Quick to test (1 seed, 30 min).

3. **Reduce stim/reset window sizes** — the current eval uses 200 stim
   sub-steps + 100 reset between each trial. Halving gives ~25% wall-clock
   savings on eval. Need to verify accuracy isn't degraded.

4. **Profile finer within t_syn and t_plast** — the sparse mat-vec and
   STDP phases account for 42% of compute. Specific optimizations
   (cached COO matrix, kernel fusion within phase) might give 1.2-1.3x
   on those phases = 1.1-1.15x overall.

5. **CUDA Graphs (deferred)** — modest savings (~1.15x) for moderate
   effort (~2 days). Lower priority than 1-4 above.

## Caveats

- Profiling under 4-way contention may overstate compute time relative
  to Python time (since GPU contention doesn't slow Python). The
  scaling correction assumes uniform 3x compute slowdown, which is
  approximate.
- A clean single-process profile (after the sweep finishes) would give
  more precise numbers. Worth re-running.
- At larger network sizes (50K+ neurons), compute-bound dominance
  would be even stronger. Smaller networks (<1K neurons) might
  actually be Python-bound.

## Tool

`research/runners/profile_step.py` — re-runnable any time. Uses the
existing `GPUConfig.enable_step_profiler` flag and prints per-section
breakdowns every 500 steps.
