# CUDA Graphs prototype — scope and analysis

**Date:** 2026-05-03 (autonomous overnight)
**Context:** User asked for analysis of CUDA Graph capture as a
speedup option. The step profile (see
`research/findings/2026-05-03-step-profile-results.md`) revised the
expected speedup downward, but the prototype is still worth scoping
so we can decide cost-benefit cleanly.

---

## What CUDA Graphs do

A CUDA Graph captures a sequence of kernel launches into a graph
object. Subsequent "launches" of the graph submit all kernels to the
device with one host-side call instead of N. The savings come from
amortizing the per-launch overhead (~5-10 μs each in CUDA, plus
Python dispatcher overhead in CuPy).

```python
# Before:
for step in range(N):
    fused_stp_kernel(...)
    fused_syn_kernel(...)
    fused_dyn_kernel(...)
    # ... 7+ more kernels
    # Each launch: ~10 μs CUDA + ~50-100 μs Python dispatcher = ~50-100 μs
    # 10 kernels × 100 μs = 1 ms of overhead per step

# After (graph capture):
g = cp.cuda.graphs.Graph()
with g.capture():
    fused_stp_kernel(...)
    fused_syn_kernel(...)
    # ... etc — same code, but recorded
graph_exec = g.instantiate()

for step in range(N):
    graph_exec.launch()
    # One launch: ~10 μs total
```

## Realistic speedup

Per the profile (single-process estimate):
- GPU compute: 3.0 ms/step (88%)
- Python orchestration: 0.4 ms/step (12%)

Of the 0.4 ms Python overhead, roughly:
- ~0.1 ms is kernel-launch overhead (10 kernels × 10 μs)
- ~0.3 ms is other Python (attribute access, conditionals, dictionary
  lookups, the `_prev_any = bool(self.cp_prev_firing_states.any())`
  kind of stuff)

CUDA Graphs would eliminate the 0.1 ms launch overhead. So:
- Best case speedup: 3.4 ms → 3.3 ms = **1.03x**

The remaining 0.3 ms of Python work happens BETWEEN logical phases
(branch decisions, computing slice indices, applying gates) and
can't be captured into a graph because it depends on per-step state.

**Verdict: CUDA Graphs alone gives ~3% speedup. Not worth the effort.**

## What CUDA Graphs would help with — only if combined with kernel restructuring

CUDA Graphs become high-ROI ONLY if we restructure to:

1. **Eliminate per-step Python branches.** Move conditional logic
   (e.g., "if STP enabled, do X; else skip") to graph-time, not
   step-time. Compile multiple graph variants for different config
   combinations.

2. **Pre-compute ALL slice indices and tensor views.** Currently
   `cp_stp_u[:actual_nnz_stp]` is recomputed every step. With graph
   capture this would be baked into the graph at capture time,
   requiring shapes to be FIXED for the duration.

3. **Eliminate any `.get()` or `.item()` calls.** These force GPU→CPU
   sync and break graph capture. Currently `bool(self.cp_prev_firing_states.any())`
   does exactly this. Need to keep the result on GPU and use it
   conditionally via masking, not Python branching.

4. **Fix-shape inputs.** External arrays like `cp_external_input_current`
   need to be in-place modified rather than reassigned, because graph
   captures reference specific tensors at specific addresses.

These restructurings (1-4) are themselves substantial work — maybe
1-2 weeks. AND they have correctness risks: removing
`_prev_any = bool(...)` removes the early-exit optimization that
skips empty work paths, potentially making things SLOWER if firings
are sparse.

## Alternative: profile-driven micro-optimizations

The profile shows the 4 biggest hotspots:
- t_dyn (2.75 ms contended ≈ 1.0 ms uncontended): neuron dynamics
- t_plast (2.36 ms ≈ 0.87 ms): STDP + eligibility
- t_syn (1.63 ms ≈ 0.6 ms): synaptic conductance
- t_stp (1.39 ms ≈ 0.51 ms): short-term plasticity

Focusing on EACH of these for 10-20% speedups (algorithmic, not
architectural) would give 1.1-1.2x compounded with much lower risk
than CUDA Graphs.

Specific candidates:
- **t_syn**: the COO conversion `cp_connections.tocoo()` is cached but
  the cache check is per-step. Verify cache hit rate is 100%; the
  fall-through cost is reportedly 40-400 ms.
- **t_plast**: STDP currently iterates per-pathway. Could batch all
  plastic-pathway weight updates into one fused kernel.
- **t_stp**: Tsodyks-Markram update has separate decay + recovery
  kernels. They were already merged into `fused_stp_decay_recovery`
  — verify it's actually being used.
- **t_dyn**: The Izh kernel reads `total_input_current_pA` which is
  built up over the step from many `+=` ops. Each `+=` is a separate
  kernel. Could fuse current accumulation with dynamics.

## Recommendation

**Do not invest in CUDA Graphs as primary speedup.** Profile shows
~3% expected gain.

**Do** invest in:
1. Parallel seeds at batch level — already done, 1.7x ✓
2. Bigger dt validation — quick test, ~2x
3. Profile finer within hot phases — algorithmic micro-opts, 1.1-1.2x
4. Sparse format alternatives — research project, 1.3-1.5x potential

If 1-4 are exhausted and we still need more speed, THEN look at:
- CUDA Graphs (after restructuring for shape fixity)
- Mega-kernel rewrite
- C++/CUDA inner loop replacement

## Why I was wrong before

In our earlier discussion I said CUDA Graphs would give 2-3x and that
the simulator was launch-overhead-bound. The actual profile shows
otherwise. The mistake was over-indexing on PyTorch experience where
the Python loop dominates for small models. Spike sims with sparse
plasticity are different — they're memory-bandwidth-bound at this
scale.

Lesson: profile before recommending. The instrumentation was ALREADY
in the codebase (`GPUConfig.enable_step_profiler`); I should have
turned it on first.
