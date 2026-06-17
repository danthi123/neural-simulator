# Scaling profile on the RTX 3090: the wall is per-op LATENCY (orchestration), not VRAM — and it's software-fixable

**Date:** 2026-06-17 (owner-requested: local-only vs cloud for small-LLM-scale conversation on the RTX 3090)
**Status:** profiled. **VRAM is not the near-term constraint; conversation LATENCY is** (~160 ms per composer op,
98% of it the resonate loop's per-step kernel-launch overhead). The latency is a software optimization, not a
hardware limit — so **local-only is viable at small-LLM scale once the op overhead is fixed; cloud is not needed
for VRAM at this scale.**
**Runners:** `research/runners/profile_scaling_3090.py`, `research/runners/_phaseB_composer_op_breakdown.py`
**Raw:** `research/findings/raw/_composer_op_breakdown.json`

## The question

Owner: before any cloud spend, what are the training and real-time-conversation slowdowns at small-LLM scale on
this machine (RTX 3090, 24 GB, 20 cores)? The prior assumption was that VRAM caps the vocabulary. This profile
tests that.

## VRAM — not the wall at small-LLM scale

A brain-based concept cortex is Izhikevich neurons + a sparse synapse CSR. Synapse math (the dominant term):
a 20k-neuron bridge at 2% recurrent ≈ 8M synapses ≈ ~100–130 MB. The fully-neural concept cortex scales ~linearly
in neurons, and a small-LLM-scale vocabulary (~thousands of concepts ≈ a few hundred-thousand neurons across the
multi-bridge cortex) is on the order of a **few GB** — comfortably inside 24 GB. (The current production composer
is even lighter: it holds concept codes as numpy + small per-op bridges, so it barely touches VRAM at all.)
**⇒ VRAM only becomes the constraint at much larger-than-small-LLM scale or for massive parallelism.** The "VRAM
caps vocab" worry is real, just not near-term.

## Latency — THIS is the wall (and it's launch-bound, not compute- or VRAM-bound)

Measured on the production `RFPhasorComposer` (D=512, GPU):

| op | cost | note |
|---|---|---|
| one bind / unbind | **~160 ms** | the unit of conversational work |
| who/what query, KB=50 | ~0.8 s/turn | scans the fact store, ~op × facts-checked |
| query vs D (128→2048) | 0.83 → 0.87 s | **flat** — composer dimension barely matters (compute is trivial) |
| query vs KB (10→100) | scales with KB | linear scan; an abstention at KB=1000 is minutes (blew the profiler timeout) |
| parser | ~710 ms/parse | same root cause (runs the bridge per-step) |

**Single-op breakdown (the decisive diagnostic):**

| sub-step | cost | share |
|---|---|---|
| **`rf_resonate_steps` (208-step loop)** | **162.3 ms** | **97.7%** — 780 µs/step |
| rf_set_complex_weights (GPU weight rebuild) | 3.6 ms | 2.2% |
| conns build / rf_kick / rf_read_phases | <0.3 ms | ~0% |

The entire op cost is the 208-step resonate loop. `_rf_advance_one` issues ~15–20 separate CuPy kernel launches
per step (rotate/decay, 4 sparse complex matvecs, zero-crossing detection, masked writes), looped 208× in Python
→ ~3,000–4,000 sequential tiny kernel launches per op. At ~40–50 µs launch overhead each, that's the 780 µs/step.
The 3090 is ~99% idle during this — it's **launch-bound**, the textbook signature of many tiny sequential GPU ops.

## The fix (software; large headroom)

In leverage order:
1. **Fuse / CUDA-graph the resonate loop.** Capture the 208-step `_rf_advance_one` loop as ONE CUDA-graph launch
   (requires making the step pure-GPU: the per-step host counter → a GPU scalar, constants hoisted). Collapses
   ~3,000–4,000 launches → ~1 → the 208 steps run at compute speed (~1–5 ms). Expected **~30–100× per op.**
2. **Batch the store scan.** A query unbinds the cue against every fact one at a time; stack all KB composites
   into ONE resonate (block-diagonal weights) → KB ops → 1 op. Expected **~KB×** at scale.
3. **Index the fact store by cue** (agent+action) so a query touches only candidates → O(KB) → ~O(1) retrieval,
   and bounds the abstention-scan worst case.
4. **Fuse `_rf_advance_one`'s element-wise ops** into 1–2 `@fuse()` kernels (a simpler partial win if the full
   graph refactor is deferred).

Stacked, a turn plausibly drops **~0.8 s → ~10–25 ms = real-time**, no hardware change.

### Proof-of-speedup prototype (DONE) — CUDA-graph gives 11× per op, demonstrated

`research/runners/_phaseB_resonate_cudagraph_prototype.py` (raw `_resonate_cudagraph_prototype.json`):
- A naive CUDA-graph capture of the loop **fails** — CuPy raises "calling cuBLAS API during stream capture is
  currently unsupported", and the bridge stores the RF weights as a sparse CSR (`cp_rf_w_re @ z` = cuSPARSE), so
  the production op hits the same wall. **Precise framing** (per the optimization-literature review,
  `2026-06-17-snn-vsa-gpu-optimization-literature.md`): this is a **CuPy capture-path** limitation (the
  library call's device-host sync during capture), *not* a categorical CUDA-graph limitation — NVIDIA's own
  forums note cuBLAS is graph-capturable in most C++ cases (exceptions: host-buffer output / host-pointer scalar
  mode). The fix below is unaffected either way: don't put the library call in the graph at all.
- **The fix that works:** the composer's bind/unbind weights are a near-diagonal permutation (post-neuron `D+k`
  ← pre-neuron `k` × phase), so the matvec is an **elementwise gather-scale** — no library call, fully
  graph-capturable. With that, capturing the 208-step loop as ONE graph and replaying gives **107 ms → 9.8 ms
  per op = 11×, measured** (n=1024). The residual 9.8 ms is the actual 208-step compute; a further single-kernel
  fusion (or shorter period) shrinks it more, and the KB-scan batching (lever 2) compounds on top.
- ⇒ the production fix is **justified and de-risked**: refactor `_rf_advance_one` to a graph-able form (diagonal
  elementwise synapse for the structured composer weights, the per-step counter as a device scalar, pre-allocated
  scratch) + a graphed `rf_resonate_steps` fast path. This is a protected `sim/` edit (byte-reviewed; the
  `tests/test_rf_*` suite asserts bit-identical RF dynamics; default-preserving for non-diagonal/general weights).

## Verdict (local vs cloud)

- **Local-only is viable for a small-LLM-scale conversational agent on the 3090** — VRAM fits it, and the latency
  wall is a software optimization, not a hardware limit.
- **Cloud is not needed for VRAM at small-LLM scale**; it would only buy headroom for much larger vocab or for
  massive-parallel training throughput.
- **The orchestration-overhead fix (lever 1) is the highest-value near-term engineering arc** for a usable
  real-time "converse with the artificial life" demo and for scaling the fact store. It is separable from the
  biology research and does not block it (validation runs at small KB where ~1 s/op is tolerable).

## Honest caveats

VRAM deltas were measured via `nvidia-smi` and are pool-noise-dominated (the synapse-math estimate is the
reliable basis). Training throughput (~270–650 bridge steps/s at 2k–20k neurons) was measured but not yet
extrapolated to a full corpus; with wall-clock not a concern (owner), training time is a secondary axis. The
profiler timed out on the largest KB/cortex points (capped wall-clock), but the decisive op-breakdown +
VRAM-math + latency-shape are complete.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_composer_op_breakdown   # the decisive 160ms breakdown
SIM_BACKEND=cupy python -u -m research.runners.profile_scaling_3090            # VRAM + latency sweeps
```
