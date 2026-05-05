# Performance roadmap — honest assessment + concrete next steps

**Date:** 2026-05-05 ~03:00 EDT
**Trigger:** User observed GPU rarely above 50% utilization with VRAM
peaking at ~5GB out of 24GB. Rightly questioned whether the sim is
actually optimized.

---

## Honest current-state

| Metric | Reading | What it means |
|---|---|---|
| Wall-time on CUDA kernels | 88-91% | We dispatch to GPU efficiently |
| GPU utilization (Windows Task Manager Compute_0) | 30-50% | But the GPU is mostly idle during those kernels |
| VRAM peak (parallel=2-3 bio scale) | ~5 GB / 24 GB | 19 GB of headroom we're not using |
| Tensor Cores | unused | We have FP32 sparse mat-vec; tensor cores are FP16/BF16 dense matmul |
| PCIe transfers per event (3-factor v3) | ~12 MB before Phase 1 | Now zero (Phase 1 GPU port shipped) |

**Translation:** the simulator IS GPU-bound (most time in kernels) but
the GPU is RUNNING SLACK during those kernels. Plenty of room to push
more parallel work in.

## Why my earlier "88-91% on GPU" framing misled

That metric measures **time in kernels vs time in Python**. It doesn't
measure **how much of the GPU's compute is being used**. A program can
spend 100% of its time in GPU kernels that use 1% of the GPU. Both
true. The user's observation of 30-50% Task Manager utilization is the
saturation metric and that's what matters for "how much room is left."

## Path to faster iteration (priority order)

### Tier 1 — cheap and high-leverage (do FIRST)

**1.1 Bump `parallelism` in YAMLs from 2 to 6.** Memory permits;
GPU is underutilized; literally just changing one number triples
sweep throughput. **Done this session** for `bio_three_factor.yaml`
and `bio_b3_validation.yaml`. Expected wall-clock: chain finishes in
**3 hours instead of 9** if you re-launch with the new value.

**1.2 GPU-port three-factor (Phase 1, this session).** Eliminates
per-event 6 MB PCIe round-trip on the 3-factor runner. **Shipped
commit `a3187e4`** with 7 unit tests passing. Expected speedup ~2×
on the 3-factor runner specifically.

### Tier 2 — modest wins, scope manageable

**2.1 FP16 mixed-precision for synaptic state (Phase 2, this session).**
Scoped narrowly to `cp_eligibility_trace` (and eventually
`cp_connections.data`). Voltages stay FP32 (precision matters near
threshold). Honest expected gain: 1.05-1.15× on bio scale. Memory
savings: ~10-20 MB per process (not the bottleneck). Real value:
halves bandwidth on the spike-propagation kernel which IS bandwidth-
bound for sparse mat-vec.

Status: `cfg.fp16_synapse_state` flag added; eligibility allocation
plumbed; `tests/test_fp16_drift.py` scaffolded but GPU tests gated
until current chain frees the GPU. **Not yet validated under load.**

**2.2 Async CUDA streams.** Currently each `bridge._run_one_simulation_step()`
synchronizes between kernel launches. Moving to a stream-per-pathway
model could overlap independent kernel runs. Expected ~1.2×.
Implementation: 1-2 weeks, touches bridge.py heavily.

**2.3 ELL-format sparse matrix.** CSR coalesces poorly on GPU when
nnz/row varies. ELL pads to fixed nnz/row → better coalescing. ~1.3-1.5×
on the t_syn step. CuPy supports; conversion is mechanical. ~1 week.

### Tier 3 — major rewrites, big payoff if pursued

**3.1 JAX with vmap for batched seeds.** This is the project's biggest
latent perf opportunity. Currently parallel=N means N separate Python
processes, each with own CUDA context. JAX `vmap` lets ONE process
batch-simulate N seeds in vectorized form — kernels become
`(batch, neurons)`-shaped, GPU compute saturates, no per-seed overhead.

Expected gain: **2-3× on multi-seed sweeps** (24 runs → 8-12 runs of
batched compute). Cost: rewriting `sim/bridge.py` from CuPy to JAX is
3-6 months of engineering. Touches every kernel, every fused op,
every type signature.

Reasonable approach if pursued:
1. Start with a minimal JAX prototype of `_run_one_simulation_step` for
   the Izhikevich path only (ignore HH, AdEx, structural plasticity).
2. Validate numerical equivalence on 1-seed (no batching).
3. Add vmap to batch over seeds.
4. Benchmark vs cupy. If gain confirms, port the rest.
5. Keep cupy implementation for fallback.

**3.2 GeNN (purpose-built SNN simulator).** If we ever scale to 100K+
neurons (whole-mouse-cortex). At our 5 K scale, GeNN would be
comparable to cupy-tuned code. Don't pursue at current scale.

**3.3 Raw CUDA C++ kernels.** Replaces `@cp.fuse()` with hand-tuned
CUDA. Realistic gain 1.3-1.5×. Cost: 6+ months. Not worth at this scale.

## Hardware vs software comparison

If we bought the suggested **2× RTX 3090** ($1.5k), we'd get 2× parallel
throughput with no code changes. If we did the **Tier 1 software work**
(parallel=6 + Phase 1 GPU port), we get 3× throughput **for free** in
the same hardware. The software wins are bigger than the hardware win
at our scale.

The 4090 + Tier 1 software combined would give ~5-6× wall-clock vs
current. That's the realistic maximum for "faster iteration on the
existing science" without a major framework change.

## Recommended order of execution

1. **Now:** Validate Phase 1 (GPU port) + parallel=6 on next sweep.
   Measure actual speedup vs current parallel=2.
2. **Today/tomorrow:** Validate FP16 (`tests/test_fp16_drift.py`) once
   GPU is free. If drift < 1mV at 1000 steps, ship it.
3. **This week:** ELL sparse format — concrete, scoped.
4. **Next quarter (if scaling matters):** JAX vmap prototype on
   Izhikevich path. Validate equivalence + benchmark. Decide whether
   to commit to full port.
5. **Hardware:** if budget allows, 2nd RTX 3090. If only one slot
   available, 4090.

## The biological-time floor

No optimization can shorten the **simulated time** that has to elapse
for biology to learn. 4000 events × 170 ms simulated = 680 sec of
biological time. At infinite GPU speed that's still 680 sec of wall
time. Currently we run at ~85× faster than realtime (8 sec wall per
seed). Doubling GPU = 170× faster. JAX-vmap with 6-seed batch = 500×
faster (effective 1.6 sec wall per seed equivalent). The hard floor
is the dynamics, not the compute.

The fastest "iteration" path is therefore:
- Smaller architecture for design iteration (cheaper)
- Bio scale only for final validation (expensive but unavoidable)
- Multi-seed via batching (vmap) once architecture is fixed
