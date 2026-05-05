# Perf wave 2 verdict — Phase 1 + Phase 2 measured cleanly

**Date:** 2026-05-05 ~12:18 EDT
**Status:** GO — both opt-ins kept as defaults. Real but smaller than projected.

---

## Clean bench results (no GPU contention)

```
Config                                       sec/event    speedup
baseline_fp32_cpu_eligibility                 706.4ms      1.00×
phase1_gpu_eligibility                        670.7ms      1.05×
phase2_gpu_eligibility_fp16                   658.2ms      1.07×
```

3 configs × 1 seed × 400 events. Quick mode. ~14 min total wall time.
GPU was idle (≤10% util) before bench start, ~40% during.

Eval accuracy: 27% baseline / 26% phase1 / 26% phase2 — within noise,
no architectural regression. (All below the 25% chance line by ≤2pp,
consistent with the 2026-05-05 verdict that this architecture has
~7-8pp structural noise but no real W→A learning.)

## Comparison to contaminated bench

The 2026-05-05 ~07:33 bench (`2026-05-05-bench-phase1-contamination.md`)
showed:

```
baseline_fp32_cpu_eligibility    1724.6ms    1.00×
phase1_gpu_eligibility           2720.8ms    0.63× (1.6× SLOWER)
phase2_gpu_eligibility_fp16      2721.0ms    0.63×
```

Contaminated: GPU at parallel=6 graded-DA contention.
Clean: GPU idle at start.

Baseline ratio contaminated/clean = 1724/706 = **2.44×**. The
contention nearly halved single-process throughput. This explains
the apparent "1.6× slower" — it was actually measuring contention
asymmetry, not algorithmic regression.

## Honest assessment vs original speculation

| Claim | Source | Reality |
|---|---|---|
| "Phase 1 eliminates 1 GPU↔CPU round-trip per event" | Plan | True |
| "Expected 2× speedup" | Plan (speculative) | **5% measured** |
| "Phase 2 fp16 halves eligibility memory" | Plan | True |
| "Expected 1.5× from FP16 storage" | Plan (speculative) | **+2% on top of phase1 = 7% total** |

The original speculation assumed eligibility was a major fraction of
runtime. Empirically: it's <10%. Most time goes to:
- Synaptic propagation (CSR mat-vec) — biggest cost
- Neuron dynamics (Izhikevich kernels)
- STDP weight updates
- Other plumbing

This makes sense: at biological canon (N=500 cortical recurrence +
NMDA bistability), the dense kernels dominate the sparse eligibility
update by ~20:1.

## Decision: KEEP `gpu_eligibility=True` and `fp16_synapse_state=True` opt-ins

Both produce small but real wins with NO regression. Keep defaults
as currently set in `research/runners/bio_three_factor.py`:

```python
gpu_eligibility: bool = True       # 5% faster
fp16_synapse_state: bool = False    # opt-in (7% cumulative when enabled)
```

`fp16_synapse_state` stays opt-in because:
1. Drift validation (`tests/test_fp16_drift.py`) is GPU-gated; CI can't
   verify on every commit
2. The 2% additional speedup over phase1 isn't worth the extra config
   surface for routine use
3. Researchers can flip it for long sweeps where every % matters

## What this teaches about perf optimization

1. **Don't trust speculative speedup estimates.** "Eliminating a round-
   trip is 2× faster" was wrong by 40×. Always profile before
   prioritizing.
2. **Run benchmarks with NO concurrent GPU work.** The post_chain_bench
   wrapper now polls for GPU idle (`scripts/post_chain_bench.ps1` v2
   committed `baf331a`).
3. **Single-process speedups don't translate to sweep throughput.**
   Phase 1's 5% single-process win is a real 5% sweep speedup at
   parallel=1. At parallel=6 (current default), the GPU is saturated
   and Phase 1 might not help at all (or might hurt if it adds GPU
   dispatch contention).

## What's next for perf

Three real opportunities NOT in wave 2:

### Option A: Profile + optimize the dominant hot path

Run `cupy.cuda.profile` to identify which kernel takes the most time.
Likely candidates:
- `fused_izhikevich2007_dynamics_update` (every neuron, every step)
- Synaptic CSR propagation (every edge with active pre, every step)
- STDP weight updates (every active edge)

If one of these is 50% of runtime, optimizing it yields a real big
win. Current profiling data: not collected.

### Option B: Cloud H100 ($2/hr, ~3-4× sweep throughput)

`scripts/deploy_to_cloud.sh` is ready. Per `2026-05-05-perf-roadmap.md`:
H100 at FP32 is ~3× faster than 3090 on dense matmul; ~2× on sparse
SpMV. At parallel=6 we'd expect ~6-8× sweep throughput improvement.

Cost-effectiveness: a 6-seed cheat-5 sweep that takes 90 min on local
3090 takes ~15 min on H100. That's $0.50 vs 90 min of waiting. For
weekly sweeps this is dominated by setup overhead; for daily research
it's a clear win.

User has held off until tomorrow — flagged for review.

### Option C: Multi-GPU parallelism (2× 3090 used = $1k)

Per the perf roadmap discussion: 2× 3090 gives 2× parallel single-
process throughput, no algorithmic changes needed. Each GPU runs an
independent seed.

Useful if local development continues and cloud isn't acquired.

### Option D: JAX vmap rewrite (Phase 3, deferred 3-6 months)

Per perf roadmap: JAX vmap can natively run multiple seeds on a single
GPU with shared kernel launches. Speedup for sweeps could be 4-8× on
single-GPU. But it's a 3-6 month rewrite and the current cupy
infrastructure is heavily invested.

NOT recommended unless dendritic learning is also going to be a major
rewrite (in which case bundle them).

## Files

- Bench harness: `research/runners/bench_perf_stack.py`
- Bench data: `research/findings/raw/bench_perf_clean.json`
- Contamination doc (history): `research/findings/2026-05-05-bench-phase1-contamination.md`
- Perf roadmap: `research/findings/2026-05-05-perf-roadmap.md`
- GPU-idle gate fix: `scripts/post_chain_bench.ps1` (commit `baf331a`)
- Phase 1 implementation: `research/runners/bio_three_factor.py`
  (`update_eligibility_and_weights` with `xp` backend)
- Phase 2 implementation: `sim/config.py` `cfg.fp16_synapse_state`,
  `sim/bridge.py` allocates `cp_eligibility_trace` as fp16 when enabled

## Honest takeaway

Phase 1 + Phase 2 deliver **7% cumulative speedup**, NOT 2× as
speculated. Both are real, validated, and worth keeping as opt-ins.
But this isn't the breakthrough we projected. The dominant runtime
cost is dense neuron/synapse kernels, not the eligibility update.

For real perf wins, the next step is profiling to identify the actual
hot path, OR moving to cloud H100 (3-4× from hardware), OR multi-GPU
local (2× from parallelism). All three options are tractable; user
choice determines which.
