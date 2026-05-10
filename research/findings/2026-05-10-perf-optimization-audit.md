# Performance optimization audit — pre-cloud-deploy menu

**Date:** 2026-05-10 (overnight, 64-word smoke observation)
**Trigger:** User directive — "continue focusing some efforts on optimization
and speed ups prior to moving to cloud compute to maximize the value we
get out of it."
**Observation:** 64-word @ n_motor=6000 takes ~700s per chunk-sleep cycle
vs 16-word @ n_motor=2000 at ~145s. ~5× slower per cycle. Per-step inner
loop dominated by synapse-side ops at 30M+ synapse arch.

---

## Current optimization stack (already shipped)

From CLAUDE.md "Recommended configuration" + `research/findings/2026-05-04-perf-speedup-stack.md`:

| Optimization | Status | Effect |
|---|---|---|
| `dt=1.0` (Izhikevich) | DEFAULT in bio_three_factor + trainer | 2× over dt=0.5 |
| `parallel-3` (multi-process GPU sharing) | YAML-only, not in chat runners | 1.5-3× when GPU underutilized |
| `cfg.fast_spike_reset` | DEFAULT True in bio_three_factor | 1.2-1.5× via cp.where masked update |
| Three-factor GPU port (Phase 1) | DEFAULT `gpu_eligibility=True` | 2× over CPU eligibility |
| `cfg.fp16_synapse_state` | **AVAILABLE BUT DEFAULT FALSE** | est 1.2-1.5× via halved memory bandwidth |
| `parallel=6` in YAMLs | Available | 6× sweep throughput when independent seeds |

## High-priority fixes (before cloud)

### 1. Enable `fp16_synapse_state=True` by default — validated, free speedup

**Status:** Already shipped + validated (`<1mV voltage drift over 1000 steps`
per CLAUDE.md "perf wave 2" 2026-05-05). Just defaults to False.

**Where:** Defaults in bio_three_factor.py:155 + consolidation_synonym_trainer.py
(not currently set). Change one default → all chat / synonym / consolidation
runners benefit.

**Expected speedup:** 1.2-1.5× on plasticity-heavy training (synapse state
is read every step; halving bandwidth saves real time at 30M+ synapses).

**VRAM savings:** 60-120 MB at 30M-synapse arch (`cp_eligibility_trace`
storage halves). Not huge but real.

**Risk:** Already validated; flipping the default just means existing runners
get the speedup automatically.

**Estimated effort:** 1 hour (flip default + benchmark on smoke + commit).

### 2. Reduce `stim_steps_per_event` from 50 → 25-30 — accuracy/speed trade

**Status:** Currently 50 stim steps + 50 reset steps per training event
(100 total). At dt=1.0 that's 100 ms of simulated time per event.

**Where:** bio_three_factor.py:131 + consolidation_synonym_trainer.py:256

**Expected speedup:** 1.5-1.7× on training (training is event × steps; halving
steps = ~halving wall clock since event-overhead is small).

**Risk:** Per-event integration window directly affects how strong each
training signal is. If too short, STDP traces don't accumulate enough.
Need a benchmark comparing 50 vs 30 vs 25 on a reproducible config.

**Estimated effort:** 1-2 hrs (sweep + verify accuracy holds at >= prior result).

### 3. Skip plasticity during `reset_steps` — should be no-op anyway

**Status:** Currently the bridge runs full plasticity during reset_steps
(quiet phase between events). But reset_steps deliberately drives nothing,
so STDP shouldn't fire. Skipping plasticity entirely during reset would
save the ops.

**Where:** Either a new flag `bridge.plasticity_off()` /
`bridge.plasticity_on()` toggled around reset blocks, OR via existing
`cp_plasticity_rate_gain` set to 0 during reset.

**Expected speedup:** 1.3-1.5× on training (50 of 100 steps per event
get plasticity skipped).

**Risk:** STDP traces continue decaying during reset (correct biology).
Need to confirm decay still happens with plasticity gate off.

**Estimated effort:** 2-3 hrs (instrument + benchmark + verify accuracy).

### 4. Sparse cross-region density — 0.1 → 0.05

**Status:** Currently `internal_density=0.1` and `pathway.density=0.1` for
most regions. Halving to 0.05 halves synapse count for those pathways.

**Where:** Various builders in `research/runners/biological_brain_regions.py`
and similar. Need to identify which pathways are dominant by synapse count.

**Expected speedup:** Linear with synapse reduction. If 50% of synapses
are in 0.1-density pathways and we drop to 0.05, that's a 25% reduction
overall → ~1.3× speedup.

**Risk:** Sparser connectivity may break learning at higher vocab tiers
(less redundancy). Need full validation pass (Phase 1.4 BRANCH A retention,
Tier 2.1 binding) before deploying. Probably tier-specific.

**Estimated effort:** 1-2 days (architectural change + full revalidation).

## Medium-priority (deeper investigation)

### 5. Profile inner sim loop with `cProfile` / `nsys`

**Goal:** Identify ACTUAL hot paths. Suspicions:
- Per-step CSR matrix-vector multiply for synaptic conductance
- Per-step STDP trace decay (read-modify-write)
- Per-step STP decay (if STP enabled)
- Per-step homeostasis (EMA update)

**Tools:** `python -m cProfile` for Python overhead; `nsys profile` for
CUDA kernels. The latter would tell us EXACTLY which CUDA op consumes
most time.

**Expected outcome:** Identifies optimization candidates we don't currently
see. May find low-hanging fruit (e.g., a Python loop that should be CuPy
batch op).

**Estimated effort:** 1 day (profile + analyze + ship targeted fix).

### 6. FP16 throughout (membrane V, conductances)

**Status:** Currently `cp_membrane_potential_v`, `cp_conductance_g_e`,
`cp_conductance_g_i`, etc. are FP32. FP16 would halve memory bandwidth.

**Risk:** Membrane potential has small dynamic range that matters for
spike detection. FP16 precision (~10⁻³) might cause spurious / missed
spikes near threshold.

**Test:** Run a smoke side-by-side at FP32 vs FP16 (full-state). Compare
output spike trains for divergence; compare end-of-training accuracy.

**Estimated speedup:** 1.5-2× on per-step compute if it works. Big.

**Estimated effort:** 2-3 days (instrument + benchmark + decide tier
where to enable).

### 7. Multi-stream GPU execution

**Status:** Currently single CUDA stream. CuPy supports multiple streams
which can overlap kernels with memory transfers.

**Where:** Bridge step pipeline has natural concurrency: while compute is
on stream A, memory transfers (CPU↔GPU) on stream B.

**Expected speedup:** 1.1-1.3× depending on how much the inner loop is
memory-bound.

**Estimated effort:** 2-3 days (rewrite step pipeline + correctness check).

### 8. Fused kernel extension

**Status:** `@cp.fuse()` already used for ~12 kernels in `sim/kernels.py`.
Some operations not fused that could be (e.g., conductance decay + current
computation are in separate kernels).

**Tools:** Identify candidates by reading the inner step pipeline + checking
which ops still produce intermediate tensors.

**Expected speedup:** 1.1-1.2× per fused kernel; cumulative over all
candidate fusions could be 1.5×.

**Estimated effort:** 2-3 days (kernel-by-kernel investigation).

## Low-priority / experimental

### 9. Multi-GPU (data parallel)

Currently single-GPU. Multi-GPU would let us run multiple seeds in parallel.
But local hardware is single-3090; cloud would be the place this matters.

### 10. JIT compilation (e.g. via Triton / Numba CUDA)

CuPy's `@cp.fuse()` is the simplest JIT we use. More aggressive JIT could
in principle squeeze more out of the inner loop, but is high effort and
research-grade.

## Recommended priority order

If the goal is "maximize cloud compute value", do these IN ORDER:

1. **#1 fp16_synapse_state default** — 1 hour, free 1.2-1.5× (already validated)
2. **#5 profile inner loop** — 1 day, identifies what to optimize next
3. **#3 plasticity-off during reset** — 2-3 hrs, ~1.3-1.5× free
4. **#2 reduce stim_steps to 25-30** — 1-2 hrs, ~1.5×, accuracy-trade
5. **#4 sparser density** — 1-2 days (after #1-3 to know where bottleneck is)
6. **#6 FP16 membrane V** — 2-3 days, biggest potential win but risky

**Cumulative best case:** Stacking #1 + #2 + #3 = 1.4 × 1.5 × 1.3 = **2.7× speedup**
on training wall-clock. With #5 + #4 + #6, potentially 4-5× total.

At cloud rates (~$2/hr A100), a 3× speedup over a 50-hour sweep saves
~$66 (~$2 × 33). Not life-changing but real. More importantly, **same
experiments finish faster locally** — find-the-ceiling could fit in a
night instead of multi-day.

## Cloud-side considerations

When we DO move to cloud:
- A100 80 GB: ~3-4× faster per token than 3090 24 GB at FP32; ~2× at FP16
- H100 80 GB: ~6-8× faster per token than 3090; significantly more on
  FP16 compute (Tensor Cores)
- 8× H100: ~$32/hr; could run ~24 parallel seeds at moderate arch sizes

So the rough multiplier of "local optimization on 3090" applies on top of
"cloud GPU is faster per second":
- 3090 local FP32 unoptimized = baseline 1×
- 3090 local FP16 + skip-plasticity + reduced steps = 2.7×
- A100 cloud FP32 unoptimized = 3-4×
- A100 cloud FP16 + same opts = 8-12×
- H100 cloud FP16 + same opts = 16-24× over baseline 3090

So **per dollar, optimizing locally first is high-value** because the
optimization compounds on top of better hardware.

## Decision recommendation

Tomorrow's session: spend 4-6 hrs on items #1-3 (fp16 default + profile + skip-
plasticity), validate, and re-benchmark 64-word smoke. Compare wall-clock
before/after. If we see 2-3× total speedup, that unblocks the find-the-
ceiling experiment to fit in one night (instead of one experiment per day).

If those land cleanly, go to items #4-6 over the following week.

THEN move to cloud with a meaningfully faster simulator.

## Provenance

- Current opt stack: CLAUDE.md "Recommended configuration"
- Perf wave 1 (2026-05-04): `research/findings/2026-05-04-perf-speedup-stack.md`
- Perf wave 2 (2026-05-05): `research/findings/2026-05-05-perf-roadmap.md`
- 64-word smoke per-chunk timing: `webapp/runtime/run_71f48c37dff6.log`
- 16-word smoke per-chunk timing: `webapp/runtime/run_ea1050bc8d10.log`
