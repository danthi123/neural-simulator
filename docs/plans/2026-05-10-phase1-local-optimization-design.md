---
type: plan
status: live
date: 2026-05-10
---

# Phase 1 local-optimization design — target 5-10× more on RTX 3090

**Date:** 2026-05-10
**Status:** DESIGN — execution starts after current arc completes
**Goal:** Make 256-512 word vocab usable on local RTX 3090, with
day-to-day continual learning sustainable. Compound with cloud H100
for ~50-100× over current baseline.
**Trigger:** User strategic direction (2026-05-10) — focus on local-
usable continual-learning agent vs SOTA-LLM-matching distributed
compute.

---

## Current optimization baseline (shipped today 2026-05-10)

| Optimization | Speedup | Status |
|--------------|---------|--------|
| `dt=1.0` (Izhikevich) | 2× | DEFAULT |
| `cfg.fast_spike_reset` | 1.2-1.5× | DEFAULT True |
| Three-factor GPU port | 2× | DEFAULT |
| **STP disabled** in chat training | **3.28×** | DEFAULT FLIPPED 2026-05-10 |
| Encoding-axis arch (n_lang=8K vs 4K, n_motor=2K vs 6K) | **~35×** at 64-word | recommended for vocab≥32 |
| `fp16_synapse_state` | 1.135× | opt-in, validated |
| `freeze_plasticity_during_reset` | TBD | opt-in, untested in real workload |
| **Cumulative current** | **~5-35× over original** | shipped today |

## Phase 1 optimization targets

Ordered by ROI (highest expected return first):

### Target 1: Bridge construction speedup (4-10× on big arches)

**Problem:** at n_lang=32768, bridge construction takes 44 min (vs 5
min at 16K). This is the wall preventing experiments at higher vocab.

**Hypothesis:** the `inject_explicit_wiring` path constructs synapses
sequentially in Python land before transferring to GPU. At ~100M+
synapses, this is dominated by Python overhead.

**Approach:**
1. Profile `inject_explicit_wiring` with cProfile + nsys
2. Identify Python hot paths
3. Move connectivity generation to GPU where possible
4. Parallelize independent pathways
5. Reduce intermediate memory allocations

**Expected speedup:** 4-10× on construction time (cuts 44 min to
~5-10 min at n_lang=32K). Enables experiments at currently-impractical
arches.

**Effort:** ~1-2 weeks of profiling + targeted rewrites.

### Target 2: Sparser cross-region density (1.5-3× on training/inference)

**Problem:** Cross-region pathways use `density=0.1` (each post-neuron
connects to 10% of pre-neurons). Most connections are noise; only
~20-30% carry meaningful binding signal.

**Approach:**
1. Multi-seed validation at `density=0.05` (halved)
2. Compare retention + binding metrics to current `density=0.1`
3. If accuracy holds within 5pp: flip default
4. Synapse count halves → memory bandwidth halves → training/inference
   1.5-2× faster

**Expected speedup:** 1.5-3× on per-step time. Compounds across all
arches.

**Effort:** ~1-2 weeks (multi-seed validation is the slow part).

**Risk:** Sparser density may break learning at higher vocab (less
redundancy). Test at 8/16/64-word vocab levels before deploying.

### Target 3: FP16 throughout (1.5-2× on per-step)

**Problem:** Membrane potential, conductances, and most state arrays
are FP32. FP16 halves memory bandwidth — the inner loop is memory-bound.

**Approach:**
1. Start with FP16 on `cp_eligibility_trace` (DONE: 1.135× measured)
2. Extend to `cp_stp_x`, `cp_stp_u` (small wins; ~100MB save)
3. **Membrane potential FP16** — the big one. Risk: small dynamic range
   matters for spike detection (~10⁻³ precision)
4. Test spike-train divergence:
   - Run identical seed in FP32 vs FP16-membrane
   - Compare spike timings step-by-step
   - If divergence < 5% over 1000 steps: deploy
5. Per-region opt-in: enable FP16 membrane only where validated safe

**Expected speedup:** 1.5-2× cumulative on per-step time.

**Risk:** Membrane FP16 may cause spurious / missed spikes near
threshold. Critical for learning quality.

**Effort:** ~2-3 weeks (validation is risky and slow).

### Target 4: Per-pathway STP gates (biology + optimization)

**Problem:** Today's STP-off flip is too coarse. Different pathways
need different STP behavior:
- Language→motor pathways: STP-off is fine (binding task)
- Hippocampus replay: STP-on for biological gamma oscillations
- Visual cortex adaptation: STP-on for sensory adaptation

**Approach:**
1. Add `RegionPathway.stp_gate: str | None` field (matches existing
   `plasticity_gate`)
2. Add `_stp_gate_to_synapses` dict in bridge
3. Per-synapse STP enable/disable: skip STP kernel on gated synapses
4. CLI / config: `cfg.stp_gate_overrides = {"language_to_motor": False, "hippo_replay": True}`

**Expected speedup:** Modest in chat training (already STP-off), but
unlocks biology-realism in other workloads while preserving the speed.

**Effort:** ~2-3 weeks (kernel changes + validation).

**Bonus:** Addresses the Tier 2.3 phrase concern from earlier today.
PFC phrase pool can keep STP for biology, language→motor can stay off
for speed.

### Target 5: Multi-stream GPU execution (1.1-1.3×)

**Problem:** Single CUDA stream. Some ops could overlap (compute vs
memory transfers).

**Approach:**
1. Identify natural concurrency: while plasticity is computing, can
   we prefetch next step's spike state?
2. Use CuPy multi-stream + asynchronous memcpy
3. Profile to ensure overlap actually happens

**Expected speedup:** 1.1-1.3×. Small but free if it works.

**Effort:** ~1-2 weeks. Some risk of correctness bugs.

### Target 6: Fused kernel extension (1.3-1.5×)

**Problem:** `@cp.fuse()` already on ~12 kernels but more candidates
exist:
- `conductance_decay_and_current` could fuse with neuron dynamics
- Multiple plasticity ops could fuse into one kernel

**Approach:** Identify intermediate-tensor ops that could fuse;
benchmark per-fusion.

**Expected speedup:** 1.3-1.5× cumulative.

**Effort:** ~2-3 weeks (kernel-by-kernel investigation).

### Target 7: CSR sparse-matrix kernel tuning (1.2-1.5×)

**Problem:** CSR matrix-vector multiply for synaptic conductance is
a dominant inner-loop cost. CuPy's default CSR ops may not be optimal
for our access patterns.

**Approach:**
1. Profile CSR ops
2. Compare to custom kernel (e.g., one that exploits row-wise
   parallelism more aggressively)
3. Consider switching to ELL or hybrid sparse format for hot pathways

**Expected speedup:** 1.2-1.5× on synaptic propagation step.

**Effort:** ~2-3 weeks.

### Target 8: Cython for inner loop (2-3×, RISKY)

**Problem:** Even with CuPy GPU ops, there's Python-side overhead
per simulation step (timer dict ops, attribute lookups, etc.).

**Approach:** Rewrite `_run_one_simulation_step` in Cython.

**Expected speedup:** 2-3× on Python overhead reduction.

**Risk:** Large rewrite. Hard to verify correctness vs Python
version. Probably defer until other targets exhausted.

**Effort:** ~3-4 weeks.

## Cumulative speedup projection

Conservative compound:
```
Current baseline                                        1.0×
+ Target 1 (bridge construction)                        1.0× during training, ~4-10× one-time
+ Target 2 (sparser density 0.05)                       2.0× (1.5-3.0)
+ Target 3 (FP16 membrane)                              3.5× (1.5-2.0)
+ Target 4 (per-pathway STP)                            3.5× (selective)
+ Target 5 (multi-stream)                               4.0× (1.1-1.3)
+ Target 6 (fused kernels)                              5.5× (1.3-1.5)
+ Target 7 (CSR tuning)                                 7.5× (1.2-1.5)
+ Target 8 (Cython, if pursued)                         18.0× (2-3)
```

**Realistic Phase 1 cumulative: 5-10× more over current** (Targets 1-3
+ part of 4-7). With Target 8: another 2-3× possible.

**Cloud multiplier:** A100 ~3-4× over 3090, H100 ~6-8× over 3090. Compounds:

- 3090 local Phase 1 optimized: ~5-10× over today
- 3090 local + cloud: same
- A100 cloud + Phase 1 optimizations: ~20-40× over today
- H100 cloud + Phase 1 optimizations: ~30-80× over today
- 8× H100 + Phase 1: ~250-650× over today (for sweep throughput, not
  single-run speedup)

That puts cloud H100 single-seed throughput at:
- 64-word smoke: 30 sec (vs current local 16 min)
- 256-word smoke: 5-10 min (vs current local would-be 10+ hours)
- 1000-word smoke: 30-60 min (vs current local infeasible)

## Phase 1 work breakdown

Critical path (~3 months focused effort):

**Month 1: highest-ROI targets**
- Week 1: Bridge construction profiling + speedup (Target 1)
- Week 2: Sparser density multi-seed validation (Target 2)
- Week 3-4: FP16 membrane validation (Target 3)

**Month 2: biology + selectivity**
- Week 1-2: Per-pathway STP gates (Target 4)
- Week 3-4: Multi-stream + fused kernel pass (Targets 5+6)

**Month 3: polishing + validation**
- Week 1-2: CSR kernel tuning (Target 7)
- Week 3-4: End-to-end validation: 256/512-word smoke + multi-seed
  with optimized stack

After Month 3: 256-512 word vocab usable on local 3090 with day-to-day
continual learning.

## Phase 1 decision gate

End of Month 3, evaluate:

**GO criteria for Phase 2 (cloud-anchored 1000-2500 word):**
- 256-word multi-seed retention > 80%
- 512-word smoke retention > 60% (loose; smoke is a probe not a
  full validation)
- Local inference latency at 512-word < 10 sec per `:speak`
- Day-to-day continual learning compute < 2-4 hr/day on 3090

**STOP criteria (Phase 1 was the wrong path):**
- Multi-seed retention degrades despite all optimizations
- Latency at 256-word > 30 sec per `:speak` (clearly unusable)
- Optimization stack doesn't compose (target speedups don't compound)

## Provenance

- This design: `docs/plans/2026-05-10-phase1-local-optimization-design.md`
- Master plan addendum: `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
- Auto-growth design: `docs/plans/2026-05-10-auto-growth-design.md`
- Optimization audit (baseline): `research/findings/2026-05-10-perf-optimization-audit.md`
- STP discovery: `research/findings/2026-05-10-stp-default-flip.md`
- Encoding-axis discovery: `research/findings/2026-05-10-encoding-axis-64word-3SEED-GO.md`
- VRAM ceiling: `research/findings/2026-05-10-vram-ceiling-probe-results.md`
