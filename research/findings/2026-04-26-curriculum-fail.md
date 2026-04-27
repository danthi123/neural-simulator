# Curriculum Learning (Drive-Gated) — NEGATIVE: hippo turn-on still disrupts

**Date:** 2026-04-26 (after combo A WTA+adaDA+hippo PARTIAL)
**Status:** **NEGATIVE** — curriculum that only gates hippo *drive* (not plasticity) doesn't crack the plastic-input-layer ceiling. True curriculum needs bridge-level staged plasticity (freeze cortex→D1 once mature).
**Companion:** [Cortex WTA + adaDA combo](2026-04-26-cortex-wta-adapda-combo.md), [Hippocampus additive fail](2026-04-26-hippocampus-additive-fail.md)

## TL;DR

Hypothesis: suppress hippocampus drive during a warmup phase so cortex+
heuristic+WTA build up cortex→D1 selectivity in isolation. Then enable
hippo drive — its plastic weights learn given that the cascade is already
mature. Tests whether plastic-input-layer ceiling is sample-efficiency
(curriculum solves it) or structural (curriculum also fails).

**Result: structural.** Drive-gated curriculum doesn't crack the ceiling.

| Variant | Sum (3-seed avg) | P1 finalQ | vs baseline |
|---|---:|---:|---|
| Baseline | 5.88 | ~2.9 | reference |
| WTA + hippo | 9.26 | ~5.0 | -58% |
| WTA + hippo + adaDA | 8.01 | ~5.0 | -36% |
| **WTA + hippo + adaDA + curriculum (warmup=600)** | **10.25** | **5.16** | **-74%** |

## Per-seed details

```
Curriculum schedule: P0 = 1200 steps fixed (6,6), P1 = 600 steps fixed (1,6)
Hippo silent during steps 0-599 (warmup); on from step 600+

seed 42: P0 finalQ=6.34  P1 finalQ=1.89  sum=8.23   n_at_goal P1=21
seed 43: P0 finalQ=4.97  P1 finalQ=7.75  sum=12.72  n_at_goal P1=8  (worst)
seed 44: P0 finalQ=3.96  P1 finalQ=5.85  sum=9.80   n_at_goal P1=14
avg: sum=10.25, P1 finalQ=5.16
```

## Why it fails

Two structural reasons:

### 1. Hippo turn-on at step 600 disrupts mature cortex→D1

P0 finalQ (measured during steps 900-1200) averages 5.09 — substantially
worse than baseline ~3.0. The agent had already built up correct cortex→D1
weights during the warmup phase (steps 0-599 with heuristic only); then
hippo turned on at step 600, injecting random-weight noise into all 4
cortex pools, which cortex→D1 then *learns* to compensate for badly.

The plasticity didn't pause — it kept running, and absorbed the noise.

### 2. The metric is unfair to curriculum (but it still loses)

Curriculum schedule has P0=1200 steps, P1=600 steps. Default has P0=300,
P1=1500. So curriculum has only 600 steps for readaptation vs 1500 in
default. P1 finalQ measures the last quarter (150 steps for curriculum,
375 for default). Less time to converge.

Even acknowledging this, the curriculum's avg sum (10.25) is *worse*
than combo A (8.01) and approaches the hippo-additive-only NEGATIVE
result (10.98). Even with the metric tilted in curriculum's favor by
giving it more learning time, it doesn't help.

### 3. Variance: one seed shows it CAN work in principle

Seed 42 has P1 finalQ=1.89 — better than baseline's ~2.9. With a fully
trained cortex+hippo system, the agent reaches Manhattan distance 1.89
from goal in steady state on a grid where random walk is ~5.5.

But seed 43 has P1 finalQ=7.75 (random-walk territory). The architecture
is unstable: same configuration, different seeds, ±3-fold variance in
phase 1 performance.

This high variance is a signature of *local optima* — sometimes the
hippo→cortex weights settle into the right asymmetry, sometimes they
don't. With more trials per seed (4500+ steps?) the variance might
converge, but that's a different (more expensive) experiment.

## What "true curriculum" would look like

The hypothesis behind drive-gated curriculum was correct in spirit but
the implementation was incomplete. Real curriculum learning would:

1. **Freeze cortex→D1 plasticity once mature.** Currently this requires
   bridge modification — there's no per-pathway runtime plasticity toggle.
2. **Re-warmup hippo→cortex from random.** With cortex→D1 frozen, hippo's
   plastic weights have a stable target — they can learn place→cortex_pool
   without simultaneously dragging cortex→D1 into a worse local optimum.
3. **Two-stage training**: first train cortex→D1 to maturity, then train
   hippo→cortex with cortex→D1 frozen, then optionally fine-tune both
   together.

This is a multi-day implementation, not a runner flag.

## Architectural insight (now firm)

Six consecutive plastic-input-layer attempts have now failed to match
baseline:

| Attempt | Sum | Status |
|---|---:|---|
| Cold-start learned perception | — | NEGATIVE (cascade silenced) |
| Informed-init perception | 12.09 | NEGATIVE |
| Hippocampus replacement | — | NEGATIVE (cascade silenced) |
| Hippocampus additive | 10.98 | NEGATIVE |
| Cortex WTA + hippo | 9.26 | PARTIAL |
| Cortex WTA + adaDA + hippo | 8.01 | PARTIAL |
| **Drive-gated curriculum** | **10.25** | **NEGATIVE** |

This now firmly closes "more flags will fix it" AND "easy curriculum
will fix it" approaches. The architectural ceiling is structural —
specifically, the cascade depends on a single clean cortex input source,
and any plastic input layer with random weights destabilizes that.

To break this ceiling requires real structural infrastructure work:
- **Bridge-level staged plasticity** (per-pathway plasticity freeze/thaw)
- OR **inverted architecture** where hippocampus learns *to drive heuristic
  output* directly, with heuristic as teacher signal
- OR **a totally different task framing** that doesn't require cortex
  selectivity to be perfectly clean

## Decision

- Keep `--curriculum --goal-schedule curriculum` flags opt-in for future
  bridge-level work.
- Default remains heuristic only (sum 5.88).
- **Stop the runner-side variant arc.** Six attempts is enough.
- Pivot to Option C: explore other research directions that don't depend
  on this specific architectural problem.

## Files

- `research/runners/g11_bg_runner.py:506-516, 794-808, 1106-1112`: curriculum implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_curriculum.json`: 3-seed acid test
- `research/findings/raw/g11_bg/g11_seed42_curriculum_smoke.json`: 50-step smoke

## Lesson

The drive-gated curriculum *was* a fair test of "does temporal sequencing
help?" — and the answer is "not without staged plasticity." The
plasticity machinery doesn't care that hippo just turned on; it processes
all spikes equally and updates all plastic pathways equally. When hippo
fires, cortex→D1 promptly absorbs the noise.

Real biological curriculum learning involves things like attention,
selective gating, neuromodulator-controlled plasticity that pauses
specific synapses during certain phases. We *have* a neuromodulator
subsystem (E.1) that could in principle gate plasticity rate — but the
bridge implementation only gates a single global `reward_learning_rate`,
not per-pathway. Building per-pathway plasticity gating is the next
real architectural step if anyone wants to revisit this arc.

For now: pivot. The plastic-input-layer ceiling is closed off until we
build that infrastructure.
