# Informed-Init Learned Perception — Doesn't Solve Cold-Start

**Date:** 2026-04-26 (final autonomous experiment)
**Status:** NEGATIVE — directional prior on sensory→cortex weights helps only marginally vs random init, both far worse than the heuristic baseline.
**Companion:** [Cold-start failure (no prior)](2026-04-26-learned-perception-cold-start-fail.md), [Phase B real-win](2026-04-25-phase-b-acid-test-real-win.md)

## TL;DR

The user's chosen direction (option 1) was: replace heuristic cortex drive with a learned sensory→cortex layer initialized via a directional prior. Hypothesis: plasticity refines the prior rather than discovers from random init.

Tested two prior strengths on 3 seeds × 1800 steps:

| Variant | Sum (3-seed avg) | Comparison |
|---|---:|---|
| Baseline (heuristic cortex drive) | **5.88** | recommended default |
| Cold-start learned perception (no prior) | 10.85 | random walk equivalent |
| **Soft informed init (α=5, base=10, signed)** | **11.41** | similar to cold-start |
| **Sharp informed init (α=8, base=0.5, positive-only)** | **12.09** | slightly worse |

**Both prior schemes underperform the heuristic baseline by ~2x.** The sharper prior is actually slightly worse than the softer one — going harder doesn't help.

## Implementation (kept opt-in: `--informed-init --learned-perception`)

```python
# For each sensory neuron i with preferred (dx_i, dy_i):
#   For each action X with direction (ax, ay) ∈ {(0,1), (1,0), (0,-1), (-1,0)}:
#     alignment = dx_i * ax + dy_i * ay
#     # Sharp version (positive-only):
#     positive_alignment = max(0, alignment)
#     weight_to_cortex_X = max(0.5, 0.5 + α * positive_alignment)
```

With α=8: aligned sensors → cortex_X = 24.5 (matches heuristic 800pA equivalent), orthogonal/anti-aligned → 0.5 (essentially silent).

Verified at smoke time: 4900 sensory→cortex synapses correctly rewritten.

## Why it doesn't work

Hypothesis (not fully validated):

1. **Multiple sensors firing simultaneously**: with Gaussian-tuned drive (σ=1.5), a goal at (6,6) from agent (3,3) activates ~5-9 sensory neurons around (dx=3, dy=3). For diagonal sensors like (3, 2), (2, 3), (3, 3), the prior gives them moderate weight to BOTH cortex_N AND cortex_E (diagonal goals require both). This sounds correct, but...

2. **Cortex→D1 is also plastic**: when multiple cortex pools fire simultaneously, multiple cortex→D1 pathways accumulate eligibility. The global reward signal then strengthens all of them — credit smearing across pathways. After 1800 trials, the BG cascade's selectivity can erode.

3. **Heuristic gives a binary signal**: only ONE cortex pool fires at a time (per goal-direction component). The BG cascade has CLEAN input asymmetry to amplify. Informed-init produces graded multi-pool input that's harder for the cascade to disambiguate.

4. **Plasticity drift**: STDP without aggressive gating tends to grow weights toward saturation when fired frequently, regardless of reward. The prior's directional structure may erode over training.

## What this means

The cold-start problem in this BG-cascade architecture is harder than just "give it an informed init." The architecture is sensitive to clean cortex pool selectivity — graded input signals seem to break the disambiguation that BG cascade depends on.

To make learned perception work, you'd need ONE of:
- **Curriculum**: fix-goal training first to lock weights, then expose to moving goal
- **Sparser encoding**: fewer sensors firing per step (one-hot or near-one-hot)
- **Plasticity gating**: freeze cortex→D1 during sensory→cortex learning, then thaw
- **Architecture change**: lateral inhibition between cortex pools to enforce winner-takes-all at the cortex level

None of these were tried in this autonomous session.

## Per-seed details

```
                       P0    P1    Sum
soft (α=5)  seed 42   4.99  4.48   9.46
soft (α=5)  seed 43   5.87  7.07  12.94
soft (α=5)  seed 44   6.96  4.86  11.82
                       avg          11.41

sharp (α=8) seed 42   5.17  4.18   9.35
sharp (α=8) seed 43   5.24  7.19  12.43
sharp (α=8) seed 44   6.05  8.43  14.49
                       avg          12.09
```

High variance both ways — the prior isn't reliably steering learning.

## Decision

- Keep `--learned-perception --informed-init` flags opt-in for future experimentation.
- Default remains heuristic cortex drive.
- The cleanest path forward (NOT pursued this session): curriculum learning or cortex-level WTA.

## Files

- `research/runners/g11_bg_runner.py:498-539, ~660 (CLI)`: informed-init implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_informed.json`: soft α=5 data
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_informed_v2.json`: sharp α=8 data

## Lesson

The cold-start finding from earlier (random init fails) was correct, but the proposed fix (directional prior) doesn't fully solve it on this architecture. The BG cascade depends on cleaner asymmetric cortex input than a soft / multi-pool sensory→cortex projection can provide. Solving learned perception properly requires more than just better initialization — likely curriculum or architectural changes.

This brings the autonomous overnight session to its natural close: Phase B's BG cascade is robust, sharpening refinements give modest task-conditional gains, but eliminating the heuristic cortex drive remains an open problem. The user's specific direction has been honestly tested.
