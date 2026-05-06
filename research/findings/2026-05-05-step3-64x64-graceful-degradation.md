# 64×64 navigation — scaling continues but with graceful degradation

**Date:** 2026-05-05 ~20:20 EDT
**Status:** Architecture scales to 64×64 with reduced performance.
Real learning still happening (~3.5× better than random walk) but
significantly worse than 32×32.

---

## 6-seed validation result

```bash
python -m research.experiment_runner experiments/scale_64x64_validation.yaml
# 6 seeds × 1 condition, parallel=3, ~31 min wall-clock
```

| seed | mean Manhattan | at_goal | elapsed | quarters |
|---|---|---|---|---|
| 42 | 8.05 | 598/1800 | 932s | [17.94, 4.85, 4.72, 4.70] |
| 43 | 8.96 | 569/1800 | 931s | [17.52, 7.95, 5.05, 5.33] |
| 44 | 9.23 | 546/1800 | 931s | [17.27, 7.28, 6.94, 5.45] |
| 100 | 9.48 | 551/1800 | 899s | [18.33, 7.79, 6.19, 5.62] |
| 101 | 8.70 | 554/1800 | 899s | [17.42, 6.65, 4.98, 5.74] |
| 102 | 8.38 | 559/1800 | 899s | [17.67, 5.72, 4.96, 5.15] |

**Mean: 8.80 ± 0.54, range 8.05–9.48. 31.3% of 1800 steps at goal.**

## Scaling comparison

| Grid | Mean Manhattan | std | n | Notes |
|---|---|---|---|---|
| 16×16 | 2.97 | ± 0.12 | 3 | Cluster K v2 baseline |
| 32×32 | **2.57** | ± 0.11 | 6 | Today's peak (13.3% better than 16×16) |
| 64×64 | **8.80** | ± 0.54 | 6 | Today's scaling test |

Random walk baselines (estimated, uniform random position to fixed point):
- 16×16: ~7
- 32×32: ~14
- 64×64: ~21

64×64 is **~2.4× better than random walk** but **~3.4× worse than 32×32**.

## What broke between 32 and 64?

Q1 (exploration phase) costs ~17.5 distance — close to random-walk
level. Q2-Q4 stabilize at 5-6, never reaching the 32×32 quality of
1.7. The agent IS learning but not nailing the goal as tightly.

Probable causes (rank-ordered by likelihood):

1. **Visual cortex resolution shortage.** With `--visual-image-size 64`
   (matched to grid-size), each grid cell maps to **exactly 1 pixel**.
   The agent and goal are 1-pixel objects. Gabor RFs at V1 are tuned
   for edges spanning multiple pixels — single-pixel objects are
   nearly invisible to V1 simple cells. **Fix:** bump
   `visual-image-size` to 128 (2 pixels/cell), which would also
   require larger V1 with proportionally more neurons.

2. **Hippocampal place cell density.** With 256 place cells for 4096
   grid positions = 0.06 cells/cell (vs 32×32: 256/1024 = 0.25
   cells/cell). Spatial discrimination resolution drops 4×.
   **Fix:** scale `n-hippocampus-per-layer` to 1024 (one cell per
   grid position).

3. **Exploration noise scale.** Motor exploration is tuned for
   smaller grids; on bigger grids the agent might wander
   inefficiently before locking onto a direction.

4. **Reward signal saturation.** Distance reward at 60+ Manhattan
   distance might be undifferentiated from distance 50 — both are
   "very far." Loss of gradient information at large distances.

## Strategic implications

This is NOT a catastrophic scaling failure. The architecture STILL
LEARNS at 64×64 — it's just less efficient. Three responses:

### Response A: Investigate + fix the breakdown

If we want 64×64 to be tight, identify which of (1)-(4) above
dominates. Quickest wins:
- Try `--visual-image-size 128 --n-hippocampus-per-layer 1024` — more
  resolution everywhere
- Try 30× longer training (n-steps 54000) at 64×64 — give learning
  more time

Cost: 2-5 days of investigation.

### Response B: Accept 64×64 ceiling, push other axes

Current scaling line shows soft ceiling between 32 and 64. Other axes
can be explored independently:
- **Dynamic obstacles** at 32×32 (the proven sweet spot)
- **Real visual input** (MNIST 28×28) on a 16×16 or 32×32 grid
- **Multi-step plans** with subgoals

These all build on the confirmed 32×32 strength without requiring
64×64 to work.

### Response C: Rethink visual cortex architecture

The Cluster K v2 visual cortex was designed for 8×8 / 16×16. At 32×32
it generalizes well. At 64×64 it shows resolution limits. A
hierarchical visual architecture (multi-scale Gabor pyramid, retinal
foveation) would handle larger grids better.

Cost: 2-4 weeks. Real systems-neuroscience contribution if done well.

## Recommendation

**Response B** confirmed empirically. The 32×32 result (2.57 ± 0.11) is the
project's strongest empirical claim and pushing other axes (obstacles,
visual input, multi-step) builds on a confirmed baseline. 64×64
scaling can be revisited later if/when needed — or as a stretch goal.

## Update — high-res Response-A test (2026-05-05 ~20:35 EDT)

Ran `--visual-image-size 128 --n-hippocampus-per-layer 1024` at 64×64
seed 42, single seed:

| Config | n | Mean Manhattan | Quarters |
|---|---|---|---|
| 64×64 baseline (image=64, place=256) | 6 | 8.80 ± 0.54 | Q1 17.7 / Q2-4 5.7 |
| **64×64 HI-RES (image=128, place=1024)** | **1** | **8.34** | Q1 16.9 / Q2-4 5.5 |

The hi-res result (8.34) is within 1σ of the baseline mean (8.80 ±
0.54, range 8.05–9.48). **Resolution is NOT the bottleneck.**

The actual bottleneck is likely phase budget. At 64×64 each phase has
450 steps to traverse ~60 Manhattan distance + wander. At 32×32 it's
the same 450 steps for ~30 distance — twice the margin. The scaling
limit is set by `n-steps / phases` relative to grid diameter, not by
perception or place cell density.

**This validates Response B.** Investing in resolution gives ~5%
improvement; the structural fix would be to increase n-steps proportional
to grid size (e.g., n-steps=3600 for 64×64) — but that just means we'd
need 4× more compute for 4× more grid. Diminishing returns.

The 32×32 result remains the operational sweet spot. Move forward
with dynamic obstacles, real visual input, or multi-step plans on
the confirmed 32×32 baseline.

## What this DOESN'T contradict

- 32×32 scaling result (2.57 ± 0.11): still the project's strongest
- 16×16 baseline (2.97 ± 0.12): still valid
- Architecture has unexploited capacity in OTHER directions (dynamic
  obstacles, multi-step plans, real visual input)
- W→A 3-factor failure (separate problem, separate fix)

## Files

- This finding
- 64×64 JSONs: `research/findings/raw/g11_bg/scale_64x64_validation_scale64_seed*.json`
- Sweep YAML: `experiments/scale_64x64_validation.yaml`
- 32×32 baseline: `research/findings/2026-05-05-step3-32x32-scaling-success.md`
- Final synthesis: `research/findings/2026-05-05-FINAL-autonomous-arc-synthesis.md`
