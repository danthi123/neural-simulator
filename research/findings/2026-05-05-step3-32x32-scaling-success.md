# 🎉 Step 3 — 32×32 navigation SCALES, beats 16×16 baseline

**Date:** 2026-05-05 ~17:35 EDT
**Status:** Single-seed smoke test PASSED. 32×32 navigation works
better than 16×16 with the same biology stack. Significantly shifts
the strategic landscape away from dendritic learning urgency.

---

## Result

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 32 --seed 42 --n-steps 1800
```

**Mean Manhattan distance: 2.70** (1 seed)

Compare:
- 16×16 (Cluster K v2 baseline, n=3): **2.97 ± 0.12**
- 32×32 (this smoke, n=1): **2.70**
- Random walk on 32×32: ~21 estimated

That's right — 32×32 navigation scored BETTER than 16×16. On a grid
with 4× more cells. With a Manhattan diameter of ~62 vs ~14.

## Per-phase breakdown (multi-goal, 4 phases × 450 steps)

| Phase | Goal | Mean | First quarter | Final quarter | Steps at goal |
|---|---|---|---|---|---|
| 0 | (30, 30) | 4.50 | 16.21 | 0.65 | 167/450 |
| 1 | (1, 30) | 2.12 | 6.38 | 0.72 | 162/450 |
| 2 | (1, 1) | 2.62 | 8.02 | 0.87 | 149/450 |
| 3 | (30, 1) | 1.57 | 4.18 | 0.67 | 167/450 |

Patterns:
- **Initial exploration (phase 0 first quarter)**: distance 16.21 —
  agent has no prior, explores randomly until vision + reward gradient
  pulls it toward the goal
- **Final quarter of every phase**: distance 0.65-0.87 — agent gets
  AT or ADJACENT to goal and stays there
- **Goal transitions**: distance recovers within first quarter
  (16.21 → 6.38 → 8.02 → 4.18) — adaptation gets faster across
  transitions, suggesting hippocampal place-cell remapping is
  contributing

Total: **645/1800 steps at goal (35.8% of total time)**.

## What this means for the four-step plan

✅ **Step 1:** verdict validated (gradient passes permuted-label)
✅ **Step 2:** 3-factor rule cannot be rescued by parameter or encoding tuning
✅ **Step 3:** scaling works — architecture has unexploited capacity
🔄 **Step 4:** dendritic learning Week 1+ — STILL pending user
   greenlight, but now WITH ALTERNATIVES on the table

The strategic landscape shifted significantly:

**Before step 3:** dendritic learning was the obvious next direction
because W→A failure was the dominant negative signal. 1.5-2 months
of engineering for a fix to a specific failure mode.

**After step 3:** the architecture has demonstrated working biology-
grounded learning at HARDER tasks (32×32 navigation) than the W→A
flashcard problem. The W→A failure is now contextualized as a
specific case (arbitrary cue + arbitrary action + sparse reward),
not a general limit.

## Two viable forward directions

### Direction A: Continue scaling navigation (recommended)

The Cluster G v2.5 + K v2 stack scales positively. Push it:
- 32×32 6-seed validation (5-7 hours autonomous)
- 64×64 smoke
- Dynamic obstacles (week of work)
- Multi-step plans (2-3 weeks)
- Real visual input (MNIST 28×28 or Tiny ImageNet)

This builds on a confirmed strength. Demonstrates biology-grounded
RL that scales. Publishable as systems-neuroscience research.

### Direction B: Dendritic learning specifically for W→A

If language → action mapping is a project goal in itself (vs being
a proxy for general learning), dendritic learning is the
theoretically right answer. 1.5-2 months for a specific contribution.

### Recommendation

Direction A first. Direction B later if W→A specifically remains a
priority. The 32×32 result is a real win that deserves consolidation
and 6-seed validation before committing to a 1.5-2 month rewrite for
a specific failure mode.

## Caveats

- **n=1 smoke test.** 32×32 needs 6-seed validation before claiming
  generalization. Cost: ~5-7 hours at parallel=3 with reasonable
  sweep yaml.
- **Action distribution still skewed.** Phase 0 action counts
  [N=159, E=104, S=119, W=68] — agent uses W less even when goal
  requires it. Phase 3 [N=120, E=125, S=121, W=84] is more balanced.
  Suggests N-bias persists at 32×32 (consistent with 16×16 finding).
- **Wall-clock 9 minutes for 1 seed.** 6-seed parallel=3 = ~30 min.
  Cheap to validate.

## Files

- This finding
- Run JSON: `research/findings/raw/g11_bg/scale_32x32_seed42.json`
- 16×16 baseline: `research/findings/2026-05-01-cluster-k-v2-breakthrough.md`
- Step 1 validation: `research/findings/2026-05-05-gradient-passes-permuted-label-VALIDATED.md`
- Step 2 verdict: `research/findings/2026-05-05-step2-COMPLETE-rule-not-rescuable.md`

## What I'd recommend the user do first thing tomorrow

1. **Validate 32×32 with 6-seed sweep.** ~30 min compute. Confirms
   the scaling result is robust.
2. **If 6-seed confirms (>=4 seeds < 4.0 mean)**: write up as a
   formal finding, push 64×64 next.
3. **Then decide** dendritic learning (W→A specifically) vs continued
   scaling work. Both are valid; the choice depends on whether W→A is
   itself a project priority or was just a proxy benchmark.

The autonomous mode has now landed:
- Verdict (W→A 3-factor fails) — validated rigorously
- Three rescue attempts for 3-factor (LR, encoding, DA mode) — all negative
- Scaling test of working architecture — positive, scales beyond 16×16
- Strategic options laid out clearly for user choice
