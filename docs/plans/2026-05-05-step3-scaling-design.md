---
type: plan
status: live
date: 2026-05-05
---

# Step 3 — scale 2D gridworld + visual cortex

**Date:** 2026-05-05
**Status:** DESIGN ONLY (kicks off after step 2 yields a verdict).

---

## Why scaling is the safest bet

Of the four-step post-verdict plan:

1. ✅ Step 1 (validate verdict) — DONE, gradient passes permuted-label
2. ⏳ Step 2 (better cross-region tasks) — high-LR sweep in flight; orthogonal cues queued; sequence learning + conditional cue-action are larger lifts
3. **Step 3 — scale what works** — biggest existing strength, lowest risk, demonstrable progress
4. Step 4 (dendritic learning) — 1.5-2 mo investment, conditional on 1-3 not yielding

The current 2D gridworld + visual cortex (Cluster K v2) result is the
project's strongest empirical claim:

> 2.97 ± 0.12 at 16×16 perception-only (NO heuristic). Closes 4 of 5
> original cheats. 5.2× better than Tier 0 vanilla perception arc.

This is biology-grounded cross-region credit assignment that ALREADY
WORKS. Pushing it further validates the architecture and produces
publishable evidence.

## Three scaling axes

### Axis 1: Grid size — 32×32 then 64×64

**Why:** Bigger grids require more spatial discrimination + longer
sequences of actions. Tests whether the policy generalizes or just
memorizes 16×16 patterns.

**Current 16×16:** ~14 Manhattan diameter. Random walk baseline ~7.
Agent gets to 2.97.

**32×32:** ~30 Manhattan diameter. Random walk baseline ~15.
Predicted agent: 4-6 if scaling holds; 10-15 if it breaks down.

**64×64:** ~62 Manhattan diameter. Random walk baseline ~31.
Predicted agent: 10-15 if scaling holds; 25-30 if breakdown.

**Cost:** existing runner supports `--grid-size N` flag. No code
changes. Just longer wall-clock per seed (more steps to navigate).

**Test plan:**
```bash
# 32×32 smoke (1 seed, 1800 steps):
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 32 --seed 42 --n-steps 1800

# 6-seed validation if smoke is positive:
# (parallel=3, ~5-7 hours total)
```

### Axis 2: Dynamic obstacles

**Why:** Static gridworld is tractable by simple gradient descent.
Adding obstacles that move or appear forces the agent to use working
memory + replanning — engaging dlpfc_wm + Cluster D hippocampus.

**Implementation cost:** Modest. Add an `--obstacle-density` flag and
update the visual cortex retina renderer to include obstacles in the
visual field. Plus reward shaping to penalize wall collisions.

**Estimated:** ~1 week of focused work to implement and validate.

### Axis 3: Multi-step plans

**Why:** Currently agent gets immediate reward gradient on each step
(closer to goal = better). True planning would require predicting
multi-step consequences (e.g., "to reach X I have to first go to Y
then turn left").

**Implementation:** Add a "subgoal" structure where the agent has to
visit a sequence of points before getting reward. Tests whether
PFC working memory holds the plan across action steps.

**Estimated:** ~2-3 weeks to implement + validate.

## Recommended sequence

1. **Smoke test 32×32** with current architecture (no code changes).
   ~30-90 min for 1 seed. Tells us immediately if scaling holds.
2. **If 32×32 holds**: 6-seed validation at 32×32. Then try 64×64.
3. **If 32×32 breaks**: investigate why (visual cortex resolution?
   exploration noise? motor pool capacity?), fix, retry.
4. **Once grid-size scaling validated**: add dynamic obstacles.
5. **Once obstacles work**: add multi-step plans.

## Cheapest first move

The 32×32 smoke test is FREE (no code changes, single seed, ~30-90
min). It's also informative: a positive result validates step 3 as
the right direction; a negative result shows where the architecture
breaks first.

This should run AFTER step 2 sweep completes (current high-LR sweep
uses parallel=3, no spare GPU). When that finishes:

```bash
nvidia-smi   # confirm idle
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 32 --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/scale_32x32_seed42.json
```

## Files

- This design doc: `docs/plans/2026-05-05-step3-scaling-design.md`
- Existing 16×16 finding: `research/findings/2026-05-01-cluster-k-v2-breakthrough.md` ⛔ RETRACTED (2026-07-16) — that doc's "NO heuristic, NO direct (gx, gy) or (x, y) access" description is WITHDRAWN: `--heuristic-strength` defaults to 1.0 and ran at full strength in the very run. The 2.97 ± 0.12 number stands; the Cluster-K-v2 visual pathway's independent contribution is unquantified. See `research/findings/2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md`.
- Existing runner: `research/runners/g11_bg_runner.py`
