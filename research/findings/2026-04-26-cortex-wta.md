# Cortex-Level WTA — PARTIAL: selectivity fix works, but readaptation penalty kicks in

**Date:** 2026-04-26 (first structural attempt at the plastic-input-layer ceiling)
**Status:** **PARTIAL** — improves over hippocampus-alone by ~16%, but still 1.6× worse than baseline. Same exploitation+/readaptation- trade-off as motor WTA at cortex level.
**Companion:** [Hippocampus additive fail](2026-04-26-hippocampus-additive-fail.md), [Motor WTA mixed result](2026-04-26-wta-lateral-inhibition-mixed.md)

## TL;DR

Cortex-level lateral inhibition (per-pool FS interneurons mirroring the
motor WTA pattern) was the most direct structural attempt at the
plastic-input-layer ceiling identified in the 4-NEGATIVE arc earlier
today. Hypothesis: WTA enforces one-cortex-pool-wins regardless of input
noise → plastic input layers can coexist with heuristic.

**Result: partial fix.** Cortex WTA + hippocampus improves over
hippocampus-alone (sum 9.26 vs 10.98, ~16%), but still substantially
worse than the heuristic-only baseline (5.88). The exact same
exploitation+/readaptation- pattern as motor WTA: phase 0 acquisition is
fine, phase 1 readaptation collapses.

## Result table

| Variant | P0 finalQ | P1 finalQ | Sum | n_at_goal P1 |
|---|---:|---:|---:|---:|
| Baseline (heuristic only) | ~3.0 | ~2.9 | **5.88** | (reference) |
| WTA only (seed 42) | 2.71 | 3.73 | **6.44** | (control) |
| WTA + hippo seed 42 | 4.41 | 3.86 | 8.27 | 53 |
| WTA + hippo seed 43 | 2.55 | 6.36 | 8.91 | 25 |
| WTA + hippo seed 44 | 5.93 | 4.68 | 10.62 | 44 |
| **WTA + hippo 3-seed avg** | **4.30** | **4.97** | **9.26** | 41 |
| (recall) Hippo additive (no WTA) | 5.85 | 5.13 | **10.98** | ~17 |

## What WTA fixes

The smoke test caught it cleanly: at 200 steps with WTA+hippo, action
counts were 53N/55E/44S/48W (favoring goal NE), vs hippo-alone where
they were ~uniform (no directional preference). WTA does enforce cortex
pool selectivity.

The full acid test confirms this in phase 0: P0 finalQ=4.30 vs hippo-alone
P0 finalQ=5.85 (~26% improvement). The agent acquires the initial policy
under hippo noise where without WTA it couldn't.

n_at_goal in phase 1 is also higher than hippo-alone (41 vs ~17, 2.4×).
The cascade is firing more meaningfully — just not the *right* action
consistently.

## What WTA breaks

Phase 1 finalQ=4.97 — far worse than baseline ~2.9. P1 action counts are
still ~uniform across all 3 seeds (e.g. seed 42: 384/384/346/386). When
the goal flips from (6,6) to (1,6), WTA's commitment to the phase-0
winner fights the cascade's attempt to switch.

This is **the same pattern as motor WTA** in Session G:
exploitation+/readaptation−. Lateral inhibition makes the winner harder
to dethrone, which helps acquisition but hurts readaptation.

WTA-only control (no hippo) confirms the readaptation penalty is
*intrinsic to WTA*, not a hippo interaction effect:

| Phase | Baseline | WTA-only seed 42 |
|---|---:|---:|
| P0 finalQ | ~3.0 | 2.71 (slightly better) |
| P1 finalQ | ~2.9 | 3.73 (worse) |
| Sum | 5.88 | 6.44 (worse) |

So even WITHOUT hippocampus, cortex WTA introduces a readaptation
penalty. With hippocampus, the penalty compounds because the slow-learning
hippocampus pathway can't keep up with the fast goal-flip while WTA is
holding the agent to phase-0's learned cortex pool.

## Why the hippocampus doesn't compensate

Hypothesis: hippocampus needs >1800 trials to build up enough plastic
weight that its drive can override WTA's locked-in winner. Looking at
n_at_goal P1 = 41 (vs ~17 for hippo-alone, vs ~75-150 expected for
strong learning), the agent is making *some* progress on phase 1, but
not enough to converge in the available steps.

Even if true, this means WTA + hippo needs much longer training to
match baseline, which is a real productivity penalty.

## Architectural insight

The plastic-input-layer ceiling has now been characterized more
precisely:

1. Without WTA: random plastic weights → uniform cortex drive → cascade
   silenced (cold-start mode, sum 10.98).
2. With WTA: random plastic weights → WTA-enforced selectivity → cascade
   active, but commits to whatever wins early → readaptation penalty
   (sum 9.26).
3. Without WTA AND without plastic input: heuristic drives selectivity
   directly (sum 5.88, the baseline).

WTA is a *structural fix for the asymmetry-injection problem* but
introduces a *new structural problem* (commitment) that re-emerges
whenever the task requires re-selection.

## Decision

- Keep `--cortex-wta` flag opt-in for future combo experiments.
- **Do NOT use as default** — net regression on 2-goal task.
- The right next move is **NOT another runner-side variant** but a
  combination test: cortex WTA + adaptive DA. Adaptive DA earlier
  improved readaptation by gating eligibility based on reward EMA. If WTA
  gives selectivity AND adaptive DA gives readaptation flexibility, the
  combo might recover baseline + add hippo memory.

## Files

- `research/runners/g11_bg_runner.py:74-85, 175-189, 405-425, 478, 540, 1083`: cortex WTA implementation (committed in 511f7b2)
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_cortexwta_hippo.json`: 3-seed acid test data
- `research/findings/raw/g11_bg/g11_seed42_cortexwta_only.json`: WTA-only seed-42 control
- `research/findings/raw/g11_bg/g11_seed42_cortexwta_smoke.json`: 100-step smoke (WTA only)
- `research/findings/raw/g11_bg/g11_seed42_cortexwta_hippo_smoke.json`: 200-step smoke (WTA+hippo)

## Lesson

The four-NEGATIVE arc identified that random plastic inputs break cascade
selectivity. The structural-fix hypothesis was: enforce selectivity at
cortex via WTA. That hypothesis was *partially confirmed* — WTA does
preserve selectivity — but the fix carries a hidden cost (commitment
that fights readaptation) that re-emerges at the new layer.

Two architectural levels (motor and cortex) now show the same
exploitation/readaptation tradeoff with WTA. This is plausibly a general
property of recurrent inhibition microcircuits: they're great at picking
a winner once, less great at switching winners on demand.

The next reasonable attempt is **WTA + adaptive DA combo** — let WTA
handle selectivity within a phase, let adaptive DA's reward-EMA gating
release the commitment when reward drops. Whether this composes well or
hits a different ceiling is an open question — both have been tested
individually but not stacked on a plastic-input-layer config.

## Next move (TBD with user)

- **A** WTA + adaptive DA + hippocampus (combo test)
- **B** Curriculum learning — train fixed-goal first to lock cortex→D1,
  then thaw input layer
- **C** Pivot — accept that 2-goal+plastic-input is a hard combination
  and explore other questions (multi-goal stress test, larger networks,
  HH biophysics, etc.)
