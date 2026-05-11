# P5 iter B seed 42 FAIL — strict two-stage gating doesn't fix it

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3 (catalog G.11 + G.13)
**Status:** Honest report. Iter B = strict two-stage gating
(McClelland 1995 CLS) + drive_lang_during_replay (Wilson &
McNaughton 1994). Result NEARLY IDENTICAL to iter A.
Architecture at ceiling.

## Comparison across iterations (seed 42)

| Metric | Original | Iter A | **Iter B** | Target |
|---|---|---|---|---|
| apple_self cosine | 0.216 | 0.227 | **0.226** | > 0.5 |
| apple_river cosine | 0.290 | 0.174 | **0.186** | < 0.4 |
| Naming ratio | 0.89x | 1.08x | **1.08x** | > 1.3x |
| Verdict | FAIL | FAIL | **FAIL** | — |

Iter A → iter B: apple_self unchanged (0.227 vs 0.226), cross
slightly higher (0.174 vs 0.186 — both well below threshold),
naming ratio identical (1.08x both). **The architecture has a
hard floor around apple_self=0.22 with these scale parameters.**

Wall clock: 306s (5 min) for iter B with strict two-stage +
lang-driven replay (custom replay loop adds ~15s).

## Diagnostic conclusion

The 3-fail iron law (superpowers:systematic-debugging Phase 4.5)
says: "if 3+ fixes fail, question architecture vs continuing to
fix symptoms." We are now at attempt 3:

1. Original: raw spike-count cosine, 100 events. FAIL multi-seed.
2. Iter A: engram-tag methodology + 300 events. FAIL seed 42.
3. Iter B: strict two-stage gating + lang drive in replay. FAIL
   seed 42.

The signals are CONSISTENT across iterations (apple_self ~0.22
to within 0.001), suggesting the architecture has a real
ceiling, not seed-variance.

## Hypothesis: semantic_cortex lacks attractor dynamics

The Patterson 2007 ATL hub theory requires the cortex to have
**stable point attractors** — when driven by a sparse input,
the recurrent network locks into one of a small number of
attractor basins (one per concept). Our current semantic_cortex:

- 500 neurons (toy scale)
- recurrent_density=0.10 (sparse)
- recurrent_weight_mean=1.0 (vs lang->wernicke=3.0,
  wernicke->semantic=4.0) — RECURRENCE IS 3-4x WEAKER THAN
  FEEDFORWARD INPUT

Implication: every fresh drive overwrites the existing pattern.
There's no attractor to "snap back to" the trained ensemble.

## Two parallel iterations launched

**Iter C (RUNNING at seed 42):** scale only.
- n_wernicke: 200 → 400
- n_semantic_cortex: 500 → 1000
- Keep iter B fixes (strict two-stage + lang in replay)

Tests whether the wernicke bottleneck (diagnosis #2) is the
issue. If scaling fixes it, the architecture is fine — just
needed more neurons.

**Iter D (pre-staged CLI flags):** attractor tuning.
- `--semantic-cortex-recurrent-density 0.25` (vs 0.10)
- `--semantic-cortex-recurrent-weight 2.5` (vs 1.0)
- `--drive-steps 300` (vs 100; give attractor time to settle)

Tests the Patterson 2007 hypothesis: real ATL hub has dense
recurrence comparable to feedforward input.

If both iter C and iter D fail, escalate to user per iron law.

## What's NOT broken (still)

- P1 trisynaptic loop: 3/3 multi-seed PASS
- P4.1 positional binding: 3/3 multi-seed PASS
- P2 engram tagging: 12 unit tests pass
- P3.1 concept replay: 5 unit tests pass
- Iter A/B engram-tag methodology IS sound (signal direction
  correct: same-concept > cross-concept consistently)
- Substrate builds + prereq checks raise on invalid combinations
- Wall clock is fine: 5 min/seed even at 300 events

The architecture is mostly built. P5's dynamic tuning needs
deeper work than CLS-theory-grounded gating.

## Path forward

1. Wait for iter C (scale-only) seed 42 result.
2. If iter C FAIL: launch iter D (attractor tuning) seed 42.
3. If iter D FAIL: 4-fail count → architectural escalation:
   a. Different cortical scaffolding (cluster-of-concepts vs
      single ATL hub per Pulvermüller distributed grounding)?
   b. Different test methodology (per-pathway weight delta
      instead of post-test reactivation)?
   c. Different training paradigm (paired drives over 5-10K
      events to grow the wernicke->semantic weights enough to
      dominate noise)?
4. If iter C or iter D PASS: launch 43/44 for multi-seed.
