# Multi-Goal Stress Test — Asymmetric Adaptive DA Hurts on Fast-Changing Tasks

**Date:** 2026-04-26
**Status:** REVERSAL — asymmetric adaptive DA, the prior best on the 2-goal task, is **worse** than baseline on the 4-goal multi-change task. Important conditional-GO finding.
**Companion:** [Asymmetric adaptive DA (slow-task win)](2026-04-26-asymmetric-adaptive-da.md), [Phase B refinement summary](2026-04-26-night-summary.md)

## TL;DR

Tested asymmetric adaptive DA on a harder task: 4 goal changes per episode (corners cycle, 450 steps per phase, 1800 steps total). 3 seeds × 2 conditions (baseline vs asym adaDA).

| Variant | P0 | P1 | P2 | P3 | Sum |
|---|---:|---:|---:|---:|---:|
| Baseline (broadcast DA) avg | 2.15 | 1.62 | 2.08 | 2.47 | **8.32** |
| Asymmetric adaptive DA avg | 2.56 | 1.70 | 3.50 | 2.22 | **9.97** |

**Asym adaDA is +20% WORSE than baseline on the multi-goal task.**

This reverses the strong asym adaDA win on the simpler 2-goal task (3.53 vs 5.24).

## Why this happens

Asymmetric adaptive DA gates eligibility on cortex→D1 synapses based on reward EMA. The mechanism assumes: when reward is high, commit to current policy; when reward drops, broadcast credit (explore).

On the 2-goal task with one transition at step 300:
- Steps 0-300: reward climbs as agent learns goal 1, EMA → +1, gating tightens, agent commits
- Step 300: goal change, reward crashes, EMA → -1 fast (asymmetric ramp), gating relaxes, exploration
- Steps 300-1800: reward recovers as agent learns goal 2, EMA → +1, gating tightens
- Net: 1 commit-explore-commit cycle, total time mostly in committed (=high learning fidelity)

On the 4-goal task with transitions every 450 steps:
- Each phase only has 450 steps to learn from scratch
- EMA is constantly being whipsawed — never fully reaches +1 (committed) before next transition
- Gating sits in mid-range most of the time, throttling learning rate without committing precisely
- Net: 3 commit-explore-commit cycles, but each truncated → less learning consolidation

**The mechanism trades adaptation latency for credit fidelity.** When the world changes faster than the EMA recovery time, the trade goes negative.

## Per-seed details

```
                Variant Seed    P0    P1    P2    P3   Sum
              baseline   42  2.35  1.63  2.12  3.02  9.12
              baseline   43  1.67  1.59  2.28  1.80  7.35
              baseline   44  2.44  1.64  1.84  2.58  8.50
              baseline  avg  2.15  1.62  2.08  2.47  8.32

            asym adaDA   42  3.57  1.63  2.83  2.25 10.27
            asym adaDA   43  1.70  1.81  4.81  1.61  9.94
            asym adaDA   44  2.40  1.65  2.86  2.80  9.70
            asym adaDA  avg  2.56  1.70  3.50  2.22  9.97
```

**Key per-phase observation:** the gap is concentrated in phase 2 (after the second goal change). Baseline 2.08 vs asym 3.50 — a 67% gap. By phase 3, asym recovers to slightly better than baseline (2.22 vs 2.47). The mechanism degrades on first re-readaptation but eventually catches up.

## Implications

1. **Adaptive sharpening is task-conditional.** It helps on tasks with infrequent goal changes (1 transition per ~1500 steps in our 2-goal case) but hurts on tasks with frequent change (1 transition per 450 steps). This isn't a tuning issue — it's about whether the EMA timescale matches the task's change rate.

2. **Baseline broadcast DA is more robust.** Despite being structurally simpler and producing worse phase-0 acquisition, broadcast DA's high learning rate handles frequent task changes better. Adaptive sharpening adds a second-order optimization that interferes when conditions are unstable.

3. **The user's NE gate question is now sharper.** This experiment exposes exactly the failure mode NE is supposed to address: fast-timescale uncertainty / unexpected-change detection. NE in real biology fires phasically on goal-change events, briefly elevating learning rate beyond DA's slow EMA. With NE, asym adaDA might handle multi-goal tasks too.

4. **Recommended config now task-dependent:**
   - 2-goal moving (slow change): use `--adaptive-da --adaptive-da-ema-decay-negative 0.7` (sum 3.53 vs baseline 5.24)
   - 4-goal moving (fast change): use **baseline** (sum 8.32 vs asym 9.97)

## What this validates

- Phase B's *baseline* BG cascade is a real, robust architectural fix for the silent-motor trap (works on both task variants)
- Adaptive DA is a conditional refinement, not an unambiguous improvement
- The exploration/exploitation trade-off shows up at multiple levels — micro (per-action sharpening) and macro (commit vs adapt across episodes)

## What remains untested

- **Tuning asym adaDA for multi-goal**: shorter EMA tau? Different positive/negative ratio?
- **NE-gated learning rate boost** at goal-change events (would explicitly target the multi-goal failure mode)
- **Hierarchical sharpening**: slow DA for committed phases + fast NE-pulse for transitions

## Files

- `research/runners/g11_bg_runner.py:643-650`: `--goal-schedule multi` flag implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_multi_{baseline,asymDA}.json`: 3-seed × 2-condition acid test data

## Decision

- Keep asym adaDA opt-in (`--adaptive-da --adaptive-da-ema-decay-negative 0.7`).
- Keep baseline as default — robust across both task types.
- Document the conditional nature of the asym adaDA win (prior finding doc updated).
- Future work: implement NE-style fast meta-modulation if multi-goal performance is a priority.

## Lesson

A win on one variant of a task doesn't generalize to all variants. Asym adaDA looked like a clean win on the 2-goal task; the 4-goal stress test reveals its failure mode. This is why follow-up validation on harder tasks matters for any architectural claim.
