# DA-Gated WTA — Adaptive Inhibition Scaling, Still Net Negative

**Date:** 2026-04-26
**Status:** NEGATIVE — confirms WTA is structurally counterproductive on this task. Even with adaptive DA scaling, adding lateral inhibition net-hurts performance.
**Companion:** [Motor WTA mixed](2026-04-26-wta-lateral-inhibition-mixed.md), [Asymmetric adaptive DA](2026-04-26-asymmetric-adaptive-da.md)

## TL;DR

Implemented "DA gate" the user asked about: motor FS→motor inhibition weights are scaled per-trial by `gating_strength` (the same reward-EMA signal driving adaptive DA targeting). When agent is winning → full WTA. When losing → WTA disabled.

3-seed acid test (1800 steps moving goal):

| Variant | P0 finalQ | P1 finalQ | Sum |
|---|---:|---:|---:|
| Baseline | 3.48 | 1.76 | 5.24 |
| WTA only | 2.40 | 2.46 | 4.86 |
| Asym adaptive DA | 1.61 | 1.92 | **3.53** |
| WTA + asym DA (vanilla) | 2.05 | 2.24 | 4.29 |
| **DA-gated WTA + asym DA** | **2.12** | **2.42** | **4.54** |

DA-gated WTA is worse than vanilla-WTA + asym-DA, and both are worse than asym-DA alone. **Adding lateral inhibition is structurally negative on this task regardless of how cleverly we gate it.**

## Why this is the right negative result

The hypothesis was: WTA's exploration penalty comes from too-aggressive winner lock-in. If we relax WTA when reward drops (= goal change), the agent should adapt faster.

What actually happens: even with WTA fully disabled at low gating (verified — 600 FS→motor synapses zeroed when gating=0), the introduction of WTA when gating ramps back up still creates motor commitment that the asymmetric adaptive DA was already handling cleanly. WTA adds a redundant constraint that interacts with credit assignment in unexpected ways.

In other words: **adaptive DA already does the WTA's job from a different angle.** Per-action DA targeting selectively reinforces the chosen action's pathway → naturally produces decisive policy. Adding motor-pool inhibition on top is double-bookkeeping that hurts more than it helps.

## Mechanism (working as designed, just not useful)

```python
# At init: identify FS->motor synapses (4 actions × 5 FS × 3 other motors × 10 motor = 600 synapses)
# Save baseline weights from cp_connections.data

# Per trial start:
bridge.cp_connections.data[fs_to_motor_indices] = baseline_weights * gating_strength

# After reward:
gating_strength = clip((reward_ema + 1) / 2, 0, 1)  # reused from adaptive DA
```

When `gating=1.0`: full WTA, FS pool fully inhibits losers
When `gating=0.5`: half-strength inhibition
When `gating=0.0`: WTA disabled, all motors compete via thal drive only

Verified working at smoke test: 600 FS→motor synapses correctly identified, weights scale per-trial.

## What this means for the user's "DA gate" question

The user asked: when do we implement DA/NE gates?

This experiment is a clean answer for one specific case: **DA-gated WTA doesn't help on the moving-goal task**. The base architecture's adaptive per-action DA already provides enough explore/exploit dynamics that adding a second adaptive layer (gated WTA) is redundant.

This doesn't generalize to "DA/NE gates are useless." Other potential gates that could help:
- DA-gated learning rate (already implemented as asym adaDA — and it's the current best)
- NE-gated WTA on a HARDER task (e.g., 5+ goal changes per episode where decisive selection matters more)
- 5-HT gated discount factor for longer-horizon credit assignment

The general lesson: gates address specific failure modes. Adding gates without a specific failure mode they target produces redundancy at best, interference at worst.

## Decision

- Keep `--da-gated-wta` flag opt-in. Useful as a "what if" knob but not part of the recommended stack.
- Asymmetric adaptive DA (`--adaptive-da --adaptive-da-ema-decay-negative 0.7`) remains the recommended Phase B configuration for moving-goal scenarios.

## Files

- `research/runners/g11_bg_runner.py:498-526, 553-559, 618-621`: DA-gated WTA implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_dagatedWTA.json`: 3-seed acid test data

## Per-seed details

| Seed | P0 | P1 |
|------|---:|---:|
| 42 | 1.41 | 2.69 |
| 43 | 1.55 | 2.32 |
| 44 | 3.40 | 2.25 |
| avg | 2.12 | 2.42 |

Phase 0 is competitive with asym DA (P0 avg 2.12 vs 1.61), but phase 1 is significantly worse (2.42 vs 1.92). The gating partially helps phase 1 by relaxing WTA after goal change, but doesn't fully recover what asym DA alone achieves without WTA at all.
