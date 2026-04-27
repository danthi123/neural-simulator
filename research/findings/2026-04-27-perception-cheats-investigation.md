# Perception Cheats Investigation — simple param changes don't enable heuristic-free navigation

**Date:** 2026-04-27 (Item 1 from session priority list)
**Status:** **CONFIRMED CEILING** — increasing initial plastic input weights (10 → 25 → 50, capped at stdp_w_max=30) doesn't enable navigation without the heuristic. The architectural change required is bigger than parameter tuning.

## TL;DR

Tested whether stronger plastic input weights (sensory_to_cortex_weight=25 or 50)
let the system navigate after the heuristic is turned off. Result: identical
collapse across all weight values:

| Variant | Steps 1500-1800 (heuristic on) | Steps 1800-2100 (heuristic off) | Steps 2100-2400 |
|---|---:|---:|---:|
| weight=10 (default) | 1.67 | **4.61** | **5.42** |
| weight=25 | 2.69 | **4.61** | **5.42** |
| weight=50 (capped to 30 by stdp_w_max) | 2.28 | **4.62** | **5.42** |

Random-walk distance on 8×8 is ~5.5. The agent collapses to random walk
regardless of input weight magnitude.

## Why parameter changes don't help

The heuristic provides a fundamentally different signal than plastic input
layers:
- **Heuristic**: 800 pA on ONE cortex pool, 0 on the other three. Clean 1-of-4
  selectivity. Magnitude scales linearly.
- **Plastic input layers**: graded drive across all 4 cortex pools (random
  initial weights, slightly biased after training). All pools get drive.

The cascade depends on **selectivity**, not just magnitude. Even with strong
input layer weights, all four cortex pools receive similar drive, so none
wins decisively, so the cascade can't select an action.

To get true asymmetry without heuristic, the system needs ONE of:
1. **Sparse 1-of-N input encoding** (e.g., 4 cardinal-direction sensors instead
   of 49 (dx, dy)-tuned ones, with each sensor exclusively firing for its
   direction). Re-introduces a hand-designed perception primitive.
2. **LTD for inactive pathways** (heterosynaptic plasticity). Currently STDP
   only does LTP+LTD via timing; inactive pathways have no spike events to
   trigger LTD. Would need an explicit decay rule.
3. **WTA at cortex level** — TESTED on 2026-04-27, causes readaptation
   penalty (motor-WTA pattern recurring at cortex layer).
4. **A different cascade architecture** that doesn't require razor-clean 1-of-N
   cortex selectivity (distributed action codes, continuous cortex). Major
   architectural rethink.
5. **Real perception from raw sensory** (V1-style features → direction cells →
   cortex). Multi-week implementation; the deepest fix.

## Biological reframing

Real animals don't navigate purely from learned associations either. They
have innate sensorimotor primitives:
- "Approach light" / "follow scent gradient" reflexes
- Vestibular reflexes
- Looming detection
- Optic flow integration

Our heuristic ("if gx > x: cortex_E gets 800 pA") is biologically less defensible
because it operates on **abstract coordinates** the agent doesn't actually sense.
But the IDEA of an innate sensorimotor primitive is biologically real.

The truly biological version would be:
1. Goal emits a sensory cue (light, scent, sound)
2. Agent has innate detectors for the cue
3. Innate reflex: "approach cue" (this is the heuristic, but operating on
   perceived cues, not coordinates)
4. Plastic layers refine the reflex with learning

This is the long-term direction but requires implementing the sensory
modality + cue generation in the environment. Multi-week work.

## What was added this session

- `--sensory-to-cortex-weight W` CLI flag
- `--hippocampus-to-cortex-weight W` CLI flag
- These propagate from CLI through `run_moving_goal_episode` to
  `build_bg_brain_regions`

Useful for future experiments tuning input layer strength.

## Decision

- The simple parameter-tweaking approach to "remove the heuristic" is
  closed off (this session's investigation).
- Item 1 (perception cheat removal) is now scoped as a multi-week
  architectural project. Subgoals identified:
  1. Replace abstract goal coords with sensory cues (environment change)
  2. Implement raw sensory → V1-style features → direction encoding
  3. Implement innate cue-following reflex as the new "heuristic"
- For now, the heuristic stays as biologically-defensible innate scaffolding.
- Continuing to other items (working memory, replay, multi-modal, etc.).

## Files

- `research/runners/g11_bg_runner.py:1334-1338`: new CLI weight flags
- `research/findings/raw/g11_bg/g11_seed42_w25_heuoff.json`: weight=25 test
- `research/findings/raw/g11_bg/g11_seed42_w50_heuoff.json`: weight=50 test

## Lesson

When you can't get a result from parameter tuning, the architecture is the
ceiling. This is the third time tonight that lesson has appeared (drive-gated
curriculum, WTA-based selectivity, weight-magnitude tweaks). The pattern is
clear: solve architectural problems with architecture changes, not config
changes.
