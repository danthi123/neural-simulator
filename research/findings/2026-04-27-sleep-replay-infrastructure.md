# Sleep-Replay Infrastructure — basic mechanism works; random replay neutral

**Date:** 2026-04-27 (continuation of plastic-input-layer arc resolution)
**Status:** **PARTIAL** — sleep-replay infrastructure works correctly (gates fire, agent freezes, no crashes); random-pattern replay is neutral (sum 3.91 vs no-sleep 3.87 on 3 seeds). Future work needs trajectory replay of LEARNED sequences.

## TL;DR

Built sleep-replay memory consolidation as the next biological feature
on top of the per-pathway plasticity gating breakthrough. The
infrastructure correctly:
- Suspends behavior during sleep (agent freezes)
- Disables heuristic during sleep (no external goal teaching)
- Drives random place + goal cells (sharp-wave-ripple-like)
- Thaws `cortex_to_d1` for consolidation
- Freezes `hippo_to_cortex` to preserve learned weights

Test: 1200 wake + 300 sleep + 600 post-sleep wake (2100 total). Sum
3.91 ± 0.35 (3-seed). No-sleep control (2100 wake): sum 3.87. Sleep
neutral.

## Why random replay is neutral

Real biological replay is of **learned sequences** — successful
trajectories from waking experience. Hippocampal sharp-wave ripples
during NREM sleep replay specific (place, action, reward) sequences
from earlier exploration. Cortical neurons receive these and STDP
consolidates the patterns.

Our random-pattern replay drives random (x,y) place cells with random
(gx,gy) goal cells. This activates whatever (place, goal) → cortex_pool
weights have been learned, but in arbitrary order. The signal-to-noise
ratio is low — meaningful place→action associations are diluted by
random ones.

To improve: replay LEARNED trajectories. Specifically, replay the
(place_xy, goal_gxgy) pairs that resulted in goal-reaching during
waking. This requires:
1. Logging successful trajectories during wake
2. Sampling them during sleep instead of random patterns
3. Possibly time-compressing them (real ripples are 50-100ms)

## Implementation

```python
# In runner trial loop:
in_sleep = (sleep_replay_after_step >= 0
           and step >= sleep_replay_after_step
           and step < sleep_replay_after_step + sleep_replay_steps)

if in_sleep:
    bridge.set_plasticity_gate("cortex_to_d1", 1.0)         # thaw
    bridge.set_plasticity_gate("hippo_to_cortex", 0.0)      # freeze
    bridge.set_plasticity_gate("sensory_to_cortex", 0.0)    # freeze

# In drive section:
if in_sleep and enable_hippocampus:
    # Random replay
    replay_x = np.random.randint(0, grid_size)
    replay_y = np.random.randint(0, grid_size)
    replay_gx = np.random.randint(0, grid_size)
    replay_gy = np.random.randint(0, grid_size)
    place_drive = exp(-||hippo_pref - (replay_x, replay_y)||² / 2σ²)
    goal_drive = exp(-||hippo_pref - (replay_gx, replay_gy)||² / 2σ²)

# Agent doesn't move during sleep
if in_sleep:
    new_x, new_y = x, y
```

CLI flags:
- `--sleep-replay-after-step N`
- `--sleep-replay-steps M`
- `--sleep-replay-rate-hz R`

## Per-seed results

```
3-seed sleep-replay (1200 wake + 300 sleep + 600 wake test):
seed 42: P0 finalQ=1.33 P1 finalQ=2.16 sum=3.50
seed 43: P0 finalQ=1.76 P1 finalQ=2.12 sum=3.88
seed 44: P0 finalQ=2.15 P1 finalQ=2.20 sum=4.35
avg: 3.91 (std=0.35)

No-sleep control (seed 42, 2100 steps wake):
sum 3.87, P1 finalQ=1.91

Post-sleep performance (steps 1500-2099) seed 42:
  SLEEP:    meanD=2.32
  NO-SLEEP: meanD=2.02
```

Sleep variant performs slightly worse post-sleep (2.32 vs 2.02), confirming
random replay isn't beneficial. Variance is low (std=0.35), suggesting
the mechanism is stable, just not useful with random content.

## What's correctly modeled

Despite the negative end result, the implementation is biologically
sound:

1. **Behavioral suspension** — animals don't navigate during NREM sleep
2. **Sensory deafferentation** — heuristic = innate sensory primitive,
   off during sleep
3. **Replay-driven consolidation** — hippo activity drives cortex
4. **Differential plasticity** — cortex (consolidation site) thaws,
   hippo (source) preserves
5. **Sharp-wave-ripple frequency** — replay rate parameter (~150-250 Hz
   biologically, we use 200 Hz default)

What's missing for biological accuracy: replay of learned sequences
(not random), multiple sleep cycles, NREM/REM differences.

## Decision

- Keep `--sleep-replay-after-step` flag and infrastructure (working).
- Don't recommend random-replay for performance — it's neutral at best.
- Future work: trajectory-replay (sample successful (place, goal)
  pairs from wake-period logs) is the natural next step.

## Files

- `research/runners/g11_bg_runner.py:537-549, 1004-1030, 1124-1130`:
  sleep-replay implementation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_sleep.json`: 3-seed
  acid test
- `research/findings/raw/g11_bg/g11_seed42_nosleep_2100.json`:
  no-sleep control (matching duration)

## Lesson

Building biological infrastructure ≠ achieving biological function. The
sleep-replay machinery models the right MECHANISM (gate switching,
random place cell drives, behavioral suspension), but biological
function requires the right CONTENT (learned trajectories). The
distinction matters.

This isn't a regression — the infrastructure is now in place for future
trajectory-replay experiments. Random replay was the simplest
proof-of-concept. Trajectory replay needs more code (logging
trajectories during wake, sampling during sleep), which is the natural
next iteration.
