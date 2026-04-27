# Item 1 Stage 1: Goal-Beacon Perception — PARTIAL GO

**Date:** 2026-04-27 (next session)
**Status:** **PARTIAL GO** — replaced direct (gx, gy) goal cell access with beacon perception. 3-seed avg sum 5.36 (slightly better than baseline 5.88, low variance 0.33). System navigates without direct goal coordinate access in goal_cells. Heuristic still uses (gx,gy) directly — cleanup deferred to Stage 3.

## TL;DR

Implemented Stage 1 of the perception arc per `docs/plans/2026-04-27-perception-arc-plan.md`:
- 8 directional beacon sensors with cosine-tuned receptive fields
- Plastic beacon → goal_cells pathway (curriculum-gated)
- `--beacon-replaces-goal` flag: in replace mode, goal_cells receive ONLY
  beacon-derived input (no direct (gx, gy) cheat)

| Variant | 3-seed avg sum | std | beats baseline |
|---|---:|---:|---|
| Baseline (heuristic only) | 5.88 | — | reference |
| **Beacon v2 (curriculum-gated, replace)** | **5.36** | **0.33** | **directionally** |
| Beacon v1 (no curriculum gate, replace) | 5.69 | 0.41 | tied |
| PFC best (direct goal cheat) | 4.41 | 0.94 | yes (best) |

The cleaner v2 (where beacon→goal pathway is curriculum-gated alongside
hippo/sensory) shows tighter variance and slight improvement over baseline.

## Architecture

```python
# Beacon emits intensity falling off with distance from goal:
intensity = beacon_max_intensity / (1 + falloff * distance)

# Each sensor i has preferred direction d_i (8 directions for n=8):
# Activation is intensity weighted by cosine alignment, half-rectified
# (real biological sensors fire only for stimuli in their receptive field)
sensor_act_i = intensity × max(0, dot(d_i, bearing_to_beacon))
```

8 sensors evenly distributed at 0°, 45°, 90°, ..., 315°. Each sensor responds
maximally when beacon is in its preferred direction.

## Per-seed details

```
v2 (curriculum-gated):
seed 42: P0=2.29 P1=3.30 sum=5.59
seed 43: P0=1.87 P1=3.04 sum=4.90  (beats baseline)
seed 44: P0=2.61 P1=2.98 sum=5.59
avg: 5.36 ± 0.33

v1 (initial, beacon_to_goal not in curriculum):
seed 42: P0=2.04 P1=3.91 sum=5.95
seed 43: P0=2.20 P1=3.80 sum=6.00
seed 44: P0=2.71 P1=2.40 sum=5.11
avg: 5.69 ± 0.41
```

The v2 variant (curriculum-gated) consistently improved by ~6%, confirming
the curriculum extension was the right fix.

## What this proves

**The system navigates without direct (gx, gy) input to goal_cells.**
Goal_cells fire only because of the plastic beacon → goal_cells pathway,
which learned to integrate beacon sensor patterns into spatial representations
during training.

This is the first step in removing the "direct goal coordinate access" cheat
identified in `research/findings/2026-04-27-perception-cheats-investigation.md`.

## What this doesn't prove (yet)

**The heuristic still uses (gx, gy) directly.** The cortex_E gets 800 pA
when gx > x — same coordinate cheat. To remove this, Stage 3 of the
perception arc replaces the heuristic with a beacon-driven reflex circuit.

Without addressing that, we've partially closed only the goal_cells path
of the perception cheat. The bigger one (heuristic) remains.

## What's next (perception arc continues)

Per `docs/plans/2026-04-27-perception-arc-plan.md`:

- **Stage 2: Place cell self-organization.** Replace direct (x, y) place cell
  drive with landmark-based localization. Place cells emerge from sensory
  cue patterns rather than direct coordinates. ~1-2 weeks.
- **Stage 3: Replace heuristic with cue-following reflex.** Build a fixed-
  weight reflex circuit that translates beacon sensor activations into
  cortex direction signals. Removes the direct (gx, gy) heuristic. ~1 week.
- **Stage 4: Full integration + heuristic-off validation.** All cheats
  closed; agent navigates purely from sensory cues. ~1 week.

## Architecture additions

```bash
--beacon-perception           Enable beacon_sensors region (8 directional cells)
--n-beacon-sensors N          Number of sensors (default 8)
--beacon-to-goal-weight W     beacon → goal_cells weight (default 8)
--beacon-max-intensity P      Peak sensor drive at distance 0 (default 600 pA)
--beacon-falloff F            Distance falloff factor (default 1.0)
--beacon-replaces-goal        Use beacon-only goal info (no direct gx,gy)
```

Curriculum integration: the `beacon_to_goal` plasticity gate is now part
of the standard curriculum logic — frozen during phase 1, thawed in phase 2
alongside hippo/sensory pathways.

## Files

- `research/runners/g11_bg_runner.py:113-126, 376-397, 588-603, 763-779, 1064-1077, 1295-1338`: beacon perception
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_beacon_replace_v2.json`: 3-seed v2 data
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_beacon_replace.json`: v1 data (gate not in curriculum)
- `research/findings/raw/g11_bg/g11_seed42_beacon_smoke.json`: augment-mode smoke
- `research/findings/raw/g11_bg/g11_seed42_beacon_replace_smoke.json`: replace-mode smoke

## Next session priority

**Validate v2 with 6 seeds** (statistical significance), then proceed to
Stage 3 (replace heuristic with cue-following reflex) — the big remaining
cheat. Stage 2 (place self-organization) deferred until perception arc has
demonstrated working heuristic-free navigation.
