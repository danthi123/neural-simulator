# Item 1: Real Perception Arc — Multi-Stage Plan

> **For Claude:** Multi-week plan for replacing the heuristic with real biology-grounded perception.

**Goal:** Replace the current "if gx > x: cortex_E gets 800pA" heuristic — which assumes the agent magically knows abstract goal coordinates — with a perception system that detects sensory cues from the environment, enabling goal-directed behavior without coordinate access.

**Architecture:** Goal-cue emitting environment → sensory neurons with biological receptive fields → V1-style feature detection → place/goal cell formation → cortex selectivity. Each layer learnable from scratch.

**Tech Stack:** Existing brain-region framework + per-pathway plasticity gating + neuromodulator subsystem. No new frameworks needed — just composition of existing tools.

---

## Why this is the biggest cheat to remove

Currently the system has 3 deeply-intertwined "perception cheats":

1. **Direct (gx, gy) goal access.** `goal_cells_(gx,gy)` fires based on the goal's coordinates being given to the agent. Real animals must perceive goal cues (light, sound, scent).
2. **Direct (x, y) position access.** `place_cells_(x,y)` fires based on agent's coordinates being given. Real animals localize from sensory cues + path integration.
3. **Heuristic cortex drive.** The runner directly maps goal-relative direction to cortex pool drive. Real animals have innate sensorimotor reflexes operating on raw sensory features (looming, optic flow, scent gradient), not abstract coordinates.

These cheats currently sidestep what would otherwise be a major fraction of the agent's neural workload. Removing them would let the system claim genuine "biology-grounded learning agent" status — currently the heuristic does the perceptual heavy-lifting.

The 2026-04-27 PFC working-memory finding showed the cascade can use plastic input layers usefully. The next step is replacing the input *content* with sensory-derived signals.

## Stage breakdown

### Stage 1: Goal-beacon perception (~1 week)

Replace direct (gx, gy) access with a "beacon" that the agent must perceive.

**Subgoals:**
1. **Environment change**: place a "beacon" at the goal position. Beacon emits a signal (e.g., scalar intensity falling off as 1/distance).
2. **Beacon sensors**: add a small region (8-16 neurons) tuned to specific (relative) positions around the agent. Each sensor fires proportional to beacon strength at its preferred relative position.
3. **Goal cells from sensor pattern**: connect beacon-sensors → goal_cells (plastic, density 1.0). With curriculum, goal_cells learn to represent the goal's relative position from sensor patterns.
4. **Test**: agent should still navigate. Performance might drop initially (uses noisier perceptual signal) but should recover with training.

**Implementation effort:** ~2-3 days
- Add `BeaconEnvironment` config (beacon position, intensity model, falloff)
- Add `beacon_sensors` region with directional tuning
- Modify trial loop to compute sensor activations from beacon position
- Tag plasticity gate `beacon_to_goal_cells` for curriculum

**Success criteria:**
- 6/6 seeds beat baseline on 2-goal task with beacon-based goal perception (sum < 5.88)
- Agent learns to localize beacon position from sensor pattern within phase 1

### Stage 2: Place cell self-organization (~1 week)

Replace direct (x, y) access with place cells that emerge from sensory cues.

**Subgoals:**
1. **Visual landmarks**: place several "landmarks" at fixed positions in the environment. Each landmark emits a unique signal.
2. **Landmark sensors**: agent has direction-tuned sensors that fire based on visible landmarks (using bearing + distance).
3. **Place cells from landmark patterns**: place_cells receive plastic input from landmark sensors. Through self-organization (Hebbian or sparse coding), each place cell becomes tuned to a specific (x, y) position based on the unique landmark pattern at that position.
4. **Test**: agent should navigate without direct (x, y) access. Place cells should self-organize into a spatial map.

**Implementation effort:** ~3-5 days
- Add `LandmarkEnvironment` with multiple unique landmarks
- Add `landmark_sensors` region with bearing-tuned cells
- Modify place_cells region to be plastic with landmark inputs
- Add Hebbian/competitive learning for place-cell self-organization (BCM rule or similar)
- Curriculum: landmark sensors → place cells trains first, then full system

**Success criteria:**
- Place cells develop position-selective firing fields without direct (x, y) supervision
- Agent navigates similarly to current system

### Stage 3: Replace the heuristic (~1 week)

With Stages 1+2 working, build an "innate cue-following reflex" that replaces the heuristic.

**Subgoals:**
1. **Cue-direction reflex**: hard-code (or pre-train) a small reflex circuit that translates beacon-sensor activation into approach cortex drive. Like real animals' "approach light/scent" reflex.
2. **Cortex selectivity from reflex**: reflex drives cortex_X based on the direction of the beacon (computed from sensor pattern), with magnitude proportional to beacon strength.
3. **Test**: same task, but heuristic replaced with reflex. Should perform comparably (slightly worse since reflex uses inferred direction vs known direction).

**Implementation effort:** ~3-4 days
- Build the reflex circuit (could be a small fixed-weight pathway)
- Wire beacon sensors → reflex → cortex
- Remove direct heuristic from runner

**Success criteria:**
- Agent navigates without direct (gx, gy) access AND without the original heuristic
- Performance within 30% of current breakthrough config (sum 4.41)

### Stage 4: Full integration and validation (~1 week)

Combine Stages 1-3 with existing PFC + curriculum + sleep-replay infrastructure.

**Subgoals:**
1. **Curriculum sequence**: phase 1 trains cortex+heuristic-equivalent; phase 2 trains place/goal cell self-organization; phase 3 fine-tunes the full system.
2. **Heuristic-off test**: with the reflex in place, test if the system maintains navigation under varying beacon strengths (analogous to "if I can barely smell the food, can I still find it?").
3. **6-seed validation**: full configuration on 2-goal task vs baseline vs current-best.

**Success criteria:**
- 6-seed sum at most 1.5× the current best (4.41) — i.e., sum < 6.6 — without ANY of the perception cheats
- Place cells visibly self-organize into spatial maps
- Agent demonstrates landmark-based navigation

## Risk assessment

**Likely problems:**
1. **Place cell self-organization is hard.** Without a good unsupervised learning signal, place cells might not develop useful tuning. Fallback: use sparse coding loss or competitive learning.
2. **Reflex tuning.** Getting the reflex strong enough to drive navigation but weak enough not to dominate is delicate.
3. **Compound errors.** Each stage adds noise; by stage 4 the cumulative error might be large.

**Mitigations:**
- Per-stage validation: each stage must work standalone before composing
- Conservative parameter tuning: start with values close to current working config
- Comparison baselines: at each stage, compare to current-best to track regression

## Decision points

After Stage 1 (goal-beacon):
- If it works (sum < 5.88 with beacon perception): continue to Stage 2
- If marginal (5.88-7.0): tune beacon parameters, try denser sensors
- If broken (>7.0): may need different perception architecture; consider attention mechanisms

After Stage 2 (place self-organization):
- If place cells visibly tune to positions: continue to Stage 3
- If maps are diffuse: try competitive learning rule (winner-take-all on place cells)
- If broken: may need pre-supervised pretraining of place cells

After Stage 3 (reflex):
- If navigation survives: full system without coordinate access — major win
- If reflex too weak: increase reflex strength, accept it as biological "innate prior"

## Files to create/modify

```
# New
research/runners/g11_perception_runner.py           # standalone runner for perception arc
sim/environment.py                                   # beacon + landmark environment
sim/perception_regions.py                            # beacon_sensors, landmark_sensors regions
research/findings/2026-04-XX-stage1-beacon.md
research/findings/2026-04-XX-stage2-place-self-org.md
research/findings/2026-04-XX-stage3-reflex.md
research/findings/2026-04-XX-stage4-full-perception.md

# Modify
research/runners/g11_bg_runner.py                   # add perception flags + region wiring
sim/regions.py                                       # may need new pathway types
tests/test_perception.py                             # new test suite
```

## Estimate: 4 weeks total

- Stage 1: 1 week
- Stage 2: 1-2 weeks (most uncertain)
- Stage 3: 1 week
- Stage 4: 1 week (validation)

Stages 1-2 can potentially be parallelized after a clean abstraction layer is in place.

## Why this matters

The user's stated goal is "biologically accurate human neural network." The current system's biggest claim-vs-reality gap is the heuristic + coordinate access. Closing this gap moves the project from "biologically-inspired" to "biologically-grounded" — meaning the agent really does have to perceive its environment to act, the way a real organism does.

This is the natural culmination of all the infrastructure built so far:
- Per-pathway plasticity gating (2026-04-27) — needed for curriculum-based perception training
- Real curriculum (2026-04-27) — staged plasticity for perceptual learning
- PFC working memory (2026-04-27) — maintains goal info across delays (essential when perception is noisy)
- Brain-region framework (2026-04-24) — composable architecture for new regions
- Neuromodulator subsystem (2026-04-24) — gates plasticity by behavioral state

All the pieces are in place. The arc is well-scoped. Total effort ~4 weeks of focused work.
