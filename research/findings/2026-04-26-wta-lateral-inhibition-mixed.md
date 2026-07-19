# Motor WTA Lateral Inhibition — Implementation Works, but Exploitation/Exploration Trade-off Hurts Readaptation

**Date:** 2026-04-26
**Status:** PARTIAL — implementation correct + verified at probe level, but acid test shows mixed result. Kept opt-in (`--motor-lateral-inhibition`); NOT made default.
**Branch:** `lateral-inhibition`
**Companion:** [Phase B real-win](2026-04-25-phase-b-acid-test-real-win.md), which identified motor lateral inhibition as one of three "what's left" follow-ups.

## TL;DR

Added FS-mediated motor lateral inhibition (cortical WTA microcircuit) per the Phase B follow-up plan. Diagnostic probes show it works as designed: from perfectly equal cortex drive, motor pool asymmetry jumps from **1.06× → 1.77×**. Decisive winners emerge.

Acid test (3 seeds × 1800 steps, moving goal):
- **Phase 0 (initial goal acquisition): -31% finalQ** (better — agent locks onto goal faster)
- **Phase 1 (goal-change readaptation): +40% finalQ** (worse — locks in old policy, slow to switch)
- BG-active rate ~unchanged (22–24% in both)

This is a real biological trade-off (exploitation vs exploration). Strong WTA commits faster to the current best action; that hurts when the optimal action changes mid-episode. Cortical WTA in real brains is gated by neuromodulators (DA, NE) so the system can RELAX commitment under task uncertainty. We don't have that gating yet — flat WTA is too rigid for moving-goal tasks.

**Decision:** keep `--motor-lateral-inhibition` flag opt-in (default OFF). Document trade-off. Next step: revisit after per-action DA targeting (#2 in the Phase B follow-up plan), which might enable adaptive WTA via DA-gated FS strength.

## Implementation

### Architecture

Added 4 motor_FS_X regions (5 FS interneurons each, exc_fraction=0.0) plus 16 pathways:
- `motor_X → motor_FS_X` (excitatory; motor's own activity drives its FS), weight=50, density=1.0
- `motor_FS_X → motor_Y` for Y != X (inhibitory; FS suppresses other motors), weight=20, density=1.0

Sign comes from source region's exc_fraction: motor_FS_X (exc=0.0) → outgoing synapses inhibitory.

```
motor_N ──exc──► motor_FS_N ──inh──► motor_E, motor_S, motor_W
motor_E ──exc──► motor_FS_E ──inh──► motor_N, motor_S, motor_W
... (4 actions × 3 cross-pool inhibition pathways = 12)
```

Total: +4 regions, +16 pathways, +20 neurons. Built into `g11_bg_runner.build_bg_brain_regions()` as opt-in kwarg `enable_motor_lateral_inhibition` and CLI flag `--motor-lateral-inhibition`.

### Weight tuning (validated on `probe_bg_wta_ambiguous`)

Initial weights of motor_to_fs=10, fs_to_motor=5 left FS pool subthreshold (FS firing ~0.4 Hz, no inhibition). Probed with stronger values:
- motor_to_fs=50: FS pools fire 25-30 spikes per 50ms (active)
- fs_to_motor=20: enough inhibition to break symmetric tie

Static probe (single-pool drive, target=W) still produces clean selection: motor_W=6.6Hz, others <0.4Hz.

## Diagnostic results

`research/probe_bg_wta_ambiguous.py`: drives cortex_N AND cortex_E equally at 800 pA, measures motor pool firing for 500 ms with plasticity off, no OU noise.

| Drive | WTA | motor_N | motor_E | Asymmetry (max/min) |
|-------|-----|---------|---------|---------------------|
| Equal (800/800) | OFF | 34 | 32 | 1.06× |
| Equal (800/800) | ON  | 13 | 23 | **1.77×** |

Symmetric input produces decisive output via WTA. E happens to win in this seed (initial conditions); winner identity is set by random initial conditions / STN dynamics, but it IS decisive rather than ambiguous.

## Acid test (3 seeds × 1800 steps moving goal)

Compared seed-matched WTA-on vs WTA-off (the no-WTA baseline from `2026-04-25-phase-b-acid-test-real-win.md`):

| Seed | Phase 0 finalQ |  | Phase 1 finalQ |  |
|------|---------------:|---|---------------:|---|
|      | no-WTA | WTA | no-WTA | WTA |
| 42   | 3.39   | **1.45** ↓ | 1.64 | **2.09** ↑ |
| 43   | 1.72   | 1.81 ≈ | 1.93 | **3.50** ↑↑ |
| 44   | 5.33   | **3.92** ↓ | 1.71 | 1.79 ≈ |
| **avg** | **3.48** | **2.40** **(-31%)** | **1.76** | **2.46** **(+40%)** |

Phase 1 BG-active rates (real BG-driven trials, not random fallback): 22-24% across all 6 runs. WTA does NOT meaningfully change activation rate; it changes the commitment of selection within active trials.

## Interpretation

The result is consistent with a known property of WTA microcircuits in cortex: they accelerate decision-making by amplifying small input asymmetries into decisive winners. This is great for stable tasks but produces a "winner lock-in" failure mode when the optimal action changes. Real brains gate WTA strength via neuromodulators:
- High DA / certainty → strong WTA (commit, exploit)
- Low DA / uncertainty / unexpected outcome → weak WTA (relax, explore)

Our current flat WTA has no such gate, so it amplifies the phase-0 weights exactly when phase 1 needs them to be plastic.

## What this means for the original "what's next" plan

In Phase B's follow-up doc, three priorities were listed:
1. ✅ **Lateral inhibition** — implemented, but not a free win on the moving-goal benchmark
2. **Per-action dopamine targeting** — now MORE interesting: a DA-gated WTA could adapt strength based on confidence
3. **Real sensory cortex with learned position encoding** — independent of WTA outcome

So the natural next move is #2: target DA per-action-pool, then optionally re-enable WTA gated by DA concentration. That gives biology its full credit/exploration cycle:
- Action A succeeds → DA up at D1_A and FS_A → WTA strengthens around A
- Action A fails (negative reward) → DA down at FS_A → WTA weakens, other actions get a chance

## Files

- `research/runners/g11_bg_runner.py:54-65`: kwargs for WTA opt-in
- `research/runners/g11_bg_runner.py:259-296`: WTA region + pathway construction (gated by `enable_motor_lateral_inhibition`)
- `research/probe_bg_wta_ambiguous.py`: diagnostic for symmetric-drive disambiguation
- `research/findings/raw/g11_bg/g11_seed{42,43,44}_wta.json`: full acid test data

## Decision

- Keep `--motor-lateral-inhibition` flag, opt-in, **default OFF**.
- Default `motor_to_fs_weight=50`, `fs_to_motor_weight=20` (validated on probe).
- This finding gets cross-referenced from Phase B follow-up plan + SCIENCE_ROADMAP §4.7.
- Pivot to #2 (per-action DA targeting) as the next experiment.
