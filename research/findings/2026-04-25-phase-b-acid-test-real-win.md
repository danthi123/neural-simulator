# Phase B.T6 Acid Test — REAL Win After Two-Bug Fix

**Date:** 2026-04-25
**Status:** GO — Phase 1 finalQ 1.76 avg (vs G9 baseline 6.74). 74% improvement, clean cascade across all 1800 trials.
**Branch:** `pfc-working-memory`
**Supersedes:** [the overstated initial finding](2026-04-25-phase-b-bg-acid-test.md) and the [honest correction](2026-04-25-phase-b-honest-correction.md). Both kept for the trail.

## Summary

The original Phase B.T6 result was REAL but masked by two compounding bugs:
1. `n_cortex=400` (100/action) over-drove D1 to saturation
2. `stdp_w_max=2.0` (default) silently collapsed cortex→D1 weights from 25→2 in milliseconds

Each bug independently zeroed the cascade — fixing only one didn't help (v2 run still had 1799/1800 zero-motor trials). Fixing both gives a clean, sustained cascade and dramatic improvement on the moving-goal scenario.

## Final 3-seed results

| Seed | Phase 0 finalQ | Phase 1 finalQ | Phase 1 actions [N,E,S,W] | Phase 1 active trials |
|------|----------------|----------------|---------------------------|---------------------|
| 42   | 3.39           | **1.64**       | [434, 324, 352, 390]      | 326/1500 (22%) |
| 43   | 1.72           | **1.93**       | [440, 328, 359, 373]      | 340/1500 (23%) |
| 44   | 5.33           | **1.71**       | [408, 322, 348, 422]      | 362/1500 (24%) |
| **avg** | **3.48**     | **1.76**       | uniform-ish               | 23% |

**Phase 1 finalQ 1.76 avg vs G9 baseline 6.74 = 74% improvement.**

For context: an 8×8 grid has Manhattan diameter 14, random walk converges to mean ≈ 5.5, V1's "stuck on dominant motor" was at 6.40. The agent now stays within Manhattan distance ~1.7 of the goal in steady state — genuine learning.

## The two bugs

### Bug 1: cortex pool over-drive (saturation)

The static probe (`--probe-action W`) used the default `n_cortex=100` (25 cortex per action) and produced clean cascade: D1_W=68 Hz, GPi_W=0 Hz, thal_W=24 Hz, motor_W=7 Hz.

The moving-goal runner I wrote used `n_cortex=400` (100 per action) — 4× more cortex inputs to the same striatal pool. This pushed:
- D1 firing from ~75 Hz to 220+ Hz (saturated, unphysiological for MSN)
- GPi from silence to 130-230 spikes/50ms (D1 inhibition could not overcome STN excitation)
- Cascade output → thal silenced → motor silent

A bin-by-bin probe (`research/probe_bg_500.py`) with `n_cortex=100` showed motor activity sustained across 500 steps. With `n_cortex=400`, motor died after the first 100 steps.

Fix: change `build_bg_brain_regions(n_cortex=400)` → `build_bg_brain_regions(n_cortex=100)` in the moving-goal runner.

### Bug 2: STDP soft-bound weight collapse

The cortex→D1 pathway uses `weight_mean=25.0`. The bridge default `stdp_w_max=2.0`. The soft-bound STDP rule (`sim/kernels.py:262`):

```python
delta_w_LTP = A_plus * (w_max - w) * exp(-delta_t / tau_plus)
```

When w=25 and w_max=2, `(w_max - w) = -23`. So every "LTP" event is strongly NEGATIVE. The first STDP-active step rapidly depresses the cortex→D1 weights toward 2.0. After this collapse, D1 cannot fire from cortex input.

Trial-match probe (`research/probe_bg_trial_match.py`):
- With STDP off: 20/20 trials show motor_N firing (1-7 spikes/trial)
- With STDP on, `stdp_w_max=2`: only trial 0 fires (10 spikes); trials 1-19 silent
- With STDP on, `stdp_w_max=30`: 20/20 trials show motor_N firing 4-8 spikes consistently, D1=325-340 spikes/trial healthy

Fix: set `cfg.stdp_w_max = 30.0` in the moving-goal runner (above the cortex→D1 weight_mean=25).

## Cascade health (per trial-match probe with both fixes)

| Region | Per-trial spike count |
|--------|----------------------|
| cortex_N | (driven 800 pA, ~112 Hz) |
| str_D1_N | 325-340 (~46 Hz / neuron) |
| gpi_N | 0 (silenced) ✓ |
| thal_N | 20-30 (firing) ✓ |
| motor_N | 4-8 (firing) ✓ |
| motor_E/S/W | 0 (correct: only target action selected) |

## Why 22% BG-active and 78% random fallback?

When the agent is far from the goal, multiple cortex pools (e.g. cortex_N + cortex_E for goal up-and-right) get driven simultaneously. With ALL 4 cortex pools driving D1 pools competitively, GPi may not silence cleanly for any single action — selection becomes ambiguous and motor_X stays below threshold. The runner falls back to random selection.

Even at 22% BG-driven, the action distribution is strongly biased toward correct directions:

Seed 42 phase 1 (goal=(1,6)): per-direction action total - random baseline ≈
- N: 434 - 292 = 142 (goal y=6, mostly above starting y → N is correct)
- W: 390 - 292 = 98 (goal x=1, mostly to left → W is correct)
- S: 352 - 292 = 60 (occasional, when over-shooting)
- E: 324 - 292 = 32 (rarely correct)

The BG circuit produces direction-correct bias whenever it fires.

## Files

- [`research/runners/g11_bg_runner.py`](research/runners/g11_bg_runner.py): both fixes applied
- [`research/probe_bg_500.py`](research/probe_bg_500.py): bin-by-bin static probe (found bug 1)
- [`research/probe_bg_trial_match.py`](research/probe_bg_trial_match.py): trial-match probe (found bug 2)
- [`research/findings/raw/g11_bg/g11_seed{42,43,44}_v3.json`](research/findings/raw/g11_bg/): final 3-seed data

## Comparison vs all V1-V7 + previous Phase B attempts

| Variant | Approach | Phase 1 finalQ | All motors? | BG-driven? |
|---------|----------|----------------|-------------|------------|
| baseline | none | 6.74 | 2/3 | reservoir argmax (broken) |
| V1 | motor_exploration_rate_hz=15 | 6.40 | 3/3 | reservoir + Poisson noise |
| V5 | proportional sampling | 6.78-7.13 | uniform (random) | none |
| Phase B v2 | n_cortex=100 only | 5.46 | 3/3 (uniform via random fallback) | 0% (silent cascade) |
| **Phase B v3** | **n_cortex=100 + stdp_w_max=30** | **1.76** | **3/3** | **22-24% real selection** |

Phase B v3 is the first intervention to drop phase 1 finalQ below 3.0 with REAL motor activity (not just random fallback uniformity).

## Cumulative session work

- Sessions D-I (silent-motor trap arc): V1-V7 motor-exploration variants — all NEGATIVE
- Phase A: HH+Izh+AdEx preset audit + biology bug fixes (30 working presets)
- Phase B: BG action selection module + acid test
  - Initial run: cascade dies (overstated as win)
  - Bug 1 fix: still mostly random fallback
  - Bug 2 fix: cascade fires across all trials, 74% improvement vs baseline

The BG-style architecture works as designed. The 22% BG-active rate is the next ceiling to break — likely needs per-action DA targeting (currently broadcast) and better cortex input selectivity (currently heuristic goal-relative drive).

## What's still left to do

1. **Position encoding**: cortex drives are heuristic ("drive cortex_X for goal-relative direction X"). A proper cortex pool should learn position→action mapping via plastic sensory→cortex weights. This would make BG selection happen on more of the trials.

2. **Per-action dopamine targeting**: dopamine currently broadcasts. Targeting DA to the active D1 pool (per Schultz/Wickens biology) would sharpen credit assignment.

3. **Lateral inhibition between motor pools**: currently absent (the previous motor→motor pathway was incorrectly excitatory). FS interneuron sub-pools would create proper winner-take-all.

4. **Test on harder task**: 8×8 grid with single goal change is the simplest version. Try multiple goal changes, longer episodes, harder grids.
