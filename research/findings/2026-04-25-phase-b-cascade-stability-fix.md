# Phase B Cascade-Stability Fix — n_cortex=400 over-drives D1, breaks GPi inhibition

**Date:** 2026-04-25
**Status:** Root cause found. 600-step verify GO. Full 3-seed run in progress.
**Branch:** `pfc-working-memory`

## The bug

The Phase B.T6 acid test [previously documented overstated](2026-04-25-phase-b-honest-correction.md): motor counts were 0 in 1799/1800 trials, agent random-walked. The static probe (`--probe-action W`) worked fine. Trial-loop runs failed.

**Root cause:** the moving-goal runner used `build_bg_brain_regions(n_cortex=400)` (100 cortex neurons per action). The static probe used the default `n_cortex=100` (25 per action). With 4× more cortex neurons projecting to the same fixed-size striatum:

- D1 firing rate jumps from ~75 Hz → ~220+ Hz (saturated, unphysiological)
- GPi gets over-driven by hyperdirect-path STN excitation past what D1 inhibition can suppress
- gpi_X = 130-230 spikes per 50ms (vs target 0)
- thal_X stays inhibited → motor_X never fires

## How it was found

A bin-by-bin probe (`research/probe_bg_500.py`) replicated the static probe pattern exactly but with `n_cortex=400`:

```
Bin 0 (steps 0-50):    d1_N=680 gpi_N=160 thal_N=10 motor_N=0
Bin 1 (steps 50-100):  d1_N=550 gpi_N=230 thal_N= 0 motor_N=2
Bin 2-9:                              gpi_N=130-220 motor_N=0
```

D1 fires at saturation; GPi never silences. With `n_cortex=100`:

```
Bin 0 (steps 0-50):    d1_N=189 gpi_N=  0 thal_N=20 motor_N=0
Bin 1 (steps 50-100):  d1_N=154 gpi_N=  0 thal_N=10 motor_N=10
Bin 2-9:               d1_N=155-180 gpi_N=0 thal_N=10 motor_N=1-5
```

D1 fires at 75 Hz (physiological MSN range), GPi correctly silenced, motor_N fires sustained. **The cascade works as designed when the input ratio is right.**

## Why this matters

Two months of "silent-motor trap" work plus one Phase B BG circuit build, and the architecture was actually correct all along — but tested at one input scale (small probe) and deployed at another (big trial run). The `--probe-action W` working at `n_cortex=100` validated the BG cascade, but the moving-goal runner I wrote used `n_cortex=400` and was effectively running the cascade in saturation.

This is a pure scaling issue, not a circuit-design issue. Striatal MSN's are designed (biologically) to fire ~10-50 Hz; pushing them to 200+ Hz means every spike falls into refractory period, lateral inhibition saturates, and downstream signaling depends on instantaneous firing rate that's pinned at the model's max.

## Verification (600-step quick run, seed 42)

With n_cortex=100 + plasticity re-enabled + reward-hold re-enabled:

```
[g11 seed=42] step 100/600  pos=(2,7)  goal=(6,6)  recent_dist=5.47  actions= 24N/ 25E/ 21S/ 30W
[g11 seed=42] step 200/600  pos=(6,1)  goal=(6,6)  recent_dist=5.18  actions= 20N/ 31E/ 27S/ 22W
[g11 seed=42] step 300/600  pos=(7,4)  goal=(6,6)  recent_dist=5.05  actions= 28N/ 26E/ 28S/ 18W
[g11 seed=42] step 300: GOAL CHANGED to (1, 6)
[g11 seed=42] step 400/600  pos=(2,4)  goal=(1,6)  recent_dist=4.57  actions= 25N/ 23E/ 21S/ 31W
[g11 seed=42] step 500/600  pos=(0,4)  goal=(1,6)  recent_dist=3.97  actions= 36N/ 19E/ 21S/ 24W
[g11 seed=42] step 600/600  pos=(5,6)  goal=(1,6)  recent_dist=4.25  actions= 29N/ 29E/ 18S/ 24W

Phase 0 goal=[6, 6] meanD=5.23 finalQ=4.99 actions=[72, 82, 76, 70]
Phase 1 goal=[1, 6] meanD=4.26 finalQ=4.83 actions=[90, 71, 60, 79]
```

Per-100-step action counts are all 18-36 (none near 0). All 4 motors firing throughout. Phase 1 actions [90N, 71E, 60S, 79W] biased toward N+W, which is the correct direction for goal=(1,6) reaching from various positions.

Phase 1 finalQ=4.83 vs G9 baseline 6.74 = 28% improvement. **And it's REAL motor activity** (not random fallback).

## Files

- `research/runners/g11_bg_runner.py` line 310: changed `n_cortex=400` → `n_cortex=100`
- `research/probe_bg_500.py`: bin-by-bin diagnostic probe
- `research/probe_bg_minimal.py`, `research/probe_bg_trial_structure.py`: earlier diagnostic attempts

## Next

- Full 3-seed run at 1800 steps (seeds 42/43/44 in progress)
- If all 3 show phase 1 finalQ < 6.0 with sustained motor activity: clean Phase B.T6 win
- If the result is still mostly random-walk: need stronger learning loop (per-action DA, eligibility tuning)

## Lesson

Static probe + acid test must use IDENTICAL configurations. The probe's correctness doesn't transfer if the deployed scale differs. From now on: the smoke probe runs at the same `n_cortex` as the moving-goal scenario.
