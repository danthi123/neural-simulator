---
type: finding
status: contributing
date: 2026-06-08
mechanism: rpe
---

# Nav place-critic Stage-1 smoke — PARTIAL: silencing FIXED + nav excellent, but the critic stays silent (drive calibration)

**Date:** 2026-06-08
**Type:** Stage-1 smoke (seed 42) of the nav value-critic redesign (commit `27f7d79a` runner + the protected mask-fix `6f73b5f0`).

## Result (seed 42, full flagship A+E+G v2.5 + `--spiking-snc --enable-neural-critic --enable-place-goal-readout`)

| Gate | Result |
|---|---|
| (i) RUNS / SNc non-zero | **✅ True** — n_windows=1800, 6.40 spikes/window |
| (iii) SNc fires | **✅** 6.40 spikes/window (≈ Stage-A's ~7 Hz) |
| (iv) Sane nav | **✅ excellent** — final-Q mean dist **1.5**, n_steps_at_goal **828/1800**, overall mean **2.13** (≈ flagship 2.57) |
| (ii) Critic learns | **❌** `cortex_it→value` weight frozen 3.011→3.011; `striosome_value` critic **never fires** (mean 0.00) |

## What this means

**The headline bug is FIXED.** The protected mask-fix (`6f73b5f0`) + the place-cell afferent
(`sensor_place_readout`, replacing the inactive ventral `cortex_it`) + the physiological GABA_B
strength (0.02) **resolved the network-silencing**: the SNc fires, the agent navigates *well*
(828/1800 steps on goal, overall distance 2.13 — comparable to the flagship). This is the decisive
contrast with the prior broken config (SNc 0 Hz, distance ~32, never reached goal).

**But the value subtraction is not yet engaged.** The `striosome_value` critic stays silent (0.00
firing) and its afferent weight never grows, so V=0 → no GABA_B inhibition reaches the SNc → the SNc
fires the *raw-reward* RPE (effectively Stage-A behavior). Nav works *because* the raw-reward RPE
still drives learning — but the brain-based-only **value subtraction** (the whole point of the
redesign) isn't happening.

## Root cause (calibration, not a fundamental block)

The critic neurons are `IZH2007_STRIATAL_MSN_D1` — MSNs have a depolarized firing threshold and need
strong, convergent drive to fire (the "up-state"). The `sensor_place_readout → striosome_value`
afferent at `weight_mean=3.0` is too weak to fire them in nav. The CPU de-risk
(`snc_stageb_critic_probe_place.py`) drove the critic *harder* (cue weight ~20 + a strong place
current) and the critic learned value-of-location robustly (3/3) — so the **mechanism is validated**;
the nav integration just needs the afferent drive calibrated to actually fire the MSN-D1 critic.

A 6-seed A/B at this state would be a **hollow pass** (no nav regression — it navigates fine — but no
neural subtraction, since the critic is silent). So the A/B is **held** until the critic fires.

## Next (calibration, runner-side only, no sim/ edits)

1. Diagnose: is `sensor_place_readout` itself firing in nav (it should, with `--enable-place-goal-readout`)?
   If yes, the afferent weight is the weak link.
2. Calibrate the critic drive to fire the MSN-D1 critic — stronger `sensor_place_readout→striosome_value`
   weight (toward the de-risk's ~15-20 range), and/or a reward-window teacher current on the critic
   (mirroring the de-risk's strong place drive), and/or a more excitable critic neuron type.
3. Re-smoke seed 42: the critic must FIRE + its weight GROW + `striov` track value, while nav stays
   sane and the SNc still fires. Then the 6-seed A/B (neural vs Stage-A).

The silencing fix is the real win banked here; the critic-firing calibration is the remaining piece to
make the value subtraction actually neural in nav.
