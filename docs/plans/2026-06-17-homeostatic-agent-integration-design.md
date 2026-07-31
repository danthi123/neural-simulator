---
type: plan
status: live
date: 2026-06-17
---

# Brain-based spiking homeostatic agent — integration design

> **Status:** design for the deferred integration build. The artificial-life **motivational core is de-risked
> across all three faces** — it learns (`2026-06-17-homeostatic-drive-rl-cheap-first-GO.md`), the drive + neural
> reward work on spikes (`2026-06-17-homeostatic-spiking-drive-mechanism-GO.md`), and it sustains life
> (`2026-06-17-homeostatic-sustained-agency-GO.md`). This doc specifies the remaining deliverable: a running
> **spiking** agent that learns to keep itself alive from a self-generated intrinsic reward.

## Goal

One `SimulationBridge` agent that **navigates to food** to satisfy a self-generated homeostatic drive, learning
the policy in spikes from the **neural drive-reduction reward** (no host distance/goal formula). The strict
brain-based standard: host code is the body + environment only; the drive, reward, and policy are neural.

## The key reuse decision — g9, NOT a from-scratch toy

The CYCLE-128 reward→plasticity toy failed (depressed in all conditions) because it used **default STDP, which
was LTD-dominant for its weights/timing**, and it lacked credit-assignment machinery. The validated **g9 learning
loop** has exactly what the toy lacked, and is the right vehicle:

- **LTP-biased STDP** — `stdp_a_plus=0.012 > stdp_a_minus=0.01` (`g9_runner.py:148-149`): the co-fire potentiates
  (positive eligibility), so a positive reward amplifies it rather than deepening depression.
- **Motor exploration** (Session G) — independent Poisson spikes into each motor pool break the silent-motor
  trap, so every action can acquire eligibility and credit can be assigned.
- The **three-factor path** already wired: `enable_stdp + enable_reward_modulation`, the eligibility trace
  (`tau≈500 ms`), and `current_reward_signal` consumed each step (`g9_runner.py:139-159`).

g9 already learns a navigation policy from `current_reward_signal` (set from a host Manhattan-distance formula,
`g9_runner.py:10-11`). **The integration is one change: replace that reward source with the neural
drive-reduction.**

## The wiring (additive; aim for no `sim/` edit)

1. **Drive region** — add `agrp` (+ optional `pomc`) `BrainRegion`s to the g9 config, driven by an interoceptive
   current proportional to the body's energy deficit (the legitimate body→sensory boundary). Validated CYCLE 127.
2. **Hunger modulator** — a `NeuromodulatorConfig` with a `from_region_firing_signed` production rule over
   `["agrp"]` (`sensitivity≈100, threshold≈0.005`, the CYCLE-127 calibration), so the hunger concentration
   tracks the drive. Read-only target (it does not gate plasticity directly).
3. **Reward swap** — each step, instead of the Manhattan-distance reward, set
   `core_cfg.current_reward_signal = −Δ(hunger concentration)` (the drive reduction; positive when reaching food
   reduces a real deficit). Read from `neuromodulator_manager.get_concentration("hunger")`.
4. **Body/environment (host, legitimate)** — energy `E` depletes each step; reaching the goal cell (food) refills
   `E` (the "eat" event). The grid + the agent's position are the environment + body, exactly as g9 already does.

## The gate (multi-seed ≥ 3, then 6)

**GO** = the spiking agent **learns to reach food** (time-to-food decreases over trials) AND **maintains
homeostasis** (energy stays above the crash floor over a long run), driven only by the intrinsic reward.

## Anti-cheats (load-bearing)

- **Lesion the drive** (zero the AgRP interoceptive current → `r=0`) → no learning, energy crashes. Self-direction
  must collapse.
- **Yoked-random reward** (shuffle the drive-reduction signal, matched marginals) → no learning.
- **Reward provenance** — assert `current_reward_signal` is read from `cp_firing_states`-driven hunger
  concentration, with **no host distance/goal term** anywhere in the reward path (the swap removes the Manhattan
  formula entirely).
- **Remapped action map** — randomize which motor pool is "toward food" per seed, so the agent cannot default to
  the right action; it must learn from the reward (the CYCLE-126 control).

## Honest risk assessment

- **Low-risk:** the drive + neural reward (CYCLE-127 GO) and g9's policy-learning-from-`current_reward_signal`
  (validated project-wide) are both established. The integration composes them.
- **The real risk** is parameter coupling: the drive-reduction reward magnitude (a small concentration delta)
  vs g9's `reward_learning_rate` and the depletion/refill balance must be tuned so the reward is a usable training
  signal and the depletion is survivable-once-learned but fatal-if-not (the CYCLE-129 window logic). A reward gain
  (a constant reward sensitivity) is acceptable (it is a scalar, applied equally across conditions, so it cannot
  manufacture the lesion/yoke contrast).
- **An honest NEGATIVE** (g9's loop fails to learn from the intrinsic reward specifically) would itself pin a
  precise wall — and is a valid deliverable.

## Build steps (for execution)

1. Fork g9's config builder; add the drive region + hunger modulator (CYCLE-127 snippet).
2. In the per-step loop, compute `r = prev_hunger − cur_hunger` and set `current_reward_signal = gain·r`; delete
   the Manhattan-distance reward.
3. Add the energy/eat body dynamics (CYCLE-129 logic) to the environment.
4. Run the 4-check gate + the 4 anti-cheats, ≥3 seeds (then 6).
5. If GO → the first **brain-based, self-directed living agent**: a spiking brain that keeps itself alive from a
   drive it generates. Finding + CLAUDE.md.
