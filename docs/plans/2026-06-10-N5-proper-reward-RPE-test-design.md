---
type: plan
status: live
date: 2026-06-10
---

# N5 — the proper test: a dopamine reward-prediction-error battery with the reward sourced from neurons

**Date:** 2026-06-10
**Why this doc exists:** the N5 nav A/B was confounded — this gridworld is *orient-solvable* (the SC perception solves it), so no reward (host or neural) is behaviorally load-bearing, and the scrambled-retinotopy control didn't regress. A reward/dopamine signal cannot be validated by a task that doesn't need it. A reward system is *defined* by its teaching signal (the RPE), so the proper test is the Schultz reward-prediction-error battery — with the reward `r` **sourced from the SC neurons**, not a host scalar.

## What's missing in the existing harness

`research/runners/snc_pavlovian_probe.py` validates the SNc RPE behavior (omission dip, monotone-in-δ) but drives the SNc with a **host** reward: `I_snc = tonic + k_r·max(0,r) − k_v·max(0,V)`, where `r ∈ {+1,0,−1}` is a number. The reward content is never neural. So it proves the *dopamine cell* is correct given a reward — not that a *neural* reward produces the right teaching signal. That's exactly the N5 question.

## The proper N5 test (reward sourced from the SC)

Build a minimal bridge: `sc_retina → sc_map (+Mexican-hat) → sc_rostral (proximity)` → `reward_us` → `snc` + the signed `dopamine` modulator (the Pavlovian harness's wiring) + (for the cue-shift) the N9 neural striosome critic `V`. The **US/reward is delivered by driving the SC with a goal-present-and-close image** — the SC bump → `sc_rostral` proximity firing → `reward_us` → SNc burst. The reward `r` is now the SC's *firing*, not a host number. The "omission" is a goal-absent/far image (no proximity → no reward).

Run the Schultz battery and require:

1. **Burst on the neural US.** A goal-close image → `sc_rostral` fires → `reward_us` → SNc rate bursts above tonic. (The neural reward drives the dopamine cell.)
2. **Monotone in proximity.** Closer image → larger SNc burst (the de-risk already showed `corr(distance, sc_rostral) = −0.844`); confirm it propagates to the SNc rate.
3. **Omission dip.** After acquisition (V learned that the cue predicts the close-goal reward), present the cue then a goal-*absent* image (US withheld) → SNc dips below tonic. Needs the signed DA rule + a non-zero learned V.
4. **Cue-shift (with the neural critic).** Over acquisition trials, the SNc burst migrates from the US (goal-close image) to the predictive CS, as the neural striosome critic `V` acquires the cue's value. The canonical Schultz signature; requires N9's state-dependent neural V (host global-EMA can't produce it).
5. **Faithful to the host reward.** Run the identical schedule driving `reward_us` from the host `sign(Δecc)` scalar instead of the SC; the RPE signatures (burst/dip/shift magnitudes) must match within tolerance → the neural reward is a faithful `r`.
6. **Load-bearing anti-cheats (the ones the nav couldn't give):**
   - **Reward-relay lesion:** zero `sc_rostral → reward_us` (or `reward_us → snc`) → the burst/dip/shift vanish (the RPE is carried by the synaptic reward, not anything else).
   - **Scrambled retinotopy:** permute `sc_retina → sc_map` → the SC bump no longer tracks the goal → `sc_rostral` is noise → no proximity-graded burst, no cue-shift. **This regresses here** (unlike the nav, where perception carried the task) because the reward *is* the dependent variable.

PASS = burst + monotone + omission-dip + cue-shift + host-faithful + both anti-cheats break it. That is a complete, load-bearing, non-confounded validation that the **neural reward produces a correct dopamine teaching signal** — the gold standard for the whole N5+N9 reward+dopamine axis, independent of whether any particular task behaviorally needs the reward.

## Reward mechanism (settled by this design)

Use `sc_rostral` *proximity* firing as `r` (corr −0.844, robust), **not** the fragile temporal-difference circuit (compound slow-NMDA+GABA_B lag ≈ 2.5 steps, and a global-GABA_B-tau collision with the N9 critic). The temporal-difference belongs in the dopamine RPE (δ = r − V), where the critic V provides the baseline — so a proximity `r` + the neural critic V *is* the correct actor-critic factorization, and more biologically faithful than mimicking the host's hand-coded derivative. The TD regions (`sc_rostral_slow`, `approach_slow_inh`, `approach_n5`) are dropped; `sc_rostral → reward_us` is the reward.

## Honest scope note (kept, not hidden)

This validates the reward *mechanism* (neural `r` → correct RPE). It does **not** claim the reward is behaviorally load-bearing in *this* gridworld — it isn't, because the task is orient-solvable. A behavioral load-bearing demonstration needs a harder task (delayed/structured reward, or a remapped-action navigation where the policy must be learned from reward); that is a separate, larger arc and is noted as future work, not smuggled into the N5 closure.

## Build

New runner `research/runners/sc_n5_rpe_probe.py` reusing `_build_snc_bridge`/`_drive_snc_and_count` patterns from `snc_pavlovian_probe.py` + the SC build helpers (`install_spiking_sc_wiring`, `render_egocentric_goal`). CPU (`SIM_BACKEND=numpy`) for the battery; the reward-source is the only thing varied vs the existing harness.
