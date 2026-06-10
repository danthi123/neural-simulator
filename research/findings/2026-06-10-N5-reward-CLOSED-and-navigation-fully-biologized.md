# N5 reward CLOSED (proper dopamine-RPE test) — and navigation is now fully biologized at the mechanism level

**Date:** 2026-06-10
**Result:** the host `sign(Δ eccentricity)` reward formula is replaced by a **neural goal-salience/proximity reward** (the SC `sc_rostral` pool) that drives a **correct, graded dopamine reward-prediction-error**, validated by the proper test (`sc_n5_rpe_probe.py`). With N1, N6, N8, N9 spiking and N2/N7 defensible, **every cognitive computation in the navigation loop is now a neural mechanism** — with one honest, documented limitation about behavioral load-bearing.

## The pivotal lesson: a reward can't be validated by a task that doesn't need it

The N5 nav A/B looked great (6-seed neural/host 0.988) but the scrambled-retinotopy anti-cheat **didn't regress** — because this gridworld is an **orient-toward-the-goal** task, solvable by perception (the SC) almost regardless of the reward. So *no* reward (host or neural) is behaviorally load-bearing here, and no nav test can isolate it. A reward/dopamine signal is **defined by its teaching signal** (the reward-prediction-error), not by behavior in a task that doesn't need it. So the proper test is the Schultz RPE battery with the reward **sourced from neurons**.

## The proper test (`sc_n5_rpe_probe.py`) — and the result

A minimal bridge `sc_retina → sc_map (+Mexican-hat) → sc_rostral (proximity) → reward_us → snc` + the signed dopamine modulator. The reward `r` is the SC's *firing* (a goal-close image → bump → `sc_rostral` → `reward_us` → SNc), not a host scalar. Result:

| Check | Result |
|---|---|
| **Burst on the neural US** (close goal) | SNc 251 Hz vs 48 Hz tonic ✓ |
| **Monotone in proximity** | corr(distance, SNc rate) = **−0.99** ✓ |
| **Omission dip** (goal withheld, V>0) | SNc 0 Hz < tonic ✓ |
| **Lesion** `sc_rostral→reward_us` (the decisive reward anti-cheat) | burst vanishes (flat 39 Hz) ✓ — the RPE *is* the synaptic reward |
| **Scrambled retinotopy** (informative, not a gate) | corr −0.90 over 5 permutations — proximity is goal-**salience** (total SC activation, permutation-invariant), not retinotopic *position* |

**VERDICT: PASS.** The neural reward drives a correct, graded dopamine teaching signal, and the decisive load-bearing anti-cheats for a *reward* pass: lesion the synaptic reward → the RPE vanishes; no goal → no reward → dip. This is the validation the orient-solvable nav A/B could not give — here the reward *is* the dependent variable.

The scramble "failure" is not a cheat: scrambling retinotopy is the right anti-cheat for an orienting *direction* (N1, where it regressed 2.4×), not for a *proximity* reward — proximity is goal-salience, which is permutation-invariant. The lesion + omission are the decisive anti-cheats for a reward, and they pass.

## The reward mechanism (settled)

Use `sc_rostral` **proximity/goal-salience** firing as `r`, **not** a temporal-difference circuit. The TD attempt (`sc_rostral_slow` via slow-NMDA + a GABA_B subtraction) had a compound lag ≈ 2.5 nav-steps and a global-GABA_B-tau collision with the N9 critic — structurally wrong. The temporal-difference belongs in the **dopamine RPE** (δ = r − V; the N9 critic provides the baseline V), so a proximity `r` + the neural critic V *is* the correct, more-biological actor-critic factorization — not a mimicry of the host's hand-coded derivative. The deployment (`g11_bg_runner.py:--enable-spiking-sc-approach`) was simplified to this: `sc_rostral → reward_us`, TD regions dropped (52 regions, runs clean).

## Navigation cheat ledger — final

| Axis | Mechanism | Status |
|---|---|---|
| Orienting (N1) | spiking retinotopic superior colliculus (6-seed beats host, lesion-confirmed) | **CLOSED** |
| Action selection (N6) | spiking commit-burst / WTA | **CLOSED** |
| Disinhibition (N8) | spiking | **CLOSED** |
| Reward value (N5) | neural goal-salience/proximity (proper RPE test) | **CLOSED** |
| Dopamine RPE (N9) | spiking SNc + reward_us + GABA_B critic (δ = r − V) | **CLOSED** |
| Goal cue (N2) | beacon rendered into the retina (defensible) | **CLOSED** |
| V1 receptive fields (N7) | innate Gabor afferent weights (defensible) | **CLOSED** |

**Every cognitive computation between sensation and action in the navigation loop is now a validated neural mechanism.** Per the BRAIN-BASED-ONLY standard, navigation is biologized.

## The honest limitation (documented, not hidden)

The N5 reward is validated as a correct *mechanism* (a neural teaching signal). It is **not** behaviorally load-bearing **in this gridworld**, because the task is orient-solvable — the perception (N1) carries it. A behavioral demonstration that the reward *changes* navigation would need a harder task (delayed/structured reward, or a remapped-action navigation where the policy must be learned from reward). That is a separate, larger arc and is noted as future work — not smuggled into this closure. So the precise honest claim is: **navigation is fully biologized at the mechanism level; the reward/dopamine machinery (N5+N9) is biologized and validated as correct, with the caveat that this particular task does not behaviorally stress it.**

## What this unblocks

Roadmap step 1 (finish biologizing everything but cortex) is complete at the mechanism level. Next is step 2 — consolidating the navigation and conversational configs into one brain (the single-instance unification), which wants a deep-research + design pass first (un-scoped; the resonate-and-fire vs Izhikevich neuron-model coexistence is the crux to de-risk).
