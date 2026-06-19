# Spiking actor-critic ADVANTAGE-routing de-risk on the hidden-goal task (2026-06-19)

## The question (STEP 2 of the limbic-core arc)

Step 1 (`2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md`) found the spiking
reward/value/dopamine limbic core is **NOT behaviorally load-bearing** on a hidden goal: the
reward lesion does not collapse navigation, and the agent drifts to a FIXED corner regardless of
goal location (a structural cascade bias, not reward-driven learning). Diagnosed mechanism: the
place->action map never forms because raw global reward-STDP does not overcome the cascade's
random-init directional bias over a single static goal in a few hundred steps (the 2026-05-05
"global scalar feedback fails at biological scale" family).

**The de-risk question (the ONLY thing this tests):** does the **Fremaux-Sprekeler-Gerstner (2013)
spiking actor-critic recipe** form the place->action map on the point-neuron substrate -- i.e.
does routing the **ADVANTAGE delta = r - V(place)** (the already-deployed spiking-SNc RPE with the
neural value critic ON) as the actor's third factor (instead of the raw global reward that failed
in 2026-05-05) let the actor LEARN a hidden goal's location? Advantage r-V is a far better credit
signal than raw reward (it is ~0 once V predicts r), so this is the canonical fix and the validated
F-S-G water-maze setting -- the point-neuron-feasible attempt BEFORE concluding the dendrite is
needed.

## Code verification: the advantage IS ALREADY routed (no new mechanism needed)

A read of `g11_bg_runner.py` + `sim/bridge.py` establishes that with
`--spiking-snc --enable-neural-critic` the actor's three-factor signal is **already the advantage
delta = r - V(place)**, not raw reward:

1. **The signed third factor in the weight update is the dopamine-modulator deviation.** The
   bridge's reward-modulation block computes
   `effective_signal = da_signal if da_signal is not None else reward_prediction_error`
   (`bridge.py:6904`), where `da_signal = conc("dopamine") - baseline` (`bridge.py:6901`), and the
   weight update is `Delta_w = effective_reward_lr * effective_signal * eligibility`
   (`bridge.py:6952`). So when a `dopamine` modulator is registered, the **signed** plasticity
   direction comes from the DA concentration's deviation from baseline -- NOT from the host scalar
   reward.

2. **The `dopamine` modulator IS the SNc firing.** With `--spiking-snc` the runner registers a
   `dopamine` modulator whose production rule is `from_region_firing_signed` over the `snc` pool
   (`g11_bg_runner.py:4318-4325`), `target_type="plasticity_rate", scope="all"`. So DA concentration
   tracks the SNc's windowed firing relative to its tonic threshold = the reward-prediction error.

3. **The SNc fires r - V(place) at the membrane** when `--enable-neural-critic` (+ `--spiking-reward-us`):
   the reward burst `r` is delivered SYNAPTICALLY by `reward_us -> SNc` (`g11_bg_runner.py:7218-7240`),
   and the **neural value critic** `striosome_value` subtracts `V(place)` at the SNc membrane via its
   GABA_B/GIRK inhibition (`striosome_value -> snc`, `g11_bg_runner.py:7199-7209, 7246-7248`). The
   critic LEARNS V(place) from the same SNc-derived DA signal through a plastic, DA-delta-gated
   place->value afferent.

**=> `effective_signal ~= delta = r - V(place)` = the advantage, and it gates the corticostriatal
actor's eligibility.** The actor is advantage-gated, not raw-reward-gated, whenever the neural critic
is on. (When `--spiking-snc` is OFF, `effective_signal = reward_prediction_error = r - reward_baseline`
= raw reward -- the path that failed Step 1 / 2026-05-05.) This probe therefore USES the existing
advantage path; it is a **configuration de-risk, not a code change**. The three Step-1 confounds:
  - **(a) sparse selective place code** -- the actor's `sensor_place_readout` uses `sigma=0.5`
    (`g11_bg_runner.py:3222`) => 1-3 cells/position with a per-position preferred grid
    (`:4046`). Already selective. [present]
  - **(b) a SINGLE long goal-stable phase** -- `goal_schedule=None => [(0, goal)]` = 1 phase
    (`:3866`); the probe runs one static goal over a long `n_steps`.
  - **(c) the structural-bias confound** -- surfaced by the goal-location anti-cheat (the agent must
    TRACK the goal across >=3 locations; a fixed-corner drift that equals one goal is NOT tracking).

## Method (cheapest-first, 1-seed GPU smoke)

Probe `research/runners/_advantage_actor_critic_probe.py` (this commit). For each of >=3 hidden-goal
locations it runs the **advantage-routed actor-critic** (full deployed core:
`enable_neural_critic + spiking_snc + spiking_reward_us`, `heuristic_strength=0`, `hidden_goal=True`,
single static goal, long `n_steps`) and reports `sum_finalQ` (per-phase final-quarter Manhattan,
LOWER=better), the END POSITION, and `end_dist_from_goal`. `--also-lesion` adds the reward-LESIONED
condition (`lesion_reward=True`) at each goal = the load-bearing contrast. GPU/cupy only (the
moving-goal runner imports cupy directly). Reference random-walk floor (grid-8, start (1,1)): ~5.52.

Verdict logic: **tracking** = reward-ON end-pos NEAR the goal (<=2 Manhattan) AND `sum_finalQ` below
the floor, at >=2/3 locations, with DISTINCT end positions across goals (the anti-cheat); the lesion
(if run) at/above floor and goal-independent.

## Results (controller-owned full multi-goal run; the subagent overran on a partial single-goal smoke)

Full 3-goal de-risk (`--goals "6,6;1,6;6,1" --also-lesion`, seed 42, 1500 steps, grid-8; random-walk
floor ~5.52; LOWER `sum_finalQ` = better; `end_dist` = Manhattan from the goal):

| goal | reward_ON `sum_finalQ` / end_dist | LESION `sum_finalQ` / end_dist |
|---|---|---|
| (6,6) | 4.40 / 6 | 1.60 / 1  — the NE fixed-drift corner (CONFOUNDED: the structural bias drifts here) |
| (1,6) | 5.36 / 5 | 5.36 / 5  — IDENTICAL (reward does nothing) |
| (6,1) | 7.85 / 10 | 6.59 / 6  — reward WORSE than the lesion |

**The advantage-routed actor does NOT track any goal** (reward_ON end-dist 5-10; it never reaches near
a goal it must LEARN), and **reward_ON <= LESION at every goal**. The away-from-drift goals (1,6)/(6,1)
are the clean signal (the (6,6) drift coincidence is confounded). Even the canonical fix — the
already-routed advantage delta = r - V (a far better credit signal than the raw reward that failed
2026-05-05) — does not form the place->action map on point neurons in this config.

## Verdict — PRELIMINARY NEGATIVE (point-neuron path NOT yet exhaustively ruled out) -> the DENDRITE-decision juncture

**Preliminary NEGATIVE:** the advantage-routed point-neuron actor-critic did not solve hidden-goal
place->action learning here. This is the **3rd independent hit on the actor-critic credit-assignment
wall** (2026-05-05 global-scalar W->A verdict + Step 1 limbic-not-load-bearing + this), the project's
deepest learning wall, where the dendrite (apical-basal credit assignment) is the named unlocker
(`feedback_dendritic_substrate_fair_game`, `docs/plans/2026-05-05-dendritic-learning-design.md`).

**HONEST CAVEAT — the point-neuron path is NOT exhaustively ruled out:** this probe ran ONE long
static-goal phase, NOT a **trial-structured protocol** (the F-S-G water-maze learns the place->action
map over MANY reset trials), and it did not confirm **cascade SYMMETRIZATION** (the (6,6) lesion
drifting to (7,6) ~ goal shows the structural NE-bias persists, masking any weak learning signal). A
CLEAN point-neuron verdict therefore requires the proper trial-structured actor-critic training + a
symmetrized cascade — which is itself the deep "make the limbic core load-bearing" build (the remaining
must-try-first point-neuron step per the dendrite memory).

**=> Genuine OWNER-SCOPED FORK (surfaced):** (A) the DENDRITE arc (months-scale; D2 Phase 0-2 infra
exists, Phase 3 pending); (B) the remaining POINT-NEURON attempt (trial-structured actor-critic
training + symmetrized cascade — the cheaper rigor step); (C) PIVOT to the conversational-scaling
primary (task #55 — UNBLOCKED, substrate exercised, not gated by this wall). Controller-default absent
redirect (per never-idle): (C), deep-research-first; (A) is proposed, not unilaterally started.

## Artifacts

- Probe: `research/runners/_advantage_actor_critic_probe.py` (this commit).
- Runner flags reused (from Step 1, additive default-OFF): `g11_bg_runner.py` `hidden_goal`,
  `lesion_reward`; the deployed limbic core `enable_neural_critic`, `spiking_snc`, `spiking_reward_us`.
  NO `sim/` edit.
- Raw: `research/findings/raw/_advantage_actor_critic_summary.json`, `_advac_*_seed42.json`.
