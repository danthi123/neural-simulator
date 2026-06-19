# F-S-G spiking actor-critic — TRIAL-STRUCTURED hidden-goal de-risk (2026-06-19)

## The question (STEP 3 — the FINAL point-neuron rigor step)

This is the last point-neuron-feasible attempt before the dendrite (apical-basal credit
assignment) is proposed for the project's deepest learning wall — the actor-critic
credit-assignment wall.

**The de-risk question (the ONLY thing this tests):** does the Fremaux-Sprekeler-Gerstner
(2013) spiking actor-critic form the hidden-goal place->action map on the **point-neuron**
substrate when given its **proper trial-structured protocol** — MANY reset trials at the
SAME hidden goal, learned weights persisting across trials — AND with the structural
cascade **symmetrized** (so the no-reward baseline sits at the random-walk floor, not a
fixed-drift corner)? This is the literature-validated water-maze setting. If the actor
STILL fails here, the point-neuron path is exhausted and the dendrite is the
clearly-proposed obvious unlocker.

## Why this is the necessary follow-up to the Step-2 probe

Step 2 (`2026-06-19-spiking-actor-critic-advantage-routing-derisk.md`) verified by
code-read that the advantage `delta = r - V(place)` is ALREADY routed to the
corticostriatal actor when `enable_neural_critic + spiking_snc + spiking_reward_us` (the
SNc fires `r` minus the striosome_value critic's `V(place)`; the `dopamine`-modulator
deviation is the signed plasticity third factor). But its probe ran **ONE long static
phase** (not a trial-structured protocol) and did **not** confirm cascade symmetrization
(the (6,6) lesion drifting to the NE corner showed a residual structural bias). It was a
**preliminary** NEGATIVE and explicitly flagged these two gaps as the remaining
point-neuron rigor.

## Method (cheapest-first, 1-seed GPU smoke)

- **New runner mechanism (additive, default-OFF, NO `sim/` edit):** `g11_bg_runner.py`
  gains `trial_reset_steps` / `trial_reset_seed`. When `trial_reset_steps > 0`, the agent
  is teleported to a fresh RANDOM start (Manhattan >= 3 from the goal) every K steps while
  the learned weights persist across trials (no plasticity reset) — the F-S-G water-maze
  training. The per-trial final distance is recorded in `results["trial_final_distances"]`
  (the learning curve). `trial_reset_steps == 0` is byte-identical to the legacy path.
- **De-risk runner:** `research/runners/_fsg_watermaze_derisk.py` runs the
  advantage-routed actor-critic (full deployed core, `heuristic_strength=0`,
  `hidden_goal=True`, value warm-up) at >=2 away-from-drift goals, reward-ON vs
  reward-LESIONED, for N reset trials each, and reports the learning curve (early-trial
  mean vs late-trial mean final distance).
- **The random-start protocol is itself the symmetrization:** resetting to a random start
  every trial averages out a fixed starting bias; the reward-OFF (lesion) curve is the
  symmetrization GUARD — it must be goal-INDEPENDENT and near the random floor (~5.52 on
  grid-8). A goal-dependent lesion curve = residual structural drift = confounded.

The load-bearing proof (owner standard `validate_signal_by_its_function`): across >=2
goals (distinct learned policies), reward-ON's learning curve must DECREASE and converge
near the goal, while the lesion stays flat at the floor (goal-independent).

Run:
```
python -X utf8 -m research.runners._fsg_watermaze_derisk \
    --seed 42 --goals "1,6;6,1" --n-trials 40 --steps-per-trial 200 --grid-size 8
```

## Results

(RESULTS_PENDING — populated after the GPU smoke completes.)

## Verdict

(VERDICT_PENDING)

## Artifacts

- Runner mechanism: `g11_bg_runner.py` `trial_reset_steps` / `trial_reset_seed` (additive,
  default-OFF; per-trial final distance in `results["trial_final_distances"]`). NO `sim/`
  edit.
- De-risk runner: `research/runners/_fsg_watermaze_derisk.py`.
- Raw: `research/findings/raw/_fsg_watermaze_summary.json`, `_fsgwm_*_seed42.json`.
