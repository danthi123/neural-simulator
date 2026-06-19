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

**Process note:** the first delegated runs were misread as "crashed" by a hand-rolled `tasklist`
process-waiter (a false signal on Windows) → duplicate concurrent runs contended the GPU. Corrected by
running ONE clean tracked job. The runner itself is fine (the tiny smoke ran clean). LESSON: these long
moving-goal GPU de-risks (~0.3 s/step) are controller-owned via the tracked background mechanism, not a
hand-rolled waiter.

**Clean tracked run (`_fsg_2goal_clean.json`, seed 42, 12 trials × 2 goals × {reward ON, lesion}, grid-8;
random-walk floor 5.52; lower = closer to goal):**

| condition | early-trial dist | late-trial dist | learn Δ | end_pos |
|---|---|---|---|---|
| goal (1,6) reward ON | 6.0 | **6.67** | −0.67 (worse) | [7,6] (dist 6) |
| goal (1,6) reward OFF (lesion) | 7.0 | 6.67 | — | [7,7] |
| goal (6,1) reward ON | 6.67 | **6.67** | 0.0 (flat) | [5,6] (dist 6) |
| goal (6,1) reward OFF (lesion) | 6.0 | 6.33 | — | [6,7] |

**No learning at either goal:** reward-ON's late distance stays ~6.7 (near the floor), does NOT decrease,
and is **no better than the reward-OFF lesion**; both lesion curves are flat, goal-independent, at the
random floor. `n_goals_learned_and_converged = 0/2`, `n_goals_on_beats_lesion = 0/2`.

**The CYCLE-241 4-trial smoke "early 7 → late 2 learning" was a small-sample artifact** — the 4-trial
"late" averaged a SINGLE trial that happened to land near the goal. The 12-trial run's 3-trial "late"
average shows the learning was not real. The rigor (more trials + the ≥2-goal distinct-policy
requirement) caught the false positive. Smoke positive RETRACTED.

## Verdict — NEGATIVE → the dendrite is the clearly-proposed obvious unlocker

The point-neuron F-S-G spiking actor-critic does NOT form the hidden-goal place→action map even with the
proper trial-structured protocol. This is the **3rd rigorous hit** on the actor-critic credit-assignment
wall (2026-05-05 global-scalar W→A + the single-phase advantage de-risk + this trial-structured run).

**HONEST CAVEAT:** 1 seed, 12 trials, and the failure is INTERTWINED with the **#5 place-selectivity
boundary** (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`: the place code is not location-selective
enough at nav scale — so the actor may fail because it cannot tell places apart, i.e. the place INPUT, not
only the credit-assignment RULE). But BOTH #5 (graded, selective place fields) and the credit-assignment
rule are the SAME point-neuron analog-computation limit, and BOTH point at the dendrite (apical-basal /
graded dendritic computation). A single dendritic-substrate build would address both.

**⇒ The fork resolves toward the DENDRITE** (`feedback_dendritic_substrate_fair_game` — "propose it
clearly the moment it's the obvious unlocker"). This is a months-scale, OWNER-SCOPED arc (the D2 Phase 0-2
two-compartment-neuron + learned-graded-cortex infrastructure already exists; Phase 3 is pending,
`docs/plans/2026-05-05-dendritic-learning-design.md`). Proposed to the owner, NOT unilaterally started; the
unblocked high-leverage alternative is the conversational-scaling primary (task #55).

## Artifacts

- Runner mechanism: `g11_bg_runner.py` `trial_reset_steps` / `trial_reset_seed` (additive,
  default-OFF; per-trial final distance in `results["trial_final_distances"]`). NO `sim/`
  edit.
- De-risk runner: `research/runners/_fsg_watermaze_derisk.py`.
- Raw: `research/findings/raw/_fsg_watermaze_summary.json`, `_fsgwm_*_seed42.json`.
