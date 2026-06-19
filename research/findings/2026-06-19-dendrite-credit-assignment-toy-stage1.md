# Dendrite credit-assignment STAGE-1 toy (CPU/numpy) — honest NEGATIVE: the apical-basal dendrite does NOT crack the RL credit-assignment task even in the favorable toy

**Date:** 2026-06-19. **Scope:** STAGE 1 only — the cheapest gate (a CPU/numpy
gridworld actor-critic toy, NO bridge, NO GPU, NO `sim/` edit) before any
on-bridge/GPU work (stage 2) or the months-scale build. **Verdict: NEGATIVE
(valid setup) — the apical-basal dendrite (both the steelman burst-dependent
form AND the project's existing Urbanczik-Senn rule) does not learn the
hidden-goal place→action map where an ideal learner can and the point-neuron
control fails.** This is the TERMINUS for stage 1: it reframes the wall as
DEEPER than apical-basal credit assignment and SAVES the months-scale build.

This result is **consistent with, and extends, the 2026-05-17 precedent**
(`research/findings/2026-05-17-dendritic-credit-assignment-NEGATIVE.md`: the
local Urbanczik-Senn rule did not do hidden credit assignment in a W2-frozen
supervised isolation test at feasible local scale). That test was supervised
and isolation-style; this one is the RL-specific question with the genuine
advantage teaching signal δ = r − V(place) — the exact gap the 2026-05-17 test
lacked — so it was worth running. It lands NEGATIVE too.

---

## What was built

`research/runners/_dendrite_ca_toy_derisk.py` (~330 lines, pure numpy, ASCII,
reuse-by-import of `sim/dendritic_neuron.py` `DendriticLayer` + the BAC
apical-depol burst gate, and `sim/dendritic_plasticity.py`
`urbanczik_senn_update`). A grid-8 gridworld actor-critic with trial resets to
random starts (weights persist across trials), a learned tabular value baseline
V(place) → the advantage δ = r − V, and **six arms** sharing the identical
environment, place code, selection path, and seeds — differing ONLY in the
learning rule that shapes the place→action weights:

| arm | rule | role |
|---|---|---|
| `oracle` | tabular-Q value iteration (max-bootstrap) | DIFFICULTY BOUND — must succeed (else the bias is uninformatively too strong) |
| `point` | global three-factor `Δw ∝ δ · elig(place,taken)` | the point-neuron CONTROL — the fair baseline that must fail |
| `dendrite` | **steelman burst-dependent** `Δw ∝ sign(δ) · burst(apical δ) · elig` (Payeur 2021; burst = `\|B_apical·δ\|` via the BAC gate) | the TEST arm |
| `dendrite_lesion` | burst arm with the apical advantage zeroed (δ→0) | apical-gate LESION — must collapse |
| `dendrite_wrongsign` | burst arm with the advantage sign flipped | WRONG-SIGN control — must fail |
| `dendrite_us` | the project's EXISTING `urbanczik_senn_update` applied verbatim to the RL advantage (the 2026-05-17 rule) | reuse cross-check |

### Mechanism, faithful to the scoping

- **basal** = the **place code** (one-hot per grid cell — held FIXED + perfectly
  SELECTIVE, so the #5 place-selectivity wall is EXCLUDED BY CONSTRUCTION; any
  result is attributable to the credit-assignment RULE, not the place input —
  scoping §4/§5.1).
- **apical** = the **advantage δ = r − V(place)** (a learned tabular critic),
  projected through the FIXED-RANDOM `B_apical` (feedback alignment; NO weight
  transport — reusing `DendriticLayer`).
- **the BAC burst** (`\|B_apical·δ\|`, the Larkum Ca²⁺ plateau magnitude) GATES the
  plasticity; the **sign of the advantage** sets LTP vs LTD (dopamine's signed
  third factor). Lesion (δ→0 into apical) ⇒ zero burst ⇒ zero plasticity.
- **the structural cascade bias** is a directional PRIOR IN THE WEIGHTS toward
  the NW corner (goal-NEUTRAL: NW is neither test goal, so it drives AWAY from
  both). Because it lives in the weights, learning CAN overwrite it — the oracle
  does; the point rule cannot reshape it fast enough (the documented
  credit-assignment failure). Calibrated to `bias_mag = 2.0` so the oracle
  SUCCEEDS but the point control FAILS (the mandatory two-sided anti-cheat).

### Honest mechanism note (why a NEGATIVE was the a-priori-likely outcome here)

For a **single trainable layer** (the actor) there are no hidden units to
assign credit to, so feedback alignment has nothing to align — the apical
compartment's role reduces to a per-action, `\|δ\|`-scaled, non-negative GAIN on
the same update direction the point rule already uses. It does NOT add input/
action specificity beyond the place pre-code + the taken-action eligibility the
point rule already has. The empirical question was whether that burst gate
nonetheless lets the dendrite overcome the structural bias where the global rule
cannot. It does not.

---

## Calibration (the setup is a FAIR test — both sides checked)

- **random-walk floor** (grid-8, 25-step episodes, final Manhattan distance):
  **5.53** (Monte-Carlo, matches the watermaze harness's documented 5.52).
- **oracle tabular-Q** converges to **0.0 greedy-eval at all 6 (goal,seed)** →
  the task is learnable in principle (difficulty bound satisfied,
  `VALID_oracle_succeeds: True`).
- **point three-factor** stays at **3.5–4.8 greedy-eval at all 6** (never ≤2.5)
  → the point-neuron control genuinely fails on the identical setup
  (`VALID_point_control_fails: True`).
- Episode horizon = 25 steps, temp = 0.4, potential-based shaping reward
  (Ng 1999: Δdistance + goal bonus), 500 trials/condition, greedy eval (12 fresh
  trials, deterministic argmax — separating the LEARNED policy from softmax
  exploration noise). Verdict convergence threshold 2.5 + learn-delta 1.0 (the
  watermaze harness's own pre-registered values).

A horizon sweep was required to land this regime: at 80 steps a random walk
reaches the goal ~34% of the time and EVERY arm (including lesion) "passed"
(VOID, too easy); at 12–15 steps NOTHING — not even the oracle — converged (too
hard). 25 steps + weight-init bias + potential shaping is the regime where the
oracle succeeds and the point control fails.

---

## Result (3 seeds × 2 goals; values are greedy-eval mean Manhattan distance)

| arm | seed 42 (1,6 / 6,1) | seed 43 | seed 44 | converged ≤2.5 |
|---|---|---|---|---|
| **oracle** | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | **6/6** ✅ task learnable |
| **point (control)** | 3.5 / 3.5 | 4.2 / 3.8 | 3.9 / 4.0 | **0/6** ✅ fair baseline fails |
| **dendrite (burst, TEST)** | 3.3 / 2.5 | 4.0 / 4.7 | 3.3 / 4.0 | **1/6** ✗ |
| **dendrite_lesion** | 7.0 / 7.0 | 6.5 / 6.3 | 8.2 / 5.6 | 0/6 (collapses) |
| **dendrite_wrongsign** | 9.8 / 9.3 | 8.8 / 8.9 | 9.3 / 7.8 | 0/6 (anti-learns) |
| **dendrite_us** | 9.8 / 5.0 | 5.0 / 4.7 | 3.6 / 5.2 | 0/6 |

**The headline:** the burst-dependent dendrite arm lands at **roughly the
point-neuron level (~3.3–4.7), converging only 1/6** (and that one is exactly at
the 2.5 threshold). It does NOT systematically beat the point control, and never
produces distinct goal-appropriate policies for both goals on any seed
(`distinct_policies_all_seeds: 0/3`).

### The controls validate the NEGATIVE (they are load-bearing, not vacuous)

- **lesion collapses** (apical δ→0 ⇒ eval ~6–8, BELOW the dendrite ~3–5) — the
  apical gate IS doing something (its presence holds the arm at point-level
  rather than collapsing to the floor); it just isn't enough to learn.
- **wrong-sign anti-learns** (flip δ ⇒ eval ~8–10, driving AWAY from the goal) —
  the advantage's SIGN is load-bearing in the rule. This is the exact
  sign-discrimination the 2026-05-17 adversarial review demanded; the toy is NOT
  a non-sign-discriminating pass.
- **the existing U-S rule also fails** (`dendrite_us: 0/6`) — consistent with
  2026-05-17.

### Robustness (the NEGATIVE is not a learning-speed artifact)

Stress sweep of the burst-dependent dendrite across `actor_lr ∈ {0.2, 0.5, 1.0,
2.0}` × `n_trials ∈ {500, 1500}`: converges **at most 1/6**, and MORE training
does NOT help (it plateaus/slightly worsens, never reaching ≤2.5 at both goals).
The dendrite is stuck at the point-neuron wall regardless of learning rate or
budget — not slow, walled.

---

## Honest scientific conclusion (no spin)

- **NOT** "dendritic credit assignment is impossible" — Guerguiev-Lillicrap-
  Richards 2017 / Payeur-Naud-Richards 2021 demonstrate it at larger scale with
  hidden layers + fuller machinery. The mechanism is real in principle.
- **IS:** in this cheap decisive RL slice — on the single trainable actor layer
  the nav task presents, with the place code held perfectly selective and the
  genuine advantage as the apical teaching signal — the apical-basal dendrite
  (steelman burst-dependent AND the project's existing U-S rule) does **not**
  learn the hidden-goal place→action map. It lands at the point-neuron level, on
  a setup where an ideal value-iteration learner converges perfectly and where
  the point control fails. **⇒ the wall is DEEPER than apical-basal credit
  assignment for this nav actor-critic.**
- **Mechanistically:** on a single output (action) layer, feedback alignment has
  no hidden units to align, so the apical burst reduces to a per-action gain on
  the point rule's update — it cannot supply the missing specificity to reshape
  structurally-biased weights faster than the point rule. The credit-assignment
  escape the literature shows requires the HIDDEN-LAYER setting the nav actor
  does not have.

## STOP / decision

- **Stage 1 = NEGATIVE (valid).** Per the pre-registered stop criterion
  (scoping §5.2) and the 2026-05-17 discipline ("an Arch-A NEGATIVE is the
  terminus, NOT a license to escalate to Arch B/C"), this is the answer. **No
  stage-2 on-bridge/GPU build, no months-scale build is warranted on the
  apical-basal-cracks-credit-assignment premise.**
- **What this points to instead** (for the owner's decision, not actioned here):
  the binding constraint is either (a) the place code as the true limiting
  factor — the **#5 place-selectivity wall** (the SEPARATE decorrelation/
  normalization dendrite, partly built as the D2 divisive gain, found
  NOT-load-bearing for the cortex code) — which this de-risk deliberately held
  fixed; or (b) a different algorithm class for the actor (e.g. a model-based /
  successor-representation actor, the off-policy `max`-bootstrap the oracle used
  — which is NOT a local three-factor rule). A single extra compartment with the
  apical advantage gate is not it.
- **Caveat the scope honestly:** this is the rate-level CPU/numpy stage-1 gate,
  not an on-bridge spiking result. But because it is NEGATIVE on the FAVORABLE
  toy (selective place code, ideal-learnable task, genuine advantage), the
  on-bridge spiking version (noisier, harder) is not expected to do better — the
  stage-1 gate is exactly designed to fail cheap before the expensive build, and
  it did.

## Anti-cheat discipline (why this NEGATIVE is trustworthy)

Pre-registered FIXED bars (the watermaze harness's own 2.5 / 1.0 values), never
tuned to a result. Multi-seed (42/43/44). **Two-sided validity gate**: the
oracle MUST succeed (task learnable) AND the point control MUST fail (fair
baseline) — both confirmed before reading the dendrite. The apical-gate LESION
and the WRONG-SIGN control both correctly fail (the lift, if any, would ride the
gate AND the advantage's sign — the exact confounds 2026-05-17 caught). The
NEGATIVE is robust to a learning-rate × trial-budget sweep (not a speed
artifact). NO `sim/` edit; reuse-by-import of the existing dendritic machinery.
The honest NEGATIVE is reported, not forced into a GO.

## Files / evidence

- Toy: `research/runners/_dendrite_ca_toy_derisk.py`.
- Raw result: `research/findings/raw/_dendrite_ca_toy_summary.json` (verdict
  block: `DERISK_VALID: True`, `oracle 6/6`, `point 0/6`, `dendrite 1/6`,
  `lesion 0/6`, `wrongsign 0/6`, `dendrite_us 0/6`, `GO: False`,
  `NEGATIVE: True`).
- Scoping (followed): `research/findings/2026-06-19-dendrite-credit-assignment-
  derisk-scoping.md`.
- Precedent (foregrounded): `research/findings/2026-05-17-dendritic-credit-
  assignment-NEGATIVE.md` + `2026-05-17-dendritic-faithful-instrument-TERMINUS.md`.
- Reused: `sim/dendritic_neuron.py` (`DendriticLayer` apical/basal + BAC gate),
  `sim/dendritic_plasticity.py` (`urbanczik_senn_update`).
- Scientific basis: Larkum 2013 (BAC firing); Urbanczik-Senn 2014; Lillicrap
  2016 (feedback alignment is training-emergent + needs hidden units to align);
  Guerguiev-Lillicrap-Richards 2017; Payeur-Naud-Richards 2021 (burst-dependent
  plasticity); Frémaux-Sprekeler-Gerstner 2013 (spiking actor-critic); Ng 1999
  (potential-based shaping).
