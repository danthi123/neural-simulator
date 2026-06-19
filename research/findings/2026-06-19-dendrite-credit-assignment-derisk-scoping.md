# Dendrite credit-assignment de-risk — scoping (the cheap-first option-1 the owner approved 2026-06-19)

**Status:** READ-ONLY deep-research + design-first scoping (the standing "deep research + catalog review FIRST at a
multiply-confirmed roadblock" move, CLAUDE.md). NO `sim/` edits, NO build, NO GPU run in this pass. Single deliverable =
this doc. **Date:** 2026-06-19. **Author role:** read-only research subagent. Every load-bearing project fact below was
re-verified against the repo (file/finding/line cited); the surprising / decision-flipping ones (the 2026-05-17
credit-assignment NEGATIVE; the two-dendrite-stories framing; the existing `DendriticLayer` internals; the actor's
plasticity wiring) were read in full, not trusted from a summary. **This is a scoping/decision doc, NOT a brain-based
result and NOT a commitment to the months-scale build.**

---

## 0. The one-paragraph answer

The point-neuron spiking actor-critic does NOT learn the hidden-goal place→action map (3 rigorous NEGATIVEs this arc:
the 2026-05-05 global-scalar W→A verdict, the single-phase advantage-routing de-risk, the 12-trial trial-structured
water-maze run). The owner approved testing whether the project's **existing two-compartment "dendrite" neuron**
(`sim/dendritic_neuron.py` = Larkum BAC + Guerguiev-Lillicrap-Richards 2017 segregated apical/basal; `sim/dendritic_plasticity.py`
= the local Urbanczik-Senn apical-gated mismatch rule) cracks this credit-assignment problem before any months-scale
build. **The mechanism map is sound and the named biology is exact** (apical = the value/RPE teaching signal, basal =
the place/state code, the apical-gated burst lowers the somatic threshold AND gates the corticostriatal plasticity so
the actor learns "in THIS place → THAT action"; Payeur-Naud-Richards 2021 burst-dependent plasticity is the canonical
form). **BUT there is a single load-bearing internal precedent that must dominate the framing: the project ALREADY ran
a dendritic-credit-assignment de-risk (2026-05-17) and it was an honest NEGATIVE** — the local Urbanczik-Senn rule with
fixed-random feedback did NOT do hidden credit assignment in the discriminating W2-frozen isolation test at feasible
local scale (loss-ratio 1.095, feedback-alignment 0.012), and the both-layers regime was non-sign-discriminating. So
the *off-bridge rate-level* version of "can the existing dendrite do credit assignment" is **already answered NO at
feasible local scale**, and a naive numpy re-run would re-derive that NEGATIVE. **⇒ The genuinely-new, decision-relevant
de-risk is NOT another off-bridge XOR/MLP toy — it is the on-bridge actor-critic question: does an apical-gated
plasticity rule on the EXISTING corticostriatal pathway (driven by the already-deployed neural-critic advantage as the
apical/burst teaching signal) crack the hidden-goal task in the EXACT `_fsg_watermaze_derisk.py` harness, where the
point-neuron control failed?** The cheapest-first version is a **rate-level / numpy off-bridge gridworld actor-critic
toy that faithfully reproduces the point-neuron failure (cascade structural-bias + sparse place code) and adds ONE
apical-gating capability** — this is afternoon-scale, decides the go/no-go for the on-bridge build, and (critically)
fixes the 2026-05-17 NEGATIVE's flaw (it was a supervised-MLP isolation test, NOT the RL credit-assignment task the nav
actor needs, and the *teaching signal it lacked* is exactly the advantage the nav critic now supplies). The honest risk:
the failure is **intertwined with #5 (the place-code-selectivity wall)** — the dendrite must be shown to crack
credit-assignment *with the place code held fixed / known-selective*, or a GO is confounded by the place input rather
than the credit-assignment rule.

---

## 1. THE MECHANISM MAP (concrete apical/basal signal assignment + the plasticity gate)

**The crisp claim (1-2 sentences):** In a two-compartment actor neuron, the **basal** compartment integrates the
bottom-up **place/state code** (`sensor_place_readout` firing — "where the agent is") and the **apical** compartment
integrates the top-down **value/RPE/dopamine teaching signal** (the deployed neural-critic advantage `δ = r − V(place)`
carried by the SNc); the **apical-driven dendritic event** (a Larkum BAC Ca²⁺ plateau / burst, triggered when basal
place-drive AND apical advantage coincide) both lowers the somatic threshold (so the *right place×action* fires more)
AND **gates the corticostriatal plasticity** so a weight on the `sensor_place_readout → cortex_action` (and
`cortex → str_D1`) synapse changes ONLY when "this place was active AND the advantage said this was good" — i.e. the
plasticity is now **place-specific AND advantage-specific**, the exact `Δw ∝ pre × post × burst(place,δ)` three-factor
form a point neuron's single-soma `Δw ∝ DA × eligibility` (global scalar) provably cannot localize (the 2026-05-05
verdict's root cause).

### 1.1 Why this is the credit-assignment fix (grounded in the project's own diagnosis)

The 2026-05-05 W→A verdict (`research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`) and the
2026-06-19 hidden-goal diagnostic both pin the *identical* root cause: **global scalar reward broadcasts the SAME
`DA × eligibility` to every corticostriatal synapse regardless of which place/action was responsible**, so at biological
scale the noise floor of "which neurons coincided by chance" swamps the credit signal — the textbook credit-assignment
problem with global feedback (Frémaux & Gerstner 2016). A supervised GRADIENT solved the same architecture 3/3 PERFECT
because gradient gives each weight a *specific* per-region error. The dendrite is the biologically-local way to recover
that specificity WITHOUT weight transport:

- **Basal (bottom-up):** the **place code** — `sensor_place_readout` firing, the egocentric `(x,y)` self-position drive
  (legitimate under BRAIN-BASED-ONLY; it is the agent's own position, not the hidden goal). In the existing
  `DendriticLayer` this is `W_basal` (`sim/dendritic_neuron.py:24`, `v_basal = leak·v_basal + x@W_basal`).
- **Apical (top-down teaching):** the **advantage** `δ = r − V(place)` — already computed on the substrate by the
  deployed neural critic (the SNc fires `r` minus the `striosome_value` critic's `V(place)` via GABA_B/GIRK; the
  `dopamine` modulator deviation is the signed third factor — verified by code-read in
  `2026-06-19-spiking-actor-critic-advantage-routing-derisk.md` §"Code verification"). In the existing `DendriticLayer`
  this routes through `B_apical` — a **FIXED RANDOM** feedback projection (`sim/dendritic_neuron.py:27`, feedback
  alignment, never learned, NO weight transport).
- **The dendritic event (BAC firing) = the gate:** apical depolarization LOWERS the somatic threshold
  (`effective_threshold = theta_high − apical_gain·|apical_depol|`, `dendritic_neuron.py:44-47`) so basal+apical
  coincidence drives a burst the soma alone would not. **Burst-dependent plasticity** (Payeur-Naud-Richards 2021) then
  makes the basal synapse's weight change proportional to that burst — `Δw ∝ pre · burst` — so a corticostriatal weight
  is potentiated ONLY for the place that was active when the advantage was positive. This is the apical-gated local rule
  `urbanczik_senn_update` already implements (`sim/dendritic_plasticity.py:17-41`: `Δw ∝ apical_gate · mismatch · pre`,
  where `apical_gate` is the post-synaptic apical activity).

### 1.2 The 2-3 most-relevant papers + catalog entries (verified)

- **Payeur, Guerguiev, Zenke, Richards, Naud (2021), *Burst-dependent synaptic plasticity can coordinate learning in
  hierarchical circuits*, Nature Neuroscience** — THE most relevant: the apical dendrite controls bursting + short-term
  plasticity to multiplex feedforward/feedback so a top-down (here: advantage) signal steers local plasticity *without
  disrupting bottom-up signaling*. This is the mechanism class the web-lit pass confirmed: "burst-dependent synaptic
  plasticity, integrated with apical dendritic activity and feedback pathways, enables effective credit assignment …
  the apical dendrite receives the plasticity signal, enabling local modification of neuron weights."
- **Guerguiev, Lillicrap, Richards (2017), *Towards deep learning with segregated dendrites*, eLife** — the segregated
  apical/basal + fixed-random feedback (feedback alignment, no weight transport) that `sim/dendritic_neuron.py` IS. This
  is the SPATIAL credit-assignment dendrite (the right family for "which synapse gets credit"), distinct from the
  temporal BTSP dendrite (Bittner-Magee 2017) the project showed point neurons handle via conductance EMAs for the TD
  cue-shift (`2026-06-18-TD-cueshift-dendrite-decision-scoping.md`).
- **Frémaux, Sprekeler, Gerstner (2013), *Reinforcement Learning … with Spiking Neurons*, PLoS Comput. Biol.** — the
  canonical spiking actor-critic the project's circuit follows (a spiking critic estimates V; the TD/advantage error
  modulates reward-STDP). Point-neuron; the de-risk asks whether a dendritic gate on the actor's plasticity lifts it
  past the credit-assignment wall the point-neuron form hits.
- **Catalog `G.02 Active dendrites` (MISSING, ~10× compute/neuron; behavioral validation = the Larkum BAC firing
  experiment, basal+apical coincidence → bursts, `:2644-2652`)** + **`C.30 Actor-critic`** (actor implemented, critic
  the named requirement — now built — NOT a dendrite per se; the dendrite is the credit-routing escape when the
  point-neuron actor-critic under-learns) + **`B.17 dendritic linearization`** (needs a multi-compartment MSN, the
  striatal side of the same gap). The catalog confirms multi-compartment + BAC firing is genuinely absent on the bridge.

### 1.3 The honest mechanism caveat (do not hand-wave)

The 2026-05-17 NEGATIVE proved that *merely having* the apical-gated Urbanczik-Senn rule + fixed-random feedback does
NOT guarantee credit assignment — at feasible local scale, in the discriminating W2-frozen isolation test, the local
rule alone did nothing (loss-ratio 1.095; feedback-alignment 0.012 ≈ zero). The mechanism is correct *in principle*
(GLR-2017/Sacramento-2018 demonstrate it at larger scale + many steps), but the project's own cheap slice found it does
not engage at feasible local scale. **So the de-risk cannot assume "dendrite ⇒ credit assignment"; it must MEASURE
whether the apical gate engages on THIS task.** Two structural differences from the 2026-05-17 test make a fresh,
RL-specific test worthwhile rather than a foregone re-NEGATIVE: (i) 2026-05-17 was a *supervised* MLP with a *constructed
label* as the teaching signal in a W2-frozen isolation — the nav actor-critic instead has a *genuine advantage teaching
signal* (`r − V`) the substrate already computes, and (ii) the nav task is a *single trainable layer* (the corticostriatal
actor) with the rest of the cascade fixed — there is no "W2 co-adaptation confound" to isolate against, because there is
no second trainable layer; the actor IS the W2-frozen-equivalent slice by construction.

---

## 2. WHAT EXISTS vs WHAT IS NEW (reuse-vs-new, file-cited)

### 2.1 Reusable (the whole point — do NOT rebuild)

| Asset | File / location | What it gives the de-risk |
|---|---|---|
| Two-compartment neuron (Larkum BAC + GLR-2017 segregated dendrites) | `sim/dendritic_neuron.py` (58 lines): `W_basal`, `B_apical` (fixed-random, no weight transport), `effective_threshold = theta_high − apical_gain·|apical_depol|` | the apical/basal split + the BAC threshold-lowering gate, off-bridge numpy, ready to drive an actor toy |
| Local apical-gated plasticity rule | `sim/dendritic_plasticity.py` (`urbanczik_senn_update`: `Δw ∝ apical_gate · mismatch · pre`, sign-corrected to descent, verified `+1.0`) | the local three-factor rule where the apical gate = post-synaptic apical activity; swap its supervised mismatch for the advantage gate |
| Feedback-alignment MLP harness (GPU-capable) | `sim/dendritic_mlp.py` (195 lines, routes through `sim.backend`) | the multi-layer scaffold + the discriminating isolation/sign/permuted-label test idioms (the 2026-05-17 instrument is preserved + re-runnable) |
| The EXACT actor-critic de-risk harness | `research/runners/_fsg_watermaze_derisk.py` + `g11_bg_runner.py` `run_moving_goal_episode` flags `--hidden-goal`, `--lesion-reward`, `trial_reset_steps`/`trial_reset_seed`, `critic_warmup_trials` | the hidden-goal, trial-structured, ≥2-goal, reward-ON-vs-lesion, goal-location-anti-cheat harness where the point neuron FAILED — the on-bridge GO target swaps in the dendritic actor |
| The deployed advantage limbic core (the apical teaching signal source) | `--enable-neural-critic` (`striosome_value` V(place); GABA_B subtracts V at the SNc), `--spiking-snc` (`dopamine` modulator = SNc firing = RPE), `--spiking-reward-us` (r delivered synaptically); `sim/td_value_critic.py` | `δ = r − V(place)` already computed ON the substrate — this IS the apical drive; NO new critic needed |
| The actor's plastic pathways (where the apical gate attaches) | `g11_bg_runner.py:1450` `sensor_place_readout → cortex_{action}` (plastic); `:1749` `cortex → str_D1` (plastic, `plasticity_gate="corticostriatal"`); reward-modulated STDP gated by `cp_eligibility_trace` | the exact synapses an apical gate must multiply; `cp_plasticity_rate_gain` / the gate machinery is the per-synapse hook |
| The step-loop protected-edit template (if/when on-bridge) | `sim/bridge.py:5805-5849` + `sim/kernels.py:252` (`fused_coincidence_plateau`) — a guarded, per-neuron, restricted-matvec sub-threshold term computed BEFORE the soma integrates, byte-identical when off | the byte-identity-when-off pattern a second `v_apical`/`v_dend` state + apical-gate term would mirror |
| Guarded dendritic flags already on the bridge | `sim/config.py:173` `enable_coincidence_detection`, `:233` `enable_dendritic_divisive_gain`, `:386` `enable_graded_lateral` (all default-False) | the precedent for additive default-OFF dendritic capability; NONE of these is compartment-routing for credit assignment, so the actor's apical gate is genuinely new |

### 2.2 New (the minimal addition)

**There is NO compartment-routing / two-compartment actor on the bridge today** (the `DendriticLayer`/U-S stack is
off-bridge numpy; the bridge dendritic flags are diagonal divisive gain + lateral decorrelation, NOT apical-gated
plasticity routing; verified `grep` of `sim/config.py` + the deep-research §(d)). So the NEW pieces, in cheapest-first
order:

1. **De-risk stage 1 (CPU/numpy, NO `sim/` edit):** a small off-bridge gridworld actor-critic toy that (a) reproduces
   the point-neuron failure (a structural-bias cascade + a sparse place code + global-scalar `DA × eligibility` ⇒ no
   place→action learning, mirroring the diagnostic), then (b) adds ONE apical gate: the actor's place→action weight
   update is gated by the burst `b(place, δ) = basal_place_drive × apical(δ)` via the existing
   `urbanczik_senn_update`/`DendriticLayer` (the advantage `δ` as the apical signal). **The point-neuron control = the
   SAME toy with the apical gate collapsed (global scalar only).** ~80-150 lines, reuse-heavy. **This is the gate for
   the on-bridge build.**
2. **De-risk stage 2 (on-bridge, ONLY if stage 1 GO — a SMALL protected `sim/` edit, flag for byte-review):** route the
   advantage to a per-neuron `v_apical` on the actor's `cortex_{action}` (and/or `str_D1`) slice and gate the
   corticostriatal STDP by the apical-burst — additive, default-OFF (`enable_apical_gated_corticostriatal` or similar),
   byte-identical when off (mirror the `fused_coincidence_plateau` guard). Then re-run `_fsg_watermaze_derisk.py`
   *verbatim* with the dendritic actor. **This is NOT the full months-scale N-compartment `NeuronModel`** — it is the
   minimal apical-gate-on-the-actor-pathway, the reduced form the 2026-05-05 design doc scoped (`target_compartment`
   routing on `RegionPathway`), the smallest thing that answers the on-bridge question.

**Is the existing plasticity rule already the right rule, or does it need the burst variant?** The existing
`urbanczik_senn_update` is `Δw ∝ apical_gate · (soma − φ(v_basal)) · pre` — an apical-*gated* mismatch rule. For
actor-critic the cleaner target is the **burst-dependent** form `Δw ∝ pre · burst(δ)` where the apical advantage `δ`
sets the burst probability (Payeur 2021) — i.e. the apical signal should *gate*, with the advantage's SIGN setting LTP
vs LTD (positive advantage → potentiate the taken action's place→action weight; negative → depress). The existing rule
is close (the apical-gate term is there) but its mismatch is a self-prediction error, not a reward advantage; the
stage-1 toy should test the burst-gated-by-advantage form explicitly (a small change to how the apical signal enters,
NOT a new module). This is a **mechanism choice to settle in stage 1**, cheaply, before any `sim/` edit.

---

## 3. THE CHEAPEST-FIRST DE-RISK (config + GO bar + controls)

**Decision logic (why two stages):** the *on-bridge* `_fsg_watermaze_derisk.py` run is the load-bearing test (it is the
exact harness the point neuron failed), but it needs a protected `sim/` edit + GPU. The *cheapest* thing that decides
whether that edit is worth making is a **stage-1 numpy gridworld actor-critic toy** that reproduces the point-neuron
failure and adds the apical gate — afternoon-scale, NO `sim/` edit, NO GPU. **Stage 1 is the gate for stage 2.** This
mirrors the project's own D1→D2 ladder (the dendritic cortex was gated by the D1 numpy toy before the protected edit).

### 3.1 Stage 1 — the cheapest gate (CPU/numpy, NO `sim/` edit, afternoon)

**The toy (faithful to the actual failure):** a tabular/sparse-code gridworld actor-critic where the point-neuron form
provably fails the same way the bridge does —
- **Environment (host, legitimate):** grid-8, hidden goal at ≥2 away-from-drift locations (1,6)/(6,1), random start each
  trial (Manhattan ≥3 from goal), trial-structured (weights persist across trials), scalar distance reward only.
- **Place code (the #5 confound — held FIXED + selective):** a sparse, *known-selective* place code (1-3 cells/position,
  per-position preferred grid — the `sensor_place_readout` σ=0.5 design). **Crucially: held FIXED and selective so the
  de-risk isolates the credit-assignment RULE, not the place input** (see §5).
- **Structural-bias cascade:** initialize the actor's place→action weights with a random directional bias that, under
  global-scalar reward-STDP, produces the fixed-corner drift (reproduce the diagnostic's NE-drift) — so the point-neuron
  control FAILS on the identical toy (mandatory anti-cheat, §3.3).
- **Critic:** a learned `V(place)` (the toy's striosome-value analogue) → the advantage `δ = r − V(place)`.
- **The dendritic actor (the test arm):** the place→action weight update is gated by the apical burst
  `b(place, δ) = basal(place_drive) × apical(δ)` via the existing `DendriticLayer`/`urbanczik_senn_update` (advantage as
  the apical signal, sign → LTP/LTD). **The point-neuron control = the SAME toy, apical gate collapsed (global
  `DA × eligibility` only).**

**Stage-1 GO bar (the contrast IS the result, multi-seed 42/43/44):**
- **STRUCTURE:** the dendritic actor's per-trial-final-distance LEARNING CURVE DECREASES and converges near the goal
  (late-trial mean < ~2.5 on grid-8, the harness's `converge_thresh`) at BOTH goals, WHILE the point-neuron control
  stays at the random floor (~5.52) — the point-neuron control MUST fail on the identical toy.
- **DISTINCT POLICIES (the anti-cheat):** the dendritic actor ends at DISTINCT, goal-appropriate locations across the ≥2
  goals (tracks the goal — not a fixed corner drift).
- **LESION:** zero the apical gate (advantage → 0 into the apical compartment) ⇒ the dendritic actor collapses to the
  point-neuron floor (proves the effect RIDES the apical gate, not a leftover toy property).

### 3.2 Stage 2 — the load-bearing on-bridge test (ONLY if stage 1 GO; small protected `sim/` edit + GPU)

Run `research/runners/_fsg_watermaze_derisk.py` **verbatim** with the dendritic actor enabled (the new default-OFF
apical-gated corticostriatal flag ON):

```bash
SIM_BACKEND=cupy python -X utf8 -m research.runners._fsg_watermaze_derisk \
    --seed 42 --goals "1,6;6,1" --n-trials 40 --steps-per-trial 200 --grid-size 8 \
    [--enable-apical-gated-corticostriatal]   # the new default-OFF flag (stage-2 edit)
```

This reuses the full deployed advantage limbic core (`--enable-neural-critic --spiking-snc --spiking-reward-us`,
`hidden_goal`, `trial_reset_steps`, `critic_warmup_trials`) — the ONLY change is the actor's plasticity is apical-gated.

**Stage-2 GO bar (inherited from the harness's own pre-registered verdict logic, multi-seed):**
- **(headline) Learning curve DECREASES + converges:** `delta_early_minus_late ≥ 1.0` AND `late_mean ≤ 2.5` at BOTH
  goals (`n_goals_learned_and_converged ≥ 2`) — the dendritic actor LEARNS where the point neuron did not.
- **Beats its own lesion:** reward-ON `late_mean` ≤ lesion `late_mean − 1.0` at both goals (`n_goals_on_beats_lesion ≥ 2`)
  — the load-bearing contrast.
- **Distinct goal-appropriate end positions** across the ≥2 goals (the goal-location anti-cheat; a fixed-corner drift
  that equals one goal is NOT learning).
- **Symmetrization guard holds:** the reward-OFF (lesion) curves stay FLAT, goal-INDEPENDENT, near the random floor
  (`lesion_goal_independent` True, `lesion_at_random_floor` True) — so the reward-ON learning is attributable to the
  apical-gated credit, not residual structural drift.
- **6-seed for the final claim** (the standing 6-seed rule for a variable effect); the 1-seed GPU smoke is the
  cheap-first gate before the 6-seed run.

**NOTE: stage 2 is CONTROLLER-OWNED GPU** (the moving-goal path imports cupy directly; `~0.3 s/step`, ~40 trials × 200
steps × 2 goals × {ON, lesion} ≈ the diagnostic's per-run cost). This de-risk is NOT delegated — the controller owns the
GPU run, per the standing rule that these long moving-goal de-risks are controller-tracked (the 2026-06-19 buggy-waiter
lesson: hand-rolled process-waiters give false "crashed" signals → duplicate contending runs).

### 3.3 Anti-cheat controls (so a GO is the apical gate, not an artifact)

- **POINT-NEURON CONTROL MUST FAIL** (the headline, both stages): the global-scalar actor MUST stay at the random floor /
  fixed-corner drift on the IDENTICAL toy/harness/seeds. A dendritic GO only counts against a point-neuron NEGATIVE on
  the same setup. (Re-tune the structural-bias magnitude until the point neuron fails before trusting any dendritic
  number — exactly the diagnostic's NE-drift.)
- **LESION-THE-APICAL-GATE → collapse** (both stages): zero the advantage into the apical compartment ⇒ the dendritic
  actor returns to the point-neuron floor. Proves the lift rides the gate. (Stage-2 = the harness's existing
  `--lesion-reward`, which zeroes the teaching signal — but ALSO a gate-specific lesion that keeps reward but cuts only
  the apical route, to separate "reward needed" from "apical-gating needed".)
- **GOAL-LOCATION ANTI-CHEAT → tracks the goal** (both stages): distinct goal-appropriate end positions across ≥2 goals
  (the diagnostic's decisive control — a fixed-corner drift that coincides with one goal is NOT learning).
- **SYMMETRIZATION GUARD** (stage 2): the lesion curve goal-independent + near the random floor (the harness asserts this).
- **ADVANTAGE-IS-NEURAL provenance** (stage 2): the apical drive is the SNc-derived `δ = r − V(place)` (the deployed
  core), NOT a host-computed advantage injected into the apical compartment — assert `current_reward_signal`-path is the
  neural critic, not a Python `r − V`. (The whole point is the brain computes the advantage; the apical gate just reads
  it. Under BRAIN-BASED-ONLY the advantage must be the substrate's, the place code the agent's own egocentric position,
  and only the environment/body are host.)
- **PERMUTED / WRONG-SIGN control** (stage 1, reuse the 2026-05-17 instrument): a wrong-sign apical gate (advantage sign
  flipped) MUST fail — guards against a non-sign-discriminating "pass" (the exact confound the 2026-05-17 adversarial
  review caught).
- **NO `sim/` edit in stage 1; the stage-2 edit is byte-identical-when-off** (the `fused_coincidence_plateau` precedent;
  byte-level diff review per the owner's standing rule for protected edits).

---

## 4. DOES IT ALSO UNLOCK #5 (the place-selectivity wall)?

**Honest answer: NO — not the SAME dendritic build, and the de-risk should hold #5 FIXED rather than rely on the
dendrite to solve it.** This is the most important framing correction.

- **#5 (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`) is a DIFFERENT dendritic function** than actor-critic
  credit assignment, and the project's own two-dendrite-stories framing
  (`2026-06-18-TD-cueshift-dendrite-decision-scoping.md` §2) draws the line cleanly:
  - **Credit assignment** (this de-risk) = the **SPATIAL / segregated-apical-basal / feedback-alignment** dendrite
    (Larkum BAC + GLR-2017 — exactly `sim/dendritic_neuron.py`): "which synapse gets credit," a *gating* operation.
  - **#5 place-selectivity** = forming MANY distinct, location-selective sparse codes from heavily-overlapping egocentric
    landmark sensors — a **per-cell nonlinear input-integration / decorrelation** problem (the Mikulasch-Priesemann
    *cross-neuron whitening* family, the OTHER dendrite, the one D2 Phase 1 built as the divisive gain). #5's own finding
    says it would plausibly need "per-cell nonlinear input integration to carve selective fields" OR a graded rate
    read-out — the *decorrelation/normalization* dendrite, NOT the apical-gating-credit dendrite.
- **So one apical-gating-credit build does NOT also deliver selective place fields.** They are orthogonal dendritic
  functions on different compartments with different rules (apical-gate-the-plasticity vs per-input-lateral-balance).
- **The intertwining is a CONFOUND, not a two-for-one.** The hidden-goal failure is *jointly* caused by (a) the
  credit-assignment rule AND (b) the place code not being selective enough — `2026-06-19-fsg-watermaze` §HONEST CAVEAT
  and the diagnostic §"What WOULD make it load-bearing" both flag this. **⇒ The de-risk MUST hold the place code FIXED +
  known-selective** (stage 1: a constructed sparse-selective place code; stage 2: the existing `sensor_place_readout`
  σ=0.5 already-selective code, NOT the #5-failing self-org place code) so a GO is attributable to the apical credit
  gate, not the place input. If the de-risk instead used the #5-failing place code, a NEGATIVE could be the place input
  (not the credit rule) and a GO could be luck — both confounded.
- **Caveat the place code's *partial* role honestly:** the diagnostic notes the actor may fail *partly* because it
  "cannot tell places apart." Holding the place code selective removes that as the cause; if the dendritic actor STILL
  fails with a selective place code, the wall is the credit-assignment rule itself (deeper than apical-basal) — a
  genuinely informative NEGATIVE (§5).

**Conclusion for the owner:** the dendrite de-risk addresses credit assignment ONLY; #5 (selective place fields) is a
SEPARATE dendritic question (the decorrelation/normalization dendrite, partly built as the D2 divisive gain, found
NOT-load-bearing for the cortex code) needing its OWN cheap-first de-risk if/when prioritized. Do NOT scope this de-risk
as solving both — and explicitly hold the place code fixed/selective so the credit-assignment result is clean.

---

## 5. HONEST RISK + THE STOP CRITERION

### 5.1 The biggest ways this de-risk could mislead

1. **The place-selectivity confound (the #1 risk).** If the place code is not held fixed/selective, a NEGATIVE could be
   "#5, not the credit rule" and a GO could be place-input luck. **Mitigation (load-bearing): hold the place code FIXED
   + known-selective** (§4). The de-risk tests the credit-assignment RULE with the place INPUT controlled.
2. **The 2026-05-17 precedent could repeat AND could be misread.** The off-bridge rate-level U-S rule already failed the
   discriminating isolation test (loss-ratio 1.095, alignment 0.012). If stage 1 naively re-runs *that* test it will
   re-derive that NEGATIVE — but that would be testing the wrong thing (a supervised MLP isolation, not the RL
   actor-critic with the genuine advantage teaching signal). **Mitigation:** stage 1 is the *RL gridworld actor-critic*
   toy with the *advantage* as the apical signal (the teaching signal the 2026-05-17 test lacked), NOT another XOR/MNIST
   isolation. Carry the 2026-05-17 sign-discriminating + wrong-sign controls so a "pass" can't be vacuous.
3. **Toy mis-calibration (the point neuron must genuinely fail).** If the structural-bias is too weak, the point-neuron
   control "succeeds" and the contrast is meaningless. **Mitigation:** the headline anti-cheat — re-tune until the
   point-neuron control reproduces the diagnostic's fixed-corner-drift floor BEFORE trusting any dendritic number.
4. **Stage-1-GO-but-stage-2-NEGATIVE (the rate-vs-spike gap).** A rate-level numpy GO may not survive the spiking-noise
   floor / the real cascade (the project has hit this repeatedly). **Mitigation:** stage 1 is the *gate*, not the claim;
   the load-bearing result is the on-bridge stage-2 `_fsg_watermaze_derisk.py` run. Report stage 1's scope explicitly
   (rate-level, off-bridge) and do not bank it as the answer.
5. **Host-shortcut leak (the advantage must be neural).** If the apical drive is a host `r − V` injected into the
   compartment, the "credit assignment" is partly host. **Mitigation:** the provenance anti-cheat (§3.3) — the apical
   drive is the deployed neural critic's SNc-derived advantage.

### 5.2 The clear three-state outcome (pre-registered)

- **GO** — the dendritic (apical-gated) actor LEARNS the hidden goal (curve decreases + converges, tracks ≥2 goals,
  beats its lesion) WHILE the point-neuron control fails on the identical setup, with the place code held selective, the
  lesion + goal-location + provenance anti-cheats clean, multi-seed (stage 1: 42/43/44; stage 2: 1-seed GPU smoke → 6
  seeds). ⇒ **the apical-basal dendrite IS the credit-assignment escape for the nav actor-critic → greenlight the fuller
  on-bridge build** (the months-scale `target_compartment`-routed two-compartment actor, owner-gated, byte-reviewed).
  This is the likely-positive direction given the literature (Payeur 2021 / GLR-2017 demonstrate it), tempered by the
  2026-05-17 feasible-local-scale caution.
- **BOUNDARY** — the dendritic actor beats the point neuron but only partially (learns one goal, not both; or converges
  slowly / above the threshold). ⇒ partial escape; informs whether the fuller form (more compartments / a burst-gated
  vs apical-gated rule / the place code also needs the decorrelation dendrite) is warranted, and whether to first
  improve the place code (#5) before re-testing.
- **NEGATIVE** — even the dendritic actor fails (stays at the floor / does not track the goal) with the place code held
  selective and the point-neuron control reproducing the failure. ⇒ **the wall is DEEPER than apical-basal credit
  assignment** — a clean, citable result that one extra compartment with the apical advantage gate does not crack the
  nav actor-critic on this substrate at feasible scale (consistent with, and extending, the 2026-05-17 NEGATIVE from the
  conversational W→A to the nav RL task). This SAVES the months-scale build and points to either the deeper
  multi-compartment form, the #5 place code as the true binding constraint, or a different algorithm (e.g. a
  model-based/successor-representation actor) — itself a major deliverable under "honest negatives are the deliverable."

**The stop criterion (when to STOP and report, not config-crank):** report the three-state outcome after the
pre-registered stage-1 multi-seed run (and, if GO, the stage-2 1-seed GPU smoke → 6-seed). Do NOT escalate a stage-1
NEGATIVE into an arch-B/arch-C config search (the 2026-05-17 discipline: "an Arch-A NEGATIVE is the terminus, NOT a
license to escalate"). A stage-1 NEGATIVE with the point-neuron control failing + the place code held selective + the
wrong-sign control failing IS the answer.

---

## 6. SUMMARY (the return)

- **Mechanism map (1-2 sentences):** the actor neuron's **basal** compartment integrates the bottom-up **place code**
  (`sensor_place_readout`), the **apical** compartment integrates the top-down **advantage** `δ = r − V(place)` (the
  deployed neural critic's SNc-derived RPE, via the fixed-random `B_apical` — no weight transport), and the apical-driven
  **BAC burst** gates the corticostriatal plasticity so `Δw ∝ pre · burst(place, δ)` — place-AND-advantage-specific
  credit a point neuron's global `DA × eligibility` cannot localize (Payeur-Naud-Richards 2021; GLR-2017).
- **Cheapest-first de-risk + GO bar:** **Stage 1 (afternoon, CPU/numpy, NO `sim/` edit)** — a gridworld actor-critic toy
  where the point-neuron control reproduces the fixed-corner-drift failure and the dendritic actor (advantage = apical
  gate via the existing `DendriticLayer`/`urbanczik_senn_update`) is the test arm; GO = the dendritic curve decreases +
  converges + tracks ≥2 goals WHILE the point-neuron control stays at the floor, lesion-the-apical-gate collapses it,
  place code held selective, wrong-sign control fails, 3 seeds. **Stage 2 (only if stage 1 GO; small default-OFF
  protected `sim/` edit + CONTROLLER-OWNED GPU)** — run `_fsg_watermaze_derisk.py` verbatim with the apical-gated actor;
  GO = the harness's own bar (`n_goals_learned_and_converged ≥ 2`, beats lesion, distinct goal-appropriate ends,
  symmetrization guard holds), 1-seed smoke → 6 seeds.
- **Reuse-vs-new:** REUSE `sim/dendritic_neuron.py` (apical/basal + BAC gate), `sim/dendritic_plasticity.py` (apical-gated
  local rule), `_fsg_watermaze_derisk.py` + `g11_bg_runner.py` (the exact hidden-goal/trial-structured/anti-cheat
  harness), the deployed advantage core (`--enable-neural-critic --spiking-snc --spiking-reward-us` = the apical teaching
  signal), the actor's `corticostriatal`-gated plastic pathways, and the `fused_coincidence_plateau` byte-identity-when-off
  template. NEW = the stage-1 toy (~80-150 lines numpy) + (stage 2 only) a minimal default-OFF apical-gate on the
  corticostriatal STDP (NOT the full N-compartment `NeuronModel`).
- **Does it unlock #5?** NO — credit assignment (the SPATIAL apical-basal dendrite) and #5 place-selectivity (the
  cross-neuron decorrelation dendrite, partly built as the D2 divisive gain, found NOT-load-bearing for the cortex code)
  are ORTHOGONAL dendritic functions; the de-risk must HOLD the place code fixed/selective to avoid the #5 confound, and
  #5 needs its own cheap-first de-risk if prioritized.
- **The decisive honest caveat for the owner:** the project ALREADY has an off-bridge dendritic-credit-assignment
  NEGATIVE (2026-05-17, feasible-local-scale, supervised MLP). This de-risk is worth running because it tests the
  *different, RL-specific, on-bridge* question with the *genuine advantage teaching signal the substrate now computes* —
  but the stop criterion respects that precedent: a clean stage-1 NEGATIVE (point-neuron control failing + place code
  selective + wrong-sign control failing) is the terminus, and reframes the wall as deeper-than-apical-basal — itself a
  citable deliverable that saves the months-scale build.

---

## 7. Sources

### Project record (re-verified this pass, file/finding cited)
- **The motivating NEGATIVE:** `research/findings/2026-06-19-fsg-watermaze-trial-structured-derisk.md` (the 12-trial
  trial-structured NEGATIVE; the harness `_fsg_watermaze_derisk.py` + `g11_bg_runner.py` flags), with
  `2026-06-19-spiking-actor-critic-advantage-routing-derisk.md` (the advantage IS already routed, code-verified) and
  `2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md` (the fixed-corner-drift mechanism + the next-lever:
  route the critic's advantage into the actor's place→action STDP).
- **The original credit-assignment verdict:** `research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`
  (global scalar fails 1/6 & 0/6; supervised gradient 3/3 → the credit-assignment RULE is the bottleneck) +
  `docs/plans/2026-05-05-dendritic-learning-design.md` (the prior apical-basal design, targeted at conversational W→A;
  the `target_compartment` routing + the reduced-2-compartment 1.5-2 month scope).
- **THE load-bearing internal precedent (a prior dendritic-credit-assignment NEGATIVE):**
  `research/findings/2026-05-17-dendritic-credit-assignment-NEGATIVE.md` (the local U-S rule did NOT do hidden credit
  assignment in the W2-frozen isolation test at feasible local scale; the sign-discriminating + wrong-sign + permuted
  controls preserved) + `2026-05-18-dendritic-fairscale-SOUND-instrument-VOID-strongest-triangulation.md`.
- **The existing dendrite infra (read in full):** `sim/dendritic_neuron.py` (Larkum BAC + GLR-2017 segregated apical/basal,
  fixed-random `B_apical`, BAC threshold-lowering), `sim/dendritic_plasticity.py` (`urbanczik_senn_update`, apical-gated,
  sign-corrected), `sim/dendritic_mlp.py` (GLR-2017 feedback alignment, GPU via `sim.backend`); the deep-research scoping
  `2026-06-14-dendritic-substrate-deep-research.md` (D1→D2 ladder, the reusable stack, the step-loop precedent) +
  `2026-06-14-dendritic-D1-cheap-derisk-GO.md` (the D1 cheap-first GO pattern, the gate-the-build discipline) +
  `2026-06-14-D2-phase1-DONE-phase2-frontier.md` (the on-bridge divisive gain delivered, byte-identical-when-off).
- **The two-dendrite-stories framing (spatial credit vs temporal/decorrelation):**
  `research/findings/2026-06-18-TD-cueshift-dendrite-decision-scoping.md` §2 (D2 = spatial decorrelation/feedback-alignment,
  not temporal) + `2026-06-17-dendritic-substrate-frontier-scoping.md` (the off-diagonal/PPMI reframe; D2 de-prioritized
  for the cortex).
- **#5 (the intertwined place-selectivity wall):** `research/findings/2026-06-19-place-code-sparsify-default-BOUNDARY.md`
  (the self-org place code is not location-selective at nav scale; needs per-cell nonlinear input integration OR a graded
  read-out — the OTHER dendrite).
- **The actor's plasticity wiring (verified):** `g11_bg_runner.py:1450` (`sensor_place_readout → cortex_{action}` plastic),
  `:1749` (`cortex → str_D1`, `plasticity_gate="corticostriatal"`), `cp_eligibility_trace` reward-modulated STDP;
  `sim/td_value_critic.py` (the critic), the deployed `--enable-neural-critic --spiking-snc --spiking-reward-us`.
- **Step-loop protected-edit template + flags:** `sim/bridge.py:5805-5849` + `sim/kernels.py:252`
  (`fused_coincidence_plateau`, guarded byte-identical-when-off); `sim/config.py:173/233/386` (the guarded dendritic flags,
  none of which is apical-gated credit routing).

### Peer-reviewed / current literature (re-confirmed this pass)
- **Payeur, Guerguiev, Zenke, Richards, Naud (2021)** "Burst-dependent synaptic plasticity can coordinate learning in
  hierarchical circuits," *Nature Neuroscience* — apical-dendrite-controlled bursting + STP multiplex feedforward/feedback
  so a top-down signal steers local plasticity without disrupting bottom-up signaling (THE actor-critic credit mechanism).
- **Guerguiev, Lillicrap, Richards (2017)** "Towards deep learning with segregated dendrites," *eLife* 6:e22901 — the
  segregated apical/basal + fixed-random feedback (= `sim/dendritic_neuron.py`); the SPATIAL credit-assignment dendrite.
- **Sacramento, Costa, Bengio, Senn (2018)** "Dendritic cortical microcircuits approximate the backpropagation algorithm,"
  *NeurIPS* (arXiv 1810.11393) — apical/basal dendritic-error microcircuit (rate-based).
- **Urbanczik & Senn (2014)** "Learning by the dendritic prediction of somatic spiking," *Neuron* — the committed
  apical-gated local rule (`sim/dendritic_plasticity.py`).
- **Frémaux, Sprekeler, Gerstner (2013)** "Reinforcement Learning … with Spiking Neurons," *PLoS Comput. Biol.* 9:e1003024 —
  the canonical spiking actor-critic (point-neuron) the nav circuit follows; the form this de-risk asks the dendrite to lift.
- **Larkum (2013)** "A cellular mechanism for cortical associations: an organizing principle for the cerebral cortex,"
  *Trends Neurosci.* — BAC firing (basal+apical coincidence → burst), the gate.
- **Cell *Patterns* (2025)** "Three-factor learning in spiking neural networks: …" — confirms the three-factor framing
  (eligibility × a neuromodulatory third factor; dopamine gates the actor's policy plasticity).
- **PNAS (2025)** "Spiking world model with multicompartment neurons for model-based reinforcement learning" (arXiv
  2503.00713) — existence proof that multicompartment SNNs scale to RL-class tasks.
- **Catalog `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`:** G.02 (active dendrites — MISSING, ~10×
  compute/neuron, BAC-firing behavioral validation, `:2644-2652`); C.30 (actor-critic — critic the named requirement);
  B.17 (MSN dendritic linearization, multi-compartment). Kandel 6e Ch 13 (passive + active dendrites).
