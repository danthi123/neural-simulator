# Tier-3 artificial-life capstone — deep-research scoping: the persistent living loop

**Date:** 2026-06-20
**Type:** read-only deep-research + catalog-review scoping (the standing deep-research-FIRST gate at a new direction). NO code written, NO experiments run. ONE findings doc.
**Roadmap tier:** Tier-3 — the artificial-life capstone (a persistent living agent). The owner-accepted post-conversational roadmap (`project_post_conversational_roadmap_tiers`) is Tier-1 (conversational loose ends) → Tier-2 (TRUE ONE BRAIN) → **Tier-3 (artificial-life capstone)** → Tier-4 (deep walls: dendrite / learned cortex). Tier-1, the shortcut burndown, the conversational primary, and Tier-2 #6 (limbic→composer) are DONE+verified (commits up to CYCLE 318). This gate scopes the next tier.
**Top-level goal this serves:** artificial life with a proper brain analogue + biology-translatable insights; capabilities are instrumental; honest negatives under strict biology ARE the deliverable (`project_actual_goal_artificial_life_brain_analogue`).
**Strict bar applied throughout:** everything cognitive between sensation and action MUST be neurons/synapses; host code is legitimate ONLY for the environment (world state + sensory render) and the body (acting on motor output) (`feedback_brain_based_only_standard`).
**Framing:** this is the OWNER's core goal and a VISION-level direction. Per the prompt, this gate scopes the MINIMAL concrete next step and presents OPTIONS for the owner to steer — it does NOT design the whole vision or commit to a build.

---

## 0. Top-line (read this first)

**A persistent living agent is much further along than a "next tier" framing suggests — the motivational core is BUILT and de-risked across every face, and the limbic reward/value/dopamine core is now co-resident on the merged "one brain."** The honest gap is NOT "build the drive" (done) and NOT "build the limbic core" (done, co-resident) — it is **assembling the validated pieces into ONE continuously-running outer loop where an interoceptive drive persists across resets and motivates behaviour, AND closing the single open mechanism wall (the actor-critic credit-assignment that makes the reward behaviourally load-bearing).**

There are TWO independent sub-gaps, and they have OPPOSITE risk profiles:

1. **The integration sub-gap (LOW risk, cheap-first achievable now):** wire the validated 2-pool spiking hunger drive onto the merged bridge (it is currently validated standalone + rate-proxy but is NOT co-resident — `grep` of `nav_conv_merged_bridge.py` returns **0** drive/hunger references), run it as a continuous `live()` outer loop where the drive persists across episode resets via the lineage, and show the drive motivates behaviour across modalities. The machinery for every piece exists; this is a wiring + scheduler job. **This is the recommended cheap-first de-risk.**

2. **The load-bearing sub-gap (HIGH risk, a characterized wall, likely Tier-4/dendrite):** for the drive's reward to *change the navigation policy* (not just ride on perception), the actor-critic must learn a place→action map from the intrinsic reward. This is the project's deepest learning wall and was hit a **3rd rigorous time on 2026-06-19** (the F-S-G water-maze NEGATIVE), with the verdict that it "resolves toward the DENDRITE" — a months-scale, owner-scoped arc. **This is NOT on the cheap-first critical path; the living loop can be demonstrated on the validated `sustain-your-energy-by-eating` behaviour (6-seed GO rate-proxy) WITHOUT solving it, because there the reward is load-bearing for survival even if the spatial policy stays simple.**

**The phased recommendation:** build the minimal persistent living loop (sub-gap 1) cheap-first — it is genuinely reachable on the existing substrate and is the smallest thing that makes the merged one-brain *a life rather than a battery of demos*. The honest scope of that demonstration is "self-regulating embodied agent with a continuous interoceptive drive" — real artificial-life progress — while being explicit that the *learned spatial policy* under it is bounded by the dendrite wall (sub-gap 2), which is the owner's call to schedule as Tier-4.

---

## 1. Verified state — what of "a persistent living agent" is ALREADY built (with evidence)

I read the actual code and findings (not the summaries) and verified every load-bearing claim. **More is built than a "scope Tier-3" prompt would assume**, exactly the situation the prompt flagged ("if MORE is already built than expected, SAY SO").

### 1a. The merged "one brain" — BUILT, and now WITH a co-resident limbic core

- `build_merged_nav_conv_bridge` (`research/runners/nav_conv_merged_bridge.py:447`) + `MergedNavConvAgent` (`:1200`) put nav (BG cascade, spiking WTA action selection) + perception (V1/IT) + conversation (parser + composer + dlPFC) on ONE `SimulationBridge`. Verified.
- **The limbic reward/value/dopamine core is now a co-resident opt-in** (`co_resident_limbic`, `nav_conv_merged_bridge.py:453`): `limbic_reward_us → limbic_snc ← limbic_striosome` (GABA_B/GIRK value subtraction) + the shared `dopamine` `from_region_firing_signed` modulator. Co-residence + nav-inertness + default-off byte-identity + moat-preservation all PASS, and the δ=r−V Schultz arithmetic works co-resident (`2026-06-18-merged-limbic-core-lift.md`, `2026-06-18-merged-config-homeostasis-boundary-RESOLVED.md`). The standalone organ passed the full Schultz RPE battery 6/6 (`2026-06-18-limbic-core-rpe-battery-GO.md`). There are also `co_resident_nav_critic`, `co_resident_td_cueshift`, and `enable_da_salience_gate` (the Tier-2 #6 DA→composer gate). **So the merged bridge already HAS a motivational substrate (reward + value + dopamine), which the 2026-06-17 scoping said it lacked** — this is the single biggest "more is built" update.

### 1b. The homeostatic DRIVE (the "why act") — BUILT + de-risked across all faces, but NOT co-resident on the merged bridge

The 2026-06-17 homeostatic arc built and validated the drive→reward→action motivational core (the catalog's #2-most-actionable addition), cheapest-first:

| Face | Status | Evidence |
|---|---|---|
| Reward STRUCTURE (intrinsic drive-reduction is learnable) | **GO 6 seeds** (rate-proxy + tabular Q) | `2026-06-17-homeostatic-drive-rl-cheap-first-GO.md` |
| Drive + reward ON REAL SPIKES (2-pool AgRP/POMC, `from_region_firing_signed`) | **GO 3 seeds** (corr(deficit,AgRP)=+1.0; eat → reward; lesion → 0) | `2026-06-17-homeostatic-spiking-drive-mechanism-GO.md` |
| Sustained "alive over time" (energy never crashes by self-directed eating) | **GO 6 seeds** (rate-proxy; lesion → crashes) | `2026-06-17-homeostatic-sustained-agency-GO.md` |
| Spiking agent INTEGRATION (place+motor+drive+hunger on one bridge, neural reward) | **BUILT + brain-faithful + functional 3 seeds**; robust policy convergence NOT achieved | `2026-06-17-homeostatic-spiking-agent-integration-BUILT.md` |

The reuse hook into the validated nav learner is real and verified: `run_moving_goal_episode(..., homeostatic_hook=None)` (`research/runners/g11_bg_runner.py:3193`, call site `:7159`-`7165`) — additive, default-`None` = byte-identical, invoked per trial after the natural reward, gates `reward *= hunger` and relocates food on an eat (`dist_after==0`). This is the documented reuse-not-fork foundation (`2026-06-17-homeostatic-g11bg-reuse-mechanism-GO-loadbearing-needs-perception-arc.md`).

**THE PRECISE GAP HERE:** the spiking drive is validated standalone (`_homeostatic_spiking_drive_mechanism_derisk.py`) and the minimal spiking agent is built on a *small* place+motor bridge (`_homeostatic_spiking_agent_integration.py`) — but the drive is **NOT a co-resident slice on the production merged one-brain.** `grep -cE "co_resident_drive|drive_hunger|agrp|hunger_drive|interocept|energy" nav_conv_merged_bridge.py` → **0**. There is no `co_resident_drive` flag. So the agent that has nav+perception+conversation+limbic on one bridge does NOT yet have the hunger drive that would motivate it.

### 1c. Continuous learning + persistence across sessions — BUILT

- `BridgeLineage` (`sim/lineage.py:140`) — atomic save (`.new` + fsync + atomic rename, `:190`/`:215`), `load` (`:238`), history snapshots, no-catastrophic-forgetting (validated Phase 1.3/1.4). `sim/auto_growth.py` (TierPromoter). So the *persistence-across-resets* machinery exists and is production.
- Memory: engram-tag API (`sim/bridge.py`), the composer's persistent fact store, hippocampal SWR consolidation (`consolidation_trainer.py`, Phase 1.3 CLS validated). **The trigger today is a scripted encoding phase, not a lived novelty/reward event** (per the 2026-06-17 diagnosis §1.4) — that residual stands.

### 1d. Embodiment — BUILT

The gridworld body (perceive→act) is the nav cascade. `run_moving_goal_episode` is the per-episode driver; `MergedNavConvAgent.navigate_to_compose_then_answer` is the perceive→ground→compose→answer behavioural task. These are **bounded function calls** (episodes), not a continuous existence.

### 1e. The honest one-line state

The agent has a competent cognitive engine (conversation, reasoning, generalization), a competent body (nav cascade), a validated motivational SUBSTRATE (limbic reward/value/dopamine, co-resident), and a validated motivational DRIVE (2-pool spiking hunger, de-risked but standalone). **What does not yet exist is (i) the drive wired onto the merged one-brain and (ii) a continuous outer loop in which that drive persists across resets and motivates the whole agent. Those two — wiring + scheduler — are the gap to "a life." The deeper wall (a *learned spatial policy* from intrinsic reward) is separately characterized and points at the dendrite (Tier-4).**

---

## 2. The minimal capability (NOT the whole vision)

**Name it precisely:** *A single continuous `live()` step-loop on the merged one-brain in which a co-resident interoceptive hunger drive (a) rises over time as a body-energy variable falls, (b) biases the agent's behaviour (action selection, and — as a stretch — conversational salience), (c) is satisfied by an eat action that drops the drive and yields a neural drive-reduction reward, and (d) persists across an episode reset via the lineage so the agent resumes the same internal state — with a drive-lesion collapsing the self-directed behaviour.*

This is the smallest thing that converts "nav + conversation + limbic co-located on one bridge" into "**one animal that lives**": it gives the agent a *reason to act* that originates inside it, runs *continuously* rather than per-episode, and *persists* as a self across time. It is explicitly NOT: a faithful free-energy solver, multiple competing drives, an amygdala aversive axis, a full day-in-the-life with circadian rhythms, or the learned-spatial-policy wall. Those are later phases / the owner's vision to steer.

The biology that grounds it (verified §6): homeostasis/allostasis as the root of motivation (Sterling; Kandel Ch 41), reward ≡ drive-reduction (Keramati-Gutkin), the AgRP/POMC push-pull (catalog O.06), incentive salience + drive-reduction reward (O.10/O.11). The field has independently converged on exactly this minimal living loop (interoceptive behaviour switching / continuous homeostatic RL, §6).

---

## 3. Reusable machinery vs the minimal NEW integration

### Reusable (high — this is mostly a wiring + scheduler job over validated subsystems)

| Need | Reusable asset | Where (verified) |
|---|---|---|
| The merged one-brain body+perception+conversation+limbic | `build_merged_nav_conv_bridge(co_resident_limbic=True, ...)` + `MergedNavConvAgent` | `nav_conv_merged_bridge.py:447,453,1200` |
| The 2-pool spiking hunger drive (AgRP/POMC) + neural drive-reduction reward | `_homeostatic_spiking_drive_mechanism_derisk.build_*` (the validated drive region + `from_region_firing_signed` "hunger" modulator) | `research/runners/_homeostatic_spiking_drive_mechanism_derisk.py`; `sim/neuromodulators.py` (`from_region_firing_signed`) |
| Gate the validated nav learner's reward by the drive + relocate food, NO fork | `run_moving_goal_episode(homeostatic_hook=...)` | `g11_bg_runner.py:3193,7159` |
| Make the drive a co-resident slice (the lift pattern) | the `co_resident_limbic` lift is the EXACT template (append regions after existing slices, index bases preserved, nav-inert, default-off byte-identical, per-region `enable_homeostasis` mask for the operating point) | `nav_conv_merged_bridge.py` + `2026-06-18-merged-config-homeostasis-boundary-RESOLVED.md` |
| Drive → incentive salience (boost deficit-cue pathway / action pool) | `ModulatorTarget(target_type="excitability_drive"/"synaptic_gain", scope="group:NAME")` | `sim/neuromodulators.py` |
| Drive → conversational salience (the DA→composer precision gate, already built) | `enable_da_salience_gate` reads the shared `dopamine` off the merged bridge → tightens the moat gate; a hunger-driven DA would flow straight in | `nav_conv_merged_bridge.py:1229`-1243 (`2026-06-18-DA-composer-precision-derisk-GO.md`) |
| Persist the internal state across a reset | `BridgeLineage.save/load` (atomic) — extend the saved attrs with the body-energy scalar + drive concentration | `sim/lineage.py:190,238` |
| Lived-memory consolidation in the loop | engram-tag API + `consolidation_trainer.py` SWR replay | `sim/bridge.py`, `research/runners/consolidation_trainer.py` |
| The drive-lesion / yoked-random anti-cheats (precedent) | the limbic-core lesion battery + the `lesion_reward` clamp | `2026-06-18-limbic-core-rpe-battery-GO.md`; `g11_bg_runner.py` (`lesion_reward`) |

### The minimal NEW integration (what does not exist yet)

1. **A `co_resident_drive` opt-in on `build_merged_nav_conv_bridge`** — append the validated 2-pool hunger drive (`agrp`/`pomc`) + the `hunger` neuromodulator as a co-resident slice, mirroring the `co_resident_limbic` lift exactly (regions after existing slices, nav-inert, default-off byte-identical, per-region `enable_homeostasis` mask for the operating point). **Likely NO `sim/` edit** (the lift template needed none; the per-region homeostasis mask + GABA_B are already-shipped). This is the single genuinely-new structural piece.
2. **A continuous `live()` driver** (a new runner, NOT a `sim/` edit) that, instead of resetting per episode: decays body-energy each step; lets the drive bias action selection + (stretch) conversational salience via the shared DA; on eat, drops the drive → neural reward → the existing learner; periodically runs the SWR consolidation phase on what was lived; and persists the body-energy + drive state through `BridgeLineage` across a reset. This is the "scheduler that makes the pieces a life," the 2026-06-17 Option 3.
3. **Lineage persistence of the body/drive scalar** — a few additional saved attributes (body-energy, drive concentration), so a reload resumes the same internal state. Likely a small additive change to the runner's save/load wrapper, not `sim/`.

**Net:** the build is one co-resident slice (templated on `co_resident_limbic`) + one outer-loop runner + a small persistence extension. The cognitive content (drive, reward, salience, consolidation) is all validated/reused. **Flag for the owner: the one place a `sim/` edit *might* surface is if the per-region-homeostasis operating point for the drive pools needs a tweak the existing mask can't express — but the limbic lift proved the mask suffices, so the prior is no-edit, additive-only.**

---

## 4. The recommended cheap-first de-risk (the decisive falsification)

**Probe (suggested):** `_persistent_living_loop_derisk.py` — a small `live()` runner on the merged bridge with the co-resident hunger drive. Cheap-first = 1-seed GPU smoke to decide GO/NEGATIVE before the 6-seed commit (`SIM_BACKEND=cupy` is required — the merged moving-goal path imports cupy directly).

**The minimal living loop under test:** the merged one-brain agent runs continuously in the gridworld; body-energy decays each step; the co-resident `agrp` drive rises as energy falls; the drive biases action selection (incentive salience via `excitability_drive` on the approach/action pool); reaching the food cell refills energy → the drive drops → the neural drive-reduction reward fires through the existing `reward_us → snc` chain; the loop persists body-energy + drive state through the lineage across one mid-run reset.

**The four decisive checks (explicit GO / BOUNDARY / NEGATIVE):**

| # | Check | GO | BOUNDARY | NEGATIVE |
|---|---|---|---|---|
| 1 | **Drive is neural + tracks the deficit co-resident.** corr(energy-deficit, `agrp` mean firing) over a free run on the merged bridge | corr ≥ +0.9 | +0.6–0.9 | < 0.6 (the co-residence broke the drive's f-I — the systemic merged-config risk the limbic lift hit) |
| 2 | **Self-directed survival (the "alive over time" property).** Over a long continuous run with energy depleting, does the agent keep itself alive (energy never crashes) by self-directed food-seeking, with NO externally-supplied goal? — the validated `2026-06-17-homeostatic-sustained-agency-GO.md` metric, now on the merged spiking bridge | min-energy stays well above the crash floor (e.g. ≥ 0.5 in the run's 2nd half); crash-rate 0% | survives but with dips near the floor | crashes repeatedly (the drive doesn't sustain behaviour) |
| 3 | **Persistence across a reset (the "self over time" property).** Save mid-run, reload, resume: the post-reload body-energy + drive concentration + behaviour resume the pre-reset state (not a cold start) | resumed energy/drive within a small tolerance of pre-save; behaviour continuous | resumes drive but not energy (or vice-versa) | a cold start (no persistence — the loop is still episodic) |
| 4 | **Cross-modal motivation (STRETCH — the "one animal" property).** When hunger is high, the shared DA (raised by the drive) tightens the conversational moat gate AND/OR biases which fact the agent volunteers, vs when sated — i.e. the SAME drive touches BOTH halves of the one brain | a measurable, drive-dependent shift in the conversational read (moat NEVER loosened) | a shift only in one modality | no cross-modal effect (the drive only touches nav) |

**Why this framing.** Checks 1+2 are the brain-based-only "is it a self-regulating life" core, re-using the already-validated drive + sustained-agency metrics but lifting them onto the *production merged bridge* (the new thing). Check 3 is the "persistent self" — the genuinely new outer-loop property (the lineage already does atomic save/load; this proves the *internal state* persists, not just weights). Check 4 is the "one brain, not two skills" property — it re-uses the already-built `enable_da_salience_gate` so the SAME dopamine the drive raises is the dopamine the composer reads, which is the cleanest possible demonstration of cross-modal motivation with **zero new mechanism** and the moat structurally safe (the DA gate can only tighten abstention, `nav_conv_merged_bridge.py:1236`-1239).

**Crucially, this de-risk is decoupled from the load-bearing-spatial-policy wall (§5/sub-gap 2):** checks 2+3 are survival + persistence, which the rate-proxy already showed are achievable WITHOUT a converged spatial policy (the short-corridor / dynamics make eating reachable; survival is the discriminator, not policy optimality). So a GO here is real artificial-life progress even while the spatial-policy wall stays open.

Multi-seed (≥6) before any "works" claim (`feedback_6seed_validation`); the cheap-first smoke is 1 seed to decide GO/NEGATIVE.

---

## 5. The anti-cheat controls the de-risk needs

1. **Drive-lesion (the decisive self-direction anti-cheat).** Zero the `agrp` drive (or the `hunger` modulator). Check 2's sustained survival MUST collapse (the agent starves like the validated `2026-06-17-homeostatic-sustained-agency-GO.md` lesion) and check 4's cross-modal effect MUST vanish. *If survival is unchanged with the drive lesioned, the "self-direction" is coming from perception/a leftover goal — NEGATIVE, exactly the trap the navigation N5 A/B fell into (`feedback_validate_signal_by_its_function`).*
2. **Yoked-random drive control.** Replace the `agrp` firing with a yoked random signal of matched mean/variance (no relation to the actual body-energy). Checks 1, 2, 4 must fail — proving the *coupling to the internal deficit* is load-bearing, not "any extra current makes it move/answer."
3. **Reward-provenance assertion (host-reward forbidden).** Assert `r` is read from `cp_firing_states` of the drive pool via `from_region_firing_signed` and that NO `r = f(distance_to_food)` host term exists (mirror the limbic-core reward-lesion: lesion `agrp → reward_us` → the RPE vanishes). Asserts the brain-based-only bar.
4. **No-persistence control (the "self over time" anti-cheat for check 3).** Run the identical loop but DON'T persist the body-energy/drive across the reset (cold-start the internal state). The post-reset behaviour must visibly differ (a re-warm transient) from the persisted run — proving the persistence is load-bearing, not that the agent re-derives the state in milliseconds anyway.
5. **No-confab moat untouched (co-residence anti-cheat).** Across the entire living run, assert the conversational `is None` abstention is byte-unchanged (the drive + the DA salience gate must never loosen the moat). The owner's standing: the moat is a plus, biologize-where-free, and the DA gate is moat-safe by construction (can only tighten) — but this must be ASSERTED, not assumed.

---

## 6. Biology grounding (verified against catalog + Kandel + literature)

Every load-bearing biological claim was checked against the actual source text.

- **Homeostatic drives as the root of motivation (the architecture).** Catalog **O.05 Hypothalamic Homeostatic Architecture** (`feature-catalog.md:4803`, Kandel 6e Ch 41 pp 1011-1013 Table 41-1): ~24 sensor→integrator→effector loops, emergent settling-points. Verified verbatim.
- **The two-pool push-pull (the mechanism).** Catalog **O.06 Arcuate POMC/AgRP/MC4R** (`:4815`, Kandel Ch 41 pp 1031-1037): "Two antagonistic populations" — AgRP (hunger) inhibits POMC (satiety); AgRP-stim in a sated mouse → ravenous eating. Maps onto an E/I region pair. Verified verbatim.
- **Reward ≡ drive-reduction (the reward definition).** Catalog **O.11 Drive Reduction Theory** ⭐ (`:4875`, Sternson group; Kandel Ch 41 pp 1038-1039): AgRP activation is itself aversive; relieving it is the reward; CPP/CPA behavioural signature. **Keramati & Gutkin, "Homeostatic reinforcement learning…," eLife 2014 (e04811)** — verified via search: proves seeking reward is *equivalent to* physiological stability when reward = the predicted reduction in the deficit (the project's exact basis; the existing dopamine-RPE δ=r−V consumes it unchanged). PubMed 25457346.
- **Incentive salience (drive → action bias).** Catalog **O.10 Incentive Motivation** (`:4863`, Berridge/Toates; Kandel Ch 41 pp 1037-1039): deprivation amplifies the *reward value* of goal cues (it does not generate behaviour directly). This is the `excitability_drive`/`synaptic_gain` salience boost.
- **The catalog's own #2-most-actionable addition is EXACTLY this.** `feature-catalog.md:4969`-4973 (verified): "Hypothalamic homeostatic drives (O.50/O.51/O.55/O.56): add hunger/thirst as slow-changing internal state variables that *modulate per-stimulus reward weights* AND act as aversive negative reinforcers… The neuromodulator subsystem already supports the right abstraction." The project has now IMPLEMENTED this (the `from_region_firing_signed` drive) — so the catalog's gap "complete absence of hypothalamic homeostatic drives" (`:4981`) is partially closed; the residual is *co-residence on the merged bridge + the continuous loop.*
- **Allostasis (the persistent/predictive framing — why a continuous loop, not a setpoint thermostat).** **Sterling, "Allostasis: A Brain-Centered, Predictive Mode of Physiological Regulation," Trends Neurosci 2020** — verified via search: the brain predicts needs and prepares before they arise, rewarding better-than-predicted outcomes with dopamine to learn regulatory behaviour; "a mean value need not imply a setpoint." This grounds the *continuous, learning, persistent* loop (the agent learns to anticipate hunger) over a static thermostat — and the dopamine-for-better-than-predicted is precisely the δ=r−V the limbic core already computes.
- **Active inference (one candidate unifying framing, NOT a recommended implementation route).** Friston (Free Energy Principle): an embodied agent acts to keep its sensations in preferred (homeostatic) ranges; expected free energy = an extrinsic value term (drives) + an intrinsic information-seeking term (curiosity). Useful as the *organizing principle* (drives + curiosity are two halves of one quantity), but a faithful EFE solver on spikes is heavy and off-bar — the project should build the *biological circuits* (hypothalamus, hippocampal-VTA) that approximate it, which is cheaper and on-brand (the 2026-06-17 scoping's honest caveat, which I endorse).
- **The field has independently converged on this minimal living loop** (verified via search): "Modularity benefits RL agents with competing homeostatic drives" (arXiv 2204.06608), "Continuous Homeostatic Reinforcement Learning for Self-Regulated Autonomous Agents," Interoceptive Mixture-of-Experts / Interoceptive Behaviour Switching (interoception switching policies), and 2025 meta-RL-in-homeostatic-regulation work. This is exactly the "internal body state biases behaviour, dynamic goal-switching, survival" loop — strong external validation that the minimal capability (§2) is the right target.

---

## 7. The honest reality check + phased options for the owner

### Is a persistent living loop achievable cheap-first on the existing substrate?

**YES for the minimal living loop (sub-gap 1) — and it is genuinely the smallest thing that makes the merged one-brain a life.** Every cognitive piece is validated; the build is a co-resident slice (templated on the proven `co_resident_limbic` lift) + an outer-loop runner + a small persistence extension; the prior is no `sim/` edit (additive-only). The "alive over time" + "self over time" properties (checks 2+3) are reachable WITHOUT solving the hard wall, because survival (not spatial-policy optimality) is the discriminator — the rate-proxy already showed 6-seed GO on exactly this. A GO here is real artificial-life progress: the project's first *continuously-living, self-regulating, persistent* embodied agent on one brain.

### What needs a major arc / the deferred dendritic substrate?

**The load-bearing *spatial policy* (sub-gap 2) is a characterized wall pointing at the dendrite.** For the drive's reward to *carve a learned place→action navigation policy* (so a hidden/relocating goal is solved by reward, not perception), the actor-critic credit-assignment must work — and on 2026-06-19 the F-S-G trial-structured water-maze de-risk returned a **3rd rigorous NEGATIVE** (`2026-06-19-fsg-watermaze-trial-structured-derisk.md`), with the verdict "resolves toward the DENDRITE" (apical-basal credit assignment + the entangled #5 place-selectivity boundary; the D2 Phase 0-2 two-compartment infrastructure exists, Phase 3 pending; `docs/plans/2026-05-05-dendritic-learning-design.md`). This is Tier-4, months-scale, owner-scoped. The honest framing: **the living loop demonstrates self-regulation NOW; a living loop whose *navigation is itself learned from intrinsic reward* is gated on the dendrite.** Do not conflate the two — the first is cheap-first reachable, the second is the deep wall.

### What an honest NEGATIVE on sub-gap 1 would look like (and would still be valuable)

The localized risk for the cheap-first de-risk is **check 1** — the systemic merged-config operating-point sensitivity that the limbic lift hit (a standalone-tuned spiking organ fires weaker co-resident; resolved for the limbic core via the per-region homeostasis mask, but each new organ re-tests it). If the hunger drive's f-I doesn't survive co-residence at a clean operating point, that is an honest BOUNDARY — "the drive works standalone but the merged operating point needs the same per-region-homeostasis recalibration the limbic core needed" — a bounded, characterized result (the fix template exists), not a mystery. The other genuine-NEGATIVE-but-valuable outcome is check 4 (cross-modal): if a hunger-raised DA does not measurably shift the conversational read, that pins "the drive motivates nav but not conversation on this substrate" — itself a mapped boundary.

### Phased options for the owner to steer (this is the OWNER's core goal — options, not a committed design)

- **Phase 3.0 (recommended cheap-first, ~days): the minimal persistent living loop.** Lift the validated hunger drive onto the merged bridge (`co_resident_drive`, templated on `co_resident_limbic`) + a `live()` outer-loop runner + lineage persistence of the body/drive state. De-risk = §4 checks 1-3 (+ stretch 4), anti-cheats §5. GO = the first continuously-living, self-regulating, persistent one-brain agent. **This is the recommended next step.**
- **Phase 3.1 (after 3.0 GO, optional, small): the cross-modal "one animal" demonstration.** Make the hunger-raised shared DA visibly touch the conversational half (check 4 promoted to a first-class result, reusing `enable_da_salience_gate`) + a second drive (thirst, catalog O.08) to demonstrate dynamic goal-switching (the literature's competing-drives result). Still point-neuron, still cheap.
- **Phase 3.2 (the lived-memory closure, medium): make consolidation lived, not scripted.** Trigger the existing SWR consolidation on a *lived* novelty/reward event during free behaviour (the 2026-06-17 residual §1.4), so the agent forms memories *because something happened to it*, then those change later behaviour. Reuses the engram + consolidation machinery; the new piece is the *trigger* (a CA1-mismatch novelty read, catalog D.23 — the 2026-06-17 Option-2 curiosity drive).
- **Phase 3.3 / Tier-4 (the deep wall, owner-scoped, months): the learned spatial policy from intrinsic reward.** The dendritic actor-critic (apical-basal credit assignment + graded place fields) that makes the drive's reward carve a *learned* navigation policy. Proposed clearly per `feedback_dendritic_substrate_fair_game`; the D2 Phase 3 build. This is the genuine frontier and the owner's call to schedule.

**Recommended order:** 3.0 → (3.1 / 3.2 in either order, both cheap) → Tier-4 dendrite when the owner chooses. The living loop is reachable now; the deep wall is the honest, separately-tracked frontier.

---

## Citations (source for every load-bearing claim)

**Project code (verified file:line):**
- `research/runners/nav_conv_merged_bridge.py:447` (`build_merged_nav_conv_bridge`), `:453` (`co_resident_limbic`), `:1200` (`MergedNavConvAgent`), `:1229`-1243 (`enable_da_salience_gate`, moat-safe-by-construction); `grep` → **0** drive/hunger references (the gap).
- `research/runners/g11_bg_runner.py:3193` (`homeostatic_hook=None` param), `:7159`-7165 (call site), `:3181` (`perceived_approach_reward`), `lesion_reward` clamp.
- `sim/lineage.py:140,190,215,238` (`BridgeLineage` atomic save/load); `sim/auto_growth.py`.
- `sim/neuromodulators.py` (`from_region_firing_signed`, `excitability_drive`/`synaptic_gain` targets).

**Project findings (verified content):**
- `2026-06-17-artificial-life-frontier-scoping.md` (the prior scoping — diagnosis + ranked options; this doc updates it with the now-built limbic core + the 2026-06-19 dendrite verdict).
- `2026-06-17-homeostatic-{drive-rl-cheap-first,spiking-drive-mechanism,sustained-agency}-GO.md` (the drive de-risked across faces); `2026-06-17-homeostatic-spiking-agent-integration-BUILT.md` (the spiking agent built, robust convergence not achieved); `2026-06-17-homeostatic-g11bg-reuse-mechanism-GO-loadbearing-needs-perception-arc.md` (the reuse hook + the load-bearing catch-22).
- `2026-06-18-limbic-core-rpe-battery-GO.md` (Schultz battery 6/6 standalone); `2026-06-18-merged-limbic-core-lift.md` (co-resident lift); `2026-06-18-merged-config-homeostasis-boundary-RESOLVED.md` (the systemic operating-point fix = the per-region homeostasis mask); `2026-06-18-DA-composer-precision-derisk-GO.md` (the DA→composer salience gate).
- `2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md` + `2026-06-19-fsg-watermaze-trial-structured-derisk.md` (the actor-critic credit-assignment wall, 3rd rigorous NEGATIVE → dendrite); `docs/plans/2026-05-05-dendritic-learning-design.md` (the Tier-4 dendrite arc).

**Catalog (verified verbatim, `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`):**
- O.05 (`:4803`), O.06 (`:4815`), O.08 (`:4839`, thirst), O.10 (`:4863`), O.11 (`:4875`); the "3 most actionable additions" #2 (`:4969`-4973); the gap note O.50/O.51 (`:4981`).

**Literature (verified via WebSearch):**
- Keramati & Gutkin, *Homeostatic reinforcement learning…*, eLife 2014 e04811 (reward ≡ drive-reduction; PubMed 25457346).
- Sterling, *Allostasis: A Brain-Centered, Predictive Mode of Physiological Regulation*, Trends Neurosci 2020 (predictive regulation; dopamine-for-better-than-predicted; no fixed setpoint).
- Friston, Free Energy Principle / active inference (the candidate unifying framing; EFE = extrinsic value + intrinsic epistemic) — adopted as organizing principle only, NOT an implementation route.
- Modularity benefits RL agents with competing homeostatic drives (arXiv 2204.06608); Continuous Homeostatic RL for Self-Regulated Autonomous Agents; Interoceptive MoE / Behaviour Switching; 2025 meta-RL-in-homeostatic-regulation — external convergence on the minimal living loop.
- Kandel 6e Ch 41 (hypothalamic homeostatic architecture, feeding loop, incentive/drive theories) — the textbook source the catalog entries cite.
