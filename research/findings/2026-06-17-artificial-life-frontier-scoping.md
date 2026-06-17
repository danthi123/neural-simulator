# Artificial-life frontier scoping: from episodic task-doer to a sustained, self-directed living brain-analogue

**Date:** 2026-06-17
**Type:** read-only deep-research + catalog-review scoping doc (gates + steers the next build)
**Standing workflow:** "deep research + catalog review FIRST at new directions" (owner directive, `feedback_deep_research_at_roadblocks`).
**Top goal this serves:** artificial life with a proper brain analogue; capabilities are instrumental; honest negatives under strict biology ARE the deliverable (`project_actual_goal_artificial_life_brain_analogue`).
**Strict bar applied throughout:** everything cognitive between sensation and action MUST be neurons/synapses; host code is legitimate ONLY for the environment (world state + sensory render) and the body (acting on motor output) (`feedback_brain_based_only_standard`).

---

## 1. Diagnosis — what actually separates the current agent from a *living* one

The substrate is mature. The agent can navigate a grid (basal-ganglia cascade selecting moves in spikes), perceive objects, compose a NEW fact about a perceived object, answer a query, abstain on the unseen, learn word meanings from a conversation stream, and do multi-hop reasoning — all multi-seed GO, all brain-based. **But every one of those is an *episodic task call*: an external caller sets a goal (a target cell, a query string), the agent runs a bounded episode, and then it stops.** Nothing inside the agent decides *that* it should act, *which* of several competing things to pursue, or *when* an episode is "done". Strip away the externally-supplied goal and the agent is inert — it has no reason to move.

Concretely, four mechanisms are missing, and they are missing as *neural* mechanisms (the host-scaffold versions don't count under the strict bar):

1. **No internal state that generates its own goals (no drive / no "why act").** The agent has no hunger, no thirst, no energy budget, no curiosity — no slow internal variable that says "I need X" and biases action toward getting X. Today's reward is *exogenous*: the navigation reward is literally a host formula — `g11_bg_runner.py:3132`, *"Default reward = +1 if Manhattan distance decreased, -1 if increased"* — a distance derivative computed in Python. Even the **biologized** reward (the validated `sc_rostral` goal-salience → `reward_us` → SNc dopamine-RPE, `2026-06-10-N5-reward-CLOSED-and-navigation-fully-biologized.md`) is still *triggered by an externally-placed goal beacon*. There is no neuron whose firing means "I am in deficit" and therefore *creates* the goal. **This is the single biggest gap to "alive".**

2. **The neural reward is not behaviorally load-bearing, because nothing makes the agent *choose*.** The N5 finding states this with unusual honesty: the gridworld is *orient-solvable* — perception (the superior colliculus) carries the behavior almost regardless of the reward, so "no reward (host or neural) is behaviorally load-bearing here, and no nav test can isolate it." The doc names the exact fix: *"a behavioral demonstration that the reward changes navigation would need a harder task (delayed/structured reward, or a remapped-action navigation where the policy must be learned from reward)."* **A self-generated drive is precisely the thing that turns reward into the dependent variable** — when the agent must satisfy one of several competing internal needs, the only way to behave well is to *learn from the reward which action reduces the active drive*. So this frontier is not orthogonal to the project's open navigation question — it *closes* it.

3. **No sustained perceive→cognize→act→learn loop over time.** Episodes are discrete function calls (`run_moving_goal_episode`, `MergedNavConvAgent.navigate_to_compose_then_answer`). There is no outer loop in which internal state (a drive level, a satiety level) *persists across episodes and shapes the next one*. The infrastructure for persistence exists (lineage, engram tags) but nothing wires a continuously-running internal-state vector into the step loop and lets it accumulate.

4. **No experiential memory growth driven by what the agent *lives*.** The agent learns word meanings from a *given corpus stream* and tags engrams *on command*. It does not yet form new memories *because something salient/novel/rewarding happened to it during free behavior*, then consolidate those, then have them change later behavior. The complementary-learning-systems machinery (hippocampal SWR replay → cortex, no catastrophic forgetting, `consolidation_trainer.py`, Phase 1.3/1.4) is built and validated — but its *trigger* is a scripted encoding phase, not a lived novelty/reward event.

**The honest one-line diagnosis:** the agent has a competent *cognitive engine* and a competent *body*, but **no motivational core** — no neural homeostatic/affective state that (a) generates goals, (b) defines reward intrinsically as the reduction of its own deficits, and (c) persists across time to make the agent self-directed. That motivational core is the hypothalamus-amygdala-VTA axis, and it is entirely absent (catalog: "complete absence of hypothalamic homeostatic drives; reward is exogenously defined rather than defended around setpoints", `feature-catalog.md` Cluster-O summary). Building it is the move from "task-doer" to "liver-of-a-life".

---

## 2. Ranked, biologically-grounded options

### Option 1 (TOP) — A neural homeostatic drive that generates its own goals and defines reward as drive-reduction

**The biology.** This is the catalog's own #2 "most actionable addition" (`feature-catalog.md`, Cluster-O summary, verbatim: *"Add hunger / thirst as slow-changing internal state variables that modulate per-stimulus reward weights (incentive motivation, O.10) AND act as aversive negative reinforcers (drive reduction, O.11). This produces dynamic goal-switching that fixed external rewards cannot. The neuromodulator subsystem already supports the right abstraction…"*). Catalog entries:
- **O.05 Hypothalamic Homeostatic Architecture** (Kandel 6e Ch 41 pp 1011-1013, Table 41-1): the hypothalamus runs ~24 sensor→integrator→effector loops, each defending an internal variable; emergent settling-points, not hard setpoints.
- **O.06 Arcuate POMC/AgRP/MC4R feeding loop** (Kandel Ch 41 pp 1031-1037): two antagonistic populations — AgRP (hunger-promoting) vs POMC (satiety) — a clean two-pool push-pull that maps onto an excitatory/inhibitory region pair. AgRP-stim in a sated mouse → ravenous eating (Sternson).
- **O.10 Incentive Motivation Theory** (Kandel Ch 41 pp 1037-1039; Berridge/Toates): deprivation does not generate behavior directly — it *amplifies the reward value (incentive salience) of the relevant goal stimuli*.
- **O.11 Drive Reduction Theory** ⭐ (Kandel Ch 41 pp 1038-1039; Sternson group): AgRP activation is *itself aversive*; relieving it is the reward. Gives a place-aversion / place-preference behavioral signature.
- **Computational unifier — Homeostatic Reinforcement Learning** (Keramati & Gutkin, *A Reinforcement Learning Theory for Homeostatic Regulation*, NIPS 2011; *Homeostatic reinforcement learning for integrating reward collection and physiological stability*, eLife 2014, e04811): proves reward-maximization and physiological-stability are the *same* objective if **reward ≡ the reduction in drive**, where drive = distance between the internal state and its setpoint, `d(Hₜ) = ‖H* − Hₜ‖`, and the per-step reward is `r = d(Hₜ) − d(Hₜ₊₁)`. This is the bridge that lets the project's *existing* dopamine-RPE actor-critic (δ = r − V, already spiking) consume a drive-reduction reward without changing the learning rule.

**The mechanism (spiking-native).** A two-pool hypothalamic region: a `drive_hunger` excitatory pool whose tonic firing *rises* as an internal "energy" variable falls below setpoint, antagonized by a `satiety` inhibitory pool (the AgRP↔POMC push-pull). The drive pool's firing does two NEURAL things via the neuromodulator subsystem: (a) **incentive salience** — `synaptic_gain`/`excitability_drive` boost on the perception→action pathway for the deficit-relevant cue (O.10), so a hungry agent's "food" percept drives action more strongly; (b) **drive-reduction reward** — when the body consumes the resource, the internal variable rises toward setpoint, the drive pool's firing *drops*, and that *drop in a neural firing rate* is read by `from_region_firing_signed` into the SNc dopamine RPE (O.11 + Keramati). The reward is therefore `r = −Δ(drive-pool firing)` — a quantity computed entirely from a neuron's firing rate, not a host distance formula.

**Spiking vs host scaffold — the line.** Legitimate host (= "the body/world", per the bar): the scalar internal-energy variable itself (it is part of the agent's *body/physiology*, exactly like the retinal image is part of the world the retina then receives — `O.05`'s "sensor" is interoceptive and is allowed as body state), and the environment delivering "food" when the agent's motor output reaches a resource cell. **Everything cognitive is neural:** the deficit→drive mapping is the drive pool's f-I curve; the goal-generation is the drive pool's firing biasing action selection; the reward is the *change* in the drive pool's firing read through the existing neural SNc. The host computes none of perception, salience, goal, or reward.

**Reuses existing machinery (high).** This is almost entirely a *wiring* job over validated subsystems — see §3. The catalog explicitly says the neuromodulator subsystem "already supports the right abstraction."

### Option 2 — A neural curiosity/novelty drive (the hippocampal–VTA loop) so the agent explores to *know*, not just to *get*

**The biology.**
- **D.23 Misplace system** (O'Keefe & Nadel 1978, Ch 2.3 pp 89-101, Ch 4.7.2 pp 195-209): CA1 "misplace" units fire when stimuli *mismatch* the stored map (novel/missing/rearranged); output drives investigative exploration and *one-trial* map update. The original hippocampal novelty-detection theory.
- **Hippocampal–VTA novelty loop** (Lisman & Grace, *The Hippocampal-VTA Loop*, Neuron 2005, 46:703-713): CA1 computes the mismatch (novelty) → subiculum → NAc → ventral pallidum → VTA → dopamine burst → DA released back into hippocampus enhances LTP/learning. A closed neural loop where **novelty is a teaching signal**.
- **Curiosity / intrinsic motivation** (Gottlieb, Oudeyer, Lopes & Baranes, *Information-seeking, curiosity and attention*, Trends Cogn Sci / PMC4193662, 2013; Oudeyer & Gottlieb learning-progress hypothesis): novelty / prediction-error / learning-progress act as *intrinsic reward* for unexplored states; midbrain DA is recruited as the motivational substrate. This is the neuroscience grounding of the RL "exploration bonus".
- **C.27 Wanting vs liking** (Kandel Ch 43 p 1068, Berridge): DA = wanting (incentive salience), not pleasure — consistent with novelty driving *approach* via DA.

**The mechanism (spiking-native).** The project's hippocampal trisynaptic loop is built and validated (D.12 pattern separation, D.13 completion, `validate_trisynaptic_loop.py`). Add a CA1 *mismatch* read: CA1's drive ∝ the *negative correlation* between current EC sensory input and the CA3-recalled pattern (exactly D.23's proposed mechanism, and what the catalog says is "missing… could be implemented as a CA1 region whose excitatory drive is modulated by the negative correlation between current sensor input and CA3-recalled pattern"). High CA1-mismatch firing → (via `from_region_firing` → a "novelty-DA" modulator → `excitability_drive` on the action/exploration pool) an **intrinsic exploration bonus computed by neurons**. Novelty is a neural mismatch, not `np.unique(states_visited)`.

**Why it's #2 not #1.** Curiosity is the *purer* "alive/self-directed" signal and reuses the validated hippocampus, but (a) it needs a behavioral arena where novelty is well-defined and *measurably* drives approach (a richer environment than the current beacon-gridworld), and (b) the CA1-mismatch read is one more moving part to de-risk than the two-pool drive. Homeostatic drive (Option 1) gives a cleaner first falsification and a sharper anti-cheat (lesion the drive → behavior loses self-direction). Curiosity is the natural *second* drive to add once the drive→reward→action loop is proven with hunger.

### Option 3 — Close the sustained loop: one continuous perceive→cognize→act→learn existence with persistent internal state

**The biology / framing.** This is the *integration* option, not a new circuit: take Option 1 (and/or 2) and run it as **one outer loop** where the internal-state vector persists across episodes and shapes the next one — "a day in the life." Grounding: O.05's homeostatic loops run continuously; the two-process and circadian machinery (N.09, N.10) frames long-timescale internal rhythms; active inference (Friston; *Free Energy Principle for Perception and Action*, Entropy 2022 / PMC8871280) is the unifying account that "priors act as drives or goals to enslave action" and that an embodied agent acts to keep its sensations in preferred (homeostatic) ranges, with the **expected-free-energy** objective decomposing into an extrinsic value term (Option 1) + an intrinsic information-seeking term (Option 2) — i.e. drives and curiosity are the two halves of one quantity.

**The mechanism.** A `live()` driver that, instead of resetting per episode: (1) lets drives decay/accumulate continuously (hunger rises with time/activity); (2) lets the *active* drive select which goal-cue the environment renders salient; (3) runs the existing nav cascade to act; (4) on resource contact, updates body state → drive drops → neural reward → dopamine learns the policy; (5) periodically enters a quiet/NREM phase that fires the existing SWR consolidation on what was lived. No new neural mechanism beyond Option 1/2 — it is the *scheduler* that makes them a life.

**Why it's #3.** It is the *goal*, but it is only meaningful once Option 1's drive→reward→action loop is shown to work in a single closed episode. Build the atom (Option 1), then the molecule (Option 3).

### Option 4 — An affective valence module (amygdala) for safety/threat, so the agent also has *aversive* self-direction

**The biology.** O.12 Amygdala fear-learning (LA/BLA/CeA; Kandel Ch 42 pp 1083-1099), O.13 amygdala appetitive/valence map, O.15 vmPFC→amygdala extinction. A ~50-100-neuron CS-US-convergence + output module gives the agent a *second* axis of motivation (avoid harm) beyond reward-seeking, and composes with the existing PFC region for top-down regulation.

**Why it's #4.** Genuinely valuable and small, but it adds a *second* motivational system before the *first* (homeostatic drive) is proven; it is best sequenced after Option 1 so the two can interact (the catalog notes biology routes threats through a separate substrate that *interacts* with reward differently). Lower leverage as the *first* step.

### Option 5 (reframe, not a build) — Treat "alive" as the active-inference objective and adopt drives + curiosity as its two terms

Not a separate build but the *organizing principle* for Options 1-3: expected free energy = extrinsic (drive-reduction value) + intrinsic (epistemic novelty) (active-inference literature above). Worth stating so the build is understood as instantiating one coherent objective on spikes, not bolting on two unrelated hacks. Honest caveat: a *faithful* free-energy/EFE agent on spikes is heavy and is NOT recommended as the implementation route — the project should build the *biological* circuits (hypothalamus, hippocampal-VTA) that *approximate* EFE, which is both cheaper and more on-brand than a variational free-energy solver.

---

## 3. Existing project machinery that is reusable (specific, load-bearing)

| Need | Reusable asset | Where |
|---|---|---|
| **Declare a drive as a slow internal modulator** | `NeuromodulatorConfig` + `NeuromodulatorManager` — declarative, opt-in, runs `manager.step(bridge)` once/step after reward modulation | `sim/neuromodulators.py` |
| **Make the drive signal NEURAL (read a population's firing)** | `ProductionRule(rule_type="from_region_firing")` and its signed sibling `"from_region_firing_signed"` — reads `bridge.cp_firing_states` over `source_regions`, EMAs it, emits concentration. *This is the exact precedent the spiking SNc dopamine already uses.* | `sim/neuromodulators.py:736` (`from_region_firing`), `:774` (`_signed`) |
| **Slow tonic build-up of a deficit over time** | `ProductionRule(rule_type="from_error_persistence")` — EMA that rises while a quantity stays above threshold (built for tonic NE under sustained stress; structurally a drive that accumulates) | `sim/neuromodulators.py:670` |
| **Drive → incentive salience (boost the deficit-cue pathway)** | `ModulatorTarget(target_type="synaptic_gain" / "excitability_drive", scope="group:NAME" / "all")` + `compute_excitability_drive_per_neuron` | `sim/neuromodulators.py:562` |
| **Drive-reduction reward into the existing dopamine RPE** | the validated neural reward chain `reward_us → snc` + signed dopamine modulator (δ = r − V, N9 critic supplies V) — feed `r = −Δ(drive firing)` here instead of the host Manhattan derivative | `g11_bg_runner.py` (`--enable-spiking-sc-approach`, `reward_us`, `snc`); `2026-06-10-N5-reward-CLOSED…md` |
| **The body + perception to act on a drive** | the merged nav+conv bridge: BG action cascade selects moves in spikes; perception renders into `cortex_it`; composer co-resident | `nav_conv_merged_bridge.py:447` `build_merged_nav_conv_bridge`, `:883 MergedNavConvAgent`; episode driver `run_moving_goal_episode` (`g11_bg_runner.py:3065`) |
| **`from_novelty` rule (Option 2)** | **reserved but UNIMPLEMENTED — emits 0** (`sim/neuromodulators.py:732`). A CA1-mismatch novelty drive can either implement this stub or, more simply, reuse `from_region_firing` sourced from a CA1 mismatch pool | `sim/neuromodulators.py:732` |
| **Hippocampal mismatch substrate (Option 2)** | validated trisynaptic loop (D.12/D.13 PASS): `build_biological_brain_regions(enable_hippocampus_consolidation=True)`, `validate_trisynaptic_loop.py` | `text_minimal_isolation.py`, `research/runners/validate_trisynaptic_loop.py` |
| **Persist internal state + grow memory across a life (Option 3)** | `BridgeLineage` (atomic save, history, no-catastrophic-forgetting), `consolidation_trainer.py` (SWR replay awake/sleep gate), engram-tag API (`start/commit_engram_tag`, `stimulate_tag`) | `sim/lineage.py`, `research/runners/consolidation_trainer.py`, `sim/bridge.py` (engram methods) |
| **Firing-rate homeostasis is a DIFFERENT mechanism — do not conflate** | `enable_homeostasis` etc. regulate per-neuron *firing-rate* around a target; the prior "integrated-loop homeostasis works" finding used THIS, not drives | `sim/config.py:251-256`; `2026-05-19-integrated-loop-iter2…md` |

**Net:** Option 1 needs **no new neural kernel** and (per the strict bar) **no `sim/` edit beyond, at most, a tiny additive interoceptive-state read** — it is a runner that declares a two-pool drive region + a drive modulator wired through the *existing* neuromodulator and SNc machinery.

---

## 4. The recommended cheapest-first de-risk (the decisive falsification, CPU/numpy)

**Probe name (suggested):** `_homeostatic_drive_reward_cheap_first_probe.py` (a tiny `SIM_BACKEND=numpy` runner; NO heavy GPU run, NO `sim/` edit).

**Build (smallest possible):** a minimal `SimulationBridge` with a 2-pool drive region — `drive_hunger` (excitatory) + `satiety` (inhibitory, AgRP↔POMC push-pull) — plus a 1-D internal "energy" scalar in *body* state. Declare ONE neuromodulator `hunger_drive` with `production_rules=[from_region_firing_signed(source_regions=["drive_hunger"], threshold=tonic_rate)]` and `targets=[excitability_drive(scope="group:approach"), synaptic_gain(scope="group:food_cue_pathway")]`. Wire the reward as `r = −Δ(drive_hunger firing)` into the existing `reward_us → snc` chain (NOT a host distance formula). Two "resource cells" in a toy 1-D corridor; the body's energy rises on contact and decays with time.

**The four decisive checks, with explicit numeric GO/BOUNDARY/NEGATIVE thresholds:**

| # | Check | GO | BOUNDARY | NEGATIVE |
|---|---|---|---|---|
| 1 | **Drive is neural + tracks deficit.** corr(internal-energy-deficit, `drive_hunger` mean firing) over a free run | corr ≤ −0.9 (firing rises as energy falls) | −0.9 < corr ≤ −0.6 | corr > −0.6 (drive doesn't track deficit → f-I mapping wrong) |
| 2 | **Self-generated goal (no external goal set).** Fraction of approach-actions toward the resource when deficit is high vs when sated, with NO externally-supplied goal cue | high-deficit approach-rate ≥ 2× sated approach-rate | 1.3×–2× | < 1.3× (drive doesn't bias action → no self-direction) |
| 3 | **Drive-reduction reward drives a correct dopamine RPE.** On resource contact (drive drops), SNc burst vs tonic; on a withheld-resource trial (deficit high, no contact), SNc dip below tonic (the Keramati/Schultz signature) | burst ≥ 3× tonic AND a measurable omission dip < tonic | burst ≥ 2× tonic, dip absent | no burst (reward not wired to DA through neurons) |
| 4 | **The reward is behaviorally load-bearing (the thing nav couldn't show).** With a *remapped* action→direction mapping the agent has never seen, does the policy learn (from the drive-reduction reward) to reach the resource faster over trials? slope of time-to-resource vs trial | significantly negative slope (learns); ≥ 30% reduction by end | shallow negative (weak learning) | flat/positive (reward not load-bearing even with a self-generated goal → the honest wall, see §6) |

**Why these numbers / this framing.** Checks 1+3 enforce the brain-based-only bar (the drive and the reward are *neural firing rates*, validated the same way the project validated the SNc — corr, burst, dip). Check 2 is the "is it alive / self-directed" test in its minimal form. Check 4 is the one the N5 finding said navigation *couldn't* give — a task where the reward is the dependent variable because the agent must *learn* the policy from its own drive-reduction signal. Multi-seed (≥6) is required before any "works" claim per `feedback_6seed_validation`, but the cheap-first numpy probe runs 3 seeds to decide GO/NEGATIVE before committing GPU.

---

## 5. The anti-cheat controls the de-risk needs

1. **Lesion the drive (the decisive self-direction anti-cheat).** Silence `drive_hunger` (or zero the `hunger_drive` modulator). Check 2 must collapse to ~1× (no deficit-vs-sated difference) and check 4's learning must vanish. *If behavior is unchanged with the drive lesioned, the "self-direction" was coming from somewhere else (a leftover external goal, perception) — NEGATIVE, exactly the trap the N5 nav A/B fell into.*
2. **Yoked-random drive control.** Replace `drive_hunger` firing with a yoked random signal of matched mean/variance (same statistics, no relationship to the internal deficit). Checks 1, 2, 4 must fail. This proves the *coupling to the internal state* is load-bearing, not just "any extra current makes the agent move."
3. **Host-novelty/host-reward forbidden — assert the source.** The probe must assert that `r` is read from `cp_firing_states` of the drive pool (the `from_region_firing_signed` path), and that NO `r = f(distance_to_resource)` host term exists. (Mirror of the N5 lesion anti-cheat: lesion `drive_hunger→reward_us` → the RPE must vanish. If it doesn't, a host shortcut is leaking the reward.)
4. **Sated-but-cued specificity (Sternson O.11 signature).** When sated (energy at setpoint) but the food cue is present, the agent should NOT preferentially approach (drive low → low incentive salience). And AgRP-stim analogue (force `drive_hunger` high) in a sated body → approach returns. This is the optogenetic AgRP place-preference/aversion test in silico — it dissociates "responding to the cue" (cheat) from "responding to the *drive*" (real).
5. **No-confab moat untouched (where the conversational slice is co-resident).** If the probe runs on the merged bridge, assert the conversational `is None` abstention is byte-unchanged across the drive run (the drive must not perturb the moat) — the standing co-residence anti-cheat.

---

## 6. Honest reality check — is sustained autonomous agency reachable now, and what would an honest NEGATIVE look like?

**Reachable, with one genuine risk localized.** The *mechanism* is unusually well-de-risked before we start: the project already has (a) the neuromodulator abstraction the catalog explicitly says is the right home for drives, (b) a *validated* way to source a modulator from a neural firing rate (`from_region_firing_signed`, proven on the spiking SNc), and (c) a *validated* neural dopamine-RPE that can consume a drive-reduction reward unchanged (δ = r − V). So the drive→reward→dopamine half (checks 1+3) is low-risk — it is re-applying a proven pattern. The two-pool hypothalamic push-pull (O.06) is a small, standard E/I region. **The genuine risk is check 4 — behavioral load-bearing.** It is entirely possible the drive cleanly generates a goal and a correct dopamine signal, yet the *policy learning* from that reward stays at the navigation finding's documented wall (the point-neuron three-factor rule learning a remapped sensorimotor policy from a sparse self-generated reward). That would be an **honest NEGATIVE of high scientific value**: it would pin the boundary precisely — "a self-generated homeostatic drive *can* be built on this substrate and *does* generate goals + a correct neural reward, but the point-neuron substrate cannot yet *learn the policy* from intrinsic drive-reduction reward in a task that requires it" — which maps exactly onto the project's actual-goal deliverable (honest negatives under strict biology ARE the science) and would name the next wall (likely the same credit-assignment frontier the project has mapped for navigation, not a new mystery). Either way — GO (the agent becomes minimally self-directed and the nav reward finally becomes load-bearing) or NEGATIVE (the precise learning wall for intrinsic reward is mapped) — the result advances the artificial-life goal. The one thing that would NOT be a clean result is skipping the anti-cheats: without the drive-lesion and yoked-random controls, an "it moves toward food!" demo is worthless, because perception alone can produce it. The controls in §5 are what make this a real test of *agency* rather than a reflex.

---

## Citations (source for every load-bearing biological claim)

- O'Keefe & Nadel, *The Hippocampus as a Cognitive Map* (1978) — D.21/D.22/D.23 misplace/novelty-driven exploration (Ch 2.3 pp 89-101; Ch 4.7.2 pp 195-209).
- Lisman & Grace, "The Hippocampal-VTA Loop: Controlling the Entry of Information into Long-Term Memory," *Neuron* 46:703-713 (2005) — neural novelty→VTA-DA→LTP loop.
- Keramati & Gutkin, "A Reinforcement Learning Theory for Homeostatic Regulation," *NeurIPS* (2011); "Homeostatic reinforcement learning for integrating reward collection and physiological stability," *eLife* 3:e04811 (2014) — reward ≡ drive-reduction.
- Gottlieb, Oudeyer, Lopes & Baranes, "Information-seeking, curiosity and attention: computational and neural mechanisms," *Trends Cogn Sci* 17:585-593 (2013, PMC4193662); Oudeyer & Gottlieb, learning-progress / intrinsic-motivation chapter (2016) — curiosity / intrinsic reward / DA.
- Berridge & Robinson — wanting vs liking / incentive salience (Kandel 6e Ch 43 p 1068; Ch 41 p 1038) — C.27/O.10.
- Sternson group — AgRP aversion / drive-reduction (Kandel 6e Ch 41 pp 1038-1039) — O.11.
- Kandel 6e Ch 41 (pp 1011-1013 Table 41-1; pp 1031-1039) — O.05/O.06/O.07/O.10/O.11 hypothalamic homeostatic architecture, feeding loop, incentive/drive theories.
- Kandel 6e Ch 42 (pp 1083-1102) — O.12/O.13/O.15 amygdala valence (Option 4).
- Friston et al., active inference / free energy: "The Free Energy Principle for Perception and Action: A Deep Learning Perspective," *Entropy* 24:301 (2022, PMC8871280); active-inference syntheses (expected free energy = extrinsic value + intrinsic epistemic) — Option 3/5.
- Schultz 1998/2007 (project-internal grounding of the existing dopamine-RPE machinery) — δ = r − V.
- Catalog (`sim-catalog/references/feature-catalog.md`): Cluster-O entries O.05/O.06/O.10/O.11/O.12/C.27; Cluster-O summary "3 most actionable additions" #2; Cluster-D D.23.
- Project findings: `2026-06-10-N5-reward-CLOSED-and-navigation-fully-biologized.md` (neural reward + the orient-solvable load-bearing limitation); `2026-05-19-integrated-loop-iter2-homeostasis-works…md` (firing-rate homeostasis ≠ drives, do-not-conflate); `g11_bg_runner.py:3132` (the host Manhattan-distance reward shortcut).
