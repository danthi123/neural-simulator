# Full spike-ification onto the shared substrate — AUDIT + prioritized roadmap + cheap-first de-risk for the limbic core

**Date:** 2026-06-18
**Type:** READ-ONLY audit + research synthesis (no code edited). The standing opening move for the owner's CYCLE-205 top-level directive: *"move every bit of the sim possible onto the shared, spiking substrate — true 'one brain'."* Two axes, both maximized: (1) SPIKE-IFY host-computed cognition (the brain-based-only bar: host code legitimate ONLY for the environment + the body); (2) CONSOLIDATE the spiking pieces onto ONE co-resident interacting `SimulationBridge`.

**Bottom line.** The project is much further along than "navigation half PARTIAL" suggests at the *mechanism* level — N1/N5/N6/N8/N9 are all built as spiking organs, and the dopamine broadcast already runs through a spiking SNc population. But two structural facts dominate the roadmap: **(A)** the spiking reward/value/dopamine machinery is **opt-in behind flags in the standalone `g11_bg_runner` and is NOT on the merged "one brain"** (the merge builder calls `build_bg_brain_regions` with *default* kwargs = a construction smoke), so the consolidated brain currently has **no limbic core at all**; and **(B)** the deployed default loops still fall back to host shortcuts (host Manhattan reward, host argmax read-out, host-numpy TD critic / predictive-coding modules that never touch a bridge). The highest-leverage move is therefore not inventing a new organ — it is **lifting the already-validated spiking SNc/critic/reward core onto the merged bridge as the shared limbic system** that both navigation and conversation draw motivation/salience/learning-gate from, and replacing the residual host glue (TD bootstrap, the WTA read-out) with spikes.

---

## 1. Audit table — each cognitive computation, classified

Legend: **SPIKING** = computed by simulated neurons/synapses on a bridge. **HOST-SHORTCUT** = computed in Python (a formula/argmax), even if biologically shaped — a shortcut by the brain-based-only bar. **SEPARATE-BRIDGE** = realized on its own bridge / numpy module, not co-resident on the merged "one brain". **HOST(env/body)** = legitimate host residual (world state, sensory render, or acting on motor output).

### 1a. Navigation sensorimotor loop

| Cognitive computation | Class | File : function | Note |
|---|---|---|---|
| Orienting reflex (which way to the goal) | **SPIKING** (N1, CLOSED) | `g11_bg_runner.py` : `install_spiking_sc_wiring` / `--enable-spiking-sc`; `sim/visual_cortex.py` retina | retinotopic `sc_retina→sc_map`(Mexican-hat)→`cortex_{N,E,S,W}` pooling; 6-seed beats host reflex, scrambled-retinotopy lesion regresses 2.4×. The legacy `sc_orienting_cardinal_from_image` host reflex is the *weanable scaffold* (still the default unless `--enable-spiking-sc`). |
| Perception — V1 simple cells | **SPIKING (innate, defensible)** | `sim/visual_cortex.py` : `build_v1_simple_weights`, `render_gridworld_to_image` | fixed Gabor afferent weights (Hubel-Wiesel; catalog E.08/L.05). N7 = defensible. |
| Perception — IT/ventral object code | **SPIKING** | region `cortex_it` (driven by the rendered image) | the perceived-state code the critic + memory read; shared with conversation (see §1b). |
| Goal cue (beacon) | **HOST(env)** (N2, defensible) | `render_gridworld_to_image(goal_pos=…)` | beacon rendered into the retina = legitimate sensory input (catalog E). |
| Reward value `r` | **SPIKING** when `--enable-spiking-sc-approach` / `--spiking-reward-us` (N5, CLOSED at mechanism level) — else **HOST-SHORTCUT** | spiking: `sc_rostral`(proximity/goal-salience)→`reward_us`(PPN-like) burst; host: `g11_bg_runner.py:6901-6946` (Manhattan / beacon-intensity / `perceived_approach_reward` sign formula) | the spiking reward is validated by the proper RPE battery (`sc_n5_rpe_probe.py`): burst on US, corr(distance, SNc)=−0.99, omission dip, **lesion `sc_rostral→reward_us` → burst vanishes**. BUT the **deployed default episode loop still computes a host `reward` scalar** (lines 6901-6946) and writes it to `current_reward_signal`. |
| Dopamine signal / RPE δ | **SPIKING** when `--spiking-snc` (N9, CLOSED) — DA broadcast = SNc firing | `g11_bg_runner.py` `_I_snc` block (~6730-6750); `sim/neuromodulators.py` : `from_region_firing_signed` over `["snc"]` | DA conc = the spiking SNc pool's windowed firing; sub-tonic firing dips conc below baseline (the omission half). The `dopamine` modulator's `plasticity_rate scope=all` gates the actor's three-factor learning. |
| Value subtraction `−V` (the critic) | **SPIKING + physiological** when `--enable-neural-critic` (Stage B) | `build_bg_brain_regions(...)` `striosome_value` MSN pool; `cortex_it→striosome_value` plastic (gate `value_input`); GABA_B/GIRK `striosome_value→snc` (E_K=−90mV) | δ=r−V at the SNc membrane is fully synaptic (r=excitation, V=GABA_B). Stage-B gap probe passes 4/4. |
| Value baseline `V` (Stage-A fallback) | **HOST-SHORTCUT** | `g11_bg_runner.py` Stage-A `_V_scaffold` = `reward_ema` | when `--enable-neural-critic` is OFF, V is the host reward-EMA (a Rescorla-Wagner baseline). |
| **TD bootstrap critic** δ = r + γV(s′) − V(s) (the cue-shift / Schultz signature (a)) | **SEPARATE-BRIDGE / HOST-numpy + MISSING on substrate** | `sim/td_value_critic.py` (pure numpy, "NO automatic differentiation"; never touches a bridge) | the **one canonical dopamine signature still not on the substrate** (catalog C.28/C.30/C.31). The deployed reward path is Rescorla-Wagner (δ=r−V), not the TD bootstrap that migrates the burst onto the earliest predictor. |
| Place / position code | **SPIKING (self-org)** when `--neural-place-selforg` — else dense host-Gaussian | `g11_bg_runner.py` `_run_critic_warmup` self-org `place`; host: `_n9_place_sensor_act` (egocentric landmark render) + `vs_place_context` Gaussian | the *sensor render* takes `(x,y)` but is a defensible body/sensory boundary (egocentric landmark bearings/distances, "(x,y) enters the brain via render"); the place *field carving* is spiking self-org. The dense `vs_place_context` Gaussian (host-rendered each step) is a scaffold. |
| Action selection / motor read-out | **HOST-SHORTCUT (argmax)** — signal source biologized; the decision-as-spike is `spiking_wta` opt-in | `g11_bg_runner.py:6865-6873` (`action_idx = max(range(N_ACTIONS), key=…)`); `readout_source ∈ {motor, thal, spiking_wta}`; `sel_{X}`/`commit_{X}` accumulator-then-commit regions | N6 = **PARTIAL** (documented residual). `--readout-source thal` reads the cleanly-selective thalamus (biologized SOURCE); `spiking_wta` adds Wang-2002 accumulator + Lo-Wang commit-burst pools so the **commit threshold-crossing IS the spiking decision** and the host argmax is a tie-break of last resort. But the **deployed default is `readout_source="motor"` = host argmax over motor spike-counts**, and the fully-spiking WTA was historically *worse* on the nav score (motor-WTA 14.7 vs thal-argmax 2.3). |
| Acting on the chosen action (move) | **HOST(body)** | `g11_bg_runner.py:6884-6892` (`x,y += ACTION_DELTAS[action_idx]`) | legitimate body residual. |

### 1b. Shared systems (serve BOTH halves — highest leverage)

| System | Class | File : function | Note |
|---|---|---|---|
| **Reward / value / dopamine limbic core** | **SPIKING in standalone nav (opt-in flags); ABSENT on the merged bridge** | nav: `g11_bg_runner` `snc`/`reward_us`/`striosome_value` + `sim/neuromodulators.py` `dopamine` (signed-firing rule); merge: `nav_conv_merged_bridge.py:506` calls `build_bg_brain_regions(n_cortex=…)` **default kwargs** | **THE central consolidation gap.** The merge note explicitly says "DEFAULT kwargs — this is the construction smoke, not the flagship", so `striosome_value`/`snc`/`reward_us`/the spiking SC are **NOT built on the one brain**. The conversation half has **no reward/value/DA at all**. A single shared spiking SNc+critic+reward limbic core would serve nav motivation AND (future) conversational salience/value. |
| Neuromodulator subsystem | **SPIKING-driven (declarative); scalar concentration, broadcast** | `sim/neuromodulators.py` : `NeuromodulatorManager.step`, `from_region_firing_signed`, `compute_plasticity_rate_multiplier` | DA/ACh/neuropeptides are **scalar concentrations driven by spiking populations** (e.g. DA conc ← SNc firing). The concentration→effect is host arithmetic but the *production* is neural (`from_region_firing[_signed]`, `from_region_firing` for D1/D2 peptides). This is the correct, reusable hinge for the limbic core. **Not currently enabled on the merged bridge.** Honest residual: the composer's RF ops bypass `_run_one_simulation_step`, so NM doesn't reach the composer (it reaches the parser). |
| Perception (V1/IT) | **SPIKING + SHARED** | `sim/visual_cortex.py`; merged `cortex_it` region (`nav_conv_merged_bridge.py` `co_resident_perception`, generalization stack `gen_perception`/`gen_concept`) | already shared: the merged-bridge generalization stack feeds perceived objects into the conversational composer (navigate-to-compose). Good model for "one organ, two consumers." |
| Working memory / dlPFC | **SPIKING + SHARED (co-resident on the merged bridge)** | `nav_conv_merged_bridge.py` : `_build_dlpfc_loop_population`, `cortex_ctx`/`dlpfc_wm` (NMDA); `unified_brain_bridge.py:_SharedDlpfcContext` | Wang-2002 NMDA attractor WM, co-resident on the one bridge, serves dialogue planning; the nav goal-WM (`--enable-dlpfc-wm`) is the same machinery on the nav side. Already consolidated. |
| Eligibility traces / three-factor plasticity | **SPIKING (on-bridge arrays), gated by the scalar DA** | `sim/kernels.py` : `fused_eligibility_trace_decay`; bridge reward-modulation block | `cp_eligibility_trace` is TD(λ) accumulating traces in all but name (catalog C.29 = implemented). The gate is the NM `dopamine` scalar. |

### 1c. Conversation residual host glue (mostly done — do NOT relitigate the FHRR bind)

| Computation | Class | File : function | Note |
|---|---|---|---|
| Whole who/what turn (parse/store/recall/abstain/negate/clauses/multi-hop/multi-turn/generate) | **SPIKING, ONE persistent co-resident bridge** | `research/runners/one_brain_composer.py` : `OneBrainComposer` (parser slice + RF registers + complex-synapse store) | production default at 320 scale; no host round-trips between ops. The real "one brain" for conversation. |
| The bind/unbind/bundle operation (FHRR resonate-and-fire) | **SPIKING (settled idealization — do NOT relitigate)** | `rf_phasor_composer.py`; `sim/bridge.py` `RESONATE_AND_FIRE` + complex synapse | the exact-inverse VSA algebra is a *principled idealization* (Eliasmith SPA), validated; the learned-bind corner is closed (a fixed self-inverse multiplicative primitive + learned codes/fillers, CYCLE 196). Not a target of this roadmap. |
| Cleanup (nearest-concept) | **HOST-SHORTCUT by default; SPIKING opt-in** | `rf_phasor_composer.py:258-263` `_cleanup` (`np.argmax(sims)` default) vs `_spiking_cleanup` (`enable_spiking_cleanup=True`, NEF matched filter + Izhikevich WTA, validated == numpy) | the spiking NEF cleanup is validated multi-seed but is **opt-in**; the production default is the numpy argmax (fast path). |
| Final word-order / readout / lexical POS lookups | **HOST-SHORTCUT (mostly biologized opt-ins)** | `rf_phasor_composer` `render_fact` `order_fn` (spiking serial-order generator, opt-in `enable_neural_render`); `attributed_parser.py`/multi-frame parser POS roles are *learned conjunctions* (spiking) | sentence word-order is neural when `enable_neural_render`; the attributed/multi-frame parsers learn role assignment in spikes. Residual host = the token strings + zipping words to spike-read roles (legitimate I/O). |

### 1d. Consolidation state — what is on the ONE bridge vs separate

| On the merged "one brain" (`nav_conv_merged_bridge.py` / `OneBrainComposer`) | Separate bridge / numpy module |
|---|---|
| Nav BG action-selection cascade (actor: `cortex_X→str_D1/D2_X→gpi_X→thal_X→motor_X`) | The spiking **reward/value/DA limbic core** (`snc`, `reward_us`, `striosome_value`, spiking SC) — **only in standalone `g11_bg_runner` behind flags** |
| Conversational parser (`parse_conj`/`parse_role`), dlPFC dialogue loop (`cortex_ctx`/`dlpfc_wm`) | `sim/td_value_critic.py` (TD bootstrap critic — numpy, never touches a bridge) |
| RF composer registers + persistent complex-synapse fact store (`co_resident_rf`) | `sim/predictive_coding.py` (Rao-Ballard predictor — numpy, "never touches a bridge") |
| Perception (`cortex_it`) + generalization stack (`gen_perception`→`gen_concept`→`gen_fact`) | The N5/N9 spiking SC + critic + place self-org regions (in `g11_bg_runner` only) |
| Cross-region interaction: language→action (`spoken_instruction_nav`), perception→memory (`navigate_to_see_then_answer`), compose-perceived (`navigate_to_compose_then_answer`) | — |

**The headline consolidation fact:** the merged bridge has the **actor** (BG cascade) and the **cortex** (parser/dlPFC/composer/perception) co-resident, but **no limbic core** (no reward, no value, no dopamine). The spiking limbic organs exist and are validated — they just live in a *different* runner. Lifting them onto the merge is the single highest-leverage consolidation step.

---

## 2. Audit headline (how much is spiking vs host vs separate)

- **Conversation half: ~fully spiking + consolidated.** The whole who/what turn runs on one persistent bridge (`OneBrainComposer`, 320-scale production default). Residual host = the *opt-in* numpy cleanup default and token I/O; the FHRR bind is a settled idealization (excluded by scope).
- **Navigation half: spiking at the MECHANISM level, but host in the DEPLOYED DEFAULT and SEPARATE from the one brain.** Every nav cognition has a validated spiking organ (N1 SC, N5 reward, N9 SNc+critic, N6 accumulate-commit), but (a) the **default episode loop still uses host reward + host argmax**, (b) the **TD bootstrap critic is numpy-only**, and (c) **none of the spiking limbic organs are on the merged bridge**.
- **Shared systems: perception + dlPFC are shared; the reward/value/DA limbic core is NOT.** This is the gap that turns "two skills sharing a GPU" into "one animal sharing a self."

So the roadmap is dominated by **consolidation of already-built spiking organs** + **two genuine spike-ification builds** (the TD bootstrap critic; the fully-spiking WTA read-out), not by inventing new biology.

---

## 3. The prioritized ROADMAP (by leverage × cheapness × dependency)

Ordered so the **shared limbic core lands first** (serves both halves), then the residual host conversions, then the harder nav sensorimotor pieces. Each item: neural target → reusable machinery → effort → point-neuron-wall risk.

**#1 — Shared reward/value/dopamine limbic core ON THE MERGED BRIDGE.** *(Leverage: maximal — the one organ both halves need; Cheapness: high — reuse-by-import; Dependency: none.)*
- **Target:** lift the validated spiking SNc (`snc`) + reward burst (`reward_us`, PPN-like) + value critic (`striosome_value` with GABA_B/GIRK δ=r−V) + the `dopamine` neuromodulator (`from_region_firing_signed` over `snc`) onto `nav_conv_merged_bridge` as a **shared limbic slice**, co-resident with the BG actor + the conversational cortex. The DA broadcast then gates BOTH the nav actor's three-factor learning AND (future) conversational salience/value.
- **Reuse:** `build_bg_brain_regions(enable_spiking_snc=…, n_vs_place_context=…, …)` (the Stage-B kwargs already exist — the merge just calls it with defaults); `sim/neuromodulators.py` `from_reward`/`from_region_firing_signed`; the GABA_B `RegionPathway(receptor="gaba_b")` (owner-approved sim/ edit, already shipped); `sim/regions.py` BrainRegion/RegionPathway.
- **Effort:** moderate. Mostly wiring (pass the Stage-B kwargs through the merge builder; register the `dopamine` modulator on the merged cfg; route `reward_us` from a neural reward source). NO new sim/ edit expected (the GABA_B/coincidence edits are already in).
- **Point-neuron-wall risk:** LOW for the mechanism (already validated on point neurons). The known residual is *behavioral load-bearing* (the gridworld is orient-solvable), not a substrate wall.

**#2 — Replace the deployed-default host reward with the neural reward, on the one brain.** *(Leverage: high; Cheapness: high; Dependency: #1.)*
- **Target:** make the merged-bridge nav episode source `current_reward_signal` from `sc_rostral→reward_us` firing (N5), not the host Manhattan/sign formula at `g11_bg_runner.py:6901-6946`. The reward becomes a *synaptic* quantity the limbic core reads.
- **Reuse:** the N5 spiking SC approach reward (`--enable-spiking-sc-approach`), the RPE battery (`sc_n5_rpe_probe.py`) as the validation.
- **Effort:** low-moderate (route + the clean-reset read protocol). Anti-cheat: the lesion `sc_rostral→reward_us` must abolish the reward.
- **Wall risk:** LOW (mechanism validated). The honest negative if it *changes* nav behavior is itself the deliverable (the orient-solvable caveat).

**#3 — The TD-bootstrap critic ON THE SUBSTRATE (close Schultz signature (a), the cue-shift).** *(Leverage: high — the last canonical dopamine signature missing; Cheapness: moderate; Dependency: #1.)*
- **Target:** replace the host-numpy `sim/td_value_critic.py` with an on-bridge critic that bootstraps δ = r + γV(s′) − V(s) — a value population whose recurrent estimate of V(s′) feeds back so the SNc phasic burst *migrates* from the US onto the earliest predictive cue across trials. Realizes catalog C.28/C.30/C.31 + C.33 (a PPN/`reward_us` driver of the cue-evoked burst).
- **Reuse:** the `striosome_value` critic + GABA_B subtraction (already δ=r−V); the eligibility kernel (`fused_eligibility_trace_decay`, C.29); a `reward_us`/PPN cue-driver region; the Pavlovian probe harness (`snc_pavlovian_probe.py`).
- **Effort:** moderate-high (the bootstrap needs a V(s′) term fed back into the SNc input — a slow-NMDA/eligibility-over-time arm). The N5 TD attempt failed on a compound lag + a global GABA_B-tau collision, so this is a *real* increment, not a knob.
- **Wall risk:** MODERATE. The temporal bootstrap on point neurons may hit the same lag/SNR limits the N5 TD attempt did — a candidate place the **deferred dendritic substrate** (`feedback_dendritic_substrate_fair_game`) could earn its keep (a dendritic eligibility-over-time). Honest negative = "the point-neuron substrate produces δ=r−V (Rescorla-Wagner) but not the TD cue-shift; the cue-shift needs dendritic/multi-timescale machinery" — a clean substrate-boundary deliverable.

**#4 — Fully-spiking motor read-out (close the N6 host-argmax residual) on the one brain.** *(Leverage: medium; Cheapness: moderate; Dependency: none, but compose with #1.)*
- **Target:** make the merged-bridge action selection the **commit-burst threshold crossing** (`spiking_wta` + `enable_commit_burst`) the *primary* decision, retiring the host argmax (currently a tie-break / the deployed default). The merge builder already supports `enable_spiking_wta_readout`.
- **Reuse:** `sel_{X}` Wang-2002 accumulators + `commit_{X}` Lo-Wang burst pools (`build_bg_brain_regions(enable_spiking_wta_readout=True)`); the Cisek urgency/collapsing-bound + loser-only-reset machinery already in the runner.
- **Effort:** moderate (operating-point tuning — historically the fully-spiking WTA scored *worse* than thal-argmax on nav). 
- **Wall risk:** MODERATE — the prior fully-spiking WTA underperformed (motor-WTA 14.7 vs thal-argmax 2.3). The honest negative (a clean spiking decision that navigates worse than the host argmax) is a documented substrate finding; the commit-burst-as-primary with the urgency bound is the path to close the silent-commit fallback.

**#5 — Neural place/position code as the shared spatial map on the one brain.** *(Leverage: medium; Cheapness: moderate; Dependency: #1, #3 (the critic reads it).)*
- **Target:** make the self-organized spiking `place` code (`--neural-place-selforg`) the merged-bridge spatial representation feeding the critic V, retiring the dense host-Gaussian `vs_place_context` scaffold. The `(x,y)` enters only via the egocentric landmark *render* (a defensible body/sensory boundary).
- **Reuse:** `_run_critic_warmup` self-org; the deterministic-transpose-matvec edit (already shipped, default-off) for reproducible self-org.
- **Effort:** moderate; needs the determinism extended to the two restricted coincidence/GABA_B matvec sites (`bridge.py` ~5771/5812) for clean multi-seed.
- **Wall risk:** LOW-MODERATE (self-org place fields validated; the residual is reproducibility, not capability).

**#6 — Wire the shared DA/NM onto the conversational composer (limbic ↔ cortex closure).** *(Leverage: medium-high (true integration); Cheapness: LOW; Dependency: #1.)*
- **Target:** let the shared `dopamine`/salience modulate the conversational composer (e.g. salience-gated cleanup gain / familiarity-gate threshold) so the limbic core actually *reaches* the cortex on both halves — the deepest form of "one self."
- **Reuse:** the NM subsystem; the composer's real knobs are `_rf_lambda` (resonate decay) + the cleanup gain.
- **Effort:** HIGH and likely needs a **sim/ edit** — the composer's RF ops bypass `_run_one_simulation_step`, so the NM subsystem doesn't reach them (a new `_rf_lambda`/cleanup-gain route). This is the one item that probably touches protected code; defer behind #1-#5.
- **Wall risk:** N/A yet (architectural, not a point-neuron wall). This is the emergent-feature #3 (neuromodulation) flagged in the emergent-features scoping.

---

## 4. Cheap-first de-risk for ITEM #1 — the shared reward/value/dopamine limbic core

**Goal of the de-risk:** before any merge wiring or GPU build, falsify (cheaply, numpy/CPU) the load-bearing claim that **a spiking dopamine population, driven by a spiking reward population and a spiking value/critic population, computes a correct reward-prediction error δ = r − V in spikes** — i.e. the limbic core *works as an organ* independent of the gridworld. (The mechanism is already validated on point neurons in the nav runner; this de-risk re-validates it as a *standalone, co-residable* organ and pins the GO bar before lifting it onto the merge.)

### Mechanism under test (the minimal spiking actor-critic limbic core)
A minimal bridge slice (numpy/CPU `SIM_BACKEND=numpy` first; tiny, ~a few hundred neurons):
- `reward_us` — a small excitatory (PPN-like, catalog C.33) population that fires on a *neural* reward event (a cue/US pattern), → `snc`.
- `snc` — the spiking dopamine pool (tonic pacemaker; catalog C.16/C.20/C.22), DA broadcast = its windowed firing via `from_region_firing_signed`.
- `striosome_value` — a GABAergic MSN critic (catalog C.30 striosome=critic) that learns V over a state code and subtracts V onto `snc` through the GABA_B/GIRK conductance (catalog B.15 SNc-lacks-KCC2 → GABA_B is the strong subtraction; the edit is already shipped+approved).
- The DA `dopamine` modulator (`plasticity_rate scope=all`, `from_region_firing_signed` over `["snc"]`) gates a downstream actor's three-factor learning.

### What to measure — the Schultz RPE battery (the reward is the dependent variable)
Because a reward/DA signal is *defined by its teaching signal* (the project's own pivotal N5 lesson: a reward can't be validated by a task that doesn't need it), the de-risk measures the **canonical dopamine signatures with `r` and `V` sourced from neurons**, not behavior:

1. **Burst on an unpredicted US** — `snc` firing ≫ tonic when `reward_us` fires and V≈0.
2. **Graded RPE** — `snc` rate monotone in reward magnitude / proximity (the N5 probe got corr=−0.99 vs distance).
3. **Reward omission → SNc dip below baseline** — withhold the US when V>0 → `snc` rate < tonic (the signed-firing dip; catalog C.28 signature (c)).
4. **Predicted reward → no SNc response** — after the critic learns V at the predicted state, δ≈0 (the US burst shrinks as V cancels r; Rescorla-Wagner, catalog C.28 signature (b)).
5. **Lesion anti-cheat (decisive for a *reward*)** — cut `reward_us→snc` (or `sc_rostral→reward_us`) → the burst vanishes (the N5 probe got flat 39 Hz). This proves the RPE *is* the synaptic reward, not a re-hidden host scalar.
6. **Critic lesion (decisive for *value*)** — zero the GABA_B `striosome_value→snc` conductance → the value subtraction collapses (predicted == unpredicted), proving V is the synaptic GABA_B (the Stage-B probe's [LESION] gate).

### Quantitative GO bar (frozen, pre-registered)
- (1) burst/tonic ratio **≥ 3×** on the unpredicted US (the N5 probe got 251/48 ≈ 5×).
- (2) **corr(reward magnitude, `snc` rate) ≤ −0.8** (graded); sign as expected.
- (3) omission `snc` rate **< tonic** (a real dip, not just attenuation).
- (4) US-burst at the predicted state **shrinks ≥ 50%** vs the unpredicted-state burst after the critic trains (δ→0).
- (5) reward lesion → burst within **±15%** of tonic (vanishes).
- (6) critic (GABA_B) lesion → predicted/unpredicted gap collapses to **≤ 1.2×** (was e.g. 3.33×).
- **Multi-seed:** ≥ 5/6 seeds pass each gate (the standing 6-seed rule for variable effects; the lesion gates are mechanistic so 3 clean is conclusive there).

### What an honest NEGATIVE means (the deliverable)
- If the **graded δ or the omission dip fails on the point-neuron substrate** → the substrate boundary is "rate-coded point neurons cannot hold a graded subtractive value signal at the SNc without [dendritic plateau / slow-channel] machinery" — which is *itself* the scientific deliverable and points exactly at where the deferred dendritic substrate (`feedback_dendritic_substrate_fair_game`) earns its keep. (Note: Stage B already got the graded δ *with* the GABA_B + coincidence-plateau edits, so a from-scratch point-neuron version failing would precisely re-confirm that those dendritic/slow-channel edits are *necessary*, not optional.)
- If the **predicted-reward δ→0 (signature (b)) holds but the cue-shift (signature (a)) does not**, that is the expected, documented Rescorla-Wagner-vs-TD boundary → feeds directly into roadmap #3 (the TD bootstrap critic).

### Reusable machinery to name
- `sim/neuromodulators.py` — `from_reward` (phasic DA from `current_reward_signal`) and `from_region_firing_signed` (DA conc ← spiking SNc firing, with the sub-tonic dip); `ProductionRule`/`ModulatorTarget`/`NeuromodulatorConfig`; `compute_plasticity_rate_multiplier`.
- `research/runners/g11_bg_runner.py` — `build_bg_brain_regions(...)` (the actor cascade + the Stage-B `striosome_value`/`snc`/`reward_us` kwargs); the spiking-SNc Stage-A/B wiring; `sc_n5_rpe_probe.py` / `snc_pavlovian_probe.py` (the RPE battery harnesses to reuse verbatim).
- `sim/regions.py` — `BrainRegion`/`RegionPathway` (incl. `receptor="gaba_b"`, `coincidence_detector`); the GABA_B/GIRK + coincidence-plateau sim/ edits are **already shipped + owner-approved**.
- `research/runners/nav_conv_merged_bridge.py` — `build_merged_nav_conv_bridge` (the merge target; the masked-region co-residence pattern for adding the limbic slice without perturbing the existing nav/conv slices).

### Anti-cheat controls (must all hold)
- **The reward/value/DA must be NEURAL** — `r` = `reward_us` *firing* (not a host scalar written to `current_reward_signal`); `V` = `striosome_value` *firing* subtracted via GABA_B; δ = `snc` *firing*. A passing run must show the lesions (5) + (6) collapse the signal (the host-scalar version would be lesion-insensitive).
- **Lesion/permuted controls that collapse** — reward-pathway lesion (5), critic-conductance lesion (6); a permuted state→value mapping should destroy signature (4) (predicted δ→0).
- **Held-out generalization** (for the value critic) — V must predict reward at states *not* seen during a held-out probe (the critic generalizes, not memorizes) where the task admits it.
- **Cognition is neural; host only for env + body** — the only host residuals permitted are the cue/US *pattern* presentation (environment) and (if any) the action on a motor pool (body). The δ, the value, the dopamine are all spikes.
- **Honest-negative framing** — a δ=r−V that holds but no TD cue-shift, or a graded-δ that needs the dendritic edits, is reported as the substrate boundary, not buried.

---

## 5. Catalog + literature anchors (cited)

- **C.28** TD error = phasic DA (δ = r + γV(s′) − V(s)); sim status *partial — gap is measurable*; the cue-shift signature requires a critic population. The single most-cited gap.
- **C.30** Actor-critic — separable policy + value with shared δ; sim status *partial — actor implemented, critic missing* (the BG cascade is actor-only / two-actor D1-D2; **no separable V(s)**). Anatomical map: SNc=δ, striosome=critic V, matrix=actor — exactly the limbic-core structure proposed in #1/#3.
- **C.31** Bootstrapping vs Monte Carlo — why phasic DA *must* bootstrap (single-trial cue-shift); the project is "windowed Monte Carlo" (eligibility window), not bootstrapping.
- **C.32** Two-component DA (detection/salience + utility-RPE) — the salience component is the conversational hook (a shared salience signal).
- **C.33** PPN → SNc reward driver — the `reward_us` population; PPN inactivation degrades the *cue-evoked* DA burst (the anti-cheat for the cue-driver).
- **C.22 / O.02** Schultz RPE — the canonical three signatures = the GO battery.
- **B.15** SNc DA neurons lack KCC2 → GABA_B/GIRK is the strong value subtraction (the basis for the already-shipped GABA_B edit).
- **C.16 / C.20** VTA/SNc anatomy + tonic-phasic firing — the `snc` pacemaker + phasic burst.

(Literature note: the spiking actor-critic / spiking RPE literature — Frémaux-Sprekeler-Gerstner reward-modulated spiking actor-critic; Potjans-Diesmann spiking TD; Nakano/Kato spiking BG actor-critic — corroborates the catalog mapping; the project's own N9 work is a validated instance. A WebSearch/bio-research pass can deepen this but the catalog C.28-C.33 cluster is sufficient to anchor the build.)

---

## 6. Honest scope / non-claims

- I did **not** relitigate the FHRR bind (settled idealization) or the generalizing-cortex question (closed-positive via PPMI stream cortex). Those are out of this roadmap by the owner's prior closures.
- The navigation spiking organs are validated at the **mechanism** level; their **behavioral load-bearing** in the orient-solvable gridworld is a known, documented limitation (not hidden) — a harder reward-load-bearing task is a separate, larger arc (relevant to #3).
- The merged bridge's "no limbic core" is the load-bearing consolidation finding; everything in §3 #1-#2 is *lifting validated organs onto the merge*, which is high-confidence; #3-#6 contain the genuine new builds + the point-neuron-wall risks.

---

## 7. EXACT NEXT

Build the **#1 cheap-first limbic-core de-risk** (numpy/CPU first): a minimal `reward_us → snc ← striosome_value(GABA_B)` slice + the `dopamine` signed-firing modulator, run the Schultz RPE battery (§4), pre-register the frozen GO bar, and report PASS or the honest substrate-boundary NEGATIVE. On PASS, lift the slice onto `build_merged_nav_conv_bridge` as the shared limbic system (pass the Stage-B kwargs through the merge builder + register the `dopamine` modulator on the merged cfg). Reuse-by-import; no new sim/ edit expected (the GABA_B/coincidence edits are already shipped + approved).
