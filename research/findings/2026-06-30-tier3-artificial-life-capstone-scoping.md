# Tier-3 artificial-life CAPSTONE — deep-research scoping (READ-ONLY): the persistent living agent

**Date:** 2026-06-30 (CYCLE 731, autonomous loop; owner-directed Tier-3 open)
**Type:** Deep-research + reference-catalog scoping. **READ-ONLY — NO code / `sim/` / GPU edit.** This doc is the
standing "deep-research FIRST at a new direction" opening move for the project's north-star
([[project_actual_goal_artificial_life_brain_analogue]], [[project_post_conversational_roadmap_tiers]] Tier 3).
**Predecessors (do NOT re-derive — this doc SUPERSEDES their framing by isolating the genuine residual):**
`2026-06-17-artificial-life-frontier-scoping.md`, `2026-06-20-tier3-artificial-life-capstone-deep-research.md`,
`2026-06-23-artificial-life-longitudinal-test-scoping.md`, and the **executed** de-risks
`2026-06-20-tier3-persistent-living-loop-derisk.md` (GO 6/6 rate-proxy) +
`2026-06-20-tier3-spiking-living-loop-derisk.md` (spiking drive CONVERTS; survival policy = honest dendrite NEGATIVE)
+ `2026-06-24-week1-develop-loop-console-capstone.md` (a brain develops over a simulated week).

---

## 0. TOP-LINE — is it largely done? YES, in PIECES. The genuine residual is the SYNTHESIS.

The north star is a **PERSISTENT LIVING AGENT**: perceive → remember → reason → act → converse, CONTINUOUSLY, on
ONE brain, accumulating open-ended experience over time. Applying the SURPASS practice (Tiers 1+2 repeatedly
turned out largely-done; isolate + quantify the *genuine* residual, don't accept a vague "it's scripted"):

**Every ingredient already exists and is separately multi-seed-validated on the merged one brain.** What does NOT
yet exist is **one continuous outer loop that JOINS them** — a body that perceives+composes, a self-generated drive
that persists across resets, and conversation about what it lived, all in ONE `live()`-style loop where the
experience is **agent-chosen (open-ended), not scripted**. Concretely, today there are THREE separate "capstone-ish"
artifacts, each missing two of the three living-agent axes:

| artifact | continuous life? | a BODY that perceives/acts? | converses about lived experience? | experience source |
|---|---|---|---|---|
| **`develop_gpu`** (the day-loop) | **days, resumable** ✓ | ✗ (listen-only; no gridworld/motor) | ✓ (MultiTurnAgent on learned codes) | **SCRIPTED curriculum** (`curriculum.day_stream`) |
| **`persistent_living_loop`** (survival) | **continuous `live()`** ✓ | ✓ (drive→act→eat, spiking drive) | ✗ (no composer/parser; moat untouched by construction) | drive-generated (survival) |
| **`navigate_to_compose_then_answer`** | ✗ (bounded episode) | ✓ (navigate→perceive→ground→compose) | ✓ (who/what + moat) | **SCRIPTED route + layout** (fixed waypoints, `max_steps=64`) |

The residual is the **intersection cell that no artifact fills**: *continuous life ∧ a perceiving/composing body ∧
conversation about what it lived ∧ open-ended (self-chosen) experience ∧ persistence.* Because all four seams
already exist as **additive, default-off, byte-preserving** hooks (§2), the FIRST persistent-living-agent slice is a
**runner-only JOIN — no `sim/` edit** (the executed living-loop de-risks both needed none), decoupled from the one
genuinely-deferred wall (the learned spatial policy = Tier-4 dendrite, §5). **This is largely-done at the
component level; the residual is a small assembly + the honest "open-ended vs scripted" upgrade.**

---

## 1. DIAGNOSIS — what EXISTS toward the capstone (cite file:line)

### 1a. The brain + body (the merged one brain) — EXISTS, spiking, default action-decision spiking
- **`build_merged_nav_conv_bridge(...)`** — `research/runners/nav_conv_merged_bridge.py:542`. ONE `SimulationBridge`
  (brain-region framework, Izhikevich, dt=1) holding the nav BG-cascade (**the body**) + the conversational parser +
  the dlPFC dialogue planner as disjoint neuron-index slices, wired by one init-time injection. All co-resident
  organs are appended LAST so index bases stay byte-stable. **The Tier-3-relevant additive, default-off hooks (each
  byte-preserving):** `co_resident_perception` (`:545`, the `cortex_it` ventral "what" stream),
  `co_resident_generalization` (`:548`, the Gabor/V1→`gen_perception`→NMDA `gen_concept` structured-perception
  stack + trained-then-frozen convergence for spikes-only grounding), `co_resident_composer`/`co_resident_rf`
  (`:543`, the `rf` FHRR composer slice), **`co_resident_drive`** (`:551`, the 2-pool SPIKING hunger organ —
  §1e), `co_resident_limbic`/`co_resident_nav_critic` (`:550`/`:553`, the reward/value/DA core).
- **`MergedNavConvAgent`** — `nav_conv_merged_bridge.py:1617`. The `BrainConversationalAgent`-compatible surface.
  Methods: `hear`/`what_does`/`who_does`/`is_it_true`/`describe`/`elaborate` (conversation), `command_move`
  (Route A language→action, `:2055`), **`perceive_and_ground(obj_word)`** (Route B perception→memory, `:2086` —
  reads the percept's LIVE spiking response, writes the grounded phasor into the composer codebook),
  `_da_confidence_gate` (`:2129`, the #6 DA-salience moat-tighten, moat-safe). **No `navigate()`** here — navigation
  is `command_move` (one decision) or the external episode runner (§1b).
- **`run_moving_goal_episode(...)`** — `research/runners/g11_bg_runner.py:3332`. The validated moving-goal RL nav
  episode; builds the merged bridge via `extra_regions`/`conv_extra_*`. `readout_source="spiking_wta"` is the LIBRARY
  DEFAULT (`:4212`; Wang-2002 accumulator + Lo-Wang commit burst, host argmax retired; 6-seed 1.16× host,
  `2026-06-19-spiking-decision-default-on-GO.md`). **`homeostatic_hook=None`** (`:3434`) is the Tier-3 seam: called
  per trial AFTER natural reward — `gated_reward, new_goal = hook(reward, x, y, gx, gy, step, dist_after)` — so a
  drive can GATE reward (`reward *= hunger`) and RELOCATE food on "eat" (`dist_after==0`), reusing the BG-cascade +
  value-critic learner with NO fork. **STATUS: EPISODIC** (per-trial loop), spiking-by-default.

### 1b. The develop-loop (a life over DAYS — but listen-only, scripted curriculum) — EXISTS + robust
- **`develop_gpu(lineage, curriculum, n_days, ...)`** — `research/runners/_longitudinal_develop_loop_gpu.py:287`.
  Per simulated day (`:372`): **WAKE** = real stream-cortex code-learning (`cortex.hear_day` `:393`) → **CONVERSE** =
  build a `MultiTurnAgent` on the learned grounded codes + teach the day's facts (`:404-419`) → **SLEEP** =
  `consolidate` (self-replay + retention re-test, `:426`) → **METRICS** (`_measure` `:431`) → **GROWTH** =
  `TierPromoter` (`:438`) → **PERSIST** = `_save_state` (`:453`) → optional per-day BUNDLE (`:460`). Pause seam
  `should_continue` (`:376`, day-boundary, zero work lost).
- **`StreamCortex`** (`:145`) is a PERSISTENT GPU cortex that LEARNS concept codes by hearing the TinyStories corpus
  window-by-window (online rate-Hebbian co-occurrence); the co-occurrence ACCUMULATES across days on the SAME bridge.
- **`GPUGradedCurriculum`** (`:134`, `_GPU_SYLLABUS` `:84`) / `CorpusGradedCurriculum`
  (`_corpus_develop_curriculum.py:73`) supply the day's `new_concepts` + `facts` from a **preset schedule**.
- **Robustness harness:** `develop_run.py` (the 3-day self-driven launcher) + `develop_loop_supervisor.py` (24/7,
  crash-proof, pausable via a `bridges/PAUSE` sentinel, fsync'd atomic persist) + `scripts/develop.ps1`.
- **THE GAP (verified by grep — 0 matches for `navigate`/`cortex_it`/`MergedNav`/`run_moving_goal`):** it has **NO
  body, NO gridworld, NO perception, NO action**. It is a **listen-and-learn-vocab-then-converse** life. And its
  "experience" is **an authored curriculum**, not autonomously generated. **STATUS: PERSISTENT (days, resumable),
  spiking cortex+composer; NOT a perceive-act life; experience SCRIPTED.**

### 1c. Navigate-to-compose-then-answer (perceive→compose→converse — but a SCRIPTED episode) — EXISTS, 6-seed GO
- **`navigate_to_compose_then_answer.py`** (`2026-06-16-navigate-to-compose-then-answer.md`, **6-seed GPU GO**). Flow
  (`run_seed` `:710`): build merged bridge (nav body + `cortex_it` + gen stack + co-resident composer) → **navigate**
  a fixed route (`run_compose_episode` `:476`, BG cascade selects each move toward `route_waypoints`) →
  **perceive+ground** on arrival (`_perceive_and_ground` `:405`, spikes-only `gen_spikes` grounding) → **compose**
  held-out facts on the `rf` slice → **answer** who/what + ABSTAIN. Four anti-cheats (lesion / held-out≠recall /
  provenance+co-residence / no-confab moat) + an iso-perception control.
- **WHY SCRIPTED (not open-ended/persistent):** hard-coded object layout (`default_object_layout(seed)`), fixed
  `start_pos`+`route_waypoints` (`:717-719`), bounded `max_steps=64`, and a fixed setup→compose→answer→verdict
  sequence PER SEED. **No continuous life, no drive, no persistence.** It proves the CAPABILITY (compose what you
  perceive, one brain); it is not a LIFE. **STATUS: EPISODIC/SCRIPTED, spiking.**

### 1d. The motivational core (the "why act") — VALIDATED across all faces
- **`TwoPoolDrive`** — `research/runners/_homeostatic_drive_rl_cheap_first_probe.py:59`. A 2-pool push-pull hunger
  drive (rate proxy of AgRP↔POMC reciprocal inhibition); `update(deficit)` returns `agrp − pomc`; `lesion`/`yoke`
  are the anti-cheats. **Validated GO (≥3 seeds, rate-proxy):** the agent LEARNS a policy from an INTRINSIC
  drive-reduction reward `r = drive_before − drive_after` (NO host distance term); corr(deficit,drive)≥+0.9; hungry
  approach ≥2× sated; time-to-resource beats lesion+yoke. `2026-06-17-homeostatic-drive-rl-cheap-first-GO.md`,
  `-sustained-agency-GO.md` (alive-over-time, never crashes vs lesion crashes).
- **Spiking realization already wired:** `co_resident_drive` (`nav_conv_merged_bridge.py:551,879`;
  `conv_extra_regions_pathways` `:1258,1295`) appends `drive_agrp`/`drive_pomc` (hypothalamic AgRP/POMC, ZERO
  out-edges → nav-inert; per-region `enable_homeostasis` mask = the merged-config operating-point fix). Validated
  seed-42 GPU: corr(deficit, AgRP firing)=**0.995**, load-bearing, **moat byte-frozen in vivo**
  (`2026-06-20-tier3-spiking-living-loop-derisk.md`).

### 1e. The persistent living loop (the missing `live()` outer loop) — the CLOSEST prototype, GO 6/6 rate-proxy
- **`persistent_living_loop_derisk.py`** (`2026-06-20-tier3-persistent-living-loop-derisk.md`, **GO 6/6, CPU
  rate-proxy**). This is THE continuous-life prototype:
  - `LivingState` (`:82`) = body energy `E` + the `TwoPoolDrive` pools + learned policy + position + RNG — the
    persisted internal "self over time".
  - **`live(state, n_steps)`** (`:140`) = THE CONTINUOUS LOOP that both `develop_gpu` (day-batched) and
    `run_moving_goal_episode` (episodic) lack: energy depletes each step, the drive tracks the deficit and biases
    action, eating drops the deficit → intrinsic reward, state mutates IN PLACE (no per-episode reset).
  - `_save_state`/`_load_state` (`:209`/`:224`) persist `LivingState` via `BridgeLineage` (a JSON `save_fn`). All 4
    anti-cheats collapse (drive-lesion starves, yoked-random starves, no-persistence cold-start re-warms,
    reward-provenance = no distance term).
- **STATUS: PERSISTENT (continuous + resumable), rate-proxy (numpy). It has NO perception/memory/conversation** — its
  moat is untouched by construction (it is a survival loop, not a conversing one). **This is the skeleton the
  capstone extends** — the `live()` shape is exactly right; what it lacks is the perceiving/composing/conversing
  body of §1a/§1c.

### 1f. Persistence (the "self over time" substrate) — EXISTS, atomic
- **`BridgeLineage`** — `sim/lineage.py:111`. Git-like history under `bridges/lineage/<name>/`: `save(...)` (`:190`,
  **ATOMIC** `.new`+`os.replace`, custom `save_fn`), `load` (`:238`), `fork` (`:343`), `rollback_to` (`:306`),
  snapshots/prune/growth-log. Backend-agnostic — every loop already persists via a JSON `save_fn`.
- **FLAGGED GAP (both develop + living loops note it):** the persisted state is **JSON `DevelopState`/`LivingState`,
  NOT the raw `cp_connections` synaptic tensor.** On resume, `develop_gpu` re-hears the cumulative vocab to
  re-instate the learned codes (a cheap stand-in). True synaptic persistence (`save_checkpoint` of the merged bridge)
  is a follow-on — NOT on the cheap-first path (the living loop's `LivingState` resume is exact for the
  survival/drive state that matters for the FIRST slice).

---

## 2. THE GENUINE RESIDUAL — isolated + quantified (the SURPASS move)

**Claim: the FIRST persistent-living-agent slice is a runner-only JOIN of already-validated, additive seams — NO
`sim/` edit, decoupled from the one deferred wall.** The seams, all default-off + byte-preserving:

1. the **`live()` continuous outer loop** (the skeleton — `persistent_living_loop_derisk.py:140`);
2. a **perceiving/composing body co-resident** on the same bridge (`co_resident_perception` + `co_resident_composer`
   + `perceive_and_ground` — `nav_conv_merged_bridge.py:545/543/2086`);
3. the **self-generated spiking drive** that makes experience self-chosen (`co_resident_drive` +
   `homeostatic_hook` — `:551`, `g11_bg_runner.py:3434`);
4. **persistence** of the internal state across a reset (`BridgeLineage` — `sim/lineage.py:190`).

**What is genuinely NOT-done (the precise residual — small):**
- **(R-a) The JOIN itself:** no runner today runs a continuous loop where the SAME persistent brain (i) is driven by
  its own interoceptive state to act, (ii) PERCEIVES + GROUNDS objects it encounters *during* that lived behaviour,
  (iii) COMPOSES/stores facts about them, (iv) can be QUERIED about what it lived (with the moat intact), and (v)
  PERSISTS across a reset and resumes. Each of (i)-(v) exists; the loop that chains them does not.
- **(R-b) "Open-ended vs scripted" — the honest upgrade:** in `navigate_to_compose_then_answer` the *what-it-perceives*
  is a fixed route + fixed layout; in `develop_gpu` the *what-it-learns* is a preset curriculum. The genuine
  living-agent property is that **which objects the agent encounters (and therefore what it can later talk about) is
  a consequence of the agent's OWN behaviour** (drive-biased exploration over a world it doesn't fully control), not
  an author's script. This is the discriminator that separates "a life" from "a scripted demo replayed."
- **(R-c) NOT on the critical path — the two deferred pieces** (do NOT scope as blockers for the first slice):
  - the **learned spatial POLICY from intrinsic reward** = the Tier-4 dendrite wall (3rd rigorous NEGATIVE
    2026-06-19; the spiking-living-loop de-risk confirmed the fixed BG-cascade can't *learn* to keep itself alive).
    The first slice is demonstrated on **survival/foraging where the reward is load-bearing even with a simple
    policy** (exactly as the rate-proxy GO 6/6 showed), and on **encounter-driven perception** (arriving-at-an-object
    is enough — no converged optimal path needed).
  - **true synaptic (`cp_connections`) persistence** of the merged bridge (§1f) — the `LivingState`/`DevelopState`
    JSON resume is sufficient for the first slice; raw-tensor persistence is a follow-on.

**Quantified:** the residual is **one new runner (~a `live_and_converse()` loop) reusing 4 existing hooks**, sized
like the `persistent_living_loop_derisk.py` prototype (~400 LOC) but on the merged spiking bridge instead of the
rate proxy — i.e. the spiking-living-loop de-risk's runner (`_tier3_spiking_living_loop_derisk.py`) **plus** the
`co_resident_perception`+`co_resident_composer` slices and a `perceive_and_ground` call on each encounter. **It is
largely done.** The "big new tier" framing overstates it: Tiers 1+2 were closed by exactly this pattern (the pieces
existed; the JOIN + a default-flip closed them).

---

## 3. THE BIOLOGY (catalog-FIRST) — what a persistent living agent needs

Read from `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (clusters O = motivation/reward, N =
sleep/arousal/circadian). **Nearly every entry below is sim-status "missing"** — expected, since drives / rhythms /
arousal state-machinery are the always-on scaffolding the current *task-driven* flagship lacks. The two implemented
anchors are the **BG action-selection loop (cluster A)** and partial **sleep-replay infrastructure**.

### The motivational core (already de-risked; the "why act") — cluster O
- **O.05** Hypothalamic Homeostatic Architecture (`:4803`, ⭐) — sensor→integrator→effector loops with emergent
  settling-points. Kandel 6e Ch 41.
- **O.06** Arcuate POMC/AgRP/MC4R feeding loop (`:4815`, ⭐) — the antagonistic AgRP(hunger)/POMC(satiety) tug-of-war
  = the `TwoPoolDrive`. Kandel Ch 41.
- **O.10** Incentive Motivation (`:4863`) — deprivation *amplifies the reward value* of goal stimuli (Berridge/Toates).
- **O.11** Drive Reduction Theory (`:4875`, ⭐) — a deficiency state is itself AVERSIVE; consuming relieves it →
  hunger as a negative reinforcer = the intrinsic drive-reduction reward. Kandel Ch 41 (Sternson CPP).
- **O.08** SFO/OVLT thirst circuit (`:4839`) — a SECOND drive → dynamic goal-switching (the noted Phase-3.1 add).

### Goal arbitration / foraging / decision over EXTENDED time — cluster O (the CONTINUING-TASK regime)
- **O.21** Average-reward formulation, undiscounted continuing tasks — **the exact RL regime for a persistent
  (non-episodic) agent: relative reward `R − R̄`, not discounted episodic reward.** Matches the project's own
  `--moving-goal` + `--adaptive-da` EMA. Sutton & Barto §11.3; Schwartz 1993. **← the single most capstone-relevant
  formal entry.**
- **O.18/O.20/O.22/O.23** actor-critic on the BG (striosome=critic `V(s)`, matrix=actor), Generalized Policy
  Iteration, striatal action-value coding, the three reward functions. The catalog flags **adding an explicit
  striosome `V(s)` critic** as "the single highest-leverage architectural upgrade" for long-horizon credit — and the
  merged bridge's `co_resident_nav_critic` already builds one (`striosome_value`).
- **O.19** vmPFC/OFC subjective value; **C.34** DA codes economic utility; **C.22** DA RPE δ. Kandel Ch 43/56.

### Intrinsic motivation / curiosity / novelty-seeking (for OPEN-ENDED experience — R-b) — distributed
- **D.23** Misplace system — hippocampal novelty detection driving EXPLORATION (`:1059`). O'Keefe-Nadel CA1
  "displace/misplace" units fire on expected-absent / novel stimuli → fornix→septum→brainstem drives investigative
  exploration, *reciprocally updating the map* via one-trial LTP. **← the biological engine for "the agent chooses
  what to experience" (novelty-seeking = open-ended experience).** Kandel Ch 54.
- **D.15 direct** EC→CA1 temporoammonic match/mismatch (`~1129`) — novelty/mismatch detection.
- **C.32/C.23/C.24** two-component DA (detection/salience precedes value-RPE) + heterogeneous DA subpopulations
  (novelty/salience cells) — the DA-novelty bonus substrate. **C.05** NE tonic-mode → exploration (Aston-Jones
  inverted-U). (The catalog has NO single "curiosity" entry; it is distributed across these.) **Note:** the
  neuromodulator subsystem has a `from_novelty` rule **RESERVED but UNIMPLEMENTED** (emits 0) — relevant only if a
  later slice pursues an explicit curiosity drive.

### Behavioral-state / arousal / circadian (the "always-on" scaffold — later slices) — cluster N
- **N.10** SCN master clock (`~24 h`), **N.08** adenosine sleep pressure (Process-S), **N.09** two-process model,
  **N.11** orexin state-stabilization, **N.01/N.02/N.03** ascending arousal + VLPO + wake-sleep flip-flop. These are
  the day/night + wake/sleep machinery for a truly open-ended life (a LATER slice — the first slice uses a simple
  "wake=live, sleep=consolidate" alternation, not the full flip-flop).

### Continual / lifelong experience integration (the develop-loop's biology) — clusters D, N, J
- **D.01** episodic encode/store/retrieve/consolidation; **D.14** engram cells (Tonegawa — already an API);
  **D.19/N.07** sharp-wave ripples (replay); **N.12** sleep-dependent consolidation; **J.27** reconsolidation;
  **J.34** schemas/gist. **N.17 (⭐ for R-b)** awake replay at behavioral PAUSES (Foster & Wilson 2006) — **a
  natural LIVED (not scripted) consolidation trigger: replay fires when the agent rests / reaches a goal**, giving a
  second trigger site beyond scheduled NREM. This is the biology for making consolidation *lived* (the develop-loop's
  scripted SLEEP phase → an event-triggered one). Kandel Ch 44/54.

**Cross-cutting catalog steer (author-flagged, `~:4969`/`~:4994`):** add hunger/thirst as slow internal state
variables (O.05/06/10/11) that *modulate per-stimulus reward weights* via the existing neuromodulator subsystem
(`excitability_drive` + `synaptic_gain`) → "dynamic goal-switching fixed external rewards cannot" — exactly the
living-agent behaviour, reusing infrastructure that exists (and already realized by `co_resident_drive`).

---

## 4. RANKED cheap-first FIRST-SLICE options (each: mechanism · reusable machinery · behavioral signature ·
anti-cheat). Scope the CHEAPEST slice that is genuinely PERSISTENT + OPEN-ENDED, NOT the whole capstone.

### ★ OPTION 1 (RECOMMENDED) — "live-and-remember": a continuous loop where a drive-biased agent PERCEIVES objects it encounters during its own lived behaviour, GROUNDS+STORES them, and can be QUERIED about what it lived — persisting across a reset.
- **Mechanism:** extend the `live()` skeleton (§1e) onto the merged spiking bridge. Each step: the interoceptive
  drive biases exploration of a small world containing OBJECTS (food + a few landmark objects); when the agent
  arrives at an object cell, the environment renders it into `gen_perception`/`cortex_it`, the agent
  `perceive_and_ground`s it into the co-resident composer, and (on eating / at a rest pause, per **N.17**) stores a
  fact about what it just encountered. The body/drive/composer state lives on ONE persistent object; a reset →
  reload resumes the SAME life; afterward the owner can ask "what did you see near the food?" and the agent answers
  from its lived, grounded memory (or ABSTAINS — the moat).
- **Reusable machinery:** `persistent_living_loop_derisk.py:140` (the `live()` loop) + `_tier3_spiking_living_loop_derisk.py`
  (the spiking-drive-on-merged-bridge version) + `co_resident_perception`/`co_resident_generalization` +
  `co_resident_composer`/`co_resident_rf` + `MergedNavConvAgent.perceive_and_ground` (`:2086`) + `homeostatic_hook`
  (`g11_bg_runner.py:3434`) + `BridgeLineage` (`sim/lineage.py:190`). **NO `sim/` edit** (all additive default-off
  seams; both executed living-loop de-risks needed none).
- **Behavioral signature:** over a continuous life, the agent (a) survives (drive keeps energy in-band, never
  crashes), (b) accumulates grounded facts about the objects IT chose to visit (fact-count grows), (c) answers
  who/what queries about those objects correctly AND abstains on never-encountered objects, (d) resumes the exact
  life + memory after a reset.
- **Anti-cheats (all must collapse; the validated-by-function bar):** **drive-lesion** → starves + encounters/stores
  nothing (survival AND experience are the drive's doing); **yoked-random drive** → starves (coupling is
  load-bearing); **grounding-lesion** (sever `gen_perception→gen_concept`) → the stored facts collapse to chance
  (memory rides the live percept, not a structural bias); **no-persistence cold-start** → visibly re-warms + loses
  its lived memory (persistence is load-bearing); **reward-provenance** (`r` = drive reduction from `cp_firing_states`,
  no distance term); **NO-CONFAB MOAT byte-unchanged** — every unstored / never-encountered query returns None
  (a breach = HARD STOP; the composer slice is array-disjoint from the nav read-out so the moat holds by
  construction, as the spiking-living-loop smoke already showed). **6 seeds** for the survival + generalization claims.
- **Why cheapest-first + genuinely persistent+open-ended:** it JOINS the exact validated pieces, adds NO new
  mechanism class, and is open-ended in the load-bearing sense (**which objects get grounded is a consequence of the
  drive-biased behaviour + world layout, not a fixed script** — the R-b discriminator), while sidestepping the
  deferred spatial-policy wall (encounter-driven grounding needs arrival, not an optimal path; survival is
  load-bearing with a simple policy). It converts the merged one brain from "a battery of demos" into **the first
  life that perceives, remembers, and can be talked to about what it lived.**

### OPTION 2 — "develop-with-a-body": give the day-loop (`develop_gpu`) a perceiving body so the daily WAKE experience is LIVED, not a curriculum.
- **Mechanism:** replace `develop_gpu`'s scripted `curriculum.day_stream` WAKE with a live foraging episode on the
  merged bridge (Option-1 loop as the "day"), so the day's new facts come from what the brain PERCEIVED that day;
  keep the validated SLEEP(replay+retention)/GROWTH/PERSIST scaffold and the console per-day bundles.
- **Reusable machinery:** all of `_longitudinal_develop_loop_gpu.py` (WAKE/SLEEP/GROW/PERSIST + per-day bundles +
  `develop_run.py`/`develop_loop_supervisor.py` 24/7 harness) + Option-1's live-perceive loop as the WAKE.
- **Behavioral signature:** watch a brain develop over a simulated WEEK where each day's knowledge is what it LIVED
  (vocab/facts grow from experience), no forgetting (retention), moat 0-FA daily, loadable per-day bundles.
- **Anti-cheat:** Option-1's set + the develop-loop's (frozen-brain via `set_plasticity_gate=0`, retention/no-replay
  arm, permuted-curriculum, persistence cold-start-drops). **Cost/risk:** MEDIUM — it is Option-1 *plus* the
  develop-loop assembly (two moving parts); best as the SECOND slice after Option-1 proves the live-perceive-converse
  loop. (Do it AFTER Option 1, not instead.)

### OPTION 3 — "cross-modal one animal": the hunger drive tightens the conversational moat (the SAME drive touches BOTH halves).
- **Mechanism:** when hunger raises the shared `dopamine`, the already-built `_da_confidence_gate`
  (`nav_conv_merged_bridge.py:2129`, moat-safe — can ONLY tighten abstention) shifts the conversational read → the
  one drive demonstrably touches BOTH the acting half and the conversing half of the one brain.
- **Reusable machinery:** `co_resident_drive` + `co_resident_nav_critic`/`co_resident_limbic` (shared DA) +
  `_da_confidence_gate`. **Behavioral signature:** a hungry brain is measurably more conservative in conversation
  (higher abstention) than a sated one, with the moat ASSERTED byte-unchanged (it only tightens).
- **Anti-cheat:** moat never LOOSENS (structurally guaranteed); the shift tracks the drive (lesion → no shift).
  **Cost:** SMALL, but it is a *property demonstration*, not a life — a cheap FOLLOW-ON to Option 1 (the scoping's
  Phase-3.1), NOT the first slice.

### OPTION 4 — "lived consolidation": make the develop-loop's SLEEP trigger event-driven (N.17 awake-replay at pauses / D.23 misplace-novelty), not a scripted phase.
- **Mechanism:** trigger SWR replay on a LIVED novelty/reward event (a `gen_concept`/CA1-mismatch spike or a
  reward pulse) rather than a scheduled encoding phase — so consolidation is a consequence of what the agent lived
  (**D.23 misplace / N.17 awake-replay**). **Cost:** MEDIUM (needs a novelty/mismatch read-out wired to the replay
  trigger). A THIRD slice — the scoping's Phase-3.2 — after Options 1+2. Kept here for completeness.

### NOT-first-slice (deferred, do NOT scope now):
- **The learned spatial POLICY from intrinsic reward** = Tier-4 dendrite wall (3rd NEGATIVE; the spiking-living-loop
  de-risk unified this frontier — the same wall blocks nav read-outs #6/#9 and the survival policy). Owner-deferred.
- **True `cp_connections` synaptic persistence** of the merged bridge (§1f) — a follow-on; `LivingState` JSON resume
  suffices for the first slice.
- **The full N-cluster arousal/circadian state-machine** (day/night, wake-sleep flip-flop) — a much later slice; the
  first slice uses a simple wake=live / sleep=consolidate alternation.

---

## 5. THE SINGLE RECOMMENDED CHEAP-FIRST DE-RISK

**Build Option 1 as `_tier3_live_and_remember_derisk.py`** (a new runner; extend the
`_tier3_spiking_living_loop_derisk.py` + `persistent_living_loop_derisk.py` pattern; reuse
`build_merged_nav_conv_bridge(co_resident_drive=True, co_resident_perception=True|co_resident_generalization=True,
co_resident_composer=True)` + `perceive_and_ground` + `homeostatic_hook` + `BridgeLineage`). **NO `sim/` edit
predicted** (both prior living-loop de-risks needed none; all seams are additive default-off).

**Ladder:** (1) **1-seed GPU smoke** — the continuous loop closes: the drive-biased agent lives, encounters +
grounds ≥2 objects during its OWN behaviour, stores facts, answers who/what about them + abstains on
never-encountered, and resumes the exact life+memory after a reset; the moat is byte-frozen in vivo (assert the
parser/composer synapses unchanged across the live run, as the spiking-living-loop smoke did). (2) **6-seed** for the
survival + generalization + all-anti-cheats-collapse claims (the standing 6-seed rule for generalization).

**Decisive checks (GO / BOUNDARY / NEGATIVE bands):**
1. **CONTINUOUS SURVIVAL** — energy stays in-band, never crashes, over the continuous life (GO), while **drive-lesion
   and yoked-random both STARVE** (the discriminator is regulation, not luck).
2. **LIVED, OPEN-ENDED EXPERIENCE** — the agent grounds+stores facts about the objects IT encountered (fact-count
   grows from behaviour); a **permuted-world / grounding-lesion** control collapses the stored knowledge to chance
   (the memory rides the LIVE percept + the agent's own trajectory, not a script).
3. **CONVERSE ABOUT WHAT IT LIVED** — who/what queries about encountered objects answer correctly AND the **no-confab
   MOAT holds** (never-encountered → None; byte-unchanged; a breach = HARD STOP).
4. **PERSISTENCE ACROSS RESET** — reload resumes the EXACT internal life-state (drive + energy + position + composer
   fact-store) — a **no-persistence cold-start visibly differs** (re-warms + has no lived memory).
5. **REWARD-PROVENANCE** — `r` = spiking drive-reduction read off `cp_firing_states`, asserted NO `r=f(distance)`
   host term.

**Honest expected boundary (a valid deliverable per the actual-goal mandate):** if the fixed BG-cascade's inability
to *learn an efficient forage policy* (the deferred dendrite wall) makes survival underperform the rate-proxy at
some world-size, that maps the substrate cost — the first slice is scoped on survival/foraging where the reward is
load-bearing with a simple policy (as the rate-proxy GO 6/6 established), so this should NOT block the slice; if it
does, that precisely-localized NEGATIVE (survival-policy = dendrite, but perceive-remember-converse = GO) IS the
deliverable.

**⇒ The FIRST persistent living agent: a merged one brain that LIVES (drive-biased behaviour it isn't scripted to
do), PERCEIVES + REMEMBERS what it encounters, and can be TALKED TO about its own lived experience — persisting
across resets — with the no-confab moat intact and NO `sim/` edit.** It is the smallest thing that is genuinely
persistent AND open-ended AND unifies perception+memory+conversation on one brain, sidestepping the one deferred
wall.

---

## 6. Anti-cheat standing rules (apply to every slice)
- **No-confab moat NEVER weakened** — a moat breach is a HARD STOP (the composer slice is array-disjoint from the
  nav read-out, so the moat holds by construction; assert it byte-unchanged in vivo).
- **Validate-by-function** — match each control to what the signal COMPUTES (drive-lesion for survival, grounding-lesion
  for lived-memory, permuted-world for open-endedness), not a task that ignores it (the 2026-06-10 lesson).
- **6-seed** for any generalization/survival claim; a single-seed smoke is a mechanics check only.
- **BRAIN-BASED-ONLY** — host code legitimate ONLY for the world (object layout, sensory render) + the body (moving on
  the motor winner, eating). The drive + reward + perception + memory + conversation are the brain's job (spiking).
- **Honest negatives are the deliverable** — a precisely-localized boundary (e.g. survival-policy=dendrite while
  perceive-remember-converse=GO) maps the substrate and is a valid result.

---

## 7. Verdict for the owner
Tier 3 is **largely done at the component level**; the genuine residual is a **runner-only SYNTHESIS** (the
continuous `live()` loop + a co-resident perceiving/composing body + persistence — all validated seams) plus the
honest **open-ended-not-scripted** upgrade. **Recommended: Option 1 (`_tier3_live_and_remember_derisk.py`)** — the
first persistent living agent that perceives, remembers, and can be talked to about what it lived, NO `sim/` edit,
moat intact, 1-seed smoke → 6-seed. The learned spatial policy stays the owner-deferred Tier-4 dendrite wall (off
the critical path). Options 2 (develop-with-a-body), 3 (cross-modal one animal), 4 (lived consolidation) are the
ranked follow-ons.
