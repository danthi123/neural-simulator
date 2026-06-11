# Functional integration — the real "one brain": synaptic cross-region coupling between the navigation and conversational brains

**Status:** DESIGN (read-only research + design pass; no code written except this doc).
**Date:** 2026-06-10 (overnight Thread C).
**Author role:** read-only deep-research + design subagent.
**Scope:** the high-leverage arc AFTER roadmap step 2's *substrate* consolidation. Step 2 put the
navigation brain and the conversational brain on ONE `SimulationBridge` but with **zero synapses
between them** — they co-reside and do not interact. This doc designs the **functional** integration:
real cross-region SYNAPTIC pathways so the two halves actually influence each other, validated by a task
that an isolated half cannot solve.

---

## 0. Terms (defined once — owner standing requirement, no undefined acronyms)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated spiking neurons stepped by one
  `_run_one_simulation_step` loop.
- **region / slice** — a contiguous block of neuron indices used for one function (a `BrainRegion`).
- **pathway** — a directed set of synapses from one region to another (`RegionPathway`).
- **navigation brain** — the basal-ganglia action-selection cascade + its perception:
  `cortex_v1/v2/it` (the ventral "what" object code), `sc_retina`/`sc_map` (the dorsal/orienting "where"
  superior-colliculus sheet), `sensor_place_readout` (place code), `cortex_{N,E,S,W}` → striatum → GPi →
  thalamus → `motor_{N,E,S,W}` → the spiking commit/WTA selection. Built by
  `research/runners/g11_bg_runner.py:build_bg_brain_regions`; run by `run_moving_goal_episode`.
- **conversational brain** — the parser (`parse_conj` 6 units + `parse_role` 3·40 role-ensemble neurons),
  the dlPFC dialogue-planning loop (`cortex_ctx` ↔ `dlpfc_wm`), and the resonate-and-fire (RF) phasor
  composer (the `rf` slice + complex synapses) that holds fact memory and does the bind/unbind algebra.
  Built/merged by `research/runners/nav_conv_merged_bridge.py:build_merged_nav_conv_bridge`.
- **transmission gate** — a per-synapse multiplier in [0,1] on a pathway's synaptic CURRENT, set at runtime
  by `bridge.set_transmission_gate(name, value)`. Pre-wire a route, hold it closed (gate 0 → no current,
  no STDP cold-start), open it on command (Logiaco-Abbott-Escola 2021 thalamocortical dynamical gating).
- **gate-from-firing coupling** — `bridge.couple_gate_to_pool(gate, control_region, threshold)`: each step
  the gate opens iff a *control population's* smoothed firing rate exceeds a threshold. This is **in
  substrate** — no Python reads the value; the firing of one region opens a route into another
  (`_apply_gate_couplings`, `sim/bridge.py:3002`).
- **engram tag** (Tonegawa, catalog D.14) — the set of neurons that fired above a threshold during a window
  (`start_engram_recording` → run steps → `commit_engram_tag`); later `stimulate_tag` re-drives exactly
  that ensemble (causal recall). Tags are global neuron-index arrays.
- **plasticity gate** — a per-synapse multiplier on weight UPDATES (`set_plasticity_gate`); 0.0 freezes a
  pathway's learning while current still flows. Used to hold the frozen conversational weights inert under
  the navigation reward-STDP stressor (the step-2 5a isolation).
- **VSA** (vector-symbolic architecture) — the composer's role-filler binding algebra (bind = role ⊗
  filler, unbind = the inverse), realised on the RF phasor substrate.

---

## 1. THE TARGET INTERACTIONS — ranked by load-bearing value × cheapness

The brain-based-only standard: the interaction MUST be carried by neurons/synapses on the merged bridge.
Python is legitimate ONLY for the **environment** (world state + rendering the agent's sensory input) and
the **body** (acting on motor output). Everything between sensation and action — including moving a percept
into memory, or turning a parsed command into an action drive — must be synaptic. A Python `bridge_a_value =
bridge_b.read(...)` copy across the two halves is exactly the shortcut to avoid.

Three candidate couplings, biology + existing machinery noted:

### (A) LANGUAGE → ACTION — a parsed command drives the navigation cascade *(CHOSEN FIRST)*

- **What:** the parser comprehends a spoken instruction ("go north"); the role assignment fires the parser's
  `parse_role` ensemble; that firing routes — synaptically — a directional drive into the navigation
  cascade's `cortex_{N,E,S,W}` action pools, biasing action selection so the body moves as instructed.
- **Biology:** language → premotor/motor action priming (Pulvermüller's action-word somatotopy, catalog
  G.20; Broca → premotor). The route into striatum/cortex is the corticostriatal projection (catalog A.12,
  the Kincaid sparse-convergence rule — a real "command cortex → action selection" synaptic route).
- **Why most load-bearing:** it is the cleanest task that **forces** a behavioral readout an isolated brain
  cannot produce — and it **directly resolves the standing nav residual** (the gridworld is orient-solvable,
  so the nav reward isn't behaviorally load-bearing). A *spoken-instruction* task makes the conversational
  channel the only source of the goal direction, so the cross-region route becomes the load-bearing signal
  and a lesion of it must regress behavior.
- **Why cheapest:** the *target* side already exists and is exercised every nav step. There is a **validated
  precedent for the exact wiring shape**: `language_input → motor_X` / `→ cortex_X` plastic pathways
  (`research/runners/b3_supervised_gradient.py`, the concept-pool architecture). The route is a small fixed
  pathway from `parse_role` (agent/action/patient blocks) into `cortex_{N,E,S,W}`, gated. The
  **in-substrate gate-from-firing primitive already exists** (`couple_gate_to_pool`) — the same primitive
  step-2's `hear_synaptic` used to open the parser→composer route — so no new gating code is needed.
- **The hard part (honest):** the parser's role ensembles encode *grammatical role*, not *which direction
  word*. The direction identity ("north" vs "south") lives in the word's concept code, not in `parse_role`.
  So the route needs the *word identity*, which means driving from a region that carries it. Resolved in §3
  by routing from the navigation cascade's OWN action-word channel (`language_input → cortex_X` already
  binds direction words to action pools), gated open by the parser's action-role ensemble — i.e. the parser
  says WHEN to listen, the word identity says WHICH direction. This keeps the route purely synaptic.

### (B) PERCEPTION → MEMORY — what the agent perceives while navigating writes into conversational fact memory

- **What:** while navigating, the agent perceives an object (the ventral `cortex_it` code, never
  coordinates); that perceived ensemble is written into the conversational memory so the agent can later
  answer "what did you see?" — by a synaptic write, not a Python copy of the percept vector.
- **Biology:** perception → episodic/relational memory (catalog D.01 episodic memory, D.02 Eichenbaum-Cohen
  relational binding: perirhinal item stream → hippocampus → cortical consolidation; D.21 locale/taxon — the
  hippocampal cognitive map binds what-is-where). This is the canonical "perception writes memory" loop.
- **Why valuable:** it is the deepest, most brain-like interaction (it closes the perception→memory→language
  loop end-to-end) and the task (navigate to see an object, then answer about it) is unsolvable by either
  isolated brain.
- **Why NOT first (cheapness + risk):** the write is the **central cross-code-transfer problem** (§6). The
  perceived code is a navigation `cortex_it` rate ensemble; the composer stores **phasor** codes (phases in
  [0,1)^D on the RF substrate). These two codes are not commensurable — a synaptic route from `cortex_it`
  into the composer would deliver a rate pattern the phasor algebra cannot bind. There is a substrate-native
  bridge for this (the **engram-tag API** writes the *perceived ensemble itself* as a recallable tag,
  bypassing the phasor codebook — see §3 alt), but it is a second mechanism with its own validation surface.
  Defer to a follow-on once (A) proves a cross-region synaptic route works on the merged bridge at all.

### (C) SHARED GROUNDED CONCEPTS — the nav perception and the composer's concept codes share grounding

- **What:** the composer's concept code for "apple" IS (a projection of) the navigation V1-grounded visual
  response to an apple, so perceiving an apple and talking about it use one representation.
- **Biology:** grounded/embodied semantics (concept codes ARE sensorimotor patterns).
- **Why NOT now:** the composer already has the *interface* for sensory-grounded codes
  (`RFPhasorComposer(grounded_codes=...)`, `rf_phasor_composer.py:81`), but its own honest header says
  *producing meaningful grounded codes is the open embodied-cognition problem*. This is representational
  alignment, not a runtime cross-region pathway — it does not by itself create a synaptic interaction and
  is the hardest of the three. It is the *eventual* substrate that would make (B) cheap (shared codes →
  the cross-code-transfer problem disappears). Park as the long-horizon enabler, not a first build.

**Ranking:** **(A) LANGUAGE→ACTION first** (most load-bearing for the nav-residual fix, cheapest, fully
synaptic with existing primitives), then **(B) PERCEPTION→MEMORY** via engram tags (the deeper loop, the
cross-code problem), with **(C)** as the long-horizon grounding that makes (B) cheap.

---

## 2. THE TASK THAT REQUIRES THE INTERACTION (resolves "reward not load-bearing")

**Task: spoken-instruction navigation ("the commanded-goal gridworld").**

- The agent is in the gridworld. **The goal direction is NOT rendered into the retina and NOT given as
  coordinates.** Instead, each episode (or each block) a 3-word instruction is presented to the
  conversational channel: e.g. *"agent go north"* (and its synonyms/passive frames the parser already
  handles voice-invariantly).
- The ONLY way the body can know which way to move is: parser comprehends the instruction → the
  conversational→navigation synaptic route biases the action cascade → the agent moves in the commanded
  direction. There is no perceptual goal cue and no coordinate goal, so the navigation brain alone has
  nothing to orient to.
- **Success metric:** fraction of steps the agent moves in the commanded direction (or reaches a
  commanded-direction target square), across a block of instructions that **changes the commanded direction
  several times** (mirroring the multi-goal schedule, 4 phases × N steps). Baseline = chance (0.25 for 4
  directions). A successful interaction scores well above chance and **tracks the instruction changes**.
- **Why it forces the interaction (both isolated controls fail):**
  - **Isolated navigation brain** (conversational route lesioned): no goal cue of any kind → it cannot
    score above chance on the *commanded* direction. (It may still wander coherently, but its movement is
    uncorrelated with the instruction.)
  - **Isolated conversational brain** (no nav cascade / no body): it can *parse* the instruction (the parser
    works), but it has no motor output — it cannot move the agent. The conversational capability matrix
    (who/what Q&A, abstention, the no-confab moat) is untouched but produces no behavior in the world.
  - Only the **coupled** brain — parser comprehension routed synaptically into the action cascade — converts
    a sentence into the correct body movement.
- **Why it resolves the nav-reward residual:** in the orient-solvable gridworld the reward wasn't
  behaviorally load-bearing because perception (N1 superior colliculus) carried the task. Here the *only*
  goal signal is the parsed command, delivered through the cross-region route; the route (and the policy
  that learns to follow it) is squarely load-bearing, and the lesion control directly measures it. This is
  the "match the control to what the signal computes" lesson (memory:
  `feedback_validate_signal_by_its_function`) applied at the system level.

**Deeper follow-on task for (B) PERCEPTION→MEMORY (specified, not built first):** *navigate-to-see-then-
answer.* Place a colored object somewhere in the grid; the agent must navigate until it perceives the object
(the ventral `cortex_it` code fires), at which point the perceived ensemble is written to memory; then,
queried "what did you see?", the agent answers the perceived object. Success = the queried answer matches the
object actually perceived, and abstains (returns None) when nothing was perceived (the no-confab moat as the
anti-confabulation control). Isolated nav can perceive but not report; isolated conv can report but never
perceives — only the coupled brain closes the loop.

---

## 3. THE SYNAPTIC MECHANISM for the chosen first interaction (LANGUAGE → ACTION)

**Design goal:** the parser's comprehension of an instruction must, by neuron firing and synaptic current
alone, bias the navigation cascade toward the commanded direction — with no Python value crossing between
the conversational and navigation halves.

### 3.1 The route (all on the merged bridge; reuse-by-import; default-off = byte-identical)

The merged bridge already holds, as disjoint slices: the navigation cascade (incl. `language_input` and
`cortex_{N,E,S,W}` when text-IO is enabled), the parser (`parse_conj`, `parse_role`), the dlPFC, and the
RF slice. The route adds:

1. **Direction-word identity already reaches the action pools.** The navigation builder's text-IO block
   wires `language_input → cortex_X` (the same shape b3/concept-pool validated). Driving `language_input`
   with the direction word "north" already biases `cortex_N`. This is the WHICH-direction signal, and it is
   already synaptic.

2. **The parser supplies the WHEN (a gate), not a value.** Add ONE transmission gate, `command_route`, on
   the `language_input → cortex_X` action pathways (pre-wired weight unchanged; the gate scales the CURRENT).
   Hold it **closed** at rest (`set_transmission_gate("command_route", 0.0)`), so spurious `language_input`
   activity does not drive the body between instructions. Couple the gate to the parser's **action-role
   ensemble** via the **existing in-substrate primitive**:
   `bridge.couple_gate_to_pool("command_route", control_region="parse_role_action_block", threshold=...)`.
   (The control region is the `parse_role` "action" sub-block; the parser fires it when it has comprehended
   the verb of the instruction.) Each step, `_apply_gate_couplings` opens `command_route` iff the parser's
   action ensemble is firing — i.e. **the parser's comprehension, in spikes, opens the route from the
   command word into the action cascade**. No Python reads or writes the routed value; the firing of one
   region gates the current into another. This is exactly the mechanism step-2's `hear_synaptic` used
   (parser ensemble → gate → composer), now pointed at the navigation cascade.

3. **Comprehend → latch → act (the validated timing).** Step-2 found a faithful timing fix (no magnitude
   change): drive the parser conjunction for a PRE-WINDOW until it fires and opens the gate, THEN run the
   action-readout window holding the parser-opened gate (`ROLE_GATE_PREWARM_CAP_STEPS` pattern,
   `unified_brain_bridge.py:79`). The same order applies: present the instruction → parser fires →
   `command_route` opens → the direction word's `language_input → cortex_X` current biases action selection
   for the movement decision. Reuse that pattern verbatim.

**Net:** a sentence enters the parser (legitimate sensory input — the environment presents text); the parser
fires; its firing opens a synaptic route that lets the commanded direction's `language_input → cortex_X`
current steer the body. Every step between the parsed sentence and the motor decision is neurons + synapses.

### 3.2 What protected `sim/` edit is needed — NONE (preferred), with one tiny additive fallback

- **Preferred (zero `sim/` edit):** the route is one extra `RegionPathway` set (`language_input → cortex_X`
  tagged `transmission_gate="command_route"`) appended via the existing `extra_pathways=` hook of
  `run_moving_goal_episode`, plus a `couple_gate_to_pool` call in the `prebuilt_post_init_hook`. Both are
  public APIs. `couple_gate_to_pool` requires the brain-region framework (the merged bridge has it), so —
  unlike step-2's `inject_explicit_wiring` bridge — the **name-based** `couple_gate_to_pool` works directly
  (no `couple_gate_to_indices` runner-shim needed). Default-off (gate never coupled) ⇒ byte-identical nav.
- **Fallback (if the action-role sub-block is not its own region):** `couple_gate_to_pool` resolves the
  control pool by *region name*. The parser is two regions (`parse_conj`, `parse_role`); the "action" block
  is a *sub-range* of `parse_role`, not its own region. Two clean options, both additive: (i) split
  `parse_role` into three regions (`parse_role_agent/action/patient`) in `parser_regions_pathways` — a
  runner-side change, no `sim/` edit, but it perturbs parser slice indices (re-validate the parser pass);
  or (ii) reuse the step-2 runner-side `couple_gate_to_indices` helper (`unified_brain_bridge.py:123`) which
  takes raw indices (the action-block indices we already hold) and appends the identical coupling dict — no
  `sim/` edit, no region split. **Recommend (ii)** (it is already written and validated, and keeps the
  parser byte-identical). So: **no `sim/` edit in either path.**

### 3.3 (B) PERCEPTION→MEMORY mechanism (specified for the follow-on, via engram tags)

For the deeper loop, the substrate-native write that **sidesteps the cross-code problem** is the engram-tag
API (already on the bridge, no `sim/` edit):

- While navigating, when the object is perceived, `start_engram_recording("seen_apple")`, run the perception
  window, `commit_engram_tag("seen_apple", region_filter=["cortex_it"], top_k=...)`. The tag is the *actual
  perceived ensemble* — no phasor code, no Python copy of a percept vector; the neurons that fired ARE the
  memory.
- To answer "what did you see?", `stimulate_tag("seen_apple")` re-drives that ensemble; the readout reads
  which conversational concept pool (or which `language_output` spelling) the reactivation evokes. The
  composer's `grounded_codes` interface is the eventual cleaner path (so the tag's reactivation maps onto a
  composer concept), but the engram tag alone closes the perceive→store→recall loop synaptically.
- This is the catalog D.14 mechanism the project already shipped + validated (engram stim-recall 87.5%
  multi-seed); the new part is *driving the tag from live navigation perception* rather than from a
  `language_input` cue. It is the right second build because it makes the perception→memory interaction
  synaptic without solving (C)'s grounding problem first.

---

## 4. THE CHEAP-FIRST DE-RISK (smallest probe that shows ANY cross-region synaptic influence)

**The single load-bearing question:** does a parser-opened synaptic route measurably bias the navigation
cascade's action pools? Probe it BEFORE any task harness, episode loop, or reward.

**Probe (CPU-cheap, no nav episode, no reward, ~minutes):**
1. Build the merged bridge (`build_merged_nav_conv_bridge`) WITH the text-IO nav block (so `language_input`
   and `cortex_{N,E,S,W}` exist) and the `command_route` gate wired + coupled to the parser action block.
2. With the gate **closed**, drive `language_input` with "north"'s code and measure `cortex_N` vs
   `cortex_{E,S,W}` firing → expect **no selective bias** (route closed: the command does not reach the
   body). This is the closed-gate control.
3. Drive the parser conjunction for "… go …" (the action verb) for the pre-window so the parser fires and
   the coupling **opens** `command_route`; hold it open; drive `language_input` with "north" → measure
   `cortex_N` ≫ `cortex_{E,S,W}`. **A measurable, direction-correct bias that appears ONLY when the parser
   has fired = the cross-region synaptic influence exists.** Four-cardinal sweep: each command word biases
   its own action pool.
4. **The lesion in the probe itself:** zero the `command_route` weights (or never couple the gate) → the
   bias vanishes even with the parser firing → confirms the bias is carried by THAT route, not by leakage.

**Cost:** CPU is fine — this is a static drive + a short settle on a few hundred-to-few-thousand-neuron
merged bridge; no 1800-step episode, no GPU needed. GPU only for the full task (the episode loop +
reward-STDP). This mirrors how step-2 de-risked `hear_synaptic` with `_step2_gated_route_probe.py` /
`_step2_synaptic_holdopen_validate.py` before any full build — reuse that probe scaffold.

**Gate to proceed:** the parser-opened bias is direction-correct on ≥3/4 cardinals and vanishes closed and
vanishes lesioned. Only then build the task harness + the 6-seed behavioral A/B.

---

## 5. ANTI-CHEATS (the interaction must be SYNAPTIC, and the task must require BOTH brains)

The whole point is that the interaction is real (synaptic) and the task genuinely needs both halves. Three
controls, each defeating a specific way to fake it:

1. **Lesion the cross-region pathway → the interaction vanishes.** Set `command_route` weights to 0 (or
   never open the gate) and re-run the task: behavior must collapse to chance on the commanded direction. If
   behavior survives, the command is reaching the body by some other path (a leak, or a Python copy) — not
   the designed synaptic route. This is the primary load-bearing test (and it is the same control that
   resolves the nav-reward residual: the route must be *necessary*).
2. **Provenance check — no Python copies the value across.** Audit the task loop: the conversational read
   (the parsed `{role: word}`) must NOT be written into any navigation drive array by host code. The ONLY
   coupling is the `couple_gate_to_pool`/`couple_gate_to_indices` gate (which transmits a 0/1 *gate state*
   from firing, not a value) plus the pre-existing `language_input → cortex_X` synapses (which carry the
   word identity the environment legitimately presents as text). Concretely: grep the harness for any
   assignment of a parser-derived quantity into `cp_external_input_current` at navigation indices; there
   must be none beyond presenting the instruction text to `language_input` (a legitimate sensory render).
   (Note the navigation runner still has its own host scaffolds — that is the separately-tracked nav cheat
   ledger; THIS anti-cheat is specifically about the *new cross-region* coupling being synaptic.)
3. **Both-brains-required (isolated controls fail).** Run the two isolated controls of §2: nav-only (route
   lesioned) scores chance on the commanded direction; conv-only (no body) produces no movement. Only the
   coupled bridge scores above chance AND tracks instruction changes. If a single isolated brain already
   solves it, the task does not actually require the interaction (re-design the task).
4. **Instruction-scramble control (the task is genuinely instruction-following).** Permute the
   word→direction mapping in the instructions (say "north" but reward moving south): a real
   instruction-following agent regresses; an agent exploiting a fixed structural bias does not. This is the
   permuted-label control the project uses elsewhere (e.g. v16 compose anti-cheat), applied to commands.

---

## 6. HONEST could-be-NEGATIVE — the central cross-code-transfer risk

**The deepest risk is representational, and it is exactly why (A) is first and (B)/(C) are deferred.**

- **Within (A), the risk is mild and routed-around.** The parser carries *role*; the *direction identity*
  must come from the word, not the role ensemble. The design routes the identity through the navigation's
  OWN `language_input → cortex_X` channel and uses the parser only as an in-substrate gate (§3.1). So (A)
  never tries to transfer the parser's code INTO the navigation code — it gates a navigation-native channel.
  The residual risk: the parser's action ensemble may fire too weakly/burstily to hold the gate open over
  the action window — but step-2 already hit and solved exactly this (the gate pre-warm /
  comprehend→latch→act timing), so it is a known, fixed failure mode, not an open one. An honest negative
  here would say "the parser firing cannot reliably gate at dt=1.0 even with pre-warm" — a real, bounded
  substrate limit, cheaply measured by the §4 probe.

- **The central cross-code-transfer problem lives in (B), and it is real.** The navigation perception is a
  **rate code** in `cortex_it` (Izhikevich firing-rate ensembles); the composer is a **phasor code** (phases
  in [0,1)^D on resonate-and-fire neurons + complex synapses). These are not commensurable: a synaptic
  current from `cortex_it` into the RF slice does NOT deliver a phasor the bind/unbind algebra can consume,
  and a phasor read does not produce a rate ensemble the nav cascade recognizes. A naive "wire `cortex_it`
  → composer role bank" route would inject a rate pattern the algebra cannot bind — a likely **honest
  negative that maps the real limit**: *the exact-inverse VSA algebra demands clean phasor codes, so a
  messy rate percept cannot be bound by it directly.* (This is the same idealization the project already
  documents — the composer is a principled VSA idealization, not a learned cortex; see CLAUDE.md
  "composer-as-idealization".)
  - **The substrate-native route around it (why (B) is still tractable):** the **engram-tag** mechanism
    (§3.3) stores the *perceived ensemble itself* as the memory and recalls it by re-stimulation — it never
    converts the rate percept into a phasor, so the cross-code mismatch does not arise. (B) via engram tags
    is therefore a real synaptic perception→memory interaction that **sidesteps** the algebra. What it does
    NOT give is *composition over perceived content* (you can recall "I saw the apple" but not algebraically
    bind the perceived apple into a novel role-filler fact) — that genuinely requires (C) shared grounded
    codes or step-3's learned cortex.
  - **(C) is the principled fix and the honest long-horizon limit.** Making the composer's concept codes BE
    the V1-grounded perceptual codes (the `grounded_codes` interface) would let perceived content flow into
    composition with one representation — but the composer's own header is explicit that *producing
    meaningful grounded codes is the open embodied-cognition problem*. So full perception→composition is
    correctly deferred; an honest negative on a premature (B)-with-binding attempt would re-confirm the
    rate-vs-phasor wall and point at (C)/step-3 as the only real fix.

- **Net honest framing:** (A) is expected to work (the route is navigation-native, gated by parser firing,
  with a known timing fix) and yields the load-bearing behavioral interaction + the nav-residual fix. (B)
  via engram tags is expected to work as a *recall* interaction but NOT as a *compositional* one — and that
  boundary IS a scientific deliverable: it maps precisely where the rate substrate and the phasor algebra
  can and cannot hand off, and motivates step-3 (a learned cortex that reads correlated/grounded codes).

---

## 7. SEQUENCING + how this sets up step-3 (the cortex)

1. **De-risk (A) [§4]** — the parser-opened-route bias probe (CPU, minutes). Gate: direction-correct bias
   that appears only on parser firing and vanishes on lesion. *Output:* the cross-region synaptic influence
   exists (or an honest negative on parser gating).
2. **Build the commanded-goal task harness [§2]** on the merged bridge: present instructions, the
   `command_route` coupling, the comprehend→latch→act order; score commanded-direction following over a
   multi-phase instruction schedule. Reuse `run_moving_goal_episode`'s `extra_regions/extra_pathways/
   prebuilt_post_init_hook` (already the integration seams) — add the route + coupling, present the
   instruction instead of a perceptual/coordinate goal.
3. **6-seed behavioral A/B + the four anti-cheats [§5]** (GPU): coupled vs route-lesioned (must collapse),
   nav-only and conv-only isolated controls (must fail), provenance audit (no Python value-copy),
   instruction-scramble (must regress). Gate: coupled ≫ chance and tracks instruction changes; all controls
   behave as predicted. *This is the moment "functional integration" is demonstrated AND the nav-reward
   residual is resolved* (the route/policy is now load-bearing, lesion-confirmed).
4. **Build (B) PERCEPTION→MEMORY via engram tags [§3.3]** (the navigate-to-see-then-answer task): the second,
   deeper synaptic interaction; expect a recall win + an honest compositional boundary. *Output:* the
   perception→memory loop is synaptic; the rate-vs-phasor wall is mapped.
5. **Step-3 (the cortex) is the principled resolution of the boundary (B)/(C) expose.** Replacing the
   composer's exact-inverse VSA algebra with a **learned spiking-cortical binder** that reads correlated /
   grounded codes (Rigotti-Fusi mixed selectivity; reuse the Phase 2.1/2.2 BPTT spiking cortex) + a separate
   spiking familiarity/no-confab gate (the "abstention is a separate familiarity signal" unlock) is exactly
   what would let a *perceived* (rate/grounded) percept be *composed* into memory — i.e. it dissolves the
   cross-code-transfer wall that (B) hits. So this functional-integration arc is the empirical motivation
   for step-3: it produces the concrete negative (perceived content cannot be algebraically bound) that the
   learned cortex is designed to fix. (A) gives the behavioral one-brain demonstration now; (B) maps the
   wall; step-3 climbs it.

**Owner-facing summary of the build order:** cheap CPU probe → commanded-goal task + 6-seed A/B (the
load-bearing language→action interaction, resolves the nav residual) → engram-tag perception→memory (the
deeper loop + the honest cross-code boundary) → step-3 learned cortex (the principled fix for that boundary).
Every step is reuse-by-import with no `sim/` edit expected (the route, the gate-from-firing coupling, the
engram-tag API, and the episode-integration seams all already exist); any `sim/` edit would be additive +
default-off + byte-reviewed, per the standing bar.
