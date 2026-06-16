# Step-3 frontier — making PERCEIVED content COMPOSABLE: deep-research scoping (2026-06-16)

**Status:** READ-ONLY deep-research + catalog/literature review (the standing "deep research FIRST at a new
direction" opening move). No `sim/` code, no GPU, no build, no experiment run. The single deliverable is this
doc. Every load-bearing project fact is cited to file + line; literature claims are cited to paper.
**Author role:** read-only computational-neuroscience research subagent.
**Question (verbatim from the tasking):** how could a LEARNED spiking cortex (replacing the fixed VSA algebra)
let a PERCEIVED (rate / grounded) percept be COMPOSED into a novel role-filler fact — i.e. dissolve the
rate-vs-phasor cross-code wall? Scope the most biologically-grounded ACHIEVABLE path + the cheapest first
experiment.

---

## 0. Terms (defined once — no undefined acronyms)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated spiking neurons stepped by one
  `_run_one_simulation_step` loop. "The brain."
- **rate code** — information in a neuron's (or pool's) firing-rate magnitude over a window. The navigation
  perception (`cortex_it`, the ventral object-identity ensembles) is a rate code on Izhikevich neurons.
- **phasor code** — information in the PHASE of a unit-magnitude complex value, in `[0,1)^D` (or
  `e^{iθ} ∈ ℂ^D`). The conversational composer stores concepts as phasor codes on `RESONATE_AND_FIRE` (RF)
  neurons + complex synapses (`sim/bridge.py:5447–5556`).
- **FHRR (Fourier Holographic Reduced Representation)** — the production VSA scheme: bind = elementwise complex
  product of phasors, unbind = multiply by the conjugate, bundle = complex sum, cleanup = max phase-cosine over
  the codebook. Realized on the RF substrate (`research/runners/rf_phasor_composer.py`).
- **role / filler / bind / unbind / bundle** — a *role* is a slot (agent/action/patient/attribute/polarity); a
  *filler* is a concept. **bind** = (role, filler) → one composite; **bundle** = sum several bound pairs into a
  fact; **unbind** = recover a filler given the composite + the role.
- **the rate-vs-phasor wall** — the load-bearing obstacle: the perceived object is a *rate* ensemble
  (`cortex_it`); the composer consumes/produces *phasor* codes. A synaptic current from `cortex_it` into the RF
  slice does NOT deliver a phasor the bind/unbind algebra can consume (design doc §6,
  `docs/plans/2026-06-10-functional-integration-one-brain-design.md:314–336`).
- **engram tag** (Tonegawa, catalog D.14) — the set of neurons that fired in a window (`start_engram_recording`
  → `commit_engram_tag`); `stimulate_tag` re-drives that ensemble (causal recall). SHIPPED on the bridge.
- **grounded code** — a concept code that is a deterministic function of the sensory features of the object
  (vs a free random code). The composer has the interface (`RFPhasorComposer(grounded_codes=...)`,
  `rf_phasor_composer.py:81–89`).
- **stream-cortex codes** — concept codes the bridge LEARNS from the raw conversation stream by rate-Hebbian
  co-occurrence + population code; graded, real-valued, moderately decorrelated (between-code cos ≈ 0.05);
  validated multi-seed at 64/320 concepts (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`).
- **NMS (nonlinear mixed selectivity)** — Rigotti-Fusi: PFC neurons respond to nonlinear COMBINATIONS of task
  variables, yielding a high-dimensional code from which arbitrary combinations are LINEARLY readable
  (jneurosci.org/content/41/35/7420; sciencedirect S0896627324002782).
- **two-compartment / Larkum-BAC neuron** — a pyramidal model with a basal (bottom-up) + apical (top-down)
  compartment; coincident basal+apical drive triggers a burst (a multiplicative AND). The project HAS this in
  numpy (`sim/dendritic_neuron.py`), NOT on the bridge.

---

## 1. DIAGNOSIS — what "compose perceived content" requires that (B)'s engram-recall lacks

### 1.1 The capability the (B) interaction has, and the one it is missing

The just-completed (B) PERCEPTION→MEMORY arc is, end-to-end, a **RECALL** loop and an honest one:
- `funcint_perception_to_memory_probe.py` (clean read-out) — perceive A → tag → later recall A, lesion-confirmed,
  6/6 (`2026-06-16-funcint-perception-to-memory-cheap-first.md`).
- `funcint_perception_to_memory_trained_probe.py` (TRAINED noisy `cortex_it→language_output` read-out) — recall
  survives a learned lossy map, 24/24, lesion 23/24 (`2026-06-16-funcint-perception-to-memory-trained-map.md`).
- `navigate_to_see_then_answer.py` — the BEHAVIORAL version on one bridge: navigate, perceive live, tag, then
  "what did you see?" → reactivation names it, **abstains** on the unseen object, 3/3
  (`2026-06-16-navigate-to-see-then-answer.md`).

**What recall does:** stores the *perceived ensemble itself* as a tag and reactivates it — so it NEVER converts
the rate percept into a phasor (design §6, "the engram-tag mechanism SIDESTEPS that wall"). This is exactly why
(B) works: it routes AROUND the wall.

**What recall cannot do (the precise gap):** the tag is an opaque, atomic pointer to "the ensemble that fired."
It is not an *operand* of the composition algebra. You can recall "I saw the apple," but you cannot:
- **bind the perceived apple into a NOVEL role-filler fact** — `store(agent=<perceived apple>, action=is,
  patient=red)` — because `store` needs a *phasor concept code* for the filler, and the percept is a rate
  ensemble / an engram pointer, not a phasor;
- **unbind a perceived object out of a stored fact** and clean it up against the *perceptual* codebook;
- **generalize** — answer about a never-before-composed (perceived-object, role) combination, which is the whole
  systematicity gift of composition.

Composition requires the percept to be a **commensurable, algebra-ready operand** — a code the bind/unbind/bundle
ops can take in and the cleanup can resolve. Recall requires only an index back to a stored ensemble. That is the
entire difference, and it is the rate-vs-phasor wall.

### 1.2 The exact obstacle — rate-vs-phasor incommensurability, stated mechanically

Two codes, two substrates, one mismatch:

| | navigation perception | conversational composer |
|---|---|---|
| code | RATE (firing-rate magnitude over a window) | PHASOR (phase in `[0,1)^D`, unit magnitude) |
| neuron | Izhikevich (`cortex_it`, V2→IT STDP-fed, `g11_bg_runner.py:2474,2577`) | RESONATE_AND_FIRE + complex synapses |
| bind operand | a vector of pool rates | a vector of phases (`e^{iθ}`) |
| consumed by | engram tag / labeled-line read-out | FHRR `_bind`/`_unbind`/`_bundle` (`rf_phasor_composer.py:117–145`) |

The composer's `_bind` (`rf_phasor_composer.py:117`) literally does `z = exp(2πi·phases)` and a diagonal complex
synapse multiply. Feed it a rate vector and there is no phase to multiply — the operation is undefined on the
wrong code. The design doc's honest prediction (§6): "a naive `wire cortex_it → composer role bank` route would
inject a rate pattern the algebra cannot bind — a likely honest negative that maps the real limit."

**The deeper, already-mapped reason a naive learned fix is hard (the 2026-06-16 capability map):** the project
just settled (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`, CLAUDE.md UPDATE
2026-06-16) that:
- a LEARNED binder is reachable on the substrate for **single-attribute** binding (real-LIF held-out 0.833 =
  100% of the numpy reference, ON/OFF rate coding), AND it generalizes systematically on the stream codes
  (held-out 0.889, `2026-06-16-learned-bind-reachable-on-stream-codes.md`);
- but **multi-attribute BUNDLING** (a fact = a superposition of bindings, the actual conversational structure) is
  **not learnable from scratch on point neurons** — additive bind has no inverse (0.193), a learned *linear*
  inverse cannot be a reciprocal (0.056, breaks even single-attribute), while a **fixed ±1 self-inverse bind
  bundles 0.989** (positive control). The reason is structural: unbinding role *t* from a superposition needs a
  **role-dependent multiplication** (`bundle / u_t`), and a shared LINEAR unbind on a point neuron cannot
  implement a role-dependent scaling — the same Mikulasch-Priesemann point-neuron limit ("multiplication is a
  dendritic operation").

**So the diagnosis is two-layered:**
1. **(the cross-code layer)** the percept (rate) is not a composer operand (phasor) — this is the wall the
   tasking names;
2. **(the binding-substrate layer)** even WITH a commensurable code, the *bundling* algebra that makes a fact
   wants a FIXED self-inverse (or dendritic-multiplicative) primitive; a from-scratch learned point-neuron bundle
   does not work. Single-attribute bind IS learnable; bundling needs the structure.

The good news that organizes the whole option ranking: **layer 2 is already solved by the production composer**
(it binds the LEARNED stream codes with the fixed ±1 / FHRR primitive — recall 0.92). The frontier the tasking
asks about is **layer 1**: get the PERCEPT into a code the composer can already bind. The cheapest dissolution of
the wall is therefore "**make the perceptual code BE (or map deterministically to) a composer concept code**,"
NOT "replace the whole bind algebra with a learned one" (which the 2026-06-16 map shows loses bundling).

### 1.3 The decisive, already-present precedent (this re-scopes the whole problem)

There is an EXISTING numpy-reference result that demonstrates layer-1 dissolution:
`research/runners/_visual_grounded_composition_probe.py` (recorded
`2026-06-04-cheat4-visual-grounding-cheap-first-RESOLVES.md:76–95`):

> Each **real V1-Gabor sensory code** is converted to a phasor via a **FIXED deterministic complex projection**
> (so the phasor code is a function of the sensory features — grounded, not free), then the composer's
> bind/bundle/unbind/cleanup runs on a 2-role fact of two visual-grounded concepts:
> **CLEAN compose = 24/24 = 100%; CORRUPTED-sensory compose (agent slot from a noisy+shifted image) = 11/12 = 92%.**

This is the tasking's question answered at the **numpy-reference** level for the visual subset: a *perceived*
(sensory-grounded) code, passed through a FIXED transcoding, **composes** through the existing algebra and
survives sensory corruption via cleanup. It proves the wall is dissolvable by **grounded codes + a fixed
rate→phasor map**, and it does NOT require a learned bind (layer 2 stays the fixed FHRR op). The OPEN gap — and
the actual step-3 build — is doing this on the **merged bridge with the LIVE navigation `cortex_it` rate
ensemble** as the source (not a numpy V1 matrix), and ideally with a LEARNED (vs fixed-random) transcoder so the
perceptual→concept alignment is itself brain-based.

---

## 2. RANKED, BIOLOGICALLY-GROUNDED OPTIONS

Ranked by **achievability now × biological faithfulness × how directly each dissolves the wall**. Each sits
downstream of the already-solved pieces (the learned stream codes, the fixed bundling primitive, the spiking NEF
cleanup, the learned no-confab familiarity gate) and changes only the percept→operand path.

### Option (a) — RECOMMEND: SHARED GROUNDED CODES — make the perceptual code map (deterministically/learned) to a composer concept code

- **Mechanism.** The composer concept code for "apple" IS (a fixed-or-learned projection of) the navigation
  perception of an apple. Concretely: `cortex_it`'s apple rate ensemble → a transcoding map M → the phasor (or
  ±1) code the composer already uses for "apple." Then `store(agent=<that code>, …)` is just `store` with the
  composer's own apple code — the percept is now a first-class operand and **the wall does not exist** because
  there is one representation. The map M can be (a1) a FIXED projection (the precedent's `_projection`,
  `_visual_grounded_composition_probe.py:20`) or (a2) a LEARNED Hebbian read-out (the `cortex_it→language_output`
  trained map the (B) trained-probe + navigate-to-see already grow by co-firing,
  `2026-06-16-funcint-perception-to-memory-trained-map.md:42–60`) extended to project onto the concept code, not
  just the spelling band.
- **Biology.** Grounded/embodied semantics — a concept's cortical code IS its sensorimotor pattern (Pulvermüller
  distributed cortical word ensembles, CLAUDE.md G.20; Barsalou perceptual-symbol grounding). Mechanistically the
  ventral hierarchy's IT object code projecting to perirhinal/association cortex that the composer reads. Catalog:
  the project's own G.20 ensemble work; the V2→IT→`cortex_it` ventral stream (`g11_bg_runner.py:2474–2593`); the
  grounding INTERFACE already on the composer (`rf_phasor_composer.py:81–89`,
  `core_sim_composition.py:200` "Concept codes are the substrate's own (grounded)").
- **Why it dissolves the wall.** It removes the cross-code transfer entirely: there is no rate→phasor *transfer*
  at compose time because the perceptual pathway DELIVERS the concept code the composer binds. This is the design
  doc's own (C) "shared grounded concepts … the cross-code-transfer problem disappears" (design §6,
  `:330–336`) and (B)'s honest pointer ("the composer's `grounded_codes` interface is the eventual cleaner path").
- **Already-present machinery — the most of any option.** The numpy-reference COMPOSES at 100%/92%
  (`_visual_grounded_composition_probe.py`); the composer's `grounded_codes` param exists; the live `cortex_it`
  perception + the trained `cortex_it→language_output` co-firing read-out exist on the merged bridge; the V1
  Gabor front-end (`sim/visual_cortex.py`) is validated. The build is "make the perceptual read-out land on the
  CONCEPT code (not just the spelling band) and feed the composer," reuse-by-import.
- **Risk (honest).** (i) The composer's own header flags the LIMIT: "producing meaningful grounded codes (real
  object images + abstract-concept grounding) is the open embodied-cognition problem"
  (`rf_phasor_composer.py:83–85`). The *visual subset* is de-risked (the precedent); *abstract* concepts ("red",
  "on", "table") have no obvious sensory grounding — so this dissolves the wall for **perceived OBJECTS bound into
  facts**, which is exactly the tasking's example ("the apple is on the table"), but not for grounding the
  abstract relata. That is an acceptable, honest first scope (the perceived object is the operand the wall blocks;
  the role/abstract fillers are the composer's existing concept codes). (ii) For a fixed-random M the alignment is
  arbitrary (the percept maps to SOME code, not necessarily the one labeled "apple") — fine for *internal*
  consistency (perceive→store→query→answer all use the same percept-derived code) but to answer with the WORD
  "apple" the map must be the LEARNED one (a2), grown so the apple percept lands on the apple concept code. The
  learned-map version is the brain-based target and is the same co-firing mechanism (B) already validated.

### Option (b) — a LEARNED CORTEX that reads the rate percept and binds (the stream-cortex + learned-binder line)

- **Mechanism.** A learned spiking cortical stage takes the `cortex_it` rate percept as input and outputs the
  bound representation directly: it learns to read the (correlated/grounded) rate code and produce the composite,
  trained by a brain-faithful local rule. This is the design's step-3 phrasing verbatim ("a learned
  spiking-cortical binder that reads correlated/grounded codes — Rigotti-Fusi mixed selectivity; reuse the Phase
  2.1/2.2 BPTT spiking cortex", design §7 step 5, `:364–372`).
- **Biology.** Rigotti-Fusi NONLINEAR MIXED SELECTIVITY: a PFC population that responds to nonlinear combinations
  of variables yields a high-dimensional code from which arbitrary combinations are LINEARLY readable
  (jneurosci 41/35/7420; "useful for linear readouts of flexible, arbitrary combinations … flexible control
  between discrimination and generalization"). A mixed-selectivity layer over (percept, role) is the canonical
  substrate for a learned, generalizing bind. Catalog: closest are the high-dimensional conjunctive/expansion
  codes (F.12 cerebellar codon expansion-recoding line 1613; D-cluster DG expansion-recoding line 1228;
  PFC mixed-selectivity noted partial at line 2732 "no DMS-style delay-period mixed selectivity").
- **Why it could dissolve the wall.** A learned read-out does not need the percept to BE a phasor — it learns to
  consume whatever (messy, correlated, rate) code arrives and produce the composite, which is the whole point of
  "a learned cortex reads whatever messy code arrives" (CLAUDE.md composer-as-idealization note). The stream
  cortex already LEARNS concept codes from co-occurrence on the substrate (multi-seed GO) and a learned binder
  generalizes single-attribute binding on those codes (0.889).
- **Risk (honest, and decisive for ranking).** The 2026-06-16 capability map already ran this to ground for the
  *conversational* (phasor) codes: single-attribute learned bind GO, but **multi-attribute BUNDLING is NOT
  learnable from scratch on point neurons** (additive 0.193; learned-linear-inverse 0.056), and a fact is a
  bundle. So a learned cortex can give you the percept→single-bind, but to make a FACT it must still route through
  the fixed self-inverse/dendritic-multiplicative bundling primitive (Option a's composer, or Option c's
  dendrite). This option's NEW content over Option (a) is "learn the percept→operand mapping with mixed
  selectivity" — which is largely what Option (a2)'s LEARNED transcoder already is, minus the (a) insight that you
  don't have to learn the BIND, only the percept→concept code. **Net: (b) collapses, in its achievable form, into
  (a2) + the existing composer**; the "fully learned bind including bundling" version is gated on Option (c)'s
  dendrite and is the months-scale call.

### Option (c) — the DENDRITIC multiplicative binder (the D2 two-compartment substrate) for the bundling point neurons can't do

- **Mechanism.** Realize a genuinely MULTIPLICATIVE bind as a bilinear gate on two-compartment (Larkum-BAC)
  neurons: basal compartment carries the filler (or percept), apical carries the role, the neuron's burst
  probability is their PRODUCT (a native dendritic multiplication), gate weights learned by a LOCAL three-factor
  rule. This is the one substrate that can do the role-dependent multiplication bundling needs, and it generalizes
  by construction (multiplication is operand-independent).
- **Biology.** Catalog G.02 active dendrites / NMDA plateau / Larkum two-layer apical-basal coincidence
  (catalog line ~2644); J.08 NMDA voltage-dependent coincidence detector (the molecular AND). Direct 2026 paper:
  "Bilinear gating of motor primitives" (arXiv 2606.10891) — two-compartment burst = product of soma×dendrite,
  trained by a LOCAL three-factor Hebbian rule, systematic/zero-shot by the multiplicative inductive bias (cited
  in `2026-06-16-onsubstrate-learned-binder-deep-research-scoping.md` §2 Option i).
- **Why it could dissolve BOTH layers.** A dendritic-multiplicative binder makes bundling LEARNABLE (the
  point-neuron limit that blocked from-scratch bundling is precisely the missing multiplication), AND a
  mixed-selectivity two-compartment layer can read a rate percept on the basal compartment — so it addresses
  layer 1 (consume the rate percept) and layer 2 (multiplicative bundling) together. This is the deepest, most
  faithful resolution.
- **Risk (honest).** The two-compartment neuron is NOT on the bridge — it is numpy-only (`sim/dendritic_neuron.py`).
  Putting it on the bridge is a PROTECTED `NeuronModel` edit (~10× compute/neuron, catalog G.02), the months-scale
  arc the owner has repeatedly flagged as a deliberate call. The D2 Phase-1 dendritic *divisive-gain* edit landed
  (verified, byte-identical-off) but its Phase-2 cortex-code use went NEGATIVE for *necessity*
  (`2026-06-14-D2-phase1-DONE-phase2-frontier.md`) — so the dendritic-substrate-on-bridge story is real but not
  yet a binder. This is the right LONG-HORIZON target and the honest "if you want the fully-learned generalizing
  binder-of-facts, this is the substrate"; it is NOT the cheapest first move.

### Option (d) — a learned rate→phasor TRANSCODER (a dedicated cross-code bridge)

- **Mechanism.** A learned spiking stage whose job is exactly the cross-code conversion: input = `cortex_it` rate
  ensemble, output = the phasor (or ±1) code the composer binds. Resonate-and-fire neurons are the natural
  output substrate — they encode a real-valued/analog input into PHASE (literature: "a population of
  resonate-and-fire neurons converts a low-dimensional input into a spatio-temporal spike representation … encode
  information in firing PHASE", semanticscholar Auge-Mueller; arXiv 2510.14515). Train the transcoder so each
  object's rate ensemble drives the RF population to the object's concept phase.
- **Biology.** Cross-modal/representational re-coding (sensory cortex → association cortex format conversion);
  the RF phase-encoding is the project's own FHRR substrate run in *encode* mode.
- **Why it could dissolve the wall.** It is the literal "build the missing transcoder," producing a phasor the
  composer consumes — the percept becomes algebra-ready.
- **Risk (honest).** This is the most ENGINEERING for the least conceptual gain over Option (a): a transcoder that
  maps percept→concept-phase IS Option (a2)'s learned map with an RF output stage. If the target phase is the
  composer's existing concept code, (d) ≡ (a) realized via RF encode. A FREE learned phasor (not aligned to a
  named concept) inherits Option (a)'s "internal-consistency only, can't say the word" caveat. So (d) is a
  *realization detail of (a)*, not a distinct strategy — listed for completeness and because the RF-encode framing
  is the cleanest way to make a learned (a2) output land in phasor space natively.

### Ranking summary

| | dissolves the wall by | bind/bundle | brain-faithful? | on bridge today? | cost | rank |
|---|---|---|---|---|---|---|
| **(a) shared grounded codes** | percept IS a composer concept code (fixed or learned map) | existing fixed FHRR (solved) | yes (the map is co-firing-learned; objects only) | **mostly** (precedent composes 100%/92%; live `cortex_it` + trained read-out exist) | **low** | **1 (recommended)** |
| (b) learned cortex reads rate + binds | learn percept→operand (mixed selectivity) | single-attr learnable; **bundle NOT (point-neuron)** | yes | partial (stream cortex + learned single-bind exist) | moderate | 2 — collapses into (a2)+composer in its achievable form |
| (c) dendritic multiplicative binder | two-compartment AND consumes rate + bundles | learnable bundle (dendritic mult) | yes (most faithful) | **no** (protected `NeuronModel` edit, months) | high | 3 (long-horizon target) |
| (d) learned rate→phasor transcoder | dedicated cross-code stage (RF encode) | existing fixed FHRR | yes | partial | moderate | 4 — a realization of (a) |

**Recommendation:** **Option (a), shared grounded codes**, in its LEARNED-map form (a2), is the most achievable
path that makes real progress and is genuinely brain-based: the perceived object's `cortex_it` ensemble drives a
co-firing-learned read-out onto the composer's concept code, so the percept enters the EXISTING (validated)
bundling algebra as a first-class operand and the wall is dissolved for **perceived objects bound into novel
facts** — exactly the tasking's "bind the perceived apple into 'the apple is on the table'." The
numpy-reference already composes grounded codes 100%/92%; the build is moving that onto the live merged bridge.
The fully-learned-bind-including-bundling resolution is **Option (c)'s dendritic substrate — a deliberate
months-scale owner call** (and the honest deepest path), NOT the first move. Option (b) in its achievable form
reduces to (a2); Option (d) is a realization of (a).

---

## 3. REUSABLE PROJECT MACHINERY (verified file/function pointers)

- **The grounded-codes-compose precedent (the load-bearing one)** — `research/runners/_visual_grounded_
  composition_probe.py`: `_projection` (fixed V1→phase complex projection, line 20), `_to_phasor` (line 27),
  the bind+bundle+unbind+cleanup over a grounded codebook (lines 61–71), the GO gate (line 78). Recorded result
  `2026-06-04-cheat4-visual-grounding-cheap-first-RESOLVES.md:76–95` (100% clean / 92% corrupted). **This is the
  thing the bridge build must reproduce with the LIVE `cortex_it` rate ensemble as the source.**
- **The composer's grounding interface (the drop-in target)** — `RFPhasorComposer(grounded_codes={word: phases})`
  (`research/runners/rf_phasor_composer.py:63,86–89`); `CoreSimComposer(concepts=...)`
  (`core_sim_composition.py:205,239–257`). Passing a `{percept-derived code}` for an object overrides its random
  code — the percept becomes the operand with no `sim/` edit.
- **The live navigation perception + the LEARNED co-firing read-out** — `cortex_it` region + the V2→IT STDP feed
  (`g11_bg_runner.py:2474,2577`); the DENSE plastic `cortex_it→language_output` route grown by Hebbian co-firing
  in `navigate_to_see_then_answer.py` (`_train_route`, the clip-snapshot fix) and
  `funcint_perception_to_memory_trained_probe.py` (`_train_route`, on/off ≈ 13×). The (a2) build extends this
  read-out's TARGET from the spelling band to the composer concept code (same mechanism).
- **The V1 Gabor front-end (the grounded source)** — `sim/visual_cortex.py` (Gabor RFs + retina); reused by
  `_visual_grounding_probe.py` / the unified grounded agents (`unified_agent_visual_grounded.py`,
  `unified_agent_realobject_grounded.py`, `spiking_unified_agent_grounded.py`).
- **The learned stream-cortex codes + their on-bridge learner (Option b input)** — `_phaseB_onbridge_stream_
  cortex_derisk.py`, `_phaseB_online_stream_cortex_derisk.py`, `_phaseB_stdp_cooccurrence_derisk.py`; cached codes
  `research/findings/raw/_phaseB_stream_codes_320_seed{42,43,44}.npy`.
- **The learned binder + its full systematicity harness (Options b/d)** — `research/runners/cortex_learned_
  binder_systematicity_probe.py` (`BilinearBinder` line 277; leakage-free `make_systematicity_splits` 465; the
  four anti-cheats 562/583/629/725). On-substrate spiking realizations: `_phaseB_spiking_bind_onoff_derisk.py`
  (ON/OFF GO 0.806), `_phaseB_onbridge_bind_nonlinearity_derisk.py` (real-LIF single-attr 0.833),
  `_phaseB_fixed_fhrr_bundled_control.py` (the 0.989 bundling positive control).
- **The two-compartment neuron + local plasticity (Option c, numpy)** — `sim/dendritic_neuron.py`
  (`DendriticLayer`: basal forward, fixed-random apical feedback, BAC threshold lowering),
  `sim/dendritic_plasticity.py` (`urbanczik_senn_update`), `sim/dendritic_mlp.py`. The on-bridge dendritic
  *divisive-gain* Phase-1 edit (verified, default-off byte-identical, `2026-06-14-D2-phase1-DONE-phase2-frontier.md`)
  is the precedent for a protected dendritic `NeuronModel` edit.
- **The solved downstream pieces (all options need them, build once)** — spiking NEF cleanup
  (`core_sim_composition.NEF_CLEANUP_OP`, `rf_phasor_composer._spiking_cleanup` line 202, == numpy at V=320); the
  learned Bogacz-Brown familiarity gate / no-confab moat (`2026-06-11-familiarity-gate-v320-GO.md`); the engram
  API + the merged-bridge integration seams (`build_merged_nav_conv_bridge`, `MergedRFComposer`).

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK (the single smallest experiment, numpy/CPU)

**The single load-bearing question:** when the perceptual code is produced by the LIVE navigation `cortex_it`
RATE ensemble (not a numpy V1 matrix) and transcoded to the composer's code, does a PERCEIVED object COMPOSE
into a NOVEL role-filler fact and unbind back to the correct percept — i.e. does Option (a) dissolve the wall
on real percept-derived codes?

> **Probe: extend `_visual_grounded_composition_probe.py` to source its grounded codes from a `cortex_it`
> rate-ensemble forward pass on a small `SimulationBridge` (the perception region + a transcoding map M), then run
> the EXISTING composer bind/bundle/unbind/cleanup on a 2-role fact built from two PERCEIVED objects, and assert
> the unbind recovers the perceived object — including from a CORRUPTED (noisy) percept.** CPU,
> `SIM_BACKEND=numpy`, minutes.

Concretely, reusing the precedent's structure with the source swapped:
1. **Perceive** each of N objects: drive the object's `cortex_it` band on a small bridge (the sensory render, as
   the (B) probes do), step a short window, **read the `cortex_it` rate vector** (`encode_percept` =
   firing-rate read, not a hand-set vector).
2. **Transcode** the rate vector → a composer code via M. Run BOTH variants in the same harness:
   - **(a1 FIXED)** M = the precedent's fixed complex projection (rate → phase). Tests "does ANY deterministic
     map of the real rate percept compose?" (the cheapest, internal-consistency form).
   - **(a2 LEARNED)** M = a co-firing-trained read-out (rate percept → the object's concept code), the
     `cortex_it→language_output`-style Hebbian map extended to the concept code. Tests the BRAIN-BASED form (the
     percept lands on the NAMED concept).
3. **Compose + unbind** with the existing composer (`RFPhasorComposer` / `CoreSimComposer`): build
   `fact = bind(agent, percept_A) + bind(patient, percept_B)`, unbind each role, cleanup over the grounded
   codebook, check `recovered == perceived`. Repeat with a NOISY percept on the agent slot (the corruption test).
4. **Held-out novelty:** compose a (perceived-object, role) PAIR never composed in any setup step — systematicity
   of the perceived-content composition (the gift recall lacks).

**The exact GATE (mirroring the precedent + the systematicity protocol):**
- **GO** if, on real `cortex_it` rate-derived codes, **clean compose ≥ 90%** (unbind agent+patient → correct
  perceived object) AND **corrupted-percept compose ≥ 80%**, multi-seed (42/43/44), with the **held-out novel
  (object, role) pair recovered ≫ chance (1/N)** and ≫ the shuffled-percept-label control — i.e. the perceived
  object composes into a NOVEL fact and unbinds back, matching the numpy-V1 precedent (100%/92%) on
  live-percept-derived codes.
- **NO-GO / honest negative** if clean compose collapses toward chance or only the FIXED map (a1) works while the
  LEARNED map (a2) cannot land the percept on the named concept (→ "shared codes compose internally but the
  brain-based percept→concept alignment is the open piece" — itself the scientific deliverable, and the precise
  pointer to Option c). If even (a1) fails on rate-derived codes (vs the V1-matrix precedent's 100%), that maps
  the rate-read fidelity as the wall (→ the population-code lift, the documented fix).

**Why this is the right cheapest-first move.** (a) It reuses the EXACT precedent that already composes
(`_visual_grounded_composition_probe.py`) and the EXACT composer + cleanup the pipeline ships, swapping only the
code SOURCE to the live `cortex_it` rate ensemble — so a GO is directly comparable to the 100%/92% reference and
*is* the bridge-level proof the wall dissolves. (b) ZERO `sim/` edits, NO protected `NeuronModel` change (Option c
deferred), CPU/numpy, **est. < 30 min for both M variants × 3 seeds**. (c) It gates the whole arc: GO → build the
behavioral "perceive-then-compose-a-novel-fact" task on the merged bridge (the compositional successor to
navigate-to-see-then-answer) with the LEARNED (a2) map; NO-GO → the honest negative localizes whether the limit
is percept→concept alignment (→ Option c dendrite) or rate-read fidelity (→ population code), without spending GPU.

---

## 5. ANTI-CHEATS the de-risk needs

The composition + grounding controls, each defeating a specific way to fake "perceived content composes":

1. **Lesion / no-host-copy provenance (primary, brain-based audit).** The composed fact's filler must be the
   PERCEPT-DERIVED code (read from `cortex_it` firing), never a host-set concept vector. Assert structurally: the
   only write into the percept code is the `cortex_it` rate read; the composer receives that code, not a labeled
   lookup. Zero the transcoding map M → composition of the perceived object must collapse (it now has no operand).
   Mirrors the (B) lesion + provenance (`navigate_to_see_then_answer.py:provenance_check`,
   `funcint_perception_to_memory_trained_probe.py` §3).
2. **Held-out novel-combination (systematicity — THE control that separates compose from recall).** A
   (perceived-object, role) pairing never composed in any setup step must unbind correctly ≫ chance. This is the
   capability recall lacks (recall reactivates a stored ensemble; compose generalizes to new combinations). Reuse
   the leakage-free split + the `memorization_floor` lookup-table control (`cortex_learned_binder_systematicity_
   probe.py:583`): a system that only RECALLS scores at the memorization floor on held-out; a system that
   COMPOSES beats it.
3. **Corrupted-percept robustness (the precedent's own control).** Compose with a NOISY/shifted percept on a
   slot; cleanup must still recover the right concept (the `_corrupt` test,
   `_visual_grounded_composition_probe.py:67`). Guards against a brittle exact-match that is really a lookup.
4. **Shuffled-percept-label control.** Score unbind against SHUFFLED true percepts → collapses to chance. Confirms
   the recovered concept is the COMPOSED percept, not a readout artifact or a fixed structural bias.
5. **The two specific failure modes called out:** **(i) the percept secretly being a host lookup** — caught by 1
   (lesion M + provenance); **(ii) "composition" that is really recall of a memorized pair** — caught by 2
   (held-out novel combo vs memorization floor). The honest-negative line: if (a2) cannot land the percept on the
   NAMED concept, report it (the brain-based percept→concept alignment is the open piece → Option c), do not
   relax to a fixed map and claim a learned win.

---

## 6. HONEST FRAMING — what is grounded vs speculative, and the deepest-path caveat

- **GROUNDED:** grounded codes COMPOSE through the existing algebra (numpy-reference 100% clean / 92% corrupted,
  `_visual_grounded_composition_probe.py` — for the VISUAL/object subset); the composer's `grounded_codes`
  interface exists and is byte-validated == random at parity; the live `cortex_it` perception + the co-firing
  read-out exist on the merged bridge; the fixed bundling primitive is solved (the production composer binds
  learned codes, recall 0.92); single-attribute learned bind is substrate-validated (0.833 real-LIF); the
  population-code lift for rate-read fidelity is documented.
- **SPECULATIVE (the live risks):** (a) whether the LIVE `cortex_it` RATE read (vs a clean numpy V1 matrix)
  yields a code clean enough to compose at the precedent's 90%/80% is **unmeasured** — the rate-code SNR could
  erode it (the documented fix = population code). (b) The LEARNED (a2) map landing the percept on the NAMED
  concept (so the agent says "apple") is the brain-based target and is **unmeasured at compose grade** — the
  (B) trained map lands on the spelling band, not the concept code; extending it is the build's real risk. (c)
  Grounding is de-risked for OBJECTS only; ABSTRACT relata ("on", "red") have no obvious sensory grounding — so
  Option (a) dissolves the wall for **perceived objects as fillers in facts whose roles/abstract fillers are the
  composer's existing concept codes**, which is the tasking's example, but is NOT full embodied grounding (the
  composer's own honest limit, `rf_phasor_composer.py:83`).
- **The deepest-path caveat (stated plainly).** The fully-faithful resolution — a LEARNED cortex that reads the
  rate percept AND learns the BIND including multi-attribute BUNDLING — is **gated on the dendritic
  multiplicative substrate (Option c), a protected `NeuronModel` edit, months-scale, a deliberate owner call.**
  The 2026-06-16 capability map proved from-scratch bundling is NOT learnable on point neurons; the production
  composer's fixed self-inverse primitive (or a dendrite) is load-bearing and biology-grounded, not a shortcut.
  So the RECOMMENDED Option (a) is honestly "the percept enters the EXISTING (fixed-primitive) bundling algebra
  as a grounded operand" — real, achievable, brain-based progress that dissolves the wall for perceived-object
  composition NOW, while the learn-the-whole-bind version stays the deferred dendritic arc. This matches the
  project's settled position (CLAUDE.md UPDATE 2026-06-16): learned representations flowing through a fixed,
  biologically-grounded coincidence/multiplicative binding primitive.

---

## SUMMARY (the requested 6–8 lines)

**Diagnosis:** composing a perceived object into a novel fact needs the percept to be a commensurable
ALGEBRA-READY operand; (B)'s engram-recall only stores an opaque pointer to the perceived ensemble — the gap is
the rate-vs-phasor wall (the `cortex_it` rate percept is not a phasor the composer binds), layered on the
already-mapped point-neuron limit that multi-attribute BUNDLING needs a FIXED self-inverse/dendritic primitive.
**Top option:** SHARED GROUNDED CODES (a) — make the perceptual code map (co-firing-learned) to the composer's
own concept code, so the percept enters the EXISTING (validated) bundling algebra and the wall dissolves for
perceived-object facts; the project ALREADY composes grounded codes at 100%/92% in numpy-reference, the only gap
is sourcing from the live `cortex_it` rate ensemble. **Options (b)** learned cortex reads rate + binds (collapses,
in its achievable form, into (a2) + the existing composer — bundling isn't learnable on point neurons),
**(c)** dendritic multiplicative binder (the deepest, most faithful, but a protected months-scale `NeuronModel`
edit — the honest long-horizon call), **(d)** learned rate→phasor transcoder (a realization of (a) via RF encode).
**Cheapest de-risk:** extend `_visual_grounded_composition_probe.py` to source grounded codes from a LIVE
`cortex_it` rate forward pass (fixed-map a1 + learned-map a2), run the existing composer bind/bundle/unbind/
cleanup on a 2-role fact of two PERCEIVED objects, GATE = clean compose ≥ 90% + corrupted ≥ 80% + held-out novel
(object,role) ≫ chance, multi-seed, CPU/numpy, no `sim/` edit, < 30 min. **Top anti-cheat:** the held-out
novel-combination vs memorization-floor control (it is what separates COMPOSE from RECALL — a recall-only system
scores at the floor on never-composed pairings), backed by the lesion-M + provenance audit (the composed filler
must be the percept-derived code, never a host lookup).

## Sources (literature consulted beyond the in-repo catalog/code)

- "Symbolic Grounding Reveals Representational Bottlenecks in Abstract Visual Reasoning" (arXiv 2604.21346, 2026)
  + "The Vector Grounding Problem" (arXiv 2304.01481) — a large part of the reasoning gap is in PERCEPTUAL
  representations, not reasoning capacity; learned/engineered front-ends extract structured codes a general
  reasoner composes over (supports Option a: fix the percept code, reuse the algebra).
- Rigotti et al., "The importance of mixed selectivity in complex cognitive tasks" + "Emergence of Nonlinear
  Mixed Selectivity in Prefrontal Cortex after Training" (J. Neurosci 41(35):7420, 2021) +
  "Mixed selectivity: Cellular computations for complexity" (Neuron, S0896627324002782) — NMS gives a
  high-dimensional code with LINEAR readout of arbitrary combinations (the substrate for Option b's learned
  generalizing bind).
- "Bilinear gating of motor primitives" (arXiv 2606.10891, 2026) — two-compartment burst = product of
  soma×dendrite, trained by a LOCAL three-factor rule, systematic by the multiplicative inductive bias (the
  direct grounding for Option c's dendritic multiplicative binder).
- Auge & Mueller, "Resonate-and-Fire Neurons as Frequency-Selective Input Encoders" (Semantic Scholar) +
  "Direct Signal Encoding with Analog Resonate-and-Fire Neurons" (arXiv 2510.14515) — RF populations encode an
  analog/real-valued input into spike PHASE (the substrate for Option d's rate→phasor transcoder).
- Frady & Sommer, "Robust computation with rhythmic spike patterns" (PNAS 2019) + Frady-Kleyko-Sommer, "Variable
  binding for sparse distributed representations" (IEEE TNNLS 2021) — VSA binding on resonate-and-fire spikes
  (the project's RF/FHRR substrate; the fixed bundling primitive Option a feeds the percept into).
- Mikulasch, Leugering, Priesemann, "Local dendritic balance…" (PNAS, PubMed 34876505, 2021) — the operations
  needing analog/multiplicative interaction (decorrelation; binding-superposition) are DENDRITIC, not
  point-neuron — explains why bundling is the point-neuron limit and Option c is the deep fix.

**No banking — reported exactly as found.**
