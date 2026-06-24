# Generative + inferential faculty for LOOSE, OPEN-ENDED conversation — deep-research scoping (2026-06-24)

> **Status:** READ-ONLY deep-research-gate scoping. NO `sim/` edits, NO experiments, NO GPU/webapp. Single
> deliverable = this doc + the final message. Catalog reviewed FIRST, then the project's existing pieces
> (every load-bearing claim re-verified against the repo), then current literature (cited). This is a
> *decision/scoping* doc, not a brain-based result and not a commitment to build. The controller should
> trust-but-verify the load-bearing claims (flagged **[VERIFY]**), push, and present the recommendation
> before building.

> **What the owner asked for (verbatim intent).** A GENERATIVE + INFERENTIAL faculty for *loose, open-ended*
> conversation that is **influenced-by but not restricted-to** the brain's stored knowledge — done the
> brain-based way. The current architecture is grounded RETRIEVE + COMPOSE + ABSTAIN (the no-confab moat). It
> cannot (1) generate open-ended ("what do you think about X" — a novel response not in stored facts),
> (2) infer beyond stored facts (relate a newly-DESCRIBED thing to existing knowledge + form a view),
> (3) do phatic/social ("hi"→"hello"); and the moat HARD-abstains (the opposite of loose talk). The owner
> wants loose-but-grounded talk WITHOUT the LLM doing the cognition and WITHOUT hallucination.

> **HARD CONSTRAINTS enforced in every option below:** NO cheats/shortcuts; everything BIOLOGICAL (neurons /
> synapses / their communication — host code legitimate ONLY for the environment + the body); SINGLE BRAIN,
> SINGLE SUBSTRATE (one `SimulationBridge`); the LLM is FLUENCY-ONLY (surface form), NEVER cognition / content
> / opinion; the no-confab moat RELAXES to **graded-confidence** (speak-while-flagging-uncertainty), NOT
> removed (`feedback_moat_not_hard_lossy_memory_ok`).

---

## 0. The one-paragraph answer (the rest is the evidence)

**The faculty is mostly an INTEGRATION of pieces the project has already built and validated — the genuinely
new work is (a) a graded-confidence "hypothesis channel" that relaxes the moat without removing it, and (b) a
neural VALUE/SALIENCE read-out that decides *what is worth saying* (the only piece with no current
substrate).** The single most important prior is **`2026-06-23-genfrontier-b2-generative-replay-derisk.md`
(GO, 6-seed):** the brain's own **generative replay** already INVENTS novel-but-plausible propositions at
**17× chance**, the learned graph is load-bearing (shuffled-graph collapses to the random floor), and the moat
is preserved (0 leaks: a proposal never passes as a *known* fact). That is the direct, biology-faithful,
moat-respecting answer to "open-ended generation" and "inference" — it just needs to be (i) *triggered by a
conversational turn* (a "what do you think about X" / a newly-described thing), (ii) routed through the
already-built **neural serial-order renderer + the spiking fluent faculty (Qwen, fluency-only) + the
gate→constrain→VERIFY loop** for the surface form, and (iii) emitted as a *flagged hypothesis* ("I'd guess
…") rather than a *known fact*, with a **calibrated graded-confidence** gauge. The inference-on-a-novel-thing
case is the **generalization arc** (`2026-06-16-generalization-*`: a newly-perceived/described item's CONCEPT
neurons fire in the right CATEGORY on real spikes) feeding b2 replay over the new node. Phatic/social is a
near-trivial learned stimulus→response reflex. The cheapest-first de-risk is therefore **not a new mechanism
but a falsification probe of the COMPOSITION**: *can a "what do you think about X" turn drive
(generalize-X-to-known-category → generative-replay a novel-but-plausible proposition about X → render it
fluently via the fluency-only LLM → VERIFY it's graph-supported + non-contradictory) such that the output is
NOVEL (not a stored fact), GROUNDED (graph-supported, beats a shuffled-graph control), FLAGGED (graded
confidence, never asserted as known), and the LESION (sever the brain's proposal) collapses it to the LLM
free-hallucinating?* The one **genuine deep new-mechanism** is the spiking VALUE/SALIENCE appraisal (a neural
"is this worth saying / do I have a view" read-out) — catalog C.23/O-cluster, currently MISSING — and the one
**owner-steer fork** is *how much* the moat relaxes (the graded-confidence threshold is a values call, not a
science call).

---

## 1. DIAGNOSIS — the gap, decomposed into mechanism-classes

Five distinct mechanism-classes hide inside "loose open-ended grounded-but-not-restricted conversation." For
each: the biology, what the project ALREADY has (reusable), and what is genuinely missing.

### 1.1 OPEN-ENDED GENERATION — "produce a novel response not in the stored facts"

- **What it computes.** Given a topic/cue, emit a *novel* proposition (never stored) that is *plausible* given
  what the brain knows — then render it as fluent surface form. This is **constructive imagination /
  generative replay** (recombine stored elements into a never-experienced-but-plausible whole), NOT retrieval.
- **Biology.** Catalog **G.09** "Imagination / future simulation as constructive memory" (DMN core: mPFC,
  PCC/precuneus, retrosplenial, lateral parietal/temporal, HC — Schacter/Addis/Buckner; *same network for
  "remember the last beach trip" and "imagine the next one"*; Kandel 6e Ch 52 pp 1300–1302). Sim-status:
  **"missing — sleep-replay can re-run trajectories but no constructive recombination of items into novel
  hypotheticals."** Mechanistically realized by **hippocampal generative replay** that resamples FICTIVE
  sequences *including never-experienced recombinations* (Stoianov-Maisto-Pezzulo 2022; Barry & Love, *Nat.
  Hum. Behav.* 2023 — "a generative model of memory construction and consolidation," whose simulations
  explicitly include *imagination, episodic future thinking, relational inference, and schema-based
  distortions*). Surface-form ordering = catalog **G.07** medial-premotor / pre-SMA internally-generated
  sequences (Sim-status **missing**) — but the project's **neural serial-order renderer** already covers the
  SVO frame (below).
- **What the project ALREADY has (the linchpin).** **`2026-06-23-genfrontier-b2-generative-replay-derisk.md`
  (GO, 6 seeds, D=128):** `GenerativeReplayProposer` (`research/runners/_genfrontier_b2_generative_replay_derisk.py`)
  resamples role-filler bindings from the **PPMI co-occurrence cortex** (the project's learned
  "how-related-are-two-words" signal) to PROPOSE novel SVO triples, GATED by (i) selectional-preference
  plausibility and (ii) non-contradiction against the RF composer's stored facts. **Novel-composition 0.752
  mean (vs the MEASURED 0.0 retrieval baseline, `2026-06-22-generation-novelty-categorical-gap-MEASURED.md`),
  17× the random-recombination floor, shuffled-graph collapses to the floor (the learned structure is
  load-bearing), 0 proposal→known-fact leaks, lesion floods nonsense.** Example invented propositions: `duck
  eat soup`, `cat sleep house`, `bird sing white`. **This IS open-ended generation, brain-based, moat-respecting
  — already proven.** Plus: the **neural serial-order renderer** (`neural_serial_order_renderer.py`,
  `2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`, 6/6 — competitive-queuing spiking word order)
  and the **fluency-only spiking faculty** (`_grounded_lang_integration_derisk.py` — Qwen renders, gate decides
  content, VERIFY re-parses + rejects drift; `2026-06-23-grounded-lang-INTEGRATION-GO.md`).
- **What's MISSING.** (a) a *conversational trigger* — b2 was an offline batch proposer, not "say something
  when the user asks what you think about X"; (b) the proposition→renderer→VERIFY *plumbing wired as one turn*;
  (c) the **graded-confidence flag** on the emitted proposition (b2 keeps proposals as a separate channel but
  doesn't *speak them as flagged hypotheses*); (d) a **fully-spiking** generative-replay sampler (b2's
  resampling loop is host bookkeeping — legitimate for a de-risk, but the brain-based ideal is SWR-gated CA3
  resampling, the project's replay machinery).

### 1.2 INFERENCE BEYOND STORED FACTS / RELATE-A-NOVEL-THING-TO-KNOWN

- **What it computes.** A newly-DESCRIBED entity ("here's a new species: it has fur, it hunts mice") →
  assimilate it to the nearest known category (it's cat-like) → infer un-stated properties from that category
  (it probably sleeps a lot) → form/emit a view. Two sub-mechanisms: **schema assimilation** (place the novel
  item in the existing semantic structure) + **property inference** (generative-replay over the assimilated
  node).
- **Biology.** **Hippocampal/neocortical generative model** (Barry & Love 2023; Stoianov-Pezzulo 2022): "the
  hippocampus retrieves partial cues and cortical networks fill in missing semantic details" — *semantic memory
  develops as a by-product of learning to predict, and the generative model fills in the un-observed.* **Schema
  assimilation** = Eichenbaum–Cohen relational memory / "memory space" (catalog **D.01**/D.02; transitive +
  flexible inference) + Tse 2007 schema reinforcement; the *frontal* schema/abstract-structure coding for rapid
  generalization (Schema-based active inference, arXiv 2601.18946, 2026). **Property inference from category** =
  the generalization the project already validated (a novel item's concept fires in the right category → recall
  a fact about *that category*).
- **What the project ALREADY has.** **The entire `2026-06-16-generalization-*` arc** (controller-verified,
  multi-seed, NO `sim/` edit): a NOVEL object perceived through the real Gabor/V1 front end makes its CONCEPT
  neurons **SPIKE in the correct category** on real LIF spikes (`_genfrontier_capstone_vision_to_concept_derisk.py`,
  cat-acc 0.75 = 3× chance; `2026-06-16-generalization-capstone-vision-to-concept.md`), via **cross-modal Hebbian
  unification** (`_genfrontier_crossmodal_unify_derisk.py`) + **NMDA graded propagation** so the converged
  assembly fires for a held-out novel cue (`_genfrontier_graded_propagation_derisk.py`). The **hybrid verbalize**
  capstone (`2026-06-16-generalization-capstone-verbalize.md`, 0.92 3-seed) keys the validated `RFPhasorComposer`
  recall by the spiking concept-category → recalls the matched category's fact. **⇒ "relate a novel thing to known
  + recall about the category" is DEMONSTRATED end-to-end** for *perceived* novelty. For *described* novelty
  (text, not vision), the comprehension parser + the PPMI cortex supply the "which known concepts is this near"
  signal directly (PPMI is literally a relatedness matrix). Property *inference* then = b2 replay over the
  assimilated node (§1.1).
- **What's MISSING.** (a) the *described-thing* path (assimilate a thing introduced **in dialogue**, not via the
  V1 retina) — needs a "bind the new word to its stated attributes, place it in PPMI space" step (the project has
  the parser + PPMI + cross-modal unify, but they're wired for the vision→concept route, not a dialogue-introduced
  entity); (b) **multi-hop / analogical** inference deeper than 2–3 hops is a MAPPED SNR boundary (below).

### 1.3 ANALOGY / RELATIONAL REASONING (the deeper inference)

- **What it computes.** "X is to Y as A is to ?" / transitive chains / generalizing a *relation* to never-seen
  items. Iterated single-hop pointer-chase for stored chains; a factorised structural code for true relational
  generalization.
- **Biology.** Eichenbaum relational memory (catalog **D.02**; Dusek & Eichenbaum 1997 — HC required for the
  *inference*, not for premise-pair learning); **TEM** (Tolman-Eichenbaum Machine, Whittington 2020; *Spiking*
  TEM, bioRxiv 2025.10.16.682754) for relation-as-reusable-transition-operator generalization.
- **What the project ALREADY has.** **Multi-hop reasoning is PRODUCTION** — `RFPhasorComposer.query_chain` /
  `BrainConversationalAgent.reason_chain` (`2026-06-17-multihop-query-chain-GO.md`, unanimous 3-seed×3-D GO; the
  cleanup re-discretizes between hops so error doesn't compound; holds to 4 hops; moat at every hop). The scoping
  `2026-06-17-multihop-reasoning-multiturn-dialogue-scoping.md` maps the exact boundary.
- **What's MISSING.** True relational GENERALIZATION to never-seen items = **Option 4 (TEM-style factorised
  structural code)** — a months-scale research build, the strategic end-state, NOT cheap-first. Pointer-chase
  over stored facts is done; the deep version is deferred.

### 1.4 PHATIC / SOCIAL ("hi" → "hello"; backchannels; "how are you")

- **What it computes.** A learned, near-reflexive stimulus→response map for social tokens that carry little
  propositional content — orthogonal to the no-confab moat (these aren't *facts*, so abstention is the wrong
  posture).
- **Biology.** Overlearned associative responses (cortico-striatal habit; the project's Tier-1 word↔action
  binding is exactly this shape) + an affective/social register. Catalog has no dedicated "phatic" entry; it's a
  trivial special case of associative recall.
- **What the project ALREADY has.** The Tier-1/embodied-Hebbian word↔response binding (6/6 multi-seed) IS a
  stimulus→response map; the `BrainConversationalAgent` surface can carry a small "social register" of learned
  greeting↔greeting / question↔acknowledgement pairs. The conversational classifier (which the parser/comprehension
  front-end already approximates) routes a phatic input to the social register instead of the fact store.
- **What's MISSING.** A tiny **intent/register router** — "is this a fact-query, an opinion-ask, a phatic
  token, or a description-of-a-new-thing?" — so the agent picks {abstain-or-answer, generate-a-view,
  phatic-reflex, assimilate-and-infer}. This is a low-risk classification step (the parser's firing already
  distinguishes sentence types; extend it to a 3–4-way register tag). The *content* of a phatic reply is a
  learned association, never the LLM's invention.

### 1.5 THE GRADED-CONFIDENCE MOAT-RELAX (the cross-cutting requirement) — and the VALUE/SALIENCE read-out

- **What it computes.** Two coupled things the current architecture lacks:
  1. **Graded confidence** — instead of {assert known fact | hard-abstain}, the agent has a third posture:
     *"I'd guess / it seems like / probably …"* — speak a proposition while FLAGGING it as a low-confidence
     hypothesis. The confidence is the **plausibility score** the brain already computes (the PPMI/replay
     likelihood) mapped to a calibrated linguistic hedge.
  2. **Value / salience appraisal** — "what is *worth* saying" / "do I even have a view on X." A loose
     conversationalist doesn't emit every plausible proposition; it picks the *salient/relevant* one and knows
     when it has nothing. This is the **limbic/value** judgment the owner flagged ("what do you think" = a
     value/salience judgment).
- **Biology.** **Graded confidence** = the brain's familiarity/uncertainty signal (the Bogacz-Brown familiarity
  gate the moat already uses — `2026-06-11-familiarity-gate-v320-GO.md` — read as a *graded* value, not a hard
  threshold). **Value/salience** = the **DMN+FPCN+Salience-network triad** for self-generated thought (the
  *salience network* — anterior insula + ACC — "detects salient internal events" and "activates DLPFC which
  selectively increases weights of task-relevant attributes"; controlled semantic retrieval is FPCN×DMN coupling
  — *Communications Biology* 2025 s42003-025-07470-9; *Comms Bio* 2024 06405-0). Neurally: catalog **C.23**
  "Heterogeneous DA subpopulations — reward, aversion, SALIENCE" (Sim-status **missing**), **C.24** DA-as-salience
  (Sim-status **partial**), and the project's own **TD value critic** (`sim/td_value_critic.py`) +
  predictive-coding **prediction-error-as-salience**.
- **What the project ALREADY has.** (a) the **plausibility/familiarity scores** are computed (PPMI relatedness,
  the Bogacz-Brown gate, the b2 plausibility gate) — the raw material for graded confidence is *already there*,
  just thresholded into a hard moat; (b) **`sim/td_value_critic.py`** (canonical TD(λ) value critic, catalog
  C.30) + **`sim/predictive_coding.py`** (Rao-Ballard prediction error) + the **limbic/DA value system** are
  built pieces for a value read-out; (c) the b2 finding already exercises the owner-sanctioned moat/generativity
  trade (proposals are a *separate flagged hypothesis channel* — the moat-relax pattern in embryo).
- **What's MISSING (the genuine deep new-mechanism).** A **spiking VALUE/SALIENCE read-out** that takes a
  candidate proposition (or a topic) and returns *how worth-saying / how confident* it is — a neural appraisal,
  not a host formula. This is the one piece with **no validated substrate** for the *conversational* use; the
  ingredients (TD critic, prediction-error, familiarity gate, DA-salience) exist but have never been wired as
  "appraise a candidate utterance." **Plus** the calibration of plausibility→linguistic-hedge (a calibration
  experiment, cheap).

### 1.6 Diagnosis summary table

| Mechanism-class | Biology (catalog/Kandel/paper) | Project ALREADY has (reusable) | Missing |
|---|---|---|---|
| **Open-ended generation** | G.09 imagination (missing); Stoianov-Pezzulo / Barry-Love generative replay | **b2 generative-replay GO (17× chance, moat-safe)**; neural serial-order renderer (GO); fluency-only Qwen + gate→VERIFY (GO) | conversational TRIGGER; turn plumbing; graded-confidence FLAG; (opt) fully-spiking SWR sampler |
| **Inference / relate-novel-to-known** | Barry-Love hippocampal generative model; D.01/D.02 relational; schema active inference (arXiv 2601.18946) | **generalization arc GO** (novel→concept fires in right category, real spikes); hybrid verbalize 0.92; PPMI relatedness; cross-modal unify | the *described-in-dialogue* entity path (vs vision route) |
| **Analogy / deep relational** | D.02 Eichenbaum; TEM / Spiking-TEM | **multi-hop query_chain PRODUCTION** (2–4 hops, moat at each) | true relation-generalization to novel items = **TEM (months-scale, deferred)** |
| **Phatic / social** | overlearned cortico-striatal association | Tier-1 word↔response binding (6/6); agent surface | a small **intent/register router** (3–4-way) |
| **Graded-confidence + value/salience** | familiarity/uncertainty signal; DMN+FPCN+**Salience-net** triad (Comms Bio 2024/2025); C.23/C.24 DA-salience (missing/partial) | plausibility/familiarity scores (PPMI, Bogacz-Brown gate, b2 gate); `td_value_critic`; `predictive_coding`; limbic/DA | **spiking VALUE/SALIENCE appraisal read-out (the deep new piece)**; plausibility→hedge calibration |

**Net:** four of five classes are an INTEGRATION + small-router job over proven pieces; the genuine new-mechanism
is the value/salience appraisal (§1.5), and the genuine deep-frontier is TEM relational generalization (§1.3,
deferred).

---

## 2. RANKED biologically-grounded options (cheapest-first)

Each option states the biology citation, the reusable project machinery, and how it stays
single-brain / single-substrate / LLM-fluency-only.

### Option 1 (RECOMMENDED, cheapest) — Wire the *existing* b2 generative-replay proposer into a conversational turn, flagged as a graded-confidence hypothesis, rendered by the fluency-only faculty + VERIFY

- **What.** Trigger b2's `GenerativeReplayProposer` on an opinion-ask / open-ended turn ("what do you think
  about ducks?"): seed the replay with the topic concept → propose a novel-but-plausible graph-supported
  proposition → emit it through the **neural serial-order renderer → fluency-only Qwen → gate→constrain→VERIFY**,
  prefixed with a **graded-confidence hedge** derived from the proposition's plausibility score ("I'd guess a
  duck might…"). Known facts still go through the normal retrieve+assert path; only the *opinion/open-ended*
  register invokes replay.
- **Biology.** G.09 constructive imagination; Stoianov-Pezzulo 2022 / Barry-Love 2023 generative replay
  (*resamples fictive sequences including never-experienced recombinations*). The hedge = the brain's graded
  familiarity/uncertainty signal (Bogacz-Brown gate read as graded).
- **Reusable machinery.** `_genfrontier_b2_generative_replay_derisk.py` (the proposer + PPMI plausibility + the
  moat-safe accept gate — GO 6-seed); `neural_serial_order_renderer.py` (GO); `_grounded_lang_integration_derisk.py`
  (gate→constrain→VERIFY + reject→regenerate); `rf_phasor_composer.py` / `one_brain_composer.py` (the fact store
  the proposer must not contradict, + the moat); `brain_conversational_agent.py` (the surface to extend).
- **Single-brain / fluency-only?** The PROPOSITION (the content + the choice) is the BRAIN's (PPMI cortex +
  composer); the LLM only renders the agreed proposition into a sentence and VERIFY re-parses it (the proven
  hallucination-catch). The moat is RELAXED to graded-confidence (a flagged hypothesis channel — the
  owner-sanctioned trade, already exercised in b2: 0 leaks to the known-fact channel). **This is ~the b2 GO +
  the grounded-lang GO composed** — the lowest-new-code, highest-confidence route.
- **Honest scope.** b2's proposer loop is currently host bookkeeping (legitimate de-risk harness; the
  load-bearing plausibility + store + moat are the brain's). A fully-spiking SWR sampler is the follow-on
  (Option 3). The graded-confidence calibration (plausibility→hedge) is a small new experiment.

### Option 2 (cheap) — A neural INTENT/REGISTER router + a phatic/social register

- **What.** Extend the comprehension parser to emit a 3–4-way **register tag** per input: {fact-query,
  opinion/open-ended, phatic, describe-new-thing}. Route each to {retrieve+assert-or-abstain, Option 1 replay,
  learned phatic reflex, Option 4 assimilate-then-infer}. The phatic register is a small learned
  stimulus→response association table (greeting↔greeting, ack↔ack).
- **Biology.** Overlearned cortico-striatal habit (the project's Tier-1 binding shape); sentence-type
  classification is already implicit in the parser's role-firing pattern.
- **Reusable machinery.** `BridgeParser` / the comprehension front-end (the parser's firing already distinguishes
  sentence types — extend to a register tag); Tier-1 word↔response binding for the phatic table;
  `BrainConversationalAgent.hear` dispatch.
- **Single-brain / fluency-only?** Routing = "which assembly fired" (legitimate, not cognition); the phatic
  reply content is a learned association, never the LLM's. Single substrate (the parser slice).
- **Honest scope.** Low-risk classification; the only subtlety is calibrating the register boundary so an
  opinion-ask isn't mis-routed to hard-abstain (the current failure the owner named).

### Option 3 (moderate) — Fully-spiking generative-replay sampler (SWR-gated CA3 resampling)

- **What.** Replace b2's host resampling loop with a **spiking** sampler: SWR-gated CA3 bursts (the project's
  replay machinery) resample role-filler bindings on the substrate, biased by the learned recurrent
  (PPMI-shaped) weights; the engram/replay gates select the proposal. Closes the "the recombination bookkeeping
  is host code" residual in Option 1.
- **Biology.** Catalog **D.19/N.07/N.17** sharp-wave ripples + **awake replay during behavioral pauses** (N.17 —
  "~50% of SWRs occur during waking immobility … forward replay of candidate trajectories, deliberative-planning
  role"); G.09 constructive imagination realized via replay. This is *exactly* the catalog's "replay influences
  online behavior, not just offline weight changes" higher-value addition.
- **Reusable machinery.** `consolidation_trainer.run_swr_replay_phase` / `run_concept_replay_phase`; the
  awake/sleep gates (`text_minimal_isolation.set_awake_gates`/`set_sleep_gates`); engram tag/stimulate
  (`bridge.start_engram_recording`/`stimulate_tag`); `learned_assoc_graph.LearnedAssocGraph` (Hebbian recurrent
  that learns concept→concept edges — the substrate likelihood the sampler walks).
- **Single-brain / fluency-only?** Fully spiking on the one bridge; the proposal *emerges* from CA3 dynamics.
  Strongest brain-based purity. The LLM still only renders.
- **Honest scope.** Higher build cost; the genuine open question is whether SWR resampling on the point-neuron
  substrate produces *plausibility-biased* (not uniform) recombinations as cleanly as the host PPMI sampler —
  a real de-risk. Do AFTER Option 1 proves the conversational composition.

### Option 4 (moderate) — Assimilate-a-described-thing-then-infer (the dialogue inference path)

- **What.** When a new entity is introduced in dialogue ("a new species: fur, hunts mice"), bind the new word
  to its stated attributes, place it in PPMI/concept space near the nearest known category (cat-like), then run
  Option 1 replay over the assimilated node to infer + emit un-stated properties as flagged hypotheses.
- **Biology.** Barry-Love 2023 ("cortical networks fill in missing semantic details"); schema-based active
  inference + frontal abstract-structure coding for rapid generalization (arXiv 2601.18946); Tse 2007 schema
  reinforcement; the generalization arc's cross-modal Hebbian convergence (ATL hub-and-spoke,
  Patterson-Lambon Ralph).
- **Reusable machinery.** The **generalization arc** (`2026-06-16-generalization-*`: cross-modal unify + NMDA
  graded propagation + the hybrid verbalize — already GO for *perceived* novelty); the parser (to bind the
  stated attributes); PPMI relatedness (to find the nearest category); Option 1's replay (to infer properties);
  V_SCHEMA in-vivo binding (`2026-05-12-V_SCHEMA-*`, anchor reinforcement).
- **Single-brain / fluency-only?** The assimilation is Hebbian convergence on the substrate; the category match
  is "which concept assembly fired"; the inference is replay (brain). The LLM renders. Single bridge.
- **Honest scope.** The generalization arc proved this for *vision*-introduced novelty; the *dialogue*-introduced
  path reuses the same convergence machinery but needs the "bind stated attributes → place in PPMI space" front
  step de-risked. Moderate.

### Option 5 (the deep new-mechanism) — Spiking VALUE/SALIENCE appraisal read-out ("is this worth saying / do I have a view")

- **What.** A neural read-out that takes a candidate proposition (or a topic) and returns a *graded
  worth-saying / confidence* value, so the agent (a) picks the *salient* proposition among many plausible ones,
  (b) knows when it has *nothing* worth saying (graceful "I don't really have a view"), and (c) sets the
  confidence hedge. Built from the existing value/prediction-error pieces, wired to appraise an utterance.
- **Biology.** DMN+FPCN+**Salience-network** triad for self-generated thought (the salience net detects salient
  internal events + activates DLPFC to weight relevant attributes — *Comms Biol* 2025; controlled semantic
  retrieval = FPCN×DMN, *Comms Biol* 2024); catalog **C.23/C.24** DA-as-salience (missing/partial); TD value
  critic (C.30); Rao-Ballard prediction-error-as-novelty/salience (Friston active inference).
- **Reusable machinery.** `sim/td_value_critic.py` (TD(λ) critic), `sim/predictive_coding.py` (prediction error),
  the Bogacz-Brown familiarity gate (`2026-06-11-familiarity-gate-v320-GO.md`), the limbic/DA neuromodulator
  subsystem (`sim/neuromodulators.py`), the b2 plausibility gate. The owner's standing directive
  (`feedback_move_everything_to_shared_spiking_substrate`) flags the **limbic reward/value core as the
  highest-leverage shared system** — this is that build, scoped to conversation.
- **Single-brain / fluency-only?** A spiking appraisal on the one bridge (the missing piece that makes
  "what is worth saying" the BRAIN's judgment, not a host formula). The LLM never appraises.
- **Honest scope.** This is the genuinely-new mechanism (no current conversational-appraisal substrate) and the
  one with real research risk — but it's also the deepest payoff (graceful degradation + "do I have a view" +
  calibrated confidence all flow from it). Recommended to scope its OWN cheap-first probe AFTER Option 1, and
  to lean on the owner's limbic-core directive (it's wanted independently).

### Option 6 (deferred end-state) — TEM-style factorised relational code (true analogy/relation generalization)

- **What.** A learned structural code so a *relation* is a reusable transition operator that generalizes to
  never-seen items (true analogy, not stored-fact chaining).
- **Biology.** Whittington 2020 TEM; *Spiking* TEM (bioRxiv 2025.10.16.682754); Eichenbaum D.02.
- **Reusable machinery.** `query_chain` (the pointer-chase floor); the place/grid navigation code (TEM unifies
  spatial + relational).
- **Single-brain?** In principle yes (spiking TEM exists), but **months-scale**, comparable to the deferred
  dendritic rewrite. **NOT cheap-first; the strategic end-state**, flagged for the roadmap if Options 1–5 hit a
  relational-generalization ceiling.

---

## 3. RECOMMENDED cheap-first de-risk (the smallest falsification probe)

**Probe the COMPOSITION, not a new mechanism** — because the load-bearing pieces (b2 generation, the renderer,
the gate→VERIFY loop) are individually GO; the open question is whether they compose into a *conversational
turn* that is novel + grounded + flagged + lesion-collapsible. Run **CPU/numpy** (`SIM_BACKEND=numpy`) — the b2
proposer + the composer numpy fast path are the right cheap instruments; promote to GPU + 6 seeds only on GO.

**The probe (a "what do you think about X" turn):**
1. Build a small grounded brain: PPMI cortex over the real TinyStories taxonomy + the RF composer's stored facts
   (the b2 testbed, reused verbatim) + a handful of NEGATED facts (so the non-contradiction gate has work).
2. For a held-out topic concept X (e.g. `duck`), run the turn:
   `assimilate(X) → propose ← GenerativeReplayProposer(seed=X) → render ← serial-order + fluency-only LLM →
   VERIFY(reparse == proposed triple, graph-supported, non-contradictory) → emit with a graded-confidence hedge
   derived from the proposition's plausibility score.`
3. Measure, against pre-registered gates + the controls below.

**Pre-registered GO / BOUNDARY / NEGATIVE (multi-seed ≥3 for the cheap probe, 6 for the build gate):**
- **GO:** the turn emits a NOVEL proposition (distinct-from-stored ratio > 1.0, novel-composition > 0 — beats the
  measured 0.0 retrieval baseline) that is GROUNDED (graph-plausible, beats the shuffled-graph control by the
  b2-style ≥3× ratio), FLAGGED (emitted with a hedge, NEVER passes the known-fact retrieval channel — 0 leaks),
  COHERENT (VERIFY accepts a re-parseable SVO; the conservative failure is over-abstention, never confabulation),
  AND the confidence hedge tracks the plausibility score (high-plausibility → assertive hedge, low → tentative).
- **BOUNDARY (still publishable):** the turn generates + flags correctly but the **graded-confidence calibration**
  is poor (the hedge doesn't track plausibility, or it over-/under-hedges) → maps the calibration gap precisely;
  OR the *described-in-dialogue* assimilation (Option 4 front step) is the weak link while the *generation*
  composes fine → localizes the remaining build to the assimilation step.
- **NEGATIVE:** the composition loses the b2 property (novel-composition ≈ 0 when wired as a turn — e.g. the
  renderer/VERIFY round-trip strips the novelty, or the LLM injects its own content that VERIFY can't bound) →
  the integration is the wall, and the recommendation shifts to hardening the gate→VERIFY contract before any
  surface-form work.

**Why this is the right cheap-first:** it composes only GO pieces, runs in minutes on CPU, and its NO-GO/BOUNDARY
outcomes each route to a *specific, bounded* next step (Option 3 fully-spiking sampler, Option 4 dialogue
assimilation, Option 5 value/salience, or the gate→VERIFY contract) — converting "is loose grounded conversation
feasible?" into an afternoon's measurement, per the cheapest-first gate discipline.

---

## 4. ANTI-CHEATS (load-bearing — the generation/opinion must be the BRAIN, not the LLM; flagged not hallucinated; single-substrate)

These are mandatory; the b2 and grounded-lang findings already established the templates.

1. **LESION / provenance (the decisive one — generation is the BRAIN, not the LLM).** Sever the brain's
   proposal (feed the fluency-only LLM the topic with NO brain-proposed proposition, or a RANDOM proposition):
   the output must collapse to (a) the LLM free-hallucinating (which VERIFY then REJECTS → no emission) or (b)
   nonsense. If the turn still produces a sensible grounded reply with the brain's proposal severed, the LLM is
   doing the cognition — FAIL. (Mirrors b2's lesion: gate ON → 100% plausible, gate OFF → 12%.) Provenance:
   every emitted proposition must trace to a brain-proposed triple, logged.
2. **Shuffled-graph control (the content is grounded, not a template artifact).** Permute the off-diagonal PPMI
   entries (preserve marginals, destroy neighborhoods): the turn's plausibility must collapse to the random floor
   (b2: 0.328 → 0.018). If novelty survives the shuffle, it's a string/template artifact, not grounded generation.
3. **Graded-confidence CALIBRATION (flagged, not hallucinated).** A reliability-diagram test: bin emitted
   propositions by their stated confidence hedge; the *empirical* graph-plausibility within each bin must track
   the hedge (high-confidence emissions are graph-strong; low-confidence ones are graph-weak). A flat curve =
   the hedge is decorative (the moat-relax is uncalibrated → over-claiming). This is the test that the moat
   *relaxed to graded-confidence* rather than *removed*.
4. **Known-fact channel integrity (the moat is preserved where it should be).** A proposed/generated proposition
   must NEVER pass the composer's KNOWN-fact retrieval (`query_patient`/`ask_yes_no` still abstain on it — b2: 0
   leaks); an explicitly-NEGATED fact must never be re-proposed (b2: 0 negated re-proposed). The agent
   distinguishes "I know X" (assert) from "X is plausible" (flagged hypothesis) from "I was told NOT-X" (refuse).
5. **VERIFY-catches-drift (the LLM can't smuggle content).** Steer the fluency-only LLM toward a fact the brain
   did NOT propose; the re-parse must REJECT it (the grounded-lang GO already demonstrated this — the 0.5B LLM
   inverted a role, VERIFY caught it). The conservative failure direction must be over-abstention.
6. **Single-substrate / brain-based.** The proposal + the value/relevance choice + the plausibility live on the
   one `SimulationBridge` (PPMI cortex, composer, replay); host code only routes "which assembly fired" + the
   de-risk bookkeeping. The fluency-only LLM is OFF-bridge during de-risk (as the grounded-lang faculty is); the
   bridge co-residence of the LLM is the separate consolidation arc, not this scoping.
7. **Multi-seed ≥6 for any variable claim** (`feedback_6seed_validation`); CuPy for decisive runs, numpy only for
   the cheap probe (`feedback_gpu_not_numpy`); frozen gates pre-registered before seeing held-out data.

---

## 5. HONEST VERDICT — compose-proven-pieces vs genuine-new-mechanism vs moat-relax-design

| Capability | Classification | Basis |
|---|---|---|
| **Open-ended generation** (novel grounded proposition) | **COMPOSE proven pieces** | b2 generative-replay GO (17× chance, moat-safe) + neural serial-order renderer GO + gate→VERIFY GO. The mechanism EXISTS and is proven; the work is wiring it into a *turn* + the flag. |
| **Inference / relate-novel (perceived)** | **COMPOSE proven pieces** | generalization arc GO end-to-end (novel→concept fires in right category on spikes; hybrid verbalize 0.92). |
| **Inference / relate-novel (described-in-dialogue)** | **mostly compose + one small new front-step** | reuses cross-modal unify + PPMI + replay; needs the "bind stated attributes → place in PPMI" assimilation step de-risked (Option 4). |
| **Phatic/social** | **trivial compose + a small router** | Tier-1 binding shape; the new piece is a 3–4-way intent/register tag (Option 2). |
| **Graded-confidence moat-relax** | **MOAT-RELAX DESIGN + calibration** | the plausibility/familiarity scores EXIST (PPMI, Bogacz-Brown, b2 gate); the work is the *design choice* (how much to relax — a values call) + a calibration experiment. The b2 finding already exercised the trade. |
| **Value/salience "what is worth saying / do I have a view"** | **GENUINE DEEP NEW-MECHANISM** | no validated conversational-appraisal substrate; ingredients exist (td_value_critic, predictive_coding, familiarity gate, DA-salience C.23/C.24 missing/partial) but never wired as utterance-appraisal. The one real new build (Option 5). |
| **Deep analogy / relation-generalization to novel items** | **GENUINE DEEP-FRONTIER (deferred)** | TEM / Spiking-TEM — months-scale, the strategic end-state (Option 6), comparable to the dendritic rewrite. Pointer-chase (2–4 hops) is already PRODUCTION. |

**Bottom line.** The owner's "loose, open-ended, grounded-but-not-restricted" faculty is **achievable now,
mostly by INTEGRATION** — the brain's generative-replay (b2, GO), the generalization arc (GO), the neural
renderer (GO), and the gate→constrain→VERIFY fluency-only loop (GO) compose into it, with a small intent router
and the moat-relaxed-to-graded-confidence design. **Two pieces are genuinely new:** (a) the moat-relax is a
*design + calibration* job (cheap, but a values call the owner must set), and (b) the **spiking value/salience
appraisal** is the one real new mechanism (Option 5) — and it aligns with the owner's standing
"limbic-reward/value-core is highest-leverage" directive, so it's wanted independently. **One piece is a
deferred deep-frontier** (TEM relational generalization). The recommended first move is the
**Option-1 composition probe** (an afternoon, CPU) — if GO, it demonstrates loose grounded conversation
end-to-end on proven machinery; its BOUNDARY/NEGATIVE outcomes each localize the next bounded build.

### Owner-steer flags

1. **[STEER — the values call] How much does the moat relax?** Graded-confidence is a *posture* choice, not a
   science result: how tentative may the agent be, and at what plausibility does it speak-vs-stay-silent? The
   owner's `feedback_moat_not_hard_lossy_memory_ok` licenses the relax; the *threshold* is the owner's to set
   (the calibration probe measures the trade, but the operating point is a preference).
2. **[STEER — sequencing] Build the value/salience appraisal (Option 5) now or after?** It's the deepest payoff
   (graceful "I don't have a view" + calibrated confidence) AND aligns with the owner's limbic-core directive,
   but it's the one real research-risk piece. Recommend: Option-1 probe first (proves the generation
   composition), then scope Option 5's own cheap-first probe.
3. **[STEER — biology purity vs speed] Host-loop b2 sampler (Option 1, fast) vs fully-spiking SWR sampler
   (Option 3, pure)?** Option 1's proposer loop is host bookkeeping (legitimate de-risk; the load-bearing
   plausibility + store + moat are the brain's). The fully-spiking SWR sampler is the brain-based ideal but a
   real de-risk. Recommend Option 1 to prove the turn, Option 3 as the purity follow-on.
4. **[STEER — scope of "described-thing" inference]** Is the *dialogue-introduced* novel entity (Option 4) in
   the near-term scope, or is the *perceived* novelty (already GO) enough for the first demo? The dialogue path
   is a small extra de-risk (the assimilation front-step).
5. **[VERIFY — load-bearing]** Confirm the b2 proposer + the neural serial-order renderer + the grounded-lang
   gate→VERIFY loop are *independently* still GREEN on the current `main` (this scoping read the findings, not
   re-ran them) before building the composition — the recommendation rests on all three being GO.

---

## Sources

### Project record (re-verified this pass)
- **`research/findings/2026-06-23-genfrontier-b2-generative-replay-derisk.md`** (the linchpin — brain invents
  novel-but-plausible propositions, 17× chance, moat-safe, GO 6-seed) + `research/runners/_genfrontier_b2_generative_replay_derisk.py`.
- `research/findings/2026-06-22-generation-novelty-categorical-gap-MEASURED.md` (the 0.0 retrieval baseline this attacks).
- `research/findings/2026-06-23-grounded-lang-INTEGRATION-GO.md` (gate→constrain→VERIFY; LLM fluency-only; hallucination caught) + `_grounded_lang_integration_derisk.py`, `_grounded_lang_p{1b,2,3}_*`.
- `research/findings/2026-06-16-generalization-capstone-{vision-to-concept,verbalize}.md`, `-crossmodal-unify-cheap-first.md`, `-graded-propagation.md`, `-onsubstrate-convergence.md`, `-optionB-visual-similarity.md` (relate-novel-to-known, GO) + the `_genfrontier_*` runners.
- `research/findings/2026-06-16-sentence-generation-serial-order-cheap-first-GO.md` + `research/runners/neural_serial_order_renderer.py` (neural word order, GO).
- `research/findings/2026-06-17-multihop-query-chain-GO.md` + `-multihop-reasoning-multiturn-dialogue-scoping.md` (pointer-chase 2–4 hops PRODUCTION; TEM as the deferred end-state).
- `research/findings/2026-06-22-generative-sequence-frontier-scoping.md` (the adjacent BPTT-SNN free-token-generation frontier — distinct from this faculty).
- `research/findings/2026-06-03-deep-research-how-the-field-gets-past-our-generative-conversation-wall.md` (VSA is associative-not-generative; the field's hybrid remedy).
- `research/findings/2026-06-11-familiarity-gate-v320-GO.md` (the Bogacz-Brown familiarity gate = the graded-confidence raw signal).
- `sim/predictive_coding.py` (Rao-Ballard PE), `sim/td_value_critic.py` (TD(λ) critic, C.30), `sim/song_hvc.py` (HVC sequencer), `sim/neuromodulators.py` (limbic/DA), `research/runners/learned_assoc_graph.py`, `content_selection_spiking.py`, `rf_phasor_composer.py`, `one_brain_composer.py`, `brain_conversational_agent.py`.
- `sim-catalog/references/feature-catalog.md`: **G.09** (imagination/future simulation — Sim-status MISSING), **G.07** (medial-premotor internally-generated sequences — MISSING), **D.19/N.07/N.17** (SWR replay incl. awake-replay/deliberation — partial/missing), **D.01/D.02/D.13** (episodic/relational/pattern-completion), **C.22** (RPE — partial), **C.23** (heterogeneous DA incl. SALIENCE — MISSING), **C.24** (DA-as-salience — partial), **G.02** (active dendrites — MISSING). Cluster legend: G=working memory/PFC/cortical integration, D=hippocampus/sequence, C=dopamine/neuromodulation, O=emotion/reward/motivation, N=sleep/arousal.

### Current literature (June 2026 pass)
- **Generative replay / hippocampal generative model:** Stoianov, Maisto & Pezzulo 2022, *Prog. Neurobiol.* (hippocampal formation as a hierarchical generative model supporting generative replay + continual learning, S0301008222001150); **Barry & Love 2023/2024, *Nat. Hum. Behav.*** "A generative model of memory construction and consolidation" (PMID 38242925 — simulates semantic memory, *imagination, episodic future thinking, relational inference, schema-based distortions*).
- **Schema assimilation / rapid generalization:** "Schema-based active inference supports rapid generalization … frontal cortical coding of abstract structure," arXiv 2601.18946 (2026); GENESIS generative model of episodic-semantic interaction, arXiv 2510.15828.
- **Open-ended generation = DMN+FPCN+Salience triad:** *Communications Biology* 2025, "Dynamic switching between brain networks predicts creative ability" (s42003-025-07470-9); *Comms Biol* 2024, "Semantic associative abilities and executive control … novelty and appropriateness of idea generation" (06405-0); the salience network (anterior insula + ACC) detects salient internal events + activates DLPFC to weight relevant attributes. Prefrontal semantic control for weakly-associated meanings: ScienceDirect S0304394023001349.
- **Spontaneous thought / future thinking dynamics:** *Nat. Commun.* 2025, "Neural dynamics of spontaneous memory recall and future thinking in the continuous flow of thoughts" (s41467-025-61807-w).
- **Relational inference / TEM (deferred end-state):** Whittington et al. 2020, *Cell* (Tolman-Eichenbaum Machine); *The Spiking Tolman-Eichenbaum Machine*, bioRxiv 2025.10.16.682754; Dusek & Eichenbaum 1997 (HC required for inference, not premise-pair learning).
- **VSA is associative-not-generative (the wall this faculty routes around):** "Bridging Cognitive Architectures and Generative Models with VSA," AAAI-SS; Kleyko et al. 2023 HDC/VSA survey; Frady & Sommer 2019.
