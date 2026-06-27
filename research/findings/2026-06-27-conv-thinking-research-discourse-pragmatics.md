# Conversation + Thinking research — DISCOURSE · REFERENCE · PRAGMATICS · THEORY-OF-MIND

**Date:** 2026-06-27
**Type:** read-only deep-research + reference-catalog review (this doc is the only write). NO `sim/` edit, NO code.
**Scope:** one capability cluster of the conversation-and-thinking frontier — the *discourse* layer that sits ABOVE
the sentence: reference resolution & definiteness, turn-taking & dialogue management, Gricean implicature,
clarification/repair, common ground, intent & speech-act recognition, and theory-of-mind.
**Trigger:** the owner's communicable-brain north-star ([[project_communicable_brain_not_rag]]) + the deep-knowledge
build ([[project_deep_knowledge_brain_fluency_build]]). Per the standing research gate, this reviews the canonical
biology (catalog clusters D/G, Kandel 6e) + current literature BEFORE any build is committed.

**Sources used.** (1) `sim-catalog/references/feature-catalog.md` — clusters **D** (Hippocampus & sequence learning,
20 entries) and **G** (Working memory / PFC / cortical integration, 20 entries); the index has **no** ToM/discourse
cluster — a real coverage gap, noted below. (2) **Kandel 6e** `references/textbooks/kandel-pns-6e/full-book.txt`
(searched; ToM in Ch 56 + Ch 62/64, language in Ch 55, decision in Ch 56). (3) **glossary.md is ABSENT** from the
catalog references dir (only `feature-catalog.md` + `biology-buildout-roadmap.md` + `textbooks/`) — substituted with
WebSearch + literature + project knowledge. (4) Literature via WebSearch + recall (landmark reviews + meta-analyses).
(5) The sim's current state read directly from `research/runners/{rf_phasor_composer,one_brain_composer,
brain_conversational_agent,multi_turn_agent,biased_competition_buffer,learned_assoc_graph}.py` + the
2026-06-17/19/21 multi-turn/multi-referent/dialogue-graph findings.

---

## 0. TL;DR — the cluster in one screen

The sim is a **strong single-sentence comprehender/answerer with NO discourse-entity model**. It learns concept
*codes*, stores **bare (agent, verb, patient) SVO triples over concept TYPES**, abstains when nothing matches, and
carries **one** discourse referent across a turn via a WM attractor. The whole discourse layer — *which* boy, asking
"which boy?", modelling what the interlocutor knows, turn-taking, implicature — is **absent or host-scaffolded**.

The decisive biological insight from the review: **the entire discourse/social stack runs on ONE network** — the
**mentalizing network (rTPJ + dmPFC + precuneus + posterior STS) ≈ the DMN dorsomedial subsystem**, sitting at the
**apex of a hierarchy of temporal-receptive-windows** (Hasson) — fed by **hippocampal relational binding** for
entity/common-ground tracking and the **ATL hub** for person/social concepts. Two computational primitives recur and
**compose**: (a) **generative-model + prediction-error** (rTPJ codes mental-state PE; turn-end prediction;
speaker–listener anticipatory coupling; the proposed repair trigger), and (b) **recursive Bayesian agent modelling**
(BToM inverse planning; RSA implicature; relevance-theoretic ostension).

**The single highest-leverage gap** is the missing **discourse-referent / entity-instance layer** (a *token*
"boy#1" distinct from the *type* "boy"). Everything else in this cluster — "which boy?", definite-vs-indefinite,
multi-referent disambiguation, common ground, anaphora at scale — is downstream of it. And it has a clean biological
home (hippocampal engram-per-instance) + reuses machinery the project already has (engram-tagging API, the
biased-competition WTA buffer, the no-confab abstention moat as the clarification trigger).

---

## 1. Reference resolution & definiteness ("the boy" / "which boy?")

### (a) Human capability + concrete example
A competent speaker distinguishes **types** from **instances** and tracks **discourse referents** (the specific
entities under discussion). The hallmarks:
- **Indefinite** introduces a *new* referent: "A boy walked in." → allocate a new entity token.
- **Definite / pronoun** *presupposes* an existing, identifiable referent: "**the** boy" / "**he**" → bind to an
  already-introduced token. If none is uniquely identifiable, the reference *fails*.
- **Clarification on failure:** when a definite cannot be resolved (no matching referent, or **several** match), the
  competent move is to **ask** — "**which** boy?" — rather than silently guess or stay mute.
- **Bridging:** "I bought a car. **The engine** is great." — *the engine* is licensed by an inference from the car.

**The owner's "which boy" case is exactly a referent-identifiability failure on a DEFINITE.** Our sim cannot produce
it because it has no entity tokens to be ambiguous about.

### (b) Biological mechanism
- **The ERP signature of referential processing is the Nref (referential negativity):** a sustained frontal (often
  left) negativity when a noun phrase / pronoun is **referentially ambiguous** (>1 candidate antecedent: "David told
  Paul that *he*…"). It is graded by the *number* of candidate referents and scales with **working-memory span** —
  high-span readers show larger Nref. It is **distinct** from the N400 (semantic-fit) and P600 (structural
  reanalysis). Referential *failure* (no antecedent) vs *ambiguity* (too many) vs *success* dissociate. *Citations:*
  **Nieuwland & Van Berkum 2006, *J. Cogn. Neurosci.*; Van Berkum, Koornneef, Otten & Nieuwland 2007, *Brain
  Research* 1146:158–171** (the review of ERP reference effects).
- **Entity binding is hippocampal/relational** — the discourse referent is bound and indexed by the MTL relational
  system (see §4). Frontal cortex (the Nref generator, consistent with WM) holds the active candidate set;
  hippocampus/MTL binds referents into the situation model; given/new + novelty processing engages hippocampus + VTA
  + dlPFC.
- **Catalog grounding:** **D.02 Relational binding / "memory space"** (Eichenbaum–Cohen; *Kandel 6e Ch 52 pp
  1301–1302*) — "items-in-context, episodes as sequences" — is exactly a discourse-referent store. **D.14 Engram
  cells** (Tonegawa; *Kandel Ch 54 pp 1357–1359*) — a *sparse activity-tagged ensemble that stores a specific
  instance and can be reactivated as a unit* — is the natural substrate for **one engram per entity token**.
- **Formal-linguistic model with a clean neural mapping:** **Discourse Representation Theory** (Kamp 1981) / **File
  Change Semantics** (Heim 1982) — discourse is parsed incrementally into a structure of **discourse referents
  ("file cards")** + conditions on them; **indefinite = allocate-new card; definite/anaphor = retrieve-and-bind to an
  accessible card.** There is *no validated neural implementation of DRT* — but the mapping **discourse referent =
  hippocampal engram/WM slot; indefinite = allocate engram; definite/pronoun = retrieve-and-bind; Nref = the cost
  signal on ambiguous/failed retrieval** is precise and is exactly what this project's engram + abstention machinery
  can express. (Open mapping → a genuine contribution opportunity, not a citable finding.)

### (c) What our sim has / lacks
- **LACKS entity instances entirely.** The composer (`rf_phasor_composer.py`, `one_brain_composer.py`) stores facts
  keyed by **concept-TYPE codes** for `(agent, action, patient)` (`store(agent, action, patient, polarity)`,
  `rf_phasor_composer.py:528`). It supports **attributed entities** ("big apple" = a single-attribute bind, lines
  532–534) but **"big apple" is still a TYPE, not a token** — two different apples collapse to the same code. There
  is no `the`/`a` determiner handling, no instance allocation, no "which".
- **No definiteness / presupposition / bridging.** `hear()` parses bare SVO; determiners are dropped.
- **The abstention moat is the closest existing primitive.** A query whose cue roles match no stored fact returns
  `None` (`rf_phasor_composer.py:13, 497`) — the no-confab moat. This is *failure detection*, but it currently fires
  silently and on FACTS, not on entity-reference resolution.

### (d) Cheap-first, biology-grounded options (ranked)
1. **★ Entity-instance layer via per-instance engrams (the keystone — see §11 target #1).** Allocate a distinct
   sparse engram (a token "boy#1") on first mention of an indefinite; bind type + attributes + role-history to it;
   resolve a definite/pronoun by **cue-driven pattern completion** over the engram store (CA3 autoassociator).
   *Reuse:* the shipped **engram-tagging API** (`bridge.start_engram_recording` / `commit_engram_tag` /
   `stimulate_tag`, catalog **D.14**, `2026-05-11`), the **sparse heteroassociator** `_D_sparse_heteroassoc.py`
   (RESOLVED, multi-seed, anti-cheat-clean), DG/CA3 pattern separation/completion (**D.12/D.13**, already validated
   in `validate_trisynaptic_loop`). *Anti-cheat:* two same-type instances ("a boy … another boy") must get
   **decorrelated** engrams (population-vector cos ≈ 0 — the D.12 separation metric) and the right one must be
   recovered by a partial cue (D.13 completion); a *permuted-binding* control must collapse; the no-confab moat must
   stay 0-FA.
2. **Definiteness → allocate-vs-retrieve switch.** Parse `a/an` → allocate-new; `the`/pronoun → retrieve-and-bind.
   Cheap because the parser already exists (`BridgeParser`); this adds a determiner→operation tag. *Anti-cheat:* an
   indefinite never binds to an existing referent; a definite never silently allocates (it either binds or triggers
   §4 clarification).
3. **Bridging via the learned association graph** (`LearnedAssocGraph`, already built): "the engine" after "a car"
   licensed by a learned car→engine co-occurrence edge. Lower priority; needs the instance layer first.

---

## 2. Multi-referent disambiguation ("the dog saw the cat — *it* ran")

### (a) Human capability + example
With several referents in mind, a bare pronoun binds the **salient / contextually-favoured** one, using **agreement
(number/gender/animacy), recency, grammatical role (subjecthood), and verb selectional restrictions**: "The dog saw
the cat. **It** ran." (favours the recent/topical), but "The boy dropped the glass. **It** broke." (verb *break* +
inanimacy → the glass). When genuinely ambiguous → ask (§4).

### (b) Biological mechanism
- The Nref (§1) is the *graded* cost of the candidate set. Selecting *which* candidate is **attentional biased
  competition** — mutual inhibition among candidates, biased by salience/agreement — **Desimone & Duncan 1995,
  *Annu. Rev. Neurosci.* 18:193–222**; the attractor implementation is **Wong & Wang 2006, *J. Neurosci.*** (recurrent
  WTA amplifying a small biased input), with the stability envelope **Rutishauser, Douglas & Slotine 2011** (α<1).
- *Catalog:* the WTA precedent is **B.04 MSN lateral inhibition** (cross-pool WTA) and the navigation `sel`/`commit`
  recipe; binding-by-synchrony FS mutual inhibition is the gamma analogue.

### (c) What our sim has / lacks — **partly built, default-OFF, host-scaffolded bias**
This is the sim's **most mature discourse-disambiguation result**, and the trail is instructive:
- **Two converging NEGATIVES** first mapped the requirement (`2026-06-17-multireferent-disambiguation-NEGATIVE.md`):
  the plain WM loop holds referents as an **equal set** — recency does NOT decide (0/3 seeds), and a **salience
  *boost* alone is also NEGATIVE** (a 4× drive boost can't flip the winner). The independent per-concept attractors
  mean a boost only *adds* activity; it does not *suppress* the competitor.
- **The fix de-risked GO + integrated** (`2026-06-19-multireferent-integration-multiturnagent.md`):
  `BiasedCompetitionContextBuffer` (mutual inhibition between held referents + a content bias) folded into the
  production `MultiTurnAgent` behind `enable_biased_competition` (default OFF, byte-identical when off). GO-arm 5/6
  seeds; all anti-cheat controls 6/6; content beats recency; the moat abstains on empty-WM and content-silent
  queries.
- **THE RESIDUAL SHORTCUT (honest):** the *which-candidate-gets-the-bias* decision is a **host helper**,
  `content_bias_target` over `ANIMACY` / `VERB_SELECTS` Python lexicons (`biased_competition_buffer.py`, flagged
  in-module for neuralization). The **WIN is brain-based** (the spiking WTA + selective inhibition + Wong–Wang
  amplification); the **scoring is the host scaffold**.
- **Two named boundaries carried forward:** (1) extreme intrinsic-attractor asymmetry (seed-100) → a fixed bias
  occasionally can't flip the winner, and the failure mode is **abstention, not confabulation** (fix: content-graded
  / homeostatically-normalized bias within α<1); (2) the **all-compatible** case (two animate same-number candidates,
  agreement silent) needs finer role/recency cues on top.

### (d) Cheap-first options (ranked)
1. **Neuralize the bias scoring** (close the residual): a learned synaptic map *pronoun-feature population ×
   candidate-feature population → bias current*, so agreement/animacy compatibility is computed by neurons. *Reuse:*
   the validated buffer + a small learned feature-compatibility matrix (Hebbian, like `LearnedAssocGraph`).
   *Anti-cheat:* spiking-bias == host-bias on the validated turns; an honest NEGATIVE here *itself maps* what the
   point-neuron substrate can compute about agreement (a legitimate deliverable).
2. **Content-graded bias** (close boundary 1): scale bias magnitude by the content signal, normalized to keep α<1.
3. **Flip `enable_biased_competition` ON for multi-referent dialogue demos** once the bias is neuralized.
   *(All three are reuse-by-import; no `sim/` edit.)*

---

## 3. Theory of Mind / mentalizing — modelling the interlocutor's knowledge

### (a) Human capability + example
Track what the **other** person knows, believes, and intends — separately from one's own knowledge. "He's looking
for the ball — but he doesn't know Anne moved it, so he'll look in the basket" (false belief). In dialogue: tailor
*what you say* to what the listener does/doesn't already know (→ §5 common ground, §6 implicature). The canonical
test is **false belief** (Sally-Anne).

### (b) Biological mechanism
- **A reliably-replicated mentalizing network:** **bilateral (esp. right) posterior TPJ + dorsomedial PFC**, with
  precuneus, posterior STS, lateral temporal cortex, temporal poles as extended nodes. The **task-general core**
  (the only regions converging across *every* ToM task type) is **bilateral posterior TPJ + dmPFC**. *Citation:*
  **Schurz, Radua, Aichhorn, Richlan & Perner 2014, *Neurosci. Biobehav. Rev.* 42:9–34** (the load-bearing
  meta-analysis); **Saxe & Kanwisher 2003, *NeuroImage*** (rTPJ for belief attribution); **Frith & Frith 2012,
  *Annu. Rev. Psychol.* 63:287–313** (implicit vs explicit mentalizing).
- **Functional split:** **rTPJ = transient situation-specific mental states** (beliefs, intentions); **mPFC = enduring
  traits + self/other**, on a ventral→dorsal self→other gradient (Denny et al. 2012).
- **Kandel 6e directly:** "When adults perform tasks that depend on **theory of mind**, the **right
  temporal-parietal junction and superior temporal sulcus** are active" (**Ch 56**, decision/consciousness; the
  Sally-Anne / false-belief paradigm). The **mind-blindness hypothesis of autism** + the social-cognition circuit
  (**OFC, ACC, amygdala, STS, IFG, fusiform**) is laid out in **Ch 62/64** (Figs 62-1, 62-2 — the Sally-Anne test).
- **Computational model (the most directly implementable):** **Bayesian Theory of Mind / inverse planning** (Baker,
  Saxe & Tenenbaum 2009/2011) — model the other as a planner with desires (utility) + a subjective belief
  distribution; invert by Bayesian inference to recover belief+desire from observed action. Mapped to the brain by
  **Koster-Hale & Saxe 2013, *Neuron* 79:836–848 ("Theory of mind: a neural prediction problem")** — **rTPJ codes a
  prediction error over mental states** (responds more to context-unexpected beliefs/desires). ⇒ a generative model
  of the other's mind + a PE signal. *(rTPJ belief-selectivity is contested — Mitchell shows it also fires for
  non-social attentional reorienting; the reconciling view is the predicted-vs-observed code that spans both.)*

### (c) What our sim has / lacks
- **LACKS any ToM / interlocutor model.** There is **no catalog cluster for mentalizing** (clusters A–Q have none).
  The agent models the world (facts), not the *speaker's mind*. There is no self/other distinction, no belief
  attribution, no false-belief capacity.
- This is a **deep structural gap**, but a tractable *minimal* version exists (a second "what the listener knows"
  fact-store) — see §5.

### (d) Cheap-first options (ranked)
1. **Minimal common-ground store first** (§5) — a *second* fact-store tagged "shared/listener-known" is the cheapest,
   highest-value slice of ToM and unlocks audience design without full belief reasoning. *Reuse:* a second composer
   instance (the architecture already supports multiple bridges/stores).
2. **A false-belief representation as a separable belief-store** (deeper): represent the *other's* belief-set as a
   distinct engram namespace that can DIVERGE from the agent's own (Sally thinks ball∈basket; agent knows
   ball∈box). The compose algebra already binds role-fillers; tagging facts with a *believer* role is a 5th role
   binding. *Anti-cheat:* the agent must answer "where will Sally look?" from **Sally's** store (basket), not its own
   (box) — the Sally-Anne test, the gold-standard control.
3. **rTPJ-as-prediction-error** (research-grade, defer): the deepest version (Koster-Hale & Saxe) needs a generative
   model of the other; flag as a long-horizon target, not a cheap-first.
   *Honest:* full ToM is the hardest item in this cluster on the current substrate; the *minimal* slices (1–2) are
   buildable and are most of the conversational value.

---

## 4. Clarification requests & conversational repair ("which X?")

### (a) Human capability + example
On a reference failure or ambiguity, the competent move is **other-initiated repair**: "Which book?" / "Wait, who?"
rather than guessing or going silent. There is a universal preference structure (self- over other-repair) and
ordered repair-initiation positions.

### (b) Biological mechanism — **honest: the neuroscience is SPARSE**
- Repair is well-characterized at the **conversation-analytic** level — universal, structurally organized, with a
  preference for self-repair; "which X?" is a canonical **next-turn other-initiated repair (OIR)** on a referential
  trouble source; **open vs restricted** formats ("Huh?" vs "Which book?") signal the trouble locus. *Citations:*
  **Schegloff, Jefferson & Sacks 1977, *Language* 53:361–382** (foundational); **Dingemanse, Roberts, … Enfield 2015,
  *PLoS ONE* 10:e0136100** (cross-linguistic universality of OIR); **Albert & de Ruiter 2018, *Topics in Cog. Sci.*
  10:279–313** (the cognition interface review).
- **Grounding** (Clark & Brennan 1991): repair resolves failures of the *grounding criterion* (sufficient mutual
  evidence of understanding).
- **The closest mechanistic bridge (a MODEL, not a finding):** repair = the behavioral output of a **comprehension
  monitoring / prediction-error signal** — when the situation model cannot uniquely resolve a referent (the §1 Nref
  ambiguity signal, or an outright failure), a **monitoring/conflict system (medial-frontal/ACC + dlPFC)** flags the
  trouble and triggers a clarification act. This Nref→repair link is *plausible and well-motivated but not directly
  established* — flag it explicitly as the project's model, grounded in CA + grounding theory + the Nref.

### (c) What our sim has / lacks — **the moat is the perfect hook**
- **LACKS clarification behaviour** (no turn-taking, no question generation).
- **BUT the no-confab abstention moat is exactly the trigger condition, already built.** Today an unresolvable query
  returns `None` *silently* (`rf_phasor_composer.py:497`); the Bogacz-Brown **familiarity gate**
  (`familiarity_gate_v320_validation.py`, `2026-06-11-familiarity-gate-v320-GO.md`, and the abstention-as-separate-
  familiarity-signal finding `2026-05-22`) is the validated neural detector of "I don't have a unique match." **The
  same detector that drives silent abstention can route to a clarification request** instead.

### (d) Cheap-first options (ranked)
1. **★ Route the abstention/ambiguity signal to a clarification act** (the cleanest unlock in this cluster — see §11
   target #2): on a **definite-reference** query, the entity-resolver (§1) returns one of {unique → bind; **empty** →
   "I don't know which X you mean"; **>1 match** → "which X — the big one or the red one?"}. The branch is gated by
   the *same familiarity/ambiguity signal* the moat already computes; the disambiguating descriptors come from the
   distinguishing attributes on the competing engrams. *Reuse:* the moat / familiarity gate + the biased-competition
   candidate set (§2) + the engram attributes (§1). *Anti-cheat:* one unique referent → no clarification (no
   over-asking); two equally-good referents → ask, and the question must name a **distinguishing** attribute (not a
   shared one); zero referents → "which X?" not a confabulated antecedent. **This keeps the no-confab discipline:
   instead of abstaining mute, it abstains *informatively*.** Per [[feedback_moat_not_hard_lossy_memory_ok]] the moat
   relaxes to graded confidence — a clarification request IS the graceful-degradation behaviour the owner wants.
2. **Self-repair on low-confidence generation** (defer): monitor the agent's own output confidence and self-correct.

---

## 5. Common ground / shared context / audience design

### (a) Human capability + example
Track what is **mutually known** and tailor utterances to it: don't re-explain what the listener already knows; use a
pronoun/definite for an entity already in common ground, a full description for a new one. "Did you find **it**?"
works only if "it" is grounded for both.

### (b) Biological mechanism
- **Speaker–listener neural coupling:** during *successful* communication the listener's brain activity becomes a
  time-lagged copy of the speaker's; the coupling **vanishes when communication fails** and its **degree predicts
  comprehension**; some listener regions (dmPFC, TPJ, precuneus, striatum) show **anticipatory** coupling that
  *precedes* the speaker. *Citation:* **Stephens, Silbert & Hasson 2010, *PNAS* 107:14425–14430** (landmark);
  **Hasson, Ghazanfar, … Keysers 2012, *Trends Cogn. Sci.* 16:114–121** (brain-to-brain framework).
- **Common ground is hippocampal/declarative:** **amnesics with hippocampal damage fail to establish and use common
  ground in collaborative reference** (they don't form/retrieve the shared referential pacts), with spared syntax.
  *Citation:* **Duff & Brown-Schmidt 2012, *Front. Hum. Neurosci.* 6:69** (the core review); **Brown-Schmidt & Duff
  2016, *Topics Cogn. Sci.* 8:177–190**.
- **Computational model:** **interactive-alignment** (Pickering & Garrod 2004, *BBS*) + **forward-model prediction**
  (Pickering & Garrod 2013) — each interlocutor runs a forward model of the other → exactly the substrate for the
  anticipatory coupling. Implementable as **mutual predictive coding between agents.**

### (c) What our sim has / lacks
- **LACKS a common-ground store.** All facts are in one undifferentiated store; there is no "shared vs private"
  partition, no model of what the listener knows, no audience design. (The multi-turn WM buffer holds the *current*
  referent but not a *shared-knowledge ledger*.)

### (d) Cheap-first options (ranked)
1. **★ A "shared/common-ground" fact partition** (cheapest ToM-adjacent win): tag each fact/entity as
   *introduced-to-the-listener* vs *private*; use the partition to decide pronoun-vs-full-description and to avoid
   re-stating known facts. *Reuse:* a per-fact "grounded" tag (the composer already stores per-fact metadata;
   `polarity` is precedent for an extra bound role). *Anti-cheat:* the agent uses a pronoun ONLY for a grounded
   referent; a brand-new entity gets a full description; a *reset* of the common-ground partition forces
   re-introduction (load-bearing control).
2. **Hippocampal grounding via engrams** (deeper, biology-faithful): the common-ground ledger as a distinct engram
   namespace updated at each accepted contribution (Clark grounding) — ties to §1/§4 and Duff & Brown-Schmidt.
3. **Predictive speaker–listener coupling** (research-grade, defer): two coupled agents with mutual forward models.

---

## 6. Grice's maxims / conversational implicature

### (a) Human capability + example
Derive meaning *beyond* the literal: scalar implicature "I ate **some** of the cookies" ⇒ *not all*; "Can you pass
the salt?" ⇒ a request, not a yes/no question (indirect speech act). Obey/exploit the maxims (quantity, quality,
relation, manner).

### (b) Biological mechanism
- Implicature is an **effortful inferential step** engaging a **left-lateralized fronto-parietal + mentalizing**
  system (a *correction* to the older "right-hemisphere = pragmatics" view): **left dlPFC/rlPFC (BA 46/10)** for the
  inference, **right TPJ/IPL + mPFC** for the ToM component, **dorsal-PFC–basal-ganglia** for literal-vs-implied
  conflict. *Citations:* **Hartshorne, Snedeker & Kim 2015, *Lang. Cogn. Neurosci.*** (scalar-implicature fMRI);
  **Bašnáková, Weber, … Hagoort 2014, *Cereb. Cortex*** (comprehending indirect/implied speaker meaning →
  mentalizing + frontal). The RH contributes to figurative language/prosody but is not exclusive.
- **Computational model:** **Rational Speech Act (RSA)** (Frank & Goodman 2012, *Science*; Goodman & Frank 2016,
  *Trends Cogn. Sci.*) — implicature as recursive Bayesian listener↔speaker reasoning over a shared lexicon. Composes
  with the §3 BToM (both nested Bayesian agent models).

### (c) What our sim has / lacks
- **LACKS implicature entirely.** Comprehension and answering are literal. No quantity/relevance reasoning, no
  indirect speech acts. The dlPFC dialogue-planner picks *what to say next* (relevance-by-spreading) but does not
  *derive non-literal meaning*.

### (d) Cheap-first options (ranked)
- **Mostly DEFER — this is a hard wall on the current substrate.** Implicature is recursive agent-modelling (RSA),
  which presupposes the ToM (§3) and common-ground (§5) layers. Honest verdict: **not cheap-first**; sequence it
  *after* §3/§5 exist. The one tractable slice: **scalar quantity** ("some" → "not all") could be a learned
  lexical-scale entailment (some<all), testable without full RSA — but low priority relative to the keystone gaps.

---

## 7. Turn-taking & dialogue management

### (a) Human capability + example
Exchange turns with a **~0–200 ms** gap — predicting *when* the other will finish and pre-loading a response while
still listening; yielding/holding the floor; minimal overlap.

### (b) Biological mechanism
- Turn-taking is **universal and astonishingly fast** (median gaps ~0–200 ms across 10 unrelated languages), which
  creates the **production–comprehension timing puzzle**: production planning takes >600 ms, so listeners **predict
  the turn-end and begin planning mid-turn**. EEG shows response planning begins as soon as the *gist/speech-act* is
  grasped (up to seconds before turn-end), indexed by a production-planning positivity + **alpha/beta power
  reduction** (attention switch to production / motor turn-end projection). Turn-end prediction uses lexico-syntactic
  content (primary) + prosody (final cue). *Citations:* **Stivers, Enfield, … Levinson 2009, *PNAS* 106:10587–10592**
  (the ~200 ms universal); **Bögels, Magyari & Levinson 2015, *Sci. Rep.* 5:12881** (mid-turn planning EEG);
  **Levinson 2016, *Trends Cogn. Sci.* 20:6–14** (review).
- **Mechanism = predictive-timing + a motor-preparation gate**: a forward model projects the turn-end and pre-loads a
  motor plan, released at the transition-relevance place. *Catalog tie-in:* the project already has bounded
  **evidence-accumulation-to-threshold** machinery (**G.16/G.17 drift-diffusion/LIP accumulator**; the navigation
  Wang-2002 accumulator + Lo-Wang commit-burst) — a turn-end estimator is structurally the *same* "ramp to a
  commit threshold" primitive, here over an utterance-completion signal.

### (c) What our sim has / lacks
- **LACKS turn-taking ENTIRELY** (confirmed: no runner, no finding mentions turn-taking/prosody/endpointing). The
  agent is request→response, not a real-time dialogue floor manager. There is no notion of "when to speak", no
  turn-end prediction, no overlap handling.

### (d) Cheap-first options (ranked)
- **DEFER for the chat-console use case** (a typed REPL has no real-time floor) — but flag the clean reuse: a
  **turn-end / "ready-to-respond" predictor as a commit-burst accumulator** (reuse the navigation `commit` recipe +
  G.16/G.17) is the obvious biology-grounded build *if/when* a spoken or streaming interface is pursued. Low priority
  for the current goal; named so it isn't reinvented.

---

## 8. Intent & speech-act recognition

### (a) Human capability + example
Recognize the **communicative act** behind an utterance — that "Can you pass the salt?" is a *request*, that pointing
is *informing*, that an utterance is *ostensive* (meant to be recognized as communicative). Distinguish a question
from an assertion from a command.

### (b) Biological mechanism
- Recognizing that an act is *communicative* (the intentional stance / ostension) engages the **mentalizing network**
  with a special role for **anterior rostral mPFC (arMPFC)** and **right posterior STS**; sender (planning a
  communicative act) and receiver (recognizing its intent) rely on **overlapping** regions (right pSTS) — a *shared*
  generative substrate (the sender uses their own recognition system to predict the receiver). arMPFC is
  specifically more active for *communicative* than private intentions. *Citations:* **Noordzij, Newman-Norlund, de
  Ruiter, Hagoort, Levinson & Toni 2009, *Front. Hum. Neurosci.*** (overlapping sender/receiver substrate); **Stolk,
  Verhagen & Toni 2016, *Trends Cogn. Sci.* 20:180–191** ("Conceptual alignment"). **Relevance Theory** (Sperber &
  Wilson 1986/95) is the dominant cognitive account of ostensive-inferential communication; composes with RSA + BToM.

### (c) What our sim has / lacks
- **PARTIAL/implicit.** The parser distinguishes a **statement** ("dog chase cat" → store) from a **question**
  ("what does dog chase?" → query) and handles **negation/yes-no** (a bound polarity role). So a *minimal* speech-act
  taxonomy (assert vs ask vs negate) is implicitly present. **LACKS** indirect speech acts (request-as-question),
  ostension, and any explicit "this is the act type" representation.

### (d) Cheap-first options (ranked)
1. **Make the speech-act type an explicit (bound) role** — assert / ask / request / acknowledge — alongside the SVO,
   so the act type is a first-class queryable feature (the project binds roles cleanly; this is a small extension of
   the existing statement/question/polarity split). *Anti-cheat:* a yes-no question and the matching assertion bind
   *different* act roles but the *same* propositional content.
2. **Indirect speech acts** (defer): "can you X?" → request needs the ToM/implicature layer (§3/§6).

---

## 9. Anterior temporal lobe — the semantic / social-concept hub (supporting substrate)

### (a) Role
The **ATL is the transmodal hub** binding modality-specific "spokes" into modality-invariant **concepts**, with the
**ventrolateral/polar ATL especially weighted toward SOCIAL concepts + person knowledge** (the substrate the
mentalizing network reasons over). Semantic dementia (bilateral ATL atrophy) → modality-general concept loss, with
social/person knowledge degraded.

### (b) Biological mechanism + citations
**Patterson, Nestor & Rogers 2007, *Nat. Rev. Neurosci.* 8:976–987** (hub-and-spoke landmark); **Lambon Ralph,
Jefferies, Patterson & Rogers 2017, *Nat. Rev. Neurosci.* 18:42–55** (semantic-cognition review); **Olson, McCoy,
Klobusicky & Ross 2013, *SCAN* 8:123–133** + **Binney, Hoffman & Lambon Ralph 2016, *Cereb. Cortex* 26:4227–4241**
(the social-concept gradient). *Model:* the Rogers–McClelland **hub-and-spoke PDP** convergence-zone network —
lesioning the hub reproduces semantic-dementia degradation.

### (c) What our sim has / lacks — **already strongly matched**
The project's **cross-modal Hebbian convergence-zone** work (the generalization arc, 2026-06-16:
`_genfrontier_*` — ATL convergence-zone, Patterson–Lambon Ralph hub, Garagnani–Pulvermüller spiking precedent) is a
faithful instance of this hub. **What's missing is the *social/person* specialization** — the sim's concepts are
generic, not person-indexed. The §1 entity-instance layer is what turns a generic-concept hub into a person/instance
hub.

### (d) Option
- **Person-concept = entity engram + ATL hub** (composes with §1): a person token is an instance engram whose
  attributes/relations are bound through the convergence-zone hub. No new mechanism beyond the §1 keystone +
  existing convergence work.

---

## 10. The DMN / temporal-receptive-window hierarchy — the architectural lesson

Not a single capability but the **organizing principle** for the whole cluster, and the most actionable
architectural insight:

- **One network does most of the discourse/social work:** the **mentalizing network (rTPJ + dmPFC + precuneus +
  pSTS) ≈ the DMN dorsomedial subsystem** (Andrews-Hanna, Smallwood & Spreng 2014, *Ann. NYAS* 1316:29–52), sitting
  at the **apex of a hierarchy of temporal-receptive-windows** — early auditory cortex tracks ms; sentence-level
  areas integrate seconds; **DMN apex regions respond only to information integrated over paragraphs/minutes**
  (**Hasson, Yang, … Rubin 2008, *J. Neurosci.* 28:2539–2550**; **Hasson, Chen & Honey 2015, *Trends Cogn. Sci.*
  19:304–313** — "hierarchical process memory"). The **hippocampal MTL subsystem** of the DMN does the
  entity/episode binding (§4).
- **Architectural takeaway for the sim:** **put slow discourse/social state in the highest, longest-timescale
  layer** — a persistent WM/engram layer that integrates *across turns*, distinct from the fast per-sentence
  composer. The project *already* has the right shape: the per-sentence composer (fast TRW) + the persistent
  `SpikingLoopContextBuffer` / multi-turn WM (slow TRW). The missing piece is making the slow layer hold **entity
  tokens + common ground**, not just one referent.
- **G.09 (Imagination/constructive memory, DMN core; *Kandel Ch 52 pp 1300–1302*)** is the catalog's only DMN
  entry — currently `missing`. Discourse-level integration is the *comprehension* face of the same DMN machinery.

---

## 11. TOP build targets — highest leverage, ranked

The review converges hard: **the discourse-entity layer is the single keystone, and the project already owns most of
the machinery.** Targets are ordered by leverage × cheapness × biological faithfulness.

### ★ Target 1 — the discourse-referent / entity-INSTANCE layer (the keystone)
**Why #1:** every other gap in this cluster is downstream of it. "Which boy?" (§4), definite-vs-indefinite (§1),
multi-referent disambiguation at scale (§2), common ground (§5), person concepts (§9), and any ToM about *a specific
person* (§3) all presuppose **entity tokens distinct from concept types** — which the sim flatly lacks (it stores
TYPE-keyed SVO). **Biology is clean and matched to existing machinery:** an entity instance = a **sparse hippocampal
engram** (D.14 Tonegawa engram-tagging API, shipped) bound into the **relational store** (D.02 Eichenbaum–Cohen,
*Kandel Ch 52 pp 1301–1302*); indefinite = allocate-new, definite/pronoun = retrieve-by-pattern-completion (D.12/D.13,
already validated in `validate_trisynaptic_loop`); the **sparse heteroassociator** `_D_sparse_heteroassoc.py` (RESOLVED,
multi-seed, anti-cheat-clean) is the recall substrate. The formal target is the DRT/file-card discourse model with a
neural mapping the project can *claim*. **Cheap-first de-risk:** two same-type instances ("a boy … another boy") →
decorrelated engrams (D.12 separation cos≈0), each recovered from a partial cue (D.13 completion); permuted-binding
control collapses; the no-confab moat stays 0-FA. **No `sim/` edit anticipated** (engram API + heteroassociator both
exist).

### ★ Target 2 — clarification-on-failure: route the abstention/ambiguity signal to "which X?"
**Why #2:** it directly delivers the owner's flagship "which boy?" behaviour, it is the *cheapest* unlock (the
trigger already exists), and it converts the no-confab moat from *silent* abstention into *informative* graceful
degradation — exactly [[feedback_moat_not_hard_lossy_memory_ok]] + [[project_communicable_brain_not_rag]] (a brain
you communicate WITH). **Biology:** repair-on-reference-failure (Schegloff 1977; Dingemanse 2015; the comprehension-
monitoring → repair model) gated by the **same familiarity/ambiguity signal** (Bogacz-Brown familiarity gate,
`familiarity_gate_v320_validation.py`, GO; the Nref-as-ambiguity-cost analogue). **Mechanism:** on a *definite*-
reference query, the entity-resolver (Target 1) returns {unique → bind; empty → "I don't know which X"; >1 → "which X
— the big one or the red one?"}, with the disambiguating descriptors read off the competing engrams' distinguishing
attributes. **Cheap-first de-risk:** unique → no clarification (no over-asking); two equal candidates → ask, naming a
*distinguishing* (not shared) attribute; zero → "which X?" not a confabulated antecedent; moat stays 0-FA. **Reuse:**
the moat/familiarity gate + the biased-competition candidate set (§2) + Target-1 engram attributes. **No `sim/`
edit.** *(Depends on Target 1 for the entity set; the trigger/branch itself is independently buildable and testable
on facts first.)*

### ★ Target 3 — minimal common ground + a shared/private fact partition (the cheapest ToM slice)
**Why #3:** the smallest viable step toward modelling the interlocutor — it unlocks audience design (pronoun-vs-full-
description, not re-stating known facts) without full belief reasoning, and it is the foundation the harder ToM (§3)
and implicature (§6) layers sit on. **Biology:** common ground is hippocampal/declarative (Duff & Brown-Schmidt
2012); the DMN/mentalizing network tailors output to it; speaker–listener coupling (Stephens et al. 2010) is its
inter-brain signature. **Mechanism:** tag each fact/entity *introduced-to-listener* vs *private* (a bound metadata
role, `polarity` is precedent); use the partition to decide pronoun-vs-description and to suppress redundant
re-statement. **Cheap-first de-risk:** pronoun used ONLY for a grounded referent; new entity → full description; a
common-ground *reset* forces re-introduction (load-bearing control); moat preserved. **Reuse:** a second composer
store / per-fact tag. **No `sim/` edit.**

**Sequencing.** Build **1 → 2 → 3** (each enables the next). All three are reuse-by-import on shipped machinery
(engram API, heteroassociator, familiarity gate, biased-competition buffer, multi-turn WM), need **no `sim/` edit**,
and never weaken the no-confab moat (they *extend* it: Target 2 makes abstention informative; Targets 1/3 add
content the moat guards). **Deferred (hard walls, named, sequenced after 1–3):** full ToM / false-belief reasoning
(§3) and Gricean implicature / RSA (§6) — both recursive agent-modelling that presuppose the entity + common-ground
layers; turn-taking (§7) — clean reuse of the commit-burst accumulator but only relevant for a streaming/spoken
interface, not the typed console.

---

## 12. Honest boundaries & caveats

- **The biggest single gap is representational, not algorithmic:** TYPE-only fact storage. Until entity instances
  exist (Target 1), "which boy?" is *unrepresentable*, not merely unimplemented. This is the load-bearing finding.
- **The multi-referent bias is host-scaffolded** (`content_bias_target` over Python animacy/verb lexicons) — the WTA
  competition is brain-based but the agreement *scoring* is not yet neural (flagged for conversion, §2(d)1). Per the
  BRAIN-BASED-ONLY standard, an honest NEGATIVE on the neural-bias version would itself be a deliverable.
- **Full theory-of-mind and implicature are genuine hard walls** on the point-neuron substrate — both are recursive
  Bayesian agent models (BToM, RSA). The *minimal* slices (common-ground partition, false-belief-as-separate-store,
  explicit speech-act role) are tractable and carry most of the conversational value; the recursive versions are
  long-horizon. Do not overclaim a "ToM" from the minimal slices.
- **Conversational-repair neuroscience barely exists** — the Nref→repair trigger is a *model* (CA + grounding +
  the referential-ambiguity ERP), not a citable circuit. The strongest move (Target 2) is to *propose* it as the
  project's contribution, explicitly flagged.
- **The catalog has no ToM/discourse cluster and glossary.md is absent** — this cluster is under-covered in the
  project's own references; the Kandel ToM material lives in the *decision* (Ch 56) and *autism* (Ch 62/64) chapters,
  not a dedicated social-cognition chapter. Future catalog work could add a social-cognition cluster.
- **rTPJ belief-selectivity is contested** (Saxe vs Mitchell); **RH-dominance of pragmatics is revised** toward
  bilateral+frontal. Cite the meta-analysis (Schurz 2014) and the predicted-vs-observed reconciliation (Koster-Hale
  & Saxe 2013) rather than the strong belief-selective claim.

---

## Appendix — key citations (drop-in)

**Reference / anaphora:** Nieuwland & Van Berkum 2006 *JoCN*; Van Berkum et al. 2007 *Brain Res.* 1146:158–171; Kamp
1981 / Heim 1982 (DRT/file-cards). **Hippocampus & language:** Duff & Brown-Schmidt 2012 *Front. Hum. Neurosci.*
6:69; Brown-Schmidt & Duff 2016 *Topics Cogn. Sci.* 8:177–190; Cohn-Sheehy … Ranganath 2023 *Neuron* 111:4137–4153.
**ToM:** Saxe & Kanwisher 2003 *NeuroImage*; Schurz et al. 2014 *Neurosci. Biobehav. Rev.* 42:9–34; Frith & Frith
2012 *Annu. Rev. Psychol.* 63:287–313; Koster-Hale & Saxe 2013 *Neuron* 79:836–848; Baker, Saxe & Tenenbaum 2009/2011
(BToM). **DMN / timescales:** Andrews-Hanna et al. 2014 *Ann. NYAS* 1316:29–52; Hasson et al. 2008 *J. Neurosci.*
28:2539–2550; Hasson, Chen & Honey 2015 *Trends Cogn. Sci.* 19:304–313; Yeshurun, Nguyen & Hasson 2021 *Nat. Rev.
Neurosci.* 22:181–192. **Common ground / coupling:** Stephens, Silbert & Hasson 2010 *PNAS* 107:14425–14430;
Pickering & Garrod 2004 *BBS* / 2013. **Implicature:** Hartshorne, Snedeker & Kim 2015 *Lang. Cogn. Neurosci.*;
Bašnáková et al. 2014 *Cereb. Cortex*; Frank & Goodman 2012 *Science* (RSA); Goodman & Frank 2016 *Trends Cogn. Sci.*
**Turn-taking:** Stivers et al. 2009 *PNAS* 106:10587–10592; Bögels, Magyari & Levinson 2015 *Sci. Rep.* 5:12881;
Levinson 2016 *Trends Cogn. Sci.* 20:6–14. **Speech acts / ostension:** Noordzij et al. 2009 *Front. Hum.
Neurosci.*; Stolk, Verhagen & Toni 2016 *Trends Cogn. Sci.* 20:180–191; Sperber & Wilson 1986/95. **ATL hub:**
Patterson, Nestor & Rogers 2007 *Nat. Rev. Neurosci.* 8:976–987; Lambon Ralph et al. 2017 *Nat. Rev. Neurosci.*
18:42–55; Olson et al. 2013 *SCAN* 8:123–133; Binney et al. 2016 *Cereb. Cortex* 26:4227–4241. **Repair:** Schegloff,
Jefferson & Sacks 1977 *Language* 53:361–382; Dingemanse et al. 2015 *PLoS ONE* 10:e0136100; Albert & de Ruiter 2018
*Topics Cogn. Sci.* 10:279–313; Clark & Brennan 1991 (grounding). **Kandel 6e:** Ch 52 pp 1296–1302 (episodic /
relational / DMN-constructive); Ch 54 pp 1340–1342, 1357–1360 (trisynaptic / engram / CA2 social memory); Ch 55 pp
1370–1387 (language, dual-stream, Broca/Wernicke); Ch 56 pp 1393–1412 (decision; ToM → rTPJ+STS); Ch 62/64 (autism /
mind-blindness, Sally-Anne, social-cognition circuit). **Catalog:** D.01, D.02, D.03, D.12, D.13, D.14, D.15 (CA2
social memory); G.06, G.08, G.09, G.10–G.14, G.16, G.17.
