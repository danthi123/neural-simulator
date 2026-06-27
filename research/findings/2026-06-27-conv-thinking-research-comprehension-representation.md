# Conversation + thinking — deep research: COMPREHENSION + KNOWLEDGE REPRESENTATION cluster

**Date:** 2026-06-27
**Type:** READ-ONLY deep-research findings doc (no `sim/`/runner edit, no commit — the controller pushes). Produced per the project's standing practice (deep research + catalog + Kandel + literature review BEFORE a new-direction / mechanism-class build). One of a set of per-cluster research docs feeding a conversation-and-thinking build roadmap.
**Cluster scope:** (1) syntactic parsing & comprehension — argument structure / verb subcategorization, wh-questions, voice, morphology, garden-paths; (2) knowledge representation — concepts vs **entity instances** (types vs tokens), properties/attributes, relations, schemas/frames, argument structure, tense/aspect, events.
**Method note:** the prompt-referenced `glossary.md` does **not exist** at `sim-catalog/references/` (verified — only `feature-catalog.md`, `biology-buildout-roadmap.md`, `textbooks/`). The catalog `feature-catalog.md` (5,640 lines) + Kandel 6e Ch 52/54/55/56 + WebSearch literature were used. Every load-bearing project-state claim was re-verified by file-read (cited inline); the controller should trust-but-verify the **[VERIFY]**-flagged ones.

---

## 0. The one-paragraph answer

The sim's comprehension front-end is, surprisingly, **the most-built part of this whole cluster** — voice-invariance, productive multi-frame word-order (SVO/VSO/OSV + neural frame *selection*), Bates–MacWhinney multi-cue competition (robust-to-imperfect-English + cross-language case), and a scoped left-corner embedded-clause parser are all DONE or de-risked behind flags. **The deep gap is not parsing words into roles; it is the POVERTY of the role/representation alphabet they parse INTO.** Verified in code: the composer's entire relational vocabulary is `("agent","action","patient","polarity","attribute","attribute2")` and `Clause` is hard-fixed to exactly `(agent, action, patient)` (`rf_phasor_composer.py:24,28`); even the dependency-parse extractor and the case parser collapse everything to a 2-argument `(subject, verb, object)` triple (`_corpus_svo_extract.py:19,74`; `case_aware_role_parser.py:12`). So the brain **cannot represent** verb-specific argument structure ("go **to** the park" — a GOAL; "give the dog **to** the boy" — THEME+RECIPIENT), **entity instances/tokens** (only the type "boy" exists, so "which boy?" is unanswerable), **tense/aspect** ("the dog *was chasing*" vs "*chased*"), or **events as first-class objects**. The biology says these are exactly **(a)** richer *frames* stored on the lexical item (Hagoort MUC "Memory" = posterior/middle temporal cortex stores each verb's structural frame; argument-count graded activation in left posterior perisylvian cortex), bound dynamically by **Unification** (Broca/LIFG); **(b)** a hippocampal **episodic index ("barcode")** that turns co-occurring concept-cell *types* into a bound *token* (Quian Quiroga concept cells + 2024 barcode models); and **(c)** **schemas** generalized in mPFC over instances kept distinct in hippocampus. The top-3 highest-leverage, cheap-first builds are: **(I) a typed-role / verb-frame representation** (extend the composer's role alphabet + make the lexical verb carry its frame — pure reuse of role-filler binding, the project's strongest validated primitive); **(II) entity instances via an episodic-index token** (reuse the shipped engram-tagging API D.14 as the "barcode" that binds a type to a discourse token); **(III) wh-question comprehension as a filler-gap dependency** (reuse the multi-frame parser + the dlPFC NMDA WM-latch — the identical mechanism the embedded-clause scoping already specified, the SAN/LAN working-memory hold). All three are reuse-by-import; none requires the dendritic substrate. The honest wall: the binding ALGEBRA (VSA/FHRR exact-inverse) is a principled idealization, and *productive* (learned-from-data, generalizing) grammar/argument-structure is the deep, research-gated frontier where a point-neuron substrate has only partial precedent (Assembly-Calculus parser).

---

## 1. What the sim ALREADY has (verified by file-read — so the gap analysis is precise)

The comprehension half of this cluster is far more built than the catalog's blanket "missing" suggests. The catalog reflects the *core-sim* status; the conversational runners have moved well past it.

| Capability | Status (verified) | Mechanism / file |
|---|---|---|
| **Voice** (active/passive, voice-invariant role assignment) | **DONE** | `BridgeParser` `(position × voice) → role`, `_GT` map (`brain_conversational_agent.py:25`). "dog chase cat" ≡ its passive frame. |
| **Productive multi-frame word order** (SVO/VSO/OSV) given the frame | **GO 6/6** | `MultiFrameParser` over `(position × FRAME)` (`_phaseB_multiframe_comprehension_derisk.py`; finding `2026-06-18-multiframe-comprehension-GO.md`). Held-out non-native frames 1.000; permuted/lesion collapse. |
| **Neural frame SELECTION** (pick the frame from a cue, not be told it) | **GO 6/6** | `FrameSelector`: verb-position cue → frame ensemble in spikes (`frame_parser.py`; `2026-06-18-frame-selection-GO.md`). |
| **Robust-to-imperfect-English multi-cue competition** | **DONE (flag-gated)** | Bates–MacWhinney Competition Model: position+animacy+verb-fit cues, **learned validities** (three-factor reward), Wong-Wang role WTA (`multicue_role_parser.py`; `2026-06-20-robust-multicue-parser-deep-research.md`). Recovers object-fronted English 0.95 where position-only→0.00. |
| **Cross-language / case-marking cue** | **DONE (flag-gated)** | `CaseAwareRoleParser` adds a 5th `case` cue (ga/wo); cross-linguistic dissociation 6/6 (`case_aware_role_parser.py`; `2026-06-19-case-cue-crosslanguage-derisk.md`). |
| **Single attribute (adj+noun)** | **DONE** | `AttributedBridgeParser` `(from-START × from-END × voice) → role`; `attribute` role on the composer. |
| **Embedded relative clauses** (left-corner parse + WM-hold) | **SCOPED + de-risk specified** | `2026-06-19-embedded-clause-parsing-scoping.md` — relativizer = PUSH cue, dlPFC NMDA latch = WM-hold, per-span reuse of `AttributedBridgeParser`. Composer *decode* of nested `Clause` already DONE (`one_brain_composer._decode_clause`). |
| **Corpus SVO fact extraction** | **BUILT** | spaCy `nsubj`/`dobj`/`dative`/`prep→pobj` → `(subj,verb,obj)` triples (`_corpus_svo_extract.py`; `2026-06-26-deep-knowledge-brain-fluency-research.md`). |
| **Relational fact store + no-confab moat** | **DONE** | `RFPhasorComposer.store/query_patient/query_agent/ask_yes_no/render_fact/query_chain`; absent cue → `None`. Role-filler FHRR binding. |

**The load-bearing observation:** *parsing words → roles* is solved across many cue families. **The representation the roles populate is the bottleneck**, and it is impoverished in five specific, verifiable ways (§2). This reframes the whole cluster: the build targets are **representational**, not parsing-mechanism, additions.

---

## 2. THE PRECISE REPRESENTATION GAP (verified in code — this is the spine of the doc)

Across the *entire* stack — parser → case parser → multi-cue parser → extractor → composer — the relational representation is locked to **two core arguments labelled agent/patient**, plus modifier roles. Concretely:

- **Composer role alphabet** (`rf_phasor_composer.py:24`): `ROLES = ("agent", "action", "patient", "polarity", "attribute", "attribute2")`. There is **no GOAL, RECIPIENT, THEME, INSTRUMENT, LOCATION, SOURCE, TIME** role.
- **`Clause` is hard-fixed** (`rf_phasor_composer.py:28`): `Clause = namedtuple("Clause", ["agent","action","patient"])`. A fact is structurally a triple; nothing else fits.
- **The extractor collapses obliques** (`_corpus_svo_extract.py:19,74-75`): `OBJ_DEPS={"dobj","dative","attr","oprd"}` and a `prep`'s `pobj` are *all* mapped into the single `object` slot — the **preposition is discarded**. "go to the park" → `(go, ?, park)`; "give the dog to the boy" → loses the recipient/theme distinction. The argument *structure* is destroyed at extraction.
- **The case parser still emits `{agent, action, patient}`** (`case_aware_role_parser.py:12`) — case marking lets a free-word-order language find the 2 core arguments, but it is still a 2-argument transitive.

From this single fact, the five capability gaps follow directly:

### GAP-A. Verb argument structure / subcategorization — MISSING
- **Human capability:** different verbs license different argument frames. *go* takes a **GOAL** ("go **to** the park"); *give* takes **THEME + RECIPIENT** ("give **the dog** **to the boy**"); *put* takes **THEME + LOCATION** ("put the ball **on the table**"); *break* is causative-alternating. The verb's lexical entry *is* its frame.
- **Conversational example the sim fails:** "where does the boy go?" → the GOAL of *go* is not a representable role, so the fact "boy go park" cannot encode that *park* is the destination, not the patient. "the dog ate" (intransitive) vs "the dog ate the bone" (transitive) are not distinguished in frame.
- **Why it matters:** this is the single biggest blocker to "natural" facts. Most real corpus sentences have PP arguments and obliques; the brain currently throws them away.

### GAP-B. Entity instances (types vs tokens) — MISSING
- **Human capability:** "a boy met a girl. **The boy** smiled." — *the boy* is a specific **token** instance of the **type** *boy*, distinct from any other boy. Discourse reference, "which boy?", counting ("two dogs"), and tracking an individual across a conversation all require tokens.
- **Conversational example the sim fails:** every "boy" is the same generic concept code; there is no individuation. "which boy?" is unanswerable; "the boy I mentioned" cannot be re-bound.
- **This is the deepest representational gap** and the one the owner flagged explicitly.

### GAP-C. Wh-questions — MISSING as comprehension (rigid probe syntax)
- **Human capability:** "**where** does the boy go?", "**who** chased the cat?", "**what** did the dog eat?" — a fronted wh-word is a **filler** that must be held until its **gap** (the missing argument position) is found, then bound to that role.
- **Conversational example the sim fails:** the parser has no wh-frame; queries go through the rigid `query_patient`/`what does X Y` API, not natural wh-questions.

### GAP-D. Tense / aspect / events as objects — MISSING
- **Human capability:** "the dog **chased** the cat" (past, completed) vs "the dog **was chasing** the cat" (past progressive, ongoing) vs "the dog **will chase**". Aspect changes the event's internal temporal structure; tense locates it. Events are first-class (you can refer to "the chasing").
- **Conversational example the sim fails:** facts are tenseless; "did the dog chase the cat *yesterday*?" cannot be distinguished from a generic.

### GAP-E. Schemas / frames as knowledge representation — MISSING
- **Human capability:** a "restaurant" schema (enter → order → eat → pay), a "giving" event-frame (giver/gift/recipient), generalization over many instances into an abstract structure (FrameNet frames; scripts).
- **Conversational example the sim fails:** no structured event-frame; can't infer the unstated ("if the boy went to the restaurant, he probably ate").

---

## 3. The biological mechanisms — region-by-region, with citations

### 3.1 The language network architecture (Friederici / Hickok-Poeppel / Hagoort)

**Dual-stream model** (catalog **G.11**, `feature-catalog.md:2762`; Kandel 6e Ch 55 pp 1380–1387; Hickok & Poeppel; Saur et al. PNAS 2008): two anatomical streams arise from **posterior superior temporal gyrus (pSTG)**.
- **Ventral stream** (sound → meaning; comprehension/semantics): superior + middle temporal → semantic interface, via the **inferior fronto-occipital fasciculus (IFOF)** + **uncinate fasciculus** toward anterior temporal/ventral frontal. Damage → Wernicke / transcortical sensory aphasia.
- **Dorsal stream** (sensorimotor mapping for production + complex syntax): pSTG → **arcuate fasciculus / superior longitudinal fasciculus** → Broca (BA44) + ventral premotor (BA6). Damage → Broca / conduction aphasia.

**Friederici's syntax network** (Physiological Reviews 2025 "Brain structural networks underlying language"; Friederici "Neurobiology of Syntax as the Core of Human Language"): **BA44** (pars opercularis of Broca) is the **core syntax region** that "creates argument hierarchies as a sentence is computed"; it has strong direct connectivity to **posterior STG (A41/42, A22c)**. pSTG/pMTG is "inactive when artificial grammar sequences lacking semantics are processed but active when natural sentences are processed" — i.e. pSTG/pMTG supplies the *lexical-syntactic* content (the stored frames), BA44 does the *structure building*. Catalog **G.12** (Broca, `:2774`; Kandel Ch 55 pp 1382–1384): "supports comprehension of grammatically complex (non-canonical) sentences"; behavioral dissociation — "The girl that the boy is chasing is tall" FAILS (grammar-dependent), "The apple the girl ate was green" SUCCEEDS (semantically constrained).

**Hagoort's MUC model** (Hagoort 2013, *Front. Psychol.* 4:416, PMID 23874313; Hagoort 2005 "On Broca, brain, and binding", *TiCS*; the "Neurobiology of Language" chapter) — **the single most directly buildable framework for this cluster:**
- **Memory (M):** **temporal cortex + angular gyrus** store the laid-down knowledge — phonological word-forms, morphology, and crucially **"the syntactic templates associated with noun, verbs, adjectives"**. Each lexical item carries a **structural frame**. Explicitly **inspired by Joshi's (Lexicalized) Tree-Adjoining Grammar**: a verb's frame *is* its argument structure (its subcategorization tree). **This is the biology for GAP-A: the verb stores its frame in temporal-cortex memory.**
- **Unification (U):** **left inferior frontal gyrus (LIFG, Broca BA44/45)** dynamically **binds** the retrieved frames into a sentence-level structure — "generate larger structures from the building blocks retrieved from memory." Non-adjacent dependencies (filler-gap) "require on-line processing memory" — **the biology for GAP-C's WM-hold.**
- **Control (C):** frontal cortex + **ACC** + attention areas — select the context-appropriate structure/output, joint-action control.
- **ERP correlates:** **N400** ≈ Memory/lexical-semantic access (peaks ~400 ms, sensitive to semantic anomaly; ventral/temporal/ATL); **P600** ≈ Unification/syntactic integration + **reanalysis** (peaks ~600 ms, syntactic violation + **garden-path repair**; Broca/dorsal). (PLOS One 2014 task-effects on N400/P600; Brouwer neurocomputational N400/P600 model, PMC5484319.) **This gives a clean biological map: the project's "Memory" = stream cortex + lexical frames; "Unification" = the composer's binding; an honest model would predict an N400-like cost for an out-of-vocabulary concept and a P600-like cost for a structural reanalysis.**

### 3.2 Verb argument structure — the neural locus (GAP-A)
(Thompson/Bonakdarpour; Frankland & Greene; Frontiers Lang. Sci. 2023 "neurofunctional network of syntactic processing"):
- **Posterior superior temporal cortex** *retrieves* a verb's stored argument structure (the frame in MUC-Memory).
- **Left posterior perisylvian** cortex shows **graded activation with the NUMBER of thematic roles** a verb licenses (1-arg < 2-arg < 3-arg) — direct evidence that argument-count is a represented quantity.
- **Parietal cortex (posterior parietal sulcus)** assigns morphosyntactically-analysed constituents to thematic roles; TMS to L-PPS disrupts **thematic reanalysis** (PMC10158617).
- **Mid-superior temporal cortex** has **separable agent-selective and patient-selective populations** (it discriminates "dog bites man" from "man bites dog" — Frankland & Greene PNAS 2015/2020). **This is the biology that thematic roles are represented as distinct neural populations** — exactly the role-ensemble design the project already uses; the build is simply *more role ensembles*.
- **LIFG canonicity effect for ditransitives** (PLOS One MEG, "Left inferior frontal activations depending on canonicity of ditransitive argument structure") — Broca cost scales with non-canonical argument order.

### 3.3 Entity instances / type-token binding — the hippocampal index (GAP-B)
The mechanism for turning *types* into bound *tokens* is the **hippocampal episodic index**:
- **Concept cells** (Quian Quiroga; *Neuron* 2026 "20 years of concept cells"): single human MTL/hippocampal neurons respond invariantly to a *concept* (the **TYPE** — "Jennifer Aniston" across photos/name). These are the type representations.
- **Episodic binding via a sparse index ("barcode")** (eLife 2024 PMC11429605 "Barcode activity in a recurrent network model of the hippocampus enables efficient memory binding"; *TiCS* 2025 "On the origin of memory neurons"): forming an episode "requires binding together disparate elements that co-occur"; the hippocampus generates **"a sparse, high-dimensional pattern of activity transiently during the formation of each memory that serves as a unique index to which the contents of the memory are bound"**; reactivating the barcode recalls the bound contents. **A barcode IS a token:** it is a unique pointer that conjoins a set of co-active concept-types into one individuated instance. Catalog **D.02** (relational binding / Eichenbaum–Cohen "memory space", `:1098`; Kandel Ch 52 pp 1301–1302): events as **items-in-context**, distinguishing overlapping episodes that share elements ("same restaurant, different visits") without interference — **precisely the type/token distinction.** Catalog **D.14** (Tonegawa engram-tagging) — the project's **shipped** `start_engram_recording`/`commit_engram_tag`/`stimulate_tag` (`sim/bridge.py:3352/3381/3466`) **is a barcode mechanism**: a co-fired ensemble tag = a sparse index over the concepts active at encoding. **The build for GAP-B is to use an engram tag as the discourse token that binds the type "boy" + its distinguishing properties into "the_boy_1".**
- Kandel Ch 25 **binding problem** (catalog **G.03**, `:2658`): once an object is selected, features bind via **gamma-band synchrony / a parietal saliency map**; "illusory conjunctions" under load are binding errors. (The neural binding problem(s), Feldman/Treisman; PMC3538094.)

### 3.4 The ATL semantic hub — coherent concepts (GAP-E + concept quality)
(Lambon Ralph et al., *Nat. Rev. Neurosci.* 2016 "The neural and computational bases of semantic cognition"; Patterson hub-and-spoke; Pobric/Jefferies TMS):
- The **bilateral anterior temporal lobe (ATL)** is a **transmodal hub** that integrates modality-specific **"spokes"** (visual, auditory, motor, verbal) into **"coherent, generalizable concepts"** — "Coherent concepts are computed in the anterior temporal lobes" (Pobric et al. PNAS 2010 107:2717). Semantic dementia (ATL atrophy) → **impaired feature integration**, loss of fine concept distinctions. ATL also represents **verb argument/abstract structure** (HBM 2024 "ATL hub processing of abstract and concrete verbs"). **This is the biology of *why* concepts generalize** and grounds the project's hub-and-spoke convergence work (the PPMI stream cortex + cross-modal Hebbian convergence are the project's ATL-hub stand-in, per `CLAUDE.md` "GENERALIZATION across SIMILAR concepts").

### 3.5 Schemas / events — mPFC generalizes, hippocampus individuates (GAP-D + GAP-E)
(Gilboa & Marlatte; Baldassano event-schema; *Nat. Commun.* 2023 PMC9995562 "Flexible reuse of cortico-hippocampal representations"; *J. Neurosci.* 2018 "Representation of real-world event schemas"):
- **mPFC generalizes across instances of the same event SCHEMA** (the abstract type — "a giving event"), while the **hippocampus maintains event-SPECIFIC representations** (the token). mPFC encodes "temporal position within a schematic sequence" and assimilates new info into schemas. The 2023 result: an **Anterior Temporal Network** represents **people/entities** (generalizing across contexts), a **Posterior Medial Network** represents **context/situation**. **This double-dissociation (schema-type in mPFC, instance-token in hippocampus, entity in ATN, context in PMN) is the architectural blueprint for GAP-D/E.**
- **Tense/aspect:** grammatical aspect modulates event representations — perfective/imperfective shift mental models (Tandfonline 2025 "grammatical aspect on mental representations of events: ERP from English and Russian": aspect-adverbial mismatch → N400; Slavic aspect → P600). Tense/aspect is **grammatical inflection on the verb** (morphology) that conditions an event's temporal frame — i.e. an **attribute on the action/event node**, which the project's `attribute` role machinery can carry.

### 3.6 Point-neuron precedents + the binding-problem honest wall (the deepest question)
- **Variable binding / role-filler** has two main neurally-motivated families: **(i) Tensor Product Representations** (Smolensky 1990) and **Vector Symbolic Architectures / HRR** (Plate; Eliasmith Spaun) — bind role⊗filler, superpose, unbind by (approximate) inverse. **This is exactly the project's FHRR composer.** (ii) **Neural Blackboard Architecture** (van der Velde & de Kamps, *BBS* 2006) — sentence structure as **connection paths** between word-assemblies via gating circuits (Wilson-Cowan populations), an explicitly *structural* (not vector-algebraic) binding. **The honest caveat (PMC3538094 "The neural binding problem(s)"; arXiv 2012.05208):** there is **no fully-settled, learned, point-neuron solution to productive variable binding** — every approach trades off (VSA needs decorrelated full-precision codes + a fixed exact-inverse algebra; blackboard needs pre-wired connection structure; gamma-synchrony binding is capacity-limited). The project already lives this trade-off (the composer = "principled idealization, not a functional cortex", `CLAUDE.md`).
- **Assembly Calculus / NEMO** (Papadimitriou & Vempala, PNAS 2020 "Brain computation by assemblies of neurons"; Mitropolsky et al. "A Biologically Plausible Parser", *TACL* 2021; "The Architecture of a Biologically Plausible Language Organ" 2023; "Center-Embedding and Constituency in the Brain" NALOMA 2022): a **point-neuron** model (stylized neurons + synapses + plasticity + inhibition) with `project`/`merge`/`reciprocal-project`/`sequence` operations that **implements an English parser handling recursion + embedding**, and (2023) a **biologically-plausible account of language ACQUISITION**. **This is the existence proof that the comprehension/representation cluster is realizable in neurons+synapses on a point-neuron substrate** — directly relevant precedent for the project's parser + the embedded-clause build, and a candidate model to study for GAP-A/C.

---

## 4. Cheap-first, biology-grounded build options — ranked per capability

Every option below is **reuse-by-import** unless flagged; the project's strongest validated primitive is **role-filler binding + the no-confab moat**, and most of this cluster is "more roles / a token index / a WM-hold," not new substrate. Anti-cheat controls follow the project's load-bearing-controls-first discipline (the same that retracted the 2026-05-14 compose/transitive claims).

### GAP-A — verb argument structure / typed roles

**A1 (RECOMMENDED, cheapest) — extend the role alphabet + a per-verb frame table (MUC-Memory).**
Add thematic roles `GOAL, RECIPIENT, THEME, LOCATION, SOURCE, INSTRUMENT, TIME` to the composer's `ROLES` tuple (and a variable-arity `Fact` beside the fixed `Clause`). Give each verb a **stored frame** (its subcategorization: *go*→{AGENT,GOAL}; *give*→{AGENT,THEME,RECIPIENT}; *put*→{AGENT,THEME,LOCATION}) — a small host lexicon = MUC's temporal-cortex Memory (legitimate: the lexicon is the *dictionary*, not the cognition, exactly the boundary `FrameParser` already occupies). The extractor (`_corpus_svo_extract.py`) **keeps the preposition** and maps `prep+pobj` → the role the verb's frame licenses for that preposition (*to*→GOAL/RECIPIENT, *on*→LOCATION). Bind the filled frame with the **existing role-filler FHRR composer** — it is role-agnostic, so more roles cost only more codebook entries.
- **Reuse:** `RFPhasorComposer` binding (role-agnostic), `_corpus_svo_extract` (add prep retention), `FrameParser` (frame = which roles to expect). The selectional-preference / PMI `_plausible` gate already exists.
- **Biology:** MUC Memory (verb frame in temporal cortex) + Unification (composer bind) + graded argument-count activation (left perisylvian) + agent/patient-selective populations (mid-STC).
- **Anti-cheat:** (1) **frame-lesion** — scramble a verb's frame → role assignment for its obliques collapses (the frame is load-bearing, not word-position). (2) **held-out verb-frame generalization** — a verb seen in 2 frames must read a *3rd* sentence in either; assert filler disjointness. (3) **moat intact** — a wrong-role / absent-argument query → `None`, 0 false-accepts. (4) the role read-out is the **spiking** role ensemble (per `feedback_brain_based_only_standard`), not a host `if`.
- **Honest scope:** the *frame inventory* is hand-authored (a scaffold, like the develop syllabus) — variety, not learned-from-data grammar. Productive frame *learning* is A3.

**A2 — multi-cue competition over the richer roles (robustness for the new args).**
Extend `MultiCueRoleParser`'s competition to assign the *new* roles (the verb-fit cue already encodes selectional preference; add the preposition as a high-validity cue for GOAL/LOCATION). Reuse-heavy; the natural composition of the DONE multi-cue work with A1. Anti-cheat: the existing position-only-collapse + cue-lesion + permuted-cue battery, now on oblique roles.

**A3 (deep, research-gated) — learned argument-structure / Assembly-Calculus parser.**
Implement a Mitropolsky-Papadimitriou-style assembly parser (or a Chang-Dell-Bock dual-path BPTT-SNN, which the project has machinery for) that *learns* verb frames from corpus sentences. **Reward HIGH / effort HIGH / variance HIGH.** This is the genuinely-productive path (generalizes to unseen verbs/frames) and the real Broca/Unification analogue. Gate it behind a de-risk; VERIFY-gate every parse against the stored fact-base (form vs truth decoupling). Honest: the categorical-novelty gap (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`) is a nearby prior negative — treat as research-gated.

### GAP-B — entity instances (types vs tokens)

**B1 (RECOMMENDED, cheapest) — engram-tag "barcode" as the discourse token.**
For each newly-introduced entity ("a boy…"), `commit_engram_tag("boy_1", region_filter=…)` over the co-active concept pool + its distinguishing attributes — the tag is the sparse **episodic index (barcode)** that binds the *type* "boy" into a *token* "boy_1". Subsequent "the boy" re-`stimulate_tag`s it; "which boy?" disambiguates by which token-tag's attributes match. Store facts about the token by binding the **token-tag identity** into the AGENT slot instead of the bare type code.
- **Reuse:** the **shipped** D.14 engram API (`sim/bridge.py:3352/3381/3466`); the multi-turn `SpikingLoopContextBuffer` (`multi_turn_agent.py`) for holding the active discourse referents; the multi-referent disambiguation (the `2026-06-17-multireferent-disambiguation-NEGATIVE.md` named the exact next mechanism: WTA biased-competition between referent attractors — *reuse the biased-competition buffer*).
- **Biology:** hippocampal barcode index (eLife 2024) + concept cells (type) + Eichenbaum–Cohen items-in-context (D.02) + Tonegawa engram (D.14).
- **Anti-cheat:** (1) **two-token discrimination** — "a boy met a boy; the **tall** boy smiled" must bind to the correct token (collapses under tag-scramble). (2) **moat** — "which boy?" with no matching token → `None`. (3) the token index is a **spiking** engram (stimulate→reactivation measured in `cp_firing_states`), not a host dict key. (4) **multi-referent**: bare-pronoun disambiguation needs the WTA inhibition (the documented mechanism), NOT recency — guard with the recency-control that already exists.
- **Honest scope:** capacity — how many concurrent tokens can the engram store / the buffer hold? (Lisman-Idiart ~7 gamma slots is the biological bound; report it.) This is the project's *deepest* representational add and the highest-leverage for "real conversation."

**B2 — positional/contextual binding (catalog D.11, P4.1 substrate).**
The project's `positional_drive_pattern` + `ec_context→dg` already bind (word, position) into distinct CA3 ensembles (`validate_positional_binding.py`). A token = (type, discourse-position/context) ensemble. **Reuse** the P4.1 substrate; complements B1 (context-indexed tokens). Rank: secondary / combine with B1.

### GAP-C — wh-questions (filler-gap)

**C1 (RECOMMENDED) — wh-question as a filler-gap dependency, reusing the embedded-clause mechanism verbatim.**
A wh-question is mechanically a filler-gap dependency — *identical* to the relative-clause WM-hold the embedded-clause scoping already specified. The fronted wh-word ("where/who/what") is the **filler**, held in the **dlPFC NMDA WM-latch** (`cortex_ctx↔dlpfc_wm`, weight 30, NMDA-dependent, survives dt=1.0; `nav_conv_merged_bridge.py`) until the **gap** (the missing argument slot) is reached, then bound to that role → which becomes the *query role* for the composer. "where does the boy go?" → wh=GOAL filler, held, gap at *go*'s GOAL slot → `query_role(agent=boy, action=go, role=GOAL)`.
- **Reuse:** the dlPFC NMDA latch (the SAN/LAN working-memory hold — Kandel **G.08** PFC WM; the literature: sustained anterior negativity = WM for the wh-filler), `MultiFrameParser` (wh-frames are just more frames + a fronting cue), the composer's existing `query_*` (the wh-word selects which role to query). Each wh-word maps to the role it questions (who→AGENT, what→PATIENT/THEME, where→GOAL/LOCATION, when→TIME — needs GAP-A's roles).
- **Biology:** filler-gap = LIFG syntactic working memory (PMID 11523277; *Neurobiology of Language* 2022 sustained neural activity across dependencies); MUC Unification + on-line memory.
- **Anti-cheat:** (1) **held-out wh-questions** over stored facts; the answer must match the stored role-filler. (2) **moat** — a wh-question whose answer isn't stored → `None` (abstain, never fabricate the answer). (3) the filler-hold is the **spiking** dlPFC latch (lesion → the filler decays → wrong/absent binding). (4) no-fronting-cue baseline fails. **GAP-C depends on GAP-A** (the queried role must exist).
- **Honest scope:** single-gap, single-clause wh first; multi-gap / island constraints are the deep frontier (and the human ~2-level WM bound applies, same as center-embedding — an honest NEGATIVE there is the deliverable).

### GAP-D — tense / aspect / events

**D1 (cheapest) — tense/aspect as an attribute bound to the action/event node.**
Bind a `tense`/`aspect` value (past/present/future; perfective/progressive) as an attribute on the ACTION using the **existing `attribute` role machinery** — morphological inflection is a legitimate lexical front-end (the same boundary the parser's morphology already occupies). "was chasing" → action=chase + aspect=progressive.
- **Reuse:** the `attribute`/`attribute2` roles + the composer bind.
- **Biology:** aspect is grammatical inflection conditioning event structure (ERP: aspect→N400/P600, Tandfonline 2025); Kandel Ch 55 morphology in MUC-Memory.
- **Anti-cheat:** "did the dog chase the cat?" (past) vs a future-marked fact must dissociate; moat on unmatched tense.

**D2 (deeper) — events as first-class nodes (event individuation via engram, reuse B1).**
Make the *event* itself a token (a bound `(agent,action,patient,tense,GOAL,…)` composite gets its own engram-index) so it can be referred to ("the chasing happened yesterday"). This is the **event-schema/mPFC** story (a token event generalizable to a schema type). Reuse B1's barcode + the mPFC-generalizes/hippocampus-individuates blueprint. Rank: after B1.

### GAP-E — schemas / frames

**E1 (cheapest) — event-frame inventory = FrameNet-lite frames (reuse A1's frame table).**
A small inventory of event frames (giving = {GIVER,GIFT,RECIPIENT}; motion = {MOVER,GOAL,SOURCE,PATH}) — this is GAP-A's verb-frame table generalized to event types. The relational graph of facts sharing arguments **already IS an Eichenbaum-Cohen relational network** (D.02; `query_chain` traverses it) — a schema is the generalization over instances of that graph.
- **Reuse:** A1's frame machinery + the `query_chain` multi-hop relational walk + the develop-loop syllabus (cumulative).
- **Biology:** FrameNet frames; mPFC schema generalization (Nat. Commun. 2023); ATL coherent-concept hub.
- **Anti-cheat:** schema-inference must collapse under a relational-scramble control; the moat must still abstain on un-inferable gaps (never *fabricate* the unstated — a schema *predicts*, it must be flagged as inference, not asserted as a stored fact, per the DiscursiveTurn VERIFY gate).

**E2 (deep, research-gated) — learned schema abstraction (mPFC generalization).**
Learn schemas by Hebbian generalization over many stored instances (mPFC generalizes; hippocampus keeps tokens). Touches the generalization arc (substrate-capped at ~+0.065 Pearson @320, `2026-06-26-gen-readiness-bar-recalibration.md`). Research-gated.

---

## 5. TOP 2–3 highest-leverage build targets for this cluster (and why)

Ranked by **reward-per-effort**, given that the parsing front-end is already strong and the bottleneck is representational. All three are reuse-by-import; none needs the dendritic substrate; all keep the no-confab moat.

### ★ TARGET 1 — Typed-role / verb-frame representation (GAP-A: A1)
**Why highest-leverage:** it unblocks the largest fraction of *natural* facts at the lowest cost. Most real corpus sentences have PP/oblique arguments that the pipeline currently **discards at extraction**; "go to the park", "give X to Y", "put X on Y" are unrepresentable today. The fix is the project's *strongest validated primitive* (role-filler FHRR binding is role-agnostic) plus a small frame lexicon (MUC-Memory) and keeping the preposition in the extractor. It is the biology's central claim (Hagoort MUC: the verb stores its frame; argument-count is graded in left perisylvian cortex; agent/patient populations are separable in mid-STC). **It is also the prerequisite for wh-questions** (Target 3 queries the roles A1 creates) and **for event-frames** (E1 is A1 generalized). Effort: LOW (extend a tuple + a lexicon + extractor prep-retention + more codebook entries). Risk: LOW (no new mechanism; the bind is validated). The honest residual is that the frame *inventory* is a scaffold, not learned grammar (A3 is the productive path).

### ★ TARGET 2 — Entity instances via an episodic-index token (GAP-B: B1)
**Why high-leverage:** it is the **owner-flagged deepest gap** ("only a generic boy — can't handle 'which boy?'") and it is what separates "a fact database" from "a conversation about specific things." The mechanism is *already shipped* — the D.14 engram-tagging API is, functionally, the hippocampal **barcode** that binds co-active concept-*types* into an individuated *token*; the 2024 barcode literature and Quian Quiroga concept cells give it a clean biological grounding. Reuses the multi-turn context buffer + the biased-competition WTA (the documented next mechanism for multi-referent disambiguation). Effort: MEDIUM (wire engram-as-token into the agent's discourse loop; the disambiguation WTA exists). Risk: MEDIUM (capacity / multi-referent interference — but the Lisman-Idiart ~7-slot bound is the *honest biological* limit, so a capacity ceiling is a deliverable, not a defect). **This converts the brain from "knows types" to "tracks individuals," the core of reference and discourse.**

### ★ TARGET 3 — Wh-question comprehension as a filler-gap dependency (GAP-C: C1)
**Why high-leverage:** it directly closes the owner-flagged "rigid query syntax (no natural wh-questions)" gap, and it is **near-free** because the mechanism is *identical* to the already-scoped embedded-clause WM-hold (the dlPFC NMDA latch holding the wh-filler = the SAN/LAN syntactic working memory in the literature) composed with the DONE multi-frame parser and the existing `query_*` API. A wh-word simply selects which role to query (who→AGENT, where→GOAL — so it rides on Target 1's roles). Effort: LOW–MEDIUM (a fronting cue + the filler-hold + wh-word→query-role map). Risk: LOW for single-gap single-clause (the human multi-gap/island bound is the honest ceiling). **It makes the brain answer questions the way a person asks them**, and it reuses the WM-latch the project will build for embedded clauses anyway — strong synergy.

**Sequencing note:** Target 1 first (it is the prerequisite alphabet for 3 and the basis for E1/D1), then Targets 2 and 3 in parallel (independent: 2 is the discourse-token substrate, 3 is the question front-end). All three compound: typed roles + entity tokens + wh-questions together turn "recall a random SVO triple" into "ask where a specific individual went and get the grounded answer or an honest abstention."

---

## 6. Honest open risks / where biology is uncertain or a point-neuron wall is likely

1. **Productive grammar is the genuine wall.** All of §4's cheap options (A1/C1/E1) use **hand-authored frame inventories** — variety and structure, but **not learned, generalizing grammar**. The productive path (A3 learned argument-structure parser, E2 learned schemas) is a real mechanism-class build with documented nearby negatives (categorical-novelty gap) and must be research-gated. **Do not overclaim a frame scaffold as "the brain learned syntax."**
2. **Variable binding has no settled point-neuron solution.** The composer's exact-inverse FHRR algebra is a **principled idealization** (`CLAUDE.md`); TPR/VSA, neural blackboard, and gamma-synchrony binding each trade off (decorrelated codes / pre-wired structure / capacity limits — PMC3538094). Adding roles (Target 1) stays *within* the validated idealization (safe); but a *learned, generalizing* binder is the deep frontier the project has repeatedly mapped to the Mikulasch-Priesemann point-neuron limit.
3. **Entity-token capacity is bounded.** The barcode/engram + gamma-slot WM has a finite concurrent-token capacity (Lisman-Idiart ~7). This is a **biology-faithful** limit (human WM ~4±1 for active referents) — report it as the deliverable, don't brute-force it.
4. **Tense/aspect-as-attribute is a simplification.** Real aspect interacts with event structure (telicity, coercion) in ways an attribute tag won't capture; the ERP evidence (aspect→N400/P600) shows it modulates *event representation*, which D2 (events-as-objects) addresses more faithfully but at higher cost.
5. **Schemas risk fabrication.** A schema *predicts* the unstated ("he probably ate") — this MUST be flagged as inference and VERIFY-gated, never asserted as a stored fact, or it breaches the no-confab moat. The moat is content-agnostic; schema-inference is exactly the case where it must hold.
6. **The catalog is core-sim-dated.** The catalog marks the G cluster "missing", but the conversational runners are well past that — verify *current* runner state (as this doc did), not the catalog's status line, before scoping.
7. **No glossary file** exists at the referenced path; future docs should not cite it.

---

## 7. Catalog entries + sources cited

**Catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`):**
- **G.10** language as hierarchical symbolic system (`:2750`; Kandel Ch 55 pp 1370–1372) — missing.
- **G.11** dual-stream model — dorsal sensorimotor + ventral semantic (`:2762`; Kandel Ch 55 pp 1380–1387) — missing; the project's stream-cortex+composer = ventral-stream stand-in.
- **G.12** Broca's area — grammatical processing + non-canonical comprehension (`:2774`; Kandel Ch 55 pp 1382–1384) — the argument-hierarchy / center-embedding locus + behavioral dissociation.
- **G.13** Wernicke — auditory→semantic, prerequisite "semantic memory store" (`:2786`; Kandel Ch 55 pp 1384–1385) — missing.
- **G.03** object-based attention & feature binding (gamma synchrony, illusory conjunctions) (`:2658`; Kandel Ch 25) — missing; the binding problem.
- **G.08** PFC working memory — persistent activity, what/where/conjunctions (`:2726`; Kandel Ch 52 pp 1292–1294) — partial (flagship WM pool); the wh-filler/WM-hold substrate.
- **G.09** imagination/constructive memory (DMN) (`:2738`) — missing.
- **G.18** logLR accumulation in LIP — reliability-weighted evidence integration (`:2850`) — the multi-cue competition substrate.
- **G.02** active dendrites (`:2644`; Kandel Ch 13 pp 293–298) — **missing; the largest abstraction** (the deep learned-binder frontier).
- **D.01** episodic memory encoding/consolidation (`:1085`; Kandel Ch 52 pp 1296–1302) — missing as a system.
- **D.02** relational binding / "memory space" / Eichenbaum-Cohen (`:1098`; Kandel Ch 52 pp 1301–1302) — missing; the type/token + items-in-context mechanism.
- **D.03** trisynaptic loop (`:1111`; Kandel Ch 54 pp 1340–1342) — DG sparsify / CA3 complete / CA1 output (the index substrate).
- **D.14** Tonegawa engram-tagging — **SHIPPED API** (`sim/bridge.py:3352/3381/3466`) — the barcode/token mechanism for GAP-B.
- **J.34** schemas/gist/false memory (`:3903`; Kandel Ch 52 pp 1306–1308) — not-applicable-flagged; the schema-prediction-vs-moat tension.

**Kandel 6e:** Ch 52 (Learning & Memory: episodic, schemas, working memory), Ch 54 (Hippocampus & explicit memory storage — PROJECT-CRITICAL), Ch 55 (Language — dual stream, Broca, Wernicke, MUC-relevant), Ch 56 (decision-making — signal detection / criterion). Textbooks dir: `E:\Documents\Projects\sim-catalog\references\textbooks\` (kandel-pns-6e, okeefe-nadel, buzsaki, etc.).

**Literature (WebSearch, June 2026):**
- Hagoort, *MUC (Memory, Unification, Control) and beyond*, Front. Psychol. 2013 (PMID 23874313) + *On Broca, brain, and binding*, TiCS 2005 — M+U+C; verb structural frames in temporal-cortex Memory (Tree-Adjoining Grammar); Broca=Unification; non-adjacent dependencies need WM.
- Friederici / *Brain structural networks underlying language*, Physiol. Rev. 2025 (doi 10.1152/physrev.00004.2025) — BA44 core syntax / argument hierarchy; pSTG-BA44 connectivity.
- Hickok & Poeppel; Saur et al., *Ventral and dorsal pathways for language*, PNAS 2008 (PMC2584675).
- Lambon Ralph et al., *The neural and computational bases of semantic cognition*, Nat. Rev. Neurosci. 2016 (wiredbrains.org PDF); Pobric et al., *Coherent concepts computed in ATL*, PNAS 2010 107:2717 — hub-and-spoke.
- Thompson/Bonakdarpour; *neurofunctional network of syntactic processing*, Front. Lang. Sci. 2023; Frankland & Greene (agent/patient populations in mid-STC); thematic reanalysis in L-PPS (PMC10158617) — verb argument structure neural loci.
- Quian Quiroga, *20 years of concept cells*, Neuron 2026; *On the origin of memory neurons*, TiCS 2025; barcode index model, eLife 2024 (PMC11429605) — type/token episodic binding.
- Filler-gap / wh: syntactic WM ERP/fMRI (PMID 11523277); *Sustained neural activity across syntactic dependencies*, Neurobiology of Language 2022 (PMC10158612) — SAN/LAN = WM-hold.
- N400/P600: PLOS One 2014 task-effects; Brouwer neurocomputational N400/P600 (PMC5484319); aspect→event representations ERP, Tandfonline 2025.
- Event schemas: *Flexible reuse of cortico-hippocampal representations*, Nat. Commun. 2023 (PMC9995562); Baldassano event-schema, J. Neurosci. 2018 — mPFC generalizes / hippocampus individuates.
- Binding problem: *The neural binding problem(s)* (PMC3538094); Smolensky TPR 1990; van der Velde & de Kamps Neural Blackboard Architecture, BBS 2006; *On the Binding Problem in ANNs* (arXiv 2012.05208).
- Point-neuron parser precedent: Papadimitriou & Vempala, *Brain computation by assemblies of neurons*, PNAS 2020; Mitropolsky et al., *A Biologically Plausible Parser*, TACL 2021 (arXiv 2108.02189); *Architecture of a Biologically Plausible Language Organ* 2023; *Center-Embedding and Constituency in the Brain*, NALOMA 2022.

**Project files / findings reviewed (file-cited):** `rf_phasor_composer.py:24,28` (role alphabet + fixed Clause — the load-bearing gap), `_corpus_svo_extract.py:19,74` (oblique-collapse), `case_aware_role_parser.py:12` ({agent,action,patient} only), `brain_conversational_agent.py:25` (BridgeParser), `frame_parser.py` / `multicue_role_parser.py` / `attributed_parser.py`; findings `2026-06-18-frame-selection-GO.md`, `2026-06-18-multiframe-comprehension-GO.md`, `2026-06-19-embedded-clause-parsing-scoping.md`, `2026-06-20-robust-multicue-parser-deep-research.md`, `2026-06-26-deep-knowledge-brain-fluency-research.md`, `2026-06-17-multireferent-disambiguation-NEGATIVE.md` (the WTA disambiguation mechanism), `CLAUDE.md` (conversational + generalization sections).
