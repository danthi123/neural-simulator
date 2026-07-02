# Research gate — open-domain SURFACE FLUENCY: the last residual on the toward-language path (2026-07-02)

**Read-only deep-research gate (mandated before accepting the "surface fluency is the transformer's last unique job"
verdict). This doc is the only write; no code edited; not committed.** Context: the emergent HTM Temporal-Memory sequence
cortex now runs fully on the spiking `SimulationBridge` and, all 6-seed GO, no `sim/` edit, covers PREDICTION
(`2026-07-02-emerge15-word-sequence-lm-GO.md`), PRODUCTION/autoregressive generation (`-emerge16-word-generation-GO.md`),
GENERALIZATION (`-emerge17`, `-emerge19-real-ppmi-generalization-GO.md`), HIGH-ORDER sequence generalization
(`-emerge18-sequence-generalization-GO.md`), and GROUNDED production with an INTRINSIC no-confab moat
(`-emerge20-grounded-moat-GO.md`). Every prior gate named the SAME remaining residual: **open-domain surface fluency** —
arbitrary-topic, connected, grammatical, multi-word prose. This gate isolates that residual, reframes it via biology,
ranks cheap-first paths on THIS substrate, and delivers a verdict.

---

## BOTTOM LINE (verdict, expanded in §4)

**"Surface fluency" is NOT one thing, and the part that is genuinely *surface form* is largely surpassable on this
substrate — cheaply.** Decomposed (§1), five concrete pieces make up "fluency": (i) syntactic FRAME / word order,
(ii) agreement & morphology, (iii) connectives / discourse cohesion, (iv) recursion / embedding, (v) novel-sentence
generativity (systematic recombination). Four of the five are **already covered or cheaply reachable** by shipped,
substrate-validated machinery — critically, the **`LearnedFrameGrammar`** (a construction-grammar-over-roles,
6/6-seed GO on real spikes, `2026-06-17-learned-multiframe-word-order-derisk.md`), the CQ serial-order renderer, the
discourse-plan connective/aggregation layer, and the recursive-clause composer. The **decisive reframe** (§2) is that a
**high-order sequence model over word CLASSES (parts of speech) with open slots IS, literally, a construction grammar**
— and the sequence cortex already does high-order prediction + generalizing codes. So the cheapest genuinely-new step is
to make the emergent cortex learn sequences over **POS/word-class frames** (not localist word columns), so novel content
words slot into learned frames → **novel grammatical sentences from learned frames**, which is exactly what surface
fluency IS on the production side.

The **genuinely-hard core that is NOT surface form** is: **open-domain SEMANTICS / world knowledge** — deciding *what to
say* about an arbitrary topic and whether it is TRUE. That is a different faculty (Wernicke/semantic memory + grounding +
the moat), it is the field's open frontier, and the honest posture is the project's existing one (grounded-retrieval +
checkable-inference + say-where-knowledge-ends), NOT "keep the transformer." **Surface form is surpassable; open-world
semantics is the real residual, and it is a knowledge/grounding problem, not a grammar problem.**

**THE SINGLE CHEAPEST NEXT DE-RISK (§4):** a new `_emerge21_pos_frame_grammar_derisk.py` — the emergent sequence cortex
learns sequences over **POS-class columns with open content slots** on a tiny POS-tagged grammar, then GENERATES a
**NOVEL grammatical sentence** (content words never seen in that frame) by rolling out the POS frame and filling each
slot's word-class with a generalizing lexical code; anti-cheated by held-out recombination + a grammaticality check +
permuted-POS control + lesion + no-teacher + 6-seed. Reuse-by-import; NO `sim/` edit.

---

## 1. ISOLATE the true residual — decompose "surface fluency", tag each HAVE / CHEAP / HARD-CORE

"Open-domain surface fluency" is habitually treated as one monolithic blocker. Decomposed into the actual linguistic
sub-capabilities, most of it is already built. Each piece: what it is · the project machinery that covers it (verified) ·
status.

| # | Fluency piece | What it is | Shipped project machinery (verified) | Status |
|---|---|---|---|---|
| F1 | **Syntactic FRAME / word order** | serial order of constituents (SVO, VSO, question order …) | `LearnedFrameGrammar` (`_phaseB_learned_multiframe_word_order_derisk.py`): a frame = a **LEARNED** per-frame primacy gradient over grammatical-ROLE slots, cue→frame SELECTION learned, emitted by the rate-coded CQ read-out (`neural_serial_order_renderer.py`). 6/6-seed GO on real spikes; permuted-frame + lesion collapse to chance, moat holds (`2026-06-17-learned-multiframe-word-order-derisk.md`) | **HAVE** (for roles/slots) |
| F2 | **Agreement & morphology** | subject-verb agreement, number/gender, inflection | none directly; BUT agreement = a **learned sequence CONSTRAINT** (an earlier feature predicts a later form) = the HTM's native high-order competence (§2c). No mechanism wired yet | **CHEAP** (a sequence constraint the HTM already learns) |
| F3 | **Connectives / discourse cohesion** | join facts into connected prose (and/but/so; aggregation) | `plan_discourse`/`compare_discourse`/`shared_discourse` (`_fluidconv_phase16_discourse_plan_derisk.py`): aggregate same-relation + Joint/Elaboration/Contrast/Additive connectives, entailment-checked, moat by construction (`2026-07-01-fluid-conversation-phase16-…GO.md`) | **HAVE** (host-authored inventory; neural producer is a deep follow-on) |
| F4 | **Recursion / embedding** | center-embedding, constituent-as-slot-filler, arbitrary depth | depth-2 recursive clauses with flat inner args (`nested_composition_agent`, `query_chain`) validated; the HTM itself is a high-order sequence memory (finite embedding) | **HAVE (bounded)** / HARD past depth-2 (see below) |
| F5 | **Novel-sentence generativity** | systematic recombination — grammatical sentences never seen | the missing piece as currently built: emerge15-20 use **localist per-word columns** → memorize/generalize *word associations*, but do not slot novel content words into learned FRAMES. This is the true residual on the production side | **CHEAP-NEW** (§2b, §3a) — the POS-frame step |

**Two genuinely-hard sub-cores, both NAMED and both NOT "surface form":**

- **Arbitrary recursion depth + non-adjacent agreement across embeddings** collapse onto the project's *already-mapped*
  point-neuron walls — the **nested-composition/SNR wall** (`2026-06-02-full-320…hierarchical null`) and the
  **multi-attribute BUNDLING NEGATIVE** (`2026-06-16-onsubstrate-learned-binder…bundling-NEGATIVE.md`). Both reduce to
  the same superposition-inverse / multiplicative-binding operation the project has four-times found to be the
  point-neuron limit. This is the *hard half* of syntax the productive-syntax scoping already parked
  (`2026-06-17-productive-syntax-scoping.md` §4) with two routes past it (dendritic multiplication / Assembly-Calculus
  disinhibition control). **But humans also struggle past ~depth-2 center-embedding** (Kandel 6e G.12 behavioral
  validation: "the girl that the boy is chasing is tall" *fails* comprehension) — so bounded recursion is the *biological*
  target, and the substrate already reaches it.

- **Open-domain SEMANTICS / world knowledge** — choosing *what to say* on an arbitrary topic and whether it is TRUE.
  **This is the real residual, and it is NOT surface form.** It is the Wernicke/semantic-memory + grounding + moat
  faculty (F3's host-authored inventory is a symptom: the *content* of a connective is entailment-checked against stored
  facts, i.e. it bottoms out in the knowledge base, not the grammar). Free single-pass abstractive synthesis + open-world
  cross-fact inference is a documented boundary (`2026-07-01-fluid-conversation-phase16-…GO.md` "the genuine wall") and
  the field's open frontier.

**Conclusion of the isolation.** The residual that is *actually surface form* — F1, F2, F4(bounded), F5 — is
**covered or cheaply reachable**. F1/F3/F4 are shipped; F2/F5 are the HTM's native high-order competence needing only a
POS-frame encoding (§2b). The thing repeatedly called "surface fluency" that is NOT yet built is **F5 (systematic
recombination via learned frames)**, and it is cheap on this substrate. What is genuinely hard — deep recursion+agreement
(a known parked wall with two routes) and open-world semantics (a knowledge/grounding problem) — is **not surface form**.

---

## 2. REFRAME via real biology — how the brain produces fluent grammatical speech, and why the residual is small

### 2a. Language-area anatomy — the production/comprehension split (cited)
From the catalog (Kandel 6e Ch 55, dual-stream / Hickok-Poeppel), verified from
`sim-catalog/references/feature-catalog.md`:
- **G.11 dual-stream** (pp 1380-1387): DORSAL stream (posterior superior-temporal → arcuate fasciculus → Broca) =
  sensorimotor mapping for PRODUCTION + complex-syntax sequencing; VENTRAL stream = sound→meaning COMPREHENSION.
- **G.12 Broca's area** (pp 1382-1384): "maps stored word-forms → sequential motor articulation; supports comprehension
  of grammatically complex (non-canonical) sentences." Damage → *labored, agrammatic* speech, **retained noun selection,
  lost function-word/verb use** — i.e. Broca is the SERIAL-ORDER / grammatical-frame engine (the F1/F5 role), and its
  loss spares the lexicon (Wernicke) but destroys the frame. Behavioral validation is a *bounded-recursion* dissociation.
- **G.13 Wernicke's area** (pp 1384-1385): "selects words matching intended meaning"; damage → *fluent but paraphasic*
  speech (grammar intact, content wrong) — i.e. Wernicke is the lexical-SELECTION / semantic role (the open-world
  content faculty). **The Broca/Wernicke double dissociation IS the surface-form vs semantics split:** you can have
  fluent grammar with wrong content (Wernicke damage) or correct content in broken grammar (Broca damage). This is the
  anatomical proof that surface form and open-world semantics are **separate faculties** — exactly the decomposition in §1.
- **G.10 hierarchical symbolic system** (pp 1370-1372): "finite units → infinite combinations via syntactic rules" —
  which is precisely what a high-order sequence predictor over a FRAME alphabet provides (F5).

**Mapping to the substrate:** HTM-TM = the temporal-cortex/Broca predictive-sequence engine (F1/F2/F4/F5 surface form);
the generalizing stream-cortex codes = the Wernicke lexical store; the grounded-knowledge + moat = the semantic/content
faculty (the hard core). **The transformer conflated all of these; the substrate splits them the way the brain does.**

### 2b. The decisive reframe — a high-order sequence model over WORD CLASSES with open slots IS a construction grammar
The usage-based / construction-grammar literature (Goldberg; Tomasello; Diessel *The Grammar Network*) models grammar as
**learned sequence frames with open slots constrained to word classes** — "constructions are constraint-based
representations in which individual slots are limited to particular syntactic/semantic/lexical fillers, represented as a
**sequence of slot-constraints**" (usage-based construction grammar, verified via web search 2026-07-02). Production =
retrieve the most-entrenched construction and fill its slots; a child produces a *novel* grammatical sentence by slotting
a new content word into a frequent frame (a "constructional frame — a frequent lexical pattern with an open slot").
Computational-linguistics grammar induction confirms the machine version: **"predict the basic word-order facts of a
novel language given only a corpus of part-of-speech (POS) sequences"** (POS-sequence grammar induction, web search
2026-07-02) — i.e. a high-order sequence model over POS categories *recovers the grammar*.

**Why this is load-bearing here:** the emergent cortex ALREADY does (i) high-order, context-specific sequence prediction
(EMERGE-15) and (ii) generalizing similarity-structured codes so similar tokens share micro-columns (EMERGE-17/19). If
the tokens fed to it are **word-CLASS symbols (POS)** rather than localist words, then:
- learning "DET ADJ NOUN VERB DET NOUN" sequences = learning the *frame* (the construction);
- generalizing codes let a *novel* content word (whose code overlaps its class) fill the class slot;
- rolling out the frame (EMERGE-16 excitability replay) + filling each slot with a class-consistent word =
  **generating a NOVEL grammatical sentence never seen in that frame = systematic recombination (F5)**.
So "surface fluency" on the production side is **the HTM's native computation with a POS/word-class alphabet** — not a
distant capability bolted on. This is the same insight the productive-syntax scoping reached from a different angle
(`2026-06-17-productive-syntax-scoping.md`: "a frame is a LEARNED primacy gradient over ROLE slots; order over ROLES not
words → generalizes to fillers never trained"), already GO for the ORDER read-out; the POS-frame sequence cortex is the
COMPREHENSION/PREDICTION analogue that supplies which frames exist.

### 2c. Serial-order PRODUCTION + agreement — competitive queuing and learned sequence constraints (cited)
When the output is a SET of items to be ordered, the biology is **competitive queuing** (Grossberg 1978; Bullock &
Rhodes 2003; Averbeck et al. 2002/2003 recorded parallel primacy-graded ensembles in PFC whose relative strength predicts
serial position; recent human ECoG confirms speech-motor sequencing in precentral gyrus — *Nat Hum Behav* 2025, "Speech
sequencing in the human precentral gyrus", web search 2026-07-02). The project de-risked this on-substrate (rate-coded CQ,
6/6 GO). **Agreement (F2)** is, biologically and computationally, a **long-range sequence constraint** — an earlier
feature (subject number) predicts a later form (verb inflection) — which is exactly what a *high-order* sequence memory
learns (the LSTM-agreement literature, *"Neural Networks as Cognitive Models of the Processing of Syntactic Constraints"*,
PMC 2024, web search 2026-07-02, shows agreement emerges from next-word prediction). On the HTM this is the SAME
high-order coincidence that carries the EMERGE-15 subject→branch dependency four words back — no new mechanism, just a
corpus with agreement structure.

### 2d. Where the honest wall is (not surface form)
The brain produces fluent grammar (Broca/CQ/frames) *independently* of whether the content is true or apt (Wernicke
paraphasia = fluent + wrong). The genuinely-hard, field-open residual is the **content/knowledge** faculty: open-domain
*what to say* and *is it true*. The project's posture there is already correct and biology-grounded: grounded-retrieval +
checkable connectives + abstention (the intrinsic moat, EMERGE-20). The path past open-world semantics is **more learned
knowledge + grounding**, not more grammar and not the transformer.

---

## 3. RANK cheap-first paths to fluent connected production ON THIS SUBSTRATE

Ordered by implementation cost. Each: mechanism · citation · smallest de-risk · anti-cheats · `sim/`-edit-or-not.

### (a) ★ CHEAPEST + highest-leverage — POS/word-class FRAME sequence cortex (systematic recombination; F5)
- **Mechanism:** feed the emergent HTM-TM sequences of **POS-class columns** (DET/ADJ/NOUN/VERB/…) with open content
  slots, instead of localist per-word columns; learn frames unsupervised (the committed `fused_htm_permanence_update`);
  GENERATE a novel sentence by rolling out a learned frame (EMERGE-16 excitability replay) and filling each class slot
  with a class-consistent content word via the generalizing lexical codes (EMERGE-17/19). This makes the cortex produce
  grammatical sentences with content words it never saw in that frame = a construction grammar.
- **Citation:** usage-based construction grammar (Goldberg; Diessel, *The Grammar Network* — filler–slot relations);
  POS-sequence grammar induction (grammar recoverable from POS sequences); Kandel 6e G.10/G.12 (hierarchical symbolic
  frames / Broca); Bouhadjar-Diesmann 2022 (the ported HTM substrate: high-order prediction + excitability replay);
  the project's own `2026-06-17-learned-multiframe-word-order-derisk.md` (learned-frame ORDER already GO).
- **Smallest de-risk:** a tiny POS-tagged grammar (e.g. `NOUN VERB NOUN` and `DET ADJ NOUN VERB DET NOUN`) over ~2 frames
  with a small class-tagged lexicon; TRAIN unsupervised on some content-word instantiations; TEST whether the cortex
  generates a **held-out recombination** — a grammatical sentence using content words never paired in that frame in
  training (e.g. train "dog chased cat", "wolf saw bird"; generate a grammatical "fox chased bird" it never saw). Score:
  (1) the generated POS sequence matches a learned frame (grammaticality), (2) each slot is filled by a class-consistent
  word, (3) the specific content combination was HELD OUT.
- **Anti-cheats (all mandatory):** (1) **held-out recombination** — the exact content combination never trained (the F5
  discriminator: novel-but-grammatical, not memorized); (2) **permuted-POS control** — shuffle the class→column mapping →
  frames become non-grammatical → generation collapses (the order was the learned frame, not a fixed bias); (3)
  **class-derangement control** — assign words to wrong classes → slot-filling produces non-class-consistent (ungrammatical)
  output → collapses; (4) **dAP-lesion** → generation halts (frame recurrence load-bearing); (5) **no-teacher** (as
  rung-4); (6) **6-seed** (42/43/44/100/101/102). A grammaticality check = "generated POS sequence ∈ the learned frame
  set" (a checkable predicate, not a host grammar judgment).
- **`sim/` edit:** **NONE** — reuse-by-import (`build_pool_bridge`/`OnBridgeLearner`/`coincidence_predict` from
  `_emerge14`; generalizing codes from EMERGE-17/19; excitability replay from EMERGE-16). The POS-column encoding + the
  slot-fill are a runner-side wiring choice.

### (b) couple the frame cortex to the CQ serial-order renderer + `LearnedFrameGrammar` (production read-out; F1)
- **Mechanism:** the POS-frame cortex (a) supplies WHICH frame; the already-GO `LearnedFrameGrammar` +
  `NeuralSerialOrderRenderer` supply the neural word-ORDER read-out (graded current → spiking-rate ranking) and the
  cue→frame selection. This closes the loop: a learned frame → ordered emission → per-slot A→W spelling.
- **Citation:** Grossberg 1978 / Bullock-Rhodes 2003 / Averbeck 2002 (competitive queuing); Kandel G.07/H.19 (pre-SMA/SMA
  sequence production); the project's `2026-06-17-learned-multiframe-word-order-derisk.md` (6/6 GO).
- **Smallest de-risk:** feed (a)'s selected frame into `LearnedFrameGrammar.emit_spiking`, spell each ordered slot via the
  A→W read-out, verify the produced sentence re-parses to the frame; held-out fillers; permuted-frame + lesion collapse.
- **Anti-cheats:** reuse the `_phaseB_learned_multiframe` suite (held-out fillers, permuted-frame, lesion, moat); 6-seed.
- **`sim/` edit:** **NONE** (both pieces reuse-by-import; already validated GO).

### (c) connective / multi-clause cohesion via the discourse-plan machinery (F3)
- **Mechanism:** join the grounded frame-generated sentences into connected prose using the existing
  `plan_discourse`/`compare_discourse` aggregation + entailment-checked connectives; recursion via the recursive-clause
  composer where depth ≤ 2.
- **Citation:** Levelt (macroplan→microplan→realize); Reiter-Dale NLG; the project's
  `2026-07-01-fluid-conversation-phase16-discourse-plan-synthesis-GO.md` (GO, moat by construction).
- **Smallest de-risk:** already GO; the new work is only re-pointing the aggregation inputs at frame-generated sentences
  and confirming grounding + moat hold. The *neural* connective PRODUCER (self-organized RST relations) is a deep
  follow-on, not on the cheap path.
- **Anti-cheats:** grounded (0 invented tokens); connective-correct (entailment-checked); lesion→hedge; 6-seed.
- **`sim/` edit:** **NONE**.

### (d) agreement / morphology as learned high-order sequence constraints (F2)
- **Mechanism:** include agreement structure in the training corpus (subject-number → verb-form dependency across the
  frame) so the HTM's high-order coincidence learns the constraint — the SAME mechanism that carries the EMERGE-15
  subject→branch dependency; extend to short-range morphology (stem+affix as a two-symbol sub-sequence).
- **Citation:** LSTM-agreement cognitive-model literature (agreement emerges from next-word prediction, PMC 2024);
  Kandel G.10 (morphology as combinable units); Bouhadjar high-order prediction.
- **Smallest de-risk:** a corpus where verb form depends on subject number across an intervening word; measure held-out
  agreement accuracy vs a bigram floor that cannot see the subject (mirrors EMERGE-15's n-gram-floor design).
- **Anti-cheats:** bigram/adjacent floor (agreement must beat order-blind), swap-follows-subject (validate-by-function),
  lesion, no-teacher, 6-seed.
- **`sim/` edit:** **NONE** (a corpus + encoding choice).

### (e) vocabulary/frame SCALE — the sparse multi-segment pool (R2), the one deferred `sim/` extension
- **Mechanism:** the dense O((cols·nE)²) cross-column potential pool is the vocab/frame-scale wall (the toward-language
  gate §2e); the canonical HTM fix (multiple distal segments per cell, each subsampling a sparse fixed potential set)
  scales capacity with segments not vocab². Needed only when real POS-frame corpora exhaust the dense pool.
- **Citation:** Hawkins-Ahmad 2016 (multi-segment HTM cells); Bouhadjar 2022 (per-neuron dendritic branches); catalog
  D.18 (committed three-term rule).
- **Smallest de-risk:** sparse-vs-dense next-frame parity at small scale + a sub-quadratic synapse-count curve.
- **Anti-cheats:** dense-pool parity, capacity curve, lesion, 6-seed.
- **`sim/` edit:** **YES** — an additive per-cell segment structure in the coincidence pathway + a per-segment gather
  kernel. The honest, biology-grounded scale mechanism (fair game per the master directive), **deferred** until (a)-(d)
  prove the frame grammar and scale demands it.

### (f) HARD CORE (not surface form; flagged, not a near-term build) — deep recursion+agreement & open-world semantics
- **Deep recursion + non-adjacent agreement across embeddings** → the parked point-neuron walls (nested-composition SNR;
  multi-attribute bundling), two routes past: dendritic multiplication (D2 two-compartment substrate) OR Assembly-Calculus
  disinhibition recursion control (Mitropolsky center-embedding, arXiv:2206.13217; the productive-syntax scoping §4 fork).
  Note humans also fail past ~depth-2 (G.12), so bounded recursion is the biological target the substrate already meets.
- **Open-world semantics / world knowledge** — the real residual; a knowledge/grounding faculty (more learned knowledge +
  grounded-retrieval + checkable inference + abstention), NOT a grammar problem and NOT the transformer.

---

## 4. VERDICT — is surface fluency surpassable on the substrate, and the single cheapest next de-risk

**Surface form is surpassable on this substrate, and cheaply. It is not a wall.** The reframe (§2b) is decisive: a
high-order sequence cortex over **POS/word-class frames with open slots is a construction grammar**, and the emergent
cortex already has the two hardest pieces (high-order context-specific prediction + generalizing similarity-structured
codes) plus generation (excitability replay) and a validated learned-frame ORDER read-out (`LearnedFrameGrammar`, 6/6 GO).
Four of the five surface-fluency pieces (F1 frame/order, F3 connectives, F4 bounded recursion, and — cheaply — F2
agreement) are already covered or one corpus-encoding step away; the genuinely-new piece (F5 systematic recombination) is
the POS-frame sequence-cortex step, reuse-by-import, no `sim/` edit. **The transformer's "surface fluency" job is
therefore surpassable** — it decomposes into pieces the substrate already does the way the brain does (Broca/CQ/frames).

**What is genuinely hard is NOT surface form:** it is **open-domain semantics / world knowledge** (what to say on an
arbitrary topic and whether it is true) — a separate faculty (the Broca/Wernicke double dissociation, §2a), the field's
open frontier, addressed by more learned knowledge + grounded-retrieval + checkable inference + abstention (the intrinsic
moat), never by the transformer. Deep recursion+agreement is a *separately parked* known wall with two routes (dendritic /
Assembly-Calculus), and bounded recursion (the human-biological target) is already met.

**THE SINGLE CHEAPEST NEXT DE-RISK (path (a), do this first):**

> **Runner:** new `research/runners/_emerge21_pos_frame_grammar_derisk.py` (reuse-by-import: `build_pool_bridge` +
> `OnBridgeLearner`/`coincidence_predict` from `_emerge14`; the generalizing-code encoding from EMERGE-17/19; the
> excitability-replay generation from EMERGE-16).
>
> **Task:** a tiny POS-tagged grammar with **word-class columns + open content slots** — e.g. two frames
> `NOUN VERB NOUN` and `DET ADJ NOUN VERB DET NOUN` over a small class-tagged lexicon (nouns {dog,cat,wolf,bird,fox},
> verbs {chased,saw,ate}, dets {the,a}, adjs {big,small}). TRAIN unsupervised on a SUBSET of content-word
> instantiations (each word encoded so its code overlaps its class's micro-columns), holding out specific content
> COMBINATIONS.
>
> **Learn + generate:** unsupervised, teacher-free, on the bridge (`fused_htm_permanence_update`); then cue a frame and
> roll it out (excitability replay), filling each class slot with a class-consistent content word via the generalizing
> codes.
>
> **Measure (the F5 deliverable):** does the cortex GENERATE a **held-out recombination** — a grammatical sentence whose
> content-word combination was NEVER trained in that frame — where (1) the generated POS sequence ∈ the learned frame
> set (grammaticality, a checkable predicate), (2) each slot is filled by a class-consistent word, (3) the combination
> was held out? This is "novel-but-grammatical", the exact thing surface fluency IS.
>
> **Config:** `vocab` = #classes×columns, `nE`=16, `act_th`=3, rung-4's validated HTM settings; CPU numpy for the
> de-risk, GPU for the confirm.
>
> **Anti-cheats (all mandatory):** (1) **held-out recombination** (novel content combo — the F5 discriminator);
> (2) **permuted-POS control** (shuffle class→column → frames non-grammatical → generation collapses); (3)
> **class-derangement control** (words in wrong classes → slot-fills non-class-consistent → collapses); (4)
> **dAP-lesion** → generation halts; (5) **no-teacher** (as rung-4); (6) **6-seed** (42/43/44/100/101/102).

If (a) is GO, chain (b) the `LearnedFrameGrammar`/CQ ordered read-out (already GO — a wiring), (c) discourse connectives
over frame-generated sentences (already GO — a re-point), and (d) agreement-as-sequence-constraint (a corpus). The one
`sim/` extension (e, sparse multi-segment pool) is deferred until real-corpus frame scale demands it. The hard core
(deep recursion+agreement; open-world semantics) is the SEPARATE next gate — and it is a knowledge/grounding problem, not
a surface-form one; the honest posture is grounded-retrieval + checkable inference + abstention, never the transformer.

---

## Artifacts / key citations
- **Substrate + chain:** `_emerge14_stageC_onbridge_learning_derisk.py`; `sim/kernels.py`
  (`fused_htm_permanence_update`); `2026-07-02-emerge{15,16,17,18,19,20}-*.md`;
  `2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`.
- **Reusable surface-form machinery:** `_phaseB_learned_multiframe_word_order_derisk.py` +
  `2026-06-17-learned-multiframe-word-order-derisk.md` (learned construction-grammar-over-roles, 6/6 GO);
  `neural_serial_order_renderer.py` (CQ serial-order); `_fluidconv_phase16_discourse_plan_derisk.py` +
  `2026-07-01-fluid-conversation-phase16-discourse-plan-synthesis-GO.md` (connectives/aggregation);
  `nested_composition_agent.py` (recursive clause, depth-2); `_phaseB_stream_codes_320_seed42.npy` (generalizing codes).
- **Prior scoping (adopt, not reinvent):** `2026-06-17-productive-syntax-scoping.md` (easy/hard-half decomposition;
  Assembly-Calculus option; the parked walls).
- **Literature:** Goldberg (usage-based construction grammar); Diessel, *The Grammar Network* (filler–slot relations,
  Cambridge); Tomasello (usage-based acquisition; constructional frames with open slots); POS-sequence grammar induction
  (grammar recoverable from POS sequences); Bouhadjar, Wouters, Diesmann, Tetzlaff 2022, *PLoS Comput Biol* 18(6):e1010233
  (HTM sequence learning/prediction/replay); Grossberg 1978 / Bullock & Rhodes 2003 / Averbeck et al. 2002-2003
  (competitive queuing); *Nat Hum Behav* 2025 "Speech sequencing in the human precentral gyrus"; "Neural Networks as
  Cognitive Models of the Processing of Syntactic Constraints", PMC 2024 (agreement from next-word prediction);
  Caucheteux & King 2023, *Nat Hum Behav* (next-word prediction = the cortical computation); Mitropolsky et al. 2022
  arXiv:2206.13217 (spiking center-embedding, CF-languages — the deep-recursion route); Garagnani & Pulvermüller 2018,
  *Front Comput Neurosci* 12:88 (brain-constrained spiking word ensembles); Hawkins & Ahmad 2016 (multi-segment HTM).
- **Catalog (Kandel 6e):** G.10 hierarchical symbolic system (Ch 55 pp 1370-1372); G.11 dual-stream (pp 1380-1387);
  G.12 Broca — production/grammar, bounded-recursion behavioral validation (pp 1382-1384); G.13 Wernicke — lexical
  selection/semantics (pp 1384-1385); G.07/H.19 pre-SMA/SMA + premotor competitive queuing (Ch 30-39); N.15 theta-gamma
  multiplex (Lisman-Idiart 1995); D.18 committed three-term permanence rule.
