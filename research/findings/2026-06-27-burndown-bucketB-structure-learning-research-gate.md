# Burndown Bucket B — the structure-learning frontier — deep-research gate (2026-06-27)

**Type:** STANDING-PRACTICE deep-research gate (READ-ONLY). No code written, no `sim/`/composer edit. Scopes
**Bucket B** — the host-DESIGNED linguistic STRUCTURE the (now-spiking) conversational operations consume — into
what is INCREMENTALLY learnable (the B1 corpus-mine → learn template) vs the genuine months-frontier.

**Directive (the bar):** the END must be FULLY SPIKING on the ONE-BRAIN shared substrate (non-negotiable, owner,
memory `feedback_end_state_fully_spiking_one_brain_path_by_efficiency`); a spiking op whose STRUCTURE
(weights/connectivity/lexicon) is host-DESIGNED is a residual shortcut — the structure must SELF-ORGANIZE
(`feedback_spiking_structure_must_self_organize`). The PATH per-capability is an efficiency call. Bucket A
(OPERATIONS — bind/serial-order/cleanup/moat) is already converging to spikes (C1–C4); Bucket B is the STRUCTURE.

**Terms (defined once):**
- *Host-designed structure* — a Python data structure (a dict / list) hand-authored by the developer that the
  spiking operation reads. E.g. `FRAME_LEXICON["go"] = [...GOAL...]` tells the renderer that `go` takes a GOAL
  argument; the brain did not learn this — a human typed it.
- *Mined / acquired structure* — the same kind of structure DERIVED from corpus statistics over the brain's OWN
  learned vocabulary, by an extractor + a learner (the **B1 template**, below). Structure ACQUIRED, not given.
- *The B1 template* — the pattern proven in `2026-06-27-regimeB-corpus-mined-axis-GO.md`: a corpus EXTRACTOR
  (counts attested co-occurrences over the brain's vocab, logs provenance) feeds a reused LEARNER (the Tier-2.3
  Betasort ordinal objective) → a self-organized relation axis, with a **permuted-mining** anti-cheat proving the
  corpus signal (not the apparatus) carries the structure. It turned a hand-coded ordinal axis into a
  corpus-acquired one by **wiring two halves the project already owned**, reuse-by-import, NO `sim/` edit.

---

## 1. ISOLATE — the host-designed-structure inventory (VERIFIED in code this session)

Every row was read in the current code. The decisive split is **genuinely hand-authored (a human typed the
structure)** vs **already-learned (the brain acquired it)**.

### 1a. ALREADY-LEARNED — NOT Bucket-B shortcuts (verified; do not re-open)

| Structure | Where | Status — why it is NOT a shortcut |
|---|---|---|
| **The concept CODES** (the brain's word meanings) | `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`; `bridges/firstchat/brain*.npz` | **LEARNED from conversation.** Rate-Hebbian co-occurrence on the real `SimulationBridge` builds the co-occurrence matrix `corr(M,C)=+0.705`; STDP is measured-negative (wrong rule); the population code reaches ~94% of host. The codes self-organize from hearing the corpus word-by-word. **Already the brain's own learned structure — not host-designed.** |
| **The B1 ordinal RELATION axis** (size) | `_regimeb_corpus_mined_axis_derisk.py`; `2026-06-27-regimeB-corpus-mined-axis-GO.md` | **MINED from the corpus** (scalar-adjective co-occurrence over the brain's vocab) → fed to the reused Betasort learner. Permuted-mining collapses (0.476/0.552 ≈ chance) ⇒ the corpus premises, not the apparatus, carry it. **Structure acquired, not given.** (Residual: the Betasort *objective* runs host-side — see §1c; the COMPARATOR is already a spiking Wang-2002 accumulator.) |
| **The Tier-2.1-A factored analogy OPERATOR** | `factored_relation_analogy.py` | The analogy *mechanism* (unbind→transform→apply→cleanup) runs on the RF spiking substrate (`use_spiking_bind=True`, == numpy, 6 seeds). The OPERATOR is fine; what is hand-curated is the RELATION KB it operates on — see §1b. |

### 1b. GENUINELY HOST-DESIGNED — the Bucket-B residual (the exact hand-authored bytes)

| Structure | Where (file:symbol) | What exactly is hand-authored |
|---|---|---|
| **The verb-frame LEXICON** | `argstructure_composer.py:65` `FRAME_LEXICON` | A dict of **8 hand-typed verb frames**: `go/come/walk/run → [agent, action, GOAL("to the")]`; `give/send → [agent, action, THEME, RECIPIENT("to the")]`; `put → [agent, action, THEME, LOCATION("on the")]`; `_default → [agent, action, patient]`. A human decided which thematic roles each verb licenses and which preposition + determiner scaffold each role carries. `FRAME_ROLES` (`:79`) and the FrameCQ frame-id map (`:102`) are derived from it. |
| **The (verb, prep) → role MAP** | `argstructure_composer.py:84` `VERB_PREP_ROLE` | 8 hand-typed `(verb, preposition) → typed-role` entries (`(go,to)→GOAL`, `(put,on)→LOCATION`, …). The extractor uses this to assign a corpus PP to a role — i.e. the corpus mining CONSULTS this hand table rather than deriving it. |
| **The wh→role MAP** + multiword cues | `wh_question_parser.py:44` `WH_ROLE_CANDIDATES`, `:54` `WH_MULTIWORD` | Hand-typed `wh-word → ordered candidate roles` (`where→[GOAL,LOCATION]`, `who→[agent,RECIPIENT]`, …) + multiword cues (`(where,from)→SOURCE`). The filler-gap lexicon. |
| **The closed-class word lists** | `argstructure_composer.py:99` `FUNCTION_WORDS`; `wh_question_parser.py:61` `_TRAIL_PREPS`, `:80` `_BARE_LEAD`; the regex aux/subject forms (`:64`, `:72`) | Hand-typed determiners/prepositions (`{the,a,an,to,on,in,…}`), the trailing-prep set, the per-role bare-answer lead (`GOAL→"to the"`), and the wh-question surface-form regexes. The grammar's function-word inventory + the surface-form parser. |
| **The morphology tables** | `argstructure_composer.py:92` `TENSE_3SG`; `entity_instance_layer.py:66` `_PAST`; `tense_aspect_composer.py` `tense_words` | Hand-typed inflection dicts: 3sg-present (`go→goes`), irregular past (`go→went`, ~36 verbs), the `['PAST','PRESENT','FUTURE']` tag codebook. |
| **The regime-A relation KB** | `factored_relation_analogy.py:164` `build_knowledge_base` | Hand-curated gender / capital_of / past-tense / comparative families (`king/queen`, `paris/france`, …) with their factored attributes typed in by hand. The analogy OPERATOR is spiking; the KB it reasons over is GIVEN. |
| **The console SIZE ladder** | `first_chat_console.py:900` `_SIZE_LADDER = ("tiny","small","big","huge","giant")` | A hand-typed 5-item ordinal ladder. **NOTE the gap (verified `:910`):** the console STILL uses this GIVEN ladder even though B1 already proved the MINED axis works — B1 is validated but **not yet wired into the console**. |
| **The common-ground TAG VALUE** | `common_ground_composer.py` `store_cg(..., common_ground='SHARED'\|'PRIVATE')` | The tag is bound spiking-RF, but the SHARED/PRIVATE VALUE is host-SUPPLIED at store time (the caller passes it; the brain does not infer "does the listener already know this?"). |
| **The tense TAG VALUE** | `tense_aspect_composer.py` `store_tensed(fact, tense=...)` | Same shape: the tag bind is spiking-RF; the PAST/PRESENT/FUTURE VALUE is host-supplied (or surface-parsed), not inferred from event semantics. |

### 1c. The precise residual shape (so a fix targets the right bytes)

The hand-authored items split into **two mechanistically different kinds**, which matters for ranking:

1. **TYPED RELATIONAL STRUCTURE** (the verb-frame lexicon, the (verb,prep)→role map, the relation KB, the ordinal
   ladder). These say *which arguments/relations an item licenses*. They have an **observable corpus signature**
   (a verb's arguments, a comparative's scalar markers) → the B1 template applies directly.
2. **CLOSED-CLASS / MORPHOLOGY LEXICON** (function words, the wh-word inventory, inflection tables, tag codebooks).
   These are the **finite function-word vocabulary** of the language. They are small, closed, high-frequency, and
   (in real acquisition) learned *early* but as a special class — the genuinely hard part is not the list, it is
   the *grammar* that assembles them recursively (§3 frontier).

The recurrent pattern across 1b: **the OPERATION is spiking, the STRUCTURE is a host dict, and in several cases the
EXTRACTOR that would mine the structure already exists but READS the host dict instead of DERIVING it.** That
last clause is the cheap-win signal (§3).

---

## 2. REFRAME via biology — how the brain ACQUIRES this structure

The owner's `feedback_spiking_structure_must_self_organize` insight (structure emerges developmentally from local
rules + usage, NOT from task-learning) is exactly what the language-acquisition literature describes for argument
structure. The reframe answers "are we testing the right hypothesis?":

- **Usage-based acquisition (Tomasello 2003).** Children do NOT start with abstract verb frames. They acquire
  **item-specific lexical schemas first** ("X hits Y", "X gives Y to Z"), then **form analogies across the roles**
  to abstract a general construction ("X Verb-transitive Y"). The frame is *induced from usage*, not innate and not
  hand-given. ⇒ a frame lexicon that is MINED-then-abstracted from the corpus is the biologically faithful form;
  the hand-authored `FRAME_LEXICON` is a scaffold standing in for the developmental induction.
- **Syntactic bootstrapping (Gleitman; Naigles).** The frame a verb appears in is a **reliable corpus signal** the
  child recruits to constrain the verb's meaning — i.e. the `(verb → argument-frame)` mapping is *present in the
  input distribution* and learnable. This is precisely what the project's spaCy SVO extractor already observes.
- **Frame induction is an established unsupervised-learning problem.** Computational acquisition models induce verb
  subcategorization frames from raw corpora (Brent 1993; Korhonen; the EM/clustering frame-induction line) — and,
  decisively for the "cognitively plausible" bar, **Buttery & Korhonen (2005)** ran subcategorization-frame
  acquisition specifically over **child-directed speech**, showing the frames are recoverable from the kind of
  input a learner actually gets. ⇒ "verb frames cannot be learned from data" is FALSE; it is a solved induction
  problem, and the project already owns the extractor half.
- **The closed-class lexicon** is acquired as a distributional class (high-frequency, low-content items cluster
  distributionally — the same frequency/contingency statistics the brain's co-occurrence cortex already computes).
  The hard part biology defers is **recursive hierarchical grammar** (Merge / phrase structure / long-distance
  dependencies) — the part that is genuinely a slow-developmental + (in this project) BPTT-SNN-scale frontier.

**The reframe verdict:** the relational structure (verb frames, relation axes) is *exactly* the B1 case —
acquired from corpus usage over the brain's own vocab. It was never a task-learning problem (which fails) nor a
"needs the dendritic substrate" problem; it is a corpus-mining + reuse-the-learner problem, and the extractor
already exists. The recursive grammar is the separate deep frontier.

---

## 3. RANK cheapest-first — incrementally learnable vs the months-frontier

### ★ B-mine-1 (RECOMMENDED FIRST) — MINE the verb-frame lexicon from the corpus (the B1 template, again)

- **The capability:** replace the hand-authored `FRAME_LEXICON` / `VERB_PREP_ROLE` with a frame table DERIVED from
  the corpus over the brain's learned verbs — "structure acquired, not given," exactly as B1 did for the ordinal
  axis.
- **Why it is cheap — the two halves already exist:**
  - **The extractor is built.** `_corpus_svo_extract.py` (`--typed-roles`, verified `:113-148`) ALREADY parses the
    corpus with spaCy and observes, per verb, its **direct objects** (`dobj/dative/attr/oprd`) and its
    **prepositional arguments** (`prep`→`pobj`, KEEPING the preposition, `:83`), with per-fact provenance + an
    attestation count (`--min-count`). It currently *consults* `VERB_PREP_ROLE`/`FRAME_ROLES` to label the slot.
    The mine is therefore: **invert that consultation** — accumulate the observed `(verb → {preposition: count})`
    and `(verb → has-direct-object)` distribution, and DERIVE each verb's licensed slots from what it attests
    (the prepositions a verb takes above threshold = its oblique frame; presence of a direct object = a
    patient/THEME slot). This is the same "count attested co-occurrences over the brain's vocab" half as
    `mine_size_scores` (`_regimeb_corpus_mined_axis_derisk.py:100`).
  - **The role-assignment learner is small and standard.** Map the mined `(verb, prep)` distribution to a typed
    role by the **prep's dominant semantics** (a corpus-derived `prep → role` clustering: `to`→GOAL/RECIPIENT
    disambiguated by whether the verb also takes a THEME; `on/in`→LOCATION; `with`→INSTRUMENT; `from`→SOURCE). This
    is exactly the unsupervised frame-induction the literature solves; at the project's small verb count it is a
    thresholded count + a tiny prep→role table that is itself corpus-justifiable (the prep's distribution across
    verbs), not a per-verb hand list.
- **Reusable machinery:** `_corpus_svo_extract.py` (the extractor, ~unchanged), `mine_size_scores`'s
  count-with-provenance + min-count discipline, the B1 anti-cheat harness (permuted-mining, provenance assert,
  moat 0-FA). The composer is unchanged — it consumes a `FRAME_LEXICON`-shaped dict whether hand-typed or mined.
- **The decisive anti-cheat (the B1 analogue):** **permuted-mining** — assign each verb a RANDOM frame (shuffle the
  mined prep→role assignments across verbs) → the typed-role recall + render must COLLAPSE (a `give`-framed `go`
  mis-renders / mis-recalls). If it does not collapse, the frame is not load-bearing. Plus: the **agrammatism
  ablation** already in `argstructure_composer.reparse_to_fact` (drop the closed-class scaffold → telegraphic) must
  still hold; the **moat** (an unlicensed wh / unstored fact → None) must stay 0-FA.
- **Rough cost:** **LOW–MEDIUM.** One new de-risk runner (`_bucketB_corpus_mined_frames_derisk.py`) that (a) runs
  the existing extractor to get `(verb → prep-distribution + dobj)`, (b) derives the frame dict, (c) feeds it to
  `ArgStructureComposer` in place of the hand `FRAME_LEXICON`, (d) runs the typed-role store/`query_role`/render
  matrix + the permuted-mining/provenance/moat anti-cheats at ≥6 seeds. Reuse-by-import, NO `sim/` edit. The honest
  caveat (same as B1): the mine is **lossy + corpus-budget-dependent**, and the brain's vocab gates which verbs are
  mineable — so the believability is the *signature* (permuted-mining collapse + provenance), not 100% frame
  accuracy.
- **Payoff:** HIGH. It converts the single largest, most-load-bearing Bucket-B structure (the verb-frame lexicon —
  the thing the whole typed-argument-structure surface depends on) from hand-authored to corpus-acquired, on the
  brain's own vocab, with the same proven template. It is the direct B1-for-relations → B1-for-frames step.

### B-mine-2 — MINE the wh→role map from the mined frames (a near-free corollary)

- **The capability:** derive `WH_ROLE_CANDIDATES` from the mined frames instead of hand-typing it.
- **Why it is nearly free:** the wh→role map is *downstream of* the frame lexicon — `where` questions an oblique
  locative/goal slot, `what` questions a direct object/theme, `who` questions the subject. Once B-mine-1 has the
  per-verb licensed roles, the wh-candidate ordering is the **inverse index** of the role inventory (which roles
  are obliques vs core), computable from the mined frames + the prep→role table B-mine-1 already builds. The
  permuted-mapping anti-cheat already exists (`wh_question_parser` threads a `role_map` for exactly this).
- **Cost:** LOW (corollary of B-mine-1). **Payoff:** MEDIUM (removes the second hand-authored relational table).

### B-wire-1 — WIRE B1's mined ordinal axis into the console (validated, not yet deployed)

- **The capability:** flip the console's `_build_ordinal_map` from the GIVEN `_SIZE_LADDER` to the B1
  corpus-mined axis. **B1 is already GO** (6 seeds + 3 spiking); the console just hasn't adopted it (`:910`).
- **Cost:** VERY LOW (swap the ladder source for the validated `mine_size_scores`+learner, gated on the brain
  having the size adjectives in vocab — `brainALL_w7000` does, `brain1454` doesn't, so keep the hand ladder as the
  fallback for vocab-poor brains). **Payoff:** MEDIUM (the console's ordinal reasoning becomes acquired-not-given;
  closes a known validated-but-unwired gap). **Caveat:** the console wire-up is being edited by another agent —
  coordinate, don't collide.

### B-frontier-A — the COMMON-GROUND / TENSE tag VALUES inferred, not host-set

- The tag BIND is spiking; the VALUE is host-supplied. Inferring SHARED/PRIVATE needs a listener model
  (the documented agent-modelling wall, `common_ground_composer` header); inferring tense from event semantics
  needs a Reichenbach reference-time representation (the documented temporal-reasoning frontier,
  `tense_aspect_composer` header). **MEDIUM–HIGH cost, deferred** — these are representation frontiers, not
  mineable lexicons. Flagged, not ranked into the cheap sequence.

### B-frontier-B (the genuine MONTHS-frontier) — learned RECURSIVE grammar + developmental structure

- **What genuinely needs the months-arc:** the step from *induced item-frames* to a **learned recursive
  hierarchical grammar** (phrase structure / Merge / long-distance + embedded dependencies / productive novel
  composition) — i.e. a grammar that GENERATES, not a frame table that LABELS. This is the categorical
  free-generation gap (`project_generative_sequence_frontier`, MEASURED 0.0 novel-composition), whose only closer
  is the **backprop-pretrained spiking generative LM (BPTT-SNN)** as a development-stand-in, consolidated onto one
  spiking bridge (C1) with no-forgetting (C2) — both gates already MET at toy scale; the open work is SCALE.
- **The deepest residual (the owner's purest reading of `self_organize`):** even the spiking OPERATIONS that the
  mined structure feeds still have **host-COMPUTED binding connectivity** (`rf_set_complex_weights` injects the
  clean-invertible bind weights; they don't self-organize). Closing that is the **developmental
  self-organization of the binding structure**, likely tied to the dendritic substrate (D2; the multiply
  native) — the genuine open binding-problem frontier. **MONTHS, high variance, deliberate owner call.**
- **Cost:** MONTHS. **Payoff:** the actual end-state (a brain that develops its own grammar + binding structure).

---

## 4. VERDICT

**There IS an incremental structure-learning win available — Bucket B is NOT all-months-frontier.** The single
most-load-bearing host-designed structure (the verb-frame lexicon) is mineable by the EXACT B1 template, and the
extractor half already exists in the codebase reading the hand dict it should be deriving.

- **The cheapest decisive Bucket-B de-risk:** **B-mine-1** — mine the verb-frame lexicon (`FRAME_LEXICON` /
  `VERB_PREP_ROLE`) from the corpus via the already-built `_corpus_svo_extract.py --typed-roles` extractor
  (inverted to DERIVE the `verb → {prep-distribution, dobj}` frame instead of CONSULTING the hand table), fed to a
  small corpus-justified prep→role assignment, with the **permuted-mining** anti-cheat as the load-bearing proof
  (a randomly-reassigned frame must collapse typed-role recall/render). ≥6 seeds, reuse-by-import, NO `sim/` edit.
  This is the literal B1-for-relations → B1-for-frames generalization, and it converts the largest Bucket-B
  structure from hand-authored to corpus-acquired. **B-mine-2** (the wh→role map) then falls out nearly free as
  the inverse index of the mined frames; **B-wire-1** (deploy B1's already-GO mined ordinal axis into the console)
  is an even cheaper validated-but-unwired flip to schedule alongside.

- **The precise deep-frontier boundary (genuinely needs the months-arc):** (1) a **learned recursive hierarchical
  grammar** that GENERATES novel language (the BPTT-SNN generative-sequence frontier — C1/C2 gates met at toy
  scale; SCALE is the open work); (2) the **developmental self-organization of the binding connectivity** (the
  host-injected RF bind weights → emerge on-substrate, likely dendritic — the open binding problem); (3) the
  **tag-VALUE inference** frontiers (a listener model for common-ground; event-semantics for tense). These are
  representation/learning frontiers with no cheap corpus-mine, deliberately deferred.

- **Recommended next:** build **B-mine-1** (the corpus-mined verb-frame lexicon) as the cheapest decisive Bucket-B
  de-risk — it reuses the B1 template + the existing SVO extractor, carries the strongest anti-cheat
  (permuted-mining), and turns the keystone hand-authored structure into acquired structure on the brain's own
  vocab. Schedule **B-wire-1** (deploy the already-GO mined ordinal axis) alongside as a near-zero-cost validated
  flip (coordinating with the agent editing the console). Treat the recursive-grammar + developmental-binding
  frontier as the tracked months-arc, not this gate's build.

**State plainly:** incremental-win-available (B-mine-1 verb-frame mining, the B1 template generalized) — NOT
all-months-frontier. The recursive-grammar + developmental-binding-structure piece is the genuine months-frontier
and is correctly deferred.

---

## 5. Sources (read-only research)

- `2026-06-27-regimeB-corpus-mined-axis-GO.md` (the B1 template) · `2026-06-27-conversation-depth-brain-based-audit-and-burndown.md` (Bucket B) · `2026-06-27-burndown-bucketA-build-plan.md` (what A covered) · `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (the codes are LEARNED).
- Code (verified this session): `argstructure_composer.py` (`FRAME_LEXICON`, `VERB_PREP_ROLE`, `TENSE_3SG`, `FUNCTION_WORDS`, `reparse_to_fact`); `wh_question_parser.py` (`WH_ROLE_CANDIDATES`, `_TRAIL_PREPS`, `_BARE_LEAD`); `_corpus_svo_extract.py` (the existing `--typed-roles` extractor); `factored_relation_analogy.py` (`build_knowledge_base`); `first_chat_console.py:885-922` (`_build_analogy_kb`, `_SIZE_LADDER`, `_build_ordinal_map`); `common_ground_composer.py` + `tense_aspect_composer.py` + `entity_instance_layer.py` (tag-value + morphology tables).
- Memories: `feedback_end_state_fully_spiking_one_brain_path_by_efficiency`, `feedback_spiking_structure_must_self_organize`, `project_generative_sequence_frontier`, `feedback_dendritic_substrate_fair_game`.
- Biology / acquisition literature: Tomasello (2003) usage-based acquisition (item-frames → abstraction by analogy); Gleitman/Naigles syntactic bootstrapping (the frame is a reliable corpus signal); Brent 1993 + Korhonen verb-subcategorization-frame induction from corpora; **Buttery & Korhonen (2005)** subcategorization-frame acquisition over child-directed speech (cognitively-plausible, the frames ARE recoverable from a learner's input).
