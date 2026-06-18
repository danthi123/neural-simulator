# Richer-syntax conversational frontier — deep-research scoping (toward basic-LLM-competitive)

**Date:** 2026-06-18
**Type:** Deep-research / catalog-review scoping (read-only; NO code built or edited). The project's STANDING
OPENING MOVE for a new direction (owner directive 2026-06-07).
**Owner PRIMARY task #55:** "Conversational architecture build-out to basic-LLM-competitive (biology-faithful,
pre-scaling)."
**Question:** given the bind + the flat-SVO pipeline are SETTLED, which richer-SYNTAX capability is the
highest-leverage cheapest-first next build toward LLM-competitive, on the EXISTING substrate, and how do we
de-risk it cheap-first with anti-cheats?

---

## TL;DR (the recommendation)

1. **Diagnosis.** The conversational stack is comprehensively complete *for flat SVO*. The **flat-SVO PARSER
   is the bottleneck** — and it is a SHARPER bottleneck than the task framing implies, because of a structural
   asymmetry the audit surfaced: the **back end is already richer than the front end.** The composer
   (`RFPhasorComposer` / `CoreSimComposer`) already has `attribute`/`attribute2` roles and already
   binds/decodes `(adjs, noun)` tuples and recursive `Clause` objects on the spiking RF substrate
   (`nested_composition_agent` validates nested decode multi-seed); the GENERATION side already learns
   **frame-conditioned word order on real spikes** (multi-frame serial-order de-risk GO 6/6). What is missing is
   *purely* the comprehension front end: the parser maps `(word-position × voice) → {agent, action, patient}`
   and produces **flat SVO only** — it cannot emit attribute bindings, cannot parse nested structure from a
   flat token stream, and assumes one fixed frame. The richer-syntax work is therefore **almost entirely a
   parser-front-end problem**, and the composer is waiting for the structure.

2. **Ranked recommendation (leverage × cheapness on the existing substrate):**
   **#1 (DO FIRST) = ATTRIBUTED ENTITIES (adjective+noun parsing).** Highest leverage-per-effort: it is the
   single most common real-language construction the agent can't yet take from natural input, the composer
   *already* stores it, and the parser change is a small, well-bounded extension of the validated conjunctive
   (position × voice → role) readout — add an **adjacency/agreement cue** (`adj` precedes its head noun) as a
   second conjunctive factor, exactly the mechanism the voice cue already is. **#2 = MULTI-FRAME comprehension
   parsing** (questions / dative / topic-first), mirroring the already-GO generation-side frame-conditioning
   onto comprehension. **#3 = EMBEDDED-CLAUSE parsing from nested natural input** — highest ceiling but
   genuinely harder (needs a neural push/pop over the gamma-slot WM), so it is the deepest of the three and is
   correctly last.

3. **The single cheap-first de-risk (prefer numpy/CPU falsification BEFORE any GPU/sim build):** a
   **conjunctive attributed-entity PARSER probe** — extend the existing closed-form/Hebbian conjunctive readout
   with an adjacency feature so that for `"big red apple"` the parser assigns `big→attribute`, `red→attribute2`,
   `apple→patient` (and `dog`/`go`/SVO words to their SVO roles), feed the parse to the *existing* composer
   `store((["big","red"],"apple"), ...)`, and round-trip `what_does → "big red apple"`. **GO bar:** held-out
   sentences (novel adj+noun combinations NEVER in the parser's training set) parse to the correct
   attribute/attribute2/patient roles AND round-trip recall == the host-routed oracle, **≥5/6 seeds**, with the
   no-confab moat intact. **Anti-cheats:** held-out generalization with a leakage assertion vs a memorization
   floor; a no-learning / permuted-feature / lesion control that collapses it; the role decision must be
   **neural** (the parser's firing selects the role, not a host adjacency rule); a **flat-only control** (the
   current SVO-only conjunctive readout, no adjacency factor) must FAIL on the same harness (proving the
   adjacency cue is load-bearing). An honest NEGATIVE — the conjunctive readout cannot separate attribute from
   patient without confounding the flat SVO roles — **is the deliverable** and maps the substrate's syntactic
   reach.

4. **Reuse (no new mechanism invented):** `conjunctive_parser.ConjunctiveParser` (closed-form PxV readout) +
   `brain_conversational_agent.BridgeParser` (the on-bridge spiking Hebbian parser) for the neural version;
   `rf_phasor_composer.RFPhasorComposer.store` (already handles `(adjs, noun)`) + `query_patient` (already
   renders `"big apple"`); the multi-frame `FrameCQ` pattern from `_phaseB_serial_order_multiframe_derisk` for
   #2; `ordered_position_wm.OrderedPositionWM` (gamma-slot position-binding spiking WM, catalog N.15) as the
   nesting buffer for #3; `song_g1_core` (`score_order` / `permuted_order_controls` / `g1_verdict`) for the
   GO-bar/anti-cheat scaffold.

---

## 1. Diagnosis — what the flat-SVO parser can and cannot do

### 1.1 The parser, precisely

Two parser implementations are in production, both flat-SVO:

- **`research/runners/brain_conversational_agent.py::BridgeParser`** (the on-bridge spiking one; also the parser
  *inside* `one_brain_composer.py::OneBrainComposer`). 6 conjunction units `(position × voice)` → 3 role
  ensembles (`agent`/`action`/`patient`), Hebbian-learned (the v16 embodied co-firing rule), voice-invariant.
  `parse(words, voice)` asserts `len(words)==3` and returns `{role: word}` by reading which role ensemble fires
  for each `(position, voice)` conjunction (`role_of`). **The parse is NEURAL** — the role firing selects the
  bind (de-risk `_phaseB_onebrain_parser_frontend_derisk`, **verdict GO, self/host 1.0, voice-inv 1.0, moat 1.0,
  permuted+lesion collapse 1.0**, `2026-06-18-one-brain-parser-frontend-GO.md`).
- **`research/runners/conjunctive_parser.py::ConjunctiveParser`** (the numpy reference). Same idea via a
  closed-form least-squares readout over the 6 `(position, voice)` combos, plus voice detection + light
  morphology so real passive sentences normalize. The science it packages: **voice-invariant role assignment
  REQUIRES conjunctive position×voice coding** (position-only P and additive P+V both score 0.000; the
  conjunctive PxV interaction scores 1.000) — because the voice flip (agent↔patient) is an *interaction*.

**CAN do:** assign agent/action/patient to a 3-content-word sentence, voice-invariantly (active "dog chase cat"
and passive "cat is chased by dog" → same agent), with light morphology, and route the parse into the composer
as spikes (no host `{role:word}` dict).

**CANNOT do (the three documented richer-syntax gaps):**

1. **Attributed entities (adjective+noun).** The composer *already* has `attribute`/`attribute2` roles and
   `store()` *already* accepts `(adjs, noun)` tuples (`rf_phasor_composer.py:317-333`, `core_sim_composition.py:
   397-413`), and `query_patient` *already* renders `"big apple"` / `"big hot apple"`. But the parser feeds
   flat SVO — it has no mechanism to recognise that `"big"` in `"big apple go north"` is an adjective modifying
   `"apple"`, so the attribute binding is never produced from natural input. **This is a pure parser gap with
   a ready back end.**
2. **Embedded-clause parsing.** The composer handles a clause *structurally* (a `Clause(agent,action,patient)`
   provided as a filler → recursive 2-level unbind, validated: `nested_composition_agent` decodes "dog see (cat
   chase bird)" 100% / multi-seed; `one_brain_composer._decode_clause`). But `BrainConversationalAgent.
   hear_clause_fact` says it explicitly: *"the clause is provided structurally here"* — the parser does NOT
   parse "the dog that chased the cat ran" from a flat token stream into the nested structure.
3. **Multi-frame word order.** The parser assumes the fixed SVO frame. Different sentence types (questions,
   datives, topicalisation) have different word orders. The **generation** side already solved frame-conditioning
   (`_phaseB_serial_order_multiframe_derisk` learns DISTINCT orders per frame, GO 6/6; the **spiking** version
   `_phaseB_serial_order_multiframe_spiking.json` also GO 6/6, true 0.99 vs perm 0.34 vs cross-frame 0.005) — but
   the **comprehension** side has no frame variable, so a non-SVO frame is mis-parsed.

### 1.2 The load-bearing asymmetry (why this is cheaper than it looks)

| Capability | Back end (composer) | Generation | **Front end (parser)** |
|---|---|---|---|
| Flat SVO | ✅ store/query/abstain | ✅ SVO order | ✅ GO (neural, voice-inv) |
| Attributed entity (adj+noun) | ✅ `attribute`/`attribute2`, decodes | ✅ renders "big apple" | ❌ **gap** |
| Embedded clause | ✅ `Clause` recursive decode (multi-seed) | ✅ renders nested | ❌ **gap (provided structurally)** |
| Multi-frame order | (frame-agnostic store) | ✅ frame-conditioned (spiking GO 6/6) | ❌ **gap (one frame)** |

The richer-syntax frontier is **almost entirely a comprehension-front-end build**. The composer and the
generator are already ahead; closing the parser gap unlocks capabilities the rest of the stack can already
express.

### 1.3 The biological mechanism + catalog grounding

- **Hagoort MUC (Memory-Unification-Control).** The dlPFC/IFG Control function unifies lexical items
  ("Memory") into hierarchical structure in a "unification space" — working memory that **stores and retrieves
  intermediate results** during nesting. The literature is explicit that biological unification needs **distinct
  input lines for DATA and CONTROL** (microprocessor-like), realizable via electrotonically segregated dendritic
  branches — i.e. the *control* signal (which role / which frame / open-a-clause) is a separate channel from the
  *content* (the word code). The project already has the analogue: the parser's role FIRING is the control
  decision; the concept code is the data; the bind's role phasor is a fixed projection. (Murphy 2024 J.
  Neurolinguistics; Hagoort MUC chapter; Martin & Doumas neuro-cognitive comprehension model.)
- **Building hierarchical nested structure by rapid neural sequences** (Calderon/Dehaene-lineage, *PNAS* 2025):
  nesting = **rapid neural sequences** writing/reading a memory (unification) space at the right point in "logical
  time"; recursion = nested function calls storing intermediate results. This is *exactly* a gamma-slot
  position-binding WM with a push/pop discipline — which the project ALREADY has on the spiking substrate
  (`OrderedPositionWM`, catalog N.15 Lisman-Idiart).
- **Catalog entries (cite by cluster+number):**
  - **G.10** Language as hierarchical symbolic system — phonemes/morphemes/words/**syntax** (the finite-rules →
    infinite-sentences combinatorics; Kandel Ch 55 pp 1370–1372). The capability this arc advances.
  - **G.12** Broca's area — **grammatical processing of non-canonical sentences.** Its behavioral validation IS
    the embedded-clause signature: *"The girl that the boy is chasing is tall" comprehension FAILS
    (grammar-dependent); "The apple the girl ate was green" succeeds (semantically constrained)* (Kandel Ch 55
    pp 1382–1384). This is the canonical biological marker for gap #2 — and a ready-made anti-cheat: a parser
    that genuinely parses grammar (not just semantics) must handle the non-canonical frame.
  - **G.11 / G.13** Dual-stream (dorsal arcuate→Broca sensorimotor + ventral semantic) + Wernicke
    auditory→semantic mapping — the comprehension stream the parser stands in for.
  - **G.07** Pre-SMA/SMA **internally generated sequences** + **G.08** PFC working memory holding "what"/"where"/
    "what+where" **conjunctions** during a delay (Rainer/Asaad/Miller) — the substrate for the
    parallel→serial + the unification-space WM.
  - **N.15** Theta-gamma cross-frequency coupling (Lisman-Idiart 1995): **≈7±2 gamma slots per theta cycle**, a
    time-multiplexed WM buffer; "shifting (rather than repeating) the assembly sequence encodes a SEQUENCE" — the
    exact substrate `OrderedPositionWM` realizes, and the buffer a clause-nesting push/pop runs over.
- **Eliasmith Spaun / SPA precedent:** syntax is captured by **binding semantic pointers (HRR/VSA) to one
  another**, with **basal-ganglia Control** doing action selection over the structure (question-answering task).
  This is the *same* representational commitment the project already made (FHRR resonate-and-fire bind), with BG
  Control = the parser's role/frame decision. The project is squarely on a validated path.

---

## 2. Ranked options (leverage × cheapness on the EXISTING substrate)

### #1 — ATTRIBUTED ENTITIES (adjective+noun parsing) — DO FIRST

- **Demonstrable claim:** the parser takes `"big red apple go north"` (natural flat input) and produces
  `attribute=big, attribute2=red, patient=apple` (+ SVO roles for the rest), the composer stores it (it already
  can), and `what_does(agent, action) → "big red apple"` round-trips == oracle, on held-out adj+noun
  combinations, multi-seed, moat intact.
- **Reuse:** `ConjunctiveParser` (extend the closed-form readout with an adjacency factor) for the cheap numpy
  falsification; `BridgeParser` (add an `attribute` role ensemble + an adjacency conjunction unit) for the
  neural version; `RFPhasorComposer.store((adjs, noun), ...)` + `query_patient` **unchanged** (the back end is
  ready); `song_g1_core` for the verdict scaffold.
- **Effort:** LOW. The mechanism is a *second conjunctive factor* (adjacency/agreement: "is this word's
  successor its head noun?"), exactly analogous to the voice factor the parser already conjoins. No new substrate;
  the composer needs zero change.
- **Dud-risk:** LOW-MEDIUM. The risk is **role confound** — adding an `attribute` role might bleed into the
  agent/patient assignment for flat SVO (regressing the validated flat parse). The de-risk's flat-only control
  + the requirement that flat SVO accuracy is UN-regressed guards exactly this.

### #2 — MULTI-FRAME COMPREHENSION (questions / dative / topic-first)

- **Demonstrable claim:** the parser comprehends ≥2 distinct frames (e.g. SVO statement vs a question frame vs a
  topic-first frame) into the SAME role structure — mirroring the generation side, where frame-conditioned order
  is already GO on spikes. "what does the dog chase?" and "the dog chases the cat" assign `dog→agent` despite
  different word orders.
- **Reuse:** the `FrameCQ` pattern (`_phaseB_serial_order_multiframe_derisk` / `_..._spiking`) is the comprehension
  mirror image — a **frame-conditioned** conjunctive readout `role = f(position, voice, FRAME)`, learned per
  frame, generalizing to held-out facts, with a cross-frame control (the SAME words parse DIFFERENTLY by frame).
  The generation de-risk's gate and anti-cheats transfer verbatim.
- **Effort:** LOW-MEDIUM. A third conjunctive factor (frame), and a (host or neural) frame-cue detector
  (question word / inversion). The generation side proves the substrate learns frame-conditioned mappings 6/6;
  comprehension is the dual.
- **Dud-risk:** MEDIUM. The frame *detector* is the soft spot — detecting the frame neurally (not a host rule)
  is the brain-based bar; a host frame-cue would be a documented shortcut (acceptable as a teaching scaffold,
  per the BRAIN-BASED-ONLY standard, but flag it). Generalization across frames may need more than two frames to
  be convincing.

### #3 — EMBEDDED-CLAUSE PARSING from nested natural input — HIGHEST CEILING, LAST

- **Demonstrable claim:** the parser takes "the dog that chased the cat ran" (a flat token stream) and produces
  the nested `Clause`-structured fact the composer already decodes — i.e. it PARSES the hierarchy, closing the
  `hear_clause_fact` "provided structurally" gap. Behavioral target = the **G.12** signature (non-canonical
  centre-embedded relative clause).
- **Reuse:** `OrderedPositionWM` (gamma-slot position-binding spiking WM, catalog N.15) as the **unification
  space** — a clause boundary triggers a PUSH (bind the partial clause to a WM slot), the relativizer/verb
  triggers a POP (read the slot back as a unit and bind it as the head's filler), exactly the
  rapid-neural-sequence nesting mechanism (PNAS 2025) + the MUC unification space. The composer's
  `Clause`/`_decode_clause` is the ready consumer. The dlPFC NMDA WM latch (`unified_brain_bridge` /
  `content_selection_spiking`) holds the open-clause state; transmission gates route the push/pop.
- **Effort:** HIGH. This is real syntax — a neural control discipline (open/close a constituent) over the WM,
  not a one-factor readout extension. It is the deepest of the three and the one most likely to surface an honest
  substrate boundary (depth-2 nesting already costs a seed in the *decode* per `nested_composition_agent`; the
  *parse* compounds a per-level control decision on top).
- **Dud-risk:** HIGH (but the honest negative is itself a deliverable — it maps where the point-neuron + gamma-slot
  WM substrate's syntactic depth ends, the G.12 boundary).

**Why this order:** #1 is the most-common construction, ready back end, smallest change, lowest risk → maximal
leverage/effort. #2 is the natural dual of an already-GO generation result → cheap, medium ceiling. #3 is the
highest ceiling but a genuine new control mechanism → correctly last, and gated behind the cheaper wins so the
arc banks capability early.

---

## 3. Recommended cheap-first de-risk (the single highest-leverage probe)

**`_phaseB_attributed_entity_parser_derisk.py`** (to be built; numpy/CPU FIRST, GPU/neural only after the CPU
falsification passes).

**What to test (CPU, numpy `ConjunctiveParser`-style, no sim build):**
1. Extend the closed-form conjunctive readout with an **adjacency factor**: features per content word become
   `(position, voice, is-followed-by-head-noun)` — concretely, a flag that the word's successor is the phrase
   head. Fit the readout to a training set of attributed sentences so it learns `adj→attribute`,
   `2nd-adj→attribute2`, `noun→patient` (and the SVO words → their SVO roles).
2. Parse a **held-out** set of sentences whose adj+noun *combinations were never in training* (leakage-asserted:
   assert the held-out adj-noun pairs are disjoint from train), feed each parse to the **existing**
   `RFPhasorComposer.store((adjs, noun), action, ...)`, and round-trip `query_patient → "big red apple"`.
3. Compare against the **host-routed oracle** (composer TOLD the correct roles) for parity, and against a
   **flat-only control** (the current SVO-only conjunctive readout) on the same sentences.

**Existing knobs:** `ConjunctiveParser.fit_readout` (the closed-form readout to extend); `RFPhasorComposer(seed,
D, vocab)` + `.store((adjs,noun),...)` + `.query_patient` (unchanged); `BridgeParser(seed, R, shared_bridge=,
index_offset=)` for the later neural version (add an `attribute` role ensemble + an adjacency conjunction unit);
`song_g1_core.g1_verdict` (FIXED bars 0.10/0.5) for the GO line.

**Quantitative GO bar (pre-register, FROZEN — never tuned):** per seed, **held-out** attributed sentences parse
to the correct `(attribute, attribute2, patient)` roles AND round-trip recall == oracle, at **≥0.90** accuracy,
clearing the floor 0.5 and beating the permuted/flat controls by **≥10%**, on **≥5/6 seeds**; flat SVO accuracy
**UN-regressed** vs the current parser; the no-confab moat unchanged (an unstored cue → None). GO ⇒ build the
neural `BridgeParser` version + wire `BrainConversationalAgent.hear` to parse attributes. BOUNDARY ⇒ localize
(the adjacency factor vs the role confound). 

**What an honest NEGATIVE means (and that it IS the deliverable):** if the conjunctive readout cannot separate
`attribute` from `patient`/`agent` without confounding the flat SVO roles (flat accuracy regresses, or held-out
attribute parsing sits at chance), that is a real finding — it maps the conjunctive-coding parser's syntactic
reach (adjectival modification needs a richer control signal than a single adjacency factor, pointing at the
dlPFC-Control / unification-space mechanism #3 sooner). Per project standard, the honest negative under strict
biology IS the scientific deliverable.

---

## 4. Anti-cheat controls (mandatory, all of them)

1. **Held-out generalization + leakage assertion vs a memorization floor.** The eval adj+noun COMBINATIONS must
   be disjoint from the parser's training set; **assert** the disjointness in code (a held-out pair that leaked
   into training silently inflates the score). Compare against a memorization floor (a parser that can only
   reproduce trained combinations). Generalization to NOVEL adj+noun pairs (Fodor-Pylyshyn systematicity) is the
   real claim, exactly as the learned-binder systematicity harness does it.
2. **No-learning / permuted / lesion control.** A no-learning readout (untrained adjacency weights) must FAIL; a
   permuted role-map (adjacency factor wired to the wrong role) must produce the WRONG bindings; a lesion (zero
   the `attribute` role ensemble / the adjacency factor) must collapse attribute recovery. Each must collapse for
   the RIGHT mechanistic reason.
3. **The parse must be NEURAL.** For the production claim, the parser's FIRING must select the role (the
   `BridgeParser` on-bridge readout), not a host adjacency rule. The CPU numpy probe is a falsification gate only;
   the production version is the spiking parser (the same standard the parser-front-end de-risk met:
   `2026-06-18-one-brain-parser-frontend-GO.md`). A host adjacency cue in the production path is a documented
   shortcut (a teaching scaffold at most), per the BRAIN-BASED-ONLY standard.
4. **The no-confab moat is NEVER weakened.** An unstored cue still returns `None`; the abstention rate and
   false-accept count are reported alongside, and must not regress. (Owner reframe `feedback_moat_not_hard_lossy_
   memory_ok`: the moat is a plus, not a hard gate — but a richer-PARSER change must not be the thing that breaks
   it; trade the moat only deliberately for the learned-lossy path, not as a side effect of syntax.)
5. **Flat-only control MUST FAIL on the same harness (the hierarchy claim's guard).** Because the claim is
   "attributed structure," the current attribute-free SVO conjunctive readout must FAIL to recover the attribute
   on the identical held-out sentences. If the flat-only readout *passes*, the attribute signal is an artifact
   (the words were separable without the adjacency cue), not real syntactic structure — the same discipline the
   multi-frame de-risk uses (the cross-frame control must show the SAME fact ordered DIFFERENTLY by frame).
6. **Flat-SVO non-regression.** The validated flat parse (`_phaseB_onebrain_parser_frontend` GO, self/host 1.0,
   voice-inv 1.0) must remain GO after the attribute extension — assert it on the same probe.

---

## 5. Machinery to reuse (exact files/functions — for trust-but-verify)

| Purpose | File · symbol |
|---|---|
| Numpy conjunctive readout to EXTEND (cheap falsification) | `research/runners/conjunctive_parser.py` · `ConjunctiveParser`, `fit_readout`, `_features` |
| The on-bridge spiking parser (neural production version) | `research/runners/brain_conversational_agent.py` · `BridgeParser` (`role_of`, `parse`, `_train`); also embedded in `research/runners/one_brain_composer.py` · `OneBrainComposer` (`hear`) |
| Parser-front-end de-risk to mirror (GO reference + anti-cheat pattern) | `research/runners/_phaseB_onebrain_parser_frontend_derisk.py` (verdict GO; lesion+permute controls) |
| Composer — ALREADY stores `(adjs, noun)` + decodes/renders | `research/runners/rf_phasor_composer.py` · `RFPhasorComposer.store` (lines ~317-333), `query_patient` (renders "big apple"), `roles = (...,"attribute","attribute2")`; mirror in `core_sim_composition.py::CoreSimComposer` |
| Composer — nested `Clause` decode (for #3) | `research/runners/rf_phasor_composer.py` · `Clause`, `_is_clause`, `_render`/`_filler_phases`; `one_brain_composer.py` · `_decode_clause`; reference agent `research/runners/nested_composition_agent.py` |
| Multi-frame order (for #2 — the comprehension dual) | `research/runners/_phaseB_serial_order_multiframe_derisk.py` · `FrameCQ`; spiking version `_phaseB_serial_order_multiframe_spiking_derisk.py`; `research/runners/neural_serial_order_renderer.py` |
| Gamma-slot position-binding WM (for #3 — the unification space) | `research/runners/ordered_position_wm.py` · `OrderedPositionWM` (`encode_sequence`, `read_slot`, `calibrate_threshold`); catalog N.15 |
| dlPFC NMDA WM latch + transmission gating (for #3 control) | `research/runners/content_selection_spiking.py` · `SpikingSpreadingController`; `research/runners/unified_brain_bridge.py`; `sim/regions.py` transmission_gate / `bridge.set_transmission_gate` |
| Verdict + anti-cheat scaffold | `research/runners/song_g1_core.py` · `score_order`, `permuted_order_controls`, `g1_verdict` |
| Agent wiring (where attribute parsing lands) | `research/runners/brain_conversational_agent.py` · `BrainConversationalAgent.hear` / `hear_clause_fact` / `what_does` |

**Anti-cheat note for the controller (trust-but-verify):** the load-bearing claim — *the composer already
supports attributed entities and nested clauses, so the richer-syntax work is a parser-front-end-only build* —
is verifiable at `rf_phasor_composer.py:317-333` (`store` handles `(adjs, noun)`), `:446-468` (`query_patient`
renders adjectives), and `nested_composition_agent.py` header (nested decode validated multi-seed). The
generation-side multi-frame result is verifiable at `research/findings/raw/_phaseB_serial_order_multiframe_
spiking.json` (true 0.99 / perm 0.34 / cross 0.005, 6/6). The flat-parse GO is at
`research/findings/raw/_phaseB_onebrain_parser_frontend.json` (verdict GO, all metrics 1.0).

---

## 6. Sources

- Murphy (2024), *J. Neurolinguistics* 70:101180 — neural parser / hierarchical nesting as nested function calls
  over a unification space; data/control on segregated dendritic branches.
  https://elliot-murphy.com/wp-content/uploads/2023/11/murphy_2024_jneuroling.pdf
- Hagoort, MUC (Memory, Unification, Control) chapter — the tripartite IFG hierarchy; unification space = working
  memory. https://pure.mpg.de/rest/items/item_2193289/component/file_2193288/content
- *Building hierarchically nested structure by rapid neural sequences*, **PNAS** (2025) —
  https://www.pnas.org/doi/10.1073/pnas.2507417122
- Martin/Doumas, *A neuro-cognitive model of comprehension based on prediction and unification*, PMC11035797 —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11035797/
- *Vector Symbolic Architectures for Context-Free Grammars* (arXiv 2003.05171 / Cognitive Computation 2021) —
  phrase-structure & parse-tree encoding via role-filler binding + bundling, position vectors for order.
  https://arxiv.org/pdf/2003.05171
- *Survey on Hyperdimensional Computing / VSA, Part II* (ACM Comput. Surv.) — role-filler binding for semantic
  frames "who did what to whom". https://dl.acm.org/doi/10.1145/3558000
- Eliasmith et al., **Spaun / Semantic Pointer Architecture** — syntax via binding semantic pointers + BG Control
  (question answering). https://en.wikipedia.org/wiki/Spaun_(Semantic_Pointer_Architecture_Unified_Network) ;
  https://compneuro.uwaterloo.ca/research/spa/semantic-pointer-architecture.html
- Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` — **G.07, G.08, G.10, G.11, G.12,
  G.13, N.15** (cited inline above); Kandel 6e Ch 55 (language), Ch 52/34 (WM/SMA), Buzsáki Cycle 12 (theta-gamma).
