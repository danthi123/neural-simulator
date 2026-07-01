# Multi-fact SYNTHESIS frontier — scoping ("it lists facts; make it synthesize like an LLM")

**Date:** 2026-07-01
**Type:** Deep-research + reference-catalog review (the standing research-gate opening move for a new direction / a
soft "honest ceiling" verdict — the Phase-10 doc closed with *"the discussion still LISTS facts; genuine single-pass
fluent synthesis over multiple facts confabulates on this 21M"*, which is exactly the DISGUISED-boundary the gate
fires on). READ-ONLY; no `sim/`/GPU edit. Companion to `2026-07-01-open-ended-grounded-discussion-scoping.md` +
`-fluid-conversation-{gap-assessment,mechanisms-roadmap}.md`.

Terms once, on first use. **NLG** = natural-language generation. **SVO** = subject-verb-object triple (the brain's
fact form). **Aggregation** = the NLG operation that fuses several propositions into fewer, denser sentences.
**RE(G)** = referring-expression (generation) — choosing "it/the dog/a dog" for a referent. **Discourse
connective** = because/but/so/and-then. **Faithful/grounded** = every asserted fact is in the store (the no-confab
moat). **The 21M** = the RA-fine-tuned TinyStories generator (fluency-only, brain-gated). **VERIFY** = re-parse the
output, drop any SVO not in the store.

---

## 0. TL;DR (read this first)

- **Most of "synthesis" is ALREADY DONE — and it is done the way real NLG and real biology do it.** The console's
  `_discuss` output *"An elephant is a mammal. It is grey. It has trunk and tusk."* is textbook NLG **microplanning**:
  it **aggregates** three `isa/is/has` propositions into grouped sentences, does **referring-expression generation**
  (the pronoun "It"), and handles article agreement (a/an, isa-noun vs is-adjective). Aggregation and RE are
  *the* synthesis operations in Levelt's and Reiter-Dale's models — the console is not "just listing," it is doing
  real sentence planning. What it is NOT doing is (e) **cross-fact INFERENCE** (deriving a NEW proposition) and (c)
  **discourse connectives** (because/but/so). Those are the genuine residual.
- **The genuine residual is TINY and split by risk.** Of the six sub-operations of "synthesis," four are ~done
  (aggregation, RE, ordering, article/agreement); two are cheap surface polish (connectives, light abstractive
  rephrasing) that the moat makes SAFE; and only **cross-fact inference** is genuinely hard AND has a documented
  project NO-GO (free analogy over corpus codes, `2026-06-27-tier2.1-analogy-NEGATIVE.md`). The "it only lists"
  complaint is ~70% cheap surface realization (connectives + better aggregation), ~30% the one hard inference piece.
- **We are testing the WRONG hypothesis.** "Give the 21M a multi-fact context and ask for a fluid summary" is FREE
  ABSTRACTIVE generation — which the project already measured makes the 21M **confabulate by mixing entities** ("dog
  eats fish"), and which the field calls out (SLMs <4B hallucinate on free summary; ~25% of SOTA abstractive
  summaries contain unsupported content, Maynez 2020). Real biology/NLG does NOT free-synthesize: it builds a
  **deterministic content plan** (macroplanning) + a **structured message** (microplanning), then only
  **surface-realizes** locally. The fix is to move MORE structure into a deterministic host-side content plan and
  ask the generator for LESS free generation, not more.
- **The #1 cheap-first START:** **PLAN-then-realize with a grounded discourse plan** — keep the per-fact-faithful
  render (never regress the moat), but (a) build the message with **deterministic, entailment-checked discourse
  connectives** over the retrieved facts (RST-style: "eats meat AND chases cats"; "is a mammal BUT cannot fly";
  "the dog eats meat and SO DOES the wolf"), and (b) improve **aggregation** (group same-subject / same-verb facts:
  "the dog eats meat and chases cats" instead of two sentences). The connective/aggregation is host-side surface
  realization from a checkable plan (legitimate "body," argued in §3); the 21M is used ONLY per-clause where its
  fluent single-fact render is already validated. This is reuse-by-import, **NO `sim/` edit**, **NO new GPU train**,
  moat-preserved by construction. It closes ~70% of the perceived gap.
- **The verdict:** grouped-fact rendering IS synthesis (aggregation + RE), and it is cheaply upgradable to
  connected, contrastive, grouped prose with a deterministic grounded plan + the existing per-clause render. The
  **genuine wall** is single-pass free abstractive synthesis + open-world cross-fact INFERENCE on a ~21M model —
  the field's hallucination frontier; the honest posture is *checkable-inference-only + grounded connectives +
  hedge/say-where-knowledge-ends*, never free abstractive combination.

---

## 1. DIAGNOSIS — isolate + QUANTIFY the genuine residual

"Synthesis" is not one thing. Decomposed into the six operations that turn a set of facts into prose (Levelt
conceptualizer→formulator; Reiter-Dale document-plan→microplan→realize), with **how much already exists** and
**how much each would improve "depth"**:

| # | Sub-operation | Status in `_discuss` today | Depth-lift if added | Difficulty | Risk (confab) |
|---|---|---|---|---|---|
| (a) | **Sentence aggregation** (fuse propositions) | **PARTIAL** — groups the topic's OWN `isa/is/has` into 3 grouped sentences (`_fluidconv_chat_repl.py:288-296`); does NOT group same-subject action facts ("eats meat" + "chases cat" stay separate) | Medium | **Cheap** (host template over retrieved facts) | **None** (each fact grounded) |
| (b) | **Referring-expression generation** (it/the/a) | **PARTIAL** — pronoun "It" after the first sentence (`:292,295`); a/an + isa-vs-is article correctness | Low-Med | **Cheap** | None |
| (c) | **Discourse connectives** (and/but/so/because) | **MISSING** — sentences concatenated with ". " only | **High** (this is most of "feels like a list") | **Cheap** (deterministic, entailment-checked) | **Low** (connective asserts no new fact) |
| (d) | **Content selection / ordering** (what to say, in what order) | **PARTIAL** — neighbourhood retrieval + dedup + max_facts cap (`_neighbourhood`, `:275-280`); dlPFC spreading-activation exists but the console uses a simple gather | Medium | **Cheap-Med** | None |
| (e) | **Cross-fact INFERENCE** (derive a NEW proposition: compare, generalize, causal) | **MISSING** | **High** | **HARD** (+ documented NO-GO for free analogy) | **HIGH** — the hallucination frontier |
| (f) | **Abstractive rephrasing** (say it in fresh words) | **MISSING** (per-fact render is near-templated) | Low-Med | Med | **HIGH** on a 21M (measured: entity-mixing) |

**Isolation of the genuine residual (the SURPASS "pin down exactly which bytes" move).** The Phase-10/11 findings
show the discussion mechanism, retrieval, moat, and per-fact render are ALL GO. The residual that produces the "it
just lists" feeling is exactly the two MISSING surface operations **(c) connectives** and **(a-completed)
same-subject aggregation**, plus the one MISSING hard operation **(e) inference**. In numbers: on the console's own
elephant example, adding connectives + same-subject grouping converts *"An elephant is a mammal. It is grey. It has
trunk and tusk."* → *"An elephant is a mammal; it is grey and has a trunk and tusks."* — visibly connected prose
with **zero** new facts and **zero** new confab surface. That single change closes the bulk of the perceived gap.
The hard part (e) is a SMALL fraction of "depth" and is the one place risk lives.

**⇒ The residual is ~70% cheap surface realization (connectives + aggregation, moat-safe by construction) and ~30%
one genuinely-hard, confab-prone cross-fact inference operation with a standing NO-GO.** Do NOT accept "synthesis is
the abstractive-synthesis wall" — that conflates the cheap 70% with the hard 30%.

---

## 2. REFRAME — how real biology / cognitive-science / NLG produce multi-fact language

We are testing the wrong hypothesis. "Condition a small LM on N facts, ask for one fluid paragraph" is **free
abstractive generation**. Neither biology nor classical NLG does that; both build a **structured plan first**, then
realize locally.

- **Levelt's speaking model (Levelt 1989).** Production is a pipeline: the **conceptualizer** does **macroplanning**
  (select + order the information to satisfy the communicative intent — language-independent) and **microplanning**
  (perspective, information structure, what is new/topical — language-specific), producing a **preverbal message**;
  the **formulator** grammatically + phonologically encodes it; the **articulator** speaks. **The synthesis (what
  to combine, contrast, foreground) happens in the MESSAGE, before words** — the formulator only realizes a plan it
  is handed. Our current design skips the message and asks the word-generator to invent the plan. (Applied
  Psycholinguistics 40 (2019) 111-136; Psychology-of-Language "Standard Model of Speech Production.")
- **Reiter & Dale NLG architecture (2000).** The field-standard 3-stage pipeline: **document planning** (content
  determination + document structuring — grouping messages, ordering, RST relations), **microplanning**
  (**sentence aggregation** + lexicalization + **referring-expression generation**), **surface realization** (apply
  grammar). **Aggregation and RE are explicitly the "microplanning" synthesis operations** — so the console's
  grouped-fact output is doing genuine NLG microplanning, not a degenerate list. The missing pieces (connectives,
  same-subject aggregation, ordering) are named sub-tasks of this pipeline, and each has a **deterministic,
  rule-based** classical implementation. (Reiter & Dale, *Building NLG Systems*, Cambridge; Ch 4 Document Planning,
  Ch 5 Microplanning/aggregation, Ch 6 Surface Realisation.)
- **Rhetorical Structure Theory (Mann & Thompson 1988).** Coherent discourse = propositions linked by a small set
  of **rhetorical relations** (Elaboration, Contrast, Cause, Sequence, Joint). "Feels like a list" is precisely
  *Joint-only* discourse (no relations). The cheap upgrade is to compute a few **checkable** relations over the
  retrieved facts (Contrast when patients differ; Joint→"and" for same-subject; Cause only if a stored causal link
  exists) and realize them as connectives. This is deterministic and grounded — the relation is entailed by the
  stored facts, not invented.
- **The brain's constructive side is real but bounded (catalog G.09).** True cross-fact recombination
  ("imagine/generalize") is the **default-mode network** (mPFC, PCC/precuneus, retrosplenial, lateral temporal, HC)
  recombining stored fragments (Schacter/Addis/Buckner; Kandel 6e Ch 52 pp1300-1302). Catalog status: **missing —
  no constructive recombination**. This is operation (e), and biology gates it through the SAME hippocampal store
  that supplies episodic recall (HC lesion degrades both) — i.e. even biological "inference" is grounded
  recombination, not free generation. This licenses ONLY grounded, checkable inference; free abstractive combination
  is not what the brain does either.
- **PFC Control sequences the plan (catalog G.08 + Hagoort MUC).** dlPFC persistent activity (Rainer/Asaad/Miller
  1998; Kandel 6e Ch 52 pp1292-1294) + Hagoort's Memory-**Unification**-Control account cast left-PFC as the CONTROL
  that ORDERS retrieved items into a coherent utterance — i.e. the content-planning/ordering the project already
  has as the `SpikingSpreadingController`/dlPFC planner. The connective/ordering plan is a Control operation, not a
  generation operation.
- **The field's SLM evidence is blunt and directly on-point.** Small LMs (<4B, and the 21M far below that)
  hallucinate on free abstractive summary; the leading fix is **plan-guided generation** — fine-tune/condition the
  model to first produce an intermediate PLAN (what to say, in what order, as a blueprint) and generate FROM the
  plan, which measurably increases faithfulness (arXiv:2504.09071 plan-guided SLM summarization; AGGGEN
  arXiv:2106.05580 "ordering and aggregating while generating" ties generation to explicit facts; QA-blueprint
  attributability arXiv:2503.23204). **Plan-then-realize is the field's answer for exactly our model scale**, and
  it coincides with Levelt/Reiter-Dale. Our per-fact-faithful render is already the extreme-safe end of this; the
  upgrade is a richer PLAN (with connectives + aggregation), still realized locally.

**⇒ Reframe:** synthesis = build a deterministic **grounded discourse plan** (aggregate + relate + order — the
message/macroplan) and realize it LOCALLY, not free-generate a paragraph. Move structure INTO the plan (host-side,
checkable), ask the 21M for LESS. This is what biology, classical NLG, and current SLM-faithfulness research all
converge on.

---

## 3. RANKED cheap-first SURPASS mechanisms (what it buys · cost · anti-cheats)

All reuse-by-import; none needs a `sim/` edit; #1 needs no new GPU train. Ordered by leverage-per-cost.

### #1 (START HERE) — PLAN-then-realize: grounded discourse plan (aggregation + checkable connectives + ordering), per-clause render kept

**What it buys.** Converts "list of sentences" → connected, grouped, contrastive prose — the bulk of the "feels
like an LLM" gap — WITHOUT any free abstractive generation. Concretely: (a) **aggregate** same-subject/same-verb
facts ("the dog eats meat and chases cat"); (b) insert **checkable discourse connectives** — Joint→"and" (same
subject), **Contrast**→"but"/"whereas" iff a checkable predicate holds (two subjects' patients differ; an isa-parent
vs a negated capability), **Additive**→"and so does X" (two subjects share a verb+patient), **Elaboration**→"; it
also…"; (c) **order** by the dlPFC spreading-activation ranking already built (`SpikingSpreadingController` /
`ordered_associates`). The generator renders each aggregated clause with its VALIDATED single-fact fluent render;
the connectives/aggregation are assembled host-side from the plan.

**Is the host-side plan a "cheat"? (explicit reasoning per the BRAIN-BASED-ONLY standard.)** The standard is:
host code is legitimate only for "the environment + the body," and **surface realization of brain-supplied content
is the grey area**. The connective/aggregation layer is defensible as **body/surface realization** for three
reasons: (1) it asserts **no new fact** — every proposition is retrieved from the brain's store and VERIFY-checked;
(2) every connective is **entailed by the grounded facts** via a checkable predicate (Contrast IFF patients differ,
etc.), so it is a deterministic transform of brain content, not host cognition; (3) the CONTENT SELECTION and
ORDERING (the macroplanning that IS cognition) is done by the brain's spiking spreading-activation, not the host.
This mirrors the project's own already-accepted grey-area calls: the neural serial-order renderer (word ORDER is
brain-produced, CLAUDE.md 2026-06-16) and the `_discuss` article/pronoun/grouping templates that already ship
(`:288-322`). It is the SAME surface layer, extended with connectives. HONEST FLAG: the *inventory* of connective
templates + the entailment predicates are host-authored (a residual host structure, like the FRAME_LEXICON) — the
fully-brain-based version (Broca produces connectives; RST relations self-organized) is the deep follow-on, tracked,
not on the cheap-first critical path.

**Cost.** One runner + a plan-builder function in the console's `_discuss` (host, checkable). NO `sim/` edit, NO
train. Hours.

**Anti-cheats.** (1) **VERIFY unchanged** — every asserted SVO ∈ store; a connective that would require an
ungrounded fact is not emitted. (2) **Entailment check** — each connective must be derivable from the grounded
facts by its predicate (Contrast IFF the two patients actually differ; "so does" IFF both facts share verb+patient)
— a connective that fails its predicate is dropped. (3) **PERMUTED** — retrieve the wrong topic → the plan is about
the wrong thing (retrieval load-bearing). (4) **LESION** — empty neighbourhood → hedge. (5) **CONFAB probe** —
inject a false fact → VERIFY drops it AND its connective. (6) **Depth metric** — output has ≥1 connective/aggregated
clause and is strictly richer than the current list, on ≥3 seeds (promote to 6).

### #2 — Grounded checkable INFERENCE (compare / gist / shared-property), entailment-only

**What it buys.** The "connect ACROSS facts" flavor the owner named ("compare X and Y", "what do predators
share") — a genuinely NEW proposition, but only ones **entailed** by stored facts. "the dog eats meat; the cat eats
fish → they eat different things" (Contrast); "the dog eats meat and the wolf eats meat → both eat meat" (gist).
Every asserted fact is grounded; the inference is a **checkable predicate over retrieved facts**, not free analogy.

**Cost.** Small — a handful of entailment predicates + the compare route already partly exists (`_discuss` compare
branch, `:375-378`). No train.

**Anti-cheats.** (1) The conclusion must be **entailed** by the grounded facts (checkable: "differ" iff patients
differ; "share X" iff both assert X) — never a free generation. (2) **Respect the documented NO-GO** — free analogy
over corpus codes is a project NEGATIVE (`2026-06-27-tier2.1-analogy-NEGATIVE.md`); build ONLY on
entailment-over-retrieved-facts (safe), never on learned-code analogy. (3) Permuted: swap one entity's facts → the
relation flips correctly. (4) 6-seed for any robustness claim.

### #3 — A small SYNTHESIS fine-tune of the 21M (plan→paragraph), moat as the safety net

**What it buys.** A genuinely fluent single-pass paragraph. Input = the deterministic grounded PLAN (the ordered,
aggregated, connective-annotated fact set from #1); output = ONE fluent grounded paragraph. This is **plan-guided
generation** (the field's SLM-faithfulness fix, arXiv:2504.09071 / AGGGEN) — the model is trained to realize a
plan, NOT to free-synthesize, so it stays inside the plan. The moat/VERIFY catches residual drift.

**Cost.** MODERATE — one local fine-tune (hours on the 3090, the same scaffold as the RA fine-tune
`_fluidconv_phase2_ra_finetune.py`, which already trains on MULTI-fact contexts). Reuse-by-import; the fine-tune is
caller-side data, no `sim/` edit.

**Anti-cheats.** (1) VERIFY re-parse over ALL emitted SVOs — a fluent-but-false render is REJECTED and the turn
FALLS BACK to the #1 deterministic plan render (graceful degradation; the moat makes an imperfect fine-tune SAFE —
QUANTIFY the fallback rate, don't assume it's low). (2) Held-out generalization (never-trained fact-sets), not
train-set memorization. (3) Untrained-generator control fails. (4) Entity-mixing probe (the measured Phase-10
failure: "dog eats fish") must be caught by VERIFY at ~100%. (5) 6-seed. **Only build #3 if #1's deterministic
render is judged not fluent-enough** — #1 already closes most of the gap and is zero-risk; #3 buys fluency polish at
a train cost and a managed drift risk.

### #4 (LAST RESORT — 2 lines) — a bigger fluency model

A 2× larger generator (e.g. the 88.6M already spiking-validated, or a small instruct model) would free-synthesize
more coherently. **Cost to the thesis:** it directly violates "MINIMIZE the transformer" — a 4-6× larger model to
buy connectives the deterministic plan (#1) already delivers for free is a bad trade. Note only as the fallback IF
#1+#2+#3 all fail the depth bar (they will not for the cheap 70%).

### Project-toolkit pieces to reuse (not new mechanisms)

- **dlPFC spreading-activation content selection/ordering** (`SpikingSpreadingController`,
  `content_selection_spiking.py:328-467`; `elaborate`/`ordered_associates`, `brain_conversational_agent.py:717-731`)
  — the macroplanning (what to say, in what order) is brain-native and GO; feed it into #1's plan.
- **`chain_of_thought` / `reason_chain`** (multi-hop, GO — `2026-06-17`, `2026-06-27-tier2.2`) — supplies grounded
  connective CHAINS ("dog chases cat, cat chases mouse") for Sequence relations.
- **The RA fine-tune scaffold** (`_fluidconv_phase2_ra_finetune.py`, already MULTI-fact-context trained) — the #3
  fine-tune reuses it directly.
- **Generative-replay proposer + flagged-hypothesis channel** (DiscursiveTurn `:429-435`) — for #2/any speculative
  inference: FLAG + never store (the moat reframe, `feedback_moat_not_hard`).

---

## 4. Recommended de-risk (what to build first, what it proves, its anti-cheats)

**Build #1 first: a grounded discourse-plan render for `_discuss`.** A new de-risk runner
(`_fluidconv_multifact_synthesis_derisk.py`) that, over the Phase-10/11 KB (+ a Wikidata-loaded topic like
"elephant"), (i) retrieves the neighbourhood (existing), (ii) builds a **deterministic discourse plan** —
aggregate same-subject/same-verb facts, attach checkable connectives (Joint/Contrast/Additive/Elaboration), order
by the dlPFC ranking, (iii) realizes each aggregated clause via the existing single-fact fluent render + VERIFY,
joined by the planned connectives.

**What it proves.** That "it lists facts" → "connected, grouped, contrastive grounded prose" is achievable with
NO free abstractive generation, NO `sim/` edit, NO train, moat intact — closing ~70% of the perceived gap and
delivering the owner's "talk about ideas" flavor safely.

**GO bar (≥3 seeds, promote to 6).** DEPTH: output contains ≥1 aggregated clause AND ≥1 checkable connective, and
is strictly richer than the current per-fact list (e.g. the elephant example becomes one connected sentence).
GROUNDED: 0 ungrounded SVOs (VERIFY). CONNECTIVE-CORRECT: every connective passes its entailment predicate
(Contrast only when patients differ, etc.) — measured, not assumed. LESION: empty neighbourhood → hedge. PERMUTED:
wrong topic → wrong-topic plan (retrieval load-bearing). CONFAB: injected false fact + its connective both dropped.

**Then, gated on #1:** add #2 (entailment-checkable compare/gist) for explicit "compare X and Y"; consider #3 (the
plan→paragraph fine-tune) ONLY if the deterministic render #1 is judged not fluent enough — and even then the moat
+ fallback-to-#1 is the safety net (quantify the fallback rate as the honest cost).

---

## 5. Genuine-wall verdict — surpassable-and-how-cheaply vs irreducible-and-why

**SURPASSABLE, and cheaply (the comfortable "it just lists / abstractive-synthesis wall" verdict does NOT survive
the SURPASS round):**
- The grouped-fact rendering IS synthesis — it already does NLG **aggregation + referring-expression generation +
  article agreement**, the microplanning operations of Levelt/Reiter-Dale.
- The "feels like a list" residual is ~70% **cheap, moat-safe surface realization**: **discourse connectives** +
  **same-subject aggregation** + **ordering**, all deterministic and entailment-checked over the brain's retrieved,
  VERIFY-clean facts (#1). This is reuse-by-import, no `sim/` edit, no train.
- Bounded **checkable cross-fact inference** (compare/gist/shared-property, #2) delivers "connect across facts"
  safely, provided it is entailment-over-retrieved-facts, NOT free analogy (respecting the documented NO-GO).
- A **plan-guided fine-tune** (#3) can add genuine single-pass fluency, made safe by the moat + graceful fallback —
  the field's own SLM-faithfulness answer, at our exact model scale.

**GENUINELY IRREDUCIBLE (hedge honestly here, precisely why):**
- **Free single-pass abstractive synthesis + open-world cross-fact INFERENCE on a ~21M model.** The project MEASURED
  it (multi-fact context → entity-mixing confab, Phase-10) and the field confirms it (SLMs <4B hallucinate on free
  abstractive summary; ~25% of SOTA abstractive summaries unsupported, Maynez 2020; free analogy over corpus codes
  is a project NO-GO). This is the point where the honest posture is checkable-inference-only + grounded connectives
  + FLAG-speculation + **say where knowledge ends** — the communicable-brain-not-RAG boundary, a feature not a
  failure.
- **Why irreducible:** genuine abstractive combination requires either (a) a much larger model (violates the thesis)
  or (b) the missing **constructive-recombination** substrate (catalog G.09 DMN, "missing — no constructive
  recombination") — a new mechanism CLASS, not a cheap surface fix. The cheap-first path (#1/#2) deliberately routes
  AROUND this wall by moving structure into a checkable plan, so the wall is not on the critical path for "talk
  about ideas in connected prose."

**One-line verdict.** "It only lists facts" is a DISGUISED boundary: grouped rendering is already NLG synthesis
(aggregation + RE), and it is cheaply upgradable to connected, contrastive, grouped prose via a deterministic
grounded discourse plan (connectives + aggregation, entailment-checked, moat-preserved) + the existing per-clause
render — closing ~70% of the gap with no `sim/` edit, no train; free abstractive single-pass synthesis + open-world
inference on a 21M is the genuine wall, hedged honestly by checkable-inference-only + grounded connectives +
saying where the brain's knowledge ends.

---

## Key citations

**Project (file:line / finding).** `_fluidconv_chat_repl.py:268-322` (`_discuss` — the grouped `isa/is/has`
aggregation + pronoun RE + a/an article layer = the synthesis-already-present; the connective/same-subject-aggregation
gap); `:375-378` (compare branch — the #2 seam); `_fluidconv_phase10_discussion_derisk.py:88-116` (render-each +
per-sentence VERIFY + the measured entity-mixing confab on multi-fact contexts — WHY free synthesis fails on the
21M); `2026-07-01-fluid-conversation-phase10-open-ended-discussion-GO.md` (honest ceiling: "still LISTS facts");
`2026-07-01-fluid-conversation-phase11-grow-grounded-knowledge-GO.md` (richness scales with the KB; the elephant
grouped example; the render-vocab lever); `2026-07-01-open-ended-grounded-discussion-scoping.md` §1.2-1.3 (the
90%/60%/30% decomposition + the isolated residual — the direct precursor); `content_selection_spiking.py:328-467`
(dlPFC spreading-activation = brain-native content selection/ordering for #1's plan);
`brain_conversational_agent.py:717-731` (`elaborate`/`ordered_associates`); `2026-06-17-multihop-query-chain-GO.md`,
`2026-06-27-tier2.2-chain-of-thought-GO.md` (grounded connective chains); `2026-06-27-tier2.1-analogy-NEGATIVE.md`
(free analogy over corpus codes = NO-GO — the (e) boundary); `_fluidconv_phase2_ra_finetune.py:74-108` (the
multi-fact-context RA fine-tune scaffold reused by #3); `feedback_moat_not_hard_lossy_memory_ok` (the moat reframe).

**Catalog (`sim-catalog/references/feature-catalog.md`).** G.07 pre-SMA/SMA internally-generated sequences (Kandel
6e Ch 34 pp822-828) — serial-order production; G.08 PFC working memory / persistent activity, Rainer/Asaad/Miller
1998 (Ch 52 pp1292-1294) — the Control that orders the plan; G.09 Imagination / future simulation as constructive
memory — DMN recombination, Schacter/Addis/Buckner, status "missing" (Ch 52 pp1300-1302) — the (e) inference
substrate + why it is a wall, not a cheap fix; G.10 language as hierarchical symbolic system (Ch 55 pp1370-1372);
G.11 dual-stream language / Hickok-Poeppel (Ch 55 pp1380-1387); G.12 Broca production + grammatical processing (Ch
55 pp1382-1384) — the fully-brain-based connective producer (deep follow-on); G.13 Wernicke comprehension (Ch 55
pp1384-1385).

**Literature (current, cited — not from memory).** Levelt 1989, *Speaking: From Intention to Articulation*
(conceptualizer = macroplanning [select+order, language-independent] + microplanning [perspective/info-structure];
formulator; articulator) — Applied Psycholinguistics 40 (2019) 111-136; Psychology-of-Language "Standard Model of
Speech Production." Reiter & Dale 2000, *Building Natural Language Generation Systems* (Cambridge) — the 3-stage
pipeline: document planning (content determination + structuring), microplanning (**sentence aggregation** +
lexicalization + **referring-expression generation**), surface realization; Ch 4/5/6. Mann & Thompson 1988,
*Rhetorical Structure Theory* — discourse coherence via rhetorical relations (Contrast/Cause/Elaboration/Joint);
"list" = Joint-only. Xu et al. 2021, *AGGGEN: Ordering and Aggregating while Generating* (arXiv:2106.05580) —
aggregation+ordering tied to explicit facts improves faithfulness. Plan-guided SLM summarization 2025
(arXiv:2504.09071) — SLMs <4B hallucinate on free summary; plan-guided grounding increases faithfulness (directly
supports #1/#3). QA-blueprint attributability 2025 (arXiv:2503.23204) — blueprint/plan → attributable generation.
Maynez et al. 2020, *On Faithfulness and Factuality in Abstractive Summarization* (ACL) + Ji et al. 2022, *Survey
of Hallucination in NLG* (ACM CSUR) — ~25% hallucination in abstractive, extractive/plan-grounded stays faithful,
hybrid is the fix. Word-level hallucination control in data-to-text (Rebuffel et al. 2022) — the (f) risk.
