# Open-ended grounded DISCUSSION — scoping (from fact-Q&A to "discuss ideas using relevant knowledge")

**Date:** 2026-07-01
**Type:** Deep-research + reference-catalog review (the standing research-gate opening move for a new direction).
READ-ONLY; no `sim/`/GPU edit. Companion to the two 2026-07-01 fluid-conversation docs (roadmap + gap-assessment);
this one scopes the SPECIFIC new owner ask below.

**Owner directive (2026-07-01) being scoped.** Push the brain's conversation from grounded FACT-Q&A toward **fluid,
open-ended, in-depth DISCUSSION of ideas/concepts the brain has RELEVANT information on but no explicit "answer"
to.** Not "what does the dog eat? → meat" (exact-match retrieval), but "what do you think about dogs?" / "tell me
about predators" / "how are dogs and cats different?" → the brain should DISCUSS using its relevant grounded
knowledge (dog eats meat, chases cat, is big; cat eats fish; …), EXTRAPOLATING / connecting / elaborating across
facts, even with no exact stored answer — while staying grounded (not fabricating facts). This is the
`project_communicable_brain_not_rag` vision.

Terms defined once, on first use. Biology cited from the catalog (`sim-catalog/references/feature-catalog.md`),
Kandel 6e, and current literature — not from memory.

---

## 0. TL;DR (read this first)

- **Most of "discuss ideas" is ALREADY BUILT and 3-seed GO — it is just not wired into the fluid console.** The
  `DiscursiveTurn` orchestrator (`_discursive_turn_stage0_derisk.py`, GO 3-seed, 24 facts) already does the exact
  behavior the owner describes: on an open question with no stored answer it runs `gather_discuss(topic)` — the
  **dlPFC spiking spreading-activation** (`SpikingSpreadingController`) walks the brain's OWN association graph to
  collect the topic's grounded neighborhood, plus FLAGGED speculation, and frames it *"Here's how I think about
  it:"* (`_discursive_turn_stage0_derisk.py:160,376-395,553`). The *fluid* console
  (`_fluidconv_chat_repl.py`) is the piece that is still **GATE-FIRST** — `what_does(subj,verb) is None → "I don't
  know."` (`_fluidconv_chat_repl.py:80-81`) — and its `_elaborate` surfaces only ONE extra single-clause fact
  (`:94-111`). **The residual is an integration gap, not a missing mechanism.**
- **Decomposition of "open discussion" by how-much-exists** (quantified in §1): **(a) multi-fact retrieval around a
  topic ≈ 90% exists** (the association graph + spreading-activation + `reason_chain`/`chain_of_thought` are all
  multi-seed GO); **(b) synthesis into fluent multi-sentence discourse ≈ 60% exists** (the RA-fine-tuned 21M
  generator is *already trained on MULTI-fact contexts* and is RA-FAITHFUL, but the current callers render each
  fact as a SEPARATE clause and concatenate — no single fluent synthesis pass); **(c) EXTRAPOLATION beyond stored
  facts ≈ 30% exists and is the genuinely-new + risky part** (a FLAGGED-hypothesis channel exists and never
  stores, but true cross-fact inference — comparison, generalization, "predators share eating-meat" — is largely
  unbuilt, and abstractive synthesis is the field's hallucination-prone frontier).
- **The #1 cheap-first START (details §3):** wire the fluid console's open-question path to the **existing
  `DiscursiveTurn` discuss channel**, and add ONE synthesis step — retrieve the topic's grounded neighborhood
  (association-graph spreading) → condition the RA generator on the **MULTIPLE retrieved facts at once** (the
  multi-fact context the fine-tune already learned) → generate ONE fluent multi-sentence grounded discussion →
  **VERIFY every asserted known-entity SVO is grounded** (the moat, unchanged), permitting non-fact narrative/
  connective glue but DROPPING any ungrounded FACT claim. This is reuse-by-import, **NO `sim/` edit**, no new GPU
  train (the fine-tune already handles multi-fact contexts), and it converts the moat from hard-abstain to
  **grounded elaboration with honest hedging** (`feedback_moat_not_hard`).
- **The honest verdict (§4):** grounded multi-fact discussion + *bounded, retrieval-traceable* extrapolation is
  cheaply achievable now. Free abstractive synthesis and open-world inference beyond the stored/adjacent facts is
  the field's genuine wall (abstractive summarization hallucinates where extractive does not; the fix is a
  hybrid **extract-then-synthesize-then-verify** pipeline — which is exactly what the project's store+moat give).
  The honest boundary is hedged by SAYING what the brain relates and FLAGGING what it is unsure of, never
  asserting an ungrounded fact.

---

## 1. DIAGNOSIS — what changes, and what already exists

### 1.1 The precise change: GATE-FIRST → DISCUSS-FALLBACK

The current *fluid* console's turn logic is a strict retrieval gate:

```
p = what_does(subj, verb)          # spiking FHRR recall
if p is None: return "I don't know."   # ← the gate-first abstain (_fluidconv_chat_repl.py:80-81)
```

An open question ("what do you think about dogs?") has no `(subj, verb)` cue → the console either falls into the
one-fact `describe` path (`:135-141`) or abstains. There is **no path that assembles a topic's whole grounded
neighborhood and discusses it.** The owner's ask is exactly to replace the bare abstain with an ENGAGE-AND-DISCUSS
fallback: *when there is no single answer, say what you DO relate about the topic.*

Crucially, the DIFFERENT (non-fluid) console already does this. `DiscursiveTurn.discuss(msg, topic=...)` routes:
"question + gate MISS → the (D) discuss-without-an-answer path (engage, not a bare abstain)"
(`_discursive_turn_stage0_derisk.py:553`). So the change is **connect the fluid console to the already-GO discuss
turn**, not invent it.

### 1.2 Decomposition of "open discussion" — how much already exists (quantified)

**(a) Multi-fact retrieval around a topic — ≈ 90% EXISTS (multi-seed GO).**
The pieces are all built and validated:
- **The association graph** — `BrainConversationalAgent._assoc_graph()` builds `concept → {concept: weight}` from
  the agent's OWN stored facts (co-occurring agent/action/patient), OR reads it from the substrate-learned sparse
  Hebbian recurrent when `enable_learned_assoc` (`brain_conversational_agent.py:697-715, 312-318`). This is the
  topic's neighborhood.
- **dlPFC spiking spreading-activation** — `SpikingSpreadingController` (`content_selection_spiking.py:328-467`)
  embodies the graph as inter-assembly synapses; driving the topic into spiking working memory SPREADS activation
  along those synapses; the most-active unsaid assembly is the selection; a `SaidTrace` inhibition-of-return
  makes the discussion PROGRESS through the topic instead of alternating two neighbors (`:344-347`). This is the
  brain-native "what to bring up next about this topic" — Collins-Loftus spreading activation realized in spikes.
- **`elaborate(topic)`** (`brain_conversational_agent.py:717-731`) — returns the next on-topic associate via the
  above controller; **`ordered_associates(topic)`** gives the whole ranked neighborhood (used by DiscursiveTurn's
  `gather_discuss`).
- **Multi-hop chase** — `reason_chain(cue, actions)` (caller-supplied relations) and **`chain_of_thought(start)`**
  (SELF-cued: the brain itself selects each next relation by learned association strength, re-cleaning between
  hops so error doesn't compound) — both multi-seed GO (`brain_conversational_agent.py:667-686`;
  `2026-06-17-multihop-query-chain-GO.md`; `2026-06-27-tier2.2-chain-of-thought-GO.md`). This is cross-fact
  connection over the graph.
The ~10% residual: retrieval is over the STORED SVO facts + corpus association graph; "relevant information the
brain has" = exactly that set. No new retrieval mechanism is needed for discussion; the neighborhood IS the
material.

**(b) Synthesis into fluent multi-sentence discourse — ≈ 60% EXISTS.**
- The **RA-fine-tuned ~21M generator** (`_fluidconv_phase2_ra_finetune.py`) is the fluency engine. Decisive detail
  for this scoping: **it was trained on MULTI-fact contexts.** Each fine-tune example's context is
  `fact + 1–2 distractor facts about OTHER subjects`, shuffled (`_make_example`, lines 74-82, 86-108), and the
  answer must use the RIGHT fact from that multi-fact context. So conditioning the generator on several retrieved
  facts at once is **already inside its trained distribution** — no re-train needed to feed it a neighborhood.
- It is **RA-FAITHFUL** — when the prompt states a fact different from its own bias, it follows the PROVIDED fact
  (Phase-2 eval RA-faithfulness arm, `_fluidconv_phase2_ra_qa_eval_derisk.py:148-158`), and it GENERALIZES to
  novel entities (Phase-5/6 GO). So it will render a topic's grounded facts fluently, grounded to what it was fed.
- `FTFaculty.answer(facts_ctx, question)` (`_fluidconv_phase2_ra_qa_eval_derisk.py:75-95`) takes an ARBITRARY
  `facts_ctx` string and a question and returns a focused fluent answer. Passing a MULTI-fact `facts_ctx` +
  a discuss-style question ("tell me about the dog", "how do dogs and cats compare?") is a one-line caller change.
The ~40% residual: **the current callers do NOT synthesize.** `DiscursiveTurn` renders each proposition
INDEPENDENTLY (`_render_verify` per SVO, `_discursive_turn_stage0_derisk.py:402-435`) and concatenates → a
paragraph of single clauses, not one flowing multi-sentence discussion. The fluid console renders ONE fact
(`_fluidconv_chat_repl.py:84-89`). The unbuilt step is a SINGLE synthesis pass conditioned on the whole
neighborhood (with the per-fact VERIFY preserved). The generator can do it; nothing calls it that way yet.

**(c) EXTRAPOLATION beyond stored facts — ≈ 30% EXISTS; the genuinely-new + risky part.**
- What exists: a **FLAGGED-hypothesis channel.** DiscursiveTurn's (N)/(D-flagged) propositions come from the
  `GenerativeReplayProposer` over the LEARNED graph, are rendered with a hedge + a HYPOTHESIS marker, and are
  **NEVER stored** (`_discursive_turn_stage0_derisk.py:429-435`). So "the brain guesses, flags it, doesn't commit
  it as fact" is already the design. The moat is already the softer "grounded elaboration + flag speculation"
  shape the owner wants (`feedback_moat_not_hard`).
- What is largely UNBUILT: **true cross-fact INFERENCE** — comparison ("dogs eat meat, cats eat fish → they
  differ in prey"), generalization/gist ("dog chases cat, cat chases mouse → chasing is common"), category
  induction ("dog and wolf both eat meat → predators eat meat"), analogy. These require operating on the RELATIONS
  between multiple facts, not just retrieving+rendering them. The project has a NEGATIVE here: **analogy over the
  corpus codes is a documented NO-GO** — only a curated factored-relation KB works
  (`2026-06-27-tier2.1-analogy-NEGATIVE.md`; gap-assessment §4). So structured relational extrapolation is a real
  frontier.
- The RISK: extrapolation is where hallucination lives. The field's evidence is blunt — abstractive synthesis
  (paraphrase/combine) introduces false information where extractive (copy) stays faithful; ~25% of
  state-of-the-art abstractive summaries contain hallucinated content (Maynez et al.; Survey of Hallucination in
  NLG). So the extrapolation MUST be constrained to be grounded-and-traceable, and the ungrounded-fact-claim
  failure mode caught by VERIFY. This is precisely why the #1 mechanism keeps VERIFY load-bearing and only relaxes
  it for non-fact connective/opinion glue.

### 1.3 The genuine residual / hard part, isolated (the SURPASS "quantify the residual" move)

Stripping away what already exists, the truly-new work is small and specific:
1. **A synthesis step** that conditions the generator on the retrieved MULTI-fact neighborhood in ONE pass (the
   generator already supports it; no caller does it). *Cheap; the bulk of the leverage.*
2. **A per-clause VERIFY over a multi-fact synthesis** — the moat must survive the shift from "verify one rendered
   fact" to "verify each asserted known-entity fact inside a flowing paragraph, allowing non-fact glue." The
   machinery exists (`_extract_all_svos` + `store_keys` membership, already used per-fact); the change is to run
   it over ALL extracted SVOs of the synthesis and DROP/regenerate if any is ungrounded. *Cheap.*
3. **Bounded cross-fact inference** (comparison / gist / category) — the genuinely-hard, higher-risk part. Most
   of it can be done SAFELY by TEMPLATED relational framing over RETRIEVED facts ("dogs eat meat; cats eat fish;
   so they eat different things") where every asserted fact is grounded and only the connective is generated —
   NOT by free abstractive inference. Free analogy over corpus codes is the documented NO-GO; do not build on it.
   *Moderate; scope tightly to grounded-traceable relational framing.*

⇒ The "blocker" is mostly-already-solved; the genuine residual is a **synthesis + verify wiring** (cheap) plus a
**tightly-scoped grounded relational-framing** step (moderate, risk-managed). No new mechanism CLASS is required
for the cheap-first step.

---

## 2. REFRAME — how REAL biology produces open-ended grounded discussion

The owner's target maps cleanly onto a well-characterized biological pattern: **spreading activation over semantic
memory, gated by prefrontal cognitive control that selects a coherent discourse path, with constructive
recombination for the extrapolative part — and reconsolidation/gist making retrieval inherently editable
(the biological license for grounded hedged extrapolation).**

- **Spreading activation over semantic memory** (Collins & Loftus 1975; formalized in **ACT-R**, Anderson &
  Pirolli 1984 / Anderson 1983): activating a concept spreads activation to associated concepts along weighted
  links; retrieval is ranked by base-level activation (recency/frequency) + contextual (goal-driven) activation.
  This is EXACTLY the project's `SpikingSpreadingController` + `SaidTrace` + `_assoc_graph`
  (`content_selection_spiking.py:328-467`) — the brain-native "bring up the next relevant thing about the topic."
  The project's implementation is *more* biological than ACT-R's symbolic version: relevance is computed in
  SPIKES (first-spike latency / sustained rate along cortico-cortical assembly synapses).
- **Prefrontal cognitive control selects the discourse path.** Catalog **G.08** (PFC working memory / persistent
  activity, content-specific, D1-DA-dependent; Kandel 6e Ch 52 pp1292-1294) holds the topic active across the
  turn; **Hagoort's "Unification"/Control** account of language (the MUC model: Memory-Unification-Control) casts
  left-PFC (BA 45/47) as the CONTROL that sequences retrieved lexical/semantic items into a coherent utterance —
  i.e. the dlPFC dialogue planner the project already uses to ORDER the discussion. (Hagoort 2005, 2013; the
  project's DiscursiveTurn `_planner` = this Control.)
- **Constructive recombination = the extrapolative part.** Catalog **G.09** (Imagination / future simulation as
  constructive memory; Kandel 6e Ch 52 pp1300-1302): the default-mode network (mPFC, PCC/precuneus, retrosplenial,
  lateral temporal, hippocampus) RECOMBINES stored elements to simulate events that were never experienced — the
  SAME network for "remember the last beach trip" and "imagine the next one" (Schacter/Addis/Buckner). This is
  the biology of "extrapolate/connect across facts": the brain does NOT only retrieve; it recombines stored
  fragments into novel-but-grounded constructions. Catalog status: "missing — no constructive recombination."
  The #1 mechanism supplies a MINIMAL, bounded form of this (generator-synthesized connective over retrieved
  fragments), not a full DMN.
- **Reconsolidation / schemas / gist — the license for grounded extrapolation and hedging.** Catalog **J.34**
  (Memory imperfections as features — schemas, gist, false memory; Kandel 6e Ch 52 pp1306-1308): human memory
  adaptively prioritizes GENERALIZABLE structure (gist) over verbatim detail, and "reconsolidation makes
  retrieval inherently editable." Biologically, then, a brain that DISCUSSES a topic by gist-level recombination
  — and is sometimes uncertain — is normal, not a bug. This grounds the moat REFRAME: from **hard-abstain** ("I
  don't know") to **grounded elaboration with honest hedging** — the brain says what it DOES relate (grounded,
  verified) and FLAGS what it is unsure of (the flagged-hypothesis channel), never ASSERTING an ungrounded fact.
  This is exactly `feedback_moat_not_hard_lossy_memory_ok`: keep the moat as a PLUS (no fabricated *assertions*),
  not a hard gate that blocks all discussion.

**The moat reframed, precisely.** The invariant is preserved: *never ASSERT a fabricated FACT.* But the turn is
allowed to (i) assert only GROUNDED (verified-in-store) facts plainly, (ii) offer FLAGGED hypotheses with a hedge
(never stored), and (iii) use non-fact connective/opinion glue that makes no factual claim. Structurally identical
to DiscursiveTurn's type-aware VERIFY gate (`_discursive_turn_stage0_derisk.py:397-435`), which the field
independently validates as the right design: SELF-RAG's critique-generate loop and FEQA's QA-based faithfulness
check are the RAG-world analogues of the project's proposer→VERIFY-by-re-parse loop.

---

## 3. RANKED cheap-first mechanisms

Each: mechanism + catalog/lit cite + reusable pieces + cheap-first de-risk + anti-confabulation controls. Ordered
by leverage-per-cost. All reuse-by-import; none needs a `sim/` edit; #1 needs NO new GPU train.

### #1 (START HERE) — Retrieve-the-neighborhood → synthesize-in-ONE-pass → VERIFY-each-fact ("grounded discussion")

**The mechanism.** On an open/topic question (no single stored answer), (i) RETRIEVE the topic's grounded
neighborhood via the association-graph spreading-activation (the brain's own "what's relevant about this"); (ii)
condition the RA generator on the MULTIPLE retrieved facts at once (a multi-fact `facts_ctx`) + a discuss-style
question; (iii) generate ONE fluent multi-sentence grounded discussion; (iv) VERIFY every asserted known-entity
SVO is in the store — allow non-fact narrative/connective/opinion glue, but DROP (and regenerate, or fall back to
per-fact rendering) if any asserted FACT is ungrounded. Biologically: spreading activation (Collins-Loftus /
ACT-R) selects the material; PFC Control (Hagoort MUC / G.08) sequences it; the generator supplies fluency; the
re-parse VERIFY is the gist-editable-but-not-fabricating guard (J.34).

**Catalog / lit cites.** Spreading activation — Collins & Loftus 1975; ACT-R Anderson 1983. PFC Control of
language — Hagoort MUC (2005, 2013); catalog G.08 (Kandel Ch 52 pp1292-1294). Multi-evidence grounded synthesis
+ faithfulness verify — MEGA-RAG (multi-evidence guided answer refinement), SELF-RAG (critique-generate),
FaithfulRAG (fact-level), FEQA (QA-based faithfulness). Extract-then-abstract hybrid faithfulness — Survey of
Hallucination in NLG (Maynez 25% hallucination in pure abstractive; hybrid is the fix).

**Reusable pieces (all present).**
- `BrainConversationalAgent._assoc_graph()` + `elaborate` / `ordered_associates` + `SpikingSpreadingController`
  (`content_selection_spiking.py:328-467`) — the neighborhood retrieval (spiking spreading-activation).
- `DiscursiveTurn.gather_discuss(topic)` (`_discursive_turn_stage0_derisk.py:376-395`) — ALREADY assembles the
  dlPFC-ordered grounded neighborhood + flagged speculation. Reuse it directly; only the RENDER changes.
- `FTFaculty.answer(facts_ctx, question)` (`_fluidconv_phase2_ra_qa_eval_derisk.py:75-95`) — takes an arbitrary
  multi-fact context; the fine-tune was TRAINED on multi-fact contexts (`_fluidconv_phase2_ra_finetune.py:74-82`)
  and is RA-FAITHFUL. Pass the neighborhood as `facts_ctx`.
- `_extract_all_svos` + `_fact_key` + `store_keys` (`_fluidconv_chat_repl.py:35,86-89`) — the per-fact VERIFY,
  applied to ALL extracted SVOs of the synthesis.

**Cheap-first de-risk (CPU, no train; scale up only if GO).** In a new runner
(`_opendiscuss_grounded_synthesis_derisk.py`): teach the 22-fact curriculum; for a set of topic prompts ("tell me
about the dog", "what do you think about predators", "how are the dog and cat different"), (a) gather the grounded
neighborhood (≥2 facts), (b) synthesize ONE discussion via `FTFaculty.answer(multi_fact_ctx, q)`, (c) VERIFY. GO
bar (≥3 seeds, promote to 6): **DEPTH** — the synthesis references ≥2 distinct grounded facts (strictly richer
than the one-fact answer); **GROUNDED** — every asserted known-entity SVO ∈ store (0 ungrounded fact-claims);
**FLUENT** — one multi-sentence paragraph, not concatenated clauses (median > 1 sentence, reads as discourse);
**ENGAGE-not-abstain** — the no-single-answer topic ("predators") produces a grounded discussion, not "I don't
know."

**Anti-confabulation controls (the load-bearing part).**
1. **VERIFY-each-fact (the moat).** Every asserted known-entity SVO must re-parse to a STORED fact; ungrounded
   fact-claims are DROPPED. Non-fact glue (connectives, opinions, "they seem different") asserts no fact → allowed.
   The INVARIANT (identical to DiscursiveTurn): the paragraph contains only {verified-stored-certain} ∪
   {flagged-hypothesis} ∪ {non-fact-glue}.
2. **LESION control** — sever the retrieval (empty neighborhood) → the synthesis must have nothing to ground on →
   either abstains or the VERIFY drops everything. Grounding must be load-bearing.
3. **PERMUTED control** — feed the generator the neighborhood of a DIFFERENT topic → the VERIFY must reject the
   asserted facts as not-about-this-topic / not-grounded (proves the discussion tracks the retrieved neighborhood,
   not a generic prior).
4. **Confab probe** — inject a plausible-but-false fact into the synthesis (the `_ConfabOneRenderer` pattern,
   `_discursive_turn_stage0_derisk.py`) → VERIFY drops the confabulated sentence while grounded ones survive.
5. **Traceability** — every emitted FACT must map to a retrieved store fact (log the provenance), so "extrapolation
   grounded in retrieved knowledge, traceable" is auditable.

**Why #1 first:** it is the highest leverage per unit cost — one runner + one caller change unlocks the owner's
core behavior (discuss-not-abstain, multi-fact, fluent), reuses the GO DiscursiveTurn discuss channel and the
already-multi-fact-trained generator, needs NO `sim/` edit and NO new train, and keeps the moat load-bearing. It
also *simultaneously* delivers the moat reframe (grounded elaboration + hedging).

### #2 — Grounded RELATIONAL framing (comparison / gist / category over retrieved facts)

**The mechanism.** The bounded-extrapolation step. For "how are dogs and cats different?" / "what do predators
have in common?", retrieve the facts for BOTH entities (or the category members), then assert a RELATIONAL
statement whose fact-content is grounded and whose relation is TEMPLATED/framed, NOT freely inferred: "the dog
eats meat; the cat eats fish; so they eat different things" (compare); "the dog eats meat and the wolf eats meat,
so they both eat meat" (gist/category). Every asserted fact is grounded; only the relational connective is
generated. Biologically: constructive recombination (G.09 DMN) constrained to grounded fragments + gist
extraction (J.34).

**Catalog / lit cites.** G.09 constructive recombination (Kandel Ch 52 pp1300-1302); J.34 gist/schema (Ch 52
pp1306-1308); conceptual combination (Gagné & Shoben; not in catalog — lit). Analogy (Gentner structure-mapping)
— cite as the *aspiration*, NOT the mechanism (see the NEGATIVE below).

**Reusable pieces.** `what_does`/`who_does` for both entities; `_assoc_graph` for shared neighbors;
`reason_chain`/`chain_of_thought` for connective chains; the FLAGGED-hypothesis channel for any relation that
isn't directly grounded.

**Cheap-first de-risk.** Comparison/gist templates over 2 entities' grounded facts; measure grounded-correctness
(both compared facts ∈ store) + that the relational claim is derivable from the grounded facts (e.g. "different"
iff the two patients differ). GO bar: comparison correct on taught pairs, 0 ungrounded fact-claims, ≥3 seeds.

**Anti-confabulation controls.** (i) The relational conclusion must be ENTAILED by the grounded facts (a checkable
predicate: "differ" iff patients differ; "share X" iff both facts assert X) — NOT a free generation. (ii) Permuted
control: swap one entity's facts → the relation flips correctly. (iii) **Explicit NEGATIVE to respect:** free
analogy over corpus codes is a documented NO-GO (`2026-06-27-tier2.1-analogy-NEGATIVE.md`) — do NOT build relational
framing on learned-code analogy; build it on entailment over RETRIEVED facts (checkable), which is safe.

**Why #2 second:** it delivers the "connect/contrast across facts" flavor the owner explicitly named ("how are
dogs and cats different"), but it is riskier (extrapolation) and must be tightly scoped to entailment-checkable
relations. Do it after #1 proves the grounded-discussion loop + moat hold.

### #3 — DMN-style constructive recombination for open hypotheticals (higher ceiling, research-gated)

**The mechanism.** For genuinely open prompts ("imagine a new animal like a dog") — the extrapolative apex — a
constructive-recombination step (G.09) recombines grounded fragments into a novel-but-grounded HYPOTHESIS,
emitted only as FLAGGED speculation. This is the full "imagine/extrapolate" behavior; it is the highest-ceiling,
highest-risk item and fires the research gate (it is a new mechanism CLASS — constructive recombination, catalog
status "missing").

**Catalog / lit cites.** G.09 DMN constructive memory (Schacter/Addis/Buckner; Kandel Ch 52 pp1300-1302); the
generative-replay proposer (`b2 GenerativeReplayProposer`) is the nearest existing seed.

**Cheap-first probe (only after #1/#2).** Does recombining ≥2 grounded fragments into a flagged hypothesis, then
VERIFYing it is NOT asserted as fact, produce coherent-and-clearly-hedged speculation? GO bar: hypotheses are
coherent, ALWAYS flagged, NEVER stored, and a confab probe on the hypothesis channel is caught.

**Anti-confabulation controls.** The hypothesis channel NEVER stores and ALWAYS hedges (structural, per
DiscursiveTurn `:429-435`); the confab + permuted + lesion arms as in #1.

**Why #3 last:** it is the deepest and least-cheap; it is the parallel higher-ceiling science bet, not the first
move. Most of the owner's ask ("discuss what you know", "compare") is delivered by #1+#2 without it.

---

## 4. VERDICT — cheaply achievable now vs the genuine wall

**Cheaply achievable NOW (reuse-by-import, no `sim/` edit, #1 needs no new train):**
- **Grounded multi-fact DISCUSSION** — engage-and-discuss on a topic using the brain's own association-graph
  neighborhood, synthesized into ONE fluent multi-sentence grounded reply, VERIFY-clean. **~90% of the pieces
  exist and are 3-seed GO** (DiscursiveTurn discuss channel + spreading-activation + the multi-fact-trained
  RA generator); the residual is a synthesis + verify WIRING. This is the owner's core ask and it is cheap.
- **Bounded, entailment-checkable relational framing** — comparison / gist / shared-property over RETRIEVED
  facts (#2), where every asserted fact is grounded and the relation is checkable. Delivers "how are X and Y
  different / what do predators share" safely.
- **The moat reframe** — from hard-abstain to grounded-elaboration-with-honest-hedging, matching
  `feedback_moat_not_hard`, WITHOUT weakening the never-assert-a-fabricated-fact invariant (the flagged channel +
  per-fact VERIFY are already the design).

**The genuine, irreducible wall (hedge honestly at this boundary):**
- **Free abstractive synthesis / open-world inference beyond the stored+adjacent facts.** The field is blunt:
  abstractive synthesis hallucinates where extractive stays faithful (~25% hallucinated content in SOTA
  abstractive summaries; Maynez / NLG-hallucination survey). Free analogy over the brain's corpus codes is a
  documented project NO-GO (`2026-06-27-tier2.1-analogy-NEGATIVE.md`). Truly open-domain discussion (topics
  outside the vocabulary/knowledge) is the same field wall the roadmap already named
  (`2026-07-01-fluid-conversation-mechanisms-roadmap.md` §4: manage via domain-constraint + retrieval-augmentation
  + abstention, not a transformer-free open-domain conversationalist).
- **How to hedge at the boundary (honestly):** the field's own fix is EXACTLY the project's shape — a HYBRID
  **extract → synthesize → verify** pipeline (extractive pre-filter isolates grounded facts; abstractive step
  synthesizes a coherent narrative; a faithfulness check verifies). The project's spiking store + no-confab moat
  ARE the distinctive extractive+verify halves that most RAG systems LACK. So the honest posture is: the brain
  DISCUSSES fluently over what it relates (grounded, traceable, verified), FLAGS its guesses, and — beyond that
  neighborhood — SAYS SO ("that's about all I relate to it" / "I'm not sure past that"), rather than fabricating.
  The abstention/hedge IS the truthful boundary, and it is a feature (the communicable-brain-not-RAG UX), not a
  failure.

**One-line verdict.** "Discuss ideas using relevant knowledge + bounded extrapolation" is ~90%-built and cheaply
finishable by WIRING the fluid console to the GO `DiscursiveTurn` discuss channel and adding ONE multi-fact
synthesis+VERIFY pass (the generator already handles multi-fact contexts) — delivering the moat reframe for free;
free open-world abstractive inference is the field's genuine wall, hedged honestly by grounding+flagging+saying
where the brain's knowledge ends.

---

## Key citations

**Project (file:line / finding):**
`_fluidconv_chat_repl.py:80-81,84-111` (the GATE-FIRST fluid console + one-fact `_elaborate` — the thing to move
beyond); `_discursive_turn_stage0_derisk.py:160,376-395,402-435,553` (the DiscursiveTurn discuss channel + type-
aware VERIFY moat — ALREADY GO, 3-seed, the mechanism to reuse); `content_selection_spiking.py:328-467`
(SpikingSpreadingController — spiking spreading-activation over the association graph); `brain_conversational_agent.py:667-731`
(reason_chain / chain_of_thought / _assoc_graph / elaborate); `_fluidconv_phase2_ra_finetune.py:74-82,86-108`
(the RA fine-tune TRAINED ON MULTI-FACT CONTEXTS — the enabler for multi-fact synthesis with no re-train);
`_fluidconv_phase2_ra_qa_eval_derisk.py:75-95,148-158` (FTFaculty.answer(arbitrary ctx) + RA-FAITHFULNESS);
`2026-06-17-multihop-query-chain-GO.md`, `2026-06-27-tier2.2-chain-of-thought-GO.md` (multi-hop / self-cued chain,
GO); `2026-06-27-tier2.1-analogy-NEGATIVE.md` (free analogy over corpus codes = NO-GO — the extrapolation boundary
to respect); `2026-07-01-fluid-conversation-mechanisms-roadmap.md`, `-gap-assessment.md` (the sibling docs).

**Catalog (`sim-catalog/references/feature-catalog.md`):** G.08 PFC working memory / persistent activity (Kandel
6e Ch 52 pp1292-1294); G.09 Imagination / future simulation as constructive memory — DMN recombination (Ch 52
pp1300-1302); G.11 dual-stream language / Hickok-Poeppel (Ch 55 pp1380-1387); G.13 Wernicke auditory-to-semantic
(Ch 55 pp1384-1385); J.34 Memory imperfections as features — schemas, gist, false memory; "reconsolidation makes
retrieval inherently editable" (Ch 52 pp1306-1308).

**Literature (current, cited — not from memory):** Collins & Loftus 1975, *A Spreading-Activation Theory of
Semantic Processing* (Psychol. Review). Anderson 1983 / Anderson & Pirolli 1984, *Spreading Activation* (ACT-R,
JVLB / J. Exp. Psychol.). Hagoort 2005/2013, the MUC (Memory-Unification-Control) model of language (TICS;
Front. Psychol.). Schacter, Addis & Buckner 2007-2012, constructive episodic simulation / DMN. Maynez et al.
2020, *On Faithfulness and Factuality in Abstractive Summarization* (ACL) + Ji et al. 2022, *Survey of
Hallucination in NLG* (ACM CSUR) — ~25% hallucination in abstractive, extractive stays faithful, hybrid is the
fix. Durmus/Diab 2020, *FEQA* (QA-based faithfulness) — the RAG analogue of re-parse VERIFY. Asai et al. 2023,
*SELF-RAG* (critique-generate loop). MEGA-RAG 2025 (multi-evidence guided answer refinement); FaithfulRAG 2025
(fact-level conflict) — multi-evidence grounded synthesis + faithfulness. Extract-then-abstract hybrid (LLM
summarization strategy reviews, 2025).
