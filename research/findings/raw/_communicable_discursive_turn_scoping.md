# Communicable-brain DISCURSIVE multi-proposition turn — scoping (READ-ONLY)

**Date:** 2026-06-25
**Scope:** The (1) communicable-brain RICHNESS frontier — the owner's TOP north-star (`project_communicable_brain_not_rag`, incl. REFINEMENT-2). Move from the current **store→retrieve→compose→render-or-abstain** loop (a one-channel-per-turn, RAG-shaped UX) to a genuine **ENGAGE-AND-DISCUSS** conversational turn: "LLM behavior, minus hallucinations, plus persistent growth/memory." This scopes the **DISCURSIVE multi-proposition turn** — the next build AFTER the Stage A (fused single-channel turn) + Stage B (production-agent wire-in) that already exist. Per the standing practice (scope a new mechanism CLASS before building). NO edits/runs/webapp performed; design only.

**The north-star, verbatim (memory `project_communicable_brain_not_rag` + REFINEMENT-2):** a natural conversation loop where the brain (a) addresses the last message + the conversation-so-far + is influenced by ALL past conversations (persistent memory), (b) **ENGAGES + DISCUSSES** rather than retrieve-or-abstain ("what is the meaning of life" → openly discusses; "what kind of animal is a dog" → not a terse "mammal" but discusses *while* answering), (c) picks its own response depth but goes DEEPER when asked, (d) adapts to the user's style, (e) the moat shifts from "abstain-when-no-fact" to "**never ASSERT a fabricated fact**" (discuss/speculate freely, FLAG the speculative, never state a falsehood as fact). The BRAIN does the cognition — **REJECT the LLM-free-generate cheat** (the LLM is fluency-only). Richness scales with the brain's KNOWLEDGE (the curriculum axis).

---

## 0. What already exists (the inputs — VERIFIED from source + the GO findings)

| Piece | Where | What it does | Granularity |
|---|---|---|---|
| **`CommunicableTurn`** (Stage A, GO 3-seed) | `_communicable_turn_stageA_derisk.py` | Routes ONE message → ONE channel: `_known_fact_channel` / `_novel_channel` / `_phatic_channel` / teaching-`feedback`. `turn(msg, cue, topic)` returns **ONE channel record**. | **single-proposition, single-channel** |
| **`RichAnswerComposer`** (GO) | `rich_answer_composer.py` | For a KNOWN-fact question, GATHERS a multi-fact set — (a) direct recall + (b) multi-hop `query_chain` + (c) dlPFC `elaborate` — and RENDERS a multi-SENTENCE paragraph, VERIFY-checking each sentence (drops any that fail). The `NeuralDiscoursePlanner` (dlPFC spiking spreading-activation) drives WHICH/ORDER/STOP. | **multi-sentence, but ALL certain/known-fact** |
| **Stage B wire-in** (built) | `brain_conversational_agent.py` (`communicable_mode`, `converse()`, `communicable_feedback()`, `speak_value_Q()`) | The production agent routes `converse(msg)` through a lazily-built `CommunicableTurn` over the agent's OWN composer/moat; learned-Q persists via the bundle. Default-OFF = byte-identical. | single-channel (delegates to Stage A `turn`) |
| GENERATE (spiking draw) | `_genfrontier_b2_generative_replay_derisk.GenerativeReplayProposer` (default-on spiking sampler) | Draws NOVEL, graph-plausible, non-contradictory, topic-relevant triples; each filler a `cp_firing_states` spiking draw. | per-proposition |
| DECIDE-to-speak (spiking WTA) | `_value_salience_appraisal_derisk.SpikingSpeakAccumulator` | A real Izhikevich speak/silence WTA on a `SimulationBridge` slice; the speak DECISION is a pool's FIRING. | per-proposition |
| LEARN-talkativeness | `_learned_talkativeness_derisk.LearnedSpeakValue` (+ `SignedLearnedSpeakValue`) | A per-context speak-value Q updated by the three-factor rule from perceived feedback; signed (elaborate +, stop −); DA-lesion-provable. | per-context |
| graded confidence | Probe-1 `plausibility_score`, `hedge_for` | maps worth/plausibility → a hedge ("I'd say…" → "I'm not sure, but I'd guess…") + a confidence scalar. | per-proposition |
| RENDER+VERIFY | `_grounded_lang_integration_derisk._extract_svo_from_prose` + `BrainConversationalAgent.parse` + `ChatBrain._verify` + `QwenRenderer`/`StubRenderer` | fluency faculty renders an SVO → re-parse the prose → require == the brain's SVO; drop/regen on drift. | per-sentence |
| multi-hop reason | `RFPhasorComposer.query_chain` / agent `reason_chain` | role-structured pointer-chase; abstains per hop (moat at every hop). | — |
| multi-turn anaphora | `multi_turn_agent.MultiTurnAgent` + `SpikingLoopContextBuffer` | a persistent spiking WM holds discourse referents; a pronoun resolves to a held concept. | — |
| onebrain composer | `one_brain_composer.OneBrainComposer` | the whole pipeline as ONE persistent spiking loop (the production-default conversational substrate). | — |

**Crucial observation (the seam):** the single-channel `CommunicableTurn` and the multi-sentence `RichAnswerComposer` **never compose with each other**. `RichAnswerComposer` is multi-sentence but every sentence is a CERTAIN known fact (it only gathers from the gate+chain+elaborate path; it has no novel/flagged channel). `CommunicableTurn` has the novel/flagged channel but emits exactly ONE proposition per turn. **Neither produces a paragraph that MIXES proposition TYPES** (some certain, some flagged-hypothesis, some "let me think out loud", a phatic acknowledgement) — which is exactly what "engage and discuss" is.

---

## 1. THE GAP — precisely what is missing

The discursive engage-and-discuss turn is **not a new neural mechanism**. Every cognitive primitive it needs is GO. What is missing is a **multi-proposition ASSEMBLY layer** that, for ONE user message, gathers a SET of propositions of MIXED type, appraises and orders them, decides how many to say (depth), renders+verifies each, and emits a coherent paragraph that visibly distinguishes certain from speculative. Concretely, four missing pieces:

### GAP-1 — Mixed-type proposition ASSEMBLY (the core gap)
Today a turn is `intent → one channel → one (or, for rich, N-certain) propositions`. The discursive turn needs `message → a SET of typed propositions across FOUR proposition channels, assembled into one reply`:

- **(C) KNOWN-FACT-CERTAIN** — the direct recall + role-chase chain + grounded elaboration (the existing `RichAnswerComposer` gather). Asserted plainly.
- **(N) NOVEL-FLAGGED-HYPOTHESIS** — generated, graph-plausible, non-contradictory propositions ABOUT the topic (the `_novel_channel` / b2 proposer), each FLAGGED with a graded hedge. Never stored.
- **(D) DISCUSS-WITHOUT-A-STORED-ANSWER** — for an open question the brain has NO crisp fact for ("what is the meaning of life"), assemble RELATED grounded fragments (what the brain DOES know that is adjacent in the PPMI graph) + flagged speculation, framed as "here's how I think about it" — NOT a terse abstain, NOT a fabricated fact.
- **(P) PHATIC / discourse-glue** — a non-claim social/connective utterance ("Good question." / "Hi!"), makes no factual claim.

The missing object is a **`DiscursiveTurn`** (working name) that produces an ordered, typed proposition list, NOT a single channel record. This is the genuinely-new glue.

### GAP-2 — Cross-type channel ROUTING + BLENDING (not either/or)
The current router (`IntentRouter.classify`) picks ONE intent. The discursive turn keeps the intent classifier but treats it as a **mixing prior**, not an exclusive switch:
- a `question` whose gate HITS → lead with (C)-certain, then OPTIONALLY add (N)/(D) elaboration if the depth budget allows and worth clears the speak threshold (the "discuss while answering 'what kind of animal is a dog'" behavior);
- a `question` whose gate MISSES → the **(D) discuss-without-an-answer** path (the meaning-of-life behavior) instead of a bare "I don't know";
- an `opinion` → lead with (N)-flagged, optionally grounded by (C) facts the brain holds about the topic;
- a `phatic` → (P) only;
- `teaching` → the LEARN update + a re-run at the new depth.

The mix per turn is decided by the SAME appraisal already built (worth = learned-Q·value + plausibility + familiarity) gating EACH candidate proposition through the SAME `SpikingSpeakAccumulator` — so "how much / which types to say" is the brain's spiking speak decision applied per-candidate, not a host policy.

### GAP-3 — The moat SHIFT for a MIXED paragraph (speak-while-flagging at proposition granularity)
The existing moat operates at two grains that are each already GO but NOT yet combined in one paragraph:
- `RichAnswerComposer`: per-sentence VERIFY drops any sentence whose re-parse ≠ its gathered (certain, stored) SVO — but it has no notion of a *deliberately-speculative* sentence (everything is meant to be certain).
- `CommunicableTurn._novel_channel`: every novel emission is FLAGGED and never stored, and a who/what on it still abstains — but it's a SINGLE proposition, not interleaved with certain ones.

The discursive turn must enforce the moat **per-proposition, by TYPE**: a (C) proposition is held to "re-parse == a STORED fact" (the hard moat); a (N)/(D) proposition is held to "FLAGGED + verify-re-parse == the BRAIN's PROPOSED SVO + never asserted as certain + never stored" (the relaxed moat). The missing piece is the **type-tagged verify+render gate** that applies the right rule to each proposition AND renders certain vs flagged visibly distinct in ONE paragraph. (Mis-tagging a flagged proposition as certain would be the catastrophic failure — the gate must make that structurally impossible.)

### GAP-4 — DEPTH adaptation from follow-ups (the brain picks depth, deeper on request)
`RichAnswerComposer` has a `max_sentences` cap + a `_gather_more` follow-up path; `LearnedSpeakValue` has a per-context Q raised by "tell me more". But **depth is not yet a single controlled variable** that (a) the brain sets per-turn from the topic's learned-Q + worth distribution, and (b) a follow-up RAISES (immediately, this turn, AND durably via the Q update). The missing piece is a **depth controller**: the number of propositions emitted = how many candidates clear the spiking speak threshold at the current talkativeness; "tell me more" raises the talkativeness (DA burst → Q↑) so MORE candidates clear next turn AND re-runs the held topic deeper now. (This composes the appraisal + the learned-Q + the follow-up walk that all already exist — it just needs to be the explicit knob.)

**Summary of the gap:** all the neurons are built. What's missing is the **assembly-and-appraisal shell** that turns the existing per-proposition primitives into a per-MESSAGE, multi-proposition, mixed-type, depth-controlled, type-safe-moat paragraph. It is the `RichAnswerComposer` generalized from "N certain facts" to "N typed propositions across {certain, flagged, discuss, phatic}", with the `CommunicableTurn` appraisal deciding the mix and depth.

---

## 2. HOW THE EXISTING PIECES COMPOSE into the discursive turn

The build is **reuse-by-import + one new orchestrator** (`DiscursiveTurn`), mirroring how Stage A fused three GO mechanisms into a routing shell. Mapping each job to its owner:

| Job in the discursive turn | Owner (reuse-by-import) | New glue? |
|---|---|---|
| classify intent → mixing prior | `IntentRouter.classify` (Stage A) | reuse; reinterpret output as a prior, not a switch |
| (C) gather certain facts (direct + chain + elaborate) | `RichAnswerComposer.gather` / `_chain_facts` / `_elaboration_facts` + `NeuralDiscoursePlanner` | **reuse VERBATIM** — this IS the (C) channel |
| (N) propose novel flagged candidates about X | `CommunicableTurn._novel_channel` internals = `GenerativeReplayProposer` (spiking draw) + `propose_candidates_about` | reuse; expose the candidate SET (not just the single best) |
| (D) discuss-without-an-answer (adjacent grounded fragments + flagged speculation) | `RichAnswerComposer._facts_mentioning`/`_elaboration_facts` (adjacent grounded) + the (N) proposer (flagged) + `NeuralDiscoursePlanner.ordered_associates` (on-topic ordering) | **new framing glue** — combine adjacent-grounded + flagged into a "here's how I think about it" set; no new mechanism |
| (P) phatic / discourse-glue | `CommunicableTurn._phatic_channel` + `PHATIC_REPLIES` | reuse; add 2-3 connective glue phrases ("Good question.") |
| appraise EACH candidate (worth) | `CommunicableTurn.worth` (learned-Q value + plausibility + familiarity, per-seed normalised) | reuse VERBATIM |
| decide emit-vs-silent PER candidate (the mix + depth) | `SpikingSpeakAccumulator.decide` (spiking WTA) | reuse; call it per-candidate to select WHICH/HOW-MANY |
| graded confidence / hedge per (N)/(D) | `hedge_for`, `plausibility_score` | reuse VERBATIM |
| render each proposition fluently | `ChatBrain.render` / `QwenRenderer.render_svo(+_regen)` / `StubRenderer` | reuse VERBATIM |
| VERIFY each rendered proposition (type-aware) | `ChatBrain._verify` + `_extract_svo_from_prose` + `agent.parse` | reuse the re-parse; **new glue** = the per-type acceptance rule (C: ==stored; N/D: ==proposed + flagged) |
| order + cap (depth) | `RichAnswerComposer` ordering + `NeuralDiscoursePlanner` STOP | reuse; **new glue** = the depth = #candidates-clearing-the-spiking-threshold, capped |
| multi-turn referent / "tell me more about that" | `MultiTurnAgent` WM loop + `RichAnswerComposer._gather_more` | reuse; tag a (N)/(D) proposition as a HYPOTHESIS-referent (owner-steer #5 in the prior scoping) |
| learned-Q persistence (style across sessions) | `developed_brain_io` bundle + `speak_value_Q` blob (Stage B) | reuse |

**The new code is ONE orchestrator** (`DiscursiveTurn`), about the size of `RichAnswerComposer` + the Stage-A routing, that:
1. classifies intent → a mixing prior;
2. assembles a TYPED candidate pool: (C) from `RichAnswerComposer.gather`, (N) from the proposer's candidate set, (D) from adjacent-grounded + flagged, (P) from the table;
3. appraises each candidate by `worth` and runs `SpikingSpeakAccumulator.decide` per candidate to SELECT the emitted set + its size (depth);
4. orders them (lead with certain when the gate hit; lead with the framing/phatic glue otherwise; `NeuralDiscoursePlanner` orders the elaboration);
5. renders+VERIFIES each with the type-aware moat rule, dropping failures;
6. emits a paragraph with certain vs flagged visibly distinct, and writes the topic + any hypothesis-referent into the WM loop.

It exposes `discuss(msg, cue=, topic=)` returning a structured record `{paragraph, propositions:[{type, svo, certain, hedge, confidence, verified, speak_margin}], depth, abstained_certain, intent}` — a strict superset of both `CommunicableTurn.turn` and `RichAnswerComposer.answer`, so the console/agent can adopt it behind the SAME default-OFF flag.

**This is the `RichAnswerComposer` lifted one level:** RichAnswerComposer already proved "gather a SET → render multi-sentence → per-sentence VERIFY → moat-safe paragraph". DiscursiveTurn keeps that exact skeleton and (i) widens the gather to MIXED types, (ii) inserts the per-candidate spiking appraisal/decision as the selector, (iii) makes VERIFY type-aware. Everything load-bearing is already validated.

---

## 3. THE BRAIN-BASED CHECK — every cognitive step is the brain's; the LLM is fluency-only

Per `feedback_brain_based_only_standard`: anything cognitive between sensation and "the body acting" must be neurons/synapses; host code is legitimate only for the environment + bookkeeping. Walking the discursive turn:

| Step | Who does it | Brain-based? | Note |
|---|---|---|---|
| classify intent (the mixing prior) | `IntentRouter` (host regex) | **host bookkeeping** — ACCEPTABLE | routing/parsing of the user's literal text = the "sensory render" boundary; it picks WHICH brain ops run, it does no cognition. (A later step could make this neural, like the spoken-instruction parser; not required.) |
| (C) recall + chain + elaborate selection | the composer's spiking recall + `query_chain` spiking hops + the dlPFC spiking spreading-activation (`NeuralDiscoursePlanner`) | **YES — spiking** | already the brain's ops |
| (N) propose novel candidates | `GenerativeReplayProposer` spiking soft-WTA draw (`cp_firing_states`) | **YES — spiking** | provenance: 0 host-rng draws on the draw path (the `_followon2` check) |
| (D) adjacent-grounded fragments | the PPMI graph + the dlPFC ordering | **YES — the brain's learned graph + spiking spread** | the "what's adjacent" is the brain's learned co-occurrence |
| appraise worth (value/plausibility/familiarity) | `worth` reads the LEARNED-Q (three-factor plasticity), the brain's PPMI plausibility, the brain's familiarity | **YES** | the VALUE axis is reward-modulated plasticity (DA-lesion-provable) |
| decide emit-vs-silent per candidate (mix + depth) | `SpikingSpeakAccumulator.decide` — Izhikevich WTA pool firing | **YES — spiking** | the speak choice is a pool's firing, not `if score>thr` |
| graded confidence / hedge | `hedge_for(worth)` (host map worth→hedge string) | **host bookkeeping — ACCEPTABLE** | the WORTH it maps is brain-computed; the string is surface bookkeeping (like rendering) |
| render prose | the fluency faculty (spiking-Qwen / template-stub) | **fluency-only — by design** | supplies SURFACE FORM only |
| VERIFY (re-parse → SVO == brain's SVO) | `agent.parse` (the brain's BridgeParser comprehension) | **YES — the brain's own comprehension** | this is what makes the LLM fluency-only enforceable |
| order / cap | host ordering + the dlPFC STOP | mostly bookkeeping; the on-topic ordering + STOP is neural (latency rank) | — |
| learned-Q update on feedback | three-factor rule | **YES** | DA-lesion abolishes it |

**Tempting shortcuts to FLAG and REJECT (the brain-based traps specific to the discursive turn):**
1. **Letting the LLM free-generate the discussion** (the obvious cheat — "just ask the 0.5B to discuss the meaning of life"). REJECTED by design: every (N)/(D) proposition is the BRAIN's proposed SVO; the LLM only phrases it; VERIFY re-parses and rejects anything the LLM injected. The LESION anti-cheat (sever the proposal → let the faculty free-generate → VERIFY must reject) is the gate. This is the SAME contract Probe-1 + the grounded-lang INTEGRATION already passed.
2. **A host "pick the top-K by score" depth selector** instead of the spiking decision. REJECTED: depth = #candidates whose `SpikingSpeakAccumulator.decide` returns speak-wins. The host may CAP (a safety ceiling, like `max_sentences`) but must not BE the cognitive selector.
3. **A host coherence/discourse-planner** (e.g. an LLM "make this flow") that reorders or invents connective content. REJECTED for content: ordering is the dlPFC latency rank; the only host connective text is the fixed phatic-glue table (makes no claim) — never LLM-generated linking claims.
4. **A host "the meaning of life is…" templated essay** for the (D) path. REJECTED: the (D) content is assembled from the brain's OWN adjacent grounded facts + its OWN flagged speculation; the host only frames ("Here's how I think about it:") — a non-claim wrapper, like the phatic table.
5. **A host fabricated-fact filter** ("does this sound true?"). NOT NEEDED and would be a cheat: the moat is structural (re-parse == a STORED fact for certain; flagged-only otherwise), not a host plausibility judgement of the surface.

---

## 4. THE NO-CONFAB MOAT SHIFT — never ASSERT a fabricated fact, while ENGAGING on a no-stored-answer

The owner's REFINEMENT-2 moves the moat from "abstain-when-no-fact" to "**never assert a fabricated fact**" — discuss/speculate freely, FLAG the speculative. The shift is realized as a **per-proposition, type-tagged gate** (each rule is ALREADY individually GO; the design combines them safely in ONE paragraph):

**The invariant (the new HARD line):** *no proposition is ever rendered as a CERTAIN assertion unless its prose re-parses to a fact the brain has STORED.* Everything else is rendered FLAGGED or not at all. Equivalently: **certainty requires a stored-fact re-parse; speculation is always hedged; nothing speculative is ever stored.**

**The gate, per proposition type:**

- **(C) certain** — accept-as-certain iff `_verify(prose) == svo AND svo ∈ stored_facts`. (= the current `RichAnswerComposer` per-sentence VERIFY, which already drops anything that fails.) Mis-render → DROP (not downgraded to flagged silently — dropping is safe; a certain claim that won't verify must not be spoken at all).
- **(N) / (D) flagged** — accept-as-flagged iff `_verify(prose) == the_brain's_proposed_svo AND the surface carries a hedge AND the svo ∉ stored_facts` (it's novel by construction). Rendered with the `hedge_for(worth)` prefix + a "hypothesis" marker. NEVER written to the store.
- **(P) phatic / glue** — makes no factual claim → no moat check (cannot fabricate a fact because it asserts none). The table is fixed, non-generative.

**Why this keeps 0-fabrication while ENGAGING on a no-stored-answer (the meaning-of-life case):**
- The brain has NO stored fact for "the meaning of life" → the (C) channel gathers NOTHING → no certain claim is even a candidate. So a fabricated certainty is structurally impossible.
- The (D) path then assembles ONLY (i) the brain's OWN adjacent grounded facts (each held to the (C) rule — so any grounded fragment IS a real stored fact, asserted as certain and on-topic) and (ii) FLAGGED speculation from the proposer (each held to the (N) rule — generated, verify-re-parsed, hedged, never stored). The paragraph engages ("Here's how I think about it: [a real adjacent fact]. I'm not sure, but I'd guess [a flagged hypothesis].") — substantive, NOT a terse abstain, and EVERY claim is either a verified stored fact or an explicitly-flagged guess. **No fabricated fact is ever asserted.**
- If even the (D) assembly is empty (no adjacent grounded facts AND no graph-plausible candidate), the brain falls back to an HONEST framed non-answer ("That's a deep one — I don't have settled knowledge there.") — the graceful version of abstain, NOT a fabrication.

**The structural guarantee (mis-tagging is impossible):** the proposition's `type` is set at GATHER time by which channel produced it (C from the gate, N/D from the proposer), and the render/verify dispatch keys on that type. A flagged proposition CANNOT be rendered certain because the certain renderer path is reachable only for type-C propositions, and a type-C proposition requires `svo ∈ stored_facts`. The console UI (a later stage) must render the two visibly distinct (a badge/hedge) — the prior console scoping already makes "certain vs hypothesis must be visually distinct" a non-negotiable; this turn FEEDS that distinction at proposition granularity.

**Anti-cheat for the moat shift (the GO bars in §5):** (i) 0 (C)-proposition leaks — a who/what on every emitted certain proposition matches a stored fact, and a who/what on every flagged proposition ABSTAINS (it was never stored); (ii) every (N)/(D) proposition carries a hedge + a hypothesis tag (no silent certainty); (iii) the LESION (free-generate the content) is caught-by-VERIFY across the MIXED paragraph; (iv) a confabulating faculty's wrong sentence is DROPPED from the mixed paragraph while the truthful + correctly-flagged ones survive (= the `_ConfabOneRenderer` adversarial probe in `rich_answer_composer.py`, now run on a mixed C+N+D paragraph).

---

## 5. CHEAP-FIRST DE-RISK PLAN (the smallest demonstrable engage-and-discuss turn) + GO BARS

**The smallest thing to build first** is a CPU de-risk on the existing testbed brain (the b2 PPMI cortex over the 8×8 taxonomy + `RFPhasorComposer` D=64 + `TemplateStubFaculty`, host-oracle draw for tractability — the EXACT harness Stage A/B already run on), with a NEW `DiscursiveTurn` that produces a MIXED-type paragraph for a handful of scripted messages. Reuse-by-import, NO `sim/` edit. This isolates the ASSEMBLY layer from the GPU faculty + the console (those are later stages, identical to the prior console scoping's Stage C/D).

### Stage 0 (the de-risk) — `_communicable_discursive_turn_derisk.py` (CPU, ≥3 seeds → 6 if GO)

**Build:** a `DiscursiveTurn` that, per scripted message, gathers the typed candidate pool (C/N/D/P), appraises + spiking-decides the emitted set + depth, renders+verifies type-aware, and emits a mixed paragraph. Scripted cases (each a named discursive behavior):
1. **discuss-while-answering** — a known-cue question ("what is a dog" with a stored `dog chase cat`-style fact) → leads CERTAIN, then ADDS ≥1 grounded elaboration and/or ≥1 flagged hypothesis (depth > 1, mixed types). Contrast: the thin single-fact answer.
2. **engage-without-an-answer** — an open question with NO stored fact ("what is the meaning of life" / an untaught topic that IS in-vocab) → a (D) paragraph of adjacent-grounded + flagged, NOT a terse abstain, EVERY claim verified-or-flagged.
3. **opinion** — "what do you think about X" → leads FLAGGED (N), optionally grounded by a (C) fact the brain holds about X.
4. **phatic** — "hi" → (P) only, no claim.
5. **depth-up** — case 1 or 2 followed by "tell me more" → MORE propositions clear the threshold this turn (immediate) AND the topic's learned-Q rises (durable); the DA-lesion abolishes the rise.
6. **moat probes** — a who/what on an EMITTED flagged proposition → ABSTAINS (never stored); a fully-unknown word → "I don't know that word yet".

**GO bars (Stage 0):**
- **MIXED ASSEMBLY:** ≥1 scripted turn emits a paragraph with ≥2 proposition TYPES (e.g. C+N or C+D), and the discuss-while-answering turn has depth ≥ 2 (strictly richer than the thin 1-fact answer). The engage-without-an-answer turn emits ≥2 grounded-or-flagged propositions (NOT an abstain).
- **MOAT (HARD, the core bar):** 0 (C)-proposition leaks — every certain proposition re-parses to a STORED fact; every (N)/(D) proposition is FLAGGED + a who/what on it ABSTAINS + it is NEVER stored. The mixed paragraph contains ONLY {verified-stored-certain} ∪ {flagged-hypothesis} — no unverified certainty, no fabricated fact.
- **BRAIN-DOES-COGNITION:** the per-candidate emit/silence decision is the spiking accumulator's firing (provenance: read from `cp_firing_states`/the WTA pool, 0 host `if score>thr` selecting content); the (N)/(D) content is the brain's proposed SVO (the LESION free-generate arm is caught-by-VERIFY across the mixed paragraph).
- **DEPTH-ADAPTS:** the "tell me more" round increases the emitted proposition count on the held topic (immediate) AND raises the learned-Q there (durable, monotone over rounds); the **DA-lesion abolishes both** the Q rise and the extra emissions.
- **NON-REGRESSION:** with the discursive layer OFF, `CommunicableTurn.turn` + `RichAnswerComposer.answer` behave byte-identically (the new orchestrator is purely additive).

**Anti-cheats (Stage 0):**
- **shuffled-PPMI-graph** → the (N)/(D) groundedness collapses ≥3× (the brain's learned structure is load-bearing, not the host framing).
- **free-generate LESION** → VERIFY rejects on the MIXED paragraph (run on the actual emitted propositions; the content is the brain's).
- **confab-one renderer** → a confabulated sentence is DROPPED from the mixed C+N+D paragraph; the truthful + correctly-flagged ones survive (= the existing `_ConfabOneRenderer` probe, on a mixed paragraph).
- **decorrelated-credit** → the depth/talkativeness rise is per-context (shuffling the eligibility flattens the taught/untaught depth gap — depth is per-topic learning, not global vigor).
- **value⊥plausibility** corr ≈ 0 (the value axis driving depth is not relabeled plausibility).

### Stage 1 — agent/console wire-in (mirror the prior console scoping's Stages B–D)
Once Stage 0 is GO: add `DiscursiveTurn` as the turn engine behind the SAME default-OFF flag the agent/`ChatBrain` already use (`converse`/`communicable_mode` → route to `discuss`); extend the `/api/brain-chat` schema with the typed-proposition list + per-proposition certain/hedge/confidence (additive); render certain vs flagged visibly distinct in the console; depth follow-up via the teaching UI; persist Q via the bundle. GO bar = flag-OFF byte-identity (the full conversational test suite passes verbatim) + the Stage-0 invariants hold on the production agent + a live console session shows a genuinely discursive, type-distinct, moat-safe paragraph. (The GPU spiking-Qwen drop-in + the latency note are identical to the prior console scoping's Stage C — wall-clock, not a VRAM wall; keep local.)

---

## 6. HONEST SCOPE

- **The discursive LOOP is buildable NOW.** Every neural mechanism it needs is GO (GENERATE, DECIDE-to-speak, LEARN-talkativeness, multi-hop, dlPFC discourse-planner, multi-turn anaphora, type-aware VERIFY) and the two precedents — `CommunicableTurn` (single-channel) and `RichAnswerComposer` (multi-certain-sentence) — bracket exactly the new object. The build is an **assembly-and-appraisal orchestrator**, reuse-by-import + one new class, NO `sim/` edit — the SAME shape as the Stage A fusion. This is engineering composition, not a new mechanism class, so the research gate's heavy round is NOT required (this doc IS the lightweight scoping; the cheap-first de-risk is the validation).
- **The RICHNESS is knowledge-gated (the curriculum axis).** A SMALL brain (the 8×8 testbed, or an early-week developed bundle) discusses **thinly but HONESTLY** — it engages, mixes a certain fact with a flagged guess, and adapts depth, but the *substance* is bounded by how much it knows. The meaning-of-life turn from a 24-fact brain will assemble a few adjacent grounded fragments + a flagged hypothesis — genuinely discursive, but shallow. **Depth and substance EMERGE as the curriculum grows** (the develop-loop / artificial-life axis): more facts → richer (C) gather + (D) adjacency + more graph-plausible (N) candidates. The de-risk validates the LOOP and the SAFETY at small scale; the richness scales with knowledge, exactly as the owner framed.
- **What this turn does NOT add (deliberate, deferred):** true free-form generation beyond the brain's SVO algebra (the composer's exact-inverse FHRR bind is still the principled idealization; the learned generative LM / BPTT-SNN free-generation is the separate `project_generative_sequence_frontier`); cross-similar-concept generalization in the discussion (the PPMI/visual generalization arc, already GO, would FEED richer (D) adjacency but is a separate input); and style-adaptation to the user (north-star (d)) beyond depth — a natural follow-on once depth-adaptation is validated (the same learned-Q machinery extended to style features). None of these block the discursive LOOP; they each make it RICHER.
- **The moat shift is conservative + sound.** "Never assert a fabricated fact" is enforced STRUCTURALLY (certainty requires a stored-fact re-parse; everything else is flagged or dropped), so the engage-and-discuss freedom is bought WITHOUT any fabrication risk — the relaxation the owner sanctioned (`feedback_moat_not_hard_lossy_memory_ok`) realized at proposition granularity. This is the load-bearing safety claim and the primary GO gate.

---

## Appendix — file map (where each thing lives / will be touched)

**Reuse-by-import (read-only, GO):**
- `research/runners/_communicable_turn_stageA_derisk.py` — `CommunicableTurn` (`_known_fact_channel`/`_novel_channel`/`_phatic_channel`/`worth`/`render_and_verify`/`propose_candidates_about`), `SignedLearnedSpeakValue`, `IntentRouter`, `build_communicable_brain`
- `research/runners/rich_answer_composer.py` — `RichAnswerComposer` (`gather`/`_chain_facts`/`_elaboration_facts`/`render_paragraph`/`_render_one_verified`), `NeuralDiscoursePlanner`, `_ConfabOneRenderer` (the adversarial probe to reuse)
- `research/runners/_genfrontier_b2_generative_replay_derisk.py` — `GenerativeReplayProposer` (the spiking GENERATE), `shuffle_graph`
- `research/runners/_value_salience_appraisal_derisk.py` — `SpikingSpeakAccumulator` (the spiking DECIDE)
- `research/runners/_learned_talkativeness_derisk.py` — `LearnedSpeakValue`, `context_code`, `code_overlap` (the LEARN)
- `research/runners/_communicable_brain_probe1_whatdoyouthink.py` — `plausibility_score`, `hedge_for` (graded confidence)
- `research/runners/_grounded_lang_integration_derisk.py` — `_extract_svo_from_prose`, `_build_inflection_map` (the VERIFY re-parse)
- `research/runners/brain_chat_tui.py` — `ChatBrain` (`gate`/`render`/`_verify`), `QwenRenderer`, `StubRenderer`, `QuestionRouter`
- `research/runners/brain_conversational_agent.py` — `BrainConversationalAgent` (`converse`/`communicable_mode`/`parse`; the wire-in seam)
- `research/runners/multi_turn_agent.py` — `MultiTurnAgent` (the discourse WM, for the hypothesis-referent)
- `research/runners/developed_brain_io.py` — bundle IO (the `speak_value_Q` persistence home)

**New code (the integration glue):**
- `research/runners/_communicable_discursive_turn_derisk.py` (Stage 0 — the CPU mixed-type assembly de-risk)
- a `DiscursiveTurn` orchestrator (the typed-candidate gather + per-candidate spiking appraisal/decision + type-aware render/verify + depth controller) — promoted to a `discuss()` entry point on `BrainConversationalAgent`/`ChatBrain` behind the existing default-OFF flag (Stage 1)
- the `/api/brain-chat` typed-proposition schema fields + the certain-vs-flagged-distinct console rendering (Stage 1, additive to the prior console scoping's Stage D)
