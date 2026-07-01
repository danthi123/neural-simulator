# Fluid-conversation gap assessment — current conversational stack vs "talk to it like an LLM" (2026-07-01)

**Read-only assessment (no code/sim/GPU edit).** Owner's NEW MAIN PRIORITY (2026-07-01): talk to the simulated
brain as if it's an LLM — real, fluid back-and-forth about almost any topic, reasonable understanding, responses
grounded in the brain's OWN knowledge + experiences AND the live conversation, while STILL growing through the
conversation; rely on the actual transformer AS LITTLE AS POSSIBLE; biologization / single spiking substrate / ONE
brain / no-cheats all bind.

This doc owns the honest CURRENT-STATE-vs-GOAL gap. A sibling subagent concurrently surveys the biology/mechanisms;
this one does NOT survey literature. Every claim is cited to code (`file:line`) or a findings doc, and flags
demo-only / numpy-oracle-only / single-vs-multi-seed status.

Terms used once: **SVO** = subject-verb-object triple (agent, action, patient). **Composer** = the FHRR
role-filler binder (`RFPhasorComposer`) that stores/recalls facts. **FHRR** = Fourier Holographic Reduced
Representation — a vector-symbolic algebra with an exact-inverse unbind (the composer's binding scheme).
**No-confab moat** = the abstain-when-no-fact-matches guard that prevents fabricated assertions. **DiscursiveTurn**
= the current top-level engage-and-discuss turn orchestrator. **Grounded codes** = per-word phasor vectors the
stream cortex LEARNED from a corpus. **Faculty** = an optional off-bridge spiking Qwen2.5-0.5B used only for wording.

---

## 1. CURRENT CAPABILITY (what a conversation with the brain looks like TODAY)

**The interaction surface.** The production chat is a typed REPL/console:
`research/runners/first_chat_console.py` (`FirstChatConsole`), run `SIM_BACKEND=numpy python -m
research.runners.first_chat_console` (`first_chat_console.py:46-52`). You type a message; the `DiscursiveTurn`
orchestrator (`_discursive_turn_stage0_derisk.py`) classifies intent, gathers a TYPED candidate pool, decides which
candidates to speak (a spiking accumulator), renders each, VERIFIES each, and emits a moat-safe paragraph. There is
also a `brain_chat_tui.py` (a richer terminal UI) and the webapp Interact tab (per CLAUDE.md B3), all wrapping the
same components.

**The scale (concrete numbers).**
- **Vocabulary: up to 3,000 concepts.** The first-chat brain is `brain1454_w7000_seed42.npz` — **1,454 concepts**,
  128-dim stream-learned phasor codes, recall 0.958, moat 0-FA (`first_chat_console.py:11-14`;
  `2026-06-26-first-chat-ready-bar.md`). A **3,000-concept** brain (`brain3000pos_w7000.npz_seed42.npz`) is
  validated behind a `RoutedComposer` (multi-bridge, `--shards N`): recall **1.000**, moat **0 false-accepts**,
  ~127 ms/query single-bridge (`2026-06-26-multibridge-3000-integration-GO.md`, lines 8-73). Facts drawn from
  **2,191 corpus-attested SVO facts** (`_facts3000.json`, 1,013 with unambiguous who/what cues).
- **Facts (what it "knows"): a stored SVO fact-set.** The console stores SVO facts the brain was "told" (built via
  the curriculum's `_make_svo_facts` from the vocab by category — `first_chat_console.py:30-34`). The console demos
  run at **24 facts** (`--n-facts 24`), the validation at up to ~1,013 unambiguous facts. Each fact is a
  structurally-valid recombination; recall + the moat are the ground truth.
- **Turns / dialogue memory:** multi-turn with a persistent spiking working-memory loop
  (`MultiTurnAgent` + `SpikingLoopContextBuffer`, `multi_turn_agent.py:1-11`) that carries **one dominant
  referent** across a turn boundary (so "it" resolves to the held concept). Multi-referent disambiguation is opt-in
  and content-biased (`multi_turn_agent.py:13-31`). There is no long-horizon discourse model beyond this.
- **Topics:** confined to the vocabulary + the corpus-learned association graph. "Almost any topic" is NOT met —
  see gap (b).

**The fluency level (what the words actually are).** Two paths:
- **Default (`--faculty stub`): TEMPLATE rendering.** The word sequence is produced by a template/serial-order
  renderer (`_grounded_lang_p3_derisk.TemplateStubFaculty`, `_inflect`, `_determiner` — `first_chat_console.py:90`),
  optionally the spiking competitive-queuing serial-order engine for SVO word ORDER (`enable_neural_render`,
  CLAUDE.md 2026-06-16). Output reads like: "The dragonfly hums cod.", "the canary eats the bear", "I don't have
  settled facts about bison, but it tends to come up alongside plantain, lynx, and buffalo — I'd be guessing past
  that." (`2026-06-27-tier2-console-integration.md:77-83`). These are grammatical but templated single-clause
  sentences, assembled into a paragraph.
- **Optional (`--faculty llm`): spiking-Qwen fluent rendering.** An off-bridge converted Qwen2.5-0.5B renders a
  GATED, VERIFIED stored fact into fluent prose ("The dog eats meat.", "The fox chased the rabbit through the
  forest." — `2026-06-23-grounded-lang-INTEGRATION-GO.md`). The LLM supplies WORDING ONLY.

**What is genuinely validated (multi-seed GO) vs demo-only vs numpy-oracle-only:**
- **Multi-seed GO (validated):** the core who/what pipeline + moat at 320 concepts (`2026-06-17-consolidated-320-
  production-conversation-GO.md`, 3-seed host + 2-seed neural); multi-hop `query_chain` (3-seed × 3-D,
  `2026-06-17-multihop-query-chain-GO.md`); multi-turn anaphora (3-seed, `2026-06-17-multiturn-anaphora-derisk-
  GO.md`); the OneBrainComposer whole-turn 320-scale (3/3, CLAUDE.md CYCLE 190); the DiscursiveTurn Stage 0
  mixed-type engage-and-discuss turn (**3-seed GO**, 24 facts — `_discursive_turn_stage0_derisk.json` verdict=GO,
  seeds [42,43,44]); the CommunicableTurn generate/decide/learn fusion (Stage A 3-seed GO); generative-replay
  proposer (6-seed), spiking speak-accumulator (3-seed), learned talkativeness (3-seed).
- **Demo-only / de-risk-scale (NOT a full multi-seed production gate):** the 1,454-brain DiscursiveTurn quality
  **rubric** is the pending final check run as a 10-prompt sample (`2026-06-26-first-chat-ready-bar.md`, "pending
  Step 2"); the grounded-Qwen integration is **de-risk scale** (4 grounded + 2 abstain + 1 drift, single-seed
  transcript — `2026-06-23-grounded-lang-INTEGRATION-GO.md`); the Tier-2 console analogy route is **data-limited**
  (a curated 64-item KB, NOT over the brain's corpus codes — `2026-06-27-tier2-console-integration.md:86-95`).
- **numpy-oracle-only vs spiking:** the CPU console runs a **numpy reference** of the spiking architecture plus
  host-authored linguistic structure — "byte-equivalent to what the validated spiking substrate computes, but not
  itself firing neurons in the CPU path, and the grammar is not self-organized"
  (`2026-06-27-conversation-depth-brain-based-audit-and-burndown.md:49-54`). The spiking forms exist and are
  default-on in the GPU one-brain path (`consolidated_320_conversation_demo` defaults `--composer onebrain`); the
  numpy path is the CPU/test oracle.

**One-sentence current state:** you can hold a typed, multi-turn, mixed-type (certain / novel-flagged /
discuss-via-adjacent / phatic) conversation with a brain that knows a few thousand corpus-learned concepts and a
set of SVO facts about them, answers who/what/yes-no/wh questions, does self-cued association chains and (curated)
analogies, discusses-without-an-answer via adjacent facts, and — crucially — never asserts a fabrication; but the
sentences are largely templated single clauses over a fixed vocabulary, not free-form open-domain prose.

---

## 2. THE ROLE OF THE TRANSFORMER TODAY

**Where the transformer (spiking Qwen2.5-0.5B) is used:** WORDING ONLY, and only on the OPTIONAL `--faculty llm`
path. It renders a *single already-gated, already-verified stored fact* into a fluent sentence
(`first_chat_console.py:39-44`; `_grounded_lang_p3_derisk.py`). It is NEVER invoked to free-generate ungrounded
content — the console ABSTAINS instead (`first_chat_console.py:44`). Its output is re-parsed back to an SVO and
REJECTED on content-mismatch (the VERIFY gate, `2026-06-23-grounded-lang-INTEGRATION-GO.md`).

**Where the BRAIN does the work (everything else):**
- **Knowledge / grounding:** the stream cortex learns each word's meaning from the corpus (population-Hebbian
  co-occurrence) → the grounded phasor codes (`first_chat_console.py:16-28`); facts stored/recalled by the spiking
  FHRR composer.
- **Comprehension:** the Hebbian-learned on-bridge `BridgeParser` maps (word-position × voice) → role
  (`brain_conversational_agent.py:28-63`); voice-invariant.
- **Content selection / "what to say":** the brain's proposed SVO (the generative-replay proposer over the learned
  graph) + the per-candidate emit/silence decision read from `cp_firing_states` (the spiking accumulator)
  (`_discursive_turn_stage0_derisk.py:49-56`).
- **Verification (the moat):** the parser re-parses any rendered prose; a certain claim must re-parse to a STORED
  fact or it is DROPPED (`_discursive_turn_stage0_derisk.py:37-47`).

**What breaks if the transformer is removed:** essentially nothing structural — the DEFAULT console is `--faculty
stub` (template renderer, numpy-CPU, no torch — `first_chat_console.py:44`). Removing Qwen loses only the
*optional* fluent surface polish on stored facts; the brain still comprehends, recalls, decides, discusses, and
guards. **The transformer is therefore a detachable fluency skin, not load-bearing.** This is exactly the design
the owner wants (rely on it as little as possible) — but it also means the fluency the LLM provides is NOT yet
matched by any brain-native mechanism (see gap (a)/(e)).

---

## 3. THE GAP TO FLUID LLM-LIKE, RANKED (most-blocking first)

### (a) [MOST BLOCKING] Brain-native FLUENCY / free-form generation — there is no brain mechanism that produces open, multi-clause, contextually-shaped prose
- **Fluid LLM-like needs:** free-form word sequences (multi-clause, connectives, hedges, register-adapted) that the
  BRAIN generates.
- **Brain does now:** template/serial-order rendering of a single SVO clause (`TemplateStubFaculty`; the spiking CQ
  serial-order engine orders ONE SVO frame's words — CLAUDE.md 2026-06-16). Multi-sentence output is an
  ASSEMBLY of independently-rendered single clauses (`RichAnswerComposer` / `DiscursiveTurn` gather-a-set →
  render-each → concatenate), not a generated sequence.
- **The delta:** the only true free-text generator in the stack is the transformer (fluency-only). A brain-native
  free generator is the **generative-sequence frontier** — the ~134K-param BPTT-SNN scale wall + the categorical
  free-generation gap (`project_generative_sequence_frontier`; CLAUDE.md 2026-06-23 "moderate-shift capacity wall
  … needs ~50-200M params"; the 100M scale-up C1 GO is a consolidatable spiking LM but is a next-token LM, not yet
  wired as the conversational generator).
- **Prior NEGATIVE/boundary:** the serial-order render only covers the SVO frame; "embedded-clause render,
  adjective-noun, dialogue replies, and MULTI-FRAME order-learning (real syntax) are bounded follow-ons"
  (CLAUDE.md 2026-06-16). Learned grammar is named a genuine Tier-3 wall
  (`2026-06-27-conversation-thinking-ROADMAP.md:101-108`).

### (b) [MOST BLOCKING] BREADTH / open-domain — the vocabulary and knowledge are a fixed, corpus-sized closed set
- **Fluid LLM-like needs:** talk about almost any topic; handle out-of-vocabulary words gracefully; broad world
  knowledge.
- **Brain does now:** a FIXED vocabulary (1,454; validated to 3,000 with multibridge routing) of stream-learned
  concept codes; knowledge = the stored SVO facts + the corpus association graph. Unknown words route to an honest
  "I don't know X yet" clarification (`2026-06-27-tier2-console-integration.md:64-65`).
- **The delta:** LLM open-domain breadth vs a few-thousand-concept closed world. Growing the vocabulary means a
  bigger corpus + more stream-cortex training + more storage bridges; the owner's own target is "~10K → 30-40K via
  bigger corpus + tail-learning + morphology" (`project_vocab_target_breadth_vs_depth`), explicitly NOT the full
  OED.
- **Prior boundary:** generalization across similar concepts is **substrate-capped** at 1,454 (over-training
  densifies the codes; ~0.05 coherent / ~0.03 Pearson — `2026-06-26-first-chat-ready-bar.md:11`); demoted from a
  gate to a reported floor. So even within the vocab, "relate a new thing to what I know" is thin. Breadth (corpus)
  ≠ depth (meaning/fluency), and the chat problem is DEPTH (`project_vocab_target_breadth_vs_depth`).

### (c) [BLOCKING] COMPREHENSION depth — parsing is confined to a small set of hand-authored frames
- **Fluid LLM-like needs:** parse arbitrary natural input — any clause structure, prepositional arguments,
  tense/aspect, questions of any form.
- **Brain does now:** a voice-invariant SVO parser (the strongest-built part of the stack), embedded clauses,
  multi-frame word order (6/6), Bates-MacWhinney multi-cue robustness (CLAUDE.md; `2026-06-27-conversation-
  thinking-ROADMAP.md:16-18`). Wh-questions parse via a **regex** (`wh_question_parser.py`), not the spiking
  parser (`2026-06-27-conversation-depth-…-burndown.md:16`). Verb-argument structure (go→GOAL-PP) comes from a
  **hand-authored `FRAME_LEXICON` host dict** (`…burndown.md:17-18`).
- **The delta:** the representation is a **skeleton** — "bare, type-keyed SVO triples … all obliques collapsed into
  one slot, discarding the preposition; no verb argument structure, no tense/aspect, no events; types not tokens"
  (`2026-06-27-conversation-thinking-ROADMAP.md:14-32`). Tiers 0-1 add typed verb frames + entity instances (mostly
  WIRING validated parts), but the frames are hand-authored SCAFFOLDS, and LEARNED productive grammar is the
  Tier-3 wall (`…ROADMAP.md:101-106`).
- **Prior state:** the roadmap explicitly diagnoses this as THE root bottleneck ("thinks in skeletons") —
  representation, not parsing-machinery. Tiers 0-2 are largely built/GO as of 2026-06-27 (entity instances, wh as
  filler-gap, tense/aspect, chain-of-thought, transitive-ordinal all have GO findings), but they enrich the
  skeleton; they do not make comprehension open-ended.

### (d) [BLOCKING] DIALOGUE COHERENCE over many turns — context integration is single-referent
- **Fluid LLM-like needs:** track the whole conversation; responses shaped by all prior turns and (per the owner)
  all past conversations; persistent growth/memory.
- **Brain does now:** a persistent spiking WM loop holding **one dominant referent** across a turn boundary
  (`multi_turn_agent.py:1-11`); opt-in content-biased disambiguation among ≥2 referents
  (`multi_turn_agent.py:13-31`); a discourse state that records the topic + any hypothesis referent
  (`_discursive_turn_stage0_derisk.py:36`). Persistent cross-session growth exists via the develop loop + lineage
  (below).
- **The delta:** no rich multi-turn discourse model (no coreference chains beyond one referent, no topic stack, no
  "influenced by ALL past conversations" beyond what got consolidated into codes/facts). Common-ground/audience
  design is a minimal slice (`2026-06-27-tier2.4-common-ground-GO.md`).
- **Prior NEGATIVE:** multi-REFERENT disambiguation needed WTA biased-competition (recency and salience-boost both
  failed — `2026-06-17-multireferent-disambiguation-NEGATIVE.md`); it is now de-risked but opt-in and narrow. Full
  theory-of-mind / recursive agent-modeling is a named Tier-3 wall (`…ROADMAP.md:102-103`).

### (e) [BLOCKING] GENERATION MECHANISM — what actually produces the word sequence
- **Fluid LLM-like needs:** ONE generator that emits the full response text.
- **Brain does now:** a PIPELINE, not a generator — gather typed propositions → per-candidate spiking
  emit/silence decision → render each (template or, optionally, Qwen) → VERIFY → order → concatenate
  (`_discursive_turn_stage0_derisk.py:20-36`). The actual word emission is either a template or the transformer.
- **The delta:** this is the flip side of (a). The pipeline is excellent for the moat (structural
  never-fabricate) but it is a "retrieve-a-set → render-each" shape, which is why even the owner notes the UX can
  feel RAG-like however spiking the internals (`project_communicable_brain_not_rag`). A brain-native sequence
  generator (the BPTT-SNN / generative frontier) would replace both the template and the transformer.

### (f) [PARTLY MET] GROUNDING responses in the brain's OWN knowledge + experiences
- **Fluid LLM-like needs:** answers reflect what the brain actually knows/experienced, not a generic prior.
- **Brain does now:** this is the STRONGEST axis. Responses are grounded in stream-learned codes + stored facts +
  the moat, and the DiscursiveTurn discusses via adjacent grounded facts when there's no direct answer
  (`_discursive_turn_stage0_derisk.py:26-27`). Perception→memory and navigate-to-compose ground *experienced*
  objects into facts (CLAUDE.md 2026-06-16). A self-knowledge arc (`_self_knowledge_*`, `rich_answer_composer.py`)
  lets the brain answer about itself.
- **The delta:** small — grounding is real. The limit is that "own experiences" is currently navigation/perception
  episodes + told facts; a rich autobiographical/episodic memory feeding open conversation is thin, and grounding
  is only as broad as (b).

### (g) [PARTLY MET but SHALLOW] GROWTH through conversation — learning new content/words live
- **Fluid LLM-like needs:** learn from the conversation itself (new facts, new words, adapt style) persistently.
- **Brain does now:** (i) **learned talkativeness** — "tell me more" raises a per-context speak-value Q via
  three-factor DA learning; "stop" lowers it (`_communicable_turn_stageA_derisk.py:28-31`, Stage A 3-seed GO;
  DA-lesion abolishes it). (ii) **the develop loop** — a brain develops over simulated days, vocab 6→24, facts
  2→11, zero forgetting, moat 0-FA daily, persists + resumes (CLAUDE.md 2026-06-24 week1 capstone;
  `_longitudinal_develop_loop_gpu.py`). (iii) runtime teach of new facts (the composer store).
- **The delta:** growth is validated at SMALL scale (vocab 6→24 over a week; 24-fact console). Learning a genuinely
  NEW word's meaning mid-conversation (not just a new fact over known words) needs stream-cortex exposure, not one
  utterance. Style adaptation = the learned speak-value only. So "grows like an LLM in-context + persistently"
  is met in mechanism (the loop closes, DA-provable) but shallow in reach.

---

## 4. WHAT'S REUSABLE (load-bearing foundations vs dead-ends)

**Load-bearing foundations (build the goal ON these):**
- **The stream cortex** (`_phaseB_onbridge_stream_conversation_derisk.py`; the 1,454/3,000 code sets) — the
  learn-word-meaning-from-corpus engine; the breadth + grounding substrate. THE knowledge foundation.
- **The FHRR composer + no-confab moat** (`rf_phasor_composer.py`, `one_brain_composer.py`) — store/recall/abstain;
  the "minus hallucinations" guarantee; spiking on-bridge (onebrain) + numpy oracle. THE reliability foundation.
- **The DiscursiveTurn + CommunicableTurn shells** (`_discursive_turn_stage0_derisk.py`,
  `_communicable_turn_stageA_derisk.py`) — the engage-and-discuss orchestration + type-aware moat + learned
  talkativeness. THE conversational-loop foundation (already the "communicable brain" the owner described).
- **The develop loop + lineage** (`_longitudinal_develop_loop_gpu.py`, `sim/lineage.py`) — persistent
  cross-session growth. THE growth foundation.
- **The on-bridge parser + Tier-0/1 typed frames + entity instances** (`brain_conversational_agent.py`;
  Tier-0/1 GO findings) — comprehension + the richer skeleton.
- **The generative BPTT-SNN + 100M spiking LM** (`sim/bptt_snn*.py`; `2026-06-30-100M-C2-scaleup-C1-GO`) — the
  ONLY brain-native free-sequence generator path; the intended replacement for the template + transformer
  (gaps a/e). **Currently a next-token LM, not wired as the conversational generator — this is the key unbuilt
  bridge.**
- **The grounded Qwen faculty + VERIFY** (`_grounded_lang_integration_derisk.py`) — a fluency SKIN + the
  hallucination-catch proof; reusable as a scaffold/oracle while the brain-native generator is built (owner wants
  it minimized, not necessarily deleted day one).
- **Serial-order render, generative-replay proposer, spiking speak-accumulator, biased-competition buffer,
  mode-unification/theta-gamma** — validated cognitive primitives the DiscursiveTurn already composes.

**Dead-ends / retracted (do NOT build on):**
- The **STDP-pathway compose** and **concept-concept engram "semantic conversation"** results (v16 compose,
  transitive-90%, pool-firing-65%) — RETRACTED as architecture-mismatch/permuted-control artifacts (CLAUDE.md
  2026-05-14 retractions).
- **regime-B analogy over corpus codes** — documented NO-GO (`2026-06-27-tier2.1-analogy-NEGATIVE.md`); only the
  curated factored-relation KB works.
- **Decorrelating the correlated codes for a learned bind on point neurons** — four mechanisms failed
  (Mikulasch-Priesemann point-neuron limit); the exact-inverse FHRR algebra remains the binder (CLAUDE.md
  2026-06-11 fork). Not a path to fluency.

---

## 5. HONEST BOTTOM LINE

**How far:** the brain has a genuinely working, biology-grounded, mostly-spiking CONVERSATIONAL LOOP that engages,
discusses, answers, reasons in small ways, grows, and never fabricates — the "communicable brain, minus
hallucinations, with persistent growth" the owner specified is BUILT in skeleton form (DiscursiveTurn 3-seed GO on
24 facts; console on a 1,454-concept brain; up to 3,000 validated). But it is NOT yet fluid-LLM-like: it speaks in
templated single clauses over a few-thousand-concept closed vocabulary, comprehends via hand-authored frames, and
its only free-text generator is a detachable transformer skin. The distance is large on FLUENCY and BREADTH,
moderate on COMPREHENSION-DEPTH and DIALOGUE-COHERENCE, and small on GROUNDING and (mechanistically) GROWTH.

**The 3-4 genuinely-hardest gaps (the most research / new-mechanism work):**
1. **Brain-native free-form generation (gaps a + e).** Replacing the template/transformer with a spiking sequence
   generator that emits open, multi-clause, context-shaped prose. This is the generative-sequence frontier — the
   BPTT-SNN scale wall + wiring a next-token spiking LM into the conversational loop while keeping the moat. Highest
   variance, deepest.
2. **Open-domain breadth + depth (gap b).** Scaling vocabulary/knowledge from a few thousand to tens of thousands
   of concepts with meaning, plus graceful out-of-vocabulary handling — bigger corpus + tail-learning + morphology,
   against a substrate-capped generalization ceiling.
3. **Learned productive grammar / open comprehension (gap c).** Moving from hand-authored typed frames to grammar
   the brain self-organizes from the corpus (the named Tier-3 wall; the dendritic-substrate + BPTT-SNN frontier) —
   so it parses and generates arbitrary structure, not a fixed frame set.
4. **Rich multi-turn discourse + ToM (gap d).** Coreference chains, topic tracking, audience design, and the
   recursive-agent-modeling apex — beyond the current single-referent WM + minimal common-ground slice; the full
   version is a genuine point-neuron wall.

The no-confab moat and the single-spiking-substrate / one-brain framing are preserved throughout the existing stack
and must remain the invariants as these gaps are closed (fluency/breadth added WITHOUT letting the transformer do
the thinking or reintroducing fabrication).

---
*Companion (biology/mechanisms) survey produced concurrently by a sibling subagent. This doc is the current-state
gap assessment only.*
