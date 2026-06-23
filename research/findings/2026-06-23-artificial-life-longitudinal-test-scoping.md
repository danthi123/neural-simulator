# Artificial-life longitudinal DEVELOPMENT test — deep-research scoping: simulate extended (weeks→years) daily conversation, watch the brain GROW, then let a human talk to the developed brain

**Date:** 2026-06-23
**Type:** READ-ONLY deep-research + code/findings scoping (the standing deep-research-FIRST gate at a new direction). NO `sim/` edit, NO build, NO experiment, NO GPU beyond reading. Single deliverable = this doc. Every load-bearing project claim re-verified against the repo (file:line / finding).
**Owner directive (2026-06-23):** give the sim a PROPER test = simulate EXTENDED INTERACTION (weeks/months/years of DAILY conversation) to see how the brain GROWS/DEVELOPS over simulated time; not just text-gen but actual INTERACTIVE, MULTI-STEP conversation; then a HUMAN interacts with the "developed" brain to see how it responds.
**North star this serves:** artificial life with a proper brain analogue + biology-translatable insights; capabilities are instrumental; honest negatives under strict biology ARE the deliverable (`project_actual_goal_artificial_life_brain_analogue`). This is the **Tier-3 capstone** of the owner-accepted roadmap (`project_post_conversational_roadmap_tiers`: Tier-1 conv loose ends → Tier-2 TRUE ONE BRAIN → **Tier-3 artificial-life capstone** → Tier-4 deep walls).
**LLM constraint (load-bearing):** lean on the LLM as LITTLE as possible. Two distinct LLM roles, scoped both ways: (i) **LLM-as-OUTPUT-FACULTY** (fluent generation) — MINIMIZE; prefer the brain's OWN generation; (ii) **LLM-as-CONVERSATION-PARTNER** (the offline interlocutor that authors the daily stream) — legitimate per the brain-based-only standard (the environment + body are allowed host code).
**Relationship to the prior Tier-3 scoping:** `2026-06-20-tier3-artificial-life-capstone-deep-research.md` scoped the **homeostatic/survival** living loop (hunger drive → eat → persist). This doc scopes a **DIFFERENT, complementary longitudinal axis the owner has now named: linguistic/cognitive DEVELOPMENT over simulated time** (vocab growth + knowledge accumulation + consolidation + conversational competence). They are two faces of the same "persistent living agent" — survival-over-time and development-over-time. This doc is the development face.

---

## 0. One-paragraph answer (the rest is the evidence)

**Every COMPONENT of the longitudinal development test is already built and de-risked IN PIECES — the gap is purely ASSEMBLY into one outer `day` loop plus one honest scale caveat.** The four developmental sub-systems all exist and are multi-seed GO standalone: (1) **learn-word-meanings-from-the-conversation-stream** (the on-bridge Hebbian PPMI stream cortex, `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` — a spiking brain that *hears the corpus word-by-word* and grows its concept codes, scaled to 320 concepts multi-seed); (2) **store/recall/abstain/reason/multi-turn conversation** on those learned codes (`MultiTurnAgent` + `BrainConversationalAgent` + `consolidated_320_conversation_demo`, the no-confab moat intact); (3) **consolidate without catastrophic forgetting** (Phase-1.3 SWR sleep replay `consolidation_trainer.py` + the generative grow-no-forget loop `2026-06-23-generative-loop-DEMONSTRATED.md`); (4) **persist + grow across sessions** (`BridgeLineage` atomic save/load `sim/lineage.py`, `TierPromoter` auto-growth `sim/auto_growth.py`). There is even a precedent *continuous outer loop with lineage persistence* — `persistent_living_loop_derisk.py` — but it is the SURVIVAL loop, not the development loop. **What does NOT exist is the assembly: a `for day in range(N): converse(day) → consolidate(sleep) → grow(if-mastered) → persist → next_day` scheduler over the conversational stack, plus development metrics tracked across simulated days, plus a human-REPL pointed at the developed lineage.** The cheapest-first path is a **compressed "1 week" horizon** (e.g. 7 "days," a small developmentally-graded daily stream) run end-to-end on the EXISTING conversational substrate to (a) prove the loop closes and (b) clock the per-day wall-clock → then scale the horizon. The **LLM-minimal path is fully viable**: the conversation partner can be a scripted/curriculum generator OR Claude-authored-offline curricula (the validated P2 "textbook author" pattern), and the brain's OWN output (the composer's neural word-ordering renderer + the generative-loop generator) can carry generation — the fluent LLM faculty is an OPTIONAL output upgrade, not a requirement. **The one honest scale caveat:** the grow-no-forget loop is GO at toy scale for a *distinct* shift but hits a model-capacity wall for *moderate/in-band* shifts (`2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md`) — and incremental developmental vocab growth IS in-band — so "the conversational brain develops" (vocab + facts via the stream cortex + lineage, which is NOT a generator-forgetting problem) is reachable NOW, while "the *generative* faculty grows its distribution without forgetting" is the part that needs the (likely-local, 50–200M) scale-up. The recommended cheapest-first de-risk is the **compressed-week development loop on the conversational stack (stream cortex + agent + consolidation + lineage)** — decoupled from the generator-capacity wall, exactly as the survival loop was decoupled from the dendrite wall.

---

## 1. Component inventory — EXISTS-de-risked vs needs-ASSEMBLING/BUILDING (file:line / finding, trust-but-verified)

I read the actual code and findings (not the CLAUDE.md summaries) and verified every load-bearing claim. The headline: **far more is built than a "scope the capstone" prompt assumes** — this is an assembly job, not a research-frontier job, for the development (vs generation) axis.

### 1A. THE DEVELOPMENT ENGINE — learn from the stream, accumulate knowledge

| Component | Status | Evidence (verified) |
|---|---|---|
| **Stream cortex: learn word meanings by HEARING the conversation** (online Hebbian co-occurrence; the brain's concept codes GROW as it listens, NO preprocessing, NO global matrix) | **EXISTS — multi-seed GO, scaled to 320** | `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`: rate-Hebbian on-bridge `corr(M,C)` +0.686 6-seed; population code lifts read-out to 100–108% of host; the FULL conversation on stream-learned codes recall 1.00 / moat 0.96–1.00 (3 seeds); **320-concept stream-scaling VALIDATED** (`stream_taxonomy_320.py`, 150K windows ~96 min, 9920-neuron bridge, recall 1.00 / moat 1.00 multi-seed). Runner `research/runners/_phaseB_stdp_cooccurrence_derisk.py`. **This is the developmental HEART: a brain that learns by listening.** |
| **Knowledge accumulation (facts) via the parser→composer** | **EXISTS — production** | `consolidated_320_conversation_demo.py` (default `--composer onebrain`); `BrainConversationalAgent.hear()` stores SVO facts; `2026-06-18-onebrain-320-scale-production-GO.md` (320 concepts, recall 1.00, 0 false-accepts). |
| **Persistent multi-fact store across the brain's own synapses** | **EXISTS** | `one_brain_composer.py` (K=32 persistent fact-store in complex synapses, register-reset-safe). |
| **Re-encode a Claude-authored curriculum + recall + abstain (the knowledge-teacher loop)** | **EXISTS — 6-seed GO** | `2026-06-23-grounded-lang-P2-GO.md`: a ~30-fact structured curriculum ingested through the validated parser→composer, recall 1.000 + moat 0-FA, 6 seeds; `_grounded_lang_p2_derisk.py`. **This is the proof the "textbook-author" conversation-partner pattern works.** |

### 1B. INTERACTIVE MULTI-TURN CONVERSATION

| Component | Status | Evidence |
|---|---|---|
| **Multi-turn dialogue with anaphora (a turn-2 pronoun resolves to a held referent)** | **EXISTS — production, multi-seed GO** | `research/runners/multi_turn_agent.py` `MultiTurnAgent` (a persistent `SpikingLoopContextBuffer` holds discourse referents across turns); `2026-06-17-multiturn-anaphora-derisk-GO.md`; `multi_turn_conversation_demo.py`. |
| **Multi-referent disambiguation (which of ≥2 held referents a bare pronoun binds)** | **EXISTS — opt-in, de-risk GO** | `MultiTurnAgent(enable_biased_competition=True)` (WTA + content bias); `2026-06-19-multireferent-biased-competition-derisk.md`. Default OFF = byte-identical. |
| **Multi-hop reasoning (role-structured pointer-chase, moat at every hop)** | **EXISTS — production** | `RFPhasorComposer.query_chain` / `BrainConversationalAgent.reason_chain`; `2026-06-17-multihop-query-chain-GO.md` (2-hop 1.00, holds to 4 hops). |
| **The full conversational core (parse/compose/recall/abstain/negate/generate/dialogue-plan)** | **EXISTS — comprehensively complete** | `brain_conversational_agent.py` + `rf_phasor_composer.py` + `one_brain_composer.py`; the no-confab moat throughout. |
| **Sustain OPEN-ENDED multi-turn over a LONG horizon** | **NEEDS BUILDING (small)** | The validated multi-turn is SHORT (2–4 turns, scripted referents). A long-horizon dialogue needs (i) a turn-budget loop, (ii) discourse-buffer eviction/refresh over many turns, (iii) handling the agent abstaining gracefully when off-curriculum. None is a research frontier — it is a loop + buffer-management policy over the existing `MultiTurnAgent`. |

### 1C. CONTINUAL LEARNING / NO-CATASTROPHIC-FORGETTING (the "development sticks")

| Component | Status | Evidence |
|---|---|---|
| **CLS hippocampus→cortex consolidation (SWR sleep replay)** | **EXISTS — validated** | `research/runners/consolidation_trainer.py` (`run_concept_replay_phase:43`, `run_swr_replay_phase:154`, awake/sleep gate `run_consolidation_training:206`); Phase-1.3 CONFIRMED (hippo-OFF retention 94%, 3/3 strict anti-cheat multi-seed). **NOTE the file header (`:24-31`) flags the trainer as SKELETON/UNTESTED-on-GPU for some eval modes — the replay PHASE functions are validated, the full awake/sleep orchestration on a production lineage is the assembly piece.** |
| **No-catastrophic-forgetting of vocab bindings (Phase-1.4 BRANCH A)** | **EXISTS — 5/6 ≥80% retention** | `continual_forgetting_eval.py`; retention via shared-pool reinforcement + the plasticity gate-freeze (`set_plasticity_gate`, `bridge.py:3153-3177`). |
| **The generative grow-no-forget LOOP (train→generate→grow→confirm-no-forget)** | **EXISTS — DEMONSTRATED end-to-end at toy scale** | `2026-06-23-generative-loop-DEMONSTRATED.md`: with the corrected FT-LR (1e-5), the consolidated spiking generator learns a NEW distribution (new-ppl −87%) WHILE retaining the old (88% with self-replay) and the no-replay control catastrophically forgets (39%, contrast 2.25×). `_genseq_C2_demo_design_derisk.py`. **All 3 bars cleared.** |
| **⚠️ The grow-no-forget SCALE CAVEAT (load-bearing for the development axis)** | **HONEST BOUNDARY** | `2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md`: the loop is GO for a *distinct* shift (41× Shakespeare) but the 3.4M toy CANNOT hold both distributions for a *moderate/in-band* shift (best 53.9% retention; no-replay forgets only 1.16×). **Incremental developmental vocab is in-band** → the *generator's* distribution-growth needs the 50–200M scale-up (likely LOCAL, <24 GB). The C1 consolidation + the C2 mechanism are both validated; only the *generative-faculty* scale remains. See §6. |

### 1D. PERSISTENCE + STRUCTURAL GROWTH ACROSS SESSIONS (the brain "lives between days")

| Component | Status | Evidence |
|---|---|---|
| **`BridgeLineage` — persistent continuous-learning state across sessions** | **EXISTS — production** | `sim/lineage.py`: atomic save (`.new`+fsync+rename `:190`), `load` (`:238`), history snapshots, `metadata.json` (vocab, tier, `cumulative_training_events`, `accuracy_history`, `growth_events`), `_growth_log.md` diary. **The metadata fields are EXACTLY a development log** (vocab list, cumulative events, accuracy-over-time, growth events with timestamps). |
| **`TierPromoter` — the brain GROWS (scales) as it masters a tier** | **EXISTS — Phase A** | `sim/auto_growth.py`: `TierLadder` (4→8→…→320 vocab, per-tier arch) + `TierPromoter` (monitors eval accuracy, promotes after N consecutive passes ≥0.90, records growth events to a lineage, weight-transfer via `bridge.set_pathway_weights`). |
| **End-to-end auto-grow loop wired to a real bridge** | **PARTIAL (orchestration EXISTS; real train/transfer hooks are stubs)** | `research/runners/auto_grow_chat.py`: the TierPromoter orchestration loop runs end-to-end, BUT with `synthetic_train_fn` / mock `_MockBridge` (`:38-55`) — the real `bio_three_factor.run_three_factor` + actual weight transfer are flagged as the next-session wiring (`:18-24`). **So "the brain grows in size as it develops" is a wiring job, not a frontier.** |
| **A continuous outer `live()` loop with lineage persistence (PRECEDENT)** | **EXISTS — but for SURVIVAL, not development** | `research/runners/persistent_living_loop_derisk.py`: a continuous `live()` loop where an interoceptive drive persists across a reset via `BridgeLineage`, with the full anti-cheat battery (drive-lesion, yoked-random, no-persistence). **This is the structural TEMPLATE for the development loop** — same outer-loop + lineage-persistence shape, different inner content (converse+consolidate+grow instead of perceive+act+eat). |

### 1E. THE OUTPUT FACULTY (generation — the LLM-minimal question)

| Component | Status | Evidence |
|---|---|---|
| **The brain's OWN word-ordering generation (neural serial-order)** | **EXISTS — GPU GATE PASS** | `research/runners/neural_serial_order_renderer.py`; `BrainConversationalAgent(enable_neural_render=True)`: `describe()`'s word order produced by a spiking competitive-queuing read-out (Grossberg/Bullock-Rhodes), NOT a host f-string; `2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`. **This is the LLM-MINIMAL output path: the brain orders its own SVO words.** Honest scope: the SVO frame is neural; embedded-clause/adjective ordering + multi-frame syntax are bounded follow-ons. |
| **The generative-sequence loop's generator (a spiking LM that free-generates)** | **EXISTS — consolidated on the bridge** | `2026-06-23-generative-loop-DEMONSTRATED.md` / the C1 work (Gen-F fully spiking on the bridge, generating byte-identical). **The brain-internal free-generation faculty (toy scale).** |
| **A FLUENT generative LLM faculty (the OPTIONAL fluency upgrade)** | **EXISTS — de-risk scale, off-bridge** | `2026-06-23-grounded-lang-INTEGRATION-GO.md`: a real spiking Qwen2.5-0.5B (T=16) renders the brain's gated facts fluently, GATED+VERIFIED, no-confab moat preserved EVEN WITH a real generative LLM in the loop; `2026-06-23-grounded-lang-SCALED-GO.md` (~67 facts, 3 seeds). **This is the WITH-LLM output path** — but it is PyTorch off the bridge (bridge co-residence = a later consolidation), and it is OPTIONAL (the brain's own renderer already speaks). |

### 1F. THE HUMAN-INTERACTION SURFACE

| Component | Status | Evidence |
|---|---|---|
| **An interactive REPL to talk to a trained brain lineage** | **EXISTS — but narrow (direction-word vocab)** | `research/runners/chat_repl.py`: loads a `BridgeLineage`, user types a word, the sim activates a motor pool + responds; modes tier1/synonym. **It is the REPL skeleton + the lineage-load pattern**, but wired to the 4/8-word direction architecture, NOT to the 320-concept conversational agent / `MultiTurnAgent`. |
| **A REPL pointed at the full conversational agent** | **NEEDS BUILDING (small)** | The `MultiTurnAgent` / `consolidated_320_conversation_demo` are the conversational engine; a thin REPL wrapper (read a line → `agent.hear`/`agent.what_does`/`agent.elaborate` → render) that loads a *developed* lineage is the assembly. The pieces (lineage-load, agent, neural render) all exist. |

### 1G. The honest one-line state

**The development test is an ASSEMBLY of validated subsystems, not a research frontier — with one scoped scale caveat on the *generative* faculty.** The developmental engine (stream cortex), the conversation (multi-turn agent + moat), the consolidation (SWR replay + grow-no-forget), the persistence+growth (lineage + TierPromoter), and BOTH output paths (the brain's own renderer + the optional fluent LLM) all exist. **What is missing is the outer `day` loop that strings them together, the development metrics tracked across simulated days, and a REPL pointed at the developed conversational lineage.** The survival-loop template (`persistent_living_loop_derisk.py`) is the structural blueprint.

---

## 2. The longitudinal-loop architecture (the thing to assemble)

**Name it precisely:** *A continuous `develop(N_days)` outer loop on ONE persistent `BridgeLineage` in which each simulated "day" (a) the brain HEARS a developmentally-graded daily conversation stream (learning word meanings via the stream cortex + storing facts via the composer + holding multi-turn context), (b) at "night" CONSOLIDATES the day's learning via SWR sleep replay so it sticks without catastrophic forgetting, (c) periodically GROWS its capacity (TierPromoter) when it has mastered its current tier, (d) PERSISTS its full developed state (codes, facts, vocab, tier, the development log) via the lineage so the next day resumes the same brain — with development METRICS (vocab size, facts known, recall accuracy, retention, brain size) logged per day so the trajectory is visible; then (e) a HUMAN opens a REPL on the developed lineage and converses with the "grown-up" brain.*

The minimal `day` step (the inner loop):

```
develop(lineage, partner, N_days):
  brain = lineage.load()                          # resume the developing brain (or init at "age 0")
  for day in range(N_days):
      # --- WAKE: lived conversation (the development) ---
      stream = partner.day_stream(day, brain_state)    # developmentally-graded (simple at day 0 -> richer)
      for turn in stream:                              # interactive multi-turn
          brain.hear(turn)                             # stream cortex learns codes + composer stores facts
          if turn.is_question: reply = brain.answer(turn)   # the brain answers (own renderer or LLM faculty)
          # (optional) partner reacts to reply -> genuinely interactive, not just a fixed script
      # --- SLEEP: consolidation (the development STICKS) ---
      run_consolidation(brain, the_day's_engram_tags)  # SWR replay; no-catastrophic-forgetting
      # --- GROWTH: the brain scales as it masters a tier ---
      if mastered(brain): brain = tier_promoter.maybe_promote(brain)   # auto-growth
      # --- METRICS + PERSIST (the brain lives between days) ---
      log_development_metrics(lineage, day, brain)     # vocab, facts, recall, retention, size
      lineage.save(brain)                              # atomic; next day resumes THIS brain
  return lineage

# --- THEN: the human meets the developed brain ---
human_repl(lineage)   # load the grown-up lineage; a person converses with it
```

**Mapping to existing assets (each line is reuse-by-import):**
- `lineage.load/save` → `sim/lineage.py:238`/`:190` (atomic). The development log → the lineage's `accuracy_history`/`growth_events`/`cumulative_training_events` (`:76-77`), already designed for exactly this.
- `brain.hear` (stream-learn + store) → the stream-cortex Hebbian path (`_phaseB_stdp_cooccurrence_derisk.py`) + `BrainConversationalAgent.hear`.
- multi-turn `brain.answer` → `MultiTurnAgent` (anaphora) + `reason_chain` (multi-hop).
- `run_consolidation` → `consolidation_trainer.run_consolidation_training` (awake/sleep + SWR replay).
- `tier_promoter.maybe_promote` → `auto_growth.TierPromoter` (real train/transfer hooks wired in — the `auto_grow_chat.py` stub becomes real).
- `human_repl` → `chat_repl.py` rewired to the conversational agent.
- The outer-loop + persistence SHAPE → templated on `persistent_living_loop_derisk.py`.

**No `sim/` edit is required for the development axis** (it is reuse-by-import over already-shipped subsystems — exactly the pattern the survival loop and the grounded-lang arc followed). The one place a `sim/` touch *might* surface is the auto-growth weight-transfer at a tier boundary if `set_pathway_weights` doesn't cover a needed pathway shape — but that is shipped and unit-tested for the vocab tiers, so the prior is no-edit.

---

## 3. The LLM-minimal vs with-LLM paths (the owner's load-bearing constraint)

The owner wants to lean on the LLM **as little as possible**, distinguishing the two roles. Both roles have an LLM-minimal path AND a with-LLM path; the loop is designed so the LLM is OPTIONAL at every point.

### Role (i): the OUTPUT FACULTY (how the developed brain SPEAKS)

| Path | Mechanism | Status | When to use |
|---|---|---|---|
| **LLM-MINIMAL (preferred)** | The brain's OWN neural word-ordering: `BrainConversationalAgent(enable_neural_render=True)` (spiking competitive-queuing serial-order read-out) for SVO replies; the generative-loop generator (Gen-F, spiking-on-bridge) for free-form continuation. | **EXISTS** (`neural_serial_order_renderer.py`; `2026-06-16-…serial-order-…GO.md`; `2026-06-23-generative-loop-DEMONSTRATED.md`) | The DEFAULT for the development loop. The brain speaks its own (terse, SVO-shaped) sentences. Honest scope: terse + SVO-frame-bound; not free-flowing prose. |
| **WITH-LLM (optional fluency upgrade)** | The spiking Qwen2.5-0.5B faculty renders the brain's GATED facts, re-parse-VERIFIED (hallucination caught by construction). | **EXISTS, off-bridge** (`2026-06-23-grounded-lang-INTEGRATION-GO.md`) | ONLY when fluent prose is wanted in the human-REPL phase (so a person reads natural sentences). The brain still supplies + verifies all content; the LLM supplies only surface fluency. PyTorch off the bridge. |

**Recommendation:** run the development loop with the **LLM-minimal output** (the brain's own renderer) — it is sufficient to MEASURE development (recall, facts, vocab) and keeps the loop fully brain-based + local + cheap. Offer the **fluent LLM faculty as an opt-in for the human-REPL phase only**, so the human reads natural sentences when they meet the grown-up brain. This honors "minimize the output LLM" while keeping a good human-facing demo available.

### Role (ii): the CONVERSATION PARTNER (who generates the daily stream = the ENVIRONMENT/teacher)

Per the brain-based-only standard, the interlocutor is the ENVIRONMENT — host code (even an LLM) is legitimate here. Three options, cheapest-first:

| Path | Mechanism | Status | Trade-off |
|---|---|---|---|
| **LLM-MINIMAL #1 — scripted/curriculum generator** | A deterministic developmentally-graded stream generator (a templated grammar over a growing vocab: simple SVO at day 0 → attributes → clauses → multi-turn Q&A as the brain develops). | **NEEDS BUILDING (small)** — the `text_train_curriculum.py` + `stream_taxonomy_320.py` patterns are the seed. | Fully controllable, reproducible, zero LLM, anti-cheat-clean (you know exactly what was taught). Less "natural," but PERFECT for a rigorous development metric (you can measure recall against the known curriculum). |
| **LLM-MINIMAL #2 — Claude-authored OFFLINE curricula (the "textbook author")** | Claude authors a graded multi-day curriculum OFFLINE (a "syllabus"); the brain ingests it at runtime with ZERO live LLM calls. | **EXISTS — 6-seed GO** (`2026-06-23-grounded-lang-P2-GO.md` — this exact pattern, P2≠Path-3, validated) | The rich model is a "textbook author," not the student's brain — the standalone stance is preserved. Richer than #1, still zero runtime LLM. **The recommended partner for the cheap-first.** |
| **WITH-LLM — live LLM interlocutor** | An offline LLM (local small model or Claude) plays "the human," generating each turn LIVE in response to the brain's replies → a genuinely interactive dialogue. | Reuses the grounded-lang LLM plumbing (the Qwen faculty / an API call). | The most "alive"/interactive, but a live LLM in the loop = the most LLM-dependence + cost + non-reproducibility. Reserve for the FINAL human-facing demo or an "adaptive teacher" stretch. |

**What a sustained developmentally-graded daily stream needs (the developmental grading):** simple at "age 0" → richer as the brain develops. Concretely: day-0 stream = a few high-frequency SVO facts about a handful of concepts (matching the stream cortex's validated low-tier); each subsequent day adds vocabulary (new concepts the cortex must learn from co-occurrence), introduces attributes → embedded clauses → multi-hop chains → multi-turn anaphora as the brain's tier promotes. The TierLadder (`auto_growth.py:43`, 4→8→…→320) IS the grading schedule — the partner's vocab budget per day tracks the brain's current tier. This is the curriculum-learning / developmental-trajectory shape (simple→complex), grounded in the project's own validated tier ladder.

**Recommendation:** the cheap-first uses **partner LLM-MINIMAL #2 (Claude-authored offline curriculum)** — validated, zero runtime LLM, reproducible — with the **scripted generator (#1) as the anti-cheat baseline** and the **live-LLM interlocutor (with-LLM) reserved for the final human-facing demo**.

---

## 4. Development metrics over simulated time + the anti-cheats

The whole point is to MEASURE "development." The lineage already records most of these (`accuracy_history`, `growth_events`, `cumulative_training_events`); the loop logs a per-day datapoint for each.

### The development metrics (logged per simulated day)

| Metric | What it measures | Source / how |
|---|---|---|
| **Vocabulary growth** | concepts the stream cortex has learned (codes with a stable familiarity signal) | count of concepts above the learnability floor; `corr(M,C)` trend; lineage `vocab` list length over days. |
| **Knowledge accumulation** | facts the composer holds | `composer.count_facts()` (`one_brain_composer.py`) over days; facts-recalled-correctly count. |
| **Conversational competence** | recall accuracy + reasoning + multi-turn over days | who/what recall %, 2-hop chain %, anaphora-resolution % on a held-out probe set, tracked per day (rising = developing). |
| **Memory consolidation / no-catastrophic-forgetting** | does day-N learning survive day-(N+k)? | the RETENTION metric from the grow-no-forget loop (`2026-06-23-generative-loop-DEMONSTRATED.md`) + Phase-1.4 `continual_forgetting_eval` — re-probe OLD facts after later days; retention should stay high (the development STICKS). |
| **Structural / size growth** | the brain GROWS as it develops | `TierPromoter` growth events + neuron/synapse count over days (lineage `growth_events`, `current_tier`). |
| **The no-confab moat (held constant)** | the brain abstains on what it was NOT taught, throughout development | false-accept rate on untaught cues, asserted unchanged across all days (the moat is a plus, never weakened for free — `feedback_moat_not_hard_lossy_memory_ok`). |

### The anti-cheats (is it REALLY developing, or memorizing/drifting?)

1. **Held-out probe (never-taught-but-in-distribution).** Development = generalization, not memorization. Each day, probe on facts/queries NOT in that day's stream but within the learned vocabulary. Rising held-out competence = genuine development; flat held-out while train-competence rises = memorization. (Mirrors the project's standing "held-out, not train, accuracy" discipline.)
2. **Retention / catastrophic-forgetting control (the no-replay arm).** Run a YOKED loop with consolidation DISABLED (no SWR replay). It MUST catastrophically forget old days (retention collapses) while the consolidated loop retains — the load-bearing causal contrast (exactly the `2026-06-23-generative-loop-DEMONSTRATED.md` no-replay control, 2.25× forget-contrast). If consolidation-off retains just as well, the consolidation isn't load-bearing → the "development sticks" claim is unproven.
3. **Frozen-brain control (is the stream load-bearing?).** A control where the brain HEARS the stream but plasticity is gated OFF (`set_plasticity_gate=0`): competence must NOT rise (proving the stream → learning coupling drives development, not test-time luck). Mirrors `feedback_validate_signal_by_its_function` — validate the development signal by its function.
4. **Permuted-curriculum control.** Teach the SAME facts in a SCRAMBLED day-order; the final developed competence should be ~equivalent (development is cumulative, not order-artifact) BUT the per-day trajectory should differ — a flat trajectory regardless of order would mean the metric isn't tracking real learning. (The project's standing permuted-label discipline.)
5. **Persistence anti-cheat (does the brain RESUME, not re-derive?).** A no-persistence control: kill + cold-start the loop mid-development; the post-reload competence must DROP (proving the lineage carries the developed state, not that it re-derives instantly) — the exact `persistent_living_loop_derisk.py` no-persistence anti-cheat, ported.
6. **The moat-never-loosens assertion.** Across the entire development run, the `is None` abstention on untaught cues stays byte-honest — development adds knowledge, it does not erode the no-confab discipline.

---

## 5. The human-interaction test on the developed brain

**Goal:** a person opens a REPL on the *developed* lineage and converses with the grown-up brain to judge how it responds (does it know more than the day-0 brain? does it answer coherently? does it abstain honestly?).

**What exists:** `chat_repl.py` is the REPL + lineage-load skeleton, but wired to the 4/8-word direction architecture. The conversational engine (`MultiTurnAgent` / `BrainConversationalAgent` / `consolidated_320_conversation_demo`) is the real brain.

**What's needed (small):** a thin REPL that:
1. Loads the developed `BridgeLineage` (the brain at "age N").
2. Constructs the conversational agent on the lineage's learned codes/facts (the agent already accepts `grounded_codes=`, so the lineage's stream-learned codes drop in — `MultiTurnAgent(grounded_codes=...)`).
3. Reads a human line → dispatches to `agent.hear` (teach it something new live) / `agent.what_does`/`who_does` (ask it) / `agent.reason_chain` (multi-hop) / `agent.elaborate` (dialogue-plan) → renders the reply via the brain's own neural renderer (LLM-minimal) OR the fluent Qwen faculty (optional).
4. Shows the development side-by-side (optional): the same query against the day-0 vs day-N lineage, so the human SEES the brain grew.

This is a UI/wrapper job over validated pieces — no new mechanism. The honest scope: the developed brain is conversationally competent within its learned vocabulary/facts (terse SVO replies by default, fluent if the LLM faculty is enabled), with the no-confab moat making it abstain honestly on what it wasn't taught — which is itself a compelling thing for a human to probe ("ask it about something you never taught it; watch it honestly say it doesn't know").

---

## 6. Compute / time scaling — the feasibility crux + the cheapest-first de-risk

### The per-component wall-clock anchors (verified from findings)

| Operation | Measured cost | Source |
|---|---|---|
| **Stream-cortex learning, 320 concepts** | ~96 min for 150K windows (~469 windows/concept), 9920-neuron bridge, GPU | `2026-06-15-…GO.md` (320-scaling) |
| **Stream-cortex learning, 64 concepts** | ~290 s/seed at 30K windows (~470 windows/concept), GPU | same |
| **A full 320-concept conversation (onebrain)** | ~1–2 min/seed (construction + the conversation), GPU | `2026-06-18-onebrain-320-scale-production-GO.md` |
| **One conversational query (onebrain, masked megakernel)** | 96 ms/query (vs rf 416 ms), GPU | CLAUDE.md A5 speed arc |
| **One grounded-render generation (spiking Qwen T=16, greedy)** | ~0.3–1.2 s/generation, PyTorch-3090 | `2026-06-23-grounded-lang-INTEGRATION-GO.md` |
| **SWR consolidation phase** | minutes (sleep replay of the day's tagged ensembles), GPU | `consolidation_trainer.py` (per-tag replay windows) |

### The naive cost of "years of daily conversation" (why a COMPRESSED horizon is mandatory)

A literal "1 year of daily conversation" is infeasible AND unnecessary. The key realization: **the stream cortex's developmental signal is windows-bounded, not wall-clock-bounded** — "1 day" in the loop is a CHOSEN number of stream windows + turns, not 86,400 real seconds. So "simulated time" is a knob:
- 1 "day" = e.g. a few thousand stream windows (learning) + tens of conversational turns + one consolidation phase ≈ **single-digit minutes to ~½ hour GPU** at the 64–320-concept scale (extrapolating the ~470-windows/concept rate + the ~1–2-min conversation + the minutes-scale consolidation).
- A compressed "1 week" (7 days) ≈ **hours of GPU** — a single overnight run.
- A compressed "1 month" (30 days) ≈ **a long local run (a day or two of GPU)** — still local.

The development SIGNAL (vocab grows, facts accumulate, retention holds, tier promotes) appears across a HANDFUL of days — you do not need a literal year to SEE the brain develop. The "weeks/months/years" framing maps to "enough simulated days that the trajectory is unmistakable," which the compressed horizon delivers.

### The cheapest-first de-risk (the recommended decisive falsification)

**Probe (suggested):** `_longitudinal_dev_loop_derisk.py` — a **compressed "1 week" (e.g. 7-day) development loop** on the EXISTING conversational stack (stream cortex + `MultiTurnAgent` + consolidation + lineage), with a Claude-authored / scripted developmentally-graded daily curriculum (partner LLM-MINIMAL), the brain's OWN neural renderer (output LLM-MINIMAL), per-day development metrics, and the §4 anti-cheats. Cheap-first = 1-seed GPU smoke to decide GO/NEGATIVE before any multi-seed/longer-horizon commit. **`SIM_BACKEND=cupy` (GPU)** for the decisive run; numpy only for a tiny CPU smoke of the loop plumbing.

**The decisive checks (explicit GO / BOUNDARY / NEGATIVE):**

| # | Check | GO | BOUNDARY | NEGATIVE |
|---|---|---|---|---|
| 1 | **The loop CLOSES end-to-end** (7 days run: hear→consolidate→[grow]→persist→next day, no crash, state resumes each day) | runs all 7 days, day-N resumes day-(N−1)'s brain | runs but a sub-phase (e.g. growth) is skipped | the loop cannot string the phases (an assembly gap) |
| 2 | **The brain DEVELOPS** (vocab + facts + recall rise monotonically over days; held-out competence rises, not just train) | clear upward trajectory on held-out competence | rises on train but flat on held-out (memorizing) | flat/declining (no development) |
| 3 | **Development STICKS** (day-1 facts still recalled at day 7, consolidation-on) AND the no-replay control catastrophically forgets | retention high with consolidation, low without (clear contrast ≥1.3×) | retention high but the no-consolidation control ALSO retains (consolidation not load-bearing) | day-1 facts lost by day 7 even WITH consolidation |
| 4 | **The brain GROWS** (TierPromoter fires when a tier is mastered; brain size + tier rise in the lineage log) | ≥1 tier promotion across the week, logged | mastery reached but promotion not wired | no growth |
| 5 | **A human can converse with the developed brain** (the REPL loads day-7 lineage; answers in-vocab queries, abstains honestly off-vocab) | coherent in-vocab answers + honest abstention; visibly knows more than day-0 | answers but the day-0-vs-day-N difference is marginal | the developed brain is no better than day-0 |

**Why this framing.** It is the smallest end-to-end run that proves the LOOP CLOSES and the DEVELOPMENT SIGNAL is real + sticks + visible to a human — on the existing substrate, LLM-minimal, decoupled from the generator-capacity wall (§1C ⚠️). A GO here is the first time the project has a *developing-over-simulated-time, persistent, conversational* brain a human can meet — the Tier-3 development capstone at demonstrable scale. The per-day wall-clock measured in this run is the input to the horizon-scaling decision (§6 next).

**Crucially, this de-risk is DECOUPLED from the two known walls:**
- The **generator grow-no-forget capacity wall** (§1C ⚠️, `2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md`): the development loop's "no-forget" is carried by the **stream cortex + lineage + SWR consolidation** (vocab/facts, NOT a generator distribution) — which IS validated to retain (Phase-1.3 94%, Phase-1.4 5/6). The *generative-faculty* distribution-growth (which hit the toy-capacity wall) is the OPTIONAL free-generation upgrade, not the development metric. So "the conversational brain develops + retains" is reachable NOW; the generator scale-up is a separate, later, opt-in deepening.
- The **dendrite / learned-spatial-policy wall** (Tier-4): irrelevant to the development-axis (that wall is navigation credit-assignment; the development loop is conversational).

### The cloud trigger (per `feedback_long_local_runs_ok_confirm_cloud_cause`)

**The development loop is LOCAL.** Cloud is only NEEDED for a genuine VRAM wall (>24 GB); wall-clock just makes runs longer (fine with an ETA). The development substrate is the 320-concept stream cortex (~10K neurons, ~25M synapses) + the conversational agent — comfortably <24 GB (the 320-stream-run fit on the 3090). A longer horizon (a "month," a "year") is a **longer LOCAL run**, not a cloud job. **The ONE thing that could trigger cloud is the OPTIONAL generative-faculty scale-up** (§1C ⚠️: the C2 grow-no-forget needs a 50–200M generator to hold in-band shifts) — and even that is *likely local* (50–200M fits in 24 GB; the finding's own call was "a LOCAL longer-wallclock scale-up, NOT cloud"). **So: build the development loop locally; size + measure the generative-faculty scale-up only if/when the free-generation upgrade is prioritized, and call local-vs-cloud then by the VRAM rule.** Do NOT propose cloud for the development loop itself.

---

## 7. The honest reality check + ranked cheapest-first plan

### Is the longitudinal development test achievable on the existing substrate?

**YES for the conversational-development loop (the development axis the owner named), as an ASSEMBLY — and it is genuinely the smallest thing that turns the conversational stack into "a brain that grows up."** Every cognitive piece is validated; the build is an outer-loop runner + per-day metrics + a REPL rewire, templated on the existing `persistent_living_loop_derisk.py` survival loop; the prior is no `sim/` edit (reuse-by-import, exactly the pattern of the survival loop + the grounded-lang arc). The development SIGNAL (vocab + facts + retention + growth) is reachable on a compressed horizon WITHOUT solving any deep wall.

### What is genuinely bounded (the honest caveats)

1. **The GENERATIVE faculty's distribution-growth is the one piece that hit a scale wall** (§1C ⚠️). This affects ONLY the *free-generation* upgrade (the brain inventing novel fluent continuations whose distribution grows without forgetting) — NOT the development metric (vocab/facts/recall via the stream cortex + lineage, which retains fine). Frame the loop's development on the validated stream-cortex + consolidation; offer free-generation as an opt-in deepening gated on the (likely-local) generator scale-up.
2. **Long-horizon open-ended multi-turn is SHORT-validated** (§1B): the validated multi-turn is 2–4 turns; sustaining many turns/day needs discourse-buffer management (a loop + eviction policy), small but real.
3. **The auto-growth real-bridge wiring is a stub** (§1D): `auto_grow_chat.py` runs the orchestration with mock train/transfer — wiring the real `run_three_factor` + `set_pathway_weights` is the genuine (small) build for the "brain grows in size" metric.
4. **The output is terse (SVO) under the LLM-minimal renderer**; fluent prose needs the optional off-bridge LLM faculty. This is a presentation choice, not a development blocker.
5. **The consolidation_trainer has a SKELETON header** for some eval modes (`:24-31`) — the replay PHASE functions are validated, the full awake/sleep orchestration on a *production lineage* is part of the assembly (verify on a smoke before the multi-seed).

### Ranked cheapest-first plan (options for the owner to steer)

- **Phase L0 (recommended cheap-first, ~days): the compressed-week development loop on the conversational stack.** Assemble `develop(7 days)` over the stream cortex + `MultiTurnAgent` + `consolidation_trainer` + `BridgeLineage`, with a Claude-authored/scripted graded daily curriculum (partner LLM-minimal), the brain's own neural renderer (output LLM-minimal), per-day development metrics, and the §4 anti-cheats. De-risk = §6 checks 1–4 (1-seed GPU smoke → 6-seed). **GO = the first developing-over-time persistent conversational brain.** This is the recommended next step.
- **Phase L1 (after L0 GO, small): the human-REPL on the developed brain.** Rewire `chat_repl.py` to the conversational agent on the developed lineage (check 5). Add the optional fluent-LLM faculty for the human-facing prose. The "meet the grown-up brain" deliverable.
- **Phase L2 (medium): wire real auto-growth + scale the horizon.** Replace the `auto_grow_chat.py` stubs with real `run_three_factor` + `set_pathway_weights` so the brain genuinely grows in size across tiers; scale the horizon (a "month") as a longer LOCAL run. Add long-horizon multi-turn discourse-buffer management.
- **Phase L3 (opt-in deepening, gated on a generator scale-up): the free-generation upgrade.** Add the generative-loop generator + the grow-no-forget loop so the developed brain can also *free-generate* novel fluent text whose distribution grows without forgetting — gated on the (size + VRAM measured, likely-local) 50–200M generator scale-up (§6 cloud trigger). The fluent off-bridge LLM faculty is the alternative surface-fluency path that needs NO generator scale-up.

**Recommended order:** L0 → L1 → (L2 / L3 as the owner prioritizes). The development loop is reachable now; the free-generation scale-up and the dendrite are the separately-tracked deepenings.

---

## 8. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / finding read):**
- **Stream cortex learns from the conversation stream, scaled to 320 multi-seed** — `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (read in full: rate-Hebbian `corr(M,C)` +0.686 6-seed; population lift to 100–108%; 320-scaling 150K windows ~96 min recall 1.00/moat 1.00 multi-seed; STDP measured-negative). The developmental engine.
- **Multi-turn agent + multi-hop + moat** — `multi_turn_agent.py` header (read: `SpikingLoopContextBuffer` cross-turn anaphora, biased-competition opt-in, reuse-by-import NO `sim/` edit). Production demos exist (`consolidated_320_conversation_demo.py`, `multi_turn_conversation_demo.py` confirmed present).
- **Lineage persistence + the development log fields** — `sim/lineage.py` (read `:40-108`: `GrowthEvent`/`AccuracyDatapoint`/`LineageMetadata` with `vocab`/`cumulative_training_events`/`accuracy_history`/`growth_events`; atomic save). The metadata IS a development log.
- **Auto-growth TierPromoter + the stub status** — `sim/auto_growth.py` (read `:38-79`: `TierLadder` 4→320, threshold 0.90/3-consecutive, `set_pathway_weights` transfer); `auto_grow_chat.py` (read header `:1-24`: orchestration end-to-end but `synthetic_train_fn`/mock bridge — real hooks are the next wiring).
- **CLS consolidation (SWR replay)** — `consolidation_trainer.py` (read `:1-60`: awake/sleep + `run_concept_replay_phase`/`run_swr_replay_phase`; the SKELETON header caveat `:24-31`). Phase-1.3 validated per CLAUDE.md (94% retention, 3/3 anti-cheat).
- **The grow-no-forget LOOP DEMONSTRATED + its scale caveat** — `2026-06-23-generative-loop-DEMONSTRATED.md` (read: FT-LR 1e-5 → retain 88% / new −87% / no-replay 39% / 2.25× contrast, all 3 bars) AND `2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md` (read: moderate/in-band shift hits the 3.4M capacity wall, best 53.9% retention) — the two reconcile (extreme shift GO at toy scale; in-band needs scale-up). **Load-bearing for framing the development "no-forget" on the stream-cortex/lineage, not the generator.**
- **The output LLM-minimal path (brain's own renderer)** — `neural_serial_order_renderer.py` present; `BrainConversationalAgent(enable_neural_render=True)` per CLAUDE.md (GPU GATE PASS). The with-LLM path — `2026-06-23-grounded-lang-INTEGRATION-GO.md` + `-SCALED-GO.md` + `-P1b-GO.md` (read: spiking Qwen renders gated facts, verified, moat preserved; ~0.3–1.2 s/gen; off-bridge PyTorch).
- **The partner LLM-minimal path (Claude-authored offline curriculum)** — `2026-06-23-grounded-lang-P2-GO.md` (read: ~30-fact curriculum ingested, recall 1.000/moat 0-FA 6-seed; P2≠Path-3 confirmed in practice).
- **The continuous-outer-loop + lineage-persistence PRECEDENT** — `persistent_living_loop_derisk.py` (read header: `live()` loop + `LivingState` persisted via `BridgeLineage` + full anti-cheat battery — the structural template; it is the SURVIVAL loop, not the development loop).
- **The REPL skeleton** — `chat_repl.py` (read `:1-70`: loads `BridgeLineage`, interactive, but tier1/synonym direction-vocab — needs rewiring to the conversational agent).
- **Wall-clock anchors** — verified from the findings (320-stream ~96 min; conversation ~1–2 min/seed; query 96 ms megakernel; render 0.3–1.2 s).

**Could NOT fully verify (flagged honestly — for the L0 smoke to confirm):**
1. **[VERIFY — most load-bearing]** That the four subsystems compose in ONE process without state-corruption across the wake→sleep→grow→persist phases on a real lineage (each is validated standalone; the awake/sleep gate transitions on a *production conversational* lineage — vs the direction-vocab lineage Phase-1.3 used — is the assembly's genuine integration risk; the `consolidation_trainer` SKELETON header is the flag). The L0 check-1 smoke resolves this.
2. **[VERIFY — the development trajectory]** That recall/held-out competence RISES monotonically over compressed days at the 64–320-concept scale (the stream cortex is validated to LEARN to a fixed point; the *incremental day-over-day* accumulation — adding vocab/facts each day on top of the prior day's lineage state — is the new thing; L0 check-2 measures it).
3. **[VERIFY — the per-day wall-clock]** The actual minutes/day at the chosen scale (extrapolated from the anchors as single-digit-min to ~½ hr; the L0 run MEASURES it → the horizon-scaling input).
4. **[VERIFY — auto-growth on a real conversational bridge]** That `TierPromoter` + `set_pathway_weights` cleanly promote a *conversational* lineage tier (validated for the vocab-tier arch; the conversational-agent arch promotion is the L2 wiring — flagged, not on the L0 critical path).

---

## Sources

### Project record (re-verified this pass, file:line / finding cited)
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (the stream cortex — the developmental engine; read in full).
- `research/runners/multi_turn_agent.py` (MultiTurnAgent; header read). `research/runners/consolidated_320_conversation_demo.py`, `multi_turn_conversation_demo.py`, `brain_conversational_agent.py`, `rf_phasor_composer.py`, `one_brain_composer.py` (present, confirmed).
- `sim/lineage.py:40-108,190,238` (BridgeLineage + the development-log metadata). `sim/auto_growth.py:38-79` (TierPromoter/TierLadder). `research/runners/auto_grow_chat.py:1-24` (orchestration end-to-end, real-hook stubs).
- `research/runners/consolidation_trainer.py:1-60` (SWR replay + the SKELETON caveat). `research/runners/chat_repl.py:1-70` (the REPL skeleton + lineage-load).
- `research/runners/persistent_living_loop_derisk.py` (the continuous-loop + lineage-persistence TEMPLATE — survival, not development).
- `research/findings/2026-06-23-generative-loop-DEMONSTRATED.md` (grow-no-forget loop GO at toy scale, FT-LR fix) + `2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md` (the in-band scale wall — the honest caveat).
- `research/findings/2026-06-23-grounded-lang-{P1b,P2,INTEGRATION,SCALED}-GO.md` (the output-faculty paths + the Claude-authored-curriculum partner pattern).
- `research/runners/neural_serial_order_renderer.py` (the brain's own generation — the output LLM-minimal path).
- `research/findings/2026-06-20-tier3-artificial-life-capstone-deep-research.md` (the prior Tier-3 scoping — the SURVIVAL axis; this doc is the complementary DEVELOPMENT axis).

### Memory pointers (the directives this scoping honors)
- `project_actual_goal_artificial_life_brain_analogue` (the north star); `project_post_conversational_roadmap_tiers` (Tier-3 capstone); `feedback_move_everything_to_shared_spiking_substrate` + `feedback_brain_based_only_standard` (the partner is the environment = legitimate host; the output prefers the brain's own generation); `feedback_long_local_runs_ok_confirm_cloud_cause` (compressed-horizon local; cloud only on a >24 GB VRAM wall); `feedback_6seed_validation` (6-seed before any "works"); `feedback_moat_not_hard_lossy_memory_ok` (the moat held but not the gate); `feedback_validate_signal_by_its_function` (the development anti-cheats).
