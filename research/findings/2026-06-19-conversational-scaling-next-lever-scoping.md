# Conversational scaling — the next-lever decision (deep-research + catalog scoping, 2026-06-19)

> **READ-ONLY deep-research + catalog scoping. No code edited, no jobs run; this doc is the only write.** Produced
> per the standing "deep research + catalog review FIRST at a new direction" directive (CLAUDE.md;
> `feedback_deep_research_at_roadblocks`). Every load-bearing project fact below was re-verified against the repo
> (file/finding cited); the decision-flipping ones (the resonator already RESOLVES nested decode; the dendrite
> double-NEGATIVE; the production agent's integration state) were read in full, not trusted from a summary. The
> controller should trust-but-verify the **[VERIFY]**-flagged claims, then push + present the recommendation.
>
> **The decision this scopes (owner-approved 2026-06-19):** the dendrite is now thoroughly ruled out for the
> current walls (both its jobs tested cheap-first → NEGATIVE — nav credit-assignment AND two-attribute binding).
> The owner pivots to the PRIMARY (task #55): build the conversational architecture toward "basic-LLM-competitive,"
> biology-faithful, on the validated spiking substrate. This doc RANKS the highest-leverage, cheapest-first next
> conversational-capability lever. The owner offered candidates (F=3 resonator for two-attribute binding;
> longer/multi-turn dialogue; bigger vocabulary; deeper reasoning) but left the choice to me.
>
> **Relationship to the two prior LLM-scoping docs** (`2026-06-17-conversational-architecture-to-basic-LLM-scoping.md`,
> `2026-06-17-capability-frontier-to-basic-LLM-scoping.md`, `2026-06-18-richer-syntax-conversational-frontier-scoping.md`):
> those ranked the *mechanism* levers and most of their #1 picks have since been BUILT + validated (ordered-WM,
> multi-frame comprehension, attributed entities, reconsolidation, multi-hop, anaphora). This doc **re-scores against
> what is now validated** and surfaces a structural fact those docs did not foreground: **the validated capabilities
> are scattered across standalone de-risk runners and are NOT integrated into the production one-brain agent.** That
> reframes the highest-leverage next lever.

---

## 0. The one-paragraph answer (the ranked recommendation)

The honest North-star framing first: this is a **structured VSA/FHRR composer on one spiking bridge** — it *parses*
facts into role-filler structure, *binds/stores* them, *recalls* with a no-confab abstention moat, and *generates*
neurally-ordered answers. "Basic-LLM-competitive" is the owner's shorthand for a **capability target**, not a literal
GPT-match: a free-form next-token LLM and a structured composer are **categorically different machines**, and the
goal is the brain analogue (capabilities instrumental), so the bar is "converse fluently and flexibly over its known
world, the biology-faithful way," not "emit open-domain paragraphs." Against that bar, the project is genuinely
*ahead* of a basic LLM on grounding-from-conversation, trustworthy abstention, and multi-hop reasoning-with-a-moat,
and genuinely *behind* on breadth (vocabulary) and on having all its syntactic flexibility *available in one agent at
once*. **The #1 lever is NOT a new mechanism — it is INTEGRATION + CONSOLIDATION: fold the already-validated richer
capabilities (attributed entities, multi-frame comprehension, nested/two-attribute decode via the resonator) into the
ONE production agent (`OneBrainComposer` / `BrainConversationalAgent`), behind opt-in flags, with the no-confab moat
re-asserted, so the agent the owner actually converses with can do what the de-risk runners proved.** This is the
highest leverage (it converts ~4 validated-but-shelved capabilities into felt conversational ability), the cheapest
(reuse-by-import, no `sim/` edit, no new science), the most aligned with `feedback_check_existing_sims_first` (Spaun's
whole point is that the pieces run in ONE model — the project has the pieces but not yet the one model), and the
correct sequencing move before any deeper mechanism work, because **the integration will localize which gap is
actually felt** when conversing with the consolidated agent. **The cheapest-first first step:** wire the
attributed-entity parse + the resonator nested-decode into `OneBrainComposer.hear`/`query_patient` behind a default-OFF
flag and run the production conversational matrix (`tests/test_one_brain_composer_agent.py` family) with attributed +
nested facts, asserting the moat holds and flat-SVO is un-regressed. On the owner's named candidates: **the F=3
resonator is already deployed for nested/two-attribute DECODE** (RESOLVED 6/6, `nested_composition_agent`) — it is the
*integration* that is missing, not the mechanism; **bigger vocabulary is the deferred cloud axis** (a corpus-grounded
taxonomy + training budget, not a local mechanism gap); **deeper reasoning and multi-turn dialogue are already
production GO** for their validated scope. The genuinely-OPEN frontiers *after* integration are two precisely-specified
mechanism builds — **multi-referent disambiguation (needs WTA biased-competition; two converging NEGATIVEs name it)**
and **embedded-clause PARSING from a flat token stream (the hard half of syntax)** — and they are correctly sequenced
AFTER the integration localizes whether they are felt.

---

## 1. HONEST target framing — what "basic-LLM-competitive" means for THIS machine

### 1.1 The categorical difference (state it plainly, don't pretend to match a GPT)

| Axis | A basic small LLM (Phi-3-mini / Llama-3.2-3B class) | This project (VSA/FHRR structured composer on one spiking bridge) |
|---|---|---|
| **Representation** | learned continuous token embeddings; meaning is statistical | role-filler bound phasors + a stream-learned PPMI cortex; meaning is **structured + grounded-from-conversation** |
| **Generation** | autoregressive next-token over an open vocabulary | **structured**: parse → bind → store → recall → neurally-ordered emit over a known vocabulary |
| **Knowledge** | implicit in ~360M+ weights, trillions of tokens | explicit facts in complex synapses + ~320 stream-learned concepts |
| **Failure mode** | **confabulates** fluently when it doesn't know | **ABSTAINS** (the no-confab moat) — the signature advantage |
| **Open-ended fluency** | yes — paragraphs of novel text | **NO — a documented terminal wall** (from-scratch spiking LM overfits at scale, SpikeGPT); out of scope, do not re-attempt |

These are **different kinds of machine.** A faithful brain analogue that converses is not a small GPT — it is a
parser + binder + recaller + planner that happens to talk. "Compete with a basic LLM" must mean **"converse
fluently and flexibly over its known world, by mechanism, the biology-faithful way"** — NOT "emit open-domain free
text." This framing is the North star (`project_actual_goal_artificial_life_brain_analogue`): capabilities are
instrumental to a biology-translatable brain analogue; honest negatives under strict biology are the deliverable.

### 1.2 Where the architecture is genuinely CLOSE (or ahead) vs genuinely BEHIND

**Genuinely ahead of a basic LLM:**
- **Grounding from conversation** — the 320-concept stream cortex LEARNS each word's meaning from a conversation
  stream (population-Hebbian co-occurrence + local PPMI normalization + population coding) and **generalizes across
  similar concepts on the point-neuron substrate** (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`).
  A small LLM is pre-trained; this learns online, biology-faithfully.
- **Trustworthy abstention (the moat)** — abstains rather than confabulating, at 320 concepts with 0 false-accepts
  (`2026-06-18-onebrain-320-scale-production-GO.md`). This is the project's signature advantage; an LLM's headline
  weakness.
- **Multi-hop reasoning WITH a moat at every hop** (`query_chain`, GO 3×3, `2026-06-17-multihop-query-chain-GO.md`).

**Genuinely behind, but the gap is BREADTH/INTEGRATION not a missing kind of cognition:**
- **Vocabulary scale** (320 vs an LLM's tens of thousands) — the **deferred cloud axis**, not a local mechanism gap.
- **Syntactic flexibility available IN ONE AGENT** — the *pieces* exist (see §2) but are scattered across de-risk
  runners; the production agent the owner converses with does not yet have them all.

**Behind and out of scope:** open-ended free-text generation (the documented terminal wall — do not re-attempt).

---

## 2. THE CAPABILITY GAP MAP (validated / boundary-with-named-fix / not-started)

Each row tagged. The **load-bearing surprise** (verified by code-read, §3) is the **integration column**: many richer
capabilities are **validated in a standalone runner but NOT wired into the production `OneBrainComposer` /
`BrainConversationalAgent`** the owner actually converses with.

| Capability | Mechanism status | In the PRODUCTION agent? | Tag |
|---|---|---|---|
| Word/concept meaning (learned, generalizing) | ✅ stream cortex, multi-seed GO (2026-06-15) | ✅ yes (320-scale production default) | **validated** |
| Store / recall SVO facts | ✅ FHRR bind + who/what, multi-seed | ✅ yes | **validated** |
| No-confab abstention (the moat) | ✅✅ 320-scale, 0 false-accepts | ✅ yes | **validated — protect, never weaken** |
| Negation / yes-no | ✅ polarity role | ✅ yes (`OneBrainComposer`) | **validated** |
| Neural word-order generation (SVO frame) | ✅ spiking CQ serial-order, GO 6/6 (2026-06-16) | ⚠️ opt-in `enable_neural_render` | **validated, partially integrated** |
| Multi-hop reasoning | ✅ `query_chain`, GO 3×3 | ✅ yes (`reason_chain`) | **validated** |
| Multi-turn anaphora (single referent) | ✅ `MultiTurnAgent`, GO 3-seed | ✅ yes | **validated** |
| Reconsolidation (correct a fact in dialogue) | ✅ `update_on_mismatch`, GO 6/6 | ✅ yes (`OneBrainComposer`) | **validated** |
| **Attributed entities (adj+noun) — comprehension+recall** | ✅ **fully neural end-to-end, GO 6/6** (CYCLE 199-201) | ❌ **standalone `AttributedBridgeParser`+`RFPhasorComposer`; NOT in `OneBrainComposer.hear`** | **validated, NOT integrated** |
| **Nested / two-attribute DECODE ("dog eat (big red ball)")** | ✅ **RESOLVED via restart-selected resonator, GO 6/6, algebra+spiking** (2026-06-03) | ❌ **standalone `nested_composition_agent`; NOT in the production one-brain agent** | **validated, NOT integrated** |
| **Multi-frame comprehension (VSO/OSV/questions)** | ✅ **GO 6/6** per-frame (CYCLE 203) + frame-SELECTION GO | ❌ **standalone `_phaseB_multiframe_comprehension`; no `FrameParser` in the agent** | **validated, NOT integrated** |
| Multi-frame word-order GENERATION | ✅ learned, GO 6/6 spikes (CYCLE 143) | ⚠️ not promoted into the agent's `describe` | **validated, NOT integrated** |
| Multi-SENTENCE / discourse emission | ✅ `MultiTurnAgentV2.narrate`, GO 6/6 | ⚠️ in `MultiTurnAgentV2`, not the one-brain default | **validated, partially integrated** |
| **Two-attribute generalizing BUNDLING (a LEARNED bind)** | ⛔ **NEGATIVE** — fixed ±1 bundles 0.989 but a *learned* bind memorizes (0.168 held-out), dendrite NEGATIVE (2026-06-19) | n/a (production uses the FIXED ±1 bundle, which works) | **boundary — production keeps the fixed primitive; learned bind is the open axis, ≠ a felt gap** |
| **Multi-referent disambiguation** (which of several held referents a bare pronoun binds) | ⛔ **NEGATIVE × 2** — not recency, not salience-boost; **needs WTA biased-competition** (lateral inhibition between referent attractors + salience bias) | n/a (single-referent works) | **boundary — NAMED fix, precisely specified, not-started** |
| **Embedded-clause PARSING from a flat token stream** | ⚠️ composer DECODES a provided `Clause`; the PARSER cannot build the nesting from "the dog that chased the cat ran" | ❌ "provided structurally" | **not-started (the hard half of syntax)** |
| Instruction → cognitive operation ("list X", "compare X and Y") | ⚠️ language→BODY GO (`spoken_instruction_nav`); language→cognitive-op absent | ❌ | **not-started (point-neuron-feasible, no wall)** |
| In-context few-shot adaptation | ❌ absent | ❌ | **not-started (lower priority for "converse")** |
| Open-ended free-text generation | ⛔ **terminal wall** (SpikeGPT overfit; ~360M+ params) | n/a | **out of scope — do not re-attempt** |

### 2.1 The decision-flipping clarification on the owner's "F=3 resonator / K=5" candidate

The owner's framing ("the F=3 resonator for two-attribute binding, to lift the K=5 wall") **conflates two different
capabilities that the project has since separated**. Holding them apart is load-bearing:

1. **Two-attribute / nested DECODE** ("big red ball" as a patient; a fact-about-a-fact) = **factoring a product of F
   unknown factors**. This is **ALREADY RESOLVED** by the resonator network (Frady-Kent-Olshausen-Sommer 2020) with
   restart-selection for the repeated-codebook case — **GO 6/6, in the algebra AND in spikes**, deployed in
   `nested_composition_agent` (`2026-06-03-resonator-decoder-cheap-first-RESOLVES…`,
   `2026-06-03-multi-modifier-attribution-resonator-restarts-RESOLVES.md`). **So the F=3 resonator is NOT a missing
   mechanism — it is a validated, spiking, scalable decoder that is simply NOT WIRED into the production one-brain
   agent.** The "cheap win" the owner intuits is real, but it is an **integration**, not a build.

2. **Two-attribute generalizing BUNDLING from a LEARNED bind** (a learned bind that generalizes to held-out attribute
   pairs) = the actual residual wall. But: the **FIXED ±1 / FHRR primitive already bundles a 3-way fact at 0.989**
   (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`); only a *learned* bind fails to
   generalize (0.168 held-out), and the dendrite NEGATIVE (`2026-06-19-dendritic-binding-toy-derisk.md`) confirms even
   a learned dendritic multiplication memorizes rather than generalizes — **worse than the fixed primitive (0.261) it
   would replace.** So a learned generalizing two-attribute bind is a *different axis* (generalization-across-similar-
   concepts, already carried by the PPMI cortex), **not a felt conversational gap** — production keeps the fixed
   primitive, which works.

⇒ **Net: the "K=5 two-attribute" item is mostly already solved** (decode via the resonator) **and the residual
(learned generalizing bundle) is NOT the lever** (the fixed primitive already does the job; the dendrite NEGATIVE
closed the learned route). The owner's candidate, examined, **points straight at the integration recommendation**:
the resonator/nested capability exists and is unintegrated.

### 2.2 The structural theme

The genuinely-missing *kinds* of cognition reduce to two precisely-specified mechanism builds (multi-referent WTA
biased-competition; embedded-clause parsing) — both correctly sequenced AFTER integration. Everything else the owner
named is either **already validated-but-unintegrated** (attributed/nested/multi-frame, neural render, narrate) or
**the deferred cloud axis** (vocabulary). This is exactly the shape that makes **integration the highest-leverage,
cheapest, lowest-risk next move.**

---

## 3. THE RANKED RECOMMENDATION

**Scoring axes** (per the task): leverage toward the North-star brain-analogue conversation × cheapness/cheap-first
de-riskability × alignment with the owner's "build on theta-multiplexing / mode-unification / generative-replay, not
RAG-ranking" steer × reuse of existing validated machinery × moat held (traded only where biological per
`feedback_moat_not_hard_lossy_memory_ok`).

### #1 (RECOMMENDED) — INTEGRATE the validated richer capabilities into the ONE production agent (attributed entities + resonator nested-decode + multi-frame comprehension + neural render), behind opt-in flags, moat re-asserted

- **What it is:** fold the four validated-but-shelved capabilities (§2: attributed entities, nested/two-attribute
  decode via the resonator, multi-frame comprehension, multi-frame/neural-render generation) into
  `OneBrainComposer` / `BrainConversationalAgent` (the 320-scale production default the owner converses with), each
  behind a default-OFF flag mirroring the existing `enable_neural_render` / `composer_kind` pattern, and validate the
  consolidated agent on the production conversational matrix with attributed + nested + multi-frame facts.
- **Why #1 (the scoring):**
  - **Leverage — HIGHEST.** It converts ~4 validated capabilities from "proved in a runner" into "felt when you talk
    to the agent." The owner's stated desire is *to converse with the artificial-life agent*; today that agent is
    flat-SVO-only even though the stack can do far more. This is the single largest jump in *felt* conversational
    ability available, at near-zero scientific risk.
  - **Cheapness — HIGHEST.** Reuse-by-import, **no `sim/` edit, no new science** — the mechanisms are validated; the
    work is wiring + a frozen-gate regression. Hours-to-a-day per capability, not weeks.
  - **`feedback_check_existing_sims_first` alignment — STRONGEST.** Spaun's entire contribution is that perception,
    binding, working memory, BG-Control routing, and question-answering run in **ONE 2.3M-neuron model**, not as
    separate demos. The project has built the pieces (FHRR bind = semantic-pointer binding; dlPFC Control = BG
    routing; ordered-position WM = order-encoded WM; parser = the comprehension front end) but **has not yet put them
    in one agent.** Integration is precisely the move the field's flagship validates as the thing that makes it "a
    cognitive model" rather than a pile of capabilities. **[VERIFY: that `OneBrainComposer` lacks resonator/attributed/
    multi-frame — confirmed by `grep` (only `brain_conversational_agent.py` + `rf_phasor_composer.py` reference them;
    `one_brain_composer.py` does not).]**
  - **Owner-steer alignment — STRONG.** This is the opposite of RAG-ranking: it consolidates the **theta-gamma
    ordered-WM / structured-binding / spreading-Control** machinery (the owner's preferred substrate) into one loop.
    It also directly serves the standing TRUE-ONE-BRAIN directive (`feedback_move_everything_to_shared_spiking_substrate`)
    — more cognition consolidated onto the one bridge.
  - **Reuse — TOTAL.** `AttributedBridgeParser`, `nested_composition_agent`'s resonator (`_resonator2`/`_resonator3`,
    depth model-selection), the multi-frame `FrameParser` pattern, `neural_serial_order_renderer` — all reuse-by-import.
- **Anti-cheat / moat:** the no-confab moat must hold for the *richer* facts too (an unstored attributed/nested cue →
  `None`); assert abstention + 0 false-accepts on the consolidated agent, and assert **flat-SVO un-regressed** (the
  richer paths must not degrade the validated flat path — the exact discipline the attributed-entity GO used). Per
  `feedback_moat_not_hard_lossy_memory_ok` the moat is a plus not a hard gate, but a *wiring* change must not be what
  breaks it — trade it only deliberately for the learned-lossy path, never as an integration side effect.
- **Host-shortcut check:** none introduced — every wired capability is already neural end-to-end in its de-risk (the
  attributed parse is in spikes; the resonator decode is the spiking resonate-and-fire ops; multi-frame is the spiking
  `BridgeParser`). The only host steps stay the environment (token string) + body (emit the ordered words).
- **Honest scope:** this is a **consolidation of EXISTING capabilities into one agent, not a new capability** (the
  same honest framing as roadmap step 2). That is exactly its value here — it is the cheapest, highest-leverage,
  lowest-risk move and it **localizes the next real gap** (you cannot tell which deeper mechanism is *felt* until the
  agent has all the validated ones at once).

### #2 — Multi-referent disambiguation via WTA biased-competition (the precisely-named NEGATIVE-fix)

- **What it is:** install **winner-take-all lateral inhibition between the held referent attractors + a salience
  bias** so an attended referent SUPPRESSES the others, letting a bare pronoun among several held referents resolve.
  This is the exact mechanism two converging NEGATIVEs named (`2026-06-17-multireferent-disambiguation-NEGATIVE.md`):
  NOT recency (NEGATIVE), NOT a salience boost alone (NEGATIVE) — biased competition (Desimone-Duncan 1995).
- **Why #2:** it is the **most precisely-specified open conversational mechanism** (the negatives did the scoping for
  us), it is genuinely point-neuron-feasible (lateral inhibition is a core motif the project already uses — FS-PV /
  MSN cross-pool inhibition), it directly extends multi-turn dialogue (a real felt gap when the agent holds >1
  referent), and it aligns with the owner steer (a WM-attention mechanism, not RAG). Cheap-first de-riskable on the
  `SpikingLoopContextBuffer` with the existing inhibition machinery.
- **Why not #1:** it is a real new build (install inhibitory cross-connections + a salience signal), so it is more
  expensive than integration; and whether multi-referent dialogue is *felt* is exactly what the #1 integration will
  reveal (single-referent multi-turn already works). Sequence it right after integration if multi-referent dialogue
  is prioritized.

### #3 — Embedded-clause PARSING from a flat token stream (the hard half of productive syntax)

- **What it is:** the parser builds the nested `Clause` structure (the composer already DECODES) from a flat token
  stream ("the dog that chased the cat ran") via a neural push/pop over the gamma-slot ordered-WM (the
  unification-space mechanism: PNAS 2025 rapid-neural-sequence nesting + Hagoort MUC). Behavioral target = the
  catalog **G.12** center-embedding signature.
- **Why #3:** highest *ceiling* (true recursion), strong catalog grounding, reuses `OrderedPositionWM` + the dlPFC
  NMDA latch + transmission gates — but it is a genuine **new control discipline** (open/close a constituent over WM),
  the highest-risk of the three, and most likely to surface an honest substrate boundary (depth-2 nesting already
  costs a seed in the *decode*; the *parse* compounds a per-level control decision). Correctly **last** — gated behind
  the cheaper wins so the arc banks capability early, and behind the integration that localizes whether deep recursion
  is the felt blocker. An honest NEGATIVE here is itself the deliverable (it maps where the point-neuron + gamma-slot
  WM substrate's syntactic depth ends — the G.12 boundary).

### #4 (lower priority for "converse") — Instruction → cognitive operation; In-context few-shot

- **Instruction→operation** (language selects which internal routine runs) is point-neuron-feasible (reuses the
  validated frame-selector WTA + transmission-gate routing + dlPFC Control) and *improves* biology-faithfulness
  (moves routine-dispatch onto neurons). A strong cheap parallel candidate, but less central to "converse with the
  agent" than syntactic flexibility. **In-context few-shot** is a real LLM capability but the least central to the
  owner's "converse" priority; lowest. Sequence both after #1-#3.

### Why NOT "bigger vocabulary" as the next lever

The owner named it; it is the **deferred cloud axis**, not a local mechanism gap. The stream-cortex mechanism already
scales in principle (more stream → wider familiarity gap; `2026-06-15`); 320→thousands needs a **corpus-grounded
taxonomy + a wall-clock training budget**, not a new mechanism. It is the right thing to do *for perceived breadth*,
but it is a data/compute axis to schedule for the cloud, not the cheapest-first *mechanism* lever this scoping asks
for. (And per `feedback_prioritize_orchestration_overhead`, the real-time wall is per-op latency, addressed
separately by the CUDA-graph arc — not vocabulary.)

---

## 4. THE CHEAPEST-FIRST FIRST STEP (for #1) + its anti-cheat

**Step (reuse-by-import, NO `sim/` edit, CPU/numpy probe → GPU regression):** wire the **attributed-entity parse +
the resonator nested-decode** into `OneBrainComposer.hear` / `query_patient` behind a single default-OFF flag
(`enable_richer_syntax`, mirroring `composer_kind` / `enable_neural_render`), reusing `AttributedBridgeParser`
(CYCLE 200) + `nested_composition_agent`'s `_resonator3` + depth model-selection. Then run the production
conversational matrix (`tests/test_one_brain_composer_agent.py` family) with a mix of **flat SVO + attributed ("dog
eat big red apple") + nested ("dog see (big cat)")** facts.

**Pre-registered GATE (FROZEN before data; the matrix is multi-seed/GPU per `feedback_6seed_validation`):**
- **GO:** on the consolidated agent — attributed round-trip ≥ 0.90 AND nested round-trip ≥ 0.90 AND **flat-SVO
  un-regressed** (== the current production accuracy) AND the **no-confab moat intact** (unstored attributed/nested
  cue → `None`, **0 false-accepts**), all on ≥5/6 seeds. ⇒ promote the flag toward default-on per the owner's
  consolidation directive; then add multi-frame comprehension the same way.
- **BOUNDARY:** the richer paths work in isolation but **interfere** with the flat path on the shared agent (flat-SVO
  regresses, or the depth model-selection mis-routes flat vs nested under the agent's bundle crosstalk) — a real
  partial result localizing *integration interference* (the agent's shared ensembles) as the next sub-problem, not a
  mechanism failure.
- **NEGATIVE:** the validated standalone capability does NOT survive co-residence in the production agent (e.g. the
  resonator decode that worked on clean phasor codes fails on the 320 stream-learned codes the production agent uses)
  — an honest, citable finding about the production codes' invertibility (echoing the documented "the resonator needs
  PHASOR codes, not the real-Hadamard 320 substrate" caveat — **a known risk, see §5**), which would itself be the
  deliverable and re-scope to "which capabilities the production code type can host."

**Anti-cheat controls (mandatory — a "success" without all of these is an artifact):**
1. **Flat-SVO non-regression** — the validated flat parse/recall stays at its current accuracy; assert on the same
   matrix (the richer paths must not degrade the old).
2. **The no-confab moat asserted intact THROUGHOUT** — abstention + 0 false-accepts on unstored attributed/nested/flat
   cues, before and after wiring; a regression here voids the result even if richer recall improves
   (`feedback_brain_based_only_standard`; trade the moat only deliberately, never as a wiring side effect).
3. **The richer path is NEURAL** — the attributed parse is the spiking `AttributedBridgeParser` firing (not a host
   adjacency rule); the nested decode is the spiking resonate-and-fire resonator (not a host factorization). Assert
   the production path uses the neural components, with host limited to environment (tokens) + body (emit).
4. **Held-out generalization + leakage assertion** — attributed/nested test fillers are novel combinations disjoint
   from training (assert disjointness in code); compare against a memorization floor (Fodor-Pylyshyn systematicity).
5. **Depth model-selection control** — flat facts must NOT be mis-decoded as nested and vice-versa (the cleanup-
   confidence / 2-factor-residual model-selection from the multi-modifier GO must hold under the production agent's
   bundle crosstalk).
6. **≥6 seeds (fractional ≥5/6 bar) on the decisive GPU regression**; the CPU smoke is the cheap-first gate.
7. **Frozen bars, no config-cranking; reuse the validated runners' own anti-cheat harnesses verbatim** (the
   attributed-entity GO's flat-only control; the resonator's single-shot control).

**Expected wall-clock:** the standalone capabilities each validated in minutes (CPU) / a GPU matrix run; the
integration is wiring + the existing matrix — hours, not weeks. **No cloud.**

---

## 5. HONEST RISK + the clear cheap-first GO vs NEGATIVE

**The biggest way #1 could disappoint — the production-code-type risk (the #1 risk).** The resonator nested-decode is
validated on **clean phasor FHRR codes**, and there is a documented, decisive finding that **the resonator does NOT
work on the real-Hadamard 320 substrate** (`_resonator_real320_probe`: 0.00 on real-Hadamard, 1.00 on phasor at the
identical D/M — because the dense real-Hadamard binding is not cleanly invertible). The production `OneBrainComposer`
is the **resonate-and-fire phasor** path (the production default since 2026-06-18), so this *should* be fine — the
resonator's home substrate. **But this must be the FIRST thing the cheap-first probe checks**: does the resonator
nested-decode survive on the production agent's actual stream-learned 320 codes (which are phasor-substrate but
LEARNED, not clean-random)? If the learned codes' correlation structure degrades the resonator's iterative
convergence, the integration would BOUNDARY/NEGATIVE on nested specifically (attributed + multi-frame, which don't
need the resonator's M^F factorization, would still GO). **Mitigation:** probe nested-decode on the production codes
FIRST, cheaply; if it fails, ship attributed + multi-frame integration (which are independent of the resonator) and
record the nested-on-learned-codes boundary as the deliverable.

**Second risk — integration interference (the BOUNDARY case).** Co-residing the richer parsers on the production
agent's shared role ensembles could let the richer paths perturb the flat path (the exact thing the flat-SVO
non-regression anti-cheat guards). This is a *localizable* engineering boundary (per-capability flags isolate it), not
a mechanism failure — and finding it is informative.

**Third risk — "integration is not a *capability*" (the scope-honesty risk).** #1 ships no NEW science; an observer
could undersell it as "just wiring." The honest framing (and the reason it is still #1): it is the **highest-leverage,
cheapest, lowest-risk** move, it is what `feedback_check_existing_sims_first` (Spaun) validates as the thing that makes
a pile of capabilities into a cognitive model, and **it is the necessary precondition for correctly choosing the next
DEEP lever** — you cannot tell whether multi-referent WTA (#2) or embedded-clause parsing (#3) is the *felt* blocker
until the agent has all the validated capabilities in one place.

**The clear three-state outcome (pre-registered, §4):** GO (richer facts round-trip on the consolidated agent, moat
intact, flat un-regressed, ≥5/6) → promote + add multi-frame; BOUNDARY (interference or depth mis-routing) → localize
the shared-ensemble sub-problem; NEGATIVE (a validated capability doesn't survive co-residence — most likely the
resonator on learned 320 codes) → ship the capabilities that DO survive + record the production-code boundary as the
deliverable. **The stop criterion:** report the three-state outcome after the pre-registered CPU probe → GPU matrix;
do NOT escalate a co-residence NEGATIVE into a config search — a clean boundary on the production codes IS the answer
and re-scopes to "which capabilities the production code type hosts."

---

## 6. SUMMARY (the return)

- **Honest target framing (1-2 sentences):** this is a structured VSA/FHRR composer on one spiking bridge (parse →
  bind → recall-with-a-moat → neurally-ordered emit), **categorically different from a free-form next-token LLM**;
  "basic-LLM-competitive" is the owner's shorthand for a capability target (converse fluently + flexibly over its
  known world, the biology-faithful way — the North star is the brain analogue, capabilities instrumental), NOT a
  literal GPT-match, and the agent is already *ahead* on grounding-from-conversation + abstention + moated multi-hop
  reasoning, and *behind* mainly on breadth (the deferred cloud axis) and on having its validated syntactic
  flexibility *available in one agent*.
- **The ranked #1 lever + why:** **INTEGRATE the validated-but-shelved richer capabilities (attributed entities +
  resonator nested-decode + multi-frame comprehension + neural render) into the ONE production agent
  (`OneBrainComposer` / `BrainConversationalAgent`), behind opt-in flags, moat re-asserted, flat-SVO un-regressed** —
  highest leverage (converts ~4 validated capabilities into felt ability), cheapest (reuse-by-import, no `sim/` edit,
  no new science), most aligned with `feedback_check_existing_sims_first` (Spaun runs its pieces in ONE model — the
  project has the pieces but not the one model) and the TRUE-ONE-BRAIN directive, and the correct sequencing move
  because it **localizes which deeper gap is actually felt.** The owner's named "F=3 resonator / K=5 two-attribute"
  candidate, examined, points straight at this: the resonator already RESOLVES nested/two-attribute DECODE (GO 6/6,
  spiking) — it is the *integration* that is missing, not the mechanism; the residual learned-generalizing-bundle is
  not a felt gap (the fixed ±1 primitive bundles 0.989; the dendrite route is a confirmed NEGATIVE).
- **The cheapest-first first step + its anti-cheat:** wire the attributed parse + resonator nested-decode into
  `OneBrainComposer.hear`/`query_patient` behind a default-OFF flag; run the production conversational matrix with
  flat + attributed + nested facts; **GO** = attributed ≥0.90 ∧ nested ≥0.90 ∧ flat un-regressed ∧ moat intact (0
  false-accepts), ≥5/6 seeds; anti-cheats = flat-SVO non-regression + moat-asserted-throughout + the path is NEURAL +
  held-out leakage-asserted + depth model-selection control + ≥6 seeds + frozen bars. **Probe the resonator on the
  production's learned 320 codes FIRST** (the documented real-Hadamard-vs-phasor risk).
- **Sequenced after #1 (the genuinely-OPEN mechanism frontiers, precisely specified):** **#2 multi-referent
  disambiguation via WTA biased-competition** (two converging NEGATIVEs named it — lateral inhibition between referent
  attractors + salience bias, point-neuron-feasible), then **#3 embedded-clause PARSING from a flat token stream** (the
  hard half of syntax — a neural push/pop over the gamma-slot WM, highest ceiling/risk, honest NEGATIVE = the G.12
  boundary deliverable). #4 instruction→operation + in-context are lower-priority cheap parallels. Bigger vocabulary is
  the deferred cloud axis, not a local mechanism lever.

---

### Catalog entries cited
**G.02** (active dendrites — for the ruled-out dendrite framing), **G.03** (feature binding — "missing"), **G.07**
(pre-SMA internally-generated sequences), **G.08** (PFC working memory / executive control), **G.10** (language as a
hierarchical symbolic system), **G.11** (dual-stream language), **G.12** (Broca's grammatical processing — the
center-embedding "the girl that the boy is chasing is tall" dissociation, the embedded-clause behavioral target),
**G.13** (Wernicke comprehension), **H.19** (premotor competitive queuing / serial order), **N.15** (theta-gamma
multiplexed cell-assembly buffer; Lisman-Idiart 1995 — the ordered-WM substrate), **N.19** (gamma binding-by-synchrony
as attention — the biased-competition substrate for #2). Catalog:
`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.

### Papers cited (links)
- Eliasmith, Stewart, Choo, Bekolay, DeWolf, Tang, Rasmussen — **Spaun: A Perception-Cognition-Action Model Using
  Spiking Neurons**, *Science* 2012 (2.3M-neuron model; syntax = binding semantic pointers; BG routing; order-encoded
  WM for question answering). https://compneuro.uwaterloo.ca/files/2012-Spaun.pdf ; the SPA:
  https://compneuro.uwaterloo.ca/research/spa/semantic-pointer-architecture.html — the `feedback_check_existing_sims_first`
  precedent: the pieces run in ONE model.
- Frady, Kent, Olshausen, Sommer — **Resonator Networks** (factoring VSA products by iterated unbind↔cleanup), *Neural
  Computation* 2020 — the validated nested/two-attribute DECODE mechanism.
- Kleyko/Frady — **Variable Binding for Sparse Distributed Representations** (arXiv:2009.06734) — binding =
  multiplication/coincidence (the dendrite framing, ruled out).
- Lisman & Idiart 1995, *Science* — theta-gamma multiplexed STM buffer (the ordered-WM substrate).
- Grossberg 1978; Bullock & Rhodes 2003 — competitive queuing for serial order (the neural render).
- Desimone & Duncan 1995, *Annu. Rev. Neurosci.* — biased-competition attention (the #2 multi-referent mechanism).
- Mitropolsky, Collins, Papadimitriou — **A Biologically Plausible Parser**, TACL 2021 (arXiv:2108.02189); Mitropolsky
  et al. — **Center-Embedding and Constituency in the Brain**, 2022 (arXiv:2206.13217) — the Assembly-Calculus account
  that learned ordered syntactic structure is point-neuron-realizable (the #3 embedded-clause grounding).
- *Building hierarchically nested structure by rapid neural sequences*, **PNAS** 2025 — the unification-space nesting
  mechanism (#3).

### Project files / findings reviewed (this pass, file-cited)
- **The pivot trigger (dendrite double-NEGATIVE):** `2026-06-19-dendritic-binding-toy-derisk.md` (learned dendritic
  two-attribute bind memorizes 0.168, < fixed FHRR 0.261; NEGATIVE), `2026-06-19-dendritic-binding-derisk-scoping.md`,
  `2026-06-19-dendrite-credit-assignment-derisk-scoping.md` (the credit-assignment NEGATIVE).
- **The validated-but-unintegrated capabilities (the #1 lever):** `2026-06-18-attributed-entity-parser-GO.md`,
  `2026-06-18-neural-attributed-endtoend-GO.md` (attributed entities, fully neural end-to-end GO 6/6, standalone),
  `2026-06-03-resonator-decoder-cheap-first-RESOLVES-the-multi-factor-nesting-decode-in-algebra.md` +
  `2026-06-03-multi-modifier-attribution-resonator-restarts-RESOLVES.md` (resonator nested/two-attribute decode,
  GO 6/6, algebra+spiking, in `nested_composition_agent`; the real-Hadamard-vs-phasor risk),
  `2026-06-18-multiframe-comprehension-GO.md` + `2026-06-18-frame-selection-GO.md` (multi-frame, GO 6/6, standalone),
  `2026-06-16-sentence-generation-serial-order-cheap-first-GO.md` (neural render).
- **The production agent's integration state (the load-bearing surprise):** `grep` of `one_brain_composer.py`
  (no resonator/attributed/multi-frame) vs `brain_conversational_agent.py` + `rf_phasor_composer.py` (which have
  them) — the validated capabilities are NOT in the production one-brain agent.
- **The boundaries with named fixes:** `2026-06-17-multireferent-disambiguation-NEGATIVE.md` (×2 NEGATIVE → WTA
  biased-competition, the #2 lever), `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`
  (fixed ±1 bundles 0.989; learned bind doesn't generalize), `2026-06-17-within-category-error-signature-NEGATIVE.md`.
- **The production resting point:** `2026-06-18-onebrain-320-scale-production-GO.md` (320-scale, moat 0 false-accepts),
  `2026-06-17-consolidated-320-production-conversation-GO.md`, `2026-06-17-multihop-query-chain-GO.md`,
  `2026-06-17-multiturn-anaphora-derisk-GO.md`.
- **The prior scoping docs re-scored:** `2026-06-17-conversational-architecture-to-basic-LLM-scoping.md`,
  `2026-06-17-capability-frontier-to-basic-LLM-scoping.md`, `2026-06-18-richer-syntax-conversational-frontier-scoping.md`.
- `CLAUDE.md` (conversational sections); `research/findings/AUTONOMOUS_STATE.md` (CYCLEs 199-249). Catalog:
  `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (clusters G, N, H).
