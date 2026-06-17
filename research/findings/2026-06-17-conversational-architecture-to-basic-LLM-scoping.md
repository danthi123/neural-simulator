# Conversational architecture → "compete with a basic LLM" — deep-research + catalog scoping (2026-06-17)

> **Read-only deep research. No code edited.** Produced before any build per the standing "deep research +
> catalog review FIRST" directive (CLAUDE.md; `feedback_deep_research_at_roadblocks`). The controller should
> trust-but-verify the load-bearing claims flagged **[VERIFY]** inline, then push + present before building.
>
> **Owner directive being scoped (2026-06-17):** the primary demo desire is to *converse with the artificial-life
> agent*, and to build the conversational ARCHITECTURE out enough to **at least compete with a *basic* LLM**
> (Phi-3-mini / Llama-3.2-3B class) under STRICT biological realism. Vocab/corpus scaling via cloud compute is
> DEFERRED — this is about the architectural MECHANISMS, not more data/compute on the current design. The agent
> converses IS the embodied agent (nav + conversation already on ONE spiking bridge), so keep one-brain in mind.

---

## 0. Goal & constraints (the bars every option below is judged against)

- **BRAIN-BASED-ONLY** (`feedback_brain_based_only_standard`): only neurons / synapses / their communication count
  as the brain doing the work; host code is legitimate ONLY for the environment (world state, sensory render) and
  the body (acting on motor output). A host formula (argmax, distance, an f-string word-orderer) is a documented
  SHORTCUT; an honest negative under strict biology is itself a deliverable.
- **The no-confab "moat" must NEVER weaken.** The agent ABSTAINS when it has no matching fact rather than
  hallucinating — the project's signature advantage over LLMs. Every de-risk asserts it intact.
- **Point-neuron substrate** (Izhikevich/HH/AdEx + resonate-and-fire). The dendritic rewrite is benched
  (months-scale). The project has repeatedly found point-neuron workarounds (local feedforward normalization for
  generalization; fixed structural binding primitives) rather than needing dendrites — **prefer
  point-neuron-feasible mechanisms.**
- **Reuse-by-import strongly preferred** over edits to protected `sim/` modules.
- **North star** (`project_actual_goal_artificial_life_brain_analogue`): the deliverable is a biology-faithful
  artificial-life brain, with biology-translatable insight. "Compete with a basic LLM" is the *capability target*;
  it must be reached by mechanism, not by abandoning biology. **Honest negatives under strict biology ARE the
  output.**

---

## 1. DIAGNOSIS — decompose "conversational ability" and locate the gap

The project's conversational stack is far more complete than a naive reading suggests. **What is already
built and substrate-validated** (read in full from CLAUDE.md + the cited findings):

- **The 320-concept STREAM cortex** (`_phaseB_onbridge_stream_*`, 2026-06-15): a cortex that LEARNS each word's
  meaning *from a conversation stream* by population-Hebbian co-occurrence + local feedforward normalization
  (PPMI-equivalent) + population coding — **generalizes across similar concepts WITHOUT a dendritic rewrite**,
  on the point-neuron substrate, multi-seed GO, moat intact. This is the headline result that retired the
  "decorrelate-then-bind fork" framing.
- **The FHRR composer** (`rf_phasor_composer.py`): role-filler binding in resonate-and-fire phasor neurons +
  complex synapses (Frady-Sommer 2019), opponency-free. Parse → store SVO facts → who/what recall → ABSTAIN →
  negate/yes-no → embedded clause → 1-attribute bind. The 320-stream codes drop into it (consolidated GO,
  `2026-06-17-consolidated-320-production-conversation-GO.md`).
- **Serial-order GENERATION** (`neural_serial_order_renderer.py`, `enable_neural_render`): word ORDER for a
  described SVO fact produced by a spiking competitive-queuing serial-order generator (Grossberg/Bullock-Rhodes;
  catalog G.07/H.19), NOT an f-string. De-risked GO 6/6 (`2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`).
- **Dialogue PLANNING** (`content_selection_spiking.py` `SpikingSpreadingController`, dlPFC `elaborate`):
  what-to-say-next chosen by a spiking spreading-activation Control over the agent's own association graph.
- **Working-memory LATCH** (NMDA bistability; `SpikingLoopContextBuffer`): holds discourse referents; the
  single-referent multi-turn agent resolves "it" across turns (`MultiTurnAgent`, multi-turn anaphora GO).
- **Multi-hop reasoning** (`composer.query_chain`): role-structured pointer-chase, moat at every hop, GO 3×3.
- **Nested composition** (resonator decoder, `nested_composition_agent.py`, 2026-06-03): a fact whose slot is
  itself a bound entity ("dog chase (big cat)"), decoded by a spiking resonator network — RESOLVED on the
  phasor substrate.
- **Binding-by-gating + sequencing** (`sim/bridge.py` transmission gate, 2026-06-03): thalamocortical dynamical
  gating (Logiaco-Abbott-Escola 2021) binds verb→motor by *which gate is open*, re-binds with zero weight
  change, and steps through an ordered plan (temporal variable binding) — SHIPPED.

**So the gap is NOT "the agent can't converse."** It is that the conversation is **fact-database-shaped**:
fixed SVO frames, single-fact answers, retrieval-and-abstain. A basic LLM is **fluent, open-domain, syntactically
flexible, and context-tracking**. The decomposition:

| Capability | Basic-LLM behaviour | Project status today | Precise architectural gap |
|---|---|---|---|
| **Word/concept meaning** | learned embeddings, generalize | ✅ **DONE** — stream cortex learns from conversation, generalizes (2026-06-15) | none — this is a genuine strength |
| **Store/recall facts** | implicit in weights | ✅ DONE — FHRR bind + who/what recall, multi-seed | none |
| **Trustworthy abstention** | ❌ LLMs confabulate | ✅✅ **DONE + the moat** (signature advantage) | none — protect, never weaken |
| **Single-fact answer** | yes | ✅ DONE | none |
| **Multi-hop reasoning** | yes (in-context) | ✅ DONE — `query_chain`, moat per hop | none |
| **Generate word-order** | yes | ✅ DONE — spiking CQ serial-order (`enable_neural_render`) | none for the SVO frame |
| **Multi-SENTENCE fluency / discourse** | yes — paragraphs, coherent | ⚠️ **single utterances**; `elaborate` picks ONE next associate | **no mechanism to emit an ordered MULTI-ITEM sequence of facts/clauses as connected output** |
| **General SYNTAX + recursion** | yes — arbitrary phrase structure | ⚠️ **fixed SVO frame + 1 embedded clause**; parser is (position×voice)→role | **no productive syntactic structure**: can't handle arbitrary constituency / center-embedding / multi-frame word order (catalog G.12's "the girl that the boy is chasing is tall" dissociation) |
| **Working memory / context window** | yes — long context | ⚠️ **single salient referent** held; multi-referent disambiguation is a mapped NEGATIVE | **WM holds a SET, not an ORDERED/ranked multi-item buffer**; no biased-competition selection among held items (`2026-06-17-multireferent-disambiguation-NEGATIVE.md`) |
| **In-context (few-shot) adaptation** | yes | ❌ absent | no mechanism to condition output on a few in-context exemplars without weight training |
| **Instruction-following** | yes | ⚠️ partial — `spoken_instruction_nav` steers the BODY (language→action GO); no general "do X" over conversation | language→action exists; language→*cognitive-operation* does not |
| **Topic flexibility / open response** | yes | ⚠️ frame-bound (SVO Q&A, elaborate, yes/no) | response surface is a fixed small frame inventory |
| **Fluent open-ended generation** | yes | ❌ **known WALL** (out of scope) | ~360M+ params + trillions of tokens; from-scratch spiking LM overfits at scale (SpikeGPT). Documented terminal. **Do not re-attempt.** |

**The single structural theme uniting the four real gaps (multi-sentence fluency, syntax/recursion, WM/context,
in-context adaptation): the project has solved *single-item* binding/recall/generation but lacks a
biologically-grounded mechanism for ORDERED MULTI-ITEM STRUCTURE held and manipulated in working memory** — a
sequence of role-slots within a sentence (syntax), a sequence of sentences across a turn (discourse), and a
ranked set of items competing for selection (attention/context). In the brain this is one mechanism family:
**theta-gamma multiplexing for ordered multi-item WM (Lisman-Idiart; catalog N.15) + binding-by-synchrony as
attention (catalog N.19) + a competitive-queuing/biased-competition read-out.** That convergence is the basis
for the ranking below.

### 1a. Critical historical nuance the controller must hold (so we don't re-tread a "terminus")

There is a **2026-05-17 META-TERMINUS** and a **2026-05-20 "5-architecture convergent ceiling"** declaring
theta-gamma / SPEAR / generative-replay conversational composition a string of honest negatives, AND a
2026-05-17 finding calling multi-hop reasoning a "corrected-NEGATIVE." **Both have since been overtaken by the
substrate pivot, and this is load-bearing:**

- Those negatives were all on the **OLD engram-tag / concept-pool RETRIEVAL substrate**, testing one specific
  *compositional-retrieval* task (cue a noun → emit a bound adjective). The whole conversational stack was
  later **rebuilt on the FHRR composer + stream cortex**, which dissolved those exact walls: multi-hop
  reasoning is now **production GO** (`2026-06-17-multihop-query-chain-GO.md`) — directly reversing the
  2026-05-17 "corrected-NEGATIVE."
- The theta-gamma *negative* was a **specific mechanism** ("cue-suppression-during-retrieve") that *hurt* — NOT
  theta-gamma per se. Theta-gamma *as ordered-sequence multiplexing* separately **PASSED at the algebra level**
  (`2026-05-24-direction-E-theta-gamma-multiplexing-ALGEBRA-VALIDATED-controls-decisive.md`: 1.000 at loads
  2/3/5/7, permutation + no-windowing controls decisive), and theta/gamma **synchronization works on the
  spiking substrate** (`2026-06-09-route-T-gamma-volley-RESULT.md`: FS-PING gamma re-times a sparse ensemble
  into a coincident volley, jitter anti-cheat PASSES).

⇒ **The opportunity is precisely to bring the algebra-validated theta-gamma ordered-sequence mechanism onto the
*new* FHRR/stream-cortex substrate as a working-memory + sequence engine — which the old-substrate negatives
never tested.** This is a genuinely-new build, not a config-crank of a dead path. (Flag this distinction
explicitly in any write-up so it is not mistaken for re-running the 5-architecture ceiling.) **[VERIFY: that the
theta-gamma negatives were substrate-specific (engram/pool) and that the algebra PASS + route-T volley stand —
both are in the cited findings.]**

---

## 2. RANKED biologically-grounded options (biggest lever first)

Bars: a candidate is preferred if it (i) closes the most gap, (ii) has direct catalog + literature grounding,
(iii) is point-neuron-feasible, (iv) reuses existing machinery, (v) needs no (or a minimal, additive) `sim/`
edit, and (vi) keeps the moat intact.

### Option 1 (RECOMMENDED — biggest single lever) — Theta-gamma ordered multi-item WM buffer as the shared sequence + attention engine

- **Mechanism (one paragraph):** Install a **theta carrier (~4–8 Hz) with nested gamma sub-cycles (~40–80 Hz)**
  over the conversational working-memory loop. Each item (a role-slot, a discourse referent, or a fact)
  occupies its own gamma sub-cycle; **ordinal position = theta phase**; the theta period sets the buffer span
  (~7±2 items, Miller's number). This is a *single* mechanism that simultaneously supplies (a) **ordered
  multi-item WM** (hold a ranked buffer, not just a set → fixes multi-referent disambiguation and the context
  window), (b) **sequence read-out** (replay the items in phase order → multi-slot syntax frames and
  multi-sentence discourse), and (c) **binding-by-synchrony as attention** (items in the same gamma cycle are
  co-bound/co-attended; offset items are segregated — catalog N.19). Adding the oscillators is "straightforward
  in the NM framework" per the catalog (sinusoidal `excitability_drive` at theta + a theta-phase-modulated
  faster modulator), so it can likely be done **reuse-by-import via the neuromodulator subsystem with no `sim/`
  edit** — the SPEAR runner already drives ACh phase-gating exactly this way (`spear_conversational_core.py`).
- **Biology source:** catalog **N.15** (theta-gamma cross-frequency coupling = multiplexed cell-assembly
  buffer; Lisman & Idiart 1995; Bz Cycle 12) + **N.19** (gamma binding-by-synchrony, ING/PING; the gamma cycle
  ≈ the STDP window) + **D.18** (theta sequences = temporal compression into the plasticity window) + **G.08**
  (PFC persistent-activity WM). Recent concrete spiking models: **Pirazzini, Ursino et al. 2023 *Neural
  Networks*** (PMC10050512 — "a model of working memory for encoding multiple items and ordered sequences
  exploiting the theta-gamma code"; reconstruct an item from partial cue, hold multiple items unordered, AND
  reconstruct an ordered sequence from a cue) and **Ursino et al. 2024 *Front. Neural Circuits*** (PMC11211613
  — theta-gamma coupling for sequential memory, imagination, dreaming). **The project already references the
  Ursino/Pirazzini line** (`pirazzini_three_layer_*`) — adopt the proven mechanism rather than reinvent
  (`feedback_check_existing_sims_first`).
- **Which gaps it closes:** multi-sentence fluency (ordered fact/clause sequences), WM/context window (ranked
  multi-item buffer), the multi-referent disambiguation NEGATIVE (theta phase + gamma segregation give the
  ranked salience the plain attractor loop lacks), and it is the *substrate* on which multi-frame syntax
  (Option 2) and in-context buffering (Option 3) sit.
- **Point-neuron difficulty:** **LOW–MEDIUM.** The oscillators are point-neuron-native (FS-PING gamma already
  emerges — route-T validated; theta is a sinusoidal drive). The known risk is **dt-bound phase resolution**:
  CLAUDE.md (one-bridge step 3) records that rank-order/latency coding ties at dt=1.0 when neighbours are
  equidistant. Mitigation: the WM/sequence engine can run at its own dt=0.5 (the dlPFC already does), and the
  algebra PASS used D=256 with biological-precision noise σ=0.05 and held 7 slots — so the headroom exists.
- **`sim/` edit needed?** **Probably NO** — the neuromodulator subsystem's `excitability_drive` +
  `synaptic_gain` targets (used by SPEAR) can impose a phase-modulated drive; the FS-PING gamma is intrinsic.
  If a clean phase clock proves easier as an additive guarded primitive, it mirrors the transmission-gate
  precedent (a small, default-off, byte-reviewed edit) — but try reuse-first.
- **Reusable machinery:** `spear_conversational_core/runner.py` (the theta-phase ACh controller — the rhythm
  plumbing), `content_selection_spiking.SpikingLoopContextBuffer` (the WM loop + clean-dynamics config), the
  resonate-and-fire neurons (**phase is ALREADY the representation** — directly compatible with phase-coded
  slots), `theta_gamma_mode_unification_core.py` (the frozen verdict instrument), `route-T` FS-PING gamma, the
  neuromodulator subsystem.

### Option 2 — Productive syntax: a learned sequence-detector / assembly-calculus parser+generator over multi-frame word order

- **Mechanism:** Replace the fixed (position×voice)→role parser and the single SVO output frame with **learned
  sequence detectors** — neural assemblies that fire to AB but not BA (Pulvermüller DCNAs), composed into a
  small inventory of *learned* syntactic frames, selected by the dlPFC, with **center-embedding handled by an
  assembly-calculus recursion mechanism** rather than a software stack. Generation then orders slots per the
  *selected frame's* learned order (multi-frame → real syntax, not one hard-coded SVO).
- **Biology source:** **Pulvermüller & Knoblauch 2009** (sequence detectors / discrete combinatorial neuronal
  assemblies); **Mitropolsky, Papadimitriou et al.** assembly-calculus parser + **"Center-Embedding and
  Constituency in the Brain"** (arXiv:2206.13217 — a biologically-plausible spiking account of context-free /
  center-embedded structure WITHOUT a stack); catalog **G.07** (pre-SMA internally-generated sequences),
  **G.10/G.12** (hierarchical syntax; Broca's grammatical processing — the catalog's own behavioral validation
  is the syntactic-complexity dissociation), **G.13** (Wernicke comprehension). A 2025 *Lang. Cogn. Neurosci.*
  neurobiologically-inspired sentence-comprehension model (lexicon+syntax+semantics modules) is a current
  reference point.
- **Which gaps it closes:** general syntax + recursion, topic/frame flexibility, and (via multi-frame order
  learning) the "real syntax vs fixed SVO" boundary CLAUDE.md flags as the open follow-on to the serial-order
  generator.
- **Point-neuron difficulty:** **MEDIUM–HIGH.** Sequence detectors are Hebbian-bindable on point neurons
  (AB-order pools), but productive recursion (arbitrary depth) is the genuinely hard part; assembly-calculus
  shows it is point-neuron-feasible *in principle*, but realizing it on this substrate at conversational scale
  is unproven. The single embedded clause already works via the resonator (nested composition); the open part
  is *arbitrary* / *multiple* embedding and learned per-frame order.
- **`sim/` edit needed?** Likely NO for the sequence-detector frames (Hebbian pools + transmission-gate routing
  exist); the recursion mechanism may need new runner machinery but not necessarily a `sim/` edit.
- **Reusable machinery:** the learned-binder (single-attribute, spike-validated), the transmission gate
  (binding-by-gating + temporal variable binding — already does "same verb bound to different slots in a
  sequence"), the CQ serial-order generator (the per-frame ordering read-out), the dlPFC frame selector, the
  resonator decoder (nested constituents).

### Option 3 — Working-memory / context expansion via biased-competition attention + hippocampal episodic integration

- **Mechanism:** Two coupled additions to the WM loop: (i) **winner-take-all biased competition** — install
  *inhibitory cross-connections between referent attractors* + a salience bias, so the attended item SUPPRESSES
  competitors (Desimone & Duncan 1995), the exact mechanism the multi-referent NEGATIVE pinned as missing; (ii)
  **hippocampal episodic integration** — use the validated trisynaptic loop + engram tagging + SWR/awake replay
  (catalog D.03/D.12/D.13/D.19/N.17) to extend the effective context beyond the active-WM span by paging
  discourse history in/out (the CLS division of labour the project already validates for no-forgetting).
- **Biology source:** Desimone & Duncan 1995 (biased competition); catalog **N.19** (gamma binding-by-synchrony
  as the attention substrate), **G.08** (PFC WM), **D.02** (relational binding — items-in-context),
  **N.17** (awake replay for online deliberation), **D.19** (SWR). The project's own NEGATIVE
  (`2026-06-17-multireferent-disambiguation-NEGATIVE.md`) already names this mechanism precisely.
- **Which gaps it closes:** multi-referent disambiguation, context window beyond the WM span, long-range
  coherence.
- **Point-neuron difficulty:** **LOW** for the biased-competition WTA (inhibitory cross-connections are a core
  primitive — MSN lateral inhibition / FS-PV feedforward inhibition motifs already exist); MEDIUM for the
  episodic-paging integration (the hippocampal machinery is built but wiring it as a *conversational* context
  store is new).
- **`sim/` edit needed?** NO — inhibitory cross-connections + engram API are reuse-by-import.
- **Reusable machinery:** `SpikingLoopContextBuffer`, the MSN/FS WTA motifs, engram tagging
  (`commit_engram_tag`/`stimulate_tag`), the trisynaptic loop builder, SWR/replay, `MultiTurnAgent`.
- **Note:** this is largely a *subset* of Option 1 (theta-gamma supplies ranked multi-item WM + binding-by-
  synchrony attention *for free*), which is why Option 1 ranks above it — but the **biased-competition WTA is
  the cheapest standalone win** and is a prerequisite either way, so it is the natural first concrete brick.

### Option 4 (lower priority for THIS goal) — Hybrid: VSA-retrieval + a small RA/distilled generator for open fluency

- **Mechanism:** Wire the validated 320-VSA grounded memory as the retrieval store for a small (≤360M)
  retrieval-augmented or distilled conventional generator, which produces fluent surface text conditioned on
  the grounded facts.
- **Biology source / field view:** the 2026-06-03 deep-research thread (RETRO-LI; BabyLlama; the AAAI "VSA is
  associative not generative → couple to a separate generator" consensus). **Honest:** the generator is
  conventional (non-spiking) — it **relaxes biology for the generator**, so under the BRAIN-BASED-ONLY +
  north-star framing it is an **engineering baseline, clearly-marked non-load-bearing**, not an artificial-life
  step.
- **Which gaps it closes:** fluent open-ended generation (the one capability that is otherwise a hard wall) +
  multi-sentence surface fluency — i.e. it is the *only* route to genuinely LLM-like fluency.
- **Point-neuron difficulty:** N/A (the generator isn't on the substrate).
- **`sim/` edit needed?** NO.
- **Recommendation:** keep as the **explicit owner fork** — if "compete with a basic LLM" means *surface
  fluency*, Option 4 is the only path and the owner must decide to relax biology for the generator. If it means
  *architectural breadth under biology* (the project's actual goal), Options 1–3 are the answer and Option 4 is
  a documented escape hatch. **Surface this decision; do not adopt unilaterally** (`project_actual_goal`
  escalation rule).

**Ranking rationale:** Option 1 (theta-gamma WM/sequence/attention engine) is the **single highest lever**
because one biologically-grounded mechanism closes three of the four real gaps at once (multi-sentence fluency,
WM/context, multi-referent attention) AND is the substrate the others sit on; it is point-neuron-native,
algebra-pre-validated on *this* project, reuse-heavy, and likely needs no `sim/` edit. Option 3's
biased-competition WTA is the cheapest standalone brick and a prerequisite (do it first as the de-risk).
Option 2 (productive syntax) is the deepest capability gain but highest-risk on point neurons — sequence it
*after* the WM/sequence engine gives it a buffer to order over. Option 4 is the only path to true surface
fluency but trades away biology — an owner fork, not a default.

---

## 3. REUSABLE MACHINERY (consolidated — the build is mostly assembly)

| Piece | File | Contributes to |
|---|---|---|
| Stream cortex (learned, generalizing codes) | `_phaseB_onbridge_stream_*`, `consolidated_320_conversation_demo.py` | the concept substrate for all options |
| FHRR composer (bind/unbind/bundle/cleanup + moat) | `rf_phasor_composer.py` | fact memory; phase IS the representation (theta-gamma-ready) |
| Resonate-and-fire neurons + complex synapses | `sim/bridge.py` (`NeuronModel.RESONATE_AND_FIRE`, `rf_*`) | phase-coded slots (Option 1); nested decode |
| Resonator decoder | `nested_composition_agent.py`, `_resonator_*` probes | recursion / nested constituents (Option 2) |
| Transmission gate (binding-by-gating + temporal variable binding) | `sim/bridge.py` (`set_transmission_gate`, `couple_gate_to_pool`), `gated_sequence_demo.py` | routing slots; sequence stepping (Options 1, 2) |
| dlPFC content-selection Control | `content_selection_spiking.py` (`SpikingSpreadingController`) | what-to-say-next; frame selection |
| Spiking loop WM buffer + NMDA latch | `content_selection_spiking.py` (`SpikingLoopContextBuffer`), `MultiTurnAgent` | the WM engine to add theta-gamma to (Options 1, 3) |
| CQ serial-order generator | `neural_serial_order_renderer.py`, `_phaseB_serial_order_*` | per-frame word ORDER read-out (Options 1, 2) |
| Theta-phase ACh rhythm controller | `spear_conversational_core/runner.py` | the rhythm plumbing for Option 1 (reuse, no `sim/` edit) |
| FS-PING gamma synchronizer | `coincidence_volley_n9_derisk.py` (route-T) | the gamma carrier (Option 1) |
| Frozen verdict instruments | `theta_gamma_mode_unification_core.py`, `spear_conversational_core.py`, `song_g1_core.py` | anti-cheat scoring (reuse verbatim) |
| Engram tagging + trisynaptic loop + SWR replay | `sim/bridge.py` engram API, `consolidation_trainer.py`, `validate_trisynaptic_loop.py` | episodic context paging (Option 3) |
| WTA / lateral-inhibition motifs | `g11_bg_runner.py` (`--enable-msn-lateral-inhibition` etc.), FS-PV feedforward | biased-competition attention (Option 3) |
| Learned single-attribute binder | `2026-06-16-onsubstrate-learned-binder-*` runners | sequence detectors (Option 2) |
| Surrogate-grad BPTT LIF (engineering-baseline only) | `sim/bptt_snn*.py`, `surrogate_grad.py` | a tiny seq2word net IF ever wanted — **clearly-marked non-load-bearing** |

---

## 4. RECOMMENDED #1 + its cheap-first de-risk

**Recommend Option 1 (theta-gamma ordered multi-item WM buffer).** But the **single cheapest decisive probe**
that gates it is the **biased-competition WTA piece (Option 3's prerequisite), which the multi-referent NEGATIVE
already pre-specified** — because the whole Option-1 value proposition rests on the WM loop being able to hold a
*ranked/ordered multi-item buffer where the attended item dominates*, and the plain loop provably cannot
(NEGATIVE: it holds a set; recency and salience-boost both failed; only WTA biased competition is left
untested). De-risk THAT first; if it passes, the theta-gamma ordering layer is the natural next build on a now-
ranked buffer.

**Cheap-first de-risk (CPU/numpy, reuse-by-import, no `sim/` edit, minutes):**

- **Setup:** extend `SpikingLoopContextBuffer` with **inhibitory cross-connections between the referent
  attractors** (each concept pattern inhibits the others, weight `w_inh`) + a **salience bias** current on the
  foregrounded referent. Write two referents (cat, then bird), then read which dominates. Reuse the EXACT
  protocol of `_phaseB_multireferent_disambiguation_derisk.py` (NATURAL / ORDER-CTRL / REFRESH conditions) so
  it is directly comparable to the recorded NEGATIVE.
- **Metric:** does the salient/foregrounded referent **dominate the read** (specificity > threshold) reliably,
  AND does the **ORDER-CONTROL flip the winner** (write bird-then-cat → cat dominates) — the discriminator the
  plain loop failed.
- **Pre-registered GATE (FIX before seeing data; ≥6 seeds per the project rule — `feedback_6seed_validation`):**
  - **GO:** the foregrounded referent dominates (specificity > 1.5) in NATURAL **AND** the ORDER-CONTROL flips
    the winner, on ≥5/6 seeds, with the competitor genuinely *suppressed* (its rate drops vs the no-inhibition
    baseline — proving suppression, not just out-driving). ⇒ build the theta-gamma ordering layer on the
    ranked buffer.
  - **PARTIAL:** dominates in NATURAL but the ORDER-CONTROL doesn't reliably flip (salience works, recency-order
    doesn't) — a real but incomplete win; localizes whether ordinal-position needs the theta-phase tag (→ go
    straight to the theta-gamma layer for the ordinal part).
  - **NEGATIVE:** WTA biased competition still can't make the attended item dominate+suppress reliably ⇒ a
    deeper WM-wiring boundary; record it (a third converging negative would be a strong biology-translatable
    finding about point-neuron WM selection) and reconsider before the larger theta-gamma build.
- **Expected wall-clock:** **minutes** on CPU (the multi-referent NEGATIVE ran 3 seeds CPU in minutes; 6 seeds
  + the inhibition sweep is still well under an hour). No GPU, no 5-bridge load.

**Why this is the right cheap-first:** it is the *minimal mechanism change* that the project's own NEGATIVE
pre-specified ("install inhibitory cross-connections + a salience bias, not a parameter tweak"), it reuses the
exact recorded protocol so the comparison is honest, it runs in minutes, and its three outcomes each cleanly
route the next move. A GO directly unblocks the ranked multi-item buffer that Option 1's theta-gamma layer needs;
a NEGATIVE is itself a deliverable. **The theta-gamma *algebra* is already validated (2026-05-24), so the
remaining risk is precisely whether the *spiking WM loop* can hold a ranked/segregated multi-item buffer — which
this probe tests at the cheapest possible point.**

---

## 5. ANTI-CHEAT CONTROLS (the de-risk needs all of these or a "success" is an artifact)

1. **The no-confab moat asserted intact THROUGHOUT.** Any WM/attention/sequence change must NOT let the agent
   answer when it should abstain. Run the standing abstention probe (unstored cue → `is None`) before and after;
   a ranked-WM that confabulates a referent when WM is empty/ambiguous is a regression even if disambiguation
   improves. The moat is the load-bearing bar (`feedback_brain_based_only_standard`).
2. **ORDER-CONTROL (the load-bearing discriminator).** Reuse `_phaseB_multireferent_disambiguation_derisk.py`'s
   ORDER-CTRL: writing the referents in the opposite order MUST flip the winner. Without it, "the salient item
   won" passes trivially whenever one item happens to have the stronger intrinsic attractor (the exact failure
   mode the recorded NEGATIVE diagnosed — `bird` won on seed 42 regardless of order or boost).
3. **Suppression check, not just dominance.** Verify the competitor's rate DROPS under biased competition (vs a
   no-inhibition baseline) — biased competition must *suppress*, not merely *out-drive* (the salience-boost-alone
   negative proved out-driving is insufficient). This separates real WTA from additive drive.
4. **Lesion control.** Remove the inhibitory cross-connections (or the salience bias) → disambiguation must
   collapse back to the recorded set-hold behaviour. If it "works" without the new mechanism, the effect isn't
   attributable to it.
5. **Permuted / scrambled-input control.** Scramble which pattern is the foregrounded referent → the read-out
   must NOT track the true foregrounded item (else the decoder is reading a fixed structural bias, not the
   salience signal). For the later theta-gamma sequence layer: a **permuted-ORDER control** (same item set,
   scrambled phase assignment) must drop ordered read-out to chance (the discriminator that validated the
   theta-gamma algebra and exposed the G1 generator failure — reuse `theta_gamma_mode_unification`'s
   no-slot-windowing control + `song_g1_core.permuted_order_controls`).
6. **≥6 seeds for any variable effect** (`feedback_6seed_validation`); CuPy for any decisive/heavy run, numpy
   only for the tiny cheap-first (`feedback_gpu_not_numpy`).
7. **Frozen bars, no config-cranking.** Pre-register the GO/PARTIAL/NEGATIVE thresholds before seeing held-out
   data; reuse the frozen verdict instruments verbatim; run all seeds to completion. **Explicitly flag that this
   is a NEW mechanism on the NEW (FHRR/stream) substrate, not a re-run of the 2026-05-20 5-architecture ceiling**
   — keep that distinction in the write-up so it is not mistaken for cranking a dead path.

---

## 6. Load-bearing claims the controller should trust-but-verify

1. **[VERIFY — most load-bearing]** The 2026-05-19→24 theta-gamma / SPEAR / Pirazzini conversational NEGATIVES
   were on the **OLD engram/concept-pool retrieval substrate** for a *specific* compositional-retrieval task,
   NOT a verdict on theta-gamma as a sequence mechanism — AND theta-gamma *ordered-sequence algebra* separately
   **PASSED** (`2026-05-24-direction-E-...ALGEBRA-VALIDATED`) and theta/gamma synchronization works on the
   spiking substrate (`2026-06-09-route-T-gamma-volley-RESULT`). If this is right, Option 1 is a genuinely-new
   build, not a re-tread. *(Read the three cited findings' headers; confirm the substrate + the specific
   mechanism that failed.)*
2. **[VERIFY]** Multi-hop reasoning is **production GO** (`2026-06-17-multihop-query-chain-GO.md`), reversing the
   2026-05-17 "corrected-NEGATIVE" — i.e. the substrate pivot really did dissolve the old walls. *(Confirms the
   general claim that the new substrate is the right place to retry sequence/WM mechanisms.)*
3. **[VERIFY]** The multi-referent disambiguation NEGATIVE pre-specifies **winner-take-all biased competition
   (inhibitory cross-connections + salience bias)** as the exact missing mechanism, and recency + salience-boost
   both failed — so the cheap-first de-risk is testing the one untested mechanism, not re-running a failure.
   *(Read `2026-06-17-multireferent-disambiguation-NEGATIVE.md` §"Follow-up" + "Where this leaves multi-turn".)*
4. **[VERIFY — feasibility]** The theta-gamma oscillators can be added **reuse-by-import via the neuromodulator
   subsystem (no `sim/` edit)** — the SPEAR runner already drives theta-phase ACh gating via
   `excitability_drive`/`synaptic_gain`, and FS-PING gamma is intrinsic (route-T). *(Confirm the
   `excitability_drive` target supports a per-step phase-modulated drive without a protected edit; if not, the
   additive default-off primitive route mirrors the transmission-gate precedent.)*
5. **[VERIFY — scope honesty]** Fluent OPEN-ENDED generation remains a hard wall (Option 4 / `~360M+ params`,
   SpikeGPT overfit-at-scale) and is OUT of scope for the brain-based path; "compete with a basic LLM" is
   interpreted here as **architectural breadth under biology** (multi-sentence, syntax, WM/context), with
   surface fluency available only via the explicitly-flagged engineering-baseline hybrid (Option 4) the owner
   must approve. *(This interpretation is load-bearing — confirm it matches the owner's intent before building;
   it is the one place the goal could be read two ways.)*
6. **[VERIFY — point-neuron risk]** Phase-coded ordinal resolution is **dt-bound** (CLAUDE.md one-bridge step 3:
   ties at dt=1.0). The mitigation (run the WM/sequence engine at dt=0.5; the algebra held 7 slots at D=256
   σ=0.05) is plausible but unproven on the merged bridge — the theta-gamma build's first internal check should
   be "how many ordered slots survive at the merged-bridge dt."

---

### Catalog entries cited
N.15 (theta-gamma multiplexed buffer), N.16/N.17 (intrinsic + awake SWR), N.18 (NREM nesting), N.19 (gamma
binding-by-synchrony), D.02 (relational binding), D.03/D.05/D.13 (trisynaptic / CA3 sequential autoassociator),
D.18 (theta sequences / compression), D.19 (SWR replay), G.07 (pre-SMA internally-generated sequences), G.08 (PFC
working memory), G.10/G.11/G.12/G.13 (hierarchical syntax / dual-stream / Broca / Wernicke), H.19 (premotor
sequential action).

### Papers cited (links)
- Lisman & Idiart 1995, *Science* — theta-gamma multiplexed STM buffer. Lisman & Jensen 2013, *Neuron* "The
  θ–γ neural code."
- Pirazzini & Ursino 2023, *Neural Networks* — WM model for multiple items + ordered sequences via theta-gamma:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10050512/ ; Ursino et al. 2024, *Front. Neural Circuits*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11211613/
- Desimone & Duncan 1995, *Annu. Rev. Neurosci.* — biased-competition attention.
- Pulvermüller & Knoblauch 2009, *Neural Networks* — sequence detectors / discrete combinatorial neuronal
  assemblies for word order.
- Mitropolsky, Papadimitriou et al. — assembly-calculus parser; "Center-Embedding and Constituency in the
  Brain": https://arxiv.org/pdf/2206.13217
- Bullock & Rhodes 2003 — competitive queuing for serial order (the validated CQ generator's basis).
- Logiaco, Abbott & Escola 2021, *Cell Reports* — thalamocortical dynamical gating (the shipped transmission
  gate).
- Frady, Kent, Olshausen & Sommer 2020, *Neural Computation* — resonator networks (the shipped nested decoder).
- 2025 neurobiologically-inspired sentence-comprehension model (lexicon+syntax+semantics):
  https://www.tandfonline.com/doi/full/10.1080/23273798.2025.2473537
- (engineering-baseline reference) Zhu et al. 2023 (SpikeGPT, overfit-at-scale); Eldan & Li 2023 (TinyStories).

### Project files / findings reviewed
`CLAUDE.md` (conversational sections); `rf_phasor_composer.py`, `brain_conversational_agent.py`,
`multi_turn_agent.py`, `content_selection_spiking.py`, `neural_serial_order_renderer.py`,
`spear_conversational_{core,runner}.py`, `theta_gamma_mode_unification_core.py`, `sim/compose_temporal_bind.py`;
findings: `2026-06-17-{consolidated-320-production-conversation-GO, multireferent-disambiguation-NEGATIVE,
multihop-query-chain-GO}.md`, `2026-06-16-sentence-generation-{biologization-deep-research,serial-order-cheap-first-GO}.md`,
`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`,
`2026-06-11-cortex-build-plan-decorrelate-then-bind.md`,
`2026-06-03-{deep-research-how-the-field-gets-past-our-generative-conversation-wall, resonator-decoder-cheap-first-RESOLVES, thalamocortical-gating-solves-compose-binding-SHIPPED}.md`,
`2026-06-09-route-T-gamma-volley-RESULT.md`,
`2026-05-{17-conversational-capability-program-META-TERMINUS, 20-THETA-GAMMA-decisive-honest-negative, 20-Pirazzini-decisive-honest-negative, 24-direction-E-theta-gamma-multiplexing-ALGEBRA-VALIDATED}.md`.
Catalog: `sim-catalog/references/feature-catalog.md`.
