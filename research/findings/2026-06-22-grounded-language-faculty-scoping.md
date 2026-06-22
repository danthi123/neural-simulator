# Grounded language faculty for the spiking brain — 3-piece architecture SCOPING (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings/literature scoping for an owner-directed architecture — a
> GROUNDED LANGUAGE FACULTY that gives the spiking brain fluent natural-language output while the BRAIN (not a
> transformer) holds the knowledge and gates what is asserted. **NO `sim/` edits, NO training, NO GPU.** Single
> deliverable = this doc. Every load-bearing project API re-verified against the repo (file:line). SOTA bounded by a
> fresh June-2026 literature pass. Builds on the Gen-F CONVERT scoping (`2026-06-22-genseq-convert-scoping.md`) +
> step-0 GO (`2026-06-22-genseq-step0-C1-consolidation-GO.md`); does NOT re-derive them. The controller should
> trust-but-verify the **[VERIFY]** items, then push + present before building. This is a SCOPING/DECISION doc, NOT a
> brain-based result and NOT a commitment to build.

---

## 0. One-paragraph answer (the rest is the evidence)

The owner's reframe is **architecturally sound and unusually cheap to start** because it decouples three things the
project already has separate, validated machinery for: **(P1) FLUENCY** — a tiny off-the-shelf LLM converted to spikes
by the SAME training-free ANN→SNN mechanism the Gen-F convert is de-risking; **(P2) KNOWLEDGE** — taught OFFLINE by a
rich model (Claude itself can author the curriculum in-session) and re-encoded the brain's OWN way through the
already-shipped parser → composer → engram → hippocampal-consolidation pipeline; **(P3) GROUNDING** — the brain's
structured stores (the composer's exact-match recall + the no-confab abstention moat + the Bogacz-Brown familiarity
gate) decide what is KNOWN and gate what the fluent faculty is allowed to assert. The single most important technical
finding for P1: **every modern fluent SLM (SmolLM2, Qwen2.5, Llama-3.2, Gemma-2) is LLaMA-family — RMSNorm + RoPE +
SwiGLU/SiLU + GQA — NOT the vanilla LayerNorm+GELU GPT that Gen-F is**, so the MBE/LAS/ECMT methods Gen-F relies on do
NOT directly cover them. BUT the 2026 SOTA closed exactly this gap: **Plug-and-Play Spiking Operators** (decomposes
RMSNorm + SiLU + Softmax, training-free, <1% loss at T∈{1,2,4} on real LLaMA-2/3 and Qwen3) and **NEXUS** (bit-exact
FP32 gate-circuits incl. RMSNorm/SiLU/RoPE/Softmax, perplexity IDENTICAL to ANN). So the fluent-faculty convert is a
SOLVED-class problem even for the LLaMA stack — the residual is engineering + cost, not research. **The cheapest, do-it-
now de-risk is P2** (Claude writes a ~30-fact structured curriculum this session → the brain learns it via the parser
+ composer → structured recall + no-confab check), because it needs ZERO model download, ZERO conversion, and ZERO GPU
beyond a normal composer run — it validates the WHOLE knowledge-and-grounding half before any LLM is touched. The
recommended PRIMARY fluent candidate is **Qwen2.5-0.5B-Instruct** (genuinely fluent, Apache-2.0, the canonical LLaMA-
stack the SOTA converts; SmolLM2-360M is the smaller Apache fallback). The explicit cloud-trigger is the LARGER fluent
convert (a ≥1.5B model's spiking forward at T≥16 over a long context) and/or a large teaching corpus — both are the
moment the 3090's per-step×matmul cost or the curriculum-generation volume exceeds local feasibility. **P2 ≠ the
DEPRECATED Path 3**: Path 3 had an external LLM do the RUNTIME cognition (`bridge_memory.py`, "no external LLM ever" at
runtime — `CLAUDE.md:1308-1310`); P2's rich model is an OFFLINE textbook-author that never runs at inference, so the
standalone-agent stance is preserved.

---

## 1. P1 — the FLUENT FACULTY (fluency only, so it can be small)

### 1a. The owner's key insight, restated precisely

Because the BRAIN holds the knowledge (P2) and gates assertions (P3), the converted LLM only has to be **fluent** —
produce grammatical, coherent surface form — NOT knowledgeable. That decouples *fluency* (a function of language
modeling, achievable at <1B params, even <10M for the TinyStories regime) from *world knowledge* (a function of scale,
which we explicitly do NOT need from the faculty). This is the same observation the TinyStories paper makes (Eldan & Li,
arXiv 2305.07759): **<10M params already yields coherent multi-paragraph English** when the model only has to be
fluent over a constrained domain. The project's own Gen-F is a 3.45M-param GPT at held-out ppl 6.1 producing coherent
story-shaped English (`2026-06-22-genseq-step0-C1-consolidation-GO.md:19-21`) — an existence proof that fluency is
cheap. The faculty is "Gen-F, but a real off-the-shelf model with broader grammatical coverage."

### 1b. SOTA tiny fluent open-weights LLMs (2024–2025 cohort + the 2026 frontier)

| Model | Params | License | Tokenizer / vocab | Architecture | Fluency (coherent multi-sentence?) |
|---|---|---|---|---|---|
| **SmolLM2-135M** | 135M | **Apache-2.0** | SmolLM tokenizer, **vocab 49152** | LLaMA-family: **RMSNorm + RoPE + GQA (9h/3kv) + SwiGLU**, tied emb, 30 layers, d=576, ctx 2048 | Marginal — terse; usable for very short grammatical spans |
| **SmolLM2-360M** | 360M | **Apache-2.0** | same, vocab 49152 | same LLaMA-family, depth-over-width | **Good** — coherent short paragraphs; smallest comfortably-fluent Apache option |
| **SmolLM2-1.7B(-Instruct)** | 1.7B | **Apache-2.0** | vocab 49152 | LLaMA-family | **Strong** — instruction-following, multi-sentence coherent |
| **Qwen2.5-0.5B(-Instruct)** | 0.5B | **Apache-2.0** | Qwen BPE, **vocab ~151,643** | LLaMA-family: **RMSNorm + RoPE(base 1e6) + SwiGLU + GQA**, **embedding tying** (small models tie; large don't), ctx 32K | **Good–strong** — the recommended primary; fluent, multilingual headroom |
| **Qwen2.5-1.5B(-Instruct)** | 1.5B | Apache-2.0 | vocab ~151,643 | same | Strong |
| **TinyLlama-1.1B-Chat** | 1.1B | **Apache-2.0** (Llama-2 *architecture*, open weights) | Llama-2 BPE, **vocab 32000** | Llama-2: RMSNorm + RoPE + SwiGLU + (MHA, no GQA at 1.1B) | Good — strong commonsense for size; older (2023) |
| **Llama-3.2-1B(-Instruct)** | 1B | **Llama 3.2 Community License** (NOT OSI; redistribution/usage terms) | Llama-3 BPE, **vocab 128256** | LLaMA-family: RMSNorm + RoPE + SwiGLU + GQA | Strong — but the license is the caveat |
| **Gemma-2-2B(-it)** | 2B | **Gemma license** (open-weights, custom terms) | Gemma SentencePiece, **vocab 256000** | RMSNorm (pre+post) + RoPE + GeGLU + GQA + logit soft-cap | Strong |
| **Phi-3-mini** | 3.8B | **MIT** | Llama-2 BPE, vocab 32064 | LLaMA-family + blocksparse attn in some variants | Strong (but 3.8B = a real convert + memory step up) |
| **OLMo-1B / 2-1B** | 1–1.2B | **Apache-2.0** (fully-open incl. data) | GPT-NeoX BPE, vocab ~50280 | LLaMA-family (RMSNorm/RoPE/SwiGLU) | Decent; the most *open* (data+code) option |

**2026 frontier (note, not primary):** SmolLM3-3B (Apache-2.0, outperforms Llama-3.2-3B/Qwen2.5-3B), Phi-4-mini-3.8B
(MIT), Gemma-3n. These are *more fluent* but *larger* — relevant only if the 0.5–1.7B tier proves insufficiently
fluent for the faculty role (unlikely, given the brain supplies content).

### 1c. The decisive convert-difficulty finding — modern SLMs are LLaMA-family, NOT vanilla GPT

**[load-bearing]** The Gen-F convert scoping established that Gen-F is a *vanilla* GPT (`nn.MultiheadAttention` +
**LayerNorm** + **GELU** + learned absolute positions; `tiny_transformer.py:11-78`), and that MBE/LAS/ECMT convert
exactly those three ops training-free. **But every fluent off-the-shelf SLM in §1b uses a DIFFERENT nonlinear stack:**

| Op | Vanilla GPT (Gen-F) | LLaMA-family SLM (SmolLM2/Qwen2.5/Llama-3.2/Gemma-2) |
|---|---|---|
| Norm | **LayerNorm** (mean + variance) | **RMSNorm** (ℓ₂ norm, no mean-centering) |
| Activation | **GELU** | **SiLU/Swish** inside **SwiGLU** (gated: `SiLU(xW)⊙(xV)`) |
| Positions | learned absolute `nn.Embedding` | **RoPE** (rotary, applied inside attention to Q/K) |
| Attention | MHA | **GQA** (grouped-query; fewer KV heads) |
| Attention nonlin | Softmax | Softmax (same) |

So MBE/LAS/ECMT (which handle Softmax/LayerNorm/GELU) do **not** directly cover RMSNorm / SiLU-SwiGLU / RoPE. This is
the honest gap the owner's "scale up the Gen-F mechanism" framing must clear. **It IS cleared by the 2026 SOTA:**

- **Plug-and-Play Spiking Operators** (arXiv 2605.20289): decomposes Transformer nonlinearities into **three
  primitives — division, exponentiation, ℓ₂ norms** — which compose **Softmax, SiLU, and RMSNorm**, each realized by
  LIF gate primitives. **Training-free** (no weight modification), **<1% accuracy drop at T∈{1,2,4}**, tested on
  **real LLaMA-2-7B, LLaMA-3-8B/70B, Mistral-7B, Qwen3-8B**. (Does not explicitly cover RoPE/GELU in the main paper —
  but RoPE is a *fixed trigonometric rotation of Q/K*, applied as a deterministic linear-ish op, not a learned
  nonlinearity; NEXUS covers it explicitly, below.) [VERIFY — code release not stated in the abstract this pass.]
- **NEXUS** (arXiv 2601.21279): **bit-exact** ANN→SNN via IEEE-754 FP32 gate circuits — explicitly implements
  **RMSNorm, SiLU/SwiGLU, RoPE (polynomial trig), and numerically-stable Softmax**, **perplexity IDENTICAL to the
  FP32 ANN** (WikiText-2 LLaMA-2-7B 5.12±0.02 = the ANN baseline), up to LLaMA-2-70B. Cost: a fixed 32-step FP32 bit
  window + ~3–4k neurons per FP arithmetic block (heavy, but *exact*); NOT training-free (surrogate-free STE, which is
  an exact identity because the forward is exact). [VERIFY — code release not stated this pass.]

**⇒ The fluent-faculty convert is a SOLVED-CLASS problem even for the LLaMA stack.** The choice is a fidelity/cost
trade: **training-free approximate (Plug-and-Play class, T∈{1,2,4}, <1% loss)** for cheap-first, escalating to
**bit-exact (NEXUS class, T≈32, exact ppl)** if approximate sampling degrades. The Gen-F convert (MBE/LAS on a vanilla
GPT) remains the cheapest *first* convert to validate the whole pipeline, then the off-the-shelf LLaMA-stack model is
the production faculty using the Plug-and-Play/NEXUS operators.

### 1d. How small can we go and still be genuinely fluent?

- **Floor for *constrained-domain* fluency:** ~3–10M (TinyStories regime; Gen-F's 3.45M is here). Coherent but
  *domain-bounded* and grammatically thin — fine as a faculty IF the brain's knowledge is the only content source and
  the domain is narrow. Risk: too thin for open who/what dialogue surface forms.
- **Floor for *broad grammatical* fluency:** ~**350M–500M** (SmolLM2-360M / Qwen2.5-0.5B). This is the recommended
  faculty tier — genuinely fluent across general English, still small enough to convert + run a spiking forward
  locally at low T.
- **Comfortable:** 1–1.7B (SmolLM2-1.7B / Qwen2.5-1.5B) — strongest fluency, but the spiking-forward cost roughly
  triples vs 0.5B (see §1e), pushing toward the cloud-trigger.

### 1e. Local-vs-cloud — RTX-3090 (24 GB) cost of a spiking forward

A spiking forward replays the dense matmuls **T times** (one per timestep), so the cost is ≈ `T × (ANN forward FLOPs)`
plus the per-op gate-circuit overhead. Order-of-magnitude for one token's forward (per layer dominated by the
attention + MLP matmuls), **weights in FP16 on the 3090**:

| Model | Params | FP16 weights | ANN fwd (1 tok) | Spiking fwd @ T=16 (approx, Plug-and-Play class) | Spiking fwd @ T≈32 (bit-exact, NEXUS class) | Local on 3090? |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | ~1 GB | ~1× | ~16× the ANN fwd | ~32× + heavy gate-neuron overhead | **Yes** (T≤16 comfortably; weights + activations fit with room) |
| SmolLM2-1.7B / Qwen2.5-1.5B | 1.5–1.7B | ~3–3.4 GB | ~3× | ~48× | ~96× + overhead | **Yes at T≤16** but tight; bit-exact T=32 starts to strain |
| Phi-3-mini / 3B-class | 3–3.8B | ~6–7.6 GB | ~6× | ~96× | ~192× + overhead | **Borderline** — fits in memory but the T× compute + gate overhead pushes wall-clock toward cloud |

Key caveats: (a) the **approximate** path (T∈{1,2,4} in Plug-and-Play) is dramatically cheaper than the assumed T=16 —
if it holds for generation it makes even 1.5B local at ~3–6× the ANN forward; (b) the **bit-exact** path's ~3–4k
neurons-per-FP-op overhead is the real memory/compute multiplier and is the most likely cloud-trigger; (c) the de-risk
(§4) runs the spiking forward **in PyTorch off the bridge**, so `SIM_BACKEND` is irrelevant at the de-risk stage —
bridge co-residence is the *later* consolidation step.

### 1f. P1 recommendation

**PRIMARY: Qwen2.5-0.5B-Instruct** — genuinely fluent, **Apache-2.0** (clean to convert + redistribute), the canonical
LLaMA-stack the 2026 SOTA converts, embedding-tied (fewer params to convert), runs locally at T≤16. **FALLBACK (smaller,
still Apache): SmolLM2-360M** if 0.5B's spiking forward is too costly or its vocab (151k) inflates the head-convert.
**AVOID as primary: Llama-3.2-1B / Gemma-2-2B** (open-weights but custom non-OSI licenses — usable, but Apache is
cleaner for a "convert + redistribute as part of the agent" stance). The faculty is the off-the-shelf model + the
Plug-and-Play/NEXUS operators; Gen-F stays the *pipeline-validation* convert.

---

## 2. P2 — the KNOWLEDGE TEACHER → brain re-encoding (verified against the project APIs)

### 2a. The pipeline (a rich model authors an OFFLINE curriculum; the brain learns it ITS OWN way)

The owner's P2: a large knowledgeable model (or **Claude itself**, in-session) generates an **offline teaching
curriculum** — facts / SVO statements / explanations — and the brain ingests them through its EXISTING, validated
machinery, producing real brain-structured knowledge with the no-confab moat. The end-to-end re-encoding path, every
stage verified in the repo:

```
  Claude / rich LLM (OFFLINE, authoring time)
        │  emits a structured curriculum (SVO facts + a word-by-word co-occurrence stream)
        ▼
  [PARSER]   BridgeParser / multicue parser — comprehends each sentence → {agent, action, patient}
        │    (Hebbian (word-position×voice)→role, voice-invariant; vocabulary-agnostic)
        ▼
  [COMPOSER] RFPhasorComposer / OneBrainComposer — binds the roles into a fact, appends to the
        │    spiking store (complex synapses); query/abstain/negate/clauses/multi-hop
        ▼
  [ENGRAM]   bridge.start_engram_recording / commit_engram_tag — Tonegawa ensemble tags for
        │    concept-concept associative memory (the multitag retrieval mechanism)
        ▼
  [CONSOLIDATION] consolidation_trainer.run_consolidation_training — awake/sleep SWR replay →
        │    hippocampus→cortex transfer, no catastrophic forgetting (Phase 1.3/1.4, CLS theory)
        ▼
  [STREAM CORTEX] PPMI online co-occurrence — the cortex HEARS the curriculum word-by-word,
             learns generalizing concept codes by rate-Hebbian co-occurrence (NO preprocessing)
```

### 2b. The APIs, verified (file:line — trust-but-verify the load-bearing claims)

| Stage | API | Verified location | What it does (verified from code/docstring) |
|---|---|---|---|
| **Parser** | `BridgeParser` (+ `.parse(words, voice)`) | `research/runners/brain_conversational_agent.py:28-143` | 6 conjunction units → 3 role ensembles on a `SimulationBridge`; Hebbian co-firing rule (`enable_hebbian_learning=True`, `:73`); ground-truth role map `_GT` (`:25`); voice-invariant; **vocabulary-agnostic** — assigns role by word-position×voice, so any vocab works (`:139-143,157`). |
| **Composer (rf)** | `RFPhasorComposer.store/query_patient/query_agent/ask_yes_no/render_fact/query_chain` | `research/runners/rf_phasor_composer.py:432,568,579,603,618,632` | role-filler FHRR bind/unbind in resonate-and-fire phasor neurons + complex synapses; SVO fact store (`store:432`); who/what Q&A; **negation** via bound AFFIRM/NEGATE polarity (`ask_yes_no:618`); recursive **clauses** + multi-hop **query_chain** (`:603`). |
| **Composer (one-brain)** | `OneBrainComposer` (API-sibling) | `research/runners/one_brain_composer.py:107-118,754,769` | the WHOLE who/what pipeline on ONE persistent co-resident bridge (parser + RF registers + persistent store + cleanup), no host round-trips; `grounded_codes=` drop-in (`:114`). |
| **Agent wiring** | `BrainConversationalAgent(composer_kind=…)` | `research/runners/brain_conversational_agent.py:146-205` | comprehend (parser) → store/recall/compose (composer); `composer_kind` ∈ {`rf`(default), `onebrain`, `rate`} (`:160-199`); delegates fact storage/retrieval to the composer; `concepts=`/`grounded_codes=` plumbed through (`:157,195,202`). |
| **Engram tagging** | `bridge.start_engram_recording / commit_engram_tag / stimulate_tag / list_engram_tags` | `sim/bridge.py:3352,3381,3466,3510` | Tonegawa (catalog D.14): accumulate per-neuron spike counts over an encoding window (`:3352`), commit a top-K or threshold-Hz ensemble tag with optional `region_filter` (`:3381-3464`), causally recall by `stimulate_tag` (`:3466`). Auto-ticked per step, zero overhead when idle (`:3368-3379`). |
| **Consolidation** | `consolidation_trainer.run_consolidation_training` (+ `run_swr_replay_phase`, `run_concept_replay_phase`) | `research/runners/consolidation_trainer.py:206,154,43` | awake/sleep loop (`:206`); awake = encoding ON, sleep = SWR replay drives CA3 (`:154-204`), `consolidation_interval` awake-events-per-sleep (`:210`); builds the hippocampus-enabled bridge (`enable_hippocampus_consolidation=True`, `:275`). Phase 1.3/1.4 CLS — cortex retains binding after hippo silence (CLAUDE.md, 3/3 anti-cheat multi-seed). |
| **PPMI stream cortex** | rate-Hebbian co-occurrence learning on-bridge | `research/runners/_phaseB_onbridge_stream_cortex_derisk.py` (+ `_phaseB_online_stream_cortex_derisk.py`); finding `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` | a cortex that HEARS the corpus word-by-word: online rate-Hebbian co-occurrence (`corr(M,C)` → +0.705, STDP measured-NEGATIVE because co-occurrence is symmetric/`Δt≈0`) + population code (47%→~94–108% of host ref). Produces **generalizing** concept codes the composer then binds. |

### 2c. What curriculum FORMAT does the brain ingest best?

The repo's machinery dictates the answer — **two complementary formats**, both of which a rich model emits trivially:

1. **SVO fact triples** (`agent action patient`, optional polarity, optional 1 attribute) → the **parser + composer**
   path. This is the *fact-knowledge* channel: each triple becomes a stored, queryable, abstaining fact. Format the
   curriculum as short canonical-order SVO sentences ("dog eat meat", "cat chase mouse", "apple is red"). The composer's
   validated scope: flat SVO + **single-attribute** entities ("big apple" — the 2-factor path, 100% on the 320 learned
   codes); 2-attribute is the documented K=5 boundary; recursive clauses + multi-hop chains are PRODUCTION
   (`query_chain`, `rf_phasor_composer.py:603`). So the curriculum should be **flat SVO with at most one adjective per
   noun**, optionally with **embedded-clause facts** ("dog know [cat eat mouse]") and **chains** for multi-hop reasoning.
2. **A word-by-word co-occurrence STREAM** → the **PPMI stream cortex** path. This is the *meaning/generalization*
   channel: the cortex learns *which concepts are similar* (so "dog"/"cat" generalize) by hearing a running stream where
   co-occurring words fire together. Format: a flat token stream where related concepts co-occur (the rich model can
   generate a topically-coherent corpus — "the dog ran. the dog ate meat. the cat chased the dog." — and the stream
   cortex extracts the co-occurrence structure online, no preprocessing). This is the channel that gives the *codes*
   the composer binds (the `grounded_codes=` / `concepts=` the agent passes through).

**Graded difficulty** matches the consolidation curriculum pattern already validated (the V_SCHEMA / Tse-2007 schema
work, and the Tier-ladder vocab scaling). Order the curriculum: (i) a base co-occurrence stream to seed the cortex's
concept codes; (ii) SVO facts over those concepts (parser+composer store); (iii) awake/sleep consolidation interleaved
so cortex retains without forgetting (`run_consolidation_training`); (iv) harder constructs (attributes, clauses,
chains) last. This is exactly the project's own "learn word meanings by listening → converse using them" loop, which
the consolidated-320 demo already closes on the stream-learned codes (CLAUDE.md, `2026-06-17-consolidated-320-...GO.md`).

### 2d. P2 ≠ DEPRECATED Path 3 — the consistency check, explicit

**This is the load-bearing distinction the owner asked to confirm.** The DEPRECATED "Path 3" (`CLAUDE.md:1099-1112,
1308-1310, 2420-2423`) had an **external LLM perform the RUNTIME cognition** — the MockLLM/orchestrator
(`sim/llm_memory_orchestrator.py`) drove tool-use, the LLM did the language *and* the reasoning, and the sim was a
bolted-on memory subsystem the LLM *called*. The user then clarified the goal is "sim as a standalone agent, **no
external LLM ever**" (`CLAUDE.md:1308`), and Phase 3.3 (real-LLM swap-in at runtime) is DEPRECATED for the primary
path.

**P2 is categorically different and consistent:**

| Axis | DEPRECATED Path 3 | P2 (this design) |
|---|---|---|
| When does the LLM run? | **At RUNTIME** (every turn) | **OFFLINE only** (authoring the curriculum, once) |
| Who does the cognition? | The external LLM (it reasons + decides + abstains) | The BRAIN (parser+composer+moat reason/abstain at runtime) |
| Who holds the knowledge at inference? | The LLM | The brain's structured stores |
| Runtime dependency on an external model? | **Yes** | **NONE** — the agent runs standalone after teaching |
| Role of the rich model | The cognitive engine | A "textbook author" — like a human teacher who wrote the lessons and then left |

The rich model in P2 is the *author of the lesson plan*, not the *student's brain*. Once the curriculum is generated
(offline) and the brain has learned it, the agent converses with **zero external-LLM calls** — the no-external-LLM-at-
runtime stance is preserved by construction. (The *fluent faculty* in P1 is likewise a converted-to-spikes model that
is now PART OF the brain, not an external service.) **Consistent.**

---

## 3. P3 — GROUNDING / anti-hallucination (the structured memory gates the fluent faculty)

### 3a. The risk, stated honestly

LLMs hallucinate — they emit fluent assertions ungrounded in any fact (the categorical *opposite* of the project's
no-confab moat). If the fluent faculty (P1) is allowed to *assert content*, it will confabulate, destroying the moat
the project spent the whole conversational arc building. **So the faculty must be confined to producing fluent SURFACE
FORM, while the brain's structured stores supply and verify the CONTENT.** The grounding mechanism is what enforces
this split.

### 3b. The project's OWN grounding primitives (verified — these are the gate)

The brain already has a stronger anti-hallucination guarantee than any retrieval-augmented LLM, because its recall is
*exact-match over stored bindings*, not soft retrieval:

1. **The composer's abstention moat** — `query_patient` / `query_agent` / `render_fact` return **`None` BEFORE any
   rendering** when no stored fact matches the cue (`rf_phasor_composer.py:589,601,654,677`), and `ask_yes_no` returns
   `"unknown"` (`:630`). The moat is structural: a cue that no bound fact matches yields *nothing to say*, not a
   fabricated answer. Multi-hop `query_chain` abstains the moment ANY hop misses (`:614`). The 320-scale production
   demo holds this at **0 false-accepts** (CLAUDE.md, `2026-06-18-onebrain-320-scale-production-GO.md`).
2. **The Bogacz-Brown familiarity gate** — a learned neural abstention decision that matches the host abstention at
   V=320 multi-seed (agreement 168/168 every seed, **zero moat-breaches**; CLAUDE.md, `familiarity_gate_v320_validation.py`,
   `2026-06-11-familiarity-gate-v320-GO.md`). This is the *neural* realization of "is this concept KNOWN?" — the
   familiarity signal that gates whether the faculty is allowed to speak about a referent.
3. **Reconsolidation prediction-error gate** — `update_on_mismatch` (`rf_phasor_composer.py:487-501`) abstains on a
   never-stored cue ("a missing trace is not fabricated", `:491`), so even *corrections* can't smuggle in confabulation.

### 3c. Grounded-generation SOTA (the external frame the brain's primitives sit inside)

The 2024–2025 grounded-generation literature converges on exactly the architecture P3 needs — **separate the fluent
generator from the verified knowledge source, and constrain/gate the generator with the source**:

- **Retrieval-augmented generation (RAG) surveys** (arXiv 2506.00054, faith.futuretechsci 297): ground generation by
  conditioning on retrieved *structured* knowledge (knowledge graphs, tables, databases) rather than the model's
  parametric memory — "a potential solution is to ground generation with retrieved structured knowledge."
- **GraphRAG / ontology-augmented** (Awesome-GraphRAG; Walk&Retrieve arXiv 2505.16849): structured (graph) knowledge
  reduces hallucination and aligns output with the knowledge base — *directly analogous to the composer's role-filler
  fact graph*.
- **Constrained / template decoding** (Hofstätter 2023 executable templates; multi-grained constrained decoding):
  retrieve a *structure* and constrain the generator's output space to it — minimizing "generative interpolation."
- **RAGTruth** (Niu 2024) + **Confidence-Calibrated RAG** (Ozaki 2025): benchmark + calibrate *when* the generator
  should trust retrieved context vs abstain.

**The mapping to this project is unusually clean: the brain's structured store IS the knowledge graph, and the
composer's abstention IS the calibrated-confidence gate** — both already validated, both stronger than soft retrieval
(exact binding-match vs cosine similarity).

### 3d. The grounding mechanism — sketch (faculty proposes form; brain supplies + verifies content)

The split that preserves the moat:

```
  USER QUERY  ──►  [PARSER]  ──►  cue {agent, action} (or topic)
                                       │
                                       ▼
                            [COMPOSER / STORE]  ── exact-match recall ──►  CONTENT  (the SVO fact)  ─┐
                                       │                                                              │
                                       ├── no match ──►  ABSTAIN (moat / familiarity gate)  ──────────┤
                                       ▼                                                              ▼
                            (a fact exists)                                              [FLUENT FACULTY (P1)]
                                                                          renders the RETRIEVED content into fluent
                                                                          surface form — CONSTRAINED to the
                                                                          composer's words/roles (the faculty
                                                                          chooses grammar/phrasing, NOT facts)
                                                                                          │
                                                                                          ▼
                                                                              VERIFY: re-parse the faculty's output;
                                                                              its asserted SVO must match the stored
                                                                              fact (else reject/regenerate)
                                                                                          │
                                                                                          ▼
                                                                                    GROUNDED FLUENT REPLY
```

Three enforcement layers, cheapest-first: (i) **gate** — the composer/familiarity-gate decides *whether there is
content to speak* (if it abstains, the faculty says nothing / "I don't know" — the moat); (ii) **constrain** — the
faculty is conditioned on the retrieved fact's *words and roles* (slot-filling / constrained decoding over the
composer's output, not free generation), so its degrees of freedom are grammar+phrasing only; (iii) **verify** — the
faculty's surface form is re-parsed and its asserted SVO checked against the stored fact (a closed loop using the SAME
parser), rejecting any output whose content drifted. Layers (i)+(iii) are the moat-preservers; (ii) is the
hallucination-reducer. This is RAG/constrained-decoding/grounded-generation SOTA, with the brain's exact-match store +
abstention substituting for soft retrieval + confidence calibration — a *stronger* guarantee than the literature's
baseline.

---

## 4. The cheap-first DE-RISKS (one per piece, cheapest first, runnable)

Ordered by cost. **P2 first** — it is the do-it-now de-risk that validates the entire knowledge-and-grounding half
with zero downloads, zero conversion, zero new GPU.

### DE-RISK P2 (CHEAPEST — do it now, in-session) — "Claude writes a curriculum → the brain learns it → structured recall + no-confab"

- **Cost:** a single composer/agent run; **no model download, no convert, no new GPU** beyond a normal
  `BrainConversationalAgent` invocation. CPU-feasible for a small vocab; GPU only for the 320-scale variant.
- **The curriculum (the controller produces this DIRECTLY this session):** Claude authors a small structured corpus —
  e.g. **~30 SVO facts** over a coherent micro-domain (animals/objects/actions: "dog eat meat", "cat chase mouse",
  "bird eat seed", "apple is red", "dog is big", plus 2–3 embedded-clause facts and a 2-hop chain
  "dog chase cat / cat chase mouse"), AND a matching **short co-occurrence stream** (a few topically-coherent
  sentences) for the stream cortex.
- **The run:** feed the SVO facts through `BrainConversationalAgent.hear(...)` (parser → composer store); optionally
  feed the stream through the PPMI stream-cortex de-risk runner to ground the concept codes
  (`grounded_codes=`/`concepts=` passthrough, `brain_conversational_agent.py:195,202`).
- **The metrics (reuse the validated harness):** (1) **structured recall** — `query_patient`/`query_agent`/`ask_yes_no`/
  `query_chain` return the taught facts (recall ≈ 1.0 on the small set); (2) **no-confab moat** — queries about
  NEVER-taught cues return `None`/`"unknown"` (**0 false-accepts** is the bar); (3) **generalization** (if the stream
  cortex is used) — a held-out concept lands in its correct category. Multi-seed; the moat check is the load-bearing
  anti-cheat (a curriculum-taught fact must be retrievable AND an untaught one must abstain).
- **GO** = facts recall correctly AND the moat holds 0-false-accept on untaught cues, ≥3 seeds. ⇒ the **knowledge
  teacher → brain re-encoding → grounded recall** loop works end-to-end. This de-risks P2 AND P3's gate simultaneously,
  and it needs P1 not at all — it proves the *brain half* before any LLM is touched. **This is the recommended first
  action.**

### DE-RISK P1 (next) — "convert a ~0.5B fluent model + measure fluency-preserved"

- **Cost:** hours, 1×3090, **NO training** (per the Gen-F convert de-risk — inference + a calibration minibatch). The
  first, cheapest variant reuses the SHIPPED Gen-F checkpoint (the Gen-F convert de-risk in
  `2026-06-22-genseq-convert-scoping.md` §3 — already scoped: spiking-rate forward for `TinyGPT` at T=16, run the
  byte-frozen Gen-F gate). That validates the *convert mechanism* on a vanilla GPT for free.
- **The faculty-specific extension:** download **Qwen2.5-0.5B-Instruct** (or SmolLM2-360M), apply the **Plug-and-Play
  Spiking Operators** decomposition (RMSNorm + SiLU/SwiGLU + Softmax → division/exp/ℓ₂ primitives; RoPE as the fixed
  rotation) training-free at T∈{4,8,16}, and measure **fluency-preserved**: held-out ppl on a small corpus (spiking vs
  ANN, target ≤~1.2× per the SOTA's ≤1.03), distinct-n-gram non-degeneracy, verbatim-copy ≤ 0.20, AND **read the
  generated text** for coherence (the load-bearing check — the SOTA reports ppl, NOT generation coherence; §4 of the
  Gen-F scoping flags this as the genuine open question).
- **GO** = spiking generation stays coherent (clears the non-degenerate + not-copying bars AND reads as fluent) at the
  lowest feasible T, ≥3 seeds. ⇒ a spiking fluent faculty exists. **NO-GO/escalate** = raise T → bit-exact (NEXUS) →
  surrogate-grad finetune (the ordered fallbacks).

### DE-RISK P3 (grounding/gating smoke) — "does the gate confine the faculty to grounded content?"

- **Cost:** small; composes the P2 store + (a stub or the real) P1 faculty.
- **The smoke:** with the P2-taught store, route a query through the gate (composer recall → abstain-or-content), have
  the faculty render the retrieved fact into fluent form (initially a *template/slot-fill* stub standing in for the
  full faculty — keeps the smoke cheap), then **VERIFY** by re-parsing the faculty's output and checking its SVO
  against the stored fact. The adversarial case: a query about an UNTAUGHT cue must produce **abstention** (the faculty
  is given nothing to render), and a faculty that tries to confabulate must be **caught by the re-parse verify**.
- **GO** = grounded queries → fluent correct content; untaught queries → abstain (moat held); injected-confabulation →
  caught by verify. ⇒ the grounding mechanism preserves the moat while the faculty supplies fluency.

---

## 5. Ranked plan + the explicit cloud-trigger

| Rank | Step | Why this order | Cost | Cloud? |
|---|---|---|---|---|
| **1** | **DE-RISK P2** — Claude authors a ~30-fact curriculum + stream; brain learns it; structured-recall + no-confab check (≥3 seeds) | Validates the WHOLE knowledge+grounding half with **zero downloads/convert/new-GPU**; the cheapest possible signal and it de-risks P3's gate too | composer run (CPU-feasible small; GPU for 320-scale) | **No** |
| **2** | **DE-RISK P1a (free)** — run the SHIPPED Gen-F convert de-risk (the `2026-06-22-genseq-convert-scoping.md` §3 experiment) to validate the *convert mechanism* on a vanilla GPT | Reuses shipped checkpoints; proves the spiking-forward + gate works before any new model | hours, 1×3090, NO training | **No** |
| **3** | **DE-RISK P1b** — convert **Qwen2.5-0.5B-Instruct** (Plug-and-Play operators, T∈{4,8,16}, training-free) + measure fluency-preserved + READ the text | The production faculty tier; the LLaMA-stack convert the SOTA covers; small enough to run locally | hours–day, 1×3090 (download + inference + calibration) | **No** (at 0.5B, T≤16) |
| **4** | **DE-RISK P3** — grounding/gating smoke (P2 store + faculty render + re-parse verify; adversarial untaught-cue abstention) | Closes the loop: fluency confined to grounded content, moat preserved | small | **No** |
| **5** | **Scale + integrate** — larger fluent convert (≥1.5B) and/or a large teaching corpus; consolidate the faculty onto the bridge | Only once 1–4 GO; this is where local feasibility ends | — | **Cloud (the trigger, below)** |

### The explicit cloud-trigger

Trigger the cloud (per the project's standing "CuPy/GPU for decisive runs, cloud only once production-worthy" stance)
on **whichever comes first**:

1. **The larger fluent-model convert** — a **≥1.5B** model's spiking forward at **T≥16** (Plug-and-Play approximate)
   or **bit-exact** (NEXUS, T≈32 + ~3–4k neurons/FP-op overhead) over a **long context**. The 3090 holds 0.5–1.7B
   weights, but `T × matmul × long-context` (and the bit-exact gate-neuron overhead) is the wall-clock ceiling — the
   moment the spiking forward stops being "hours on the 3090," it goes to cloud.
2. **A large teaching corpus** — if P2's curriculum needs to scale from ~30 facts / a short stream to a
   **corpus-grounded taxonomy** (the documented 320-concept tier needed a curated taxonomy; a richer faculty wants a
   broader stream), the *generation* of that corpus (a large rich-model run) and the *stream-cortex training* over it
   (many windows — the documented window-budget wall) is the second cloud-trigger.

Everything before Rank 5 is local. The decisive cheap signal — **P2, do-it-now, in-session** — needs no GPU at all at
small scale.

---

## 6. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / file read):**
- **Parser** = Hebbian (position×voice)→role on `SimulationBridge`, vocabulary-agnostic: `BridgeParser`
  `brain_conversational_agent.py:28-143` (rule `:73`, `_GT` `:25`, `.parse` `:139-143`), read in full.
- **Composer (rf)** API: `store:432`, `query_patient:579`, `query_agent:568`, `ask_yes_no:618`, `render_fact:632`,
  `query_chain:603`; **abstention** = `return None`/`"unknown"` BEFORE rendering (`:589,601,614,630,654,677`) —
  `rf_phasor_composer.py`, read.
- **Composer (one-brain)** = whole pipeline on one bridge, `grounded_codes=` drop-in: `one_brain_composer.py:107-118`
  (constructor `:113-118`), read.
- **Agent** wiring + `composer_kind` ∈ {rf,onebrain,rate} + `concepts=`/`grounded_codes=` passthrough:
  `brain_conversational_agent.py:146-205`, read.
- **Engram** API: `start_engram_recording:3352`, `commit_engram_tag:3381`, `stimulate_tag:3466`,
  `list_engram_tags:3510`; auto-tick `_tick_engram_recordings:3368` (zero overhead idle) — `sim/bridge.py`, read.
- **Consolidation**: `run_consolidation_training:206`, `run_swr_replay_phase:154`, `run_concept_replay_phase:43`,
  awake/sleep gates + `enable_hippocampus_consolidation=True:275` — `consolidation_trainer.py`, read.
- **PPMI stream cortex** = rate-Hebbian co-occurrence (STDP measured-negative): finding
  `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md:1-60`, read; runner
  `_phaseB_onbridge_stream_cortex_derisk.py` on disk.
- **Path 3 deprecation framing** (external LLM = RUNTIME cognition, "no external LLM ever"): `CLAUDE.md:1099-1112,
  1308-1310, 2420-2423`; `sim/llm_memory_orchestrator.py` is the deprecated runtime path.
- **Gen-F convert facts** (vanilla GPT: MHA+LayerNorm+GELU; MBE/LAS/ECMT cover those; 3.45M; ppl 6.1):
  `2026-06-22-genseq-convert-scoping.md` (read this pass) + `2026-06-22-genseq-step0-C1-consolidation-GO.md` (read).

**SOTA (fresh June-2026 web pass; abstracts + key full-texts read):**
- **Modern SLMs are LLaMA-family** (RMSNorm/RoPE/SwiGLU/GQA), NOT vanilla GPT: SmolLM2 (RMSNorm+RoPE+GQA+SwiGLU, vocab
  49152, tied emb — HF SmolLM blog / model cards), Qwen2.5-0.5B (RMSNorm+RoPE(1e6)+SwiGLU+GQA, vocab ~151,643, tied —
  Qwen2.5 tech report). [VERIFIED via web; the exact per-model layer counts are secondary.]
- **Plug-and-Play Spiking Operators** (arXiv 2605.20289): training-free RMSNorm+SiLU+Softmax via division/exp/ℓ₂
  primitives, <1% loss at T∈{1,2,4} on LLaMA-2/3, Mistral, Qwen3. **Full-text read this pass.**
- **NEXUS** (arXiv 2601.21279): bit-exact ANN→SNN incl. RMSNorm/SiLU/SwiGLU/RoPE/Softmax, ppl identical to ANN
  (WikiText-2 LLaMA-2-7B 5.12), 32-step FP32 window, surrogate-free STE. **Full-text read this pass.**
- **Grounded-generation / anti-hallucination SOTA**: RAG survey (arXiv 2506.00054), GraphRAG (Awesome-GraphRAG,
  Walk&Retrieve 2505.16849), RAGTruth (Niu 2024), Confidence-Calibrated RAG (Ozaki 2025), constrained/template
  decoding (Hofstätter 2023). Abstracts read.
- **TinyStories** (Eldan & Li, 2305.07759): <10M params → coherent multi-paragraph English (the fluency-is-cheap
  existence proof).

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — most load-bearing] That a converted SLM preserves GENERATION coherence, not just perplexity.** Every
   SOTA number (Plug-and-Play, NEXUS, MBE, LAS) is ppl/accuracy; NONE directly measured post-conversion free-generation
   quality. This is the P1 de-risk's read-the-text + distinct/copy checks — the hypothesis the experiment tests, NOT a
   settled result. (Inherited verbatim from the Gen-F convert scoping §4.)
2. **[VERIFY] Code release for Plug-and-Play (2605.20289) and NEXUS (2601.21279)** — not stated in the abstracts this
   pass. LAS (`github.com/lc783/LAS`), ECMT (`github.com/h-z-h-cell/Transformer-to-SNN-ECMT`), QCFS
   (`github.com/putshua/SNN_conversion_QCFS`) ARE released (verified in the Gen-F scoping) and cover the vanilla-GPT
   ops; the LLaMA-stack operators may need re-implementation from the papers' method sections if no code drops.
3. **[VERIFY] The exact per-token spiking-forward FLOP/memory constants in §1e** are order-of-magnitude estimates
   (`T × ANN-forward` + gate overhead), not a profiled measurement. The P1 de-risk produces the real numbers.
4. **The licenses in §1b** are from secondary sources (the model-card / survey web pass), not re-read from each LICENSE
   file. Apache-2.0 for SmolLM2/Qwen2.5/TinyLlama/OLMo and custom-community for Llama-3.2/Gemma-2 are well-established,
   but the controller should confirm the exact license before redistributing a converted model.

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `research/runners/brain_conversational_agent.py` (parser `:28-143`; agent + `composer_kind` `:146-205`).
- `research/runners/rf_phasor_composer.py` (composer API `:432,568,579,603,618,632`; abstention moat `:589,601,614,630,654,677`).
- `research/runners/one_brain_composer.py` (one-brain pipeline `:107-118`; `grounded_codes=` `:114`).
- `sim/bridge.py` (engram API `:3352,3381,3466,3510`; auto-tick `:3368`).
- `research/runners/consolidation_trainer.py` (consolidation `:206,154,43`; hippo `:275`).
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (PPMI stream cortex).
- `research/findings/2026-06-22-genseq-convert-scoping.md` + `2026-06-22-genseq-step0-C1-consolidation-GO.md` (the
  Gen-F convert + step-0 GO this builds on).
- `CLAUDE.md` (Path 3 deprecation `:1099-1112,1308-1310,2420-2423`; familiarity gate / 320-scale moat; consolidated-320).

### Current literature (June 2026 pass)
- **Plug-and-Play Spiking Operators: Breaking the Nonlinearity Bottleneck in Spiking Transformers** — arXiv 2605.20289
  (training-free RMSNorm/SiLU/Softmax, <1% at T∈{1,2,4}, LLaMA-2/3 + Qwen3). Full-text read.
- **NEXUS: Bit-Exact ANN-to-SNN Equivalence via Neuromorphic Gate Circuits** — arXiv 2601.21279 (bit-exact incl.
  RMSNorm/SiLU/RoPE/Softmax, ppl identical, 32-step FP32). Full-text read.
- **MBE** (2508.07710), **LAS** (2505.09659, code `github.com/lc783/LAS`), **ECMT** (ACM-MM 2024, code
  `github.com/h-z-h-cell/Transformer-to-SNN-ECMT`), **QCFS** (Bu 2023, code `github.com/putshua/SNN_conversion_QCFS`)
  — the vanilla-GPT-op convert methods (inherited from the Gen-F scoping).
- **SmolLM / SmolLM2** — HuggingFace blog + model cards (LLaMA-family, Apache-2.0, vocab 49152, RMSNorm/RoPE/GQA/SwiGLU).
- **Qwen2.5 Technical Report** — arXiv 2412.15115 (0.5B–72B, RMSNorm/RoPE/SwiGLU/GQA, vocab ~151,643, tied emb at small
  scale, Apache-2.0).
- **RAG survey** (2506.00054), **GraphRAG** (Awesome-GraphRAG; Walk&Retrieve 2505.16849), **RAGTruth** (Niu 2024),
  **Confidence-Calibrated RAG** (Ozaki 2025) — grounded-generation / anti-hallucination SOTA.
- **TinyStories** — Eldan & Li, arXiv 2305.07759 (<10M → coherent English; fluency-is-cheap).
