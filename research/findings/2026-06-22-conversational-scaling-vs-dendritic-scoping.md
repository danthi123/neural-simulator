# Conversational scaling vs the dendritic / learned-cortex unlock — load-bearing direction scoping (2026-06-22)

> **Status:** READ-ONLY deep-research + catalog/literature scoping (the standing "deep research FIRST before
> committing GPU/months to a wall" move). NO `sim/` edits, NO experiments, NO GPU. Single deliverable = this doc.
> Every load-bearing project claim re-verified against the repo (file/finding/line cited); the surprising ones read
> in full, not trusted from a prior summary. SOTA bounded by a fresh literature pass. **This is a scoping/decision
> doc, NOT a brain-based result and NOT a commitment to build.**

---

## 0. The one-paragraph answer (the rest is the evidence)

**The gap to a small LLM is overwhelmingly CATEGORICAL, not scale — and the dendritic / learned-cortex rewrite is
NOT the cheap unlock that closes it; on the project's own evidence it is neither cheap nor, for the conversational
goal, an unlock at all.** The production composer (`OneBrainComposer`/`RFPhasorComposer`) already does the things
that *scale* cleanly — vocabulary to 320 concepts (the documented "age-5" tier), K=32 facts/store with zero
cross-talk, who/what Q&A, the no-confab abstention moat at 100%, negation/yes-no, single- AND two-attribute binding,
multi-hop reasoning, multi-turn anaphora, learn-from-conversation codes (the PPMI "stream cortex"), and frame-selected
(SVO/VSO/OSV) comprehension. What it categorically *cannot* do is what a small LLM is *for*: **open-domain arbitrary
text, learned grammar beyond a small set of fixed frames, and FREE generation** (it emits stored facts through fixed
templates, it does not produce novel fluent sentences). Those three are categorical gaps that **no amount of scaling
the current architecture closes** — they require a fundamentally different generative learned-sequence mechanism. The
critical finding for the owner's question is that **the dendritic build was already de-risked to a fork (2026-06-14,
"build D2"), then the project's own follow-up work overturned the premise**: (a) the D2 Phase-1 dendritic gain was
built on the bridge and found NOT load-bearing — the spiking threshold + read-out integration substitute for it; (b)
the generalizing cortex was *delivered on point neurons* via local PPMI normalization, no dendrite; (c) the
off-diagonal residual the dendrite might still buy was de-risked NEGATIVE (local PPMI already reaches the whitening
ceiling); and (d) BOTH named dendrite jobs — multi-attribute binding and apical-basal credit assignment — were
cheap-first tested and came back NEGATIVE (the learned dendrite *memorizes* two-attribute bindings but does not
generalize; worse than the fixed FHRR primitive it would replace). **⇒ The dendrite is NOT the unlock for the
categorical LLM gaps.** It closes the project's *biology-fidelity* science question (already done) but does not buy
free generation or learned open-domain syntax — those are a *different* missing mechanism (a learned generative
sequence model on the substrate, e.g. the benched surrogate-grad SNN cortex), not a dendritic one. **The honest
ranked plan: (1) bank the cheap conversational-scaling wins that are already in flight (production-wire the
validated frame-parser / attributes / learned-syntax / bigger-vocab — weeks, low-risk); (2) recognize that
"small-LLM-competitive" (free open-domain generation) is a SEPARATE multi-month research bet on a learned generative
substrate, NOT the dendritic cortex; (3) the dendrite stays a legitimate artificial-life capstone but is
de-prioritized as a CONVERSATIONAL unlock.** §6 gives a cheap, no-retrain ceiling-probe that quantifies exactly
where the current arch breaks, runnable today against the shipped bridges.

---

## 1. THE GAP, by capability dimension — small-LLM vs the current composer

A "small LLM" reference = Phi-3-mini (3.8B) / Llama-3.2-3B class: open-weights, runs on the project's RTX 3090,
trained on trillions of tokens, does open-domain conversation, free generation, in-context learning, multi-step
reasoning. The composer is a fixed exact-inverse FHRR (Fourier Holographic Reduced Representation) vector-symbolic
**algebra** over role-filler bindings on a spiking resonate-and-fire substrate (`rf_phasor_composer.py`,
`one_brain_composer.py`). The contrast:

| Dimension | small LLM (Phi-3 / Llama-3.2-3B) | current composer (verified) | gap type |
|---|---|---|---|
| **Vocabulary scale** | ~32k–128k subword tokens, open | **320 curated concept words** (`vocab_ceiling_probe.py`, V=320 6-seed GO); the V-list is the only input, codes self-generate | **SCALE** (closeable; bridge-size / corpus-bound, not mechanism-bound) |
| **Arbitrary / open-domain text** | any English (any topic, OOV-robust via subwords) | **closed curated vocab + fixed SVO/role grammar**; an OOV word has no concept code → abstains | **CATEGORICAL** (no subword tokenizer, no open lexicon; the moat *correctly* refuses unknowns) |
| **Learned syntax** | full grammar, learned from data (any construction) | **fixed SVO + a small frame set** — SVO/passive (`BridgeParser`), and SVO/VSO/OSV via the neural `FrameParser` (frame-selection GO 6/6, CYCLE 204). Real morphology/case/recursion-depth not learned | **CATEGORICAL for open grammar** (frames are hand-enumerated, not learned); **SCALE** for *adding more fixed frames* |
| **Free generation** | autoregressive — produces novel fluent sentences token-by-token | **templated emission of STORED facts**: `render_fact`/`describe` decode a stored fact's roles and join them; word ORDER is neural (the competitive-queuing serial-order renderer, GO) but the CONTENT is a retrieved fact, never a novel composition | **CATEGORICAL** (no autoregressive next-token generator; cannot say anything it was not told) |
| **Compositional novelty** | generalizes to unseen word combinations (systematic) | **binds novel role-filler combinations** (the algebra is systematic by construction; held-out combos recover 1.000) — BUT only over the closed concept set, flat-distinct codes | **PARTIAL** (compositional within the closed vocab ✓; not open-ended) |
| **Context length** | 4k–128k tokens | **K=32 facts/store** (`one_brain_multifact_store` GO, zero cross-talk), multi-turn anaphora over a short discourse buffer | **SCALE** (K bounded by neuron budget; multi-bridge sharding extends it) |
| **Reasoning** | multi-step chain-of-thought, arbitrary | **multi-hop relational pointer-chase** (`query_chain`, GO 3-seed, holds to 4 hops, moat at every hop) — structured retrieval over stored facts, not free inference | **PARTIAL** (relational reasoning over stored facts ✓; open-ended inference / arithmetic / world-knowledge ✗) |
| **In-context learning** | few-shot from the prompt | none (learning is offline stream-cortex code formation + fact storage) | **CATEGORICAL** |
| **No-confab reliability** | hallucinates confidently | **abstains** ("I don't know") — the no-confab moat, 100% (20/20) every cell at V=320, 0 false-accepts | composer **WINS** here (the algebra buys it ~free; an LLM does not have it) |

**Reading.** The composer's strengths and the LLM's strengths are almost disjoint. The composer is a reliable,
abstaining, compositional **structured memory + relational-query engine**; the LLM is an open-domain **generative
fluent text model**. Three of the gaps (open-domain text, learned open grammar, free generation) are *categorical* —
they are the LLM's defining capabilities and the composer has no mechanism for any of them. The scale gaps (vocab,
context-K, more frames) are real but closeable by the current arch. **The owner's question "how far can the spiking
conversational system scale toward small-LLM-competitive" therefore has a sharp answer: it scales cleanly along the
SCALE axes and hits a hard categorical wall on the three GENERATIVE axes that *define* an LLM.**

---

## 2. The current architecture's CEILING (evidence-based)

### 2.1 What scales cleanly (verified GO, multi-seed)

- **Vocab → 320 concepts.** `vocab_ceiling_probe.py`, V=320, 6 seeds (42–47): the full capability matrix GO; the
  no-confab moat 20/20 every cell; two-attribute (the old K=5 boundary) **lifted on the production agent**; the only
  code-dimension dependence is the embedded clause (needs D≥256 at V=320). (`2026-06-10-vocab-ceiling-multiseed-GO.md`.)
- **K=32 facts/store, zero cross-talk.** Per-fact-isolated tiled complex-weight blocks; recall 1.00 == oracle at
  K=8/16/32, unused-block peak exactly 0 (the moat holds at scale), per-fact recall independent of K. Capacity is
  **bounded by neuron budget, not matched-filter SNR** (the tiled store sidesteps the superposition SNR wall); the
  320-concept multi-bridge sharding route extends beyond a single bridge.
  (`2026-06-18-one-brain-multifact-store-GAP-A-GO.md`.)
- **Learn-from-conversation codes.** The PPMI "stream cortex" hears a corpus window-by-window (online Hebbian
  co-occurrence + running frequency, NO preprocessing, NO global matrix), reaches the host distributional-semantics
  target (`corr(M,C_stream) +0.885`), generalizes (held-out 0.86), and the full conversation runs on the
  stream-learned codes at 320 concepts, 3-seed, moat clean. **This is the genuine "learned cortex" the project has —
  on point neurons.** (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`, CLAUDE.md 2026-06-15.)
- **Multi-hop reasoning, multi-turn anaphora, negation, generation word-order** — all individually GO (CLAUDE.md
  2026-06-17 arc), all reuse-by-import, moat never weakened.
- **Frame-selected comprehension** (SVO/VSO/OSV) — neural verb-position→frame selection GO 6/6 (CYCLE 204).

### 2.2 The categorical wall (the fixed FHRR algebra is the load-bearing idealization)

The honest, project-documented limitation (`2026-06-06-composer-vsa-idealization-known-limitation.md`,
`rf_phasor_composer.py:1-13`): the composer binds with a **clean, exactly-invertible algebra that DEMANDS
decorrelated full-precision codes**. Three consequences cap it categorically:

1. **No free generation.** `render_fact`/`query_patient`/`describe` (`rf_phasor_composer.py:632`, `:579`) DECODE
   stored facts and emit them through a fixed template (`f"{agent} {ac} {pt}"`, or the neural serial-order renderer
   for word ORDER only). There is **no autoregressive generator** — the composer cannot produce a sentence it was not
   explicitly told. This is the single biggest categorical gap and it is structural, not a tuning knob.
2. **No learned open grammar.** Comprehension is a fixed role-assignment (SVO/passive + the enumerated frames). The
   grammar is *hand-specified*, not *learned from data*; real syntax (arbitrary recursion depth, case/agreement
   morphology, novel constructions) is not acquired. The frame-selection mechanism scales by *adding more
   hand-built frames*, not by learning grammar.
3. **Multi-attribute binding is a fixed primitive, not learnable.** Two-attribute binding works on the production
   agent (the FHRR F=3 resonator), but is the documented K=5-load boundary and **degrades to ~29% on the correlated
   LEARNED codes** (`2026-06-19` resonator-on-learned-codes). A from-scratch *learned* multi-attribute bind is
   NEGATIVE (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`): additive has no inverse
   (0.193), a learned linear inverse cannot be a reciprocal (0.056). The fixed self-inverse algebra is **load-bearing
   and not replaceable by learning** on point neurons.

**The ceiling, stated precisely:** the current architecture is a *reliable compositional structured-memory + relational-query
+ templated-readout system* that scales to ~320 concepts / K=32 facts / multi-hop / multi-turn. It will **never** be a
free-generation open-domain text model, because the fixed exact-inverse algebra is a *retrieval/binding* mechanism, not
a *generative-sequence* mechanism. That is the categorical wall, and it is the algebra itself.

---

## 3. Is the dendritic / learned-cortex the genuine UNLOCK? — NO (the premise was overturned by the project's own work)

This is the load-bearing question, and the answer changed *after* the fork was framed. The chain of evidence
(all re-verified in full this pass):

### 3.1 The fork was framed (2026-06-11 → 2026-06-14): "build D2, the dendritic cortex"
`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md` and the fork-resolved decision
`docs/plans/2026-06-14-step3-cortex-fork-resolved-dendritic-D2-decision.md` concluded: a semantically-FLAT cortex (A)
is achievable on point neurons but cannot generalize; a semantically-STRUCTURED generalizing cortex (B) "cannot be
done on point neurons" (five mechanistically-distinct NEGATIVEs → the Mikulasch-Priesemann analog/pre-spike whitening
limit), so the dendritic substrate is the validated escape → **build D2** (owner-gated, ~1.5–2 months). The D1/D1.5/D1.6
de-risk ladder was GO, and Phase 0 (D1.7, a full spiking two-compartment LIF probe) SURVIVED. **At the 2026-06-14
fork, the dendrite genuinely looked like the unlock for the generalizing cortex.**

### 3.2 Then THREE findings overturned the premise (the corrections the prompt's framing predates)

**(a) The D2 Phase-1 dendritic gain was BUILT and found NOT load-bearing.** The protected `sim/` edit
(`enable_dendritic_divisive_gain`, 5 guarded sites, byte-identical when off — 18/18 GPU conversational tests incl. the
moat) was DELIVERED, but Phase 2's clean-readout control **inverted** the result: point-neuron +0.167 (gen 0.422) vs
dendritic +0.042 — i.e. *with enough read-out temporal integration the point neuron recovers the structure on its own
and the gain HURTS.* The D1 ladder over-stated the dendritic advantage because its point-neuron control was a single
*rate-level* global gain, lacking the real spiking substrate's threshold + temporal integration. (Verbatim from
`2026-06-14-D2-phase1-DONE-phase2-frontier.md` §CORRECTION, read in full via the 2026-06-17 scoping §1.3a.)

**(b) The generalizing cortex was DELIVERED on point neurons — no dendrite.** The "off-diagonal red herring" / PPMI
reframe (CYCLE 88–96): generalization needs **feedforward LOCAL normalization** (PPMI = log + per-hub + per-concept
mean-subtraction + ReLU — all point-neuron ops), NOT cross-neuron decorrelation (which would *destroy* generalization
by whitening away the similarity). PPMI reaches host (+0.518 > host +0.442 > offline ZCA +0.49), generalizes
(held-out 0.86), lands in the binding sweet spot, and ships the full pipeline at 320 concepts, multi-seed, moat clean,
on the real spiking substrate. **The capability D2 Phase 3 targets — generalization-in-conversation about a held-out
concept via a similar known one, moat intact — is already passed on point neurons** (`2026-06-16-generalization-capstone-verbalize.md`,
0.92 3-seed; the 320-stream cortex). So D2 Phase 3 would *re-deliver a shipped capability* via a months-scale build.

**(c) The off-diagonal residual the dendrite might still uniquely buy was de-risked NEGATIVE.** The one residual a
dendrite was hypothesized to add (the cross-neuron low-rank decorrelation PPMI's diagonal normalization leaves on the
table) was tested cheap-first (`2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md`, 3 seeds): the
online-local circuit reaches +0.519 — but its **lesion (no learned off-diagonal gains) gives the SAME +0.519**, and
the effective rank is 53 not ~8. The learned gains are **inert**: local PPMI-centering *already reaches the whitening
ceiling* (+0.519 ≈ ZCA +0.524). Verdict: "the months-scale dendritic off-diagonal rewrite is NOT required for the
generalizing conversational cortex; ship the flat 2,048-concept cortex."

### 3.3 And BOTH named dendrite *jobs* were assessed cheap-first — both NEGATIVE

CLAUDE.md's recent commits log it directly: *"BOTH dendrite jobs assessed, both NEGATIVE → dendrite thoroughly ruled
out for current walls."*

- **(job a) learnable multi-attribute BINDING via dendritic multiplication** —
  **NEGATIVE** (`2026-06-19-dendritic-binding-toy-derisk.md`): the learned dendritic sigma-pi/plateau conjunction
  *memorizes* two-attribute bindings (bundle-train 0.422) but **does NOT generalize** (held-out 0.168, train→held gap
  +0.254 = the memorization signature), and is **worse than the production fixed FHRR primitive (0.261)** on the same
  held-out test. The dendrite's native multiplication lets it fit, but not generalize. ⇒ the binding wall is NOT the
  missing dendritic multiplication.
- **(job b) apical-basal CREDIT ASSIGNMENT** — **NEGATIVE**
  (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`): a single-layer actor has nothing to route; the dendrite's
  credit-assignment value needs a deep multi-layer network the conversational pipeline doesn't have.

### 3.4 What Phase 3 concretely entails (and why it is redundant for the conversational goal)

`docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md` line 21–22, verbatim: Phase 3 = "the learned graded codes →
the already-validated bind/unbind → attractor cleanup → the full conversational matrix. **GATE:**
generalization-in-conversation (answer about a held-out concept via a *similar* known one — what the idealized composer
algebra could NOT) WITH the no-confab moat intact, multi-seed." **Phases 0–2 status (pinned):** Phase 0 (D1.7)
DONE/SURVIVES; Phase 1 DELIVERED (the gain is a verified, harmless, byte-inert-when-off protected edit); Phase 2
HONEST NEGATIVE for the gain's *necessity*; **Phase 3 PENDING but redundant** (its target capability ships on point
neurons, §3.2b). The D2 groundwork that exists is the off-bridge numpy dendritic stack (`sim/dendritic_neuron.py`
58-line two-compartment Larkum/GLR model; `sim/dendritic_plasticity.py` Urbanczik-Senn; `sim/dendritic_mlp.py` GLR2017
feedback-alignment; all sound) + the built-and-verified `enable_dendritic_divisive_gain` on the bridge.

### 3.5 The decisive point: the dendrite is the wrong tool for the CATEGORICAL gaps

Even setting aside that the dendrite was overturned for the generalizing cortex, **none of the three categorical
LLM gaps (§1) is a dendritic problem:**
- *Free generation* is a learned **generative-sequence** problem (autoregressive next-token / serial-order
  production) — not a binding or normalization problem a dendrite addresses.
- *Open-domain text* is a **lexicon/tokenizer + corpus-scale** problem — not a per-neuron-compartment problem.
- *Learned open grammar* is a **structured-sequence-learning** problem (recursion, agreement) — the dendrite's
  named jobs (multiplicative binding, credit assignment) were both tested NEGATIVE for the binding wall, and grammar
  is a yet-different mechanism.

So even a *successful* D2 would buy biology-fidelity for the *representation* (graded learned codes), not the
*generative* capabilities that separate the composer from an LLM. **The dendrite is not the unlock for
"small-LLM-competitive."** It is a legitimate artificial-life capstone (the catalog's largest missing capability,
G.02 "active dendrites — MISSING, ~10× compute/neuron"), but for the conversational/LLM goal it is de-prioritized by
the project's own evidence.

**The genuine unlock for the categorical gaps, if pursued, is a LEARNED GENERATIVE SEQUENCE MODEL on the substrate**
— the benched surrogate-gradient SNN cortex (`sim/bptt_snn*.py`, `sim/surrogate_grad.py`; Phase 2.2 trained a 4-layer
SNN on Tiny Shakespeare, loss 14.1→2.24, ~9.4 perplexity, on the RTX 3090). That is the mechanism class that does
free generation; it was shelved because *at toy scale* char-level features didn't transfer to the binding task
(`2026-05-07-Phase-2.3a-NEGATIVE`), and because the project pivoted to the reliable VSA composer. **It, not the
dendrite, is the path toward LLM-style generation — and it is a separate multi-month research bet with its own
honest uncertainty (toy-scale was ~4 orders below the scale that makes it work).**

---

## 3b. SOTA check — spiking / biologically-grounded language models (what's achievable, and what they require)

A fresh literature pass (June 2026) to bound what a spiking LM can reach and what it costs:

| System | What it reaches | What it REQUIRES (that this substrate lacks) |
|---|---|---|
| **SpikeGPT** (arXiv 2302.13939, 2023) | 45M/125M/260M params; competitive with non-spiking on its benchmarks; ~20× fewer ops on neuromorphic HW | RWKV transformer block, **backprop-through-time training**, GPU pretraining on a large corpus. Binary spikes are an *activation* substitution on a standard ANN architecture — not point-neuron biology |
| **SpikingBrain-7B** (arXiv 2509.05276, 2025) | **Transformer-comparable** capability; 100× speedup on 4M-token TTFT; 69% sparsity | **ANN-to-SNN conversion** of a linear/hybrid-attention transformer + adaptive spiking neurons; **~150B tokens** of continual pretraining; custom training framework. Capability comes from the transformer + the corpus, the spikes are an efficiency layer |
| **SpikeLLM** (arXiv 2407.04752, 2024) | LLAMA-7B-class accuracy via saliency-based spiking quantization | A **pretrained LLAMA** to quantize; surrogate-free / saliency training. It *compresses* an existing LLM, it does not learn language from biology |
| **Neuro-symbolic VSA / HDC** (LARS-VSA arXiv 2405.14436; "Attention as Binding" arXiv 2512.14709, 2025) | Compositional/relational reasoning, abstract-rule learning; HRR interprets LLM internals | These are the SAME VSA family the project's composer already uses (HRR binding/unbinding/superposition). "Attention as Binding" shows transformer attention *is* approximate VSA binding — i.e. the composer is a *clean, exact* version of what an LLM does *approximately*. They confirm the composer's mechanism is principled; they do **not** add free generation |

**The SOTA bound, stated plainly:** *every* spiking system that reaches LLM-class capability does so by **training (or
converting) a standard transformer architecture with backprop/surrogate-gradient on a large corpus**, then spiking the
activations for efficiency. **None of them is a biologically-grounded point-neuron model that learns language from
local rules.** The capability lives in the *architecture + corpus + backprop*, not in the *neuron model*. This is the
hard external bound on the prompt's question: **a point-neuron, locally-learned, no-backprop substrate is categorically
not on the SOTA spiking-LM path** — the SOTA path is "ANN with spiking activations." The project's composer is the
*neuro-symbolic VSA* alternative (reliable, compositional, abstaining), which the 2025 VSA literature validates as
principled but which is, by construction, a *structured-memory* engine, not a generative LM. There is no published
spiking-LM result that a point-neuron local-learning substrate reaches; the achievable ceiling for THIS substrate is
the VSA-composer ceiling (§2), not the SpikingBrain ceiling.

---

## 4. The cheap intermediate wins — what scaling the current arch buys (and what it does NOT)

**Buys (cheap, weeks-scale, mostly production-wiring of already-validated GO mechanisms — see the in-flight
`docs/plans/2026-06-22-production-wiring-execution-plan.md`):**
- **Bigger vocabulary** — V=320 is validated; the curated-list + multi-bridge sharding extends it (the 320 tier was
  the documented age-5 target; pushing to a richer taxonomy is corpus work, not a mechanism change).
- **Frame-parser wire-in** — SVO/VSO/OSV comprehension as a production opt-in (`enable_multiframe`, frame-selection
  GO 6/6) — more word-order coverage.
- **Attributed entities** — adj+noun ("big apple") as a production default (`enable_attributed`, single-attribute the
  holding path on learned codes).
- **Neural sentence generation (word ORDER)** — the competitive-queuing serial-order renderer
  (`enable_neural_render`, GO) replaces the host f-string for *ordering* stored facts.
- **Richer dialogue / multi-turn / multi-hop** — already GO; consolidate into the one persistent agent.
- **Fuller no-confab + reconsolidation** — the moat + in-place fact correction, all validated.

**Does NOT buy (the categorical LLM gap — no overclaim):**
- **Free open-domain generation** — the composer emits *stored facts through templates*; it cannot generate novel
  fluent text on an arbitrary topic. No amount of vocab/frame scaling changes this — it is the fixed-algebra wall.
- **Learned open grammar** — adding frames is hand-work, not grammar learning; real morphology/agreement/arbitrary
  recursion is not acquired.
- **Arbitrary text / OOV robustness** — closed curated lexicon; an unknown word abstains (correctly, but it is not
  "understanding open text").
- **In-context few-shot learning, world knowledge, arithmetic, long-form coherence** — none of these is in the
  composer's mechanism class.

**Honest framing for the owner:** the cheap wins make the *structured-conversation* artifact richer and more complete
(more words, more frames, attributes, neural generation-order, fuller dialogue) — a genuinely better "age-5
structured conversationalist with a perfect no-confab moat." They do **not** move it toward "small-LLM-competitive" in
the sense of free open-domain generation. That target is a different mechanism (§3.5), not a scaled composer.

---

## 5. RANKED cheap-first plan (separating "cheap conversational scaling" from "the generative bet")

Ranked by leverage-per-cost for the project's actual north star (artificial life / biology-translatable; capabilities
instrumental; honest negatives are the deliverable; explicitly NOT chasing LLM fluency per the owner's standing
framing). **The dendritic build is NOT ranked as a conversational unlock — it is reframed as an artificial-life
capstone, gated and de-prioritized.**

| Rank | Item | Cost | What it proves / buys | Category |
|---|---|---|---|---|
| **1** | **Finish the in-flight production-wiring pass** (`2026-06-22-production-wiring-execution-plan.md`): flip the validated spiking defaults (frame-parser, attributes, neural-render, spiking cleanup, integrated-loop, learned-assoc graph) on the one production agent | **weeks**, low-variance, NO `sim/` edit, moat-gated | Makes the existing analogue *whole* — one persistent fully-spiking agent doing the full structured-conversation loop by default. Banks every cheap conversational win. | cheap scaling |
| **2** | **The cheap CEILING-PROBE (§6)** — quantify exactly where the current arch breaks (syntax depth, generation novelty, K-capacity, multi-attribute load) on the SHIPPED bridges, no retrain | **hours**, CPU/GPU, NO retrain | Pins the *measured* ceiling (the gap is currently asserted from per-capability docs, not one head-to-head) → a citable boundary the owner can see | cheap scaling / decision |
| **3** | **Richer fixed-grammar coverage** — add more frames / a small set of learned-syntax constructions (the serial-order generator already learns ORDER per frame); push vocab toward a corpus-grounded richer taxonomy | **weeks–1mo** | More word-order + lexical coverage *within* the structured-conversation paradigm; an honest "how far does fixed-frame + learned-order scale" boundary | cheap scaling |
| **4** | **IF the owner wants the generative axis: scope the learned generative-sequence substrate** (the benched surrogate-grad SNN cortex, `sim/bptt_snn*.py`) — a deep-research + cheap-first pass on whether a substrate-native autoregressive generator can produce novel sentences over the learned codes, with the moat | **months (research bet)**, high-variance | The ONLY path to free generation. Honest: toy-scale was ~4 orders below the scale that works; this is a genuine, uncertain research arc, NOT a wire-in. Owner-gated. | **the generative bet (the real LLM-gap closer, NOT the dendrite)** |
| **5** | **The dendritic D2 build** — REFRAMED as an artificial-life capstone, NOT a conversational unlock | **months (1.5–2 floor)**, highest-variance, protected hot-path `sim/` edit | Biology-fidelity for the *representation* (graded learned codes via per-compartment normalization). Does NOT close any categorical LLM gap (§3.5). Both dendrite jobs tested NEGATIVE for current walls. | artificial-life capstone (de-prioritized) |

**The decision the plan encodes:** *do (1) and (2) now* (cheap, bank the structured-conversation wins + measure the
ceiling). *Recognize that "small-LLM-competitive" is item (4) — a separate generative-substrate research bet — NOT the
dendrite.* *The dendrite (5) is a legitimate eventual artificial-life build but is the wrong tool for the
conversational/LLM gap and is correctly de-prioritized by the project's own NEGATIVEs.* This is the opposite of the
prompt's hypothesis ("is the dendritic rewrite the genuine unlock") — the evidence says **no, and there is a clearly
different (also-hard) mechanism (learned generative sequences) that is the actual unlock for the generative gap.**

---

## 6. The cheap empirical CEILING-PROBE design (no months-scale retrain; reuse shipped bridges)

**Goal:** QUANTIFY the §1/§2 gap with one runnable experiment, so the controller/owner sees the measured ceiling, not
an asserted one. **No retrain** — reuse the existing `vocab_ceiling_probe.py` harness and the self-generating codes
(the RF composer generates phasor codes from the seed; only a word-list is needed, so *no trained bridge load and no
GPU-hours of training are required* — it builds the agent and scores the matrix at ~3 min/cell).

**Entrypoint to reuse:** `research/runners/vocab_ceiling_probe.py` (builds `BrainConversationalAgent` = Hebbian parser
+ `RFPhasorComposer` + dlPFC; scores the 8-capability matrix pass/fail with the abstention-floor + shuffled-fact
anti-cheats already wired). Driver pattern: `research/findings/raw/_run_vocab_ceiling_multiseed.sh`.

**What to measure (four ceiling sweeps, each a cheap extension of the existing matrix):**

1. **K-capacity (fact load) ceiling.** Store K facts (K ∈ {8,16,32,64,128}) and measure recall + moat-separation +
   per-fact recall vs K. *Metric:* recall accuracy and unused-block-peak / stored-peak separation. *GO/NO-GO:* recall
   stays 1.00 and separation > (the abstain floor) until a measured K\* where it breaks → **K\* is the per-bridge
   capacity ceiling** (expected from the GAP-A doc: bounded by neuron budget, not SNR — so this measures the *neuron*
   budget cap, the genuinely-new number).
2. **Syntactic-depth ceiling.** Sweep embedded-clause nesting depth (1 → 2 → 3) at V=320, D ∈ {128, 256, 512}.
   *Metric:* recursive-clause recall pass/fail. *GO/NO-GO:* find the (depth, D) frontier where the recursive bound
   code falls below the algebra's SNR → **the syntactic-recursion ceiling and its code-dimension cost** (the V=320
   doc already shows depth-1 needs D≥256; this extends it to depth-2/3 = a measured grammar-depth wall).
3. **Multi-attribute load ceiling.** Sweep #attributes bound to one entity (1 → 2 → 3) on BOTH random codes and the
   learned/correlated stream codes. *Metric:* attributed-entity recall. *GO/NO-GO:* confirm the documented
   single-attribute-holds / two-attribute-degrades-on-learned-codes (~29%) boundary and find where 3-attribute
   collapses → **the binding-load ceiling, quantified on learned codes** (the F=3 resonator wall).
4. **Generation-novelty ceiling (the categorical-gap probe — the most important).** Store N facts, then ask the agent
   to `render_fact`/`describe` and **measure how many DISTINCT sentences it can produce vs how many were stored.**
   *Metric:* unique-generated-sentences / stored-facts ratio + a "novel-composition" check (can it emit any SVO triple
   it was NOT explicitly told?). *Expected (and the point):* the ratio is **≤ 1.0 and the novel-composition check is
   0** — i.e. it generates *only* stored facts, never a novel sentence. *GO/NO-GO interpretation:* a 0 novel-composition
   score **quantitatively confirms the categorical free-generation gap** — this is the number that makes "no free
   generation" measured, not asserted, and is the cleanest single piece of evidence for the owner that the LLM gap is
   categorical, not scale.

**Anti-cheats (carry the harness's existing battery):** the abstention floor (unstored cues → "I don't know", must
hold every sweep — the moat is *not* tuned on the test), the shuffled-fact permuted control (wrong queries abstain,
zero false hits), multi-seed (42/43/44 minimum). **The generation-novelty probe needs one new check** (~30 lines): a
set of held-out SVO triples never stored, assert `render_fact`/`query_patient` either abstains or returns only stored
content (never the held-out triple) → the novel-composition score.

**Why this is the right probe:** it is *hours not months*, *no retrain* (self-generating codes), *reuses a validated
harness*, and it converts the four gap claims (§1/§2) from documented-assertion into *measured ceilings on the shipped
system* — including the decisive generation-novelty=0 number that makes the categorical LLM gap concrete. A GO on
sweeps 1–3 (clean ceilings with the expected breaks) + the expected novelty=0 on sweep 4 is the empirical foundation
for the §5 decision: *scale the structured-conversation arch (cheap) up to its measured ceiling; the generative gap is
categorical and needs a different (generative-sequence) mechanism, not the dendrite.*

---

## 7. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (read in full / file+line cited):**
- The composer is a fixed exact-inverse FHRR algebra with templated emission — `rf_phasor_composer.py:1-13` (header),
  `:632` (`render_fact` = decode-stored + `f"{agent} {ac} {pt}"`), `:579` (`query_patient`), `:230` (`_bundle`),
  `:267` (`_unbind_phases`). Read in full.
- V=320 6-seed GO + the full capability matrix + moat 20/20 + two-attribute lifted + embedded-clause D≥256 floor —
  `2026-06-10-vocab-ceiling-multiseed-GO.md`, read in full.
- K=32 store, zero cross-talk, capacity bounded by neuron budget not SNR — `2026-06-18-one-brain-multifact-store-GAP-A-GO.md`,
  read in full.
- The dendritic fork ("build D2") — `docs/plans/2026-06-14-step3-cortex-fork-resolved-dendritic-D2-decision.md` +
  `2026-06-14-D2-dendritic-cortex-build-plan.md` (Phase 3 gate verbatim, line 21–22), read in full.
- **The three premise-overturning findings** (the most load-bearing claims in this doc): (a) D2 Phase-2 gain-not-load-bearing
  inversion (point-neuron +0.167 vs dendritic +0.042) — via `2026-06-17-dendritic-substrate-frontier-scoping.md` §1.3a
  quoting `2026-06-14-D2-phase1-DONE-phase2-frontier.md` §CORRECTION; (b) PPMI generalizing cortex delivered on point
  neurons — `2026-06-17-...-scoping.md` §1.3b + CLAUDE.md 2026-06-15; (c) off-diagonal de-risk NEGATIVE (lesion ==
  mechanism, +0.519) — `2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md`, read in full.
- **Both dendrite jobs NEGATIVE** — `2026-06-19-dendritic-binding-toy-derisk.md` (job a: memorizes 0.422 / held-out
  0.168, worse than FHRR 0.261), read in full; job b via the same doc §"thoroughly assessed" + CLAUDE.md commit log.
- Multi-attribute learned-bind NEGATIVE / fixed primitive load-bearing — `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`,
  read in full.
- Catalog G.02 "active dendrites — MISSING, ~10× compute/neuron" — `sim-catalog/.../feature-catalog.md`, read directly.
- SOTA: SpikeGPT (arXiv 2302.13939), SpikingBrain-7B (arXiv 2509.05276, ANN-to-SNN + 150B tokens), SpikeLLM (arXiv
  2407.04752), VSA/HDC (LARS-VSA arXiv 2405.14436; "Attention as Binding" arXiv 2512.14709) — fresh web pass; abstracts
  read.

**Could NOT fully verify (flagged honestly):**
1. **The exact §1 capability-by-capability gap is assembled from per-capability finding docs, not one head-to-head
   run.** That is *exactly* what §6's ceiling-probe is designed to measure — the gap is well-evidenced per-capability,
   but the single quantified head-to-head (esp. generation-novelty=0) is currently un-run. Confidence: high on the
   *categorical* nature (the composer structurally has no generator), to-be-measured on the exact frontier numbers.
2. **Whether a learned generative-sequence substrate (item §5.4) could reach useful generation** — genuinely open; the
   toy-scale SNN was ~4 orders below the working scale (`2026-05-07-Phase-2.3a-NEGATIVE`). I do not predict it; it is a
   research bet, flagged as such.
3. **The exact PPMI decimals (+0.518 / host +0.442 / ZCA +0.49)** — read from the CYCLE 88–96 findings, not re-run
   (read-only). Internally consistent across multiple docs; not load-bearing for the *direction* (the direction holds
   on the qualitative result: generalizing cortex shipped on point neurons, dendrite overturned).

---

## Sources

### Project record (re-verified this pass, file/finding cited)
- `research/runners/rf_phasor_composer.py`, `research/runners/one_brain_composer.py` (the production composers).
- `research/findings/2026-06-06-composer-vsa-idealization-known-limitation.md` (the algebra is a principled idealization).
- `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md` (the flat-vs-structured fork).
- `docs/plans/2026-06-14-step3-cortex-fork-resolved-dendritic-D2-decision.md` + `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md` (the D2 build plan + Phase 3 gate).
- `research/findings/2026-06-17-dendritic-substrate-frontier-scoping.md` (the premise-overturning re-examination — the single most important prior doc).
- `research/findings/2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md` (the off-diagonal NEGATIVE).
- `research/findings/2026-06-19-dendritic-binding-toy-derisk.md` (dendrite job a NEGATIVE); `2026-06-19-dendrite-credit-assignment-toy-stage1.md` (job b NEGATIVE).
- `research/findings/2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` (fixed primitive load-bearing).
- `research/findings/2026-06-10-vocab-ceiling-multiseed-GO.md`; `2026-06-18-one-brain-multifact-store-GAP-A-GO.md`; `2026-06-18-onebrain-320-scale-production-GO.md`.
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (the PPMI stream cortex — the learned cortex on point neurons).
- `docs/plans/2026-06-22-production-wiring-execution-plan.md` (the in-flight cheap-wins pass).
- `sim/dendritic_neuron.py`, `sim/dendritic_plasticity.py`, `sim/dendritic_mlp.py`, `sim/bptt_snn*.py`, `sim/surrogate_grad.py` (the dendritic + generative-substrate code).
- `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` G.02 (active dendrites — MISSING).

### Current literature (June 2026 pass)
- **SpikeGPT** — Zhu et al., arXiv 2302.13939 (generative SNN LM, 45M–260M, BPTT-trained, RWKV block).
- **SpikingBrain** — arXiv 2509.05276 (7B, ANN-to-SNN conversion + ~150B-token continual pretraining, Transformer-comparable).
- **SpikeLLM** — arXiv 2407.04752 (saliency-based spiking quantization of LLAMA).
- **LARS-VSA** — arXiv 2405.14436 (VSA for abstract-rule learning); **"Attention as Binding"** — arXiv 2512.14709 (transformer attention ≈ approximate VSA binding).
- Mikulasch-Priesemann PNAS 2021 (arXiv 2010.12395, the point-neuron analog/pre-spike whitening limit — the dendritic rationale, now overtaken by PPMI for the conversational cortex).
