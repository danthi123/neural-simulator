# Fluid conversation with a minimized transformer — mechanisms, biology, and a cheap-first roadmap

**Date:** 2026-07-01
**Type:** Deep-research + reference-catalog review (the standing research-gate opening move for a new direction).
**Owner directive (2026-07-01):** the NEW main priority is to *talk to the simulated brain like an LLM* — fluid
back-and-forth on almost any topic, understanding + responses grounded in the brain's OWN knowledge/experiences +
the running conversation, growing through the conversation. HARD constraint: rely on the actual transformer AS
LITTLE AS POSSIBLE; biologization / single spiking substrate / ONE brain / no-cheats all still bind.
**Scope of this doc:** the MECHANISMS + biology + literature + the transformer-minimization strategy + the ranked
cheap-first ROADMAP. A sibling assessment covers the current-state gap inventory. READ-ONLY; no `sim/`/GPU edit.

Terms defined once, on first use. Biology is cited from the catalog (`sim-catalog/references/feature-catalog.md`),
Kandel 6e, and current literature — not from memory.

---

## 0. TL;DR (read this first)

- **The transformer-minimization verdict is a spectrum, and the honest sweet spot is the MIDDLE, not the extreme.**
  Fully *transformer-free, brain-native, from-scratch* open-domain fluency is a **genuine wall** on a single GPU —
  it is the field's wall too (SpikeGPT overfits at scale; VSA "is fundamentally associative, not generative"; general
  conversation empirically needs ~360M+ params + trillions of tokens). The project has already reproduced every one
  of these walls independently (`2026-06-03-deep-research-how-the-field-gets-past-our-generative-conversation-wall.md`).
- **But the project is FAR closer to "minimal transformer" than the framing assumes.** It has already run an **88.6M
  generative transformer as a faithful SPIKING forward pass on its own resonate-and-fire substrate** (ppl_ratio 1.0
  over 12 layers — `2026-06-30-100M-C2-scaleup-C1-GO-C2-nuanced.md`). So "the transformer" is not an external black
  box bolted on: it is *already a spiking network that runs on the one brain*. The lever is to make that spiking
  generator **smaller, brain-trained, brain-gated, and brain-grounded** — not to delete it.
- **The single cheapest-first de-risk to START:** a **TinyStories-scale (~10–30M) generator, GATED + GROUNDED +
  VERIFIED by the brain, driven word-by-word in the incremental predictive-coding loop the brain actually uses**
  (Broca next-word prediction / N400 prediction error). This replaces the external Qwen-0.5B with a ~20–50× smaller
  model the project can train locally, keeps the no-confab moat and grounding, and is the minimal honest transformer.
  It needs **NO `sim/` edit** and **one modest local GPU run** (the TinyStories curriculum train). It is composing
  already-validated pieces (Gen-F trainer + spiking-forward + composer gate/verify), not a new mechanism class.
- **The one genuinely-new, research-gated mechanism worth a cheap probe** (higher ceiling, higher risk) is
  **thalamocortical dynamical gating** (Logiaco-Abbott-Escola 2021) — the field's leading *circuit-level generative
  sequencer*, distinct from everything the project tried, and **partly already in the g11 BG→thalamus cascade**. It is
  the only credible path to *transformer-free* generation that is compositional-by-construction. Probe it AFTER the
  cheap TinyStories slice lands, as a parallel science bet.

---

## 1. The transformer-minimization verdict — the spectrum, honestly ranked

The question "can the brain's own spiking mechanisms be fluent open-domain WITHOUT a transformer?" resolves onto a
spectrum. Ranked from most-brain-native to most-pragmatic, with the honest cost/feasibility of each.

### (i) Transformer-FREE brain-native generation — how fluent, at what cost?

The project's transformer-free generators and where each tops out:

- **VSA composer render (`OneBrainComposer` / `RFPhasorComposer`, `rf_phasor_composer.py:824-845`) + neural
  serial-order renderer (`neural_serial_order_renderer.py`).** "VSA" = *vector-symbolic architecture*: concepts and
  roles are high-dimensional vectors, bound by an invertible algebra. The project's is **FHRR** = *Fourier Holographic
  Reduced Representation* (unit-magnitude phasors; bind = phase addition via complex synapses), run on **resonate-and-
  fire** neurons (a neuron whose complex internal state rotates; a zero-crossing = a spike carrying phase). This
  produces **grounded, hallucination-proof, structured output**: it decodes a stored fact (agent/action/patient) and a
  spiking competitive-queuing generator orders the words (catalog **G.07 pre-SMA / SMA internally-generated sequences**;
  **H.19 premotor/SMA sequential action**; Grossberg 1978, Bullock-Rhodes 2003). **Ceiling: this is fact reconstruction
  in fixed frames, NOT open-domain generation.** The field agrees categorically: the one paper asking this exact
  question ("Bridging Cognitive Architectures and Generative Models with VSA," AAAI-SS) concludes VSA binding is
  *"fundamentally associative rather than generative"* and open generation requires *coupling VSA to a separate neural
  generator*. Spaun (2.5M spiking neurons) outputs drawn digits, never utterances. **Cost to make it fluent: unbounded
  — it is the wrong tool for free generation.** Its value is the *grounding + verification half*, which it does better
  than anything else the project has.

- **BPTT-SNN generator (`sim/bptt_snn.py`, `bptt_snn_gpu.py`, `surrogate_grad.py`).** A spiking network trained
  end-to-end by *backpropagation-through-time* with a *surrogate gradient* (a smooth stand-in for the non-
  differentiable spike). This is the from-scratch-spiking route. **Ceiling: the project measured it and hit the field's
  exact wall** — `2026-06-02-generative-ceiling-spiking-LM-NEGATIVE-overfit-not-size.md`: the spiking LM got *worse*
  with scale (overfit), which SpikeGPT's own authors report as *"potentially suffering from over-fitting"* (surrogate-
  gradient mismatch + vanishing temporal gradients + limited-data). Literature confirms **no direct-trained spiking net
  at single-GPU scale produces coherent open conversation.** SpikeGPT (46M/216M, spiking-RWKV) reaches GPT-2-small-tier
  perplexity on *easy* corpora (WikiText-2 test ppl ~18, competitive with GPT-2-small's ~37.5) but **makes no coherence
  claims and falls behind on WikiText-103** — it cannot absorb large-corpus world knowledge from scratch. **Honest
  cost: from-scratch spiking fluency is not feasible on this hardware.**

- **The critical nuance that reframes (i):** the project does NOT train a spiking transformer from scratch. It trains a
  small ANN transformer (`sim/tiny_transformer.py` — a 4-layer decoder-only GPT) and runs it as a **faithful spiking
  forward pass** on the RF substrate (`_genseq_allspiking_forward_compose_derisk.py:28-42` — every matvec on the RF
  complex-synapse accumulator; LayerNorm/GELU/softmax through shipped spiking circuits). This *conversion* route is
  exactly what the field found works: *"the capable spiking LMs are conversions/distillations of pretrained
  transformers"* (SpikeLLM, SpikingBERT, SpikeLM, FAS). **So the project's "transformer-free at run-time" is already
  achieved for an 88.6M model** — the residual "transformer" is (a) the ANN *training* and (b) the host-distilled
  *weights* (an explicitly-deferred residual, per the BRAIN-BASED-ONLY standard). This is why the verdict is not
  "delete the transformer" but "shrink and brain-integrate it."

### (ii) MINIMAL transformer — smaller / brain-trained / on-substrate (the honest sweet spot)

This is the recommended target. Three independent levers each shrink or biologize the transformer:

1. **Shrink it 20–50× by constraining the domain/curriculum (TinyStories regime).** The decisive external result:
   **TinyStories (Eldan & Li 2023)** shows models **below 5–10M params produce fluent, grammatical, coherent
   multi-paragraph English** when the *training data* is simplified to a 3–4-year-old vocabulary, whereas ~125M is the
   threshold for coherent open-domain English on *standard* corpora. A 28M TinyStories model beat GPT-2-XL (1.5B) on
   its domain. **This means a ~10–30M generator is enough for fluent conversation at the project's actual scale (the
   develop-loop vocab is tens→hundreds of words), and the project already runs 88.6M spiking-faithfully** — so the
   compute headroom is proven. The Qwen-0.5B is ~15–50× larger than needed for the current domain.

2. **Brain-train it (continual pretraining, tiny data) instead of importing frozen weights.** **SpikingBrain (2025,
   arXiv:2509.05276)** — brain-inspired hybrid-linear + spiking-attention LLMs (7B linear; 76B hybrid-MoE) — reaches
   transformer-baseline quality with **continual pre-training on <2% of the usual data (~150B tokens)** and ~69%
   activation sparsity, and demonstrably trains on **non-NVIDIA hardware**. The transferable principle (not the 7B
   scale): a **hybrid architecture (sliding-window local attention + linear/low-rank global kernels + adaptive spiking
   neurons)** is *much cheaper to train and more brain-shaped* than full quadratic attention, and can be **grown by
   continual pretraining** rather than from scratch. This is the architecture to prefer if/when the project trains its
   own generator, and it dovetails with growth-through-conversation.

3. **Make the attention itself linear/recurrent (RWKV-style) so it maps onto the substrate's own dynamics.** **RWKV /
   SpikeGPT** replace quadratic self-attention with a **linear-cost recurrent state** — O(1) inference, a single
   fixed-size recurrent state, *"on par with similarly-sized transformers"* at matched scale. This is important because
   the brain generates language **incrementally, word-by-word, with a recurrent state**, NOT by parallel all-to-all
   attention over the whole context (see §2, comprehension). A recurrent/linear generator is therefore *both* the
   biologically-faithful choice *and* the cheap-inference choice. Honest limit: linear attention is weaker at
   long-context exact recall — but the project's grounding memory (VSA store) supplies exact recall separately, which is
   precisely the hybrid the field recommends.

**⇒ The minimal transformer is: a ~10–30M hybrid-linear/RWKV-style spiking generator, TinyStories-curriculum-trained
(and later continually pretrained on the develop-loop stream), run as a spiking forward on the RF substrate, GATED +
GROUNDED + VERIFIED by the brain.** Cost: one modest local train (hours, not the 3-day 88.6M run), no `sim/` edit for
the first slice. Feasibility: HIGH — every piece is validated or externally demonstrated.

### (iii) Current grounded-lang faculty (Qwen-0.5B fluency + brain grounding) — the FALLBACK

`2026-06-23-grounded-lang-INTEGRATION-GO.md`: the converted spiking Qwen2.5-0.5B renders the brain's retrieved facts
into fluent prose, **GATED** (composer decides whether there is content) + **VERIFIED** (parser re-parses the output,
rejects role-inversion drift). This already works end-to-end and *already caught a real hallucination* — the moat holds
with a real generative LLM in the loop. **This is the honest fallback if the smaller brain-trained generator
underperforms.** It is the most pragmatic and least-brain-native point on the spectrum; the owner's directive is to
move OFF it toward (ii). Keep it as the reference ceiling and the numpy-CPU/test path (mirroring how `rf` composer is
retained as the test oracle).

---

## 2. Ranked cheap-first mechanisms — one per gap

Each: the biological mechanism (catalog-cited) + reusable project machinery + the transformer-minimization angle + a
cheap-first de-risk + the anti-cheats. Ordered by leverage-per-cost.

### GAP A — Fluent free-form generation *(the headline gap; highest leverage)*

- **Biology.** Language production is incremental and predictive: Broca's area (**catalog G.12**) maps stored word-forms
  to articulation and builds structure left-to-right; the brain **predicts the next word** and emits a prediction-error
  signal (the N400) when it mismatches (Nature Rev. Psychology 2024; Kandel 6e Ch 55). Frame-and-slot planning
  (Hagoort Unification) sets the sentence skeleton; a serial-order generator (**G.07 / H.19**, competitive queuing)
  converts the parallel plan into a word sequence.
- **Reusable machinery.** `sim/tiny_transformer.py` + `tiny_transformer_train.py` (Gen-F trainer, kill-safe resume);
  the all-spiking forward (`_genseq_allspiking_forward_compose_derisk.py`); `sim/bpe_tokenizer.py`; the RF spiking
  substrate; the composer gate + parser-verify from the grounded-lang arc.
- **Transformer-minimization angle.** Swap the external 0.5B for a **~10–30M TinyStories-curriculum generator**
  (proven-fluent-at-scale by TinyStories) → 15–50× smaller, trainable locally, and small enough that **bridge
  co-residence is cheap** (the 88.6M already ran; 10–30M is trivial by comparison). Prefer a **hybrid-linear/RWKV**
  block so generation is the recurrent, word-by-word process the brain uses and inference is O(1).
- **Cheap-first de-risk.** Train the ~10–30M generator on TinyStories (local GPU, hours). Run it through the EXISTING
  grounded-lang loop (gate→constrain→verify) in place of Qwen. Measure: fluency (held-out ppl on the domain),
  grounded-correctness (re-parse == taught fact), abstention (untaught → no sentence), drift-caught (steered-to-false →
  VERIFY rejects). GO bar: fluent-correct ≥ the Qwen baseline's 4/4 pattern on the de-risk curriculum, moat 0-FA.
- **Anti-cheats.** (1) The verify re-parse is the moat — a fluent-but-false render must be REJECTED, never emitted.
  (2) Held-out generalization (never-trained sentences), not train-set memorization. (3) An untrained-generator control
  (random init) must fail. (4) 6-seed for any robustness claim.

### GAP B — Open-domain breadth *(the true scale wall; manage, don't "solve")*

- **Biology.** Semantic breadth is the ventral "what" stream + anterior temporal hub (**catalog G.11 dual-stream / G.13
  Wernicke**); word meanings are learned from exposure and stored in distributed cortical ensembles (Pulvermüller
  G.20). Humans acquire ~tens of thousands of words over years of streamed exposure.
- **Reusable machinery.** The **stream-cortex** that learns word meaning online from the conversation stream
  (`2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md` — online Hebbian co-occurrence + running-
  frequency normalization reaches batch-PPMI quality, +0.513, generalizes 0.91); the **develop loop** (`develop_run.py`,
  vocab 6→24 over simulated days); the **VSA retrieval + no-confab store** as the grounded knowledge base.
- **Transformer-minimization angle.** Breadth is where a from-scratch brain-native model genuinely cannot compete on a
  single GPU (BabyLM: 10–100M words buys grammar but *"not world knowledge"*; general conversation needs ~360M+ params
  + trillions of tokens). The honest move is **retrieval-augmentation**: the brain's validated VSA store + abstention IS
  the hard, distinctive half of a RAG (retrieval-augmented generation) system that most RAG lacks. A small generator
  *conditioned on the retrieved grounded facts* covers far more ground than its parameter count would (RETRO: 25×
  smaller matches larger; RETRO-LI works at small sparse DBs — the project's regime).
- **Cheap-first de-risk.** No new build first — CHARACTERIZE the boundary: measure the domain where the TinyStories
  generator + VSA-retrieval is fluent-AND-grounded vs where it must abstain. The abstention IS the honest breadth story
  (the brain says "I don't know that" rather than confabulating).
- **Anti-cheats.** Abstention-floor 0-FA on untaught topics; retrieval must be load-bearing (lesion the store → the
  generator loses grounding, VERIFY rejects).

### GAP C — Arbitrary-input comprehension *(parser breadth)*

- **Biology.** Comprehension is the ventral sound→meaning stream (**G.11/G.13 Wernicke**); the dorsal stream maps to
  production. Comprehension is robust to word order / case / morphology via cue-competition (Bates-MacWhinney).
- **Reusable machinery.** The Hebbian-learned `BridgeParser` (voice-invariant role assignment); the robust multi-cue
  parser direction (owner note `project_conversational_primary_robust_multicue_parser`); the stream-cortex for OOV
  (out-of-vocabulary) word meaning.
- **Transformer-minimization angle.** Comprehension is *encoding-into-the-store*, not generation — the transformer is
  not needed here at all. The brain's own parser + stream-cortex already handle it; the frontier is breadth of
  syntactic frames, not fluency.
- **Cheap-first de-risk.** Feed the parser held-out sentence structures; measure role-assignment accuracy vs a permuted
  control. (This is largely a follow-on to the existing parser arc, not a new mechanism.)
- **Anti-cheats.** Permuted-word control at chance; the stream-cortex OOV path lesion collapses novel-word comprehension.

### GAP D — Multi-turn dialogue coherence

- **Biology.** Discourse referents are held in prefrontal working memory (**catalog G.08 PFC persistent activity**,
  Rainer/Asaad/Miller 1998, D1-dependent) across turns; theta-gamma multiplexing indexes ordered items in WM (Lisman-
  Jensen). Pronoun resolution binds a current word to a held referent.
- **Reusable machinery.** `MultiTurnAgent` + the `SpikingLoopContextBuffer` (persistent discourse referents across
  turns; validated 2026-06-17 anaphora GO); the theta-gamma **mode-unification** + **SPEAR** runners (Hasselmo separate-
  phases-of-encoding-and-retrieval; theta = encode/retrieve gating); the dlPFC dialogue planner.
- **Transformer-minimization angle.** Coherence is a *state-maintenance* problem the brain solves with recurrent WM +
  theta gating — exactly what a linear/recurrent (RWKV) generator's recurrent state ALSO provides. A recurrent generator
  is inherently more multi-turn-coherent per parameter than a context-window transformer. The known limit — **multi-
  REFERENT disambiguation needs winner-take-all biased-competition inhibition** between referent attractors (two
  converging NEGATIVEs, `2026-06-17-multireferent-disambiguation-NEGATIVE.md`) — is the precise specified next mechanism.
- **Cheap-first de-risk.** The multi-referent WTA inhibition probe: two held referents, a bare pronoun, biased-
  competition inhibition between the attractors → the salient/grammatically-cued referent wins. GO bar: correct
  binding > recency baseline, lesion (no inhibition) collapses to chance.
- **Anti-cheats.** Empty-WM abstains; reset/lesion breaks the carry; permuted salience at chance.

### GAP E — Grounding in the brain's OWN knowledge

- **Biology.** Grounding = the same distributed ensembles that store perception/experience supply the content for
  language (embodied semantics, Pulvermüller; ATL convergence hub, Patterson-Lambon Ralph). Meaning is not in the words
  but in the sensory/experiential codes the words point to.
- **Reusable machinery.** This is the project's STRONGEST asset — already comprehensively validated: perceive→ground→
  compose on one brain (`navigate_to_compose_then_answer.py`, 6-seed GO); shared grounded codes dissolve the rate-vs-
  phasor wall for perceived objects; the stream-cortex learns codes FROM conversation; the no-confab moat + parser-
  verify catch ungrounded assertions.
- **Transformer-minimization angle.** Grounding is precisely the half the transformer must NOT do — the owner's whole
  decoupling. The generator supplies fluency; the BRAIN supplies + verifies content. This gap is essentially SOLVED for
  the current scale; the work is to keep grounding load-bearing as the generator shrinks (the gate/verify already do
  this).
- **Cheap-first de-risk.** Already done repeatedly; the drop-in test is: does the smaller generator still get its
  content GATED by the brain (untaught → abstain) and VERIFIED (drift → rejected)? (Folded into GAP A's de-risk.)
- **Anti-cheats.** The drift-caught + untaught-abstain arms from the grounded-lang arc, verbatim.

### GAP F — Growth through conversation

- **Biology.** New knowledge is encoded by the hippocampus and consolidated to cortex during sleep replay (McClelland
  1995 complementary learning systems; sharp-wave-ripple replay, Buzsaki), preventing catastrophic forgetting; new word
  meanings accrue via streamed Hebbian exposure.
- **Reusable machinery.** The **develop loop** (`develop_run.py` / `_longitudinal_develop_loop_gpu`) — real stream-cortex
  Hebbian learning grows vocab + facts over simulated days with retention 1.0, moat 0-FA, resumable; **self-replay
  consolidation** causally prevents forgetting; and the **C2 grow-without-forget** result on the *generative* model at
  100M (`2026-06-30-100M-C2-...` — the spiking-consolidated 88.6M generator learns a new in-band task AND retains its
  domain, replay dose-monotone).
- **Transformer-minimization angle.** This is where **SpikingBrain's continual-pretraining-on-tiny-data** principle and
  the project's **self-replay** converge: a small generator can be **grown by continual pretraining on the develop-loop
  stream** (the conversation itself becomes the corpus), consolidated by self-replay, without retraining from scratch.
  The generator LEARNS from talking, exactly the owner's goal.
- **Cheap-first de-risk.** Extend the C2 loop to the small generator: talk → new facts enter the store (VSA) AND the
  stream-cortex → periodic self-replay consolidation → confirm retention + no-forget. GO bar: new content recalled +
  old retained ≥85% + replay dose-monotone (the C2 bars, already met at 88.6M).
- **Anti-cheats.** Frozen-brain control learns 0 (validated in the develop loop); no-replay control forgets more (dose-
  monotone); moat 0-FA maintained across growth.

---

## 3. Phased roadmap — cheapest, highest-leverage first

The order is: prove the minimal generator closes the headline gap, THEN broaden, THEN take the higher-ceiling science
bet. Each phase names whether it is *composition of validated pieces* or a *research-gated new mechanism*.

**PHASE 0 — the minimal-transformer generator slice (cheapest, highest leverage; COMPOSITION).**
- Train a **~10–30M hybrid-linear/RWKV-style generator on TinyStories** (local GPU, hours; reuse `tiny_transformer_train`
  scaffold, swap the block to a linear/recurrent variant OR keep the current decoder-block and just shrink — start with
  the shrink, it is zero-new-code).
- Drive it through the EXISTING grounded-lang gate→constrain→verify loop in place of Qwen-0.5B.
- **This one slice closes/tests GAP A (fluency), GAP E (grounding, drop-in), and half of GAP B (abstention as the
  breadth story) at once.** No `sim/` edit. Verdict target: fluent-correct + moat-intact at ≥ the Qwen baseline on the
  de-risk curriculum, with a 15–50× smaller, locally-trained, brain-gated generator.

**PHASE 1 — recurrent/on-substrate + multi-turn (COMPOSITION, one small optional `sim/` seam).**
- Move the Phase-0 generator to a **recurrent/linear block** (if Phase 0 started with the plain shrink) so generation is
  word-by-word with an O(1) recurrent state (biologically faithful; cheap inference; better multi-turn per parameter).
- Run it as a **spiking forward on the RF substrate** (the 88.6M path already proves this composes; a 10–30M model is
  trivial) → the generator is now literally on the one brain.
- Add the **multi-referent WTA biased-competition** probe for GAP D (the one specified missing dialogue mechanism).

**PHASE 2 — growth through conversation (COMPOSITION of the develop loop + C2).**
- Wire the generator into the **develop loop**: conversation → store + stream-cortex → self-replay consolidation →
  no-forget confirmation (the C2 bars). The brain grows its knowledge AND (optionally) continually-pretrains the small
  generator on its own conversation stream (SpikingBrain principle). Closes GAP F end-to-end.

**PHASE 3 — the research-gated generative bet (NEW MECHANISM CLASS; parallel science track).**
- **Thalamocortical dynamical gating** (Logiaco-Abbott-Escola 2021, *Cell Reports*) — the only credible path to
  *transformer-free* compositional generation. Cortex stores no sequence; it is a fixed recurrent dynamic pattern-
  generator whose *mode* is switched by BG-disinhibited thalamic low-rank perturbation → any-to-any transitions with no
  transition-specific learning. **Partly already the g11 BG→thal cascade** (`cortex_X → str_D1_X → gpi_X → thal_X`); the
  missing pieces are (a) a reconfigurable recurrent cortical generator (current pools are static attractors) and (b) a
  shared generic preparatory loop. This is the higher-ceiling, higher-risk route; it fires the research gate because it
  is a new mechanism class. Cheap probe: does thalamic low-rank gating of a fixed recurrent cortical RNN produce
  arbitrary motif orderings? Honest caveat: validated for MOTOR sequencing; language application is a reasonable
  extrapolation, not established.
- **(Optional, folds into Phase 0–1 as a training-protocol change)** a **resonator-network decoder + noise injection**
  (Frady 2020; Kymn 2024, ≥50× capacity; spiking stochasticity supplies the noise free) between VSA-unbind and emission
  — the field's universal fix for "algebra works, substrate fails," recommended as the strongest near-term VSA lever in
  the 2026-06-03 synthesis. This lifts the compose ceiling but is about the *grounding/composition* half, not free
  generation, so it is secondary to Phase 0.

---

## 4. Honest boundaries — where the walls are real

- **From-scratch transformer-free open-domain fluency is a genuine wall on this hardware.** Independently reproduced by
  the project (`2026-06-02-generative-ceiling-...`, spiking LM overfits with scale) and the field (SpikeGPT's own
  overfitting admission; no direct-trained spiking net is fluent open-domain at single-GPU scale). **The lever here is a
  minimal transformer (converted/distilled/continually-pretrained), not more from-scratch spiking effort.** This is the
  point-neuron / rate-code family of walls already documented: a rate code cannot do what an analog pre-spike
  computation does, and open-vocabulary generation is not a VSA operation.

- **VSA/composer cannot free-generate — by theorem, not bug.** Binding is associative, capacity is a crosstalk theorem
  (Frady-Sommer: ~½ bit/neuron, SNR ∝ √D/√M). The composer is the grounding + verification half; it will never be the
  fluency half. Accept this and keep it in its role.

- **Open-domain BREADTH needs scale the project doesn't have from scratch** (~360M+ params + trillions of tokens for
  general conversation; BabyLM shows 10–100M words buys grammar but not world knowledge). **The honest answer is
  retrieval-augmentation** (the brain's VSA store + abstention is the distinctive hard half) + domain-constraint
  (TinyStories regime makes small models fluent) + the abstention moat as the truthful "I don't know" boundary.

- **Where the DENDRITIC substrate / SCALE is the real lever, not a cheap fix:** *generalization across similar concepts*
  was resolved WITHOUT the dendritic rewrite (2026-06-16 arc), so that is NOT a wall here. The remaining substrate-deep
  item is *learned multi-attribute binding from scratch* (bundling has no inverse on point neurons — 2026-06-16), but
  that is a composer-internal frontier, not on the fluency critical path. Scale (a bigger corpus for the base generator)
  is the lever for lower base perplexity (the 88.6M model is *data-bound* at 41M tokens), but a bigger corpus is a data
  problem, not a substrate wall.

- **Where an honest minimal-transformer IS the pragmatic answer:** the fluency half, full stop. The owner's constraint
  is "as little as possible," and the honest minimal is a small (~10–30M), brain-trained-or-continually-pretrained,
  brain-gated, brain-grounded, spiking-on-substrate generator — not zero transformer. Claiming zero-transformer fluency
  would be an overclaim the project has already falsified.

---

## 5. Recommendation — the single cheapest-first de-risk to START

**START PHASE 0: train a ~10–30M TinyStories-curriculum generator and run it through the EXISTING grounded-lang
gate→constrain→verify loop in place of the Qwen-0.5B.**

- **Why this first:** it is the highest leverage per unit cost — one modest local train tests the entire
  transformer-minimization thesis (a 15–50× smaller, locally-trained generator can be fluent AND stay brain-gated +
  brain-grounded + hallucination-proof), and simultaneously exercises GAP A (fluency), GAP E (grounding drop-in), and
  the GAP B abstention story. It builds directly on the two strongest validated assets: the Gen-F trainer/spiking-
  forward and the grounded-lang gate/verify loop.
- **Does it need a `sim/` edit?** **No** — it is reuse-by-import (a new/shrunk generator config + the existing loop). The
  later bridge co-residence of the small generator reuses the already-validated 88.6M spiking-forward path (no new
  `sim/` mechanism; a small model is trivial for a path that held 88.6M).
- **Does it need a long GPU run?** **A modest one** — a TinyStories-scale train is hours on the 3090 (the 88.6M base was
  the 3-day run; ~10–30M is far cheaper), well inside the "long local runs OK with ETA" policy, and no cloud (no VRAM
  wall — the 88.6M already fit).
- **Does it need a new mechanism class?** **No.** Phase 0 composes validated pieces. The only research-gated NEW
  mechanism on the roadmap is Phase 3 (thalamocortical dynamical gating), which is the *parallel higher-ceiling science
  bet* and explicitly fires the research gate — it is NOT the first move.

**Verdict in one line:** the brain's own spiking mechanisms can supply *grounding, verification, comprehension, dialogue
state, and growth* — but NOT from-scratch open-domain *fluency*; the honest minimal-transformer answer is a small,
brain-trained/gated/grounded, spiking-on-substrate generator, and the cheapest first step to prove it is a
TinyStories-scale generator dropped into the existing grounded-lang loop.

---

## Key citations

**Project (file:line / finding):** `sim/tiny_transformer.py` (4-layer decoder GPT); `research/runners/tiny_transformer_train.py`
(Gen-F trainer); `_genseq_allspiking_forward_compose_derisk.py:28-42` (all-spiking forward on RF); `rf_phasor_composer.py:824-845`
(templated render_fact); `neural_serial_order_renderer.py` (spiking competitive-queuing order, G.07/H.19);
`2026-06-30-100M-C2-scaleup-C1-GO-C2-nuanced.md` (88.6M spiking == ANN; grow-without-forget); `2026-06-23-grounded-lang-INTEGRATION-GO.md`
(Qwen-0.5B fluency + brain gate/verify, hallucination caught); `2026-06-02-generative-ceiling-spiking-LM-NEGATIVE-overfit-not-size.md`
(from-scratch spiking overfits); `2026-06-03-deep-research-how-the-field-gets-past-our-generative-conversation-wall.md`
(the field-wall synthesis + resonator/thalamocortical/hybrid escapes); `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md`
(stream-cortex learns word meaning); develop loop (`develop_run.py`); `2026-06-17-multireferent-disambiguation-NEGATIVE.md`
(the specified WTA dialogue mechanism).

**Catalog (`sim-catalog/references/feature-catalog.md`):** G.07 pre-SMA internally-generated sequences (Kandel 6e Ch 34 p822-828);
G.08 PFC working memory (Ch 52 pp1292-1294); G.10 language as hierarchical symbolic system (Ch 55 pp1370-1372); G.11 dual-stream
language / Hickok-Poeppel (Ch 55 pp1380-1387); G.12 Broca production (Ch 55 pp1382-1384); G.13 Wernicke comprehension (Ch 55
pp1384-1385); H.19 premotor/SMA sequential action (Ch 34 p822-835).

**Literature (current, cited):** Eldan & Li 2023, *TinyStories* (arXiv:2305.07759) — <5–10M coherent on constrained domain, ~125M
threshold on standard corpora. Zhu et al. 2023, *SpikeGPT* (arXiv:2302.13939) — spiking-RWKV 46M/216M, WikiText-2 ppl ~18, overfits
at scale, no coherence claim. Peng et al. 2023, *RWKV* (arXiv:2305.13048) — linear-cost RNN on-par with transformers at matched scale.
*SpikingBrain* 2025 (arXiv:2509.05276) — hybrid-linear + spiking-attention LLM, continual pretraining on <2% data, non-NVIDIA training.
*Training-Free ANN-to-SNN Conversion for Spiking Transformers* 2025 (arXiv:2508.07710) — near-lossless conversion (ViT/RoBERTa/GPT-2).
Nature Rev. Psychology 2024, *Neural evidence of word prediction*; N400 predictive-coding accounts — incremental word-by-word production
+ prediction error. Logiaco, Abbott & Escola 2021, *Cell Reports* — thalamocortical flexible sequencing. Frady, Kent, Olshausen & Sommer
2020, *Neural Computation* — resonator networks; Kymn et al. 2024 (arXiv:2412.00354) — noise injection ≥50× capacity. Lake & Baroni 2023,
*Nature* — MLC (1.4M params, single GPU). Hoffmann et al. 2022 — Chinchilla scaling. Reservoir/ESN + HTM: never scaled to generalized
language modeling (Nature Comms 2024 reservoir review; HTM surveys) — an honest negative for those transformer-free routes.
