# Simulate-Broca — REPLACE the ANN generator with SIMULATED SPIKING CIRCUITRY for language PRODUCTION (research gate, read-only scoping)

**2026-07-03. Read-only deep-research SCOPING doc.** No code edited/built/run. Produced BEFORE committing build/GPU
resources, per the standing "deep-research FIRST at a new direction" gate (CLAUDE.md "Standing practice" + the master
directive `project_master_directive_relentless_biological_emergence`). Controller to trust-but-verify the load-bearing
claims (flagged inline), push, present the recommendation.

**THE DIRECTION.** The north-star wire is complete (EMERGE-56/57/58 GO): the emergent spiking substrate DISCOVERS
categories, REASONS (inheritance / per-dimension cancellation / taxonomy / sibling-abstain), and now SPEAKS its grounded
answers FLUENTLY behind a gate-first no-confab moat. BUT the fluent "Broca articulation" is a re-fine-tuned **~21M
TinyStories transformer ANN** — a TRACKED TEMPORARY SCAFFOLD. The master directive is explicit: **"simulate Broca, don't
bolt on an LLM"** — the end state SIMULATES the production circuitry (fully-spiking, one brain, emergent, no permanent
external ML artifact). This doc scopes the honest biological path to REPLACE that ANN with SIMULATED SPIKING CIRCUITRY for
language PRODUCTION = **surface realization**: turning a grounded message/proposition into a fluent word sequence.

This is a SURPASS round (the four mandatory moves below), NOT "diagnose + rank."

---

## MOVE 1 — ISOLATE + QUANTIFY THE GENUINE RESIDUAL

Most of "fluent production" is ALREADY spiking-realizable in this repo. Accounting for the project's OWN validated
production machinery so we don't re-scope solved parts:

### 1a. What the surface-realization job decomposes into (the Levelt/Bock blueprint — verified against literature)

Psycholinguistics is unambiguous that speech production is NOT one monolithic language model. Levelt's *Speaking* /
Levelt-Roelofs-Meyer 1999 blueprint + Bock-Levelt grammatical encoding + Dell 1986 decompose surface realization into
SEPARABLE sub-processes (`pages.ucsd.edu/~scoulson/cogs179/Levelt.pdf`; Bock & Levelt 1994):
1. **Message / conceptual** — what to say (the proposition). **← already the brain's job, DONE**: EMERGE reasons and
   emits `(gate, subject, property, polarity, frame-kind)`; the dlPFC content-selection Control (`content_selection_spiking.py`)
   picks WHAT. Not in scope here.
2. **Lemma selection (lexical access)** — pick the abstract word (meaning + grammatical class), NOT its sound. Frame-and-slot
   grammatical encoding: a syntactic **frame** with **slots** labelled by grammatical class, filled by selected lemmas
   ("the moment of selection is determined by the developing syntactic frame"). **← mostly DONE**: the emergent lexicon
   (EMERGE-30..55 discovered category codes + G.20 sparse concept codes) IS the lemma store; A→W read-out
   (`concept_speak_demo`, 100% multi-seed) maps a selected concept-pool → its word.
3. **Positional / grammatical encoding (serial order of the frame)** — order the slots (SVO vs who/what vs yes-no vs
   modal-ability), insert function words, agreement. **← the SERIAL-ORDER part is DONE on spikes**: the rate-coded
   COMPETITIVE-QUEUING serial-order generator (`neural_serial_order_renderer.py` + `_phaseB_serial_order_{cq,spiking}_derisk.py`),
   6/6-seed GO on the real spiking substrate (findings `2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`):
   a frame sets a primacy gradient (graded current) → per-pool spiking RATE ranking = emission order, beating
   permuted-order + equal-drive controls. And **`FrameCQ`** (`_phaseB_serial_order_multiframe_derisk.py`, 6/6 GO on
   spikes) already learns **frame-CONDITIONED** orders (`prim[frame][role]`) with a decisive cross-frame control (0.000)
   — distinct orders for distinct reply frames = the seed of syntax, spiking.
4. **Phonological / phonetic encoding + articulation** — spell each ordered lemma into its form and emit. **← the
   word-level spell is DONE** (A→W); sub-word phonology is out of scope (we emit whole word-forms, legitimately — the
   English orthographic word IS the emission unit for a text console, the "body" boundary).

**So of "fluent English production", the parts covered by already-validated spiking machinery are: message (brain),
lemma store (emergent lexicon), lemma→word read-out (A→W), single-frame serial order (CQ, spikes), and frame-CONDITIONED
serial order (FrameCQ, spikes).**

### 1b. Is "run the trained generator as a spiking forward" a legitimate on-substrate realization? — NO (verified)

The repo HAS a validated spiking-forward conversion: the 24-layer Qwen ran on the SimulationBridge RF substrate bit-exact
(`2026-06-23-bridge-coresidence-DEMONSTRATED.md`, ppl 7.041 == ANN, logit cos 1.0), and the ~88.6M generator's spiking
forward == the ANN at ppl_ratio 1.0 (`2026-06-30-100M-C2-scaleup-C1-GO-C2-nuanced.md`). **Tempting to call the scaffold
"already spiking." It is NOT a legitimate close, for a precise reason tied to `feedback_spiking_structure_must_self_organize`:**
running the transformer as a spiking forward makes the OPERATIONS spike, but the STRUCTURE (all weights) was **trained by
backprop off-substrate on TinyStories + RA + EMERGE frames, then INJECTED**. Per the owner's 2026-06-20 challenge, that is
"spiking at RUNTIME, host-DESIGNED at the STRUCTURAL level" — the exact residual that (i) breaks the "self-contained,
develops-its-own-structure, ONE emergent brain" claim, and (ii) requires the host to compute + inject the weights for any
neuromorphic port. It is ALSO a permanent external-ML artifact (the master directive's explicit "don't bolt on an LLM").
⇒ the spiking-forward conversion is a legitimate way to *run* a converted model, but NOT the close for "simulate Broca";
the residual is that the PRODUCTION KNOWLEDGE must be **learned/self-organized on-substrate from experience**, not
backprop-trained off-substrate.

### 1c. THE GENUINE RESIDUAL — named in mechanism terms (the ~25%)

Coverage estimate (honest, not a measured %): **~75% of the ANN's actual job in THIS console is already
spiking-realizable** by composing (lemma store + A→W spell + FrameCQ serial order + the gate-first moat). The generator's
role in EMERGE-57/58 is narrow — render a handful of fixed frames (`the {S} can {V}.`, `the {S} {intr3sg}.`, "yes"/"no",
"I don't know…"), plus subject/verb inflection. The **TRULY-MISSING ~25%** is the part FrameCQ+A→W do NOT yet supply — a
from-experience-LEARNED recurrent spiking language-production cortex that GENERALIZES surface realization:

- **(R1) Function-word insertion** — the articles/copulas/modals/negators ("the", "a", "can", "does not", "is") that are
  NOT content lemmas selected upstream. Currently the ANN emits them; FrameCQ orders only the CONTENT role-slots. **This
  is the single biggest concrete residual and is CHEAP** (function words are a tiny closed class — they are frame-slot
  furniture, exactly what Bock-Levelt says the frame supplies).
- **(R2) Morphological inflection** — 3sg `-s` (fly→flies), the copula, verb-form selection. Currently a host table
  (`emerge_v3`) + the ANN. Real biology: a morphological slot in the phonological-encoding stage.
- **(R3) Frame SELECTION must be neural** — WHICH frame (statement / who-what / yes-no / modal-ability / negated-exception)
  must be chosen by the brain (dlPFC), not a host `if`. FrameCQ orders a GIVEN frame; picking the frame is upstream.
- **(R4) Novel/longer word sequences beyond the fixed inventory** — genuinely open generation (arbitrary connected prose)
  is the KNOWN WALL (the ~134K-param from-scratch BPTT-SNN was judged ~4 orders too small vs ~1B,
  `2026-05-07-Phase-2.3a-NEGATIVE`; the whole Generator S/D/E/F/G/H arc). This part is **NOT** cheaply on-substrate and is
  honestly deferred (see Move 4).

**The precise residual for a cheap first step: R1+R2+R3 — a LEARNED spiking frame that supplies function-word + inflection
slots and is neurally SELECTED — is what replaces the ANN for EMERGE's actual frame inventory.** R4 (open fluent prose) is
a separate, harder, deferred question.

---

## MOVE 2 — REFRAME VIA "HOW DOES REAL BIOLOGY ACTUALLY DO THIS?"

**Am I testing the wrong hypothesis?** The scaffold implicitly frames production as "learn a big LM." Biology does NOT.
The reframe (grounded below) is: **biology COMPOSES an utterance from a syntactic FRAME whose slots are filled by selected
lemmas + supplied function words + inflected forms — each a small circuit — driven by a serial-order (competitive-queuing)
engine.** That is EXACTLY the shape of the project's already-validated FrameCQ + A→W, missing only the function-word /
inflection / frame-selection slots. So the right hypothesis is "learn the FRAME + its closed-class furniture," not "learn
a language model."

**Catalog grounding (verified `E:/Documents/Projects/sim-catalog/references/feature-catalog.md`):**
- **G.12 Broca's area** (Kandel 6e Ch 55 pp 1382–1384, Fig 55-6): "Maps stored auditory word-forms to motor articulation;
  supports … grammatical processing." Damage → **agrammatic** speech with **retained noun selection but LOST function-word
  / verb use** — i.e. Broca's specific job is the FRAME + function words + grammatical morphology, precisely R1–R3. **Sim
  status: missing.** This is the entry we are building.
- **G.11 Dual-stream** (Ch 55 pp 1380–1387): dorsal (production/repetition) vs ventral (comprehension) — matches
  "Wernicke decides → Broca articulates" already in the wire.
- **G.10 language-as-hierarchical-symbolic-system** (Ch 55 pp 1370–1372): finite units → infinite combinations via
  frames — the frame-and-slot grounding. **Sim status: missing.**
- **G.07 pre-SMA/SMA internally-generated sequences** (Ch 34 pp 822–828) + **H.19 premotor/SMA sequential action** — the
  serial-order-PRODUCTION substrate; catalog basis for competitive queuing (already the project's CQ mechanism).
- **NO catalog entry exists for HVC/birdsong, CQ, DIVA/GODIVA, or Levelt** (verified) — those are project-internal
  (`sim/song_hvc.py`) or literature-only. The catalog language-production cluster G.10–G.14 is uniformly "missing," which
  is the honest statement that this IS new-direction work (gate fires).

**Literature grounding (the reframe, cited):**
- **Levelt, Roelofs & Meyer 1999** (*BBS*, "A theory of lexical access in speech production") + **Bock & Levelt 1994**
  (grammatical encoding) + **Dell 1986** (*Psych Review* spreading-activation): three separable strata — conceptual →
  **lemma** (grammatical class, tense, function) → form. Frame-and-slot grammatical encoding: "frames represent syntactic
  structures with slots labelled with the grammatical classes of the lemmas that may fill them." ⇒ function words +
  agreement are FRAME properties, not selected content.
- **Averbeck, Chafee, Crowe & Georgopoulos 2002/2003** + **Kornysheva et al. 2019** (bioRxiv 383364, *"Neural competitive
  queuing of ordinal structure underlies skilled sequential action"*): direct neural evidence that the cortex holds an
  upcoming-sequence **ordinal template** as a parallel activation gradient, produced by competitive queuing —
  position-generalizing across specific actions. This is the *measured* biological instantiation of the CQ mechanism the
  project already runs. ← load-bearing for ranking CQ/FrameCQ first.
- **Hartley & Houghton 1996** (*"Serial Control of Phonology in Speech Production: A Hierarchical Model"*) + Houghton 1990:
  the canonical frame-slot serial-order model of SPEECH specifically — a competitive-queuing planning layer + syllable/word
  frames whose slots are filled position-wise. Confirms the project's CQ choice is the field-standard serial-order-of-speech
  mechanism, and that **function/structure lives in the FRAME**.
- **Dell, Chang & Griffin 1999** / **Chang, Dell & Bock 2006** ("Becoming syntactic," the *dual-path* model): a recurrent
  connectionist sentence-production net LEARNS syntactic frames + function words + agreement from message→sequence
  experience (a *small* recurrent net, not a 1B LM) — the closest computational precedent for "learn the frame from
  experience," and the template for R1–R3 as a LEARNED spiking recurrent producer.
- **2024–2026 spiking-LM landscape (WebSearch):** fully-spiking LMs (SpkGPT; "NeuronSpark"; the ~0.9B from-random-init
  spiking LM) are still (i) trained by off-substrate backprop next-token prediction, (ii) capped well below transformer
  scale (prior work ≤216M). ⇒ NO off-the-shelf emergent-on-substrate fluent generator exists; the honest emergent path is
  the SMALL-CIRCUIT frame-based one, and open fluent prose (R4) remains the field wall — corroborating the deferral.

**Verdict of the reframe:** we were about to test the wrong hypothesis (shrink/convert an LM). The right hypothesis —
"learn a small, neurally-selected syntactic FRAME with function-word + inflection slots, driven by the already-spiking
CQ serial-order engine, filled by the emergent lexicon via A→W" — has a cheap answer and is exactly Broca's catalogued
job (G.12).

---

## MOVE 3 — RANK CHEAP-FIRST SPIKING MECHANISMS PAST THE RESIDUAL

Bars: (i) reuse-by-import of already-validated pieces over a from-scratch build; (ii) ONE variable per rung, gated; (iii)
full anti-cheats (held-out-only, permuted-ORDER control, no-learning/lesion control, memorization-floor, host-baseline
reported, no-confab moat verified); (iv) multi-seed (≥6 for any generalization claim); (v) leverage-per-cost.

### Rung A (RECOMMENDED first — EMERGE-59 candidate) — the FRAME-AS-SLOTS neural renderer: extend FrameCQ so the frame carries FUNCTION-WORD + INFLECTION slots, learned from examples, and DROP the ANN for EMERGE's fixed frames

- **What:** treat each EMERGE reply frame as an ordered sequence of slots that includes **fixed function-word slots** and
  **content slots**, e.g. modal-ability = `[det("the"), CONTENT:subject, modal("can"), CONTENT:verb_bareinf]`;
  intransitive-exception = `[det("the"), CONTENT:subject, CONTENT:verb_3sg]`; yes/no = `[POLARITY]`. The **order + which
  function-word slot** is the LEARNED `prim[frame][slot]` gradient (already the FrameCQ mechanism); content slots spell via
  A→W; function-word slots spell via the SAME A→W read-out over dedicated closed-class concept pools (function words are
  just more lemmas in the emergent lexicon — cheap to add as pools). Inflection (R2) = a per-slot morphological read-out
  (3sg-form pool vs bare-form pool selected by the frame slot's tag — a learned 2-way choice, not a host table).
- **Why cheapest/highest-leverage:** it is **assembly of validated parts** — FrameCQ (6/6 GO spikes, frame-conditioned
  order with a decisive cross-frame control) + A→W (100% multi-seed) + the gate-first moat (EMERGE-56/57/58) — extended by
  the smallest new thing: adding function-word + inflection SLOTS to the frame. It directly replaces the ANN for the exact
  frame inventory EMERGE emits (ability-affirm / describe / intransitive-exception / abstain), which is ALL EMERGE-57/58
  actually needed the 21M for.
- **Anti-cheats:** held-out FACTS only (never in the frame-teaching set); **permuted-slot-order control** (reuse
  `song_g1_core.permuted_order_controls` — must drop to chance); **cross-frame control** (the same content under a
  different frame must NOT match — FrameCQ already has this, 0.000); **no-learning control** (equal/untrained primacy →
  chance); **function-word ablation** (remove the learned function-word slots → agrammatic output, proving they're
  learned-slot-supplied not host-inserted); **memorization floor** (a lookup baseline can't generalize to held-out); the
  **no-confab moat verified** (abstain → the renderer NEVER emits, reuse the EMERGE-56 `render_call_count==0` hard
  assertion); host-template baseline reported. ≥6 seeds.
- **Cost:** CPU/numpy phase (seconds) + one small spiking bridge (`SIM_BACKEND=numpy`, minutes), then a GPU multi-seed
  confirm. No 5-bridge load required for the de-risk. **This is the concrete cheapest single experiment.**
- **Outcome routing:** GO → this IS the ANN replacement for EMERGE's frames; wire into `_emerge58` behind the gate-first
  moat and RETIRE the 21M for those frames. PARTIAL (order/function-words learn but inflection lost) → localizes the wall
  to the morphological read-out (R2), a bounded follow-on. NEGATIVE (can't beat permuted with function-word slots) →
  informative: the closed-class furniture needs a different mechanism (Rung B).

### Rung B — LEARNED syntactic frames as Pulvermüller sequence-detector assemblies (DCNAs), neurally SELECTED by the dlPFC (closes R3)

- **What:** each frame = a Hebbian-bound "fires-to-AB-not-BA" sequence-detector assembly (Pulvermüller & Knoblauch 2009
  DCNAs); the dlPFC content Control SELECTS the frame from the message features (statement vs question vs negated
  exception), so frame selection is NEURAL not a host `if` (closes R3). Layers ON TOP of Rung A's serial-order engine.
- **Why second:** with only a handful of frames, a host frame-selector is itself a shortcut; this makes selection neural.
  But it's a higher-integration build than Rung A and only matters once Rung A proves the frame-as-slots renderer works.
- **Anti-cheats:** frame-selection accuracy vs a permuted message→frame mapping (must collapse); dlPFC-lesion collapses
  selection; held-out messages; moat preserved. ≥6 seeds.
- **Cost:** medium (new DCNA assemblies + dlPFC wiring). Defer behind Rung A.

### Rung C — a SMALL LEARNED recurrent spiking sentence-producer (Chang/Dell dual-path), trained on-substrate by the fact-as-teacher, for BEYOND-fixed-inventory local generalization

- **What:** a small recurrent spiking net (message + prev-word → next-word) learns frames + function words + agreement
  from message→sequence experience, à la Chang-Dell-Bock 2006 dual-path — but SMALL (dozens–hundreds of neurons, a few
  frames, ≤320 lemmas), teacher = the structured fact's own word sequence (the EXTERNAL teacher that dodged the G1
  self-judge trap), NOT open prose. Uses surrogate-gradient BPTT (`sim/bptt_snn*.py`, `surrogate_grad.py`) as the on-
  substrate learner. This is the "learned-in-the-loop, generalizes to novel short sequences" upgrade over Rung A's
  fixed-slot frames.
- **Why third (not first):** BPTT-trained weights are still gradient-computed; whether that counts as "emergent
  self-organized" is the open question (`feedback_spiking_structure_must_self_organize`) — a three-factor / local-plasticity
  variant would be more faithful but higher-variance. Only pursue once Rung A shows the fixed frames aren't enough (i.e.
  when a genuinely novel short sequence is needed). Reuses the exact `song_g1_core` held-out + permuted-order gate.
- **Anti-cheats:** held-out sequences, permuted-order primary gate, memorization-floor (in-sample vs held-out gap),
  the recorded G1 trap avoided (fact-as-teacher, not self-comprehension judge), moat preserved. ≥6 seeds.
- **Cost:** medium-high; the honest bridge toward R4 without attempting open 1B-scale prose.

### NOT ranked / explicitly deferred
- **Shrinking or converting the 21M ANN further** — it is a permanent external ML artifact regardless of size; the master
  directive forbids it as the END state. Only kept as the current TEMPORARY scaffold until Rung A retires it frame-by-frame.
- **Open fluent arbitrary prose (R4)** — the known ~4-orders-too-small wall; NOT cheaply on-substrate; deferred (Move 4).
- **Theta-gamma slot multiplexing** (Lisman-Jensen, catalog N.15) — most biologically faithful "word-order-as-slots" but
  the project has NO theta/gamma generator in this path + dt-fragility (one-bridge step-3 finding). Follow-on ONLY if Rung
  A hits a flat order-production wall (per the 2026-06-16 scoping's identical deferral).

---

## MOVE 4 — VERDICT: SURPASSABLE, AND HOW CHEAPLY

**SURPASSABLE — and cheaply, for the part that matters now.** The scaffold ANN is NOT irreducible: **~75% of what it does
in this console is already spiking-realizable** by composing validated pieces (emergent lexicon lemmas + A→W spell +
FrameCQ frame-conditioned serial order + gate-first moat). The genuine residual is the closed-class **furniture** —
function-word slots (R1), inflection (R2), and neural frame-selection (R3) — which is **exactly Broca's catalogued job
(G.12: retained noun selection, lost function-word/verb use)** and is CHEAP because function words are a tiny closed class
that the frame supplies (Bock-Levelt), not content to be modelled.

**The exact next cheap experiment (EMERGE-59 candidate) — Rung A:** extend the already-GO `FrameCQ` so each EMERGE reply
frame carries LEARNED **function-word + inflection SLOTS** alongside content slots, spell every slot (content AND
function-word) via the validated A→W read-out over the emergent lexicon, gate it with the EMERGE gate-first moat, and grade
**held-out** facts with `song_g1_core.score_order` vs the **permuted-slot-order** + **cross-frame** + **function-word-ablation**
+ **no-learning** controls, ≥6 seeds. GO ⇒ this renders EMERGE's fixed frames FLUENTLY on spikes and **retires the 21M ANN
for those frames** — the first genuine "simulate Broca" step. It is CPU-first (seconds→minutes), reuse-by-import, one
variable (add function-word/inflection slots to the frame), and its three outcomes each route the next move cleanly.

**The genuinely-irreducible part, precisely why:** OPEN fluent arbitrary prose (R4) — beyond the fixed frame inventory —
is NOT cheaply on-substrate: the from-scratch spiking LM path is ~4 orders too small (`2026-05-07-Phase-2.3a-NEGATIVE`),
and 2024–2026 fully-spiking LMs are still off-substrate-backprop-trained + sub-transformer-scale. Per the master directive
this is an UNDISCOVERED MECHANISM, not an endpoint — but it is a SEPARATE, harder question. The honest framing: **Rung A
retires the ANN for the bounded frame inventory the emergent brain actually uses today; R4 (open prose) stays scaffolded by
the tracked temporary ANN until a from-experience-learned recurrent producer (Rung C) or a larger emergent mechanism
clears it.** No "wall" is accepted — R4 is named as the next mechanism-search, sequenced after Rung A.

---

## SEQUENCING vs THE OTHER TWO LIVE POST-WIRE FRONTIERS (honest)

1. **THIS (simulate-Broca / generator replacement)** — highest alignment with the master directive's "don't bolt on an
   LLM / emergent single substrate." Rung A is CHEAP and retires the scaffold for the actual frame inventory. **Recommend
   FIRST** — it removes the single most conspicuous permanent-external-ML shortcut, and it's mostly assembly of GO parts.
2. **Deepen fluid conversation (multi-fact synthesis / open-domain)** — the DISCUSS/synthesis + open-domain frontier
   (CLAUDE.md "genuine walls"). This OVERLAPS R4 (open prose) — doing Rung A first clarifies exactly which synthesis needs
   open generation vs which is frame-composable. **Sequence AFTER Rung A**; it partly depends on the same open-prose
   mechanism.
3. **Dendritic credit-assignment emergence (EMERGE-1b burst-multiplexed plasticity)** — the deepest lever (the master
   directive's core "emergent from experience" engine) and the eventual substrate for a from-experience-learned producer
   (Rung C) AND for R4. Higher-variance, longer-horizon. **Runs in PARALLEL as the deep track**; it is what ultimately
   makes Rung C's weights self-organized rather than BPTT-injected, and is the honest long-term answer to R4.

**Net:** do **Rung A (EMERGE-59) next** — cheapest, highest directive-alignment, retires the ANN for EMERGE's frames;
keep the dendritic emergence track running in parallel as the deep lever that will later carry Rung C + R4.

---

## Files / provenance reviewed
- Project machinery: `research/runners/neural_serial_order_renderer.py`, `_phaseB_serial_order_{cq,spiking,multiframe}_derisk.py`
  (FrameCQ), `sim/song_hvc.py`, `sim/bptt_snn*.py`, `sim/surrogate_grad.py`, `research/runners/song_g1_core.py`
  (`score_order` + `permuted_order_controls` + `g1_verdict`), `concept_speak_demo.py` (A→W), the EMERGE-56/57/58 wire
  (`_emerge5{6,7,8}_*` runners).
- Findings: `2026-07-03-emerge5{6,7,8}-*.md` (the wire), `2026-06-16-sentence-generation-{biologization-deep-research,
  serial-order-cheap-first-GO}.md`, `2026-06-23-bridge-coresidence-DEMONSTRATED.md`,
  `2026-06-30-100M-C2-scaleup-C1-GO-C2-nuanced.md`, `2026-05-07-Phase-2.3a-NEGATIVE-next-char-features.md`,
  `2026-05-16-generator-G1-songbird-NEGATIVE.md`.
- Catalog: `E:/Documents/Projects/sim-catalog/references/feature-catalog.md` **G.10–G.14** (language production, all "Sim
  status: missing"), **G.07 / H.19** (SMA serial order); verified NO HVC/CQ/DIVA/Levelt catalog entry.
- Memory: `project_master_directive_relentless_biological_emergence`, `feedback_spiking_structure_must_self_organize`,
  `feedback_brain_based_only_standard`.
- Literature (WebSearch + verified refs): Levelt-Roelofs-Meyer 1999 (*BBS*); Bock & Levelt 1994; Dell 1986 (*Psych Rev*);
  Chang, Dell & Bock 2006 (dual-path, *Psych Rev*); Kornysheva et al. 2019 (bioRxiv 383364, neural competitive queuing);
  Averbeck et al. 2002/2003; Hartley & Houghton 1996 (serial phonology); Pulvermüller & Knoblauch 2009 (DCNAs); Grossberg
  1978 / Houghton 1990 / Bullock & Rhodes 2003 (competitive queuing); 2024–2026 spiking-LM survey (SpkGPT / NeuronSpark /
  ~0.9B from-scratch spiking LM — all off-substrate-trained, sub-scale).
