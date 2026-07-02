# Research gate — from the emergent HTM Temporal-Memory sequence cortex to REAL LANGUAGE (2026-07-02)

**Read-only deep-research gate (mandated before a NEW direction).** Context: rung-4 is COMPLETE — the whole unsupervised,
self-organizing HTM Temporal-Memory (Hawkins-Ahmad; the spiking port of Bouhadjar-Diesmann-Tetzlaff 2022, *PLoS Comput
Biol* 18(6):e1010233) runs on the real spiking `SimulationBridge`: BOTH inference AND learning, teacher-free, 6-seed GO,
scaling to 8 contexts sharing one middle (`_emerge14_stageC_onbridge_learning_derisk.py`,
`2026-07-02-emerge14-stageC-onbridge-learning-GO-rung4-complete.md`). The task so far is abstract overlapping SYMBOL
sequences. The owner directive ([[project_master_directive_relentless_biological_emergence]]): a SIMULATED recurrent
sequence/language cortex is the honest, self-contained path to language production — the minimized ~21M transformer, the
VSA composer, discourse templates, and intent dispatcher are all TEMPORARY scaffolds to be REPLACED by simulated
circuitry. "If Broca drives articulation, we simulate Broca."

**BOTTOM LINE (verdict, expanded at the end):** the gap is NOT a wall. High-order next-symbol prediction over a WORD
vocabulary is — literally, in the computational-neuroscience literature — a biological language model (§2). The current
HTM-TM already has the two hardest pieces (unsupervised high-order context-specific prediction + on-substrate learning);
the missing pieces are mostly **plumbing of already-validated project machinery**: a word→SDR encoder (shipped), an
emit/decode read-out (shipped: the A→W read-out + the competitive-queuing serial-order renderer), and an autoregressive
feed-back loop (a ~10-line runner change — the Bouhadjar network already does autonomous roll-out via a single
excitability flag, §2). The one genuinely-new de-risk (§4) is: **feed the on-bridge HTM-TM real word tokens from a tiny
corpus, learn next-word prediction unsupervised, and measure held-out next-word accuracy against an n-gram baseline** —
the smallest honest "spiking language model on one brain."

---

## 1. ISOLATE the true gap — symbol-sequence memory (have) vs word-sequence production/comprehension (want)

The current HTM-TM (`_emerge14_stageC_onbridge_learning_derisk.py`, verified by direct read) is precisely:
- **Encoding:** `build_pool_bridge(vocab, nE, …)` — `vocab` columns × `nE` cells/column; **each symbol = one dedicated
  column** (a LOCALIST symbol code). A symbol's arrival = fire its column's cells. There is **no word-SDR** (no
  similarity structure between symbols) and no external drive encoder wired — the "symbol" is an integer column id.
- **Prediction:** the bridge's weighted-coincidence recurrence (`coincidence_predict` → `_prime_from_winners`) primes the
  apical/dAP compartment of the cells of the predicted continuation; the read-out returns the predicted **column** =
  `cell_idx // nE` = the predicted next **symbol index**. So a symbol-level decode ALREADY EXISTS (predicted column ⇒
  predicted symbol). The dAP-lesion collapses it → load-bearing.
- **Learning:** `fused_htm_permanence_update` (committed `sim/` kernel) applies the Bouhadjar three-term rule to
  `cp_connections.data` per symbol, teacher-free.
- **Roll-out:** NONE. It does single-step branch prediction only (`predict_branch` reads one continuation at a
  divergence point). It does not iterate its own prediction to GENERATE a sequence.

**Enumerated concrete missing pieces, each tagged HAVE / PLUMB / NEW:**

| # | Missing piece | What it is | Status | Where it lives |
|---|---|---|---|---|
| G1 | **Word→SDR input encoder** | map a word token to a sparse code injected as drive | **PLUMB** (shipped) | `sim/text_embeddings.py`: `orthogonal_drive_pattern`, `vocab_to_drive_pattern`, `positional_drive_pattern`; `concept_pool_sparse_distributed.generate_sparse_patterns` (K-of-N SDRs) |
| G2 | **Emit / production read-out** | turn the predicted cell-set into an emitted WORD | **PLUMB** (shipped) | predicted column ⇒ symbol id (already); word spelling = the A→W read-out (`concept_speak_demo`, 100% multi-seed); word ORDER for multi-slot output = the competitive-queuing `NeuralSerialOrderRenderer` (`neural_serial_order_renderer.py`, 6-seed GO) |
| G3 | **Autoregressive generation (roll-out)** | feed the predicted next word back as input; repeat | **PLUMB** (≈10-line runner loop) — the Bouhadjar substrate does this natively via a replay/excitability flag (§2) | new runner around `OnBridgeLearner` |
| G4 | **Real vocabulary** | tens→hundreds→thousands of words, not 8 symbols | **NEW (scale)** — needs (a) similarity-structured word codes and (b) a sparse multi-segment pool (§3, R2) to avoid the O(N²) dense pool | stream-cortex codes (G5) + HTM multi-segment |
| G5 | **Similarity-structured word codes** | so "dog"/"cat" overlap → generalization, not localist columns | **PLUMB→NEW** — the stream-learned PPMI cortex codes already exist (64 concepts, `corr(M,C)=+0.885`); wiring them as the HTM's column-assignment is the new step | `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`; `_phaseB_onbridge_stream_cortex_derisk.py` |
| G6 | **Syntax / grammar** | frame-dependent order; long-range dependencies | **PARTIAL** — high-order context IS the HTM's core competence (context-specific branch = a learned bigram/trigram distinction); frame-conditioned ORDER already de-risked | HTM-TM (high-order) + `_phaseB_serial_order_multiframe_derisk.py` (frame-conditioned order, cross-frame control 0.000) |
| G7 | **Grounding to meaning + no-confab** | produce GROUNDED word sequences, abstain when unknown | **PLUMB** (shipped) — the gate→constrain→verify loop + moat | `2026-06-23-grounded-lang-INTEGRATION-GO.md`; `rf_phasor_composer.py`; `brain_conversational_agent.py` |

**Conclusion of the isolation.** The genuinely-new residual is SMALL and specific: **(a)** learn next-WORD prediction on
real word tokens (G1+G3 plumbing → validate as an LM), and **(b)** scale the pool from a dense localist O((vocab·nE)²)
potential pool to a **sparse, similarity-structured, multi-segment** one (G4+G5). Everything else is composition of
validated pieces. Nothing here requires the transformer or the VSA composer as a permanent faculty — they are the
FLUENCY/binding scaffolds the emergent sequence cortex is meant to subsume.

---

## 2. REFRAME via real biology — the brain's sequence machinery, and why word-prediction IS a language model

### 2a. Next-symbol prediction over words = a biological language model (the key reframe)
The neocortex is, to a first approximation, a hierarchical next-input predictor. Caucheteux & King (2023, *Nat Hum Behav*,
"Evidence of a predictive coding hierarchy in the human brain listening to speech", 304 participants, 285 citations)
showed that (i) modern language-model activations linearly map onto brain responses to speech, and (ii) the brain
predicts a **hierarchy** of upcoming representations across timescales — frontoparietal cortex predicts longer-range,
more contextual content than temporal cortex. Jiang & Rao (2023, *PLoS Comput Biol*, "Dynamic predictive coding")
formalize the same as a hierarchical sequence model where higher levels span longer timescales. **Implication:** a network
that does high-order, context-specific next-element prediction over a WORD alphabet is, by construction, computing what a
biological language cortex computes — a predictive language model. The HTM-TM's core competence (given a shared middle,
predict the correct continuation FROM the earlier context) is exactly the "high-order n-gram / Elman-style context
sensitivity" that distinguishes a language model from a bag-of-words. So **word-level next-word prediction is not a
distant goal bolted onto the HTM — it is the HTM's native computation with a word alphabet.**

Numenta's own line makes this concrete: HTM temporal memory + **Semantic Folding** (Cortical.io, De Sousa Webber) is the
canonical HTM NLP pipeline — words are encoded as SDRs by "semantic folding" (a topographic word-similarity space →
sparse binary code), and HTM temporal memory then learns/predicts sequences of those word-SDRs. The project ALREADY has
the semantic-folding equivalent: the stream-learned PPMI co-occurrence cortex codes (G5). So the two canonical HTM-NLP
ingredients (word-SDR encoder + temporal memory) both exist here.

### 2b. Sequence GENERATION in the Bouhadjar spiking network — it is a READ-OUT MODE, not a new learning problem (load-bearing)
The Bouhadjar-Diesmann 2022 paper (which the project's rung-4 ports) generates full sequences autonomously, and does so
WITHOUT any additional learning machinery. Verified from the paper: in **replay mode** the network is run with
**increased neuronal excitability**, "such that the somatic depolarization caused by a dAP alone makes the neuron fire a
somatic spike." A cue drives the first element; its dAP-primed next-element cells then fire from the dAP alone; those in
turn prime element n+2; "the network autonomously reactivates all sequence elements in the correct order." Transition
timing is set by synaptic delay/time-constants, independent of the training ISI. Excitability is raised biologically "by
neuromodulators, attention, or propagating waves during sleep."

This is decisive for the verdict: **autoregressive generation on THIS substrate is a single flag** (raise excitability so
dAP-alone fires → the already-learned permanences roll the sequence out), NOT a separate, hard, self-supervised
generation-learning problem. It is the SAME learned synapses used in prediction, run in a self-sustaining regime. The
project's prior generation NEGATIVEs (§ project machinery item 6: SongHVC synfire self-supervision, closed-loop
predictive-coding roll-out) were a DIFFERENT mechanism — a generator whose *self-comprehension critic* couldn't read
order back, so it got zero gradient. That failure mode does not apply to Bouhadjar replay, which needs no critic and no
extra learning. (Bouhadjar et al. 2023, *PLoS Comput Biol* 19(5):e1010989, "Coherent noise enables probabilistic sequence
replay", adds noise-driven spontaneous replay/branch-selection on top — a later, optional lever.)

### 2c. Serial-order PRODUCTION for multi-item output — competitive queuing (Grossberg; Bullock-Rhodes; Averbeck)
When the output is a SET of items to be ordered (a phrase's words, a word's phonemes), the biology is **competitive
queuing (CQ)**: a planning layer holds all items in parallel with a **primacy gradient** (first item = highest
activation), and a choice WTA emits the highest-activity item then suppresses it (inhibition-of-return), iterating to
serialize (Grossberg 1978; Bullock & Rhodes 2003, "Competitive queuing for planning and serial performance"). Averbeck et
al. (2002/2003, PNAS/J Neurophysiol) recorded exactly this in prefrontal cortex: distinct ensemble patterns for each
sequence item present in parallel during preparation, their relative strength predicting serial position. The project
already de-risked this on-substrate (`neural_serial_order_renderer.py`, 6-seed GO, rate-coded CQ; the multi-frame
extension learns DISTINCT orders per frame with a cross-frame control at 0.000 — the seed of syntax). Catalog: G.07/H.19
(pre-SMA/SMA internally-generated sequence production, Kandel 6e Ch 30-39); the verbal working-memory subsystem =
phonological store (posterior parietal) + rehearsal (Broca's), Kandel 6e Ch 52 pp 1293-1297.

### 2d. Language-area anatomy (the "if Broca drives articulation, we simulate Broca" targets — cited)
From the catalog (Kandel 6e Ch 55, dual-stream / Hickok-Poeppel):
- **G.11 Dual-stream model** (pp 1380-1387): DORSAL stream (posterior superior-temporal → arcuate fasciculus → Broca)
  = sensorimotor mapping for PRODUCTION + complex-syntax sequencing; VENTRAL stream = sound→meaning COMPREHENSION.
- **G.12 Broca's area** (pp 1382-1384): maps stored word-forms → sequential motor articulation; supports non-canonical
  (grammatically complex) sentence structure. = the serial-order/production role (G2/G3 + CQ).
- **G.13 Wernicke's area** (pp 1384-1385): word selection matching intended meaning (the comprehension/lexical-selection
  role = the stream-cortex codes + the composer's cue-match).
- **G.10 hierarchical symbolic system** (pp 1370-1372): finite units → infinite combinations via sequential/syntactic
  rules = exactly what a high-order sequence predictor over words provides.
- Supporting timing substrate: **D.11 time cells** / **D.24 theta-paced sequence compression** / **N.15 theta-gamma
  multiplex** (Lisman-Idiart 1995; 7±2 gamma cycles per theta = word-buffer capacity) — the biological clock for
  word/phoneme timing; MISSING in-sim but not required for the first LM de-risk (optional later scaling lever for
  multi-word working-memory buffers).

The mapping is clean: **HTM-TM = the cortical sequence-prediction engine (temporal cortex predictive hierarchy); the CQ
serial-order renderer + A→W = the Broca/production read-out; the stream-cortex codes = the Wernicke/lexical-selection
front-end; the gate→verify moat = the grounded-selection constraint.** No permanent transformer needed for the
PREDICTION+PRODUCTION core; the transformer's only unique job (open-domain surface fluency) is the last thing to retire.

### 2e. How far is HTM-TM capacity from a usable word model? (measure, don't guess)
Current validated scale (from the findings, not guessed):
- **Contexts / branches:** numpy TM to 32 overlapping contexts (1.000); on-bridge learned to n_seq=8 sharing one middle.
- **Sequence length:** overlapping middles L=4+; the source paper tests order-10 (12-element) sequences.
- **Alphabet:** the paper's benchmark = 14 symbols; the project's runner is parameterized on `vocab`.
- **The wall is representational + memory-footprint, not conceptual:** the current pool is a DENSE cross-column
  potential pool = O((vocab·nE)²) synapses (verified: `build_pool_bridge` wires every cross-column (pre,post) pair). At
  vocab=8, nE=16 that is ~16k synapses; at vocab=2000, nE=16 it is ~10^9 — infeasible. **Canonical HTM fixes this
  exactly:** multiple **distal segments per cell**, each subsampling a SPARSE fixed number of potential synapses (not
  all-to-all). This is the project's already-named R2 multi-segment extension and is the true scale lever (§3(d)). So
  the distance to a usable word model = (i) similarity-structured codes (G5, exist) + (ii) sparse multi-segment pool
  (R2, a `sim/` extension) + (iii) more training tokens — all bounded, none conceptual.

---

## 3. RANK cheap-first paths (from current HTM-TM → language-capable sequence cortex)

Ordered by implementation cost on THIS substrate. Each: mechanism · citation · smallest de-risk · anti-cheats ·
`sim/`-edit-or-not.

### (a) ★ CHEAPEST — feed WORD tokens + validate next-word prediction as a language model (the smallest spiking LM)
- **Mechanism:** replace the localist integer symbol with a WORD token whose column is assigned by an existing encoder;
  run the unchanged `OnBridgeLearner` (HTM permanence learning) on a tiny real corpus's word stream; read out the
  predicted next column = predicted next word. Add a trivial autoregressive feed-back loop for generation.
- **Citation:** Bouhadjar-Diesmann 2022 (unsupervised local-rule sequence prediction); Numenta HTM + Semantic Folding
  (word-SDR → temporal memory = the canonical HTM LM); Caucheteux-King 2023 (next-word prediction = the cortical
  computation).
- **Smallest de-risk:** take ~8-16 words from a tiny hand-made grammar or a TinyStories snippet with genuine high-order
  structure (a shared bigram with two context-dependent continuations, e.g. "the dog ran / the cat ran" then "…
  home"/"… away" conditioned on dog-vs-cat earlier), encode each word to its column (localist first — cheapest), learn
  unsupervised, measure **held-out next-word accuracy vs a bigram/trigram Markov baseline** (the HTM must beat the
  order-blind Markov floor by using the earlier context = the whole point of high-order memory).
- **Anti-cheats:** (1) **Markov/bigram floor** — HTM must exceed it (else it's not using high-order context); (2)
  **dAP-lesion** → collapses to the floor (prediction is load-bearing); (3) **permuted-corpus** control (shuffle word
  order → structure gone → accuracy → chance); (4) **no-teacher** (fully unsupervised, as rung-4 already is); (5)
  **multi-seed** 6×; (6) **held-out** continuations never trained.
- **`sim/` edit:** NONE (reuse-by-import: `build_pool_bridge` + `OnBridgeLearner` + `orthogonal_drive_pattern`). The
  generation loop is a ~10-line runner addition; the excitability-replay flag is the Bouhadjar mode.

### (b) autoregressive GENERATION via excitability-replay (roll out full sentences)
- **Mechanism:** after (a) learns, run the network in replay mode (raise excitability so a dAP-alone fires) → cue the
  first word → the network self-continues, emitting a learned word sequence; feed each emitted word's column back for
  the next step (teacher-free roll-out).
- **Citation:** Bouhadjar-Diesmann 2022 replay mode (verified: autonomous reactivation in order via raised excitability);
  Bouhadjar 2023 (coherent-noise probabilistic replay for branch selection).
- **Smallest de-risk:** cue-then-generate on the (a) corpus; score generated continuations against held-out true
  continuations; compare cued-replay vs cold (no cue) and vs high-excitability-but-untrained.
- **Anti-cheats:** permuted-order control (generated order must match the TRUE learned order, not a scramble — reuse the
  `song_g1_core` verdict bar: margin ≥10% over permuted, abs ≥0.5); lesion; untrained control; multi-seed.
- **`sim/` edit:** likely NONE (excitability is a runtime drive/threshold offset; if the megakernel needs an
  excitability scalar exposed, that is a tiny additive default-inert flag). Explicitly NOT the failed SongHVC
  self-supervision path (§2b) — no critic, no extra learning.

### (c) similarity-structured word codes (generalization, not localist columns)
- **Mechanism:** assign each word's HTM cells from the stream-learned PPMI co-occurrence cortex codes (so related words
  share cells → the HTM generalizes predictions across similar words), instead of disjoint localist columns.
- **Citation:** the project's on-bridge Hebbian co-occurrence cortex (`corr(M,C)=+0.885`, generalizes held-out 0.86);
  Pulvermüller-Garagnani spiking distributed word ensembles (Garagnani & Pulvermüller 2018, *Front Comput Neurosci*
  12:88 — brain-constrained spiking model where word cell-assemblies self-organize by Hebbian learning); Semantic
  Folding (word-similarity SDR).
- **Smallest de-risk:** repeat (a) but with stream-cortex codes as column assignments; measure whether prediction
  GENERALIZES to a never-seen word substituted from the same category (dog↔cat) vs localist (which cannot).
- **Anti-cheats:** category-derangement control (shuffle which code goes to which word → generalization collapses);
  held-out-word substitution; permuted-corpus; multi-seed.
- **`sim/` edit:** NONE (code assignment is a wiring choice in the runner; the stream cortex is reuse-by-import).

### (d) sparse multi-segment pool — the vocabulary/context SCALE lever (R2)
- **Mechanism:** replace the dense O(N²) cross-column potential pool with the canonical HTM structure — **multiple
  distal segments per cell, each subsampling a small fixed set of potential synapses** — so capacity scales with
  segments, not with vocab². This is the project's already-named R2 multi-segment extension.
- **Citation:** Hawkins-Ahmad 2016 (HTM: cells have many distal segments, each a sparse coincidence detector); Bouhadjar
  2022 (per-neuron dendritic branches). Catalog D.18 (the three-term permanence rule already committed).
- **Smallest de-risk:** on a larger vocab (e.g. 64-128 words) show the sparse multi-segment pool matches the dense pool's
  next-word accuracy at a fraction of the synapses; measure the synapse-count vs vocab curve (must be sub-quadratic).
- **Anti-cheats:** dense-pool parity (sparse ≈ dense at small scale); capacity curve; lesion; multi-seed.
- **`sim/` edit:** YES — a genuine additive `sim/` extension (per-cell segment structure in the coincidence pathway +
  the kernel gathering per-segment). This is the one path needing protected work, and it is the honest scale mechanism
  (fair game per the master directive — biology, not a cheat). Deferred until (a)-(c) prove the word-LM works.

### (e) ground the emitted word sequences + no-confab (produce GROUNDED sentences, replacing the transformer's fluency)
- **Mechanism:** couple the generating HTM to the grounded-knowledge gate→constrain→verify loop + the moat, so it emits
  sequences that are (i) grounded in stored facts and (ii) abstain when unknown — the role currently played by the
  gate around the transformer.
- **Citation:** `2026-06-23-grounded-lang-INTEGRATION-GO.md` (gate→constrain→verify catches a real role-inversion
  hallucination; moat 0-FA).
- **Smallest de-risk:** have the HTM generate the next word CONSTRAINED to a stored fact's continuation; verify the
  emitted sequence re-parses to the fact; abstain on untaught cues.
- **Anti-cheats:** untaught-cue abstention (0 false-accepts); drift injection caught by verify; multi-seed.
- **`sim/` edit:** NONE (reuse-by-import).

---

## 4. VERDICT — the single cheapest next de-risk, named concretely

**The gap is surpassable and cheap. It is not a wall.** The two hardest computations (unsupervised high-order
context-specific prediction + on-substrate teacher-free learning) are DONE (rung-4). "Word-level language model" is the
SAME computation with a word alphabet (§2a), autoregressive generation is a built-in read-out MODE of the Bouhadjar
substrate (§2b, one excitability flag — NOT the failed self-supervised generator), and the production/decode/grounding
pieces are all shipped and reuse-by-import (§1 table G1,G2,G5,G7). The transformer and VSA composer are therefore
surpassable as PERMANENT faculties: the emergent HTM-TM + the CQ serial-order renderer + the stream-cortex codes + the
moat cover prediction, production, lexical selection, and grounded-abstention. The transformer's ONLY genuinely-unique
residual value is open-domain SURFACE fluency (arbitrary-topic grammatical English), which is the LAST thing to retire and the
genuinely-hard, field-open part — everything upstream of it is now clearly buildable on-substrate.

**THE SINGLE CHEAPEST NEXT DE-RISK (path (a), do this first):**

> **Runner:** new `research/runners/_emerge15_word_sequence_lm_derisk.py` (reuse-by-import: `build_pool_bridge` +
> `OnBridgeLearner` from `_emerge14_stageC_onbridge_learning_derisk.py`; `orthogonal_drive_pattern` from
> `sim/text_embeddings.py` for the localist word→column map).
>
> **Task:** a tiny high-order WORD corpus (~8-16 words) with a genuine context-dependent branch — e.g. two sentences
> sharing a middle whose continuation depends on an EARLIER word: `["dog","chased","the","ball","home"]` vs
> `["cat","chased","the","ball","away"]` (shared "chased the ball", branch "home"/"away" determined by dog-vs-cat). This
> is the word-level analogue of the exact overlapping-sequence task rung-4 already passes.
>
> **Learn:** unsupervised, teacher-free, on the bridge (the committed `fused_htm_permanence_update`), 40 epochs, as in
> rung-4.
>
> **Measure:** held-out next-word prediction accuracy at the branch point (predicted column == correct next word),
> **against a bigram/trigram Markov baseline** (the Markov model CANNOT use the earlier dog/cat context → the HTM must
> beat it — that margin is the deliverable: "high-order word prediction the order-blind baseline cannot do").
>
> **Config:** `vocab` = corpus word count, `nE` = 16 cells/word, `act_th`=3, `p_init`=0.0 (rung-4's validated
> settings); CPU numpy backend acceptable for the de-risk, GPU for the confirm.
>
> **Anti-cheats (all mandatory):** (1) **Markov/bigram floor** — HTM must exceed it (proves high-order context use, not
> co-occurrence); (2) **dAP-lesion** → collapses to floor (prediction load-bearing); (3) **permuted-corpus** control
> (shuffle word order → chance); (4) **no-teacher** (already); (5) **held-out** branch continuations; (6) **6-seed**
> (42/43/44/100/101/102).

If (a) is GO, immediately chain (b) generation (excitability-replay roll-out on the same corpus) and (c) similarity codes
(swap in stream-cortex codes, test dog↔cat generalization) — both reuse-by-import, no `sim/` edit. The one `sim/`
extension (d, sparse multi-segment pool) is deferred until the word-LM is proven and vocabulary scale demands it — it is
the honest, biology-grounded scale mechanism, not a shortcut.

**Genuinely-hard residual (named honestly, per the directive — a mechanism to be found, not an endpoint):** open-domain
SURFACE fluency (grammatical English on ARBITRARY topics beyond the learned corpus/grammar) is the field's open frontier
and the transformer's last unique job. The path past it is not "keep the transformer" but (i) scale the on-substrate
word-LM (paths a-e) on a real streamed corpus so its learned high-order structure IS the grammar, and (ii) treat the
minimized transformer strictly as a temporary fluency teacher whose competence the emergent cortex progressively
absorbs — exactly the scaffold-then-replace program. That is the next research gate AFTER the word-LM de-risk lands, not
a reason to stop.

---

## Artifacts / key citations
- **Substrate:** `_emerge14_stageC_onbridge_learning_derisk.py`; `sim/kernels.py` (`fused_htm_permanence_update`);
  `2026-07-02-emerge14-stageC-onbridge-learning-GO-rung4-complete.md`.
- **Reusable machinery:** `sim/text_embeddings.py` (word→SDR encoders); `neural_serial_order_renderer.py` +
  `_phaseB_serial_order_multiframe_derisk.py` (CQ serial-order production, frame-conditioned);
  `_phaseB_onbridge_stream_cortex_derisk.py` + `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`
  (stream-cortex codes); `rf_phasor_composer.py` / `brain_conversational_agent.py` +
  `2026-06-23-grounded-lang-INTEGRATION-GO.md` (gate→constrain→verify moat).
- **Literature:** Bouhadjar, Wouters, Diesmann, Tetzlaff 2022, *PLoS Comput Biol* 18(6):e1010233 (sequence learning,
  prediction, replay; the ported substrate — replay = raised-excitability autonomous roll-out); Bouhadjar et al. 2023,
  *PLoS Comput Biol* 19(5):e1010989 (coherent-noise probabilistic replay); Caucheteux & King 2023, *Nat Hum Behav*
  (predictive-coding hierarchy for speech — next-word prediction is the cortical computation); Jiang & Rao 2023, *PLoS
  Comput Biol* (dynamic predictive coding, hierarchical sequence learning); Grossberg 1978 / Bullock & Rhodes 2003
  (competitive queuing); Averbeck et al. 2002/2003 (parallel primacy-graded ensembles in PFC); Garagnani & Pulvermüller
  2018, *Front Comput Neurosci* 12:88 (brain-constrained spiking word ensembles); Hawkins & Ahmad 2016 (HTM multi-segment
  cells); Cortical.io Semantic Folding (word-SDR encoder for HTM NLP).
- **Catalog (Kandel 6e):** G.10 hierarchical symbolic system (Ch 55 pp 1370-1372); G.11 dual-stream (pp 1380-1387);
  G.12 Broca (pp 1382-1384); G.13 Wernicke (pp 1384-1385); G.07/H.19 pre-SMA/SMA sequence production (Ch 30-39); verbal
  working memory Ch 52 pp 1293-1297; D.11 time cells / D.24 theta sequences (Ch 54); N.15 theta-gamma multiplex (Lisman
  & Idiart 1995); D.18 Bouhadjar three-term permanence rule (committed kernel).
