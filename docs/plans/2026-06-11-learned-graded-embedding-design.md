# The LEARNED graded-similarity cortex embedding — the design opening move (the one unbuilt piece of the dual/CLS architecture)

**Status:** READ-ONLY deep-research + design opening move (the project's standing "deep research FIRST at a new direction",
CLAUDE.md). No `sim/` code, no build, no GPU. Single deliverable: this doc + one commit. **Date:** 2026-06-11.
**Author role:** read-only design subagent. Every load-bearing project fact below is file/line-cited and re-verified against
the project's own record; the surprising one (an existing Hebbian-co-occurrence runner) was read in full, not trusted from a
summary.

**Why this doc exists (in one paragraph).** The dual / complementary-learning-systems (CLS) architecture for a cortex that
GENERALIZES ("a cat is like a dog" because related concepts get similar codes) is now **fully de-risked on the real
substrate — but only with SYNTHETIC graded codes**. The proof chain is complete: the *shape* (numpy round-trip +0.877,
`2026-06-11-dual-CLS-architecture-proof-GO.md`), the on-substrate *encode* (strong stable drive → reproducible 1.000 AND
decorrelated sparse code, `…-strong-encode-derisk-BOUNDARY.md`), and the on-substrate *recall/round-trip + generalization*
routed through the cortex channel (+1.000 round-trip, generalization 1.000 = 4× chance, with orthogonal + permuted controls
collapsing, `…-cortex-channel-derisk-GO.md`, commit `343c721d`). **The ONE unbuilt piece is the LEARNED graded-similarity
cortex embedding**: a representation in which RELATED CONCEPTS GET PROPORTIONALLY-SIMILAR CODES (cat near dog, dog near wolf,
both far from bicycle), LEARNED on neurons from experience — because (verified, §1) NO existing project code carries graded
*semantic* similarity. This doc is the design opening move for that learned embedding and, critically, its **cheap-first
de-risk** (the load-bearing falsification BEFORE the months-scale build).

---

## 0. Terms (defined once — owner standing requirement; no undefined acronym)

- **graded similarity** — related concepts get *proportionally*-similar codes: cosine(cat, dog) HIGH, cosine(cat, wolf)
  slightly lower, cosine(cat, bicycle) LOW, *systematically tracking meaning*. This is the property generalization REQUIRES
  and the thing the embedding must learn.
- **distributional semantics** — the hypothesis (Harris 1954; Firth 1957, "you shall know a word by the company it keeps")
  that a concept's *meaning* is captured by the *contexts it co-occurs in*: words appearing in similar contexts get similar
  representations. The statistical principle behind word2vec / GloVe — and, the literature says (§2), behind how the brain
  itself learns word meaning.
- **co-occurrence** — two concepts appearing together (in the same sentence / fact / window). The learning *signal* for a
  distributional / Hebbian embedding: repeated co-occurrence → strengthened association → (with the right read-out) similar
  codes for concepts that share contexts.
- **Hebbian learning** — "cells that fire together wire together" (Hebb 1949). The project's STDP is its spike-timing form
  (catalog L: NMDAR-dependent Hebbian plasticity is "the *common substrate* of refinement in every system," `feature-catalog.md`
  L-cluster). The candidate brain-based rule for learning co-occurrence structure.
- **BPTT = backpropagation-through-time** — unrolling a recurrent/spiking network over timesteps and backpropagating the
  error; the project's accepted "learned cortex" approximation (Phase 2, surrogate-gradient SNN, `sim/bptt_snn*.py`).
- **CLS = complementary learning systems** (McClelland-McNaughton-O'Reilly 1995; Kumaran-Hassabis-McClelland 2016) — a slow
  cortex that extracts *structured, overlapping* knowledge (generalization) + a fast hippocampus that stores
  *pattern-separated* specifics (no interference), LINKED by replay. The architecture this embedding plugs into.
- **the dual/CLS architecture** — the de-risked design (`docs/plans/2026-06-11-dual-CLS-architecture-design.md`): a "cortex"
  population holds graded codes (generalization), a linked "hippocampal" decorrelated expansion (between-cos ≈ 0.05) holds
  the codes the FHRR binder reads, coupled by a DG-style encode and a CA1/cortical-reinstatement retrieve. **Everything
  except the learned graded cortex code is built + validated** (`…-cortex-channel-derisk-GO.md` §"how much already exists").
- **the cortex channel** (the de-risked routing) — generalization runs *directly on the cortex codes in place* (a
  similarity read-out over the graded codebook); the round-trip reinstates the recall-identified concept's *stable cortex
  code* (cortical pattern reinstatement) rather than decoding the degraded hippocampal settled state. **This is where the
  learned graded codes live and are read.**

---

## 1. The spec — exactly what the learned embedding must produce

The learned embedding's output is **a codebook**: one code vector per concept (dimension D, the project's cortex/language-
input space). It is the drop-in replacement for the synthetic `build_graded_codebook` that every de-risk harness currently
calls. It must satisfy three concrete, *already-implemented-as-acceptance-gates* requirements — the de-risk harnesses
(`research/runners/dual_cls_architecture_proof_probe.py`, `…cortex_channel_derisk_probe.py`) test exactly these, on LEARNED
codes swapped in for synthetic.

### 1.1 The three load-bearing properties (= the gates the harnesses already encode)

| # | Property the learned codes must have | Concrete gate (the harness checks it ALREADY) | Why |
|---|---|---|---|
| **(i)** | **GRADED similarity** matching the intended semantic structure | `codebook_similarity_stats(...)["is_graded"]` = `True` (within-cluster cos ≫ between-cluster cos, margin > 0.25); AND held-out-neighbour generalization `run_generalization(...)` accuracy ≥ 0.7 (≥ ~3× chance) | the cat~dog generalization (Probe A / A1) |
| **(ii)** | **STRONG + REPRODUCIBLE** enough for the spiking encode | the codes, used as the stable DG-ensemble drive in the strong-encode regime (drive ≥ 800 pA into the assigned sparse ensemble, k=40), reproduce at cosine 1.000 AND decorrelate to between-cos ≈ 0 — the validated operating point | the strong-encode de-risk's regime (`…-strong-encode-derisk-BOUNDARY.md`) — the encode is fixed *given* a strong stable code |
| **(iii)** | **passes the de-risked ARCHITECTURE end-to-end** | swap the learned codes for `build_graded_codebook` and re-run the cortex-channel harness → cortex-channel round-trip Pearson high (≫ permuted), generalization 1.000 on graded, COLLAPSES on orthogonal + permuted | the whole point: the architecture is proven; the codes must plug in without breaking it |

### 1.2 The acceptance gates, stated precisely (reuse the de-risk harnesses verbatim, codes swapped)

The de-risk harnesses are the spec made executable. The build's acceptance is: **the learned codebook passes the SAME gates
the synthetic codebook passed**, with two ADDED anti-cheats that are specific to a *learned* (not constructed) embedding:

- **G1 — graded-ness (the learned codes are similarity-preserving):** `is_graded=True` AND the learned within/between
  margin tracks a *ground-truth* semantic structure (the toy co-occurrence design, §4) — measured by the
  correlation between the learned cosine matrix and the ground-truth relatedness matrix. *(Stronger than the synthetic
  proof, which constructed the structure: here the structure must be RECOVERED from experience.)*
- **G2 — generalization (Probe A, the headline):** `run_generalization` accuracy ≥ 0.7 (chance = 1/n_props) on the learned
  codes; **A2** the IDENTICAL test on `load_orthogonal_codes` (the project's `generate_sparse_patterns`, between-cos ≈ 0.05)
  COLLAPSES to chance; **A3** the permuted-similarity control (`run_generalization_permuted`) COLLAPSES to chance.
- **G3 — architecture pass (Probe C / cortex channel):** the cortex-channel round-trip Pearson ≥ 0.7 and ≫ permuted, with
  binding identity ~1.000 (reuse `cortex_channel_roundtrip` + `recall_identity_and_settle`).
- **G4 — strong-encode compatibility (property ii):** the learned codes drive the spiking DG at the validated operating
  point with repro 1.000 AND decorr ≈ 0 (reuse `StrongDGEncoder` from `dual_cls_strong_encode_derisk_probe.py`).
- **G5 (NEW, learned-embedding-specific — the two anti-cheats):**
  - **permuted-CO-OCCURRENCE control** (NOT the same as A3's permuted-property): scramble the *training* co-occurrence
    signal (random contexts) → re-learn → the learned codes must NOT be graded (`is_graded=False`) and generalization
    must collapse. *This is the headline learned-embedding anti-cheat* — it proves the graded structure came from the
    REAL co-occurrence statistics, not from the architecture/initialization. (The synthetic proof could not test this
    because it never learned; here it is mandatory.)
  - **beats orthogonal/random baseline:** the learned codes must generalize *strictly better* than (a) the project's
    orthogonal codes and (b) the random-Gaussian `text_embeddings.embed` codes — both of which are the project's current
    non-graded defaults.

**The honest scope of "generalization" (carried forward unchanged from the architecture proof):** the gate measures
*similarity-based property inheritance* (a held-out concept inherits a property from its trained nearest neighbours), a
real, measurable, CLS-grounded capability — NOT open-ended analogy / schema reasoning (that is a separate, larger claim,
out of scope, `…architecture-proof-GO.md` caveat 2).

---

## 2. The mechanism — ranked, biologically-grounded options

The question: **how does a brain-based rule learn graded semantic codes on neurons?** The literature (searched fresh) gives
a clear, biologically-grounded answer and a clear ranking. All three options below are brain-based or project-accepted; the
recommendation is **(A)**, with **(C)** as the higher-ceiling fallback and **(B)** as the most-novel-but-riskier route.

### 2.0 The biological anchor (why this is well-posed, not speculative)

Three independent literature results establish that **the brain learns graded semantic representations from co-occurrence
statistics, and a spiking Hebbian network can reproduce this**:

- **Garagnani & Pulvermüller 2018** (*A Neurobiologically Constrained Cortex Model of Semantic Grounding With Spiking
  Neurons and Brain-Like Connectivity*, Front. Comput. Neurosci. 12:88; PMC6232424): a neuroanatomically-realistic
  **spiking** multi-area cortex model with **Hebbian learning + local lateral inhibition + area-specific global regulation
  + uncorrelated white noise during learning** forms **cell assemblies with category-specific (graded) distributions** for
  word meanings (object words ground in perceptual areas, action words in motor areas). **This is essentially the target
  architecture and it already exists in the literature as a spiking Hebbian model — and every ingredient it needs (spiking
  pyramidals, Hebbian/STDP, FS lateral inhibition, OU/white noise, multi-region connectivity) the project already ships.**
- **Białas, Mirończuk & Protasiewicz 2020** (*Biologically Plausible Learning of Text Representation with Spiking Neural
  Networks*, arXiv:2006.14894 / PPSN XVI): an **STDP**-trained SNN whose neurons "become responsive only to groups of words
  that co-occur frequently," producing a low-dimensional spike-based text code (80% on 20-newsgroups). Direct precedent that
  a *spike-timing Hebbian* rule learns distributional structure.
- **Hultén et al. 2021** (*The neural representation of abstract words may arise through grounding word meaning in language
  itself*, Hum. Brain Mapp. 42:4973): a **Hebbian + word-co-occurrence** statistical model accounts for the neural
  (MEG) representation of BOTH concrete AND abstract words; "word abstractness emerged from the statistical regularities of
  the language environment." The biological validation that co-occurrence statistics ARE how cortex encodes graded meaning.

**Take-away:** the brain-based mechanism is *Hebbian learning of co-occurrence structure* (option A), it is well-precedented
in spiking models, and the project already has every component. The risk is not "is it biologically possible" (it is) — the
risk is "can it produce *strong-enough* graded structure for real generalization at scale" (§5, risk i).

### 2.1 ⭐ OPTION A (RECOMMENDED) — Hebbian / distributional co-occurrence embedding

**One line:** concepts that occur in *similar contexts* (co-occur with overlapping sets of other concepts) develop
*similar codes*, learned by Hebbian growth on a recurrent (or two-layer) concept population — a spiking-Hebbian
word2vec/CBOW analogue, and **the project already ships a validated prototype of the core mechanism**
(`research/runners/learned_assoc_graph.py`).

- **Mechanism.** A concept = a sparse K-of-N pattern in a pool with a PLASTIC excitatory recurrent (or a hidden "embedding"
  layer between an input concept layer and a context-prediction layer). Training co-fires concepts that appear together in
  a fact/sentence/window; Hebbian/STDP growth on the recurrent (or input→hidden) weights LEARNS the pairwise co-occurrence.
  The **graded code** is then *derived* from the learned associative structure: two concepts that co-occur with overlapping
  neighbour-sets end up with overlapping recurrent in/out weight profiles → their pool-activity patterns (when each is
  cued, after a few settle steps of the recurrent) become *similar* in proportion to their shared context — i.e. the
  settled/spread pattern is the graded code. (This is the distributional principle realized as Hebbian spreading: "second-
  order" co-occurrence — *shared neighbours* — produces graded similarity even when two concepts never directly co-occur,
  exactly as cat~dog can be close via shared contexts without ever appearing together.)
- **Reuses (substantial — the headline efficiency finding):**
  - **`research/runners/learned_assoc_graph.py` (`LearnedAssocGraph`)** — ALREADY BUILT + validated: `store_fact` co-fires
    a fact's concept patterns and "the recurrent LEARNS their pairwise co-occurrence by Hebbian growth (NOT set)";
    `graph()` reads the learned recurrent weights. **Validated multi-seed: the learned graph's top associates match the
    Python co-occurrence oracle** (the dlPFC picks the same associates). It uses the validated `_D_sparse_heteroassoc`
    sparse heteroassociative memory (Marr/Treves-Rolls CA3 autoassociator), the brain-region framework, Hebbian, and
    `generate_sparse_patterns` — **NO `sim/` edits.** This is the *direct-co-occurrence* (first-order) version; the
    embedding build extends it to read out *graded codes* (the second-order, shared-neighbour similarity), which is a
    read-out + a small architectural addition, not a from-scratch arc.
  - the concept pools (`concept_pool_sparse_distributed.py`) — the strong + reproducible learned pools (property ii), and
    `generate_sparse_patterns` for the sparse concept assignment.
  - the project's STDP / Hebbian plasticity (`cp_plasticity_rate_gain` gates to freeze/thaw), FS lateral inhibition (the
    Garagnani-Pulvermüller "local lateral inhibition"), OU noise (their "white noise during learning").
- **Brain-based-ness:** HIGH. Pure Hebbian/STDP co-occurrence is the catalog's L-cluster "common substrate" and the
  Garagnani-Pulvermüller / Białas / Hultén mechanism. No backprop, no host-trained embedding. Fully on-substrate.
- **Data/experience signal it needs:** a co-occurrence corpus — *which the project can supply three ways* (§3): the
  conversational agent's own fact KB (facts = co-occurring concept tuples, exactly `learned_assoc_graph`'s `store_fact`
  input), a small text corpus (Tiny-Shakespeare-style, already tokenized by `sim/bpe_tokenizer.py`), or grounded
  multimodal co-occurrence (V1/ventral features co-active with words).
- **Build cost:** MEDIUM. The core (Hebbian co-occurrence on a recurrent pool) is built; the new work is (1) the graded-
  code *read-out* (cue a concept → settle the recurrent → read the spread pattern as the code), (2) scaling the co-occurrence
  source beyond toy facts, (3) tuning so the codes are strong+reproducible (property ii) AND graded (property i)
  simultaneously (the central tension, risk ii).
- **Risk:** the strong-vs-graded tension (risk ii) and whether Hebbian co-occurrence produces graded-*enough* structure for
  real generalization (risk i). Both are exactly what the cheap de-risk (§4) targets.

### 2.2 OPTION C (fallback / higher ceiling) — surrogate-gradient BPTT contrastive/next-token cortex

**One line:** train the project's accepted "learned cortex" (the Phase 2 surrogate-gradient SNN) on a *next-token or
contrastive* objective over a corpus; the hidden layer's learned representation IS a distributional embedding (graded by
construction of the objective), then read the per-concept hidden code as the cortex codebook.

- **Mechanism.** Next-token prediction (or a contrastive "same-context-positive / random-negative" loss) on a corpus is
  *exactly* the word2vec/CBOW objective; a network trained on it develops hidden codes where words in similar contexts are
  close (this is why word2vec works). The project's Phase 2 stack (`sim/bptt_snn.py` / `bptt_snn_gpu.py` /
  `surrogate_grad.py` + `sim/bpe_tokenizer.py` for word-level tokens) already does surrogate-gradient BPTT next-char
  prediction on Tiny Shakespeare (Phase 2.2: loss 14.1→2.24, perplexity ~9.4). Switching char→word tokens (BPE) + reading
  the hidden layer = a distributional word embedding.
- **Reuses:** `sim/bptt_snn_gpu.py` (the validated GPU BPTT SNN), `sim/surrogate_grad.py` (ATan/fast-sigmoid surrogates),
  `sim/char_tokenizer.py` / `sim/bpe_tokenizer.py` (the corpus tokenizer, BPE is word-level + drop-in compatible),
  `sim/bptt_snn.py` (the numpy reference + gradient check). **On the `path-f-hybrid` branch** (Phase 2 lives there, not
  main).
- **Brain-based-ness:** MEDIUM — the project's *accepted* approximation, explicitly documented as "the project-accepted
  learned-cortex" (CLAUDE.md, the composer-idealization note treats BPTT as legitimate). NOT a host sklearn embedding
  pasted in (that would be a cheat); it is a spiking network trained end-to-end. But backprop-through-time is not local
  Hebbian — so it is the *accepted-shortcut* tier, below option A's pure-Hebbian tier.
- **Data/experience signal:** a text corpus (Tiny Shakespeare is in-repo; FineWeb-Edu / a children's-book corpus for
  richer semantics). Word-level (BPE) so the per-word hidden code is the embedding.
- **Build cost:** MEDIUM-HIGH. The BPTT stack is built but on a side branch; the new work is word-level tokenization at
  scale, a contrastive/next-word head, reading the hidden codebook, and the known scale problem (Phase 2.3a NEGATIVE:
  toy 134K-param char-level features did NOT transfer to word-action binding — "~4 orders too small," CLAUDE.md). Word-level
  + a larger corpus + the *embedding read-out* (not the downstream task) may clear it, but scale is the live risk.
- **Risk:** the Phase 2.3a scale gap (risk i, directly evidenced) — a toy BPTT cortex produced WEAK transfer. Higher
  ceiling than Hebbian IF scaled; the de-risk must check whether even a *small* contrastive objective yields graded codes
  that pass the architecture.

### 2.3 OPTION B (most novel, riskier) — predictive / successor-representation temporal-context embedding

**One line:** concepts predicted by (or predicting) similar *temporal contexts* get similar codes — a successor-
representation (SR) embedding, the hippocampal-cortical "cognitive map of semantic space."

- **Mechanism.** The SR encodes each state by its *expected future occupancy* of other states; states with similar
  successor profiles get similar codes. Applied to concept sequences (sentences / fact chains), concepts that lead to /
  follow from similar concepts cluster — a *predictive* distributional embedding. **Stoewer et al. 2022** (Sci. Rep. 12:3818)
  built exactly this: a neural network learns a "cognitive map" of semantic space from 32 animals as feature vectors via
  *multi-scale successor representations*, recovering biological-class clusters (mammals/amphibians/insects) AND
  interpolating novel/incomplete input at up to 95% — i.e. the generalization capability, from a brain-grounded predictive
  map. **Fang et al. 2022** (eLife 11:e80680) gives *biologically-plausible local learning rules* that compute the SR in a
  recurrent network (synaptic weights → transition matrix; gain → predictive horizon). **Ekman 2022** (eLife) shows human
  V1+hippocampus actually carry SR-like predictive codes.
- **Reuses:** the brain-region framework + recurrent pools + Hebbian (Fang's rules are Hebbian-like); conceptually overlaps
  the project's SPEAR/temporal-context interest (MEMORY.md `feedback_conversational_path_resolution`). No SR runner exists
  yet — more new code than A.
- **Brain-based-ness:** HIGH (Fang's rules are biologically plausible + local; the SR is the hippocampal predictive-map
  theory). But the *project* has no SR machinery, so it is more from-scratch than A.
- **Data/experience signal:** concept *sequences* (ordered) — sentences or fact chains where order/transition matters
  (richer than A's unordered co-occurrence; the agent's KB would need sequence structure).
- **Build cost:** HIGH (new SR learning + read-out + sequence data). **Risk:** most novel, least project-precedent;
  attractive *if* the owner wants the predictive/temporal-context framing (which MEMORY.md flags as the owner's preferred
  conversational direction), but a bigger bet than A.

### 2.4 Recommendation

**Build OPTION A (Hebbian co-occurrence), because the core mechanism is ALREADY BUILT AND VALIDATED on the substrate
(`learned_assoc_graph.py` learns concept co-occurrence by Hebbian growth, multi-seed-matched to the co-occurrence oracle),
it is the highest brain-based tier (pure Hebbian/STDP, the catalog L-cluster + the Garagnani-Pulvermüller spiking precedent),
and it reuses the most project machinery — so it is the cheapest path to a brain-based graded embedding.** Keep **C (BPTT)**
as the documented higher-ceiling fallback if Hebbian graded structure proves too weak for real generalization (risk i), and
**B (SR)** as the option to revisit if the owner prioritizes the predictive/temporal-context conversational framing.

---

## 3. Reusable machinery + the data/experience source (where the graded structure comes from)

**The single most important concrete question for a from-scratch agent: WHERE does the learning signal (the co-occurrence /
context structure) come from?** Three project-available sources, ranked by readiness:

| Source | What it is | Project asset (file-cited) | Readiness |
|---|---|---|---|
| **(1) the agent's own fact KB** ⭐ | the conversational agent's stored facts ARE co-occurring concept tuples (SVO facts: dog–go–north, cat–run–south). Concepts sharing facts → shared context → graded. | `learned_assoc_graph.py` `store_fact(concept_list)` ingests *exactly* these tuples; the 320-concept flat-distinct KB (`2026-06-02-…`) + the agent's `_assoc_graph`/`learned_assoc_graph` co-occurrence is the live source. | **HIGHEST** — built, validated, on-substrate, no external data |
| **(2) a small text corpus** | unordered co-occurrence windows (option A) or token sequences (options B/C) from a real corpus. | `sim/bpe_tokenizer.py` (word-level BPE, in-repo), Tiny Shakespeare corpus (Phase 2.2, in-repo), `sim/char_tokenizer.py`. | MEDIUM — corpus + tokenizer exist; the co-occurrence/sequence extraction is small new code |
| **(3) grounded multimodal co-occurrence** | words co-active with V1/ventral perceptual features (a cat-image's ventral code co-fires with "cat") → grounds the embedding in perception (Garagnani-Pulvermüller's object-vs-action grounding). | `sim/visual_cortex.py` (the real V1 Gabor bank, graded *perceptual* similarity), the cheat-#4 grounded pipeline. | LOWER — richest + most biological, but the most integration; the perceptual graded structure is real (bar_0deg~bar_22deg) but the production path ZCA-decorrelates it (risk) |

**Recommendation for the data source: (1) the agent's own fact KB, with (2) a small corpus as the scaling extension.** The
KB is built, validated, on-substrate, needs no external download, and is the most honest "from-scratch agent learns from its
own experience" story. **Honest caveat (the data-poverty risk, risk iii):** a from-scratch agent's KB is *small* (hundreds
of facts), and good semantic embeddings classically need *large* co-occurrence data. The cheap de-risk (§4) must therefore
use a *controlled toy co-occurrence with KNOWN ground-truth structure* (so "did it learn the intended graded structure" is
measurable), and the months-scale build must confront whether the agent's real (small) experience is rich enough — or
whether source (2)/(3) must augment it.

Other reused machinery (the full inventory): the **dual/CLS plumbing** the embedding plugs into is ~80% built and validated
(decorrelated codes, DG separation D.12, CA3 completion D.13, CA1→cortex link, SWR-replay write-back N.14 at 94% retention,
engram index, FHRR binder, NEF cleanup, no-confab gate, the merged one-bridge host — full table in
`…-cortex-channel-derisk-GO.md`). The embedding is the **last** piece; everything it must connect to is waiting.

---

## 4. The cheap-first de-risk (the load-bearing falsification, specified precisely)

**The single open question:** *can a BRAIN-BASED learning rule produce graded codes that PASS the de-risked architecture —
and does the strong-reproducible requirement (for the encode) conflict with the graded requirement?* Everything downstream
(binding, separation, completion, the cortex-channel routing, generalization-given-graded-codes) is ALREADY validated
(§1, the de-risk chain). The de-risk isolates exactly the NEW claim. **The principle: prove a brain-based rule can LEARN the
graded codes BEFORE committing the months-scale build — gate the expensive build on the cheap learning proof, exactly as the
architecture proof gated the architecture.**

### 4.1 The probe (CPU/numpy first, then tiny GPU; reuse-by-import; multi-seed 42/43/44)

**New probe:** `research/runners/learned_graded_embedding_derisk_probe.py` (the build's first artifact; the doc specifies it,
it is NOT written here). It does FOUR things:

**STEP 1 — define a TOY but REAL co-occurrence signal with KNOWN ground-truth graded structure.** Build a small set of
concepts with a *defined* cluster structure (e.g. the architecture-proof's 8 clusters × 5 concepts = 40 concepts), then
generate a **co-occurrence corpus** that REFLECTS that structure: concepts in the same cluster co-occur frequently (appear
together in many synthetic "facts"/"sentences"), concepts in different clusters rarely. Critically, **make cat~dog close via
SHARED NEIGHBOURS, not (only) direct co-occurrence** — include the second-order case (two concepts that never directly
co-occur but share contexts) so the embedding is tested on real distributional structure, not just direct association. The
ground-truth relatedness matrix `S_true` (from the cluster structure) is the target the learned codes must recover.

**STEP 2 — LEARN the codes with the brain-based rule (option A).** Reuse **`learned_assoc_graph.LearnedAssocGraph`**: ingest
the toy co-occurrence corpus via `store_fact` (Hebbian growth on the plastic recurrent learns the co-occurrence). Then
**read out the graded codes**: for each concept, cue its sparse pattern, settle the learned recurrent a few steps, and read
the spread pool-activity pattern as that concept's code (the graded read-out — the new piece). Tiny scale: the
`_D_sparse_heteroassoc` pool (n_pool ~1500), CPU/numpy (`SIM_BACKEND=numpy`) for the harness check, then tiny GPU for the
real read.

**STEP 3 — measure GRADED-NESS against ground truth (G1).** `codebook_similarity_stats(learned_codes, labels)` →
`is_graded` must be `True`; AND Pearson(off-diag of learned cosine matrix, off-diag of `S_true`) must be high (the learned
structure recovers the intended structure — the *recovery* gate, stronger than the synthetic proof's *constructed* structure).

**STEP 4 — run the de-risked ARCHITECTURE gates on the LEARNED codes (G2/G3/G4).** Swap `learned_codes` for the synthetic
`build_graded_codebook` output and call the EXISTING harness functions verbatim:
- `run_generalization(learned_codes, …)` ≥ 0.7 (**A1**); `run_generalization(load_orthogonal_codes(...), …)` collapses
  (**A2**); `run_generalization_permuted(learned_codes, …)` collapses (**A3**). [Probe A, from
  `dual_cls_architecture_proof_probe.py`.]
- the cortex-channel round-trip (`cortex_channel_roundtrip` + `recall_identity_and_settle` from
  `dual_cls_cortex_channel_derisk_probe.py`) → Pearson high ≫ permuted, binding identity ~1.000 (**G3/Probe C**).
- the strong-encode compatibility (`StrongDGEncoder` from `dual_cls_strong_encode_derisk_probe.py`): drive the spiking DG
  with the learned codes' sparse ensembles at the validated operating point → repro 1.000 AND decorr ≈ 0 (**G4** — this is
  where the strong-vs-graded tension, risk ii, is directly measured).

### 4.2 The brain-based bar (mandatory — the rule must be Hebbian/predictive/local OR the project-accepted BPTT)

The learning rule in STEP 2 MUST be the project's Hebbian/STDP (via `learned_assoc_graph`, which is) — NOT a host-trained
sklearn / numpy SVD / gensim word2vec embedding pasted in. **If a host embedding is used at all, it is ONLY as a labelled
CEILING** (the "best a tuned objective achieves on this toy co-occurrence" reference), reported as the ceiling, with the
brain-based Hebbian result as the deliverable. (Mirrors the de-risk chain's "clean-decode ceiling" positive-control
convention.) This keeps the de-risk honest about the BRAIN-BASED claim, per the owner's standing BRAIN-BASED-ONLY directive
(MEMORY.md `feedback_brain_based_only_standard`).

### 4.3 Anti-cheats (all mandatory — the de-risk is rejected without them)

- **PERMUTED-CO-OCCURRENCE control (the HEADLINE learned-embedding anti-cheat, G5):** scramble the toy co-occurrence corpus
  (random contexts, same concepts) → re-learn → the learned codes must NOT be graded and generalization must COLLAPSE to
  chance. **This proves the graded structure came from the REAL co-occurrence statistics, not from the architecture /
  initialization / the read-out.** It is the load-bearing control the synthetic proof could not run (it never learned), and
  the analogue of the architecture proof's A3 permuted-similarity headline.
- **HELD-OUT-NEIGHBOUR generalization on the LEARNED structure (A1, reused):** a concept never trained in a property table
  inherits the property from its learned nearest neighbours — the cat~dog test, now on *learned* codes.
- **ORTHOGONAL-codes contrast (A2, reused):** generalization collapses on `generate_sparse_patterns` codes — proves it is
  similarity-driven.
- **BEATS-baseline (G5):** the learned codes generalize strictly better than the random-Gaussian `text_embeddings.embed`
  codes AND the orthogonal codes (the project's current non-graded defaults) — so "learned" is doing real work.
- **PERMUTED-S round-trip baseline (reused):** the cortex-channel Pearson on row-shuffled codes ≈ 0.
- Native code conventions (mean-removed, unit-norm) asserted; multi-seed 42/43/44.

### 4.4 Decision logic (stated explicitly)

- **GO** (justifies the months-scale build): G1 (learned codes recover the graded structure) AND G2 (generalize, with A2+A3
  collapsing) AND G3 (pass the cortex-channel architecture) AND G4 (strong-encode compatible) AND G5 (permuted-co-occurrence
  collapses + beats baseline), all multi-seed. ⇒ "a brain-based Hebbian rule CAN learn graded codes that pass the de-risked
  architecture" — the months-scale build is justified end-to-end.
- **BOUNDARY_weak_graded** (the most likely partial — risk i): G5's permuted control collapses cleanly (the structure is
  real) AND graded-ness recovers the structure (G1) BUT generalization is only marginal (e.g. 0.4–0.6, above chance but
  below 0.7). ⇒ the brain-based rule learns *some* graded structure but not strong enough for the bar; the next probe is
  more co-occurrence data / second-order tuning, OR escalate to option C (BPTT). *This is a useful, expected boundary — it
  localizes "Hebbian is too weak" cheaply, before the build.*
- **BOUNDARY_strong_vs_graded_conflict** (risk ii): the codes can be made graded (G1) OR strong-reproducible for the encode
  (G4) but NOT both at one operating point. ⇒ the dual architecture needs the *strong sparse* code (hippocampal side) and
  the *graded* code (cortex side) to be DIFFERENT populations linked by a learned map (which is, in fact, the dual design's
  intent), not the same vectors — report it, it sharpens the architecture.
- **NEGATIVE_not_co-occurrence_driven:** generalization passes on graded BUT the permuted-co-occurrence control ALSO passes
  (the "graded structure" is an artifact of the read-out / architecture, not the learned co-occurrence). ⇒ the mechanism is
  not actually learning from experience; the claim is false. No banking.

### 4.5 Cost of the de-risk

CPU/numpy harness check + tiny GPU for the spiking read/encode; reuse-by-import; ~minutes per seed (the `learned_assoc_graph`
prototype runs facts in seconds; the architecture harnesses are the validated ~3.5 s numpy / ~167 s GPU multi-seed). **No
`sim/` edits** (the Hebbian co-occurrence is a brain-region-framework + plasticity-gate operation; the read-out and the
architecture gates are readout/cognitive operations — exactly the constraint the whole de-risk chain held).

---

## 5. Honest risk register (every load-bearing assumption, flagged)

### 5.1 ⚠️ (i) A brain-based Hebbian/predictive rule may produce only WEAK / COARSE graded structure — generalization may be marginal (THE BIGGEST RISK)
**The load-bearing risk, and the one the cheap de-risk must target first.** Good semantic embeddings classically need LARGE
data + a TUNED objective (word2vec/GloVe train on billions of tokens); biological Hebbian learning is *weaker* than
backprop-on-big-data (this is the classic "biological learning < backprop" gap, and the project has already MET it once —
Phase 2.3a: a toy BPTT cortex's features did NOT transfer, "~4 orders too small," CLAUDE.md / `2026-05-07-Phase-2.3a-…`).
**So the realistic failure mode is `BOUNDARY_weak_graded`: the Hebbian rule learns the RIGHT structure (permuted control
collapses → it IS co-occurrence-driven) but the generalization is only marginally above chance, not ≥ 0.7.** **Mitigation:**
(a) the de-risk uses a *controlled toy co-occurrence with strong, KNOWN ground-truth structure* so weak-but-real is
distinguishable from absent; (b) the *second-order* read-out (shared-neighbour similarity, not just direct association) is
specifically the thing that lifts coarse first-order association into graded similarity — test it explicitly; (c) the
fallback ladder is explicit (more data → option C BPTT → option B SR), each a documented escalation, NOT a surprise at build
time. **Do not let "the architecture is proven" hide "the LEARNED codes may be too weak to exploit it" — that is the entire
remaining unknown.**

### 5.2 ⚠️ (ii) STRONG-reproducible (for the encode) may CONFLICT with GRADED (for generalization)
The dual architecture needs the codes to be BOTH strong+reproducible (property ii — to drive the spiking DG into the
reproducible+decorrelated regime, the strong-encode de-risk's requirement) AND graded (property i — overlapping for similar
concepts). **These pull opposite ways:** the project's *strong + reproducible* learned codes are the concept pools, which
are ORTHOGONAL by construction (`--orthogonal-codes`, the v14 breakthrough made them *more* separable); a *graded* code is
by definition NOT orthogonal. The strong-encode de-risk used `generate_sparse_patterns` (orthogonal) as the strong stable
DG drive — so "strong stable" was demonstrated on NON-graded codes. **Open question: is there a strong-AND-graded regime?**
**Mitigation (already in the dual design's logic):** the architecture does NOT require the *same* vectors to be both — the
**cortex** codes are graded (read for generalization, in place), the **hippocampal/DG** codes are decorrelated+strong (read
for binding), and they are LINKED by a learned map. The encode does not need the *cortex* code to be strong-reproducible; it
needs the *DG ensemble assignment* to be strong-stable (which it is, independent of the cortex code's graded-ness). So the
likely resolution is "graded cortex code → learned encode → strong stable DG ensemble," and G4 in the de-risk tests exactly
whether a graded cortex code can drive a clean DG encode. **If G4 fails on graded codes, that is `BOUNDARY_strong_vs_graded_
conflict` and it sharpens the architecture (the two codes must be different populations) rather than killing it.** Flag:
this conflict is real and under-explored; surface it as a first-class de-risk output.

### 5.3 (iii) WHERE the experience/data comes from on a from-scratch agent (data poverty)
A from-scratch agent has no corpus; its co-occurrence experience is its own (small) fact KB (hundreds of facts at V=320).
Good embeddings want large data. **Risk:** the real agent's experience may be too sparse to learn graded structure even if
the toy de-risk passes (the de-risk uses a *controlled* corpus richer than the agent's real one). **Mitigation:** (a)
source (2) — a small text corpus (Tiny Shakespeare in-repo, or a children's-book corpus) — augments the KB; (b) the de-risk's
toy corpus should be sized comparably to the agent's plausible real experience (NOT artificially huge), so the GO is honest
about the data regime; (c) report the corpus size at which graded structure emerges (the data-efficiency curve) as a
build-scoping output. **Flag: the de-risk must NOT use an unrealistically large toy corpus and call it a GO for a data-poor
agent.**

### 5.4 (iv) The months-scale build could balloon (scope creep)
The embedding is "the one unbuilt piece," but it is the DEEP one (the architecture proof's caveat 1: "comparable in scope to
the dendritic rewrite Option B was trying to avoid"). **Risk:** the read-out tuning + data scaling + the strong-vs-graded
resolution + on-substrate integration could each be a sub-arc. **Mitigation:** the cheap de-risk's verdict structure
(GO / BOUNDARY_weak / BOUNDARY_conflict / NEGATIVE) explicitly bounds the build: a clean GO means "scale the validated
mechanism" (bounded); a BOUNDARY means "the mechanism needs X before scaling" (the build is gated on fixing X first, not
discovered mid-build). The incremental/resumable trainer (`concept_pool_sparse_distributed --resume-from`) already chunks
long substrate trainings. **Flag: do NOT start the months-scale build until the cheap de-risk returns GO or a localized
BOUNDARY with a named fix.**

### 5.5 (v) "Generalization" may be a shallow kNN trick, not a brain capability (scope honesty)
A similarity-weighted vote over graded codes generalizing is, at one level, "kNN works on an embedding." **Risk:** a GO could
over-claim "the cortex reasons." **Mitigation (carried from the architecture proof, caveat 2):** scope the claim as
*similarity-based property inheritance* — a real, CLS-grounded capability the flat composer cannot do — NOT analogy / schema
reasoning. The honest deliverable is "a brain-based rule learns codes that support similarity-based generalization on the
de-risked architecture," nothing more.

### 5.6 Assumption ledger (load-bearing claims this design rests on)
- **A-1 (VERIFIED).** No existing project code carries graded *semantic* similarity (denoise64 correlated-but-not-semantic
  0.81±0.04 uniform; V1 graded-but-perceptual + ZCA-discarded; concept pools orthogonal-by-construction;
  `text_embeddings.embed` random-Gaussian near-orthogonal — read at `sim/text_embeddings.py:49`). *Load-bearing: it is why
  the embedding is genuinely-new.* (Re-verified the architecture-proof's §3 verdict + read `text_embeddings.py` directly.)
- **A-2 (VERIFIED).** The dual/CLS architecture is fully de-risked on the real substrate with synthetic graded codes
  (shape +0.877; encode repro 1.000 + decorr ≈ 0; cortex-channel round-trip +1.000 + generalization 4× chance, multi-seed)
  — `…architecture-proof-GO`, `…strong-encode-derisk-BOUNDARY`, `…cortex-channel-derisk-GO` (commit `343c721d`).
  *Load-bearing: everything except the learned codes is ready.*
- **A-3 (VERIFIED).** A spiking Hebbian/STDP co-occurrence rule CAN learn distributional / graded semantic structure —
  Garagnani-Pulvermüller 2018 (spiking + Hebbian + lateral inhib → category-graded cell assemblies), Białas 2020 (STDP →
  co-occurring word groups), Hultén 2021 (Hebbian + co-occurrence → graded concrete+abstract meaning, MEG-matched).
  *Load-bearing: the mechanism is biologically real and well-precedented; the open question is STRENGTH at scale, not
  possibility.*
- **A-4 (VERIFIED).** The project ALREADY ships a validated Hebbian-co-occurrence prototype (`learned_assoc_graph.py`:
  Hebbian growth learns concept co-occurrence, multi-seed-matched to the oracle, NO sim/ edits). *Load-bearing: option A is
  assembly-plus-a-read-out, not from-scratch — read the runner in full.*
- **A-5 (UNVERIFIED — the de-risk's job).** A brain-based rule produces graded codes STRONG ENOUGH to pass the de-risked
  architecture's generalization bar (≥ 0.7), and a graded cortex code is compatible with the strong spiking encode (or the
  conflict resolves via separate linked populations). *This is the entire point of §4; tested, not assumed.*

---

## Verdict

**The recommended mechanism: OPTION A — a Hebbian / distributional co-occurrence embedding** (concepts in similar contexts
get similar codes, via Hebbian growth on a plastic recurrent concept pool + a graded read-out). **One-line why:** the core
mechanism is ALREADY BUILT AND VALIDATED on the substrate (`research/runners/learned_assoc_graph.py` — Hebbian growth learns
concept co-occurrence, multi-seed-matched to the co-occurrence oracle, NO `sim/` edits), it is the highest brain-based tier
(pure Hebbian/STDP, the catalog L-cluster "common substrate," directly precedented by the Garagnani-Pulvermüller 2018 spiking
semantic-grounding model), and it reuses the most project machinery — the cheapest path to a brain-based graded embedding.

**The data/experience source it needs:** the conversational agent's own fact KB (SVO facts = co-occurring concept tuples,
exactly `learned_assoc_graph.store_fact`'s input; on-substrate, no external download), with a small in-repo corpus
(`sim/bpe_tokenizer.py` + Tiny Shakespeare) as the scaling extension. **Honest data-poverty caveat:** a from-scratch agent's
KB is small; good embeddings classically need large data — so the de-risk uses a controlled toy co-occurrence sized to the
plausible real regime, and the build must confront whether real experience is rich enough.

**The precise cheap-first de-risk:** a new CPU/numpy-first, tiny-GPU, reuse-by-import, multi-seed probe
(`learned_graded_embedding_derisk_probe.py`) that (STEP 1) defines a TOY but real co-occurrence corpus with KNOWN
ground-truth graded structure (including second-order shared-neighbour cat~dog), (STEP 2) LEARNS the codes with the
brain-based Hebbian rule (reuse `learned_assoc_graph` + a graded read-out), (STEP 3) verifies the learned codes RECOVER the
graded structure (G1, Pearson vs ground truth), and (STEP 4) runs the de-risked ARCHITECTURE gates on the LEARNED codes
(swap synthetic→learned): generalization ≥ 0.7 with the orthogonal (A2) + permuted-property (A3) controls collapsing, the
cortex-channel round-trip closing, and strong-encode compatibility (G4). **The brain-based bar:** the rule is the project's
Hebbian/STDP (NOT a host sklearn/word2vec embedding pasted in; a host embedding is allowed ONLY as a labelled ceiling).
**The headline anti-cheat:** a PERMUTED-CO-OCCURRENCE control — scramble the training co-occurrence → the learned codes must
NOT be graded and generalization must COLLAPSE (proving the structure came from the real statistics, not the architecture);
plus beats-the-random/orthogonal-baseline.

**The single biggest risk:** **can a brain-based rule produce graded-ENOUGH structure for real generalization** — biological
Hebbian learning is classically weaker than backprop-on-big-data, and the project has already met this gap once (Phase 2.3a
toy BPTT features did not transfer). The realistic partial is `BOUNDARY_weak_graded` (the rule learns the RIGHT structure but
only marginally generalizes). **Closely behind: does STRONG-reproducible (for the encode) CONFLICT with GRADED (for
generalization)** — the project's strong+reproducible codes are orthogonal-by-construction, and a graded code is by
definition not; the likely resolution is that the cortex (graded) and DG (strong+decorrelated) codes are DIFFERENT linked
populations (the dual design's intent), which G4 tests directly. Both risks are exactly what the cheap de-risk targets
BEFORE the months-scale build is committed.

**No banking.** Reported exactly as found — including the parts that reshape the arc (the Hebbian-co-occurrence mechanism is
already partly built and validated; the strong-vs-graded tension is real and under-explored; biological Hebbian may be too
weak and the de-risk must localize that cheaply).

## Sources

- McClelland, McNaughton & O'Reilly 1995, *Why there are complementary learning systems…* (Psychol. Rev. 102:419);
  Kumaran, Hassabis & McClelland 2016 (Trends Cogn. Sci. 20:512; PMID 27315762) — CLS theory.
- **Garagnani & Pulvermüller 2018**, *A Neurobiologically Constrained Cortex Model of Semantic Grounding With Spiking
  Neurons and Brain-Like Connectivity* (Front. Comput. Neurosci. 12:88; PMC6232424) — spiking + Hebbian + lateral inhibition
  → category-graded cell assemblies for word meaning. The direct architectural precedent for option A.
- **Białas, Mirończuk & Protasiewicz 2020**, *Biologically Plausible Learning of Text Representation with Spiking Neural
  Networks* (arXiv:2006.14894; PPSN XVI) — STDP-trained SNN learns co-occurring-word-group codes (distributional, 80% on
  20-newsgroups).
- **Hultén et al. 2021**, *The neural representation of abstract words may arise through grounding word meaning in language
  itself* (Hum. Brain Mapp. 42:4973; ~28 citations) — Hebbian + co-occurrence statistical model accounts for MEG
  representation of concrete AND abstract words; graded meaning emerges from language statistics.
- **Stoewer et al. 2022**, *Neural network based formation of cognitive maps of semantic spaces and the putative emergence
  of abstract concepts* (Sci. Rep. 12:3818) — multi-scale successor-representation learns a graded cognitive map of semantic
  space (animal classes cluster; novel input interpolated to 95%). The option-B precedent.
- **Fang, Aronov, Abbott & Mackevicius 2022**, *Neural learning rules for generating flexible predictions and computing the
  successor representation* (eLife 11:e80680) — biologically-plausible local rules compute the SR in a recurrent network.
- **Ekman, Kok & de Lange 2022**, *Successor-like representation guides the prediction of future events in human visual
  cortex and hippocampus* (eLife) — empirical SR-like predictive codes in human V1 + hippocampus.
- **Chersoni et al. 2021**, *Decoding Word Embeddings with Brain-Based Semantic Features* (Comput. Linguist.; ~60 citations)
  — distributional embeddings map onto neurobiologically-motivated semantic features (Binder 2016) — the bridge between
  word-embedding similarity and brain-based semantic dimensions.
- Catalog (`sim-catalog/references/feature-catalog.md`): L-cluster (NMDAR-dependent Hebbian plasticity = "the common
  substrate of refinement in every system" = the project's STDP); G.10–G.14 (language, dual-stream, Wernicke
  "selects words matching intended meaning," "semantic memory store" prerequisite). Kandel 6e Ch 55 (language).

## Project cross-references (internal, all re-verified)

- The spec the learned codes must pass (the de-risk chain): `research/findings/2026-06-11-dual-CLS-architecture-proof-GO.md`
  (shape +0.877), `…-dual-CLS-strong-encode-derisk-BOUNDARY.md` (encode fixed; strong-vs-graded data point),
  `…-dual-CLS-cortex-channel-derisk-GO.md` (cortex-channel round-trip +1.000 + generalization 4× chance, commit `343c721d`);
  the architecture design `docs/plans/2026-06-11-dual-CLS-architecture-design.md` (§2 what's built, §3 graded-code verdict).
- The de-risk harnesses to reuse VERBATIM (codes swapped synthetic→learned): `research/runners/dual_cls_architecture_proof_probe.py`
  (`build_graded_codebook` ← the swap target; `codebook_similarity_stats`, `run_generalization`,
  `run_generalization_permuted`, `assign_properties`, `load_orthogonal_codes`); `…dual_cls_cortex_channel_derisk_probe.py`
  (`cortex_channel_roundtrip`, `recall_identity_and_settle`); `…dual_cls_strong_encode_derisk_probe.py` (`StrongDGEncoder`,
  `assign_sparse_dg_ensembles` — the G4 strong-encode compatibility check).
- The brain-based learning machinery to reuse: **`research/runners/learned_assoc_graph.py`** (`LearnedAssocGraph` — the
  validated Hebbian-co-occurrence prototype, option A's core; read in full); `research/runners/_D_sparse_heteroassoc.py`
  (the sparse heteroassociative pool it builds on); `research/runners/concept_pool_sparse_distributed.py`
  (`generate_sparse_patterns`, `train_concept_sparse`, the strong+reproducible pools); `sim/bptt_snn.py` / `bptt_snn_gpu.py`
  / `surrogate_grad.py` / `char_tokenizer.py` / `bpe_tokenizer.py` (option C, the BPTT cortex + word-level tokenizer, on
  `path-f-hybrid`); `sim/text_embeddings.py` (the random-Gaussian placeholder = the non-graded baseline + the swap point);
  `sim/visual_cortex.py` (option-(3) grounded co-occurrence source).
- The data source: the agent's fact KB (`research/runners/brain_conversational_agent.py` `_assoc_graph`; the 320-concept
  flat-distinct KB, `2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md`); the merged one-bridge host
  (`research/runners/nav_conv_merged_bridge.py`).
- The scale-gap evidence (risk i): `research/findings/2026-05-07-Phase-2.3a-NEGATIVE-next-char-features.md` (toy BPTT cortex
  features did not transfer — "~4 orders too small"); the v14 orthogonal-codes breakthrough (CLAUDE.md, the strong+reproducible
  codes are orthogonal-by-construction — risk ii).
