# Research gate — EMERGENT STRUCTURE FROM EXPERIENCE (R-c): the shared category / is-a codes that inference rides on must be LEARNED from a co-occurrence stream, UNSUPERVISED, on the HTM spiking substrate (2026-07-02)

**Read-only deep-research gate (mandated before a NEW mechanism-class direction).** The inference triad is now DEMONSTRATED on the
emergent spiking HTM cortex — generalization (EMERGE-17/18/19), Collins-Quillian INHERITANCE single-level + multi-level with
cancellation (EMERGE-26/27), and TRANSITIVE relational inference (EMERGE-28), each 6-seed GO, emergent, one spiking brain, NO `sim/`
edit. **Every one of those findings closes with the SAME honest residual, verbatim:** the inference rides on shared / is-a /
superordinate codes that are **HOST-DESIGNED** — `SUPER = {"robin":"BIRD", "sparrow":"BIRD", ...}`, with the shared "BIRD" columns
`[24,25]` **hand-assigned** in the encoder. The master directive requires EMERGENCE: the structure must be **LEARNED FROM
EXPERIENCE**, not handed as `robin isa bird`. **That is R-c, and it is now the gating residual.**

**BOTTOM LINE (verdict, expanded in §4): R-c is surpassable and CHEAP — it is not a wall.** The genuinely-missing operation is
narrow and already half-built: **make the shared superordinate MICRO-COLUMN BLOCK EMERGE from a co-occurrence stream** so that
concepts appearing in similar contexts DEVELOP overlapping columns, WITHOUT a teacher labeling the category — then run EMERGE-26's
inheritance on those LEARNED (not hand-assigned) shared cells. The biology names the exact mechanism: the **HTM Spatial Pooler**
(Cui, Ahmad & Hawkins 2017) — competitive Hebbian learning + homeostatic boosting that maps inputs to SDRs and **provably preserves
the semantic similarity of the input space** (similar inputs → overlapping columns). It is the *sister* algorithm to the HTM
Temporal Memory already committed on the substrate (`fused_htm_permanence_update`), and the project **already learns co-occurrence
into overlapping codes on the bridge** (`_phaseB_onbridge_stream_cortex_derisk.py`, `corr(M,C) +0.885` in the raw result, generalizes held-out). The
one missing step is **discretizing that learned overlap into a shared column BLOCK** a class-property can attach to. **The single
cheapest de-risk — EMERGE-30 (§4)** streams a co-occurrence corpus, lets a competitive kWTA+boosting pooler DEVELOP a shared column
block for co-occurring members, and shows a held-out member INHERITS a class property taught only to the emergent shared cells —
EMERGE-26 inheritance on LEARNED superordinate codes. Reuse-by-import; NO `sim/` edit expected.

---

## 1. ISOLATE + QUANTIFY the true residual

**What is ALREADY solved (do not re-derive):**

| Capability | EMERGE | Status |
|---|---|---|
| Inheritance over is-a structure (Collins-Quillian, with cancellation) | 26/27 | GO — but on HAND-ASSIGNED superordinate blocks |
| Transitive relational inference (recombine overlapping premises, B>D) | 28 | GO — but on HAND-ASSIGNED disjoint item codes |
| Generalization across similar words (shared family micro-columns) | 17/18 | GO — but on HAND-ASSIGNED family blocks |
| Generalization on REAL learned PPMI similarity (cosine-clustered) | 19 | GO — codes LEARNED, but categories found by host cosine-clustering, and inference not attached to a shared *block* |
| On-bridge Hebbian co-occurrence cortex (population code) | `_phaseB` | GO — `corr(M,C) +0.885`, generalizes held-out 0.86, permuted-clean |

**The isolated residual — three sub-pieces, from smallest-open to hardest:**

- **R-c-1 — the SHARED superordinate BLOCK must EMERGE (the smallest genuinely-open piece; the recommended de-risk).** In EMERGE-26/27
  the members share a *hand-assigned* set of columns (`CONTENT["BIRD"] = [24,25]`; `SUPER = {"robin":"BIRD",...}`). R-c-1 replaces
  that with: stream observations in which robin / sparrow / canary appear in shared contexts → an **unsupervised competitive-coding
  step DEVELOPS a shared column block for them** (they come to activate overlapping cells because they co-occur with shared
  contexts), with NO teacher labeling "BIRD". Then attach a class property to the *emergent* shared cells and test a held-out
  member's never-taught inheritance. **This is the exact EMERGE-26 runner with ONE change: the superordinate block is LEARNED, not
  a `dict` literal.** Quantified gap: EMERGE-19 already shows LEARNED codes carry similarity that transfers an association, but (i)
  that similarity is *read out by host cosine-clustering*, not consolidated into a discrete shared **block** on the substrate, and
  (ii) EMERGE-19 tests *association transfer*, not *class-property INHERITANCE with cancellation* on the learned code. R-c-1 = close
  both: the shared block is a substrate structure (developed by a competitive pooler), and inheritance runs on it.

- **R-c-2 — the co-occurrence → overlap map must run ON the spiking substrate as competitive coding (not host cosine).** EMERGE-19's
  "top-Kc code dims as micro-columns" is a host post-hoc read of pre-computed PPMI codes. R-c-2 wants the overlap to be produced by
  a **spiking competitive-coding layer** (kWTA + homeostatic boosting = the HTM Spatial Pooler) so similar-context inputs *fire*
  overlapping columns online — the neocortical operation, on-bridge. The project's `_phaseB_onbridge_stream_cortex_derisk.py`
  already learns the co-occurrence into synapses; R-c-2 adds the competitive read that turns learned co-occurrence into a shared
  ACTIVE column set.

- **R-c-3 — a hierarchy of emergent blocks (multi-level taxonomy from experience) + acquiring it at scale (the hardest, deferred).**
  EMERGE-27 stacks hand-assigned ANIMAL ⊃ BIRD blocks. R-c-3 is: do the emergent superordinate at TWO granularities (a broad
  ANIMAL-level shared block AND a finer BIRD-level shared block emerge from coarse-vs-fine context sharing), and scale to a real
  vocabulary. This is the Saxe-McClelland-Ganguli "hierarchy emerges in stages from feature-prediction" prediction on the spiking
  substrate — a research direction, gated after R-c-1 proves a *single* emergent level supports inheritance.

**Quantification of "how much is genuinely open":** R-c-1 is ~one runner. The ONLY new machinery is a competitive-coding
(kWTA + boosting) step that consumes the on-bridge co-occurrence and emits a shared column block — and the project already has the
pieces (rate-kWTA cleanup `cortex_dg_ratekwta_cleanup_probe.py`; the on-bridge co-occurrence learner; the EMERGE-26 inheritance
runner). So the "must acquire structure from experience" wall is, on inspection, a **competitive-coding + block-consolidation**
question bolted onto the front of the already-validated inheritance machinery — the same *"the ONLY change is the encoding"* shape as
the EMERGE-17 generalization win, but now the encoding is DEVELOPED rather than hand-set.

---

## 2. How REAL biology / recent theory does this (catalog + Kandel + literature, cited) — and the load-bearing REFRAME

### 2a. The named mechanism: the HTM Spatial Pooler = competitive Hebbian + homeostatic boosting → SDRs that PRESERVE input similarity
The single most load-bearing citation: **Cui, Ahmad & Hawkins 2017, "The HTM Spatial Pooler — A Neocortical Algorithm for Online
Sparse Distributed Coding", *Front. Comput. Neurosci.* 11:111** (bioRxiv 085035). The spatial pooler is the *feedforward sister* of
the HTM Temporal Memory the substrate already runs: it converts inputs into fixed-sparsity SDRs by **competitive Hebbian learning on
proximal (feedforward) synapses + homeostatic excitability control ("boosting")**, and its central proven property is that it
**"preserves the semantic similarity of the input space"** — *similar inputs are mapped to SDRs that share active columns.* That is
EXACTLY the emergent-superordinate operation R-c needs: concepts that co-occur with shared contexts (hence have similar feedforward
input) DEVELOP overlapping columns — an emergent shared "BIRD" block, with no teacher. Boosting is the key homeostatic term: it
forces under-used columns into service so the sparse code stays balanced *and* it drives the "similar-input → shared-column" mapping
(without boosting, a few columns hog all inputs). ⇒ *Reframe #1:* the emergent superordinate is not a new faculty to invent — it is
the OUTPUT of running the SP's own algorithm (already the project's HTM lineage) on a co-occurrence stream. The substrate has the TM
half committed (`fused_htm_permanence_update`); R-c is the SP half (competitive kWTA + boosting), which the project can realize
reuse-by-import (rate-kWTA + a boost term) with likely NO `sim/` edit.

### 2b. Semantic structure = distributed overlapping codes that EMERGE from experience (Rogers-McClelland; Saxe-McClelland-Ganguli)
Category structure is not stored symbolically; **conceptual similarity is coded by representational overlap, and taxonomic structure
EMERGES from the statistics of experience.** Rogers & McClelland 2004, *Semantic Cognition* (MIT Press); **Lambon Ralph, Jefferies,
Patterson & Rogers 2017, "The neural and computational bases of semantic cognition", *Nat Rev Neurosci* 18:42** (the hub-and-spoke
ATL: a transmodal hub binds modality-specific spokes; similarity = shared hub representation). **Saxe, McClelland & Ganguli 2019, "A
mathematical theory of semantic development in deep neural networks", *PNAS* 116:11537** — PROVES that when a network learns to
predict an item's features, a taxonomic hierarchy (superordinate → basic → subordinate) emerges in its representations IN STAGES,
and items sharing a superordinate come to share representational components. ⇒ *the exact thing R-c wants is a theorem:*
inheritance-by-shared-representation is a mathematical consequence of learning to predict features from items — the substrate's HTM
next-state predictor + a competitive code IS a feature-prediction learner. **2024-2025 update:** ECoG in ventral ATL reveals a
"graded, multidimensional semantic space" expressing both broad and fine-grained structure simultaneously ("Representational
similarity learning reveals a graded multidimensional semantic space in the human anterior temporal cortex", bioRxiv 2022 → PMC
2024) — i.e. the brain's semantic code is exactly the overlapping-graded structure the SP produces, and it is *distributed* across
cortical hubs, not a single symbolic node ("A Distributed Network for Multimodal Experiential Representation of Concepts",
*J. Neurosci.* 42:7121). Catalog anchor: **G.13 Wernicke's area** (Kandel 6e Ch 55 pp 1384-1385) lists prerequisite *"semantic
memory store"* as `missing` — the emergent overlapping-code cortex is that store's biology.

### 2c. Distributional/co-occurrence learning IS a brain mechanism, but grounded in experience (the R-c corpus caveat)
Recent work: "word meaning is encoded primarily in terms of experiential information, and distributional (co-occurrence) models
align to empirical data to the extent they reflect experiential information" (the ATL/experiential-semantics literature above). ⇒
*Reframe #2:* pure text-co-occurrence is a legitimate *proxy* for the shared-context statistics, but the honest end-state grounds
the "shared context" in the brain's EXPERIENCE (perceptual features, shared frames). For R-c-1 the cheapest faithful version uses a
co-occurrence STREAM (members share contexts) — which is what `_phaseB_onbridge_stream_cortex` already does — and the perceptual /
feature-norm grounding (members share visual/sensory features → shared code, per the project's Gabor/V1 + cross-modal-unify wins) is
the scale-up, not the first de-risk.

### 2d. The DG/CLS pattern-separation-vs-completion axis — R-c needs the COMPLETION/generalization (neocortical) side
Catalog **D.12 (Pattern separation — DG sparsifies overlapping inputs; Kandel Ch 54 pp 1357-1360)** and **D.13 (Pattern completion —
CA3; pp 1360-1361)** frame the trade-off: DG *orthogonalizes* similar inputs (separation), CA3 *merges* partial cues (completion).
**R-c wants the OPPOSITE of DG separation:** similar-context concepts must come to SHARE cells (overlap), which is the **neocortical
slow-learning, overlapping-to-generalize** half of Complementary Learning Systems (**McClelland, McNaughton & O'Reilly 1995**),
NOT the hippocampal fast-sparse-separating half. This is a crucial design guard: the competitive pooler must be tuned to PRESERVE
overlap for similar inputs (the SP's proven property), not to separate them (DG). The catalog's own **D.12 note** ("too much
completion → confused episodes; too little → no generalization") is the tuning knob: R-c-1 lives at the generalization end. The
project's `cortex_dg_ratekwta_cleanup_probe.py` is the reusable kWTA competitive-code machinery; the tuning target is overlap-
preserving (SP-like), the opposite of the DG-separation probe's target.

### 2e. Replay consolidates emergent structure into inferable schema (the R-c-3 mechanism; deferred)
**Catalog N.14 (Hippocampal-neocortical dialogue / systems consolidation; Kandel Ch 52 p 1299, Ch 54 p 1366):** SWR-locked
reactivation transfers HC→neocortex, extracting generalizable structure (schema) over episodic detail. **2024-2025 literature:**
"Constructing future behavior in the hippocampal formation through composition and replay" (*Nat Neurosci* 2025) — replay COMPOSES
structural building blocks into generalizable state spaces; "An inhibitory plasticity mechanism for world structure inference by
hippocampal replay" (bioRxiv 2022→) — replay + inhibitory plasticity INFERS latent world structure; "Abstract representations emerge
in human hippocampal neurons during inference behavior" (PMC11338822, 2024) — abstract/disentangled codes emerge AFTER learning to
infer; Spens & Burgess 2023 (*Nat Hum Behav* 7:1965) — replay trains a generative model yielding relational inference + schema
generalization. ⇒ the biology-native engine that turns raw co-occurrence into durable, hierarchical, inferable structure is **replay
+ competitive consolidation**, and the project has the excitability/NREM replay used for generation (EMERGE-16/23). Replay is the
R-c-3 path (offline recombination + hierarchy formation), reusing machinery on the substrate — deferred until R-c-1 proves a single
emergent level works.

**Net of §2 — is the current framing testing the WRONG hypothesis? Sharpened, not wrong.** The prior gate already reframed
"semantics" as *emergent inference over overlapping codes* and proved it (EMERGE-26/27/28). R-c is the honest *last* reframe: the
overlapping codes themselves must be DEVELOPED, and the biology names the exact developer — the **HTM Spatial Pooler's competitive
Hebbian + homeostatic boosting** (Cui-Ahmad-Hawkins 2017), which is provably similarity-preserving and is the feedforward sibling of
the TM the substrate already runs, backed by the *theorem* that taxonomic inheritance emerges from feature-prediction over such codes
(Saxe-McClelland-Ganguli 2019). The residual is "develop the shared block, then run the validated inheritance on it" — a
competitive-coding + block-consolidation de-risk, NOT a new inference engine.

---

## 3. RANK cheap-first, biologically-grounded mechanisms (make the shared/is-a code EMERGE on THIS substrate)

Ordered by implementation cost. Each: mechanism · citation · reusable project machinery · the specific cheap-first experiment
(single-variable, anti-cheated, multi-seed 42/43/44) · what a GO shows · `sim/`-edit-or-not.

### (a) ★ CHEAPEST — EMERGENT SUPERORDINATE BLOCK via competitive kWTA + homeostatic boosting on a co-occurrence stream, then inheritance rides it (R-c-1)
- **Mechanism:** members of a latent category are streamed so they co-activate a set of SHARED CONTEXT columns (robin/sparrow/canary
  each co-occur with the same context tokens). A **competitive-coding layer (kWTA select + homeostatic boosting)** — the HTM Spatial
  Pooler operation — learns feedforward permanences from context→concept via the committed three-term kernel, so a concept's ACTIVE
  column set comes to include the columns its shared context drives. Because the members share contexts, their SP-codes come to SHARE
  columns → an **emergent superordinate block**, with NO teacher labeling the category. Then teach a class property by potentiating
  the emergent shared cells → the property token (`emergent-BIRD-cols → flies`), and test a HELD-OUT member (never taught the
  property): its SP-code contains the emergent shared cells → the class pathway primes `flies` → **inheritance on a LEARNED
  superordinate.**
- **Citation:** Cui, Ahmad & Hawkins 2017 *Front. Comput. Neurosci.* 11:111 (SP: competitive Hebbian + boosting → similarity-
  preserving SDRs); Saxe-McClelland-Ganguli 2019 *PNAS* (taxonomic inheritance emerges from feature-prediction over such codes);
  Collins & Quillian 1969 (inheritance); Lambon Ralph 2017 *Nat Rev Neurosci* (overlap = similarity). Catalog: D.12/D.13 (kWTA
  sparse coding, tuned to the overlap-preserving/generalization end, NOT DG-separation); G.13 (semantic store).
- **Reusable machinery:** the on-bridge co-occurrence learner `_phaseB_onbridge_stream_cortex_derisk.py` (streams windows,
  rate-Hebbian co-occurrence into synapses, population read); the rate-kWTA competitive code `cortex_dg_ratekwta_cleanup_probe.py`
  (the SP-style k-winner select — tuned overlap-preserving); the EMERGE-26 inheritance runner
  (`_emerge26_emergent_inheritance_derisk.py`) VERBATIM for the read-out (graded apical-drive argmax + cancellation + moat), with the
  ONLY change: `SUPER`/`CONTENT["BIRD"]` are no longer literals but the LEARNED shared column set from the pooler; `_emerge14`
  (`build_pool_bridge`/`apply_kernel_update`/`coincidence_predict`) + `sim.kernels.fused_htm_permanence_update`.
- **Cheap-first experiment:** a small synthetic stream: 2 latent categories (BIRD-context words vs FISH-context words), 3 members
  each, each member co-occurring with its category's shared context tokens (+ member-unique tokens). Run the competitive pooler over
  the stream → read each member's emergent active-column set → verify (by overlap, RSA vs the latent labels — label-free) that
  same-category members SHARE columns and cross-category do NOT. Then teach `emergent-shared-BIRD → flies` / `emergent-shared-FISH →
  swims` on the LEARNED shared cells; HOLD OUT one member per category from ever seeing the property; test the held-out member
  inherits. CPU numpy-backend, 6-seed.
- **Anti-cheats (all mandatory; reuse the EMERGE-17/26 control patterns):**
  1. **PERMUTED-CONTEXT / category-derangement of the STREAM** — shuffle which members share which context → NO shared block emerges →
     inheritance collapses to chance (isolates the LEARNED co-occurrence structure as the cause; the EMERGE-19 shuffled-code control
     analogue).
  2. **RSA / overlap provenance (label-free)** — the emergent shared columns must track the latent categories by co-occurrence
     statistics alone, no labels used in forming them (proves EMERGENCE, not injection).
  3. **dAP-LESION** — coincidence off → no priming → inheritance → 0.00 (the dendritic-plateau recurrence is load-bearing).
  4. **HELD-OUT member** never taught the property (built into the design; the answer cannot be a memorized member→property pathway).
  5. **MOAT** — a concept that co-occurs with NO shared context develops no shared block → drives no class pathway → ABSTAINS (the
     intrinsic no-confab moat holds through the EMERGENT structure; it cannot confabulate an inheritance it never developed).
  6. **NO-LEARNING control** — skip the pooler (random/untrained codes) → no shared block → inheritance at chance (isolates the
     competitive learning as the cause, not a wiring prior).
  7. **6-seed** (42/43/44/100/101/102), no teacher labeling the category.
- **GO shows:** the FIRST emergent-STRUCTURE-from-experience on the substrate — a concept INHERITS an untold class property through a
  superordinate block that was **LEARNED from co-occurrence**, not hand-assigned. It converts the last honest residual of the
  inference triad ("the is-a codes are host-designed") into a demonstrated emergent capability: the substrate ACQUIRES the relational
  structure AND infers over it, on one spiking brain. `sim/` edit: **expected NONE** (rate-kWTA + boosting compose from existing
  cfg + host orchestration exactly as `_phaseB`/`_emerge14`/`ratekwta` already do; if a boosting term needs a tiny additive flag, it
  is default-off/byte-identical — resolve at build).

### (b) EMERGENT structure on the REAL PPMI stream codes (scale R-c-1 to the project's learned vocabulary)
- **Mechanism:** replace the synthetic 2-category stream with the project's REAL stream-cortex PPMI codes
  (`_phaseB_stream_codes_320_seed42.npy`, 320×300, LEARNED similarity). Run the SP-style competitive pooler over the real codes to
  DEVELOP shared blocks for the genuinely-tight clusters (EMERGE-19 identified them by cosine); attach a class property to an
  emergent block; test held-out inheritance on the real codes.
- **Citation:** as (a) + EMERGE-19 (the real codes carry LEARNED similarity that transfers; here consolidated into a shared BLOCK +
  inheritance + cancellation, not just association transfer).
- **Reusable machinery:** the real codes file; (a)'s pooler + read-out; EMERGE-19's cluster-finding.
- **Cheap-first experiment:** on the tight real clusters, develop shared blocks, teach a class property to the emergent block, test
  held-out inheritance; SHUFFLED-CODE control collapses (as EMERGE-19).
- **Anti-cheats:** SHUFFLED-CODE (destroys real similarity → no block → no inheritance); graded-by-cosine (tight clusters inherit,
  loose don't — the correct SDR-overlap≈similarity behaviour, EMERGE-19); dAP-lesion; held-out; 6-seed.
- **GO shows:** emergent inheritance on the project's REAL learned vocabulary similarity — the mechanism is real, not a synthetic
  toy. `sim/` edit: NONE. Do AFTER (a).

### (c) TWO-LEVEL emergent taxonomy (broad + fine shared blocks emerge from coarse-vs-fine context sharing) (R-c-3, deepen)
- **Mechanism:** stream so members share a BROAD context (all animals share "eats/moves") AND a FINER context (birds share "nest/
  wings"). The competitive pooler develops a broad shared block (ANIMAL) AND a finer nested shared block (BIRD) — the
  Saxe-McClelland-Ganguli staged hierarchy. Attach `ANIMAL-emergent → breathes`, `BIRD-emergent → flies`; test a held-out member
  inherits from BOTH emergent levels (EMERGE-27's multi-level, but on LEARNED blocks) + per-dimension cancellation survives.
- **Citation:** Saxe-McClelland-Ganguli 2019 (hierarchy emerges in stages); Collins-Quillian multi-level; EMERGE-27.
- **Reusable machinery:** (a)'s pooler + EMERGE-27's multi-level read-out.
- **Cheap-first experiment:** 2-level synthetic stream; verify broad + fine emergent blocks (RSA at two granularities); test 2-hop-up
  + 1-hop-up inheritance on the emergent hierarchy; dimension-isolation cancellation.
- **Anti-cheats:** per-level derangement; RSA at both granularities; held-out; lesion; 6-seed; flat-vs-nested control.
- **`sim/` edit:** NONE. Do after (a)/(b).

### (d) PERCEPTUAL / feature-grounded emergent structure (ground the "shared context" in EXPERIENCE, not text) (R-c, the honest end-state)
- **Mechanism:** the shared context is not co-occurring TOKENS but shared PERCEPTUAL FEATURES — members that look/behave alike
  (shared Gabor/V1 features) develop a shared block via the same competitive pooler on the perception stream. This is the
  master-directive-faithful version (structure from the brain's experience), reusing the project's validated visual-similarity +
  cross-modal-unification wins (Gabor/V1 `build_v1_simple_weights`; the generalization-frontier cross-modal-unify + on-substrate
  convergence GO's).
- **Citation:** Lambon Ralph 2017 (hub-and-spoke: modality spokes → transmodal hub); the project's Option-B visual-similarity + cross-
  modal-unify GO's; Patterson-Lambon-Ralph convergence-zone biology.
- **Reusable machinery:** the Gabor/V1 front end; the cross-modal-unify de-risks; (a)'s pooler.
- **Cheap-first experiment:** render shape-structured percepts → competitive pooler develops shared blocks for visually-similar
  objects → attach a class property to the emergent perceptual block → held-out inheritance from a NOVEL-perceived member.
- **Anti-cheats:** category-derangement of the perceptual features; RSA pixel-provenance (label-free, per Option-B r=0.99); lesion;
  held-out; 6-seed.
- **`sim/` edit:** likely NONE (reuse). Higher cost (perception pipeline); do after the text-stream de-risks prove the pooler works.

### (e) REPLAY-DRIVEN consolidation of emergent structure (R-c-3, the CLS/schema mechanism; deepest, deferred)
- **Mechanism:** use SWR/excitability replay to CONSOLIDATE + hierarchize the emergent blocks offline (replay interleaves category
  members so the pooler sharpens the shared block and forms nested levels during "sleep"), turning a stream of episodes into durable,
  hierarchical, inferable neocortical schema.
- **Citation:** Kandel N.14 (systems consolidation); Spens-Burgess 2023 *Nat Hum Behav*; *Nat Neurosci* 2025 (composition + replay);
  "An inhibitory plasticity mechanism for world structure inference by hippocampal replay" (bioRxiv 2022→); Tse 2007 (schema);
  Bouhadjar 2023 *PLoS Comput Biol* 19(5):e1010989 (coherent-noise probabilistic replay — the branch-recombining replay already in
  the ported paper).
- **Reusable machinery:** the Bouhadjar excitability-replay roll-out (EMERGE-16/23) + NREM scaffolding + (a)'s pooler.
- **Cheap-first experiment:** stream only sparse/unbalanced observations → the block is weak; run a replay phase → the emergent block
  sharpens and inheritance improves vs a no-replay control (replay CAUSALLY builds the structure, mirroring the generative-loop
  self-replay result).
- **Anti-cheats:** no-replay control (structure weaker); permuted-replay (scrambled → no gain); lesion; 6-seed.
- **`sim/` edit:** possibly a tiny additive excitability flag if not exposed; otherwise NONE. Deferred — highest value after (a)-(d).

---

## 4. VERDICT — surpassable, and how cheaply. The single recommended next de-risk: EMERGE-30

**R-c is surpassable and CHEAP — it is not a wall.** The biology names the exact developer of the emergent shared code — the **HTM
Spatial Pooler's competitive Hebbian learning + homeostatic boosting** (Cui, Ahmad & Hawkins 2017), which is *provably* similarity-
preserving (similar inputs → shared columns) and is the feedforward SIBLING of the HTM Temporal Memory the substrate already runs —
and the theory guarantees the payoff — **taxonomic inheritance emerges from feature-prediction over overlapping codes** (Saxe-
McClelland-Ganguli 2019 *PNAS*; Rogers-McClelland; Lambon Ralph 2017). The substrate already learns co-occurrence into overlapping
population codes on the bridge (`_phaseB_onbridge_stream_cortex`, `corr(M,C) +0.885`, generalizes 0.86) and already infers over shared
blocks (EMERGE-26/27/28). The residual is the ONE missing step between them: **consolidate the LEARNED co-occurrence overlap into a
discrete shared column BLOCK** (via a competitive kWTA + boosting read) **that a class property can attach to, then run the validated
inheritance on it.** That is a competitive-coding + block-consolidation de-risk of the exact same *"the only change is the encoding —
now DEVELOPED not hand-set"* shape as the EMERGE-17 generalization win. The genuinely-irreducible part — a rich MULTI-LEVEL hierarchy
acquired from raw PERCEPTION at scale (R-c-3/d/e) — is a real research direction, but it is DOWNSTREAM of first showing a SINGLE
emergent superordinate supports inheritance, and it too reuses existing machinery (the PPMI stream cortex + Gabor/V1 + replay), not a
new mechanism class.

**THE SINGLE CHEAPEST NEXT DE-RISK — EMERGE-30 (path (a), do this first):**

> **Runner:** new `research/runners/_emerge30_emergent_superordinate_derisk.py` — reuse-by-import: the co-occurrence stream +
> population-Hebbian learning from `_phaseB_onbridge_stream_cortex_derisk.py`; the rate-kWTA competitive select from
> `cortex_dg_ratekwta_cleanup_probe.py` (tuned OVERLAP-PRESERVING, the SP/generalization end, NOT DG-separation); the inheritance
> read-out (graded apical-drive argmax + cancellation + moat) + `build_pool_bridge`/`apply_kernel_update`/`coincidence_predict` from
> `_emerge26`/`_emerge14`; `sim.kernels.fused_htm_permanence_update`. **NO `sim/` edit expected** — the ONLY new thing is that the
> superordinate block is DEVELOPED by the competitive pooler from the stream, not the `SUPER`/`CONTENT["BIRD"]` dict literals of
> EMERGE-26.
>
> **The stream (the "experience"):** 2 latent categories, BIRD-context {robin, sparrow, canary} and FISH-context {trout, salmon,
> pike}. Each member co-occurs, window-by-window, with its category's SHARED context tokens (e.g. BIRD-context: nest, wings, sky) +
> its own member-unique tokens. NO label "BIRD"/"FISH" is ever presented — only who co-occurs with what.
>
> **How the shared code EMERGES:** run the competitive pooler (kWTA select of the k most-driven columns + homeostatic boosting of
> under-used columns = the HTM Spatial Pooler) over the stream, learning feedforward context→concept permanences with the committed
> three-term kernel. Because same-category members share context tokens, their SELECTED (winning) columns come to OVERLAP → the
> emergent shared block. VERIFY (label-free): same-category members share columns (overlap / RSA vs the latent labels) and cross-
> category do not.
>
> **The held-out inheritance test (on the EMERGENT block):** teach ONLY class-level facts by potentiating the EMERGENT shared cells →
> the property (`emergent-BIRD-cols → flies`, `emergent-FISH-cols → swims`) — the shared cells are READ FROM THE POOLER, not a dict.
> HOLD OUT one member per category (e.g. canary, pike) from ever seeing the property. Present the held-out member; its pooler-code
> contains the emergent shared cells → the class pathway primes the property → GO = the never-taught property is inherited
> (`canary → flies`, `pike → swims`), 6/6 seeds. Add the EMERGE-26 CANCELLATION (`penguin → walks` beats inherited flies, via a
> stronger member-specific pathway) and MOAT (a no-shared-context concept ABSTAINS) on the emergent structure.
>
> **Anti-cheats (all mandatory, all reuse EMERGE-17/19/26 control patterns):**
> 1. **PERMUTED-CONTEXT** — shuffle which members share which context tokens → NO shared block emerges → inheritance collapses to
>    chance (isolates the LEARNED co-occurrence as the cause; the EMERGE-19 shuffled-code analogue).
> 2. **NO-LEARNING control** — random/untrained codes (skip the pooler) → no shared block → inheritance at chance (isolates the
>    competitive learning, not a wiring prior).
> 3. **dAP-LESION** — coincidence off → inheritance → 0.00 (dendritic-plateau recurrence load-bearing).
> 4. **RSA / overlap provenance (label-free)** — the emergent shared columns track the latent categories from co-occurrence alone, no
>    labels used to form them (proves EMERGENCE).
> 5. **HELD-OUT members** never taught the property (built in).
> 6. **MOAT** — a no-shared-context concept develops no block → ABSTAINS (no confabulated inheritance).
> 7. **6-seed** (42/43/44/100/101/102), no category teacher.
>
> **What GO shows:** the FIRST emergent-STRUCTURE-from-experience on the substrate — a concept INHERITS an untold class property
> through a superordinate block **LEARNED from co-occurrence statistics**, unsupervised, with cancellation and the no-confab moat
> intact, on one real spiking brain, NO `sim/` edit. It closes the last standing honest residual of the entire inference arc: the
> substrate now ACQUIRES the relational structure from experience AND infers over it — the master-directive core.

**Chain after EMERGE-30 GO (all reuse-by-import):** (b) emergent structure on the REAL PPMI stream codes (scale to the learned
vocabulary) → (c) two-level emergent taxonomy (broad + fine blocks from coarse-vs-fine context) → (d) PERCEPTUAL feature-grounded
emergent structure (ground the shared context in the brain's EXPERIENCE via Gabor/V1 + cross-modal-unify — the honest end-state) →
(e) replay-driven consolidation/hierarchization (the CLS/schema offline engine). Each rung is one controlled runner; the deep,
genuinely-open frontier (R-c-3 at scale — a rich multi-level taxonomy discovered from raw perception, not curated streams) is named
honestly and is the experience-driven-learning direction of the master directive, gated after the single emergent level is proven.

**Honest scope / what would make this a real wall (and why it isn't yet):** EMERGE-30's stream is a curated SYNTHETIC co-occurrence
corpus (members deliberately share context tokens), so a GO proves "the substrate DEVELOPS a shared superordinate from co-occurrence
and inherits over it," NOT yet "it discovers rich hierarchical structure from raw open-world perception." The latter (R-c-3/d/e:
multi-level, perceptually-grounded, replay-consolidated, at real-vocabulary scale) is the genuinely-irreducible residual, and it is a
research direction — but per the SURPASS discipline it is surpassable-and-how (paths b/c/d/e all reuse existing machinery: the PPMI
stream cortex, Gabor/V1, cross-modal-unify, excitability replay), not a wall. The cheapest first step (EMERGE-30) is unambiguous,
airtight-anti-cheatable, and reuse-by-import with NO `sim/` edit expected. A secondary honest note: the competitive pooler must be
tuned to the OVERLAP-PRESERVING (generalization) regime, NOT the DG pattern-SEPARATION regime (catalog D.12/D.13) — over-
sparsification would orthogonalize the members and kill the shared block; this is a known, single-knob tuning target (the SP's proven
similarity-preservation is the design goal), not a mechanism risk.

---

## Artifacts / key citations
- **Substrate + reusable machinery:** `_phaseB_onbridge_stream_cortex_derisk.py` (on-bridge co-occurrence learner, `corr(M,C) +0.885`,
  generalizes 0.86; the emergent-overlap engine); `_phaseB_stream_codes_320_seed42.npy` (320×300 real learned PPMI codes);
  `cortex_dg_ratekwta_cleanup_probe.py` (rate-kWTA competitive code — reuse tuned overlap-preserving); `_emerge26_emergent_inheritance_derisk.py`
  (inheritance read-out + cancellation + moat — the ONLY change: LEARNED not hand-assigned superordinate); `_emerge27_multilevel_taxonomy_derisk.py`
  (multi-level read-out, for path c); `_emerge28_transitive_inference_derisk.py`; `_emerge17_generalizing_word_codes_derisk.py` +
  `_emerge19_real_ppmi_generalization_derisk.py` (overlap-generalization + shuffled-code control); `_emerge14_stageC_onbridge_learning_derisk.py`
  (`build_pool_bridge`/`apply_kernel_update`/`coincidence_predict`); `sim/kernels.py` (`fused_htm_permanence_update` — the committed
  three-term kernel, the TM sibling of the SP); the Gabor/V1 front end (`sim.visual_cortex.build_v1_simple_weights`) + the cross-modal-
  unify de-risks (for path d). Prior gate: `2026-07-02-open-world-semantics-knowledge-acquisition-research-gate.md`; the honest-residual
  source in every inference finding: `2026-07-02-emerge26-emergent-inheritance-GO.md`, `-emerge27-`, `-emerge28-` (all flag R-c verbatim).
- **Catalog (Kandel 6e / O'Keefe-Nadel):** **D.12 Pattern separation — DG sparsifies (Ch 54 pp 1357-1360)** + **D.13 Pattern completion
  — CA3 (pp 1360-1361)** (the separation-vs-completion axis; R-c wants the completion/generalization end — cluster J sparse-coding via
  inhibition); **D.02 Relational binding / transitive inference — Eichenbaum-Cohen (Ch 52 pp 1301-1302)**; **N.14 hippocampal-
  neocortical dialogue / systems consolidation (Ch 52 p 1299, Ch 54 p 1366)** (replay → schema, path e); **G.13 Wernicke's area —
  semantic store prerequisite (Ch 55 pp 1384-1385)**; cluster L (development/self-organization; topographic Hebbian refinement).
- **Literature:** **Cui, Ahmad & Hawkins 2017, "The HTM Spatial Pooler — A Neocortical Algorithm for Online Sparse Distributed
  Coding", *Front. Comput. Neurosci.* 11:111** (bioRxiv 085035 — competitive Hebbian + homeostatic boosting → SDRs that PRESERVE input
  similarity; THE named emergent-shared-code mechanism, the SP sibling of the committed TM); **Saxe, McClelland & Ganguli 2019, "A
  mathematical theory of semantic development in deep neural networks", *PNAS* 116:11537** (taxonomic inheritance EMERGES from feature-
  prediction over overlapping codes — the theorem); Rogers & McClelland 2004, *Semantic Cognition* (MIT Press); **Lambon Ralph,
  Jefferies, Patterson & Rogers 2017, "The neural and computational bases of semantic cognition", *Nat Rev Neurosci* 18:42** (hub-and-
  spoke; overlap = similarity); "Representational similarity learning reveals a graded multidimensional semantic space in the human
  anterior temporal cortex" (bioRxiv 2022 → PMC 2024 — graded overlapping semantic code in ATL); "A Distributed Network for
  Multimodal Experiential Representation of Concepts", *J. Neurosci.* 42:7121 (distributed, experiential); Collins & Quillian 1969
  (*J Verbal Learning Verbal Behav* 8:240 — inheritance); McClelland, McNaughton & O'Reilly 1995 (CLS — the neocortical overlapping-
  to-generalize half); Spens & Burgess 2023, "A generative model of memory construction and consolidation", *Nat Hum Behav* 7:1965
  (replay → relational inference + schema); "Constructing future behavior in the hippocampal formation through composition and replay",
  *Nat Neurosci* 2025; "An inhibitory plasticity mechanism for world structure inference by hippocampal replay" (bioRxiv 2022);
  "Abstract representations emerge in human hippocampal neurons during inference behavior" (PMC11338822, 2024); Tse et al. 2007
  (schemas); Bouhadjar et al. 2022 *PLoS Comput Biol* 18(6):e1010233 (the ported TM substrate) + 2023 19(5):e1010989 (coherent-noise
  probabilistic replay). "Semantic representations emerge in biologically inspired ensembles of cross-supervising neural networks"
  (arXiv 2510.14486, 2025) + "Modelling concrete and abstract concepts using brain-constrained deep neural networks" (PMC9674741 —
  overlapping cell-assemblies with shared neurons implementing semantic feature sharing) corroborate the emergent-overlap thesis.
