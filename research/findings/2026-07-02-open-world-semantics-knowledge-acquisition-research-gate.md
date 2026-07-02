# Research gate — OPEN-WORLD SEMANTICS / knowledge-acquisition-and-inference on the emergent HTM spiking sequence cortex (2026-07-02)

**Read-only deep-research gate (mandated before a NEW mechanism-class direction).** Context: EMERGE-15..24 completed the
*toward-language* chain on ONE real spiking `SimulationBridge` (point-neuron Izhikevich + a two-compartment dendritic-plateau
"dAP" apical compartment + the committed HTM three-term permanence kernel `fused_htm_permanence_update`, the spiking port of
Bouhadjar-Diesmann-Tetzlaff 2022): the emergent cortex now PREDICTS high-order context-specific sequences, PRODUCES sentences,
GENERALIZES across similar words (shared "family" micro-columns → overlapping SDRs), GROUNDS with an intrinsic no-confab MOAT,
has a POS-frame construction GRAMMAR (systematic recombination), and GROWS (learns a new told fact live, retains the old, keeps
the moat) — each 6-seed GO, emergent, unsupervised, NO `sim/` edit. Every prior gate (EMERGE-20/22/24) names the SAME remaining
wall: **open-world SEMANTICS** — a brain that learns MEANING and WORLD-KNOWLEDGE from EXPERIENCE and can INFER BEYOND told
facts, rather than only reproducing/generalizing/growing told SVO sequences.

**BOTTOM LINE (verdict, expanded in §4):** the residual is **NOT a wall, and most of it is already latent in the substrate.** The
project has been reading "semantics" as a hard, separate faculty; the biology + the recent computational literature converge on a
sharper, cheaper reframe: **the inference the residual demands (Collins-Quillian inheritance; transitive inference) EMERGES from
exactly the machinery the substrate already has — overlapping/shared codes + high-order next-state prediction — with NO explicit
"inference engine."** EMERGE-17's shared-family-micro-column generalization is *structurally the same operation* as is-a
inheritance ("a robin shares the BIRD cells → inherits CAN-FLY"); recent theory (Kay 2024; the PNAS relational-generalization
theory 2024; Saxe-McClelland-Ganguli 2019) proves transitive/relational inference falls out of overlapping-code encoding + a rank
read-out *without* transitive structure being built in. The genuinely-new, genuinely-small residual is: **(R-a)** does INHERITANCE
emerge on THIS spiking substrate from shared superordinate (is-a) micro-columns + the coincidence-prediction pathway (untold facts
inferred), and **(R-b)** does the same substrate support hippocampal-style transitive inference by RECOMBINING overlapping learned
pairs. Both are cheap, reuse-by-import de-risks on the exact EMERGE-17/20/24 machinery, NO `sim/` edit. §4 names EMERGE-26.

---

## 1. ISOLATE + QUANTIFY the true residual

The `toward-language` chain (verified from the EMERGE-15..24 findings + the runners they cite) covers, all emergent + on-bridge:

| Capability | EMERGE | Status |
|---|---|---|
| High-order context-specific next-symbol/word PREDICTION | 14/15 | GO — the HTM TM core |
| Autoregressive PRODUCTION (excitability roll-out) | 16/23 | GO |
| GENERALIZATION across similar words (shared family micro-columns → overlapping SDRs) | 17/18/19 | GO |
| Intrinsic no-confab MOAT (disjoint code → no coincidence → abstain) | 20 | GO |
| POS-frame construction GRAMMAR / systematic recombination | 22 | GO |
| Full grammatical grounded sentence production | 23 | GO |
| GROWTH — learn a new told fact live, retain old, keep moat | 24 | GO |

**What is therefore NOT the residual (already solved — do not re-derive):** surface form/grammar (EMERGE-22/23); storing a TOLD
SVO fact (EMERGE-14/20); generalizing an association to a *lexically-similar* word (EMERGE-17, dog→wolf via family cells);
growing the knowledge set from told facts (EMERGE-24). The prior VSA/composer substrate ALSO had a transitive-isa CHASE
(`_fluidconv_phase15`: iterate the learned `isa` pointer hop-by-hop over *stored* triples — dog→mammal→vertebrata→chordata), but
that is (i) on the *idealized* FHRR composer, not the emergent HTM cortex, and (ii) a **retrieval walk over explicitly-told isa
edges**, not *inference of an UNTOLD fact*.

**The isolated residual — three nested pieces, from smallest-open to hardest:**

- **R-a — INFERENCE BEYOND TOLD FACTS via inheritance (the smallest genuinely-open piece).** Collins & Quillian (1969): a
  category hierarchy is stored *economically* — properties are attached at the most general level, and specific concepts INHERIT
  them without being told. "A robin is a bird; a bird can fly" ⟹ "a robin can fly" — **never told directly.** The project can
  today store + recall + chase told isa edges, and can generalize an association to a lexically-similar word, but it has **not**
  shown the cortex INFER an unstated property from a superordinate. Quantified gap: on the emergent HTM cortex, teach `bird → flies`
  (a property attached to the class) and `robin → isa → bird` (membership), and ask `robin → ?` for the *flies* property that was
  **never** taught for robin. Does it emerge? This is one controlled runner away.

- **R-b — RELATIONAL / TRANSITIVE inference by RECOMBINING overlapping learned pairs (Eichenbaum-Cohen; catalog D.02).** Given
  premise pairs A>B and B>C (or A→B, B→C) that share element B, infer the untrained A?C relation. This is the hippocampal
  relational-memory hallmark and the *general* form of R-a (inheritance is a special case where the shared element is a
  superordinate). Quantified gap: none of the EMERGE runs test cross-pair recombination where the inferred pair was never
  co-presented; EMERGE-17 tests lexical-family transfer, which is a DIFFERENT axis (similar surface code, same association) from
  relational chaining (dissimilar items linked through a shared middle).

- **R-c — ACQUIRING the relational structure from EXPERIENCE, not from told SVO triples (the hardest, deferred).** Learn the is-a /
  co-occurrence / feature-norm structure from a *stream of observations* (statistical/latent learning, perception), so the
  categories the cortex infers over were themselves *discovered*, not handed as `robin isa bird` triples. The stream-cortex PPMI
  codes (EMERGE-19) already discover *lexical* similarity from co-occurrence; R-c is the relational/hierarchical extension.

**Quantification of "how much is genuinely open":** R-a is ~one runner (reuse EMERGE-17/20 machinery verbatim; the ONLY new thing
is the CODE DESIGN — give `robin` a code that *shares the superordinate micro-columns of* `bird`, then attach the property to
those shared cells and test the never-taught inheritance). R-b is a second runner (overlapping-pair corpus + a read-out of the
recombined relation). R-c is a research direction (statistical structure learning), deferred until R-a/R-b prove the substrate
*can* infer at all. **So the "hard wall" is, on inspection, mostly a CODE-DESIGN + READ-OUT question on already-validated
machinery — the same shape as the EMERGE-17 "the ONLY change is the word→code encoding" result.**

---

## 2. How REAL biology does this (catalog + Kandel + literature, cited) — and the load-bearing REFRAME

### 2a. Semantic memory = a distributed hub-and-spoke, with taxonomic structure emerging from statistics (Rogers-McClelland)
Semantic knowledge is not a symbolic database; it is a distributed representation in which **conceptual similarity is coded by
representational overlap**, and **category structure (is-a hierarchies, feature inheritance) EMERGES from the statistics of
experience** rather than being explicitly stored. This is the Rogers-McClelland PDP theory of semantic cognition (Rogers &
McClelland 2004, *Semantic Cognition*, MIT Press; Lambon Ralph, Jefferies, Patterson & Rogers 2017, "The neural and computational
bases of semantic cognition", *Nat Rev Neurosci* 18:42-55 — the **hub-and-spoke** model: a transmodal ATL hub binds modality-specific
"spokes"). Saxe, McClelland & Ganguli 2019 (*PNAS* 116:11537, "A mathematical theory of semantic development in deep neural
networks") **proves** that when a network learns to predict an item's features, a *taxonomic hierarchy* (superordinate → basic →
subordinate) emerges in its representations in stages, and that items sharing a superordinate come to share representational
components — i.e. **inheritance-by-shared-representation is a mathematical consequence of learning to predict features from items.**
⇒ *Reframe #1:* the project already has the mechanism the theory names — EMERGE-17's shared "family" micro-columns ARE the
"items-sharing-a-superordinate-share-representation" structure; attaching a property to the SHARED cells and reading it back for a
member IS Collins-Quillian inheritance. The substrate is not missing a semantics engine; it is one controlled code-design away from
demonstrating the emergent-inheritance the theory predicts.

Catalog anchors: **G.13 Wernicke's area** (Kandel 6e Ch 55 pp 1384-1385) — "selects words matching intended meaning", explicit
prerequisite *"semantic memory store"* (currently `missing`). The hub-and-spoke ATL is the biology of that store; the project's
generalizing PPMI stream-cortex + shared-family codes are the closest existing analogue.

### 2b. Collins-Quillian inheritance = economical storage + retrieval over a class hierarchy (the R-a biology)
Collins & Quillian 1969 (the classic sentence-verification RT result: "a canary can fly" is verified FASTER than "a canary has
skin" because *fly* is stored at BIRD and *has-skin* at ANIMAL — retrieval traverses the is-a hierarchy). The property is stored
ONCE at the general node and INHERITED by traversal. **On a shared-code substrate, "traversal" is not a graph walk — it is the
member's code activating the superordinate's cells, which carry the property association.** This is exactly the EMERGE-17
mechanism run on an is-a (rather than a lexical-family) sharing relation. Catalog: this is the functional content D.02/J.34 gesture
at; Kandel 6e Ch 52 pp 1306-1308 (schemas/gist — generalizable structure prioritized over episodic detail).

### 2c. Transitive/relational inference = recombining overlapping experiences (Eichenbaum-Cohen; the R-b biology)
**Catalog D.02 (Relational binding / "memory space", Eichenbaum-Cohen; Kandel 6e Ch 52 pp 1301-1302):** the hippocampus "networks
[episodes] via overlapping events allowing flexible inference (e.g., transitive)" and supports "inference on overlapping
experiences (transitive inference)." **Catalog D.21 (Cognitive-map theory, O'Keefe-Nadel 1978; Kandel Ch 54):** "novel inferences
(shortcut taking, transitive choices, latent learning) drop out of the framework's geometry, not out of stored sensorimotor
associations" — the map's *structure* supports inference the individual associations do not. Sim-status for both: **missing — "no
relational binding primitive."** But the recent computational literature shows the primitive need not be a bespoke relational store:

- **Kay, Biderman, ... Miller 2024, "Emergent neural dynamics and geometry for generalization in a transitive inference task",
  *PLOS Comput Biol* (PMC11125559):** recurrent networks trained ONLY on adjacent premise pairs generalize transitively *"despite
  lacking overt transitive structure prior to training"* — TI **emerges** from encoding overlapping associations.
- **"A mathematical theory of relational generalization in transitive inference", *PNAS* 2024 (10.1073/pnas.2314511121):** a
  read-out from **item-wise** (overlapping) representations computes a per-item RANK that *transfers* to overlapping pairs, whereas
  a **conjunctive** (pair-memorizing) read-out does NOT transfer. ⇒ *Reframe #2:* transitive inference is a READ-OUT geometry over
  overlapping item codes, not a separate inference module — and the substrate already produces overlapping item codes (shared
  micro-columns) and a next-state read-out (coincidence prediction). The question is whether the HTM read-out is the transferring
  (item-wise) kind or the memorizing (conjunctive) kind — an *empirical* question one runner answers.

### 2d. The cognitive-map / TEM / successor-representation frame — structure factored from content, replay builds inference
- **Whittington et al. 2020, "The Tolman-Eichenbaum Machine", *Cell* 183:1249:** medial-EC cells form a **basis for structural
  knowledge**, hippocampal cells **conjoin** that structure with sensory content; this factorization is *sufficient* to generalize
  relational knowledge (the same "structure" transfers across environments/domains — space AND non-space). ⇒ the biology of
  generalizable relational inference is **factorize (structure × content) then conjoin** — and the project's three-block word code
  (POS-CLASS / content / family, EMERGE-22) is *already a factorization*: the CLASS block is a structural basis shared across
  members, the content block is the specific item. Adding an **is-a / relation block** shared by class members is the TEM move on
  this substrate.
- **Stachenfeld, Botvinick & Gershman 2017, "The hippocampus as a predictive map", *Nat Neurosci* 20:1643:** place cells encode the
  **successor representation** — expected future state occupancy — so the map *predicts* and thereby *generalizes* to novel
  trajectories. A **next-state predictor over a relational graph IS an implicit inference engine** (the SR of a graph encodes
  multi-hop reachability). ⇒ *Reframe #3:* the HTM TM *is* a next-state predictor; run it over a relational (is-a / associative)
  graph and multi-hop inference (A→B→C ⟹ A reaches C) is latent in its predictions, exactly as the SR makes multi-hop reachability
  latent in a one-step predictive map.
- **Replay builds compositional inference (2023-2025):** *"Constructing future behavior in the hippocampal formation through
  composition and replay"* (*Nat Neurosci* 2025); *"A generative model of memory construction and consolidation"* (Spens & Burgess,
  *Nat Hum Behav* 2023, 7:1965 — replay trains a generative model that yields *relational inference and schema-based* generalization);
  *"Human hippocampal ripples coordinate planning sequences and compositional representations in neocortex"* (*Nat Neurosci* 2026).
  ⇒ the biological mechanism that CONSOLIDATES and COMPOSES relational primitives into inferable structure is **SWR replay** — and
  the project has replay infrastructure (NREM scaffolding) + the Bouhadjar excitability-replay roll-out already used for generation
  (EMERGE-16/23). Replay is thus the biology-native path to R-c (offline recombination), reusing machinery already on the substrate.

### 2e. Systems consolidation / schema (the R-c biology — how experience becomes durable inferable structure)
**Catalog N.14 (Hippocampal-neocortical dialogue / systems consolidation; Kandel Ch 52 p 1299, Ch 54 p 1366):** repeated
SWR-locked reactivation gradually transfers memory HC→neocortex, extracting **generalizable structure** (schema) over episodic
detail. Tse et al. 2007 (schemas accelerate assimilation of consistent new facts) + McClelland-McNaughton-O'Reilly 1995 (CLS:
hippocampus fast/sparse, neocortex slow/overlapping-to-generalize). ⇒ the *generalizing* half of CLS is exactly what the emergent
overlapping-code cortex is; the fast/sparse half + replay is the acquisition-from-experience engine for R-c. The project's EMERGE-24
online growth is already the "assimilate a consistent new fact into an existing schema" move at the single-fact level.

**Net of §2 — is the current framing testing the WRONG hypothesis? YES, in a productive way.** "Open-world semantics is a hard
separate faculty" over-scopes it. The biology + theory say: *relational/semantic inference EMERGES from (overlapping/shared codes)
× (a next-state predictor) × (offline replay-recombination)* — and the substrate already has all three ingredients (EMERGE-17
shared codes; HTM TM predictor; excitability/NREM replay). The residual is not "build a semantics engine"; it is "give the codes a
shared *superordinate/relational* block and show inference emerges" (R-a/R-b), then "acquire that shared structure from experience"
(R-c). That is the same cheap **code-design + read-out** shape as the EMERGE-17 generalization win.

---

## 3. RANK cheap-first, biologically-grounded mechanisms (past the residual on THIS substrate)

Ordered by implementation cost. Each: mechanism · citation · reusable project machinery · the specific cheap-first experiment
(single-variable, anti-cheated, multi-seed 42/43/44) · what a GO shows · `sim/`-edit-or-not.

### (a) ★ CHEAPEST — EMERGENT INHERITANCE via a shared is-a (superordinate) micro-column block (R-a)
- **Mechanism:** give each concept a THREE-block SDR exactly as EMERGE-22 already does, but repurpose the shared block as an
  **is-a / superordinate** block: all members of a category share their superordinate's micro-columns (robin, sparrow, canary all
  carry the BIRD block). Attach a property association to the SHARED (superordinate) cells: teach `bird → flies` by potentiating
  the coincidence pathway from the BIRD block to `flies`. Then present `robin` (never taught `flies`): robin's SDR contains the
  BIRD block → the shared cells drive the learned BIRD→flies coincidence → `flies` is primed → **inheritance emerges**, never told.
- **Citation:** Collins & Quillian 1969 (economical hierarchical storage + inheritance-by-traversal); Rogers & McClelland 2004 /
  Lambon Ralph 2017 (*Nat Rev Neurosci* — similarity = shared representation); Saxe-McClelland-Ganguli 2019 *PNAS* (taxonomic
  inheritance emerges from feature-prediction). Catalog: G.13 (semantic store); Kandel Ch 52 pp 1306-1308 (schema).
- **Reusable machinery:** VERBATIM EMERGE-17/22 code-design + EMERGE-20 grounded-moat runner; `build_pool_bridge` /
  `apply_kernel_update` / `coincidence_predict` from `_emerge14`; the three-block encoder from `_emerge22`. NO new mechanism.
- **Cheap-first experiment:** superordinate BIRD = {robin, sparrow, canary}, FISH = {salmon, trout}. Teach class-level properties
  `BIRD-block → flies`, `FISH-block → swims`. Hold out `robin`/`canary`/`trout` from *ever* being taught the property. Test:
  `robin → ?` primes `flies`, `trout → ?` primes `swims` — the **never-taught inherited property**. Numpy backend, 6-seed.
- **Anti-cheats (all mandatory):** (1) **DERANGED superordinate** — assign each member a random (wrong) superordinate block →
  inheritance collapses to chance (isolates the shared is-a block as the cause, exactly as EMERGE-17's orthogonal-code control
  gave 0.000); (2) **dAP-LESION** → collapses to 0 (the coincidence recurrence is load-bearing); (3) **DISJOINT-code control** — a
  member whose code shares NO superordinate cells inherits NOTHING (moat: cannot confabulate an inheritance it has no pathway for);
  (4) **specific-over-general override** — teach `penguin → NOT-flies` directly and verify the *specific* fact beats the inherited
  default (Collins-Quillian cancellation; a stored exception overrides inheritance — this is the load-bearing "penguins don't fly"
  test that distinguishes real inheritance from blind overlap); (5) **held-out members** never trained on the property; (6) **6-seed**.
- **GO shows:** the emergent cortex INFERS an untold property from category membership + a class-attached property — Collins-Quillian
  inheritance, emergent, on one spiking brain, NO `sim/` edit. The FIRST demonstration of inference-beyond-told-facts on the
  substrate. `sim/` edit: **NONE** (code-design + reuse).

### (b) TRANSITIVE / RELATIONAL inference by recombining overlapping learned pairs (R-b)
- **Mechanism:** teach adjacent premise pairs that share a middle element (A→B, B→C, C→D, D→E) as separate facts; test the untrained
  relation between non-adjacent items (A?C, B?D) by whether the read-out over the overlapping item codes TRANSFERS (item-wise rank)
  vs MEMORIZES (conjunctive). On the HTM substrate: if B's code overlaps between the A→B and B→C pathways, chaining A→B then
  B→C should let A prime C's neighborhood; the empirical question is whether the coincidence read-out is the transferring kind.
- **Citation:** Eichenbaum-Cohen (catalog D.02); Kay 2024 *PLOS Comput Biol* (TI emerges without overt transitive structure);
  "A mathematical theory of relational generalization in TI" *PNAS* 2024 (item-wise read-out transfers, conjunctive does not);
  Dusek & Eichenbaum 1997 (hippocampal TI).
- **Reusable machinery:** `_emerge14` on-bridge learner (multi-fact store, K=32 validated); the EMERGE-18 high-order
  shared-middle corpus is *the same shape* (shared middle, context-dependent branch); the query/read-out from EMERGE-20.
- **Cheap-first experiment:** a 5-item ordered chain via 4 overlapping premise pairs; test the 6 untrained non-adjacent relations;
  score transfer vs a conjunctive (pair-memorizing) control read-out. 6-seed.
- **Anti-cheats:** (1) **conjunctive-read-out control** — must NOT transfer (isolates the item-wise geometry, per the PNAS theory);
  (2) **permuted-chain** control (shuffle premise pairs → no consistent order → transfer collapses); (3) **dAP-lesion**;
  (4) **held-out non-adjacent pairs** never co-presented; (5) **symbolic-distance signature** — accuracy should RISE with rank
  distance (the behavioral SDE, a positive functional signature, not just above-chance); (6) 6-seed.
- **GO shows:** relational/transitive inference emerges on the substrate by recombining overlapping learned pairs — the
  hippocampal relational-memory hallmark, on one spiking brain. `sim/` edit: **likely NONE** (reuse). Do AFTER (a) proves the
  simpler shared-superordinate inheritance works.

### (c) MULTI-HOP inheritance / the is-a taxonomy chain, emergent (deepen R-a toward the composer's chase, but as inference)
- **Mechanism:** stack (a) — `robin isa bird`, `bird isa animal`, property `animal → breathes` — and test `robin → breathes`
  (a TWO-hop inherited property, robin→bird→animal→breathes, never told at robin OR bird). If each level shares its superordinate's
  block transitively (robin carries BIRD ⊃ ANIMAL cells), the property attached at ANIMAL is reachable from robin.
- **Citation:** Collins-Quillian multi-level hierarchy; the composer's validated dog→mammal→vertebrata→chordata chase
  (`_fluidconv_phase15`) — but here as EMERGENT inheritance on the HTM cortex, not a retrieval walk over told edges.
- **Reusable machinery:** (a)'s runner; the HTM TM as the SR-style multi-hop predictor (Stachenfeld 2017).
- **Cheap-first experiment:** 3-level taxonomy; property attached at each level; test inheritance at 1, 2, 3 hops of distance;
  measure whether accuracy holds across hops (the "error doesn't compound because cleanup re-discretizes each hop" property the
  composer's multi-hop query already showed).
- **Anti-cheats:** deranged taxonomy; per-hop held-out; lesion; 6-seed; nested-block-vs-flat control.
- **`sim/` edit:** NONE. Do after (a).

### (d) ACQUIRE the relational structure from EXPERIENCE — statistical/latent learning (R-c, harder, deferred)
- **Mechanism:** instead of being handed `robin isa bird`, DISCOVER the shared-superordinate block from a stream of observations —
  co-occurrence statistics (robin/sparrow/canary appear in the same contexts → their online-Hebbian codes come to share cells,
  forming an emergent BIRD block) — the relational extension of the EMERGE-19 PPMI stream cortex (which already discovers *lexical*
  similarity from co-occurrence). Then run (a) on the DISCOVERED (not told) categories.
- **Citation:** Saxe-McClelland-Ganguli 2019 (hierarchy emerges from feature-prediction statistics); the project's own on-bridge
  Hebbian co-occurrence cortex (`corr(M,C)=+0.885`, generalizes held-out 0.86); O'Keefe-Nadel latent learning (catalog D.21:
  "no-reward exploration produces a map that drives later behavior"); Spens-Burgess 2023 (replay trains the generative model).
- **Reusable machinery:** `_phaseB_onbridge_stream_cortex_derisk.py` + the PPMI codes (`_phaseB_stream_codes_320_seed42.npy`);
  NREM/excitability replay for offline recombination.
- **Cheap-first experiment:** stream a corpus with latent category structure (members co-occurring in shared frames), learn codes
  online, verify the emergent codes share a superordinate block (RSA against the ground-truth categories), THEN show (a)'s
  inheritance works on the *discovered* categories.
- **Anti-cheats:** category-derangement of the STREAM (destroys the latent structure → no shared block emerges → no inheritance);
  RSA provenance (the shared structure is statistical, label-free); lesion; 6-seed.
- **`sim/` edit:** likely NONE (reuse the stream cortex); this is the acquisition-from-experience direction of the master directive.
  Deferred until (a)-(c) prove the substrate can INFER over given structure at all.

### (e) OFFLINE REPLAY-DRIVEN consolidation of inferred structure (R-c, the CLS/schema mechanism; deepest, deferred)
- **Mechanism:** use SWR/excitability replay to RECOMBINE learned premise pairs offline (replay A→B interleaved with B→C so the
  cortex forms the recombined A→C pathway during "sleep"), consolidating inference into durable neocortical structure — the
  biological mechanism that turns episodic pairs into inferable schema.
- **Citation:** Kandel N.14 (systems consolidation); Spens-Burgess 2023 *Nat Hum Behav*; *Nat Neurosci* 2025 (composition + replay);
  Tse 2007 (schema); Bouhadjar 2023 (coherent-noise probabilistic replay — the branch-recombining replay mode already in the paper).
- **Reusable machinery:** the Bouhadjar excitability-replay roll-out (EMERGE-16/23) + NREM scaffolding.
- **Cheap-first experiment:** after training only premise pairs, run a replay phase, then test whether the untrained recombined
  relation improved vs no-replay (replay causally builds the inference, mirroring the EMERGE-generative-loop self-replay result).
- **Anti-cheats:** no-replay control (inference should be weaker); permuted-replay (scrambled replay → no gain); lesion; 6-seed.
- **`sim/` edit:** possibly a tiny additive excitability flag if not already exposed (per the EMERGE-16 generation note); otherwise
  NONE. Deferred — highest-value but only after (a)-(d).

---

## 4. VERDICT — surpassable, and how cheaply. The single recommended next de-risk: EMERGE-26

**The residual is surpassable and CHEAP — it is not a wall.** The biology (Collins-Quillian; Eichenbaum-Cohen relational memory,
catalog D.02; O'Keefe-Nadel cognitive-map inference, D.21) and the recent computational theory (Kay 2024; the PNAS 2024 TI theory;
Saxe-McClelland-Ganguli 2019; TEM/Whittington 2020; Stachenfeld 2017 SR) converge on ONE load-bearing point: **relational/semantic
inference EMERGES from overlapping/shared codes × a next-state predictor — without an explicit inference engine.** The substrate
already has both ingredients (EMERGE-17 shared-family micro-columns; the HTM TM predictor), plus replay for the offline half. The
"open-world semantics" wall is, on inspection, a **code-design + read-out** question of the exact same shape as the EMERGE-17
generalization win ("the ONLY change is the word→code encoding"). The genuinely-irreducible part — *acquiring* the relational
structure from raw experience at scale (R-c) — is a real research direction, but it is DOWNSTREAM of first showing the substrate
can INFER over given structure, and it too reuses existing machinery (the PPMI stream cortex + replay), not a new mechanism class.

**THE SINGLE CHEAPEST NEXT DE-RISK — EMERGE-26 (path (a), do this first):**

> **Runner:** new `research/runners/_emerge26_emergent_inheritance_derisk.py` — reuse-by-import: `build_pool_bridge` /
> `apply_kernel_update` / `coincidence_predict` from `_emerge14_stageC_onbridge_learning_derisk.py`; the three-block SDR encoder
> from `_emerge22_pos_frame_grammar_derisk.py`; the grounded-moat production/abstain read-out from `_emerge20_grounded_moat_derisk.py`.
> **NO `sim/` edit** — the ONLY new thing is repurposing the shared block as an is-a/superordinate block and attaching the property
> to the shared cells.
>
> **Codes:** each concept = a THREE-block SDR: a **superordinate (is-a) block** shared by all members of its category (robin,
> sparrow, canary share the BIRD block; salmon, trout share the FISH block), a **content block** (unique per concept), and
> (optionally) a family block. Superordinates BIRD, FISH are disjoint from each other.
>
> **Facts taught (told):** class-level properties ONLY — `BIRD-block → flies`, `FISH-block → swims` — by potentiating the
> coincidence pathway from the shared superordinate cells to the property token, with the committed `fused_htm_permanence_update`,
> unsupervised, on the bridge (rung-4 settings: `nE=16`, `act_th=3`, `p_init=0.0`, ~40-60 epochs). The individual members
> (robin, canary, trout) are **NEVER taught the property.**
>
> **Held-out inheritance test:** present `robin` / `canary` / `trout` (whose SDRs contain the shared superordinate block) and read
> out the primed property. GO = the never-taught property is inherited (`robin → flies`, `trout → swims`), 6/6 seeds.
>
> **Anti-cheats (all mandatory, all reuse the EMERGE-17/20 control patterns):**
> 1. **DERANGED-superordinate control** — assign each member a random wrong superordinate block → inheritance collapses to chance
>    (isolates the shared is-a block as the cause; the analogue of EMERGE-17's orthogonal-code 0.000 control).
> 2. **dAP-LESION** — coincidence off → inheritance → 0.00 (the dendritic-plateau recurrence is load-bearing).
> 3. **DISJOINT-code / MOAT control** — a concept whose code shares NO superordinate cells inherits NOTHING and ABSTAINS (the
>    intrinsic no-confab moat holds through inference; it cannot confabulate an inheritance for which it has no shared pathway).
> 4. **SPECIFIC-OVERRIDE (Collins-Quillian cancellation)** — teach `penguin → NOT-flies` directly; verify the specific stored fact
>    BEATS the inherited default (real inheritance with exceptions, not blind overlap — this is the discriminating test).
> 5. **HELD-OUT members** never trained on the property (built into the design).
> 6. **6-seed** (42/43/44/100/101/102), no teacher.
>
> **What GO shows:** the FIRST emergent inference-beyond-told-facts on the substrate — Collins-Quillian property inheritance from
> category membership + a class-attached property, on one real spiking brain, with the intrinsic moat and the specific-over-general
> override intact, NO `sim/` edit. It converts "open-world semantics" from a feared wall into a demonstrated emergent capability.

**Chain after EMERGE-26 GO (all reuse-by-import, no `sim/` edit until R-c scale demands it):** immediately (b) transitive/relational
recombination (overlapping premise pairs, item-wise-vs-conjunctive read-out, symbolic-distance signature) → (c) multi-hop
inheritance (2-3 level taxonomy) → then research-gate R-c (acquire the structure from experience via the PPMI stream cortex (d) +
replay-driven offline recombination (e)). Each rung is one controlled runner; the deep, genuinely-open frontier (R-c at scale —
discovering rich relational structure from raw perception/observation, not curated facts) is named honestly and is the
experience-driven-learning direction of the master directive, to be gated after the substrate's *ability to infer* is proven.

**Honest scope / what would make this a real wall (and why it isn't yet):** EMERGE-26 tests inference over HOST-DESIGNED shared
codes (the is-a structure is given, not discovered) — that is a legitimate de-risk of the *inference mechanism*, but the shared
is-a block is a designed structure, so a GO proves "the substrate INFERS over relational structure," NOT "it ACQUIRES relational
structure from experience." The latter (R-c: the shared superordinate block must EMERGE from co-occurrence/perception statistics,
per Saxe-McClelland-Ganguli + the PPMI stream cortex) is the genuinely-irreducible residual, and it is a research direction, not a
one-runner de-risk. But per the SURPASS discipline: this is surpassable-and-how (paths d/e reuse the stream cortex + replay, both on
the substrate), not a wall — and the cheapest first step (EMERGE-26) is unambiguous and airtight-anti-cheatable.

---

## Artifacts / key citations
- **Substrate + reusable machinery:** `_emerge14_stageC_onbridge_learning_derisk.py` (`build_pool_bridge`/`apply_kernel_update`/
  `coincidence_predict`); `_emerge17_generalizing_word_codes_derisk.py` (shared-family-micro-column generalization = the
  inheritance seed); `_emerge20_grounded_moat_derisk.py` (grounded production + intrinsic moat); `_emerge22_pos_frame_grammar_derisk.py`
  (three-block factored code = structure × content); `_emerge24_online_growth_derisk.py` (online fact growth); `sim/kernels.py`
  (`fused_htm_permanence_update`); the PPMI stream cortex (`_phaseB_onbridge_stream_cortex_derisk.py`,
  `_phaseB_stream_codes_320_seed42.npy`); the composer transitive-isa chase (`_fluidconv_phase15_wikidata_breadth_derisk.py`, the
  *told-edge retrieval* precursor on the idealized substrate). Prior gate:
  `2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`.
- **Catalog (Kandel 6e / O'Keefe-Nadel):** D.01 episodic memory (Ch 52 pp 1296-1302); **D.02 relational binding / transitive
  inference — Eichenbaum-Cohen (Ch 52 pp 1301-1302)**; D.21 cognitive-map inference / latent learning / shortcut (O&N 1978 Ch 2,
  13-14; Kandel Ch 54); D.03-D.06 trisynaptic loop + CA3 autoassociator + place cells (Ch 54 pp 1340-1366); **J.34 schemas / gist
  (Ch 52 pp 1306-1308)**; **N.14 hippocampal-neocortical dialogue / systems consolidation (Ch 52 p 1299, Ch 54 p 1366)**; G.13
  Wernicke's area — semantic store prerequisite (Ch 55 pp 1384-1385).
- **Literature:** Collins & Quillian 1969 (*J Verbal Learning Verbal Behav* 8:240 — hierarchical semantic memory + inheritance);
  Rogers & McClelland 2004 *Semantic Cognition* (MIT Press); Lambon Ralph, Jefferies, Patterson & Rogers 2017, "The neural and
  computational bases of semantic cognition", *Nat Rev Neurosci* 18:42 (hub-and-spoke); Saxe, McClelland & Ganguli 2019, "A
  mathematical theory of semantic development in deep neural networks", *PNAS* 116:11537 (taxonomic inheritance emerges from
  feature-prediction); Kay et al. 2024, "Emergent neural dynamics and geometry for generalization in a transitive inference task",
  *PLOS Comput Biol* (PMC11125559 — TI emerges without overt transitive structure); "A mathematical theory of relational
  generalization in transitive inference", *PNAS* 2024 (10.1073/pnas.2314511121 — item-wise read-out transfers, conjunctive does
  not); Eichenbaum & Cohen 2014 (relational memory); Whittington et al. 2020, "The Tolman-Eichenbaum Machine", *Cell* 183:1249
  (factorize structure × content → generalize relational knowledge); Stachenfeld, Botvinick & Gershman 2017, "The hippocampus as a
  predictive map", *Nat Neurosci* 20:1643 (successor representation → generalization/inference); Spens & Burgess 2023, "A generative
  model of memory construction and consolidation", *Nat Hum Behav* 7:1965 (replay → relational inference + schema); "Constructing
  future behavior in the hippocampal formation through composition and replay", *Nat Neurosci* 2025; "Human hippocampal ripples
  coordinate planning sequences and compositional representations in neocortex", *Nat Neurosci* 2026; Tse et al. 2007 (schemas);
  McClelland, McNaughton & O'Reilly 1995 (CLS); Bouhadjar et al. 2022 *PLoS Comput Biol* 18(6):e1010233 (the ported substrate) +
  2023 19(5):e1010989 (coherent-noise probabilistic replay).
