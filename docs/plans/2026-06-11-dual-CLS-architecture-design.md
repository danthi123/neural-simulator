---
type: plan
status: live
date: 2026-06-11
---

# Dual / complementary-learning-systems (CLS) architecture — the design opening move (Option B pivot)

**Status:** READ-ONLY deep-research + design opening move (the project's standing "deep research FIRST at a new
direction", CLAUDE.md). No `sim/` code, no build, no GPU. Single deliverable: this doc + one commit. **Date:**
2026-06-11. **Author role:** read-only design subagent. Every load-bearing project fact below is file/line cited and
re-verified against the project's own record; the surprising ones (graded-similarity verdict) were measured-on-paper,
not trusted from a summary.

**Why this doc exists (the pivot, in one paragraph).** The owner chose **Option B** — a cortex that GENERALIZES
("a cat is like a dog" because similar concepts have similar codes). The direct path to B was *whiten the brain's
similar concept codes in place so they become binding-ready*. That path is now **falsified**: the afternoon-scale
`option_B_whitening_derisk_probe.py` showed that **even the IDEAL (god's-eye) whitening cannot co-satisfy
decorrelation (≤0.1) + reproducibility (≥0.9 at noise σ=0.1) + composition** on the brain's real `denoise64` codes,
multi-seed 42/43/44 (`research/findings/2026-06-11-option-B-whitening-derisk-NEGATIVE.md`). The codes are
sub-reproducible at σ=0.1 even **raw** (0.16), and whitening makes reproducibility *worse* (amplifies the noise) — and
the codes carry no graded similarity to preserve anyway (off-diagonal cosine **0.81 ± 0.04**, uniform — verified).
The falsification's decisive recommendation, which the project's own build-plan independently names as the resolution
of its "deep tension" (`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md` §"The genuinely-deep open
tension"), is the **biology-faithful DUAL / complementary-learning-systems architecture**: keep SIMILAR codes in a
"cortex" representation for generalization, plus a LINKED DECORRELATED "hippocampal" expansion that the binder reads.
This doc is the design opening move for that architecture.

---

## 0. Terms (defined once — owner standing requirement; no undefined acronym)

- **CLS = complementary learning systems** (McClelland, McNaughton & O'Reilly 1995; updated Kumaran, Hassabis &
  McClelland 2016). The theory that an intelligent agent needs TWO learning systems: a SLOW one (neocortex) that
  gradually extracts *structured, overlapping* knowledge (generalization), and a FAST one (hippocampus) that quickly
  stores the *specifics of individual experiences* in *pattern-separated* (decorrelated) form (episodic, no
  interference). They are LINKED: the hippocampus encodes from cortex and, via replay, gradually teaches cortex.
- **DG = dentate gyrus.** The hippocampal input stage. Performs **pattern separation**.
- **pattern separation** — making similar inputs MORE distinct (decorrelating). The DG's job (catalog D.12).
- **pattern completion** — reconstructing a full stored pattern from a partial/noisy cue. The CA3 recurrent
  autoassociator's job (catalog D.13). D.13's own note states the separation↔completion balance directly: *"too much
  completion → confused episodes; too little → no generalization."* This IS the CLS trade-off, in the catalog.
- **CA3 / CA1** — hippocampal subfields. CA3 is the recurrent autoassociator (completion); CA1 is the output stage
  that projects back toward cortex (the retrieval/consolidation **link**).
- **graded similarity** — similar concepts get proportionally-similar codes (cat~dog cosine HIGH, cat~bicycle LOW,
  systematically tracking meaning). The substrate generalization REQUIRES. The load-bearing open question of this doc.
- **decorrelated codes** — codes whose pairwise cosine is ≈0 (orthogonal-ish), so binding/unbinding is invertible and
  cleanup is reliable. The project's sparse-distributed codes (between-cos ≈ 0.05) are this.
- **hippocampal indexing theory** (Teyler & Rudy 2007, building on 1986) — the complementary biological framing: the
  hippocampus does NOT store the content; it stores a sparse **index / pointer** to the distributed cortical
  ensembles, and on recall the index reactivates them. This is exactly the engram-tag + sparse-code role the project
  already ships, and it is the conceptual model for the LINK (§1, §2).
- **binding (FHRR / VSA)** — the production composition algebra: roles and fillers are vectors; bind = element-wise
  complex product; unbind = multiply by the conjugate; cleanup = nearest-codebook readout. Realised on the bridge's
  resonate-and-fire neurons + complex synapses. Requires decorrelated codes to be reliably invertible.
- **the merged one-bridge substrate** — `research/runners/nav_conv_merged_bridge.py`: navigation + parser + dlPFC +
  RF composer as disjoint slices on ONE `SimulationBridge` (roadmap step 2, DONE). The substrate any new piece runs on.

---

## 1. The architecture (the CLS dual design, drawn on the project's own regions)

**The architecture in two sentences.** A **"cortex" representation** holds SLOW, OVERLAPPING, graded-similar concept
codes (so similar concepts have similar codes → generalization), and a **linked "hippocampal" decorrelated expansion**
holds FAST, PATTERN-SEPARATED codes (between-cos ≈ 0.05) that the FHRR binder + cleanup read reliably; the two are
COUPLED by an **encode** path (cortex → decorrelated expansion, the project's DG-style pattern separation / sparse
index) and a **retrieve/consolidate** path (expansion → cortex, the project's CA1→cortex link + SWR replay). Binding,
episodic memory, and the no-confab moat live on the decorrelated side (already validated); generalization lives on the
graded-cortex side (the new piece); the link lets a query enter on either side and round-trip.

### 1.1 The data flow, concretely on the project's regions

```
                            ┌──────────────────────── CORTEX side (generalization) ─────────────────┐
                            │  GRADED-SIMILAR concept codes (cat~dog close, cat~bicycle far)         │
   sensory / linguistic ───►│  • candidate substrates §3: V1→ventral grounded codes, learned        │
   input ("dog", an image)  │    semantic embedding, concept-pool codes                              │
                            │  • slow, overlapping, redundant — a real cortex reads "whatever        │
                            │    messy code arrives"                                                  │
                            │  • USE: similarity-based inference / generalization (read a property    │
                            │    of a held-out neighbour because its code is close to a trained one)  │
                            └───────────────┬───────────────────────────────▲──────────────────────┘
                                            │ ENCODE                         │ RETRIEVE / CONSOLIDATE
                       (pattern separation: │ DG sparsifies the overlapping  │ (CA1 → cortex link;
                        D.12, the           │ cortical code into a           │  SWR replay gradually
                        decorrelator)       │ decorrelated sparse index)     │  teaches cortex — N.14)
                                            ▼                                │
                            ┌──────────────────────── HIPPOCAMPAL side (binding + episodic) ─────────┐
                            │  DECORRELATED sparse codes (between-cos ≈ 0.05)                          │
                            │  • DG (separation) → CA3 (completion / attractor) → CA1 (readout link)   │
                            │  • the project's sparse-distributed codes (generate_sparse_patterns)     │
                            │  • USE: FHRR bind / unbind / bundle + localist NEF cleanup +             │
                            │    familiarity (no-confab) gate  — the VALIDATED composer                │
                            └─────────────────────────────────────────────────────────────────────────┘
```

The decisive insight (from the falsification): the two representations should NOT be the same vectors put through a
transform — that is the whitening path that died. They are **two distinct populations with two distinct code statistics**
(one graded, one decorrelated), **linked by trainable projections**, exactly as cortex and hippocampus are distinct
structures linked by the perforant path (in) and the CA1→subiculum→EC→cortex output limb (out). Binding never runs on
the graded codes (it cannot — they are too correlated, that is the whole point of the dead path); it runs on the
decorrelated expansion. Generalization never runs on the decorrelated codes (it cannot — they are equidistant by
construction, the documented Option-A limitation); it runs on the graded cortex codes. **Each capability lives on the
representation that supports it, and the link moves between them.** This is why the dual architecture is the resolution
and the single-representation whitening was not.

### 1.2 What each side delivers (and why neither alone suffices — the CLS argument)

- **Cortex side alone (the prior Option A "semantically-flat" composer):** binds reliably, passes the full V=320
  capability matrix (who/what, abstention, negation, clause, two-attribute, dialogue — `2026-06-10-vocab-ceiling-
  multiseed-GO.md`), but **cannot infer cat~dog** because every concept is equidistant. Conversational FUNCTION
  complete; biological FIDELITY (generalization) absent.
- **Hippocampal side alone:** fast episodic storage + pattern separation/completion (the project's trisynaptic loop,
  P1-validated), but a pure pattern-separator destroys similarity by design — it makes "two visits to the same
  restaurant" *distinct*, the opposite of generalization.
- **The dual, linked:** the cortex generalizes; the hippocampus binds and stores specifics without interference; the
  link lets a novel/episodic fact be stored fast (hippocampal) and gradually generalized (consolidated into cortex).
  This is precisely Kumaran-Hassabis-McClelland 2016's "two systems an intelligent agent needs."

---

## 2. How much ALREADY EXISTS (the key efficiency finding)

**Headline: the BINDING half and the DECORRELATED-EXPANSION + LINK half are LARGELY BUILT and VALIDATED. The
genuinely-new piece is the GENERALIZATION half — graded-similarity cortex codes and reading them for inference.** The
dual architecture is *much* less far than "a new arc from scratch" implies, because the project already shipped the
hippocampal machinery (it was built for continual learning) and the binder.

| CLS piece | Existing project machinery (file-cited) | REUSE or NEW? |
|---|---|---|
| **Decorrelated codes** (the binding substrate) | `concept_pool_sparse_distributed.generate_sparse_patterns` (K=100/N=2000 sparse random; between-cos ≈ 0.05, verified 0.0002–0.0027 in the positive control). 320 flat-distinct concepts validated (`2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md`). | **REUSE — validated** |
| **Pattern separation / DG** (the decorrelator = the ENCODE path) | `build_biological_brain_regions(enable_hippocampus_consolidation=True)` (`text_minimal_isolation.py`) wires EC→DG→CA3→CA1 with DG PV-basket feedforward inhibition. **P1-validated D.12: DG cosine 0.218 from input 0.800 (58pp orthogonalization)**, `validate_trisynaptic_loop.py`, multi-seed 3/3. | **REUSE — validated as a decorrelator** |
| **CA3 completion** (episodic autoassociator) | Same builder; **P1-validated D.13: CA3 cosine 0.748 (>0.7) on the direct-CA3 Marr test**, seed 42 (43/44 within 3%). Plus the on-bridge spiking heteroassociative attractor `_D_sparse_heteroassoc.py` (permuted-control-clean, multi-seed, 2/2 at production cycles). | **REUSE — validated** |
| **CA1 → cortex link** (RETRIEVE path) | `build_biological_brain_regions` adds `ca1 → motor_X` / `ca1 → language_output` consolidation pathways; CA1 readout integrates CA3 + direct EC. | **REUSE (structure built); the *graded-cortex* target of the link is NEW** |
| **Consolidation link / SWR replay** (cortex write-back, the CLS "teach cortex slowly") | `consolidation_trainer.py` (`run_swr_replay_phase`, `run_concept_replay_phase`) + awake/sleep gates. **Validated: hippo-OFF retention 94%** — memory genuinely transferred to cortex (`2026-05-07-Phase-1.3-CONSOLIDATION-CONFIRMED`, `2026-05-08-Phase1.3-Tier2.1-*` 3/3 strict anti-cheat). Catalog N.14. | **REUSE — validated as the cortex-write-back link** |
| **Hippocampal index / pointer** (Teyler-Rudy indexing = a sparse tag pointing at distributed cortex) | The engram-tag API (`sim/bridge.py`: `start_engram_recording`/`commit_engram_tag`/`stimulate_tag`) — a sparse activity-tagged ensemble that reactivates distributed populations. Multitag cue retrieval 90% FULL / 100% PARTIAL multi-seed. | **REUSE — validated as an index** |
| **FHRR binder** (bind/unbind on the decorrelated side) | `NeuronModel.RESONATE_AND_FIRE` + complex synapses; `rf_phasor_composer.py`; co-resident on the merged bridge (roadmap step 2b DONE). | **REUSE — validated** |
| **Localist cleanup + no-confab familiarity gate** | NEF/TPAM spiking cleanup (`2026-06-05-composer-cleanup-NEF-GO`, == numpy 27/27); anti-Hebbian familiarity gate de-risked +0.982 margin (`2026-06-10-cortex-learned-cleanup-derisk-PARTIAL` TEST 3). | **REUSE — validated** |
| **The merged one-bridge substrate** | `nav_conv_merged_bridge.py` (nav + parser + dlPFC + composer on ONE bridge). | **REUSE — DONE** |
| **GRADED-SIMILAR cortex codes** | *(see §3)* — **the gap.** | **NEW (the real cost)** |
| **Reading graded codes for INFERENCE** (similarity-based generalization) | none — Option A is equidistant by design; no existing runner does held-out-neighbour inference. | **NEW** |

**The honest reconciliation of the "is it built?" question:** **~80% of the CLS *plumbing* exists and is
individually validated.** What is built: the decorrelated binding substrate, the DG decorrelator, the CA3 completion,
the CA1→cortex structural link, the SWR-replay cortex-write-back, the engram index, the binder, the cleanup, the
no-confab gate, and the one-bridge host. What is **NOT** built and is the real work: **(i)** a cortex code that
carries *graded semantic* similarity (§3 — the load-bearing gap), and **(ii)** a read-out that *uses* that similarity
to generalize (similarity-based inference), plus **(iii)** wiring the existing encode/retrieve links between the new
graded-cortex population and the existing decorrelated expansion (the pieces exist, but they have never been connected
as a graded-cortex ↔ decorrelated-hippocampus pair — the trisynaptic loop today separates *arbitrary* sparse inputs,
not graded cortical codes). The dual architecture is therefore an **assembly-plus-one-new-substrate** arc, not a
from-scratch arc — but the "one new substrate" is the deep one (graded codes), so do not under-scope it.

---

## 3. The load-bearing open question: graded-similarity codes — the verdict

**The question.** Generalization is *impossible* without a code where similar concepts get proportionally-similar codes.
Which project codes carry graded SEMANTIC similarity?

**The verdict: NO existing project code carries graded *semantic* similarity. This is the central new sub-problem.**
The candidates, with on-paper assessment:

### 3.1 `denoise64` (the brain's captured concept codes) — NO graded similarity (verified, decisive)
These are the obvious candidate ("the brain's own codes") and they are the ones Option B wanted to keep. **They do not
qualify.** Their off-diagonal cosine is **0.81 ± 0.04, uniform** (`2026-06-11-option-B-whitening-derisk-NEGATIVE.md`
§"Generalization (d)": std only 0.033–0.037, range 0.15–0.18). That is high *coherence* (a shared common mode), NOT
*graded* structure — there is no systematic "some concept pairs are closer than others tracking meaning." Reading the
generator confirms *why*: `load_real_codes` (`cortex_storkey_ca3_cleanup_probe.py:62`) takes `obs__<word>` = the
mean captured spiking of each concept POOL on a bridge trained with **orthogonal per-word drive patterns**
(`orthogonal_drive_pattern`, each word a disjoint band) over an arbitrary vocab list. The codes' similarity reflects
the shared substrate dynamics (the common mode) and the arbitrary vocab order, **not** semantic relatedness — there is
no mechanism by which apple~river would be closer than apple~dog. **denoise64 is correlated-but-not-semantic; it is
exactly the wrong code for generalization (and the whitening de-risk already showed it is also the wrong code for
binding).**

### 3.2 V1 Gabor → ventral-hierarchy grounded codes — graded *PERCEPTUAL* similarity only (NOT semantic)
The project's grounded codes (`sim/visual_cortex.py`, the cheat-#4 arc) are the most interesting candidate, and the
assessment is nuanced. The cheap-first probe measured it directly (`2026-06-04-cheat4-visual-grounding-cheap-first-
RESOLVES.md`): 12 visual concepts → real V1 Gabor bank (8192 cells) → **pairwise cosine mean 0.252, max 0.709, and the
single high-cosine pair is `bar_0deg ~ bar_22deg` — "adjacent orientations SHOULD be similar."** So V1 codes carry
**genuine graded similarity for genuinely perceptually-similar inputs** (a 0° bar is close to a 22.5° bar, far from a
90° bar — proportional to geometric similarity). **This is graded similarity — but it is PERCEPTUAL, not SEMANTIC.**
Two decisive caveats:
  1. **Cat~dog are not perceptually graded-close in V1.** Two photographs of a cat and a dog do not produce close V1
     Gabor responses just because they are semantically related; V1 codes oriented-edge statistics, not category. The
     graded structure V1 gives is "edges at similar orientations/positions are close," which is not the "cat is like
     a dog" generalization Option B wants.
  2. **The 320-concept agent integration THREW AWAY the V1 similarity.** When the V1 pipeline was scaled to the full
     320-concept benchmark (`2026-06-04-cheat4-visual-grounding-agent-integration.md`), each concept got a *synthetic
     distinct texture* (arbitrary, no semantic structure) AND the agent only worked **after a ventral-hierarchy ZCA
     DECORRELATION step that drove inter-code cosine to ~0** (mean/max ~0/~0). The decorrelation was *required* for
     two-attribute composition. So the production grounded path **deliberately removes** whatever similarity V1
     induced — confirming, again, that binding and similarity pull opposite ways, and the project's existing grounded
     pipeline is on the binding (decorrelated) side, NOT the generalization (graded) side.

  **Net:** V1/ventral codes are a *proof that the substrate CAN carry graded perceptual similarity*, and a real
  ventral hierarchy (V1→V2→V4→IT) does build toward graded *object/category* similarity (Tanaka 1996 IT) — but the
  project's current pipeline (a) only grounds visually-similar inputs perceptually, (b) has no IT-level semantic
  category code, and (c) discards the similarity it does have. **Perceptual graded similarity: present. Semantic
  graded similarity (cat~dog): absent.**

### 3.3 Concept-pool / Wernicke learned codes — NO graded semantic similarity (orthogonal by construction)
The concept-pool architecture (`concept_pool_*`, v14/v16) and the P5 ventral semantic stream deliberately use
**orthogonal drive codes** (`orthogonal_drive_pattern`, `--orthogonal-codes`) precisely to maximize discriminability
— each concept a disjoint band. These are engineered to be EQUIDISTANT, the opposite of graded. The learned
lang_input→pool weights bind a word to a pool, but the pools themselves carry no graded inter-concept structure (the
whole v14 breakthrough was making them *more* separable). **No graded semantic similarity.**

### 3.4 The verdict, stated precisely
**The project has no code where semantically-related concepts are systematically closer.** The closest thing is V1
perceptual similarity (real but perceptual, and discarded in production). Therefore:

> **The central NEW sub-problem of Option B is: the cortex codes must be LEARNED to be similarity-preserving — a
> learned semantic embedding in which related concepts cluster (cat~dog close, cat~bicycle far), grounded in
> co-occurrence / shared-context / shared-attribute structure, realized on the spiking substrate.**

This is a real arc, not a parameter sweep. It is the genuinely-open half of the dual architecture, and the honest cost
of Option B. The biology is well-posed (cortex DOES learn graded semantic representations — the distributed
Pulvermüller word ensembles G.20, the IT object code, the "semantic hub" — and the brain learns them from statistics),
but **no project code currently realizes it**, and the §4 de-risk must therefore either (a) identify a project signal
that can be SHAPED into graded codes, or (b) use a controlled synthetic graded codebook to validate the *architecture*
while flagging the learned-embedding substrate as the follow-on. **Recommendation: the §4 de-risk uses a synthetic
graded codebook to prove the dual architecture works in principle and is similarity-driven, BEFORE committing to the
learned-embedding substrate build** — because if the architecture does not deliver generalization even on ideal graded
codes (e.g. the encode→decorrelate→retrieve link destroys the similarity — §5 risk ii), the learned-embedding arc is
moot, and that is the cheapest thing to falsify first.

---

## 4. The cheap-first de-risk (the single load-bearing falsification, specified precisely)

**The single open question for the dual architecture is: does keeping graded-similar codes in a "cortex" representation
actually deliver a generalization capability that the flat decorrelated codes CANNOT — AND does it survive the link to
the decorrelated binding side?** Everything else (binding, separation, completion, consolidation) is already validated
(§2). The de-risk isolates exactly the NEW claim. CPU/numpy, NO substrate rewrite, reuse-by-import, multi-seed
42/43/44. It has **three gates that must ALL pass**, mirroring the structure that made the whitening de-risk decisive.

### 4.1 Probe A — GENERALIZATION on graded codes (the new capability), with the decisive orthogonal-control contrast
**This is the headline gate and the reason Option B exists.** A held-out-neighbour inference test.

- **Codes.** Build a controlled **synthetic graded codebook** (§3.4 recommendation): K concept "clusters" (e.g.
  {dog, cat, wolf, fox} = canids/felids cluster; {car, truck, bike} = vehicles cluster), where within-cluster cosine
  is HIGH (e.g. 0.7) and between-cluster cosine is LOW (e.g. 0.1), constructed by a low-rank "category factor + concept
  residual" generator so similarity is *graded and semantic-by-construction*. **Report the cosine matrix** (the
  unit-check: within-cluster ≫ between-cluster, graded). *Parallel candidate, if cheap:* the V1 codes for a set of
  genuinely-perceptually-graded stimuli (§3.2) as a real-substrate cross-check — but the synthetic codebook is the
  primary, because it isolates *semantic* graded structure cleanly.
- **Task.** Train a simple relation/property read-out on a SUBSET of concepts within each cluster (e.g. teach
  "property P holds for dog and wolf"); **test inference on a HELD-OUT cluster-neighbour never trained in that
  relation** (e.g. query property P for "cat"). The read-out is a similarity-weighted retrieval (a learned linear
  read-out or a nearest-trained-neighbour vote over the graded codes — the simplest mechanism that *can* generalize;
  this is the cortex's "read whatever code arrives" stand-in, NOT the exact-inverse algebra).
- **GATE A1 (generalization PASSES on graded codes):** held-out-neighbour inference accuracy ≫ chance (target: clears
  a pre-registered bar, e.g. ≥ 0.7 with chance = 1/K).
- **GATE A2 (the DECISIVE CONTRAST — it FAILS on orthogonal codes):** run the IDENTICAL test on the project's
  **generated orthogonal sparse codes** (`generate_sparse_patterns`, between-cos ≈ 0.05). Inference on the held-out
  neighbour MUST collapse to **chance** — *because the inference only works if similar concepts have similar codes.*
  This contrast is what proves generalization is **similarity-driven**, not an artifact: graded → generalizes,
  orthogonal → cannot. (This is the dual of the positive control's GATE B: there the attractor worked on decorrelated
  and collapsed on correlated; here inference works on graded and collapses on orthogonal.)
- **ANTI-CHEAT A3 (permuted-similarity control):** shuffle which concepts are "similar" (break the code-similarity ↔
  semantic-similarity correspondence: assign each concept a random cluster label decoupled from its code) → held-out
  inference must collapse to chance. Otherwise the "generalization" is code-overlap unrelated to meaning. **This is the
  mandatory headline anti-cheat** (the analogue of the whitening de-risk's reproducibility headline) — a generalization
  number without the permuted-similarity control is incomplete and must be rejected.

### 4.2 Probe B — BINDING preserved (reuse the positive control, no new work)
Confirm the binding/episodic side is untouched: bind/retrieve on the decorrelated sparse expansion. **Reuse
`cortex_sparse_attractor_poscontrol_probe.py` verbatim** — it already shows argmax/Hopfield parity 1.000 on the
decorrelated codes, multi-seed, with the noise-cue no-hallucination anti-cheat. **GATE B: PASS = the existing 1.000.**
(No new code; this gate exists only to assert the dual architecture does not regress the validated binding side.)

### 4.3 Probe C — the LINK round-trip (the deepest, most-novel gate)
**The crux of the dual architecture and the place it can fail (§5 risk ii).** Does a graded cortex code survive the
encode→decorrelate→bind→retrieve→decode round-trip with BOTH binding AND similarity intact?

- **Encode** a graded cortex code → its decorrelated sparse expansion. Use the **project's DG-style pattern separation
  as the encoder** — either the trisynaptic-loop DG (`validate_trisynaptic_loop.py`'s `measure_region_response` on the
  `dg` region, which P1-validated takes input cos 0.80 → DG cos 0.22) OR, for the cheap numpy pass, a fixed
  random-projection-then-top-k sparsifier as the DG stand-in. **Report the encode map's cosine behaviour:** it MUST
  decorrelate (graded cortex cos → expansion cos ≈ 0.05, so binding works).
- **Bind + retrieve** on the expansion (Probe B's validated path) → recover the stored decorrelated code.
- **Decode** the retrieved decorrelated code back toward the graded cortex code (the CA1→cortex link — a learned
  linear read-out from expansion to cortex codebook, the project's consolidation pathway analogue).
- **GATE C1 (binding round-trips):** the decoded concept identity is correct (the right concept comes back) — parity
  with Probe B.
- **GATE C2 (the LOAD-BEARING new gate — similarity survives the round-trip):** the decoded cortex codes, for a set of
  concepts, **preserve the graded similarity structure** — i.e. cosine(decoded_cat, decoded_dog) ≫ cosine(decoded_cat,
  decoded_bicycle), tracking the original cortex similarity (report the correlation between the original and
  round-tripped cosine matrices; target: high, e.g. Pearson ≥ 0.8). **If the round-trip preserves binding but
  DESTROYS the similarity (the decorrelation is irreversible), the dual architecture's link is broken** — the cortex
  could generalize and the hippocampus could bind, but a query that goes round the loop loses exactly what
  generalization needs. **This is the single most important number in the de-risk** (the inverse of the binding problem
  — §5 risk ii).
- **Honest note on what C2 actually tests:** generalization (Probe A) does NOT have to go through the round-trip in the
  final architecture — a generalization query can be answered *directly on the cortex side* without ever decorrelating
  (the cortex read-out reads the graded codes in place). So C2 is testing the *strong* form of the link (can a stored
  episodic fact be generalized after consolidation). If C2 fails but A passes, the fallback architecture is "cortex
  generalizes in place; hippocampus binds in place; the link is one-way (encode-only) for episodic storage, and
  consolidation teaches cortex via the SLOW replay path (§2, validated at 94% retention) rather than a per-query
  decode" — which is *still* a valid dual architecture (it is, in fact, closer to the biology: consolidation is slow,
  not a per-query inverse). C2 thus *sharpens* the architecture rather than gating it binary: PASS → the link is a
  fast bidirectional codec; FAIL → the link is encode-fast / consolidate-slow (the biological default). Report which.

### 4.4 Harnesses / codes to reuse (verbatim), gates, anti-cheats — summary
- **Reuse:** `cortex_sparse_attractor_poscontrol_probe.py` (Probe B + the noise-cue anti-cheat); `generate_sparse_patterns`
  (the orthogonal control codes for GATE A2 + the expansion in Probe C); `validate_trisynaptic_loop.py`'s
  `measure_region_response` + the DG region (the encoder for Probe C, if running on-bridge); the cosine/unit-check
  conventions from the positive control (native binary, mean-removed — assert it).
- **All gates (Option-B-dual GO requires all):** A1 (graded generalizes) **AND** A2 (orthogonal does NOT — the
  contrast) **AND** A3 (permuted-similarity collapses — the anti-cheat) **AND** B (binding parity 1.000, reused) **AND**
  C1 (round-trip identity) **AND** C2 (round-trip similarity preserved, OR the documented encode-fast/consolidate-slow
  fallback).
- **The headline anti-cheat is A3 (permuted-similarity)** — front-and-center, reported alongside A1, so "generalizes by
  code overlap unrelated to meaning" cannot masquerade as a win. (Exactly as reproducibility-≥0.9 was the headline for
  the whitening de-risk.)
- **On-substrate confirmation (only on numpy GO):** wire the encoder (DG) on a small `SimulationBridge` (`SIM_BACKEND=
  numpy` then `cupy`), run Probe C through the real DG separation + the existing CA1 link, re-run A/B/C on the bridge
  output. Only on that GO does any GPU build + the V=320 acceptance matrix proceed.

### 4.5 The anti-cheat that the WHOLE Option-B cortex must still pass (unchanged)
Generalization (§4.1) is the NEW gate; the dual architecture must ALSO not regress the validated capability matrix: the
full who/what-Q&A + **abstention / no-confab moat 100% (20/20 every cell)** + negation + embedded clause (D≥256) +
two-attribute, at **V=320 multi-seed (42–47)**, shuffled-fact permuted control at zero false hits, on the merged
one-bridge substrate. Option B must add generalization WITHOUT regressing anything Option A already does — and since
binding runs on the unchanged decorrelated side (§1.1), this should hold by construction, but it is asserted, not
assumed.

---

## 5. Honest risk register (every load-bearing assumption, flagged)

### 5.1 ⚠️ (i) NO project code has graded SEMANTIC similarity → generalization needs a new learned-embedding sub-arc (the real cost)
**The biggest honest cost.** §3's verdict: denoise64 is correlated-but-not-semantic, V1 is graded-but-perceptual (and
discarded in production), concept-pool codes are orthogonal-by-design. So the "keep the brain's similar codes" framing
that motivated Option B does NOT have a ready code to keep — the graded-semantic cortex code must be **learned from
scratch** (a semantic embedding where related concepts cluster, from co-occurrence / shared-attribute statistics, on
the spiking substrate). This is a months-plausible arc on its own, comparable in scope to the dendritic rewrite Option
B was trying to avoid. **Mitigation / framing:** the §4 de-risk uses a *synthetic* graded codebook to prove the
architecture is sound + similarity-driven BEFORE that arc is committed — so the expensive learned-embedding build is
gated on the cheap architecture proof, not assumed. If the architecture proof passes, the learned-embedding arc is
justified; if it fails (e.g. §5.2), it is avoided. **Do not let "the dual architecture is mostly built" (§2, true of the
plumbing) hide "the generalization substrate is not built and is the deep piece" (true and load-bearing).**

### 5.2 ⚠️ (ii) The encode→decorrelate→retrieve round-trip may DESTROY the similarity structure (a fatal risk for the strong link)
The inverse of the binding problem, and the deepest technical risk. **Decorrelation is, by design, similarity-removing**
(pattern separation makes similar things distinct — that is D.12's whole function). So the ENCODE path
(cortex→DG→expansion) deliberately throws away the graded similarity to make binding work. The question Probe C2 tests:
can the DECODE path (expansion→cortex) put it back? If the decorrelation is irreversible (information-lossy in the
similarity dimension), then a fact that round-trips through the hippocampal side comes back *similarity-stripped* —
binding survives, generalization does not, and the "link" cannot carry generalization. **This is genuinely uncertain
and is the single load-bearing technical unknown.** **Mitigation (already designed into §4.3):** C2 is structured as a
*sharpening*, not a binary gate — if the fast bidirectional codec fails, the architecture falls back to "generalize on
the cortex side in place; bind on the hippocampal side in place; consolidate slowly via replay (the validated 94%
path)" — which is *more* biologically faithful (consolidation IS slow) and still a valid dual architecture. The risk is
only fatal to the *strong/fast* form of the link; report which form survives.

### 5.3 (iii) "Generalization on graded codes" may be trivial / uninteresting vs a real capability
A nearest-neighbour vote over similar codes generalizing is, at one level, just "kNN works on a graded embedding" — not
obviously a *brain* capability worth the arc. **The risk:** the de-risk could show A1/A2/A3 pass and yet the
"generalization" is a shallow retrieval trick, not the rich inference Option B's "a cat is like a dog" framing implies
(property inheritance, analogy, novel-combination reasoning). **Mitigation:** scope the generalization claim honestly
in the de-risk — A1 demonstrates *similarity-based property inference* (a real, measurable, CLS-grounded capability),
NOT open-ended reasoning. If the owner's target is richer (analogical inference, schema-based reasoning per Tse 2007 /
the project's V_SCHEMA work), that is a SEPARATE, larger claim the de-risk does NOT establish — flag it as out of scope
for the architecture proof. The honest deliverable is "the dual architecture supports similarity-based generalization
that the flat composer cannot," not "the dual architecture reasons."

### 5.4 (iv) The consolidation link is SLOW (many replay cycles) — practicality
The cortex-write-back link (the CLS "teach cortex gradually") is validated but *slow*: the Phase 1.3 consolidation took
~25 min wall-clock per seed for 4–8 word vocab, and CLS theory *requires* it be slow (fast cortical learning = the
catastrophic interference CLS exists to avoid). **Risk:** at V=320 the consolidation arc is expensive, and a per-query
"consolidate then generalize" loop is impractical. **Mitigation:** the architecture does NOT require per-query
consolidation — generalization (§4.1, A) runs *directly on the cortex side* (no consolidation in the loop); the slow
link is only for the offline episodic→semantic transfer that CLS theory assigns to sleep. So the slowness is a
*training-time* cost (acceptable, offline, the incremental/resumable trainer `concept_pool_sparse_distributed
--resume-from` already chunks such runs), not a *query-time* cost. Flag: do not design a query path that requires the
slow link.

### 5.5 (v) The graded-cortex ↔ decorrelated-hippocampus pair has never been WIRED (the plumbing is built but not connected this way)
§2's honest caveat: the trisynaptic loop today separates *arbitrary sparse inputs* (the P1 test drove `language_input`
with random sparse patterns), not *graded cortical codes*. The DG-as-decorrelator and the CA1-as-link are validated in
isolation, but the specific composition "graded cortex population → DG → expansion → binder, and back" is new wiring (no
runner does it). **Risk:** an integration surprise (the DG separation may behave differently on graded structured input
than on random sparse input; the CA1 link may not decode to a *graded* target). **Mitigation:** Probe C tests exactly
this composition on numpy first (cheap), and the on-substrate confirmation (§4.4) runs it through the real DG before
any GPU build — the integration risk is front-loaded into the cheap de-risk, not discovered at build time.

### 5.6 Assumption ledger (load-bearing claims this design rests on)
- **A-1.** denoise64 has no graded semantic similarity (off-diag 0.81 ± 0.04 uniform) — **VERIFIED** (de-risk doc +
  generator read). *Load-bearing: it kills "keep the brain's codes" and forces the learned-embedding arc.*
- **A-2.** The decorrelated binding substrate + DG separation + CA3 completion + CA1 link + SWR consolidation + binder
  + cleanup + no-confab gate are individually validated — **VERIFIED** (P1, Phase 1.3, positive control, NEF cleanup,
  familiarity gate findings). *Load-bearing: it is why this is assembly-plus-one-piece, not from-scratch.*
- **A-3.** V1 carries graded *perceptual* (not semantic) similarity, and the production grounded path discards it via
  ZCA — **VERIFIED** (cheat-#4 probes). *Load-bearing: V1 is a similarity proof-of-concept, not the semantic code.*
- **A-4 (UNVERIFIED — the de-risk's job).** A graded cortex code delivers held-out-neighbour generalization that
  orthogonal codes cannot, and the encode→retrieve link preserves (or the slow path consolidates) the similarity.
  *This is the entire point of §4; it is assumed nowhere and tested everywhere.*
- **A-5 (UNVERIFIED — the real cost).** A graded-semantic cortex code can be LEARNED on the spiking substrate. *Gated
  behind A-4; not attempted before the architecture proof.*

---

## Verdict

**The architecture (2 sentences):** A "cortex" representation holds slow, graded-similar codes (similar concepts →
similar codes → generalization) while a linked "hippocampal" decorrelated expansion (between-cos ≈ 0.05) holds the
pattern-separated codes the FHRR binder + cleanup read reliably, coupled by a DG-style encode path (cortex →
decorrelated) and a CA1/SWR retrieve-and-consolidate path (decorrelated → cortex). Binding and the no-confab moat live
on the decorrelated side (validated); generalization lives on the graded cortex side (new); the link lets a query enter
on either side — which is exactly the resolution of the binding-vs-generalization tension that single-representation
whitening could not achieve.

**How much already exists:** ~80% of the CLS *plumbing* is built and individually validated (decorrelated codes, DG
pattern separation D.12, CA3 completion D.13, CA1→cortex link, SWR-replay cortex write-back N.14 at 94% retention,
engram index, FHRR binder, NEF cleanup, no-confab gate, the merged one-bridge host). **NEW (the real cost):** the
graded-similarity CORTEX codes and a read-out that uses them for inference, plus connecting the existing encode/retrieve
links between a graded-cortex population and the decorrelated expansion (the pieces exist; this pairing is unbuilt).

**Graded-similarity-codes verdict:** **NONE qualify.** denoise64 is correlated-but-not-semantic (0.81 ± 0.04 uniform,
verified); V1/ventral codes carry graded *perceptual* similarity only (cat~dog are not perceptually close, and the
production path ZCA-decorrelates them away); concept-pool/Wernicke codes are orthogonal by construction. ⇒ **the central
new sub-problem is a LEARNED similarity-preserving semantic embedding** (related concepts cluster), a months-plausible
arc — the honest cost of Option B.

**The single cheapest-first de-risk:** a CPU/numpy, multi-seed, reuse-by-import probe with the decisive contrast at its
core — held-out-neighbour inference **PASSES on a synthetic graded codebook and FAILS (collapses to chance) on the
project's orthogonal sparse codes** (proving it is similarity-driven), with a **permuted-similarity anti-cheat** as the
headline control; **binding parity reused verbatim from the positive control** (1.000); and the **encode→decorrelate→
retrieve→decode round-trip** measured for whether the similarity structure SURVIVES the link (the load-bearing new
number — and structured so that if the fast codec fails, the architecture sharpens to the biologically-faithful
encode-fast/consolidate-slow link rather than failing). Run it on a synthetic graded codebook FIRST so the expensive
learned-embedding build is gated on a cheap architecture proof.

**The biggest risk:** the encode→decorrelate→retrieve round-trip may **destroy the graded similarity** (decorrelation is
similarity-removing by design — the inverse of the binding problem); Probe C2 is the load-bearing test, mitigated by the
designed fallback to the slow consolidation link. Closely behind: **no project code has graded semantic similarity**, so
Option B's generalization half is a genuinely-new learned-embedding arc, not a reuse — gated behind the cheap de-risk so
it is committed only if the architecture proves out.

**No banking.** Reported exactly as found, including the parts that reshape the arc (Option B has a ready binding half
but NO ready generalization code; the link's similarity-preservation is the deepest unknown).

## Sources

- McClelland, McNaughton & O'Reilly 1995, *Why there are complementary learning systems in the hippocampus and
  neocortex* (Psychological Review 102:419) — the foundational CLS theory.
- Kumaran, Hassabis & McClelland 2016, *What learning systems do intelligent agents need? Complementary learning
  systems theory updated* (Trends Cogn. Sci. 20:512–534; PMID 27315762) — CLS updated; cortex = slow structured
  generalization, hippocampus = fast pattern-separated specifics, linked by replay with goal-dependent weighting.
- Teyler & Rudy 2007, *The hippocampal indexing theory and episodic memory: updating the index* (Hippocampus 17:1158;
  PMID 17696170; orig. Teyler & DiScenna 1986) — the hippocampus stores a sparse INDEX / pointer to distributed
  cortical content; recall reactivates cortex via the index. The conceptual model for the link + the engram-tag index.
- Catalog (`sim-catalog/references/feature-catalog.md`): D.12 (DG pattern separation), D.13 (CA3 pattern completion —
  the "too much completion → confused; too little → no generalization" trade-off IS the CLS balance), D.14 (engram
  index), N.14 (hippocampal–neocortical dialogue / systems consolidation — the cortex write-back link). Kandel 6e
  Ch 54 pp 1340–1367, Ch 52 p 1299.

## Project cross-references (internal, all re-verified)

- The falsification that forced the pivot: `research/findings/2026-06-11-option-B-whitening-derisk-NEGATIVE.md`; the
  opening research it served: `docs/plans/2026-06-11-option-B-dendritic-substrate-research.md` (§6.4 the three-operating-
  points tension); the fork/build-plan that names the dual resolution: `docs/plans/2026-06-11-cortex-build-plan-
  decorrelate-then-bind.md` (§"The genuinely-deep open tension", §"The fork").
- The binding half (validated): `2026-06-11-cortex-sparse-attractor-poscontrol-GO.md`; `2026-06-02-full-320-flat-
  distinct-composition-RESOLVES-multiseed.md`; `concept_pool_sparse_distributed.generate_sparse_patterns`; the FHRR
  binder + NEF cleanup (`2026-06-05-composer-cleanup-NEF-GO.md`); the familiarity gate (`2026-06-10-cortex-learned-
  cleanup-derisk-PARTIAL.md`).
- The hippocampal half (validated): `research/runners/validate_trisynaptic_loop.py` + `2026-05-11-P1-trisynaptic-loop-
  validation.md` (D.12 58pp separation; D.13 0.748 completion); `build_biological_brain_regions(
  enable_hippocampus_consolidation=True)` in `research/runners/text_minimal_isolation.py`; the consolidation link
  (`research/runners/consolidation_trainer.py`; `2026-05-07-Phase-1.3-CONSOLIDATION-CONFIRMED.md`; `2026-05-08-
  Phase1.3-Tier2.1-*` 3/3 strict anti-cheat, 94% hippo-OFF retention); the engram index (`sim/bridge.py`
  engram-tagging API); the on-bridge spiking attractor (`research/runners/_D_sparse_heteroassoc.py`).
- The graded-similarity evidence: `2026-06-04-cheat4-visual-grounding-cheap-first-RESOLVES.md` (V1 graded perceptual
  similarity, mean 0.25 / max 0.71, bar_0deg~bar_22deg); `2026-06-04-cheat4-visual-grounding-agent-integration.md` (the
  ZCA decorrelation that discards it); `sim/visual_cortex.py` (the real V1 Gabor bank); the denoise64 generator
  (`research/runners/cortex_storkey_ca3_cleanup_probe.py:62` `load_real_codes`).
- The substrate: `research/runners/nav_conv_merged_bridge.py` (the merged one-bridge host, roadmap step 2 DONE).
