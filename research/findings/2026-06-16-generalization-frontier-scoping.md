# Generalization-across-similar-concepts frontier — scoping (deep-research + catalog review BEFORE build)

**Date:** 2026-06-16
**Type:** read-only deep-research + reference-catalog scoping (the standing opening move for a new direction).
**Author:** research subagent (no code edited, no heavy GPU run).
**Frontier scoped:** the agent composes *perceived* content into novel facts (validated, one spiking bridge) but
uses **flat-distinct codes**, so it does NOT generalize across **similar** concepts (it cannot treat "dog" and
"cat" as related because their codes are unrelated). A proper brain analogue generalizes: perceiving/hearing a
novel-but-similar concept should partially transfer. This doc maps the path.

**Verified-against-source:** every load-bearing project claim below was checked by opening the cited file
(line numbers given); every load-bearing biology claim is cited to the catalog (`sim-catalog`), Kandel 6e
(via the catalog's chapter cites), or current literature (URLs in §Literature).

---

## 1. Diagnosis — exactly what exists, and the exact gap

### 1a. What exists on the CONVERSATION side (already generalizes)

The conversation cortex **already generalizes across similar concepts**, on the real spiking substrate:

- **The PPMI "stream cortex."** A cortex that hears the corpus word-by-word learns **correlated semantic codes**
  from word co-occurrence: online Hebbian co-occurrence (`M[target, hub] += 1` over a working-memory window) +
  a running per-word frequency estimate + a **log-domain double-centring** read-out (the local normalization).
  These codes reach the host PPMI reference (Pearson +0.513) **and generalize to held-out role-filler
  combinations** (held-out ≈ 0.86–0.91, chance 0.12). Verified: `_phaseB_online_stream_cortex_derisk.py:80-105`
  (the online-Hebbian + double-centre + held-out-generalization), and the multi-seed result in
  `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md` (+0.513, gen 0.91, 3 seeds).
- **It is realized ON the spiking substrate.** Rate-Hebbian co-occurrence learning on a real `SimulationBridge`
  gives `corr(M,C)` +0.686 (6 seeds); the **population code** lifts the single-neuron read-out from 47% to
  100–108% of host; the full who/what recall + the no-confab abstention moat run on the stream-learned codes
  (recall 1.00; moat 0.96, restored to 1.00 with more stream). Verified:
  `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (the STDP-negative / Hebbian-GO
  mechanism table, the population lift table, the conversation-on-stream-codes table), and
  `_phaseB_stdp_cooccurrence_derisk.py` (the on-bridge runner; population via `--n-per`).
- **CYCLE 88's reframe (load-bearing, verified):** generalization needs **feedforward LOCAL normalization**
  (PPMI = `ReLU(log(count) − log(per-hub total) − log(per-concept total))`, all local ops), **NOT cross-neuron
  decorrelation** — which would *destroy* generalization (whitening over-processes: similarity-matching
  whitens → +0.35; PPMI just normalizes → +0.52). The "off-diagonal decorrelation wall" was a wall for the
  *wrong* approach. Verified: `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md:29-35,
  62-76` (the decomposition + the explicit "whitening destroys generalization" statement).
- **The compose ALGEBRA tolerates correlation.** The fixed VSA bind/unbind holds up to code-similarity ≈0.98
  (degrades only when codes are ~99.9% identical). Verified: `2026-06-16-step3-live-cortex-grounded-compose-cheap-first.md:138-167`
  (the α-sweep table + the crucial caveat). **So "the algebra tolerates correlation" is NOT the blocker** — and
  the same finding states the caveat verbatim: *"The algebra tolerates correlation" ≠ "correlation provides
  generalization."* Tolerating correlation in the binder is necessary but not sufficient; the codes must CARRY
  the right similarity structure.

### 1b. What exists on the PERCEPTION side (does NOT generalize) — the precise gap

The navigation perception's grounded object codes are **flat-distinct (orthogonal), with no semantic
similarity structure**, for two concrete, verified reasons:

1. **The objects are rendered as orthogonal bands, bypassing any visual-feature sharing.** The grounded-compose
   probes drive each object into `cortex_it` via `orthogonal_drive_pattern(cue_idx=obj, ...)` — disjoint,
   maximally-separable bands by construction. Verified: `funcint_perception_to_memory_probe.py:101-106, 259-267`
   (the `PERCEPT_SPARSITY` orthogonal render) and `_step3_live_cortex_grounded_compose_probe.py:88-106` (the live
   rate read uses the same orthogonal render). The grounding is then a **fixed RANDOM complex projection** of the
   rate vector (`_step3_..._probe.py:58-67`, `_step3_grounded_codes_production_composer_derisk.py:51-53`) — a
   random projection of orthogonal inputs yields orthogonal outputs, so the phasor codes carry no semantic
   similarity. The grounded-compose finding itself flags this honestly:
   `2026-06-16-step3-live-cortex-grounded-compose-cheap-first.md:105-110` ("the percepts here are orthogonal …
   FAITHFUL to how the navigation perception actually renders objects (flat-distinct) … the
   semantically-CORRELATED regime … is the separate, deferred frontier").
2. **The deployed visual hierarchy carries POSITION, not OBJECT-CATEGORY, structure — because the gridworld has
   no visually-distinct objects.** The real nav stack does have a Gabor → V1 → V2 → IT pipeline with V2/IT
   plastic (`g11_bg_runner.py:2465-2580`; `sim/visual_cortex.py`), but the gridworld renders only the agent and
   goal as bright/dim blocks at positions (`visual_cortex.py:155-207` `render_gridworld_to_image`). There are no
   "dog" / "cat" / "apple" object shapes with shared visual features, so `cortex_it` never had category structure
   to learn — `g11_bg_runner.py:8244` even notes `cortex_it` "was position-INVARIANT + inactive." The object
   *identity* used by the compose arc is injected by the orthogonal render, not learned from shared features.

**The exact gap (one sentence):** the two cortices use **different, unlinked codes** — the conversation cortex
has correlated, generalizing PPMI codes (learned from co-occurrence), while the perception cortex has
flat-distinct orthogonal codes (rendered, no shared-feature structure) — so (a) the agent cannot generalize
across *perceived* similar objects, and (b) a perceived "apple" and the heard word "apple" do not converge on
one concept code, so perception inherits none of the conversation cortex's generalization.

### 1c. The decisive prior (confirmed, not re-derived)

The 2026-06-11 fork framed generalization as needing the **deferred dendritic substrate rewrite** ("decorrelate
the correlated codes → Mikulasch-Priesemann analog/pre-spike whitening" — `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md:66-73`).
**That framing was superseded by CYCLE 88** (verified above): generalization needs *local normalization*
(achievable on point neurons), not cross-neuron decorrelation; the dendritic rewrite is **not** required for a
generalizing cortex. CLAUDE.md's 2026-06-15 UPDATE records exactly this supersession. This doc **confirms and
extends** that prior to the perception side.

---

## 2. Ranked, biologically-grounded options

Ranking criterion: leverage toward "a proper brain analogue that generalizes across similar concepts" × cheapness
× reuse of validated machinery × biological fidelity.

### Option A (TOP) — Cross-modal UNIFICATION onto shared concept codes via Hebbian convergence (the deepest win)

**The idea.** Make a perceived object's grounded code and the word's PPMI code **converge on ONE concept code**,
by co-activating perception and the conversation cortex when the agent perceives object X *while* the word "X"
streams, and letting Hebbian plasticity bind them into a shared multimodal assembly. Perceiving an apple and
hearing "apple" then drive the same concept neurons → perception **inherits the conversation cortex's
correlated, generalizing codes for free**, AND you get true multimodal grounding (the project's stated goal).

**The biology (cited).**
- **Hub-and-spoke / convergence-zone model (Patterson & Lambon Ralph; Damasio).** Modality-specific "spokes"
  (vision in IT, sound, action) converge on a **cross-modal hub in the anterior temporal lobe (ATL)** that
  forms **modality-invariant, generalizable concept representations**; the hub is where similar concepts become
  similar (deeper-than-surface conceptual similarity). Literature: ATL hub-and-spoke + convergence-zone evidence
  (§Literature [2]). Catalog: this is the role of **E.12 ventral "what" stream → IT** (object/category cells,
  Kandel 6e Ch 24) feeding an association hub; **E.22 Multisensory integration** (Kandel 6e Ch 17/25 — bimodal
  convergence) is the catalog's convergence-zone primitive. *(Both are catalogued `Sim status: missing` — i.e.
  this is genuinely new build, but the primitive — a convergence region that pools two modalities — is small.)*
- **The mechanism is Hebbian co-activation — and there is a published SPIKING precedent.** Pulvermüller &
  Garagnani built **"A Neurobiologically Constrained Cortex Model of Semantic Grounding With Spiking Neurons and
  Brain-Like Connectivity"** (Frontiers Comput. Neurosci. 2018): spike-driven Hebbian plasticity across
  frontal/temporal/occipital areas spontaneously forms **distributed, stimulus-specific cross-modal cell
  assemblies** that ground words in action + perception (§Literature [4]). This is exactly the substrate and
  rule the project already uses (rate-Hebbian co-activation on a `SimulationBridge`). The general principle —
  "fire together, wire together" across co-occurring unisensory areas → multimodal concept neurons — is the
  established account of cross-modal binding (§Literature [5]); "Cortical circuits for cross-modal generalization"
  (2025) makes the generalization payoff explicit (§Literature [5a]).

**What it buys.** (1) Generalization on the perception side **for free** (it is the conversation cortex's code).
(2) True multimodal grounding (perceive-apple ≡ hear-apple), squarely the project's actual goal. (3) Reuses the
already-validated population-Hebbian co-occurrence machinery — the convergence is *the same rule applied across
modalities*. (4) The no-confab moat + binder are unchanged (they ride whatever codes arrive; the codes just
become shared).

**The risk.** (1) The perception code must be *good enough* to anchor the convergence — if the rendered object
code is pure orthogonal noise, Hebbian convergence onto the PPMI code still works (the PPMI side carries the
similarity), but the *transfer to a novel perceived object* needs the perception code to itself carry
shared-feature similarity (→ Option B is the natural prerequisite/partner for *novel-perceived* generalization).
(2) Catastrophic interference when binding modalities on one bridge — but the project has the per-synapse
plasticity gate + the merged-bridge isolation results (de-risk 5a) to manage it. (3) Honest scope: a 2-modality
convergence hub is the minimal version; full hub-and-spoke (≥3 spokes) is a follow-on.

**Why #1:** it is the single move that (a) gives generalization, (b) gives multimodal grounding, (c) reuses the
most validated machinery, and (d) has a direct published spiking precedent. It is also *cheap to falsify* (§4).

### Option B — Perception-side similarity-preserving codes via SHARED VISUAL FEATURES (the perception analogue of PPMI)

**The idea.** Make `cortex_it` carry **semantic similarity directly**: similar objects (dog, cat) share visual
features → overlapping IT ensembles → similar codes → the agent generalizes across *perceived* similar objects,
exactly as PPMI does for words. The mechanism is **not** a decorrelator; it is the *natural* output of a visual
hierarchy with **shared low-level features + local normalization**, the perception-side mirror of PPMI's local
normalization.

**The biology (cited, strongly supportive).**
- **IT object representations are dominated by SHARED VISUAL FEATURES / shape similarity** — this is the
  decisive literature finding. Op de Beeck et al.; Kiani et al. 2007; Kriegeskorte RSA: the structure of monkey
  & human IT population responses is accounted for by **shape / low-level visual similarity** (visually similar
  objects → similar IT codes), with semantic category largely *explained by* visual similarity (§Literature [1]).
  ⇒ a similarity-preserving perception code arises **automatically** from shared visual features; **no
  decorrelation, no whitening, no dendrite is needed** to *produce* similarity (the opposite problem from the
  binder). Catalog: **E.12** (IT category cells, Kandel 6e Ch 24); the hierarchy that produces shared features is
  Gabor V1 (**E.08/E.09**) → V2 → IT, *which the project already has* (`sim/visual_cortex.py`,
  `g11_bg_runner.py:2465-2580`).
- **The olfactory combinatorial code (E.19) is the project's catalogued similarity-preserving precedent:** odors
  with shared chemical features activate overlapping glomerular maps → similar odors → similar codes (Kandel 6e
  Ch 29). It is the canonical "shared features → overlapping ensembles → similarity-preserving code" motif, and
  it is in the catalog (`E.19`, `Sim status: missing`).

**What it buys.** Generalization across *perceived* similar objects (the literal question-1 ask). Pairs perfectly
with Option A: B gives perception its own similarity structure; A then fuses it with the word's.

**The risk.** The gridworld has no visually-similar objects (verified §1b) — so realizing this *in the live nav
task* needs either (i) a richer object-rendering environment (objects with shared visual features), or (ii) a
**learned similarity-preserving projection** that maps the rendered object onto a code carrying a *given*
similarity structure (a "perception PPMI": drive `cortex_it` so that objects sharing features/contexts get
overlapping codes — the same local-normalization trick, sourced from a feature-overlap matrix instead of a
word-co-occurrence matrix). Option (ii) is the cheap, environment-free version and is what §4 de-risks. Risk:
if the object-feature overlap is hand-imposed, it must be justified as *sensory rendering* (shared pixels are the
environment's), not smuggled-in semantics — the anti-cheat in §4 controls this.

**Why #2:** it directly answers question 1 and is biologically the *cleanest* (similarity is the natural output
of the existing visual hierarchy), but its *live-task* payoff is gated on object diversity in the environment,
whereas Option A delivers generalization immediately by inheriting the conversation cortex.

### Option C — A learned similarity-preserving cross-modal PROJECTION (perception rate → shared PPMI code)

**The idea.** Keep the perception front end flat-distinct, but **replace the fixed random grounding projection
with a LEARNED projection** that maps each object's `cortex_it` rate code onto **the conversation cortex's PPMI
code for that word**. Train it with the (perceived-object, heard-word) pairs the agent experiences — i.e., learn
the perception→concept map by Hebbian/regression onto the already-generalizing PPMI target.

**The biology (cited).** This is the *learned* version of the convergence zone: the hub's "variable and arbitrary
mappings of features into coherent concepts" (hub-and-spoke, §Literature [2]) realized as a learned
spoke→hub map. word2vec = PMI factorization (Levy & Goldberg 2014, §Literature [3]) confirms the PPMI target is
the right concept geometry to project onto. Mechanistically it is a learned read-out / labeled-line, which the
project builds routinely (concept-pool topographic maps; the `cortex_it → language_output` read-out).

**What it buys.** Generalization on the perception side *without changing the environment*: a novel-but-similar
perceived object lands near its neighbours in PPMI space because the learned map preserves the rate-code
neighbourhood → PPMI neighbourhood relation. It is a strict upgrade of today's fixed random projection.

**The risk.** It only generalizes if the *input* rate code already has *some* neighbourhood structure shared with
the PPMI target (garbage-in if the render is pure orthogonal). So C is strongest *with* B (B gives the input
structure) or *with* A (A supplies the target by co-activation). Standalone, C is a learned grounding upgrade, not
a generalization guarantee. **Honest:** C overlaps heavily with A — A is the *unsupervised Hebbian* convergence,
C is the *supervised/regression* convergence onto an explicit target. Prefer A (more biological); keep C as the
fallback if unsupervised convergence is unstable.

### Option D (deprioritized) — the deferred DENDRITIC substrate rewrite

**Assessment: NOT required for this frontier.** The 2026-06-11 fork put generalization here, but CYCLE 88
superseded that (verified §1c): generalization comes from **local feedforward normalization** (PPMI; achievable
on point neurons), and **decorrelation/whitening would destroy generalization**. The dendritic rewrite's job
(analog/pre-spike whitening = *decorrelation*) is the *opposite* of what generalization needs. The dendritic
arc remains a legitimate long-horizon fidelity project (apical/basal compartments, sub-threshold computation),
but it is **not on the critical path** to generalization-across-similar-concepts. Decisive call in §5.

---

## 3. Reusable project machinery (this is mostly assembly of validated parts)

| Piece | File / function | Status |
|---|---|---|
| **PPMI / generalizing codes (numpy)** | `research/runners/_phaseB_online_stream_cortex_derisk.py:80-105` (online Hebbian co-occurrence + log-double-centre + `heldout_generalization`) | validated, gen 0.91 (3 seeds) |
| **On-bridge population-Hebbian co-occurrence** | `research/runners/_phaseB_stdp_cooccurrence_derisk.py` (rate-Hebbian; `--n-per` population; reads the learned `hub→target` block from `cp_connections`) | on-substrate GO; pop lift to 100–108% host |
| **The grounded-code map (perception rate → composer phasor)** | `research/runners/_step3_live_cortex_grounded_compose_probe.py:58-67` (`_projection` / `_to_phasor`); live rate read `:81-106` | validated (flat-distinct); **the random projection is the piece Options B/C replace with a similarity-preserving / learned one** |
| **The perception bridge (`cortex_it` + read-out)** | `research/runners/funcint_perception_to_memory_probe.py:140-250` (`build_probe_bridge`) | validated cheap CPU substrate |
| **The production composer + grounding INTERFACE** | `research/runners/rf_phasor_composer.py:62-89` (`RFPhasorComposer(..., grounded_codes=...)`) — already accepts `{word: phases}`; docstring states "meaningful grounded codes … the open problem" | the drop-in point; grounded codes already verified `== random at parity` 6-seed (`_step3_grounded_codes_production_composer_derisk.py`) |
| **The visual hierarchy (Gabor→V1→V2→IT)** | `sim/visual_cortex.py` (Gabor RFs, gridworld render); `g11_bg_runner.py:2465-2580` (V2/IT plastic pathways) | exists; carries position (no object diversity) — the Option-B front end |
| **Co-resident composer on the one bridge** | `research/runners/nav_conv_merged_bridge.py` (`MergedRFComposer`, masked RF ops) + `2026-06-10-step2b-...COMPLETE.md` | nav+conv+composer co-resident, capability-equivalent |
| **Plasticity isolation for multimodal binding on one bridge** | per-synapse `cp_plasticity_rate_gain` gate; merge de-risk 5a (`2026-06-10-unification-5a-...PASS-with-clip-caveat.md`) | the tool to bind modalities without interference |
| **No-confab familiarity gate** | learned Bogacz-Brown gate (`2026-06-11-familiarity-gate-v320-GO.md`); rides any codes | the moat that must survive |

The headline: **Options A/B/C are reconfigurations of existing, validated pieces** — the genuinely-new content is
(A) co-activating the two cortices and committing a shared assembly, or (B/C) swapping the *fixed random*
grounding projection for a *similarity-preserving / learned* one. No new mechanism class.

---

## 4. Recommended CHEAP-FIRST de-risk (smallest CPU/numpy probe; numeric GATE; anti-cheats)

**Target the TOP option (A), in its cheapest numpy form**, because it is the highest-leverage and is falsifiable
without GPU. The probe answers one question: *does Hebbian co-activation of a (flat-distinct perception code) and
a (correlated PPMI word code) yield a shared concept code that GENERALIZES across similar concepts on the
perception channel — i.e., perceiving a NOVEL object whose word is similar to a trained one partially recalls the
trained associate?*

**Probe (`_genfrontier_crossmodal_unify_derisk.py`, numpy, ~minutes, NO sim/ edit):**
1. Take F=16 concepts in 4 semantic categories. **Word side:** their real PPMI codes (reuse
   `_phaseB_online_stream_cortex_derisk` to produce the 16 codes; correlated, generalizing — verified to carry
   category structure). **Perception side:** flat-distinct orthogonal object codes (the current nav regime).
2. **Cross-modal convergence:** learn a Hebbian/linear map `W: perception_code → concept_code` on a TRAIN split
   of concepts, where the concept_code is the PPMI (word) code (the hub target). This is Option A's convergence
   in its simplest form (the unsupervised-vs-supervised distinction is a follow-on; for the cheap gate, learn the
   map on the co-activation statistics).
3. **The generalization test (the load-bearing measurement):** for a HELD-OUT concept (never used to fit `W`),
   present *only its perception code*, map through `W`, and ask whether the resulting concept code is **nearest
   its own PPMI code AND closer to its same-category neighbours than to other-category concepts** (the
   similarity-transfer signature). Score = held-out same-category neighbour rank / cosine margin.

**GATE (multi-seed 42/43/44):**
- **GO** if held-out concepts map to a code that (i) is nearest their own PPMI code at ≥ 0.80 (vs a flat-code
  baseline ≈ chance 1/16 = 0.0625), **AND** (ii) same-category mean cosine exceeds other-category mean cosine by
  a margin ≥ 0.15 (the generalization signature — the perception channel inherits the word cortex's similarity),
  **AND** (iii) the no-confab moat survives: an absent/novel concept whose word was never streamed abstains
  (familiarity gap present ≫ absent, gate at the midpoint, 0 false-accepts).
- **PARTIAL** if (i) holds but (ii) is weak (convergence works but similarity doesn't transfer — points to
  Option B being needed: the perception input itself must carry shared-feature structure).
- **NEGATIVE** if held-out mapping ≈ chance (cross-modal convergence does not generalize on flat-distinct
  perception input → Option B is a prerequisite, not optional).

**Anti-cheat controls (mandatory, run as part of acceptance):**
1. **Held-out generalization vs a MEMORIZATION / flat-code baseline.** The flat-distinct baseline (perception code
   with no shared structure, no convergence) must score at chance on held-out same-category transfer. A
   lookup-table memorizer of TRAIN concepts must score 0 on held-out (no entry). The discriminating evidence is
   the **gap** (the same control structure as `_step3_live_cortex_grounded_compose_probe.py:127-160`).
2. **No-leakage split.** The held-out concepts are excluded from fitting `W` (assert `train ∩ held-out = ∅`); the
   category labels used for the *scoring* must come from the PPMI/word side (the legitimate semantics), not be
   injected into the perception code (which would manufacture the similarity). Assert the perception codes are
   orthogonal (between-cos ≈ 0) *before* convergence — so any transferred similarity comes from the word cortex,
   not a pre-seeded perception structure (this is the "is it sensory rendering or smuggled semantics?" control).
3. **The no-confab moat must survive** (gate (iii) above) — never weaken it; a moat breach is a hard stop.
4. **Permuted-label control:** shuffle the (perception, word) pairing → held-out transfer must collapse to chance
   (proves the convergence is learned, not a coincidence of the projection geometry).

**Why this is the right cheap gate:** it isolates the single new claim (cross-modal Hebbian convergence transfers
the word cortex's *generalization* to the perception channel) on CPU in minutes, with the project's standard
held-out-vs-memorization + no-leakage + moat anti-cheats. A GO greenlights the GPU build (co-activate the two
cortices on the merged bridge, commit shared assemblies, validate the who/what matrix on *perceived* novel
similar objects). A PARTIAL/NEGATIVE cleanly routes to Option B (the perception input needs its own
similarity-preserving structure first) — a sharp, characterized next step, not a dead end.

---

## 5. Honest scope + is the dendritic rewrite required? (decisive call)

**Decisive call: NO — the dendritic substrate rewrite is NOT required for generalization-across-similar-concepts,
and the project's strong prior is CONFIRMED (and extended to the perception side).**

Reasoning, three independent legs:

1. **The conversation side already generalizes on point neurons** (PPMI stream cortex, gen 0.86–0.91, realized
   on the spiking substrate with population coding) — verified, multi-seed, on-bridge. Generalization is an
   *existence proof* on the point-neuron substrate today; the dendritic rewrite is not a precondition for it.
2. **Generalization needs LOCAL normalization, not decorrelation** (CYCLE 88, verified). The dendritic rewrite's
   purpose — analog/pre-spike *whitening* = *decorrelation* (Mikulasch-Priesemann) — is the **opposite** of what
   generalization needs (whitening *destroys* the similarity structure that generalization rides). So the
   dendritic rewrite is not merely unnecessary here; pointing it at this frontier would be counterproductive.
3. **The literature says perception-side similarity is the NATURAL output of shared features + a feedforward
   hierarchy** (IT is dominated by shape/visual-feature similarity — §Literature [1]), and **cross-modal
   unification is a Hebbian convergence-zone computation with a published SPIKING precedent** (Pulvermüller-
   Garagnani — §Literature [4]). Both are point-neuron, feedforward/Hebbian mechanisms the project already runs.
   Neither requires sub-threshold dendritic computation.

**What IS genuinely required (the real scope):**
- The **cross-modal convergence build** (Option A): co-activate perception + conversation cortex, commit a shared
  multimodal assembly via Hebbian learning, on the merged bridge. New build, but small and reuse-heavy.
- For generalization across *novel perceived* objects (not just inheriting via a known word), the **perception
  code must itself carry shared-feature similarity** (Option B) — which in the current gridworld needs either a
  richer object-rendering environment OR a learned similarity-preserving projection (Option C). This is the one
  place where "the environment has no visually-similar objects" is a real limiter — but it is an *environment /
  rendering* limiter, not a *substrate* limiter, and is solved by adding object diversity (legitimate sensory
  rendering) or learning the map, not by a dendritic rewrite.
- **The dendritic rewrite stays a deliberate, separate, long-horizon FIDELITY arc** (it would make the substrate
  more biologically faithful in general), but it is **off the critical path** to this frontier. That keeps it an
  owner call, made for fidelity reasons, not blocked-on for generalization.

**One honest caveat I will not over-sell:** the cheap probe (§4) tests cross-modal convergence + similarity
*transfer* in numpy at F=16. The on-bridge realization (population Hebbian convergence across two real cortices,
the interference management via the plasticity gate, and scaling to 320 concepts) is build/engineering with
known tools, but it has not been run — the GO from §4 is the trigger to commit those GPU resources, per the
standing "present before building" practice.

---

## Literature (load-bearing sources, verified by search this session)

- **[1] IT object representations dominated by shared visual features / shape similarity** (the basis for a
  natural similarity-preserving perception code; Option B): Op de Beeck/Kriegeskorte RSA; Kiani et al. 2007
  ("Object Category Structure in Response Patterns of Neuronal Population in Monkey IT"). Visual/shape similarity
  accounts for IT structure; semantic category largely explained by visual similarity.
  - https://pubmed.ncbi.nlm.nih.gov/26493748/ (visual features as stepping stones toward semantics)
  - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003167 (shape similarity > semantic membership in IT)
  - https://journals.physiology.org/doi/abs/10.1152/jn.00024.2007 (Kiani 2007, object category structure in IT population)
- **[2] ATL semantic hub / hub-and-spoke / convergence zones** (the cross-modal unification target; Option A):
  Patterson & Lambon Ralph hub-and-spoke; Damasio convergence zones. The ATL hub forms modality-invariant,
  generalizable concepts by integrating modality-specific spokes.
  - https://www.sciencedirect.com/science/article/pii/S0010945219302527 (structural connectivity convergence zone in ventral/anterior temporal lobe)
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3884130/ (ATL critical for acquiring new conceptual knowledge; feature integration)
- **[3] word2vec = PMI factorization** (confirms the PPMI cortex is the cortical form of distributional semantics;
  Option C target geometry): Levy & Goldberg 2014, "Neural Word Embedding as Implicit Matrix Factorization."
  - https://lovit.github.io/nlp/2018/04/22/context_vector_for_word_similarity/ (review of Levy & Goldberg)
  - https://www.ruder.io/secret-word2vec/ (word2vec implicitly factorizes shifted PMI)
- **[4] Spiking cross-modal semantic grounding (the substrate + rule precedent for Option A):** Garagnani &
  Pulvermüller, "A Neurobiologically Constrained Cortex Model of Semantic Grounding With Spiking Neurons and
  Brain-Like Connectivity," Front. Comput. Neurosci. 2018 — spike-driven Hebbian plasticity forms distributed,
  stimulus-specific cross-modal cell assemblies grounding words in action + perception.
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6232424/
  - https://link.springer.com/article/10.1007/s12559-009-9011-1 (recruitment/consolidation of word cell assemblies by Hebbian learning + competition)
- **[5] Cross-modal binding by Hebbian co-activation → multimodal concept neurons** (the general mechanism):
  multisensory "fire together, wire together"; co-activated unisensory areas form stronger cross-modal
  connections → multimodal neurons.
  - https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1181760/full (crossmodal interactions in learning/memory)
  - **[5a]** https://pmc.ncbi.nlm.nih.gov/articles/PMC12106601/ ("Cortical circuits for cross-modal generalization," 2025 — the generalization payoff)

## Catalog hooks (sim-catalog/references/feature-catalog.md — verified)

- **E.12** Ventral "what" stream → IT (object/category cells, viewpoint-invariant; Kandel 6e Ch 24) — the
  perception-side category code (Option B front end). `Sim status: missing` (the hierarchy exists in code but
  carries position, not category, in the gridworld).
- **E.19** Olfactory glomerular map & combinatorial code (Kandel 6e Ch 29) — the catalogued **similarity-
  preserving** precedent (shared chemical features → overlapping glomeruli → similar codes). The cleanest
  biological model for "shared features → similarity-preserving code." `Sim status: missing`.
- **E.22** Multisensory integration (bimodal convergence; Kandel 6e Ch 17/25) — the catalog's convergence-zone
  primitive (Option A). `Sim status: missing`.
- **E.08/E.09** V1 simple/complex (Gabor; the shared-feature front end that already exists in `sim/visual_cortex.py`).
- **D.14** Engram tagging (Tonegawa) — the mechanism that already binds perceived ensembles (the perception→memory
  recall milestone); the shared-assembly commit in Option A is its cross-modal extension.
- (No dedicated "semantic hub / convergence zone" catalog entry exists — a genuine catalog gap; E.12+E.19+E.22 are
  the assembling primitives. Worth adding a Cluster-E/G "semantic hub (ATL)" entry as a follow-on.)
