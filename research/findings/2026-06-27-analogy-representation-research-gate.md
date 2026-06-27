# Analogy / representation research gate — is proportional analogy unlockable on THIS substrate without a months-arc?

**Date:** 2026-06-27
**Type:** READ-ONLY deep-research gate (standing practice: deep research + catalog + Kandel + literature review BEFORE committing build/GPU/`sim/` effort to overcome a confirmed boundary). **No code, no composer/`sim` edit** — the de-risk/build is a separate later step that this doc gates.
**Trigger:** the gate fired on a confirmed boundary (`2026-06-27-tier2.1-analogy-NEGATIVE.md`: NO-GO) **and** a known representation-geometry family (the 2026-06-11 cortex-fork "correlated codes can't bind", the 2026-06-15 "decorrelation red herring", the 2026-06-16 learned-binder bundling-NEGATIVE). Conditions (a) confirmed boundary + (b) known family + (d) new-mechanism-class → the gate is mandatory.

> **Reader caveat — this is the project's MOST-BURNED capability.** The 2026-05-14 "transitive inference 90% multi-seed" was an architecture-mismatch artifact retracted to ~chance. The 2026-06-27 analogy NO-GO is itself a *correctly-caught* false-GO (v1 control too weak → acc 1.000; sharpened control → the true 0.000). Every recommendation below therefore carries the mandatory permuted-relation / held-out / lesion / moat controls, and the verdict is stated plainly as unlockable-cheaply vs genuine-months-frontier with an explicit falsification bar.

---

## 0. One-paragraph answer

The analogy *mechanism* (`unbind → transform → apply → cleanup`, Komer-Stewart / Eliasmith-SPA) is **sound and runs on the real RF spiking substrate** (acc 1.000 through the spiking `_bind`); the boundary is **purely representational**. The genuine residual is **one specific, measurable thing**: the production codes carry **first-order similarity** (cat–dog +0.43) but **no shared additive second-order relational offset** (delta-alignment over 37,950 disjoint pair-pairs = **+0.0015 ≈ random**). The decisive question the brief poses — *would binding the relation as a FACTORED typed-role slot (reusing the ArgStructureComposer's additive roles) supply the additive structure?* — **answers cleanly: a factored relation slot makes analogy TRIVIAL but only on relations the agent has been TOLD (an explicit `RELATION=is_a` slot is itself a clean phasor, so `B−A` recovers the relation phasor exactly), and it does NOT manufacture analogy over the RAW concept-code geometry where the relation is implicit (king−man≈queen−woman over learned codes).** These are two different capabilities. The biology agrees: the brain does relational reasoning by **factorising structure from content** (Whittington-Behrens Tolman-Eichenbaum Machine: medial-EC = a structural basis, hippocampus = content-conjunction; Eichenbaum-Cohen D.02 relational memory; rLPFC relational integration) — i.e. it makes the relation an **explicit factored axis**, it does NOT read analogy off raw similarity codes. **Verdict: the FACTORED-RELATION form of analogy is unlockable cheaply (days, reuse-by-import, NO `sim/` edit); the RAW-CODE-GEOMETRY form (analogy emerging on learned concept codes the way it does in large word2vec/GloVe) is a genuine corpus-scale / months frontier and is gated on the deep-knowledge build, not a clever circuit.** The single cheapest decisive de-risk is below (§4).

---

## 1. ISOLATE the genuine residual (exactly what is missing, how big, what is already present)

### 1.1 The residual, stated precisely

Proportional analogy A:B::C:D on a VSA works iff the representation makes the **transform** `T = B ⊗ A⁻¹` a *clean, shared* operator that, applied to C, lands on D. On the FHRR phasor composer (info in phase) bind = phase-ADD, unbind = phase-SUBTRACT, so:

```
T_phase = (B − A)                  # the relation A→B as a phase offset
rec     = apply(T, C) = (C + B − A) # the prediction
D       = argmax_w cos(rec, code_w) over the codebook, operands excluded
```

This is exactly `king − man + woman = queen`. It requires **`B − A ≈ D − C`** as phase vectors — a **second-order** property (a *difference of differences* being small). The production codes have:

| property the codes HAVE | property analogy NEEDS | measured |
|---|---|---|
| first-order similarity (cat–dog +0.426, bird–cat +0.408) | second-order shared delta (`B−A ≈ D−C` across distinct pairs) | **delta-alignment +0.0015 over 37,950 disjoint pair-pairs ≈ random** |
| bundled SVO fact codes (`_bundle(_bind(role,filler)…)`) | a clean extractable transform from the code | bundle cross-terms swamp `B⊗A⁻¹` → near-identity → returns C, acc **0.000** |

**Size of the gap on RAW codes:** the held-out analogy accuracy is **0.000, BELOW the 0.267 memorization floor** (chance). It is not "small and almost there" — on raw learned/bundled codes the relational geometry is **absent**, not weak. (Contrast: on hand-built ADDITIVE relational codes the same mechanism is **1.000**, and the lesion + permuted controls both collapse to the 0.267 floor — proving the mechanism is correct and the boundary is the geometry.)

### 1.2 Is ANY of it already present? — the factored-relation question (the crux)

The brief asks whether the **ArgStructureComposer's** additive typed roles (`GOAL/RECIPIENT/THEME/LOCATION/SOURCE/INSTRUMENT/TIME`, `argstructure_composer.py:49`) could supply the additive structure so analogy runs over the **relation-role** rather than raw concept arithmetic. Reading the code settles it:

- The composer binds **role-label ⊗ filler** where the filler is a **concept code** (`_encode`: `_bind(self.roles[r], self._filler_phases(fact[r]))`, `rf_phasor_composer.py:261`; `query_role` binds `roles[r]` to `_filler_phases(fact[r])`, `argstructure_composer.py:187`). The typed roles are **clean random phasors** drawn from a disjoint stream (`prng.uniform(0,1,D)` per role, `argstructure_composer.py:172`).
- So an **explicit relation slot is ALREADY a clean additive primitive in this architecture.** If a fact is stored as `{SUBJ:king, RELATION:gender_to, OBJ:man}` and the relation `gender_to` is one of these role-phasors, then **the relation IS a clean phasor by construction** — `unbind(fact, SUBJ)` etc. recover it exactly, and analogy over the relation-role is trivial phase arithmetic (precisely the ADDITIVE-codes 1.000 case). **The clean transform analogy needs already exists for any relation that is an explicit slot.**

**But this does NOT solve the boundary as posed, because of WHICH relations the agent has.** Two regimes:

| regime | is the relation an explicit factored phasor? | does analogy work? | is it "real analogy"? |
|---|---|---|---|
| **A — TOLD relation** (`RELATION=is_a` stored as a slot) | YES (a clean role-phasor) | YES, trivially (= ADDITIVE 1.000) | partly — the agent must be GIVEN the relation; it does not *discover* it from the concepts. Genuinely useful for **stored relational facts** (taxonomy, kinship). |
| **B — IMPLICIT relation in concept codes** (king:man::queen:woman, where "man-ness" lives inside the learned `king`/`man` codes) | NO (the relation is a latent direction in correlated codes, not a slot) | NO (acc 0.000 — the residual) | this is the *fluid-intelligence* analogy — inferring an un-named relation from surface items. |

**⇒ The factored-relation slot supplies the additive structure for regime A (and that is cheap + already-machinery), but it does NOT manufacture regime-B analogy over raw concept codes.** Conflating the two would repeat the 2026-05-14 over-claim: storing `is_a` slots and then "doing analogy" would demo beautifully while testing nothing about the substrate's ability to read relations off learned geometry. The honest framing: **a factored relation slot converts analogy-over-stored-relations into a clean additive lookup (regime A, cheap); analogy-over-learned-similarity-codes (regime B) remains the open frontier.**

### 1.3 Where the production fact representation makes it worse (the BUNDLED problem)

Even in regime A, the *production fact* is a **bundle** (`agent ⊗ a + action ⊗ v + patient ⊗ p`, `_encode`/`_bundle`, `rf_phasor_composer.py:245,261`). Extracting a transform from a bundle is the same superposition-crosstalk wall the 2026-06-16 learned-binder hit (`bundling NEGATIVE 0.193`; a *fixed* ±1 self-inverse bundles 0.989). So regime-A analogy must operate on a **single bound relation pair** (`bind(RELATION, X)`), not on the SVO bundle — exactly the "explicit relation slot, not superposition" caveat the NEGATIVE doc already flagged (its option 1). This is a design constraint, not a new wall: bind one relation at a time, unbind it cleanly.

---

## 2. REFRAME via biology — how does the brain actually do relational reasoning?

The biology is unusually clear and points the **same way as the factored-slot analysis**: the brain does NOT read analogy off raw similarity; it builds an **explicit, factored relational structure** and reasons over *that*.

### 2.1 The Tolman-Eichenbaum Machine (Whittington, Muller, Mark, Chen, Barry, Burgess, Behrens 2020, *Cell*) — the decisive reframe

TEM's central principle is **factorisation**: *factorise the relationships between experiences (structure) from the content of each experience.* "Medial entorhinal cells form a basis describing structural knowledge, and hippocampal cells link this basis with sensory representations." After learning, the structural basis units display **grid, band, border, and object-vector** properties; the conjunction with content gives place-like cells. The payoff is **generalisation of structural knowledge to new situations and transitive inference** — exactly because the *structure* (the relational axis) is represented **separately** from the *content* (the items). This is the biological vindication of "make the relation an explicit factored slot": the EC structural basis IS a learned set of relation/role axes, factored out of content, over which inference is a geometric operation.

### 2.2 Eichenbaum-Cohen relational memory / cognitive map (catalog D.02; Kandel 6e Ch 52 pp 1301–1302)

D.02 ("relational binding / memory space"): the hippocampus "stores events as items-in-context… and networks via overlapping events allowing flexible inference (**e.g., transitive**)." Behavioural validation = **transitive inference; selective deficit on configural learning after dorsal-HC lesion.** Crucially, the catalog's own O'Keefe-Nadel supplement notes the **map** already provides this binding *architecturally* — "novel inferences supported by traversal of the map", "place hypotheses can be tested without reactivating any specific stimulus." Inference is a **traversal of a learned low-dimensional structure**, NOT a similarity read-off. (This is precisely the insight the project *missed* in 2026-05-14 when it tried transitive inference as explicit spreading-activation chaining and got an artifact.)

### 2.3 rLPFC relational integration + the analogy hub (Bunge/Wendelken/Badre; reasoning-cluster doc §2.5)

Rostrolateral PFC (frontopolar, ~BA10) integrates **relations between relations** (second-order relations) — the step that distinguishes analogy from simple relation-matching. It operates on **structured representations held in WM** (the operands as factored role-filler structures), not on raw perceptual similarity. This matches the VSA literature: VSAs do analogy by `unbind → transform → apply` over **role-filler-structured** vectors (Gayler; Eliasmith SPA; Komer-Stewart 2020; Hersche et al. 2023 *Nat Mach Intell* solve Raven's at SOTA by exactly bind/unbind/bundle). The Eliasmith/Rasmussen Raven's solver codes the per-cell transition as a **circular convolution (a bound transform)** between **structured** cell representations and **averages** the transforms into a rule vector — it never reads the rule off raw item similarity.

### 2.4 The reframe verdict

**The right hypothesis is "make the relation an explicit factored axis" (TEM structure/content factorisation; D.02 map; rLPFC over role-filler structure) — which is CHEAP because the composer's role-filler binding already IS a factorisation primitive — NOT "learn relational geometry from a richer corpus so it emerges on raw concept codes" (which is the expensive, corpus-scale, regime-B path).** The biology says the brain factors the relation out; it does not expect analogy to fall out of similarity. This is the single most important steer in this gate: **stop trying to read analogy off the learned codes (regime B); represent relations explicitly (regime A) — and the substrate already supports it.**

A second biological gift for the hard case (transitive/ordinal): TEM/D.02 say the **ordinal/transitive** relation should be a **learned 1-D map geometry** (Euclidean distance = inferred relational distance), which the project's own PPMI/Hebbian co-occurrence code-geometry machinery can learn — this is regime A's natural extension for orderings, and it comes with a *unique* anti-cheat the artifact could not fake (the **symbolic-distance effect**, §3 option d).

---

## 3. RANK cheap-first options

Ranked by (substrate-fit × cheapness × leverage toward genuine relational reasoning × credibility-recovery on the burned front). All are reuse-by-import; none requires the dendritic substrate.

### (a) ⭐ Factored relation slots — analogy over an EXPLICIT relation-role (CHEAPEST; the recommended de-risk)
- **What:** add a `RELATION` (and optionally `SUBJ/OBJ`) role to the composer's alphabet (or reuse an ArgStructureComposer typed role), store relational facts as a **single bound pair per relation** (`bind(RELATION, gender_to)` etc., NOT in the SVO bundle), and run `unbind → transform → apply → cleanup` over the **relation-role**. Because the relation is then a clean phasor, this is the ADDITIVE-codes case (proven 1.000 with collapsing controls).
- **Reusable machinery:** `RFPhasorComposer._bind/_bundle/unbind/_cleanup` (verbatim); `ArgStructureComposer` typed-role pattern + disjoint-stream role codes (`argstructure_composer.py:49,172,187`); the no-confab moat; the anti-cheat tooling (`v16_compose_permuted_check.py`, `_genfrontier_*_derisk.py`).
- **Cost:** ~days; numpy de-risk first (CPU), then through the real RF `_bind` (the NEGATIVE doc already confirmed the spiking op carries it). NO `sim/` edit.
- **Payoff:** a **real, anti-cheatable relational-reasoning capability** for *stored* relations (taxonomy "is robin warm-blooded?" via `is_a`+`has`; kinship; part-of) — and a published-mechanism redemption of "reasoning" under proper controls. **Honest scope it buys:** regime A only (TOLD relations). It does **not** claim regime-B fluid analogy.

### (b) A small learned relational projection over the concept-code space (MEDIUM; likely fragile)
- **What:** learn a linear/MLP map `R: code-space → code-space` per relation (RLPFC-style relational integration) from a handful of exemplar pairs, then apply to C.
- **Reusable machinery:** the project's bilinear/learned-binder harness + leakage-free systematicity protocol.
- **Cost:** ~1–2 weeks. **Risk flag (from the project's OWN evidence):** the 2026-06-16 learned-*linear*-inverse was NEGATIVE (a learned linear map cannot be a reciprocal; 0.056, broke even single-attribute). A learned relational *projection* (not an inverse) is less doomed, but the project has repeatedly found learned linear maps over correlated codes fragile. **Expected payoff: LOW-confidence.** Not recommended ahead of (a).

### (c) Richer-corpus relational geometry — analogy emerges on learned codes (regime B; EXPENSIVE; the genuine frontier)
- **What:** scale the corpus so the learned PPMI/stream codes acquire real second-order relational geometry (king−man≈queen−woman), the way large word2vec/GloVe do.
- **Reusable machinery:** the PPMI online-stream cortex (CYCLE 88–96) + the owner's deep-knowledge/breadth build.
- **Cost:** **months / corpus-scale** (this IS the deep-knowledge arc, not a side-build). **Honest uncertainty:** the +0.0015 delta-alignment is at the *current* (≤25–64 word developmental) vocab; whether relational geometry emerges at 10K–40K-word scale on THIS local-normalization cortex is **genuinely unknown** (word2vec gets it at billions of tokens; PPMI is a different, more-local objective). **Payoff: high if it works, but it is a research bet gated on the corpus build, orthogonal to the composer mechanism (which is ready when the codes are).**

### (d) Cognitive-map / TEM ordinal geometry — the transitive/ordinal case (MEDIUM; the redemption build for transitive inference)
- **What:** learn a 1-D/N-D ordinal **map** from adjacent trained pairs (A>B, B>C…) so item positions encode the order; infer B vs D by **comparing map positions**, not chaining (TEM factorisation; Park 2020; Garvert/Behrens 2016 *eLife*; D.02).
- **Reusable machinery:** PPMI/Hebbian code-geometry learning; cleanup/attractor read-out.
- **Cost:** ~2–3 weeks. **Unique anti-cheat the 2026-05-14 artifact could NOT fake:** the **symbolic-distance effect** (accuracy/latency grades with map distance — |B−D| easier than |B−C|) + held-out inferred pairs + permuted-order rank-1/N! + lesion. **Payoff: directly converts the most-burned retraction into a clean, biologically-correct GO.** Recommended as the *second* build after (a), if transitive/ordinal reasoning is prioritised.

---

## 4. VERDICT — the single cheapest decisive de-risk + anti-cheats + falsification bar

### The recommendation

**Run option (a): the factored-relation-slot analogy de-risk.** It is the single cheapest test that tells us whether relational reasoning is unlockable on THIS point-neuron substrate WITHOUT a months-arc — because it isolates the *one* thing in question (does an explicit factored relation supply the additive transform the mechanism needs) from the *one* thing that is genuinely expensive (does relational geometry emerge on raw learned codes). It reuses validated machinery, needs NO `sim/` edit, and runs cheap-first numpy → real RF spiking.

### The de-risk protocol (read-only spec; build is the separate later step)

1. Define a tiny **explicit-relation** fact set: store relational pairs as single bound pairs `bind(RELATION_k, value)` over a few relations (e.g. `gender_to`, `is_a`, a kinship/ordinal relation), with the relations as clean role-phasors (reuse `ArgStructureComposer`'s disjoint-stream role codes).
2. Run `T = unbind(B, A)` (the relation transform) → `rec = apply(T, C)` → `D = cleanup(rec)` over the relation-role, operands excluded from cleanup.
3. Hold out the (C,D) pair used to *score* — the transform is built from a *different* pair of the same relation.
4. Confirm **through the real RF `_bind`/`_resonate` op** (not just host phase arithmetic) on ≥1 seed, matching the numpy reference (the NEGATIVE doc's spiking-faithfulness check pattern).

### Anti-cheats (ALL mandatory — this is the burned-capability bar; any failure → STOP, write the honest NEGATIVE, do not over-claim)

- **(i) held-out ≫ memorization floor.** The scored (C,D) relation pair is NEVER used to build the transform; operands excluded from cleanup. Held-out acc must be **≫ the lookup floor** (the floor = best-constant / nearest-C-neighbour baseline, ~0.267 in the NEGATIVE doc's harness). **Falsification: held-out ≤ floor → NO-GO** (exactly the 2026-06-27 gate that already fired on raw codes).
- **(ii) permuted-relation collapses.** Shuffle which value the relation maps to (or shuffle the relation→pair assignment). The TRUE relation must be **uniquely best (rank 1/k)** and the permuted transforms must **collapse to the floor** (the `v16_compose_permuted_check` rank-1/N! discipline that exposed the 2026-05-14 artifact). **Falsification: permuted does not collapse / TRUE not uniquely best → NO-GO** (this is the exact failure mode of the v1 false-GO).
- **(iii) lesion collapses.** Skip the unbind (`T := B`, no transform) → must drop **to the floor** (proves the transform is load-bearing, not similarity-to-C). **Falsification: lesion ≈ full → NO-GO.**
- **(iv) scrambled source → chance.** Random A,B (no real relation) → chance.
- **(v) no-confab moat 0-FA.** A relation/cue with no stored pair must **abstain (None)**, 0 false-accepts; correct-analogy cleanup sims must sit far above random-query sims (clean confidence separation). **Falsification: any forced answer on an unknown relation → moat breach → NO-GO.**
- **(vi) 6-seed.** All of the above must hold across 6 seeds (project standing rule) before any GO claim.

### The plain verdict

- **Regime A (factored / explicit relation) — UNLOCKABLE CHEAPLY (days, reuse-by-import, NO `sim/` edit).** The substrate already supports it (an explicit relation slot IS a clean additive phasor; the mechanism is proven 1.000 with collapsing controls on additive codes; the spiking op carries it). The de-risk above is expected to GO **for stored relations**, giving a real, anti-cheatable relational-reasoning capability (taxonomic inheritance, kinship, part-of) under a published mechanism — and redeeming "reasoning" under proper controls. **Honest scope: this is analogy/inference over relations the agent is GIVEN, not fluid analogy discovered from concept similarity.**
- **Regime B (analogy emerging on raw learned concept codes) — GENUINE-MONTHS-FRONTIER, gated on the corpus-scale deep-knowledge build, NOT a clever circuit.** The residual there is a true representational absence (+0.0015 delta-alignment), and whether corpus scale fixes it on the local-normalization PPMI cortex is an open research bet. The composer mechanism is ready when the codes are; do not chase regime B with a learned-linear projection (the project's own NEGATIVE warns it is fragile).
- **Transitive/ordinal (option d) — a separate MEDIUM redemption build** (TEM ordinal map + the symbolic-distance anti-cheat), recommended *after* (a) if transitive reasoning is prioritised, as the biologically-correct replacement for the retracted explicit-chaining approach.

**This gate's bottom line:** the boundary is **not** the spiking substrate and **not** the binding algebra — it is **which relations are represented explicitly.** Make relations explicit factored axes (cheap, biology-grounded by TEM/D.02, already-machinery) → relational reasoning is unlockable now for stored relations. Expect analogy to fall out of raw similarity codes → that is the corpus-scale frontier. Build option (a) first under the full anti-cheat bar; treat any regime-B claim with maximum skepticism until corpus scale is in hand.

---

## 5. Sources

**Project (verified by file-read):** `2026-06-27-tier2.1-analogy-NEGATIVE.md` (the boundary + the additive/bundled/random table + the +0.0015 delta-alignment); `2026-06-27-conv-thinking-research-reasoning-thinking.md` (§2.5 analogy, §2.4 transitive, the reusable-machinery table, the hard-walls); `2026-06-27-conv-thinking-research-comprehension-representation.md` (the representation-gap spine; TEM/blackboard/Assembly-Calculus precedents); `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` (bundling not learnable from scratch; fixed self-inverse bundles 0.989; learned-linear-inverse NEGATIVE); `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md` (CYCLE 88–90: PPMI codes generalize AND bind; "the genuine open problem is binding correlated codes, located at the binder"); `rf_phasor_composer.py:24,234,245,261,282` (ROLES alphabet; `_bind`=phase-add, `_bundle`=superposition, `_encode`=bundle-of-binds, `unbind`=conj); `argstructure_composer.py:49,172,187` (typed roles as clean disjoint-stream phasors bound to concept fillers).

**Catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`):** D.02 relational binding / Eichenbaum-Cohen "memory space" — transitive inference, dorsal-HC configural-learning deficit, the O'Keefe-Nadel map-traversal supplement (pp 1098–1109); D.07 grid cells / medial-EC metric (pp 1163+); D.03–D.05 trisynaptic / CA3 autoassociator. (Glossary `glossary.md` ABSENT from the catalog dir — only `feature-catalog.md`, `biology-buildout-roadmap.md`, `textbooks/`; substituted WebSearch + domain knowledge per instructions.)

**Literature (web-verified this gate):**
- Whittington, Muller, Mark, Chen, Barry, Burgess & Behrens 2020, "The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation," *Cell* — [cell.com](https://www.cell.com/cell/fulltext/S0092-8674(20)31388-X), [PMC7707106](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106/) (factorisation of structure from content → generalisation + transitive inference; EC = structural basis, HC = content conjunction).
- Rasmussen & Eliasmith 2011 *Topics in Cognitive Science* + Eliasmith 2013 *How to Build a Brain*; the spiking Raven's-matrices model — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0160289613001542), [Rasmussen rule-generation PDF](https://compneuro.uwaterloo.ca/files/Rasmussen.RuleGeneration.pdf) (transitions coded as circular convolutions over structured semantic pointers, transforms averaged into a rule vector; NEF spiking).
- Komer & Stewart 2020, "Analogical and Relational Reasoning with Spiking Neural Networks," IJCNN / arXiv 2010.06746 (proportional analogy A:B::C:? via unbind/transform/apply in spiking neurons).
- Hersche et al. 2023, *Nature Machine Intelligence* (neuro-vector-symbolic RPM at SOTA via bind/unbind/bundle).
- Gayler, "Vector Symbolic Architectures: A New Building Material for AGI" — [ResearchGate PDF](https://www.researchgate.net/publication/215991898_Vector_Symbolic_Architectures_A_New_Building_Material_for_Artificial_General_Intelligence); Furlong & Eliasmith spatial-semantic-pointer slot-filler work — [Neural Computation](https://direct.mit.edu/neco/article/33/8/2033/102625/Simulating-and-Predicting-Dynamical-Systems-With) (relations as role-filler bound slots; analogy over structured, not raw-similarity, representations).
- Bunge / Wendelken / Badre, rostrolateral-PFC relational integration & analogy (PMIDs 18052787, 26663572, 27012301 — analogy = second-order relations over WM-held structured representations).
