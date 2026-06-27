# Transitive inference research gate — can the RETRACTED capability be redeemed on THIS point-neuron substrate, and what makes it real this time?

**Date:** 2026-06-27
**Type:** READ-ONLY deep-research gate (standing practice: deep research + catalog + Kandel + literature review BEFORE committing any build / GPU / `sim/`-edit effort). **NO code, NO composer/`sim` edit** — the de-risk/build is a separate later step that this doc gates.
**Trigger:** the gate fires on a known family + a confirmed boundary. Transitive inference (given A>B and B>C, infer the UNSTORED A>C) was **CLAIMED then RETRACTED** on 2026-05-14 (a "90% multi-seed" that collapsed to ~chance under a corrected architecture + permuted control). The analogy gate (`2026-06-27-analogy-representation-research-gate.md`) ranked the redemption as its option (d): a Tolman-Eichenbaum-Machine ordinal-map mechanism, gated by the **symbolic-distance-effect** anti-cheat. Conditions (a) confirmed boundary + (b) known family [the relational-geometry / spreading-activation family] + (d) new-mechanism-class → the gate is mandatory.

> **Reader caveat — this is the project's MOST-BURNED capability.** Two retractions sit behind it: the 2026-05-14 transitive "90%" was an architecture-mismatch monkey-patch artifact (`2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`), and the 2026-06-27 Tier-2.1 analogy was a *correctly-caught* false-GO (a too-weak control gave acc 1.000; the sharpened control gave the true 0.000). **Every recommendation below therefore carries the full anti-cheat bar, and the headline control is one the artifact provably could not fake: the symbolic-distance effect.** The verdict is stated plainly as unlockable-cheaply vs genuine-months-frontier with an explicit falsification bar.

---

## 0. One-paragraph answer

Transitive inference is **redeemable cheaply on this substrate** — but ONLY if it is built as the biologically-correct mechanism (a learned **ordinal map**: place items on a line so an unstored pair's order is *read from positions*) and NOT as the thing that was retracted (spreading activation / edge-chaining over a co-occurrence graph). The genuine residual the project lacks is precisely **a learned 1-D ordinal embedding + a spiking comparison of two map positions** — and the literature gives the decisive reason this is the real test: **reinforcement-learning / associative-strength models "routinely fail to perform transitive inferences… and respond at chance levels"** (Jensen et al., the Betasort paper), while the mechanisms that DO work (TEM map geometry; Park 2020 vmPFC/EC Euclidean map; the Betasort biased-ordinal update) all produce the **symbolic-distance effect** — accuracy and confidence *increase* with the ordinal distance between the queried items, and the brain recruits the hippocampus *more* for the *harder, closer* pairs. **A lookup table, a memorized edge-set, or a co-occurrence-similarity artifact mathematically cannot produce a monotonic distance curve** (a stored edge either exists or doesn't — there is no graded "how far apart" for an *unstored* pair unless a metric map exists). That monotonic curve is the exact signature the 2026-05-14 version lacked. **Verdict: a learned ordinal map over 7–12 items, with the symbolic-distance curve as the mandatory anti-cheat, is unlockable at MEDIUM cost (~2–3 weeks, reuse-by-import, no `sim/` edit expected) and is the clean redemption of the retraction.** The cheapest *decisive* de-risk is §4. The scaling to many items / cross-domain ordinal maps, and the fully-learned (vs hand-built-roles) embedding, are the bounded follow-ons; the deep general-relational-reasoning version is a genuine months-frontier (the TEM-grade structural learner). What this gate firmly establishes: **the boundary is not the spiking substrate and not the binding algebra — it is "does a metric ordinal map exist," and the project's own PPMI/Hebbian code-geometry machinery can learn one.**

---

## 1. ISOLATE the genuine residual — what does transitive inference REQUIRE that query_chain and analogy do not?

### 1.1 The three capabilities are *distinct operations*, not degrees of one thing

The single most important conceptual move in this gate is to stop conflating three things the project has repeatedly blurred:

| capability | the operation | does the project have it? | the retraction risk |
|---|---|---|---|
| **multi-hop chaining** (`query_chain`) | FOLLOW stored edges hop-by-hop: A —eat→ B —eat→ C, reading each stored patient | **YES, PRODUCTION, anti-cheated** (`2026-06-17-multihop-query-chain-GO.md`) | low — every hop is a *stored* fact, abstains on a miss |
| **proportional analogy** (A:B::C:?) | extract a transform `T=B⊖A` and apply to C; `D=cleanup(C⊕T)` | regime A (explicit relation slot) cheap; regime B (raw codes) is the corpus-frontier (`2026-06-27-analogy-research-gate.md`) | medium |
| **transitive inference** (A>B,B>C ⊢ A>C) | INFER an **UNSTORED** order by reading **positions on a learned metric line** | **NO** — the retracted capability; the biologically-correct mechanism is not built | **highest** (the burned one) |

### 1.2 Why `query_chain` does NOT already solve transitive inference (the load-bearing distinction)

`query_chain(cue, [a₁, a₂, …])` is, verbatim from `rf_phasor_composer.py:721`, `x = cue; for a in actions: x = self.query_patient(x, a); if x is None: return None`. Every hop **matches a STORED (agent, action) fact and reads its STORED patient**. It is *path-following over edges that exist in the fact store*.

Transitive inference is the opposite: the queried pair's relation is **specifically NOT stored**. The defining property of a transitive-inference test is that you train only **adjacent** premise pairs (A>B, B>C, C>D, D>E) and then probe a **non-adjacent, never-trained** pair (B>D). There is **no edge B→D to follow** — `query_chain('B', ['greater_than'])` would simply return C (the one stored edge), not D, and would have no way to compare B and D at all.

To answer "is B greater than D?" without a stored B-vs-D fact, the only options are:

1. **Search the transitive closure at retrieval** — chain B>C, C>D, conclude B>D. This is *deductive search over stored edges*, NOT what brains do (it is slow, serial, and — critically — would make the *farther* pair B>E **harder** than B>C, because more hops; the brain shows the **opposite**: farther pairs are *easier* — the symbolic-distance effect, §3). It is also exactly the spreading-activation family that produced the 2026-05-14 artifact.
2. **Read it off a metric map** — if A,B,C,D,E sit at learned positions x_A < x_B < x_C < x_D < x_E on a line, then "B vs D" is a single comparison of x_B and x_D. No chaining. The unstored relation **drops out of the geometry** (O'Keefe-Nadel's exact phrase, catalog D.21). Farther pairs are *easier* because |x_B − x_D| is a bigger, more separable difference. **This is the residual.**

### 1.3 The genuine residual, stated precisely

> **What transitive inference REQUIRES that the project lacks: a learned ordinal/metric EMBEDDING that places items on a (1-D, or low-D) manifold such that the order of ANY pair — including an unstored one — is recoverable from a comparison of their map positions, PLUS a spiking comparison operator that reads "which position is greater" with a margin that grows with the positional gap.**

The machinery inventory (read directly from the repo, §3.4) confirms this is genuinely absent: the project has **discrete slot-position binding** (`OrderedPositionWM`, gamma slots — items bound to position-*roles* pos0…pos6, Lisman-Idiart), **co-occurrence code geometry** (PPMI/Hebbian stream cortex — distance encodes *categorical similarity*, not *ordinal rank*), and a **spiking accumulator** (the BG cascade / Wang-2002 — a comparator, but currently fed perceptual evidence, not two map positions). **What does not exist anywhere: a continuous 1-D ordinal scale, a rank embedding, a number-line, or any symbolic-distance / distance-effect read-out** (grep for "symbolic", "ordinal", "distance_effect", "number line", "rank" → zero dedicated runners). The residual is real, isolated, and small in mechanism-count (one learned embedding + one comparator) — it is **not** the whole hippocampal formation.

### 1.4 How big is the residual? — small in mechanism, but it is a NEW representation

The residual is **one learned metric axis + one comparator**, both reusing existing pieces (§3.4). But it is genuinely a *new representation* the project has never built: an ordinal code where geometry = order. It is **not** "almost there" on the current codes — the PPMI codes encode *similarity*, and similarity is **non-transitive** (cat~dog~wolf does not order them). So this cannot be read off existing code geometry; an ordinal training signal (adjacent-pair comparisons) must shape a *new* axis. That is the residual's true size: a bounded build (one embedding objective + one read-out), but a real one, not a recomposition of already-GO ops the way `query_chain` was.

---

## 2. REFRAME via biology — how does the brain actually do transitive inference?

The biology is unusually clear and points the SAME way as the residual analysis: **the brain does NOT chain edges at retrieval; it lays the items out on a learned low-dimensional map and reads the order off the geometry.**

### 2.1 The cognitive-map account — O'Keefe-Nadel (catalog D.21) + Eichenbaum-Cohen (D.02)

Catalog **D.21** (O'Keefe & Nadel 1978) states the mechanism in one sentence the project quoted but did not act on: *"novel inferences (shortcut taking, **transitive choices**, latent learning) **drop out of the framework's geometry, not out of stored sensorimotor associations**."* The locale system places items in a unitary metric frame; an unstored relation is recovered by **traversal/comparison on the map**, not by reactivating a specific stored stimulus (D.02 supplemental: *"place hypotheses can be tested without reactivating any specific stimulus that was originally present"*). Catalog **D.02** (Eichenbaum-Cohen relational memory): the hippocampus "networks via overlapping events allowing flexible inference (**e.g., transitive**)"; behavioral validation = "**transitive inference**; selective deficit on configural learning after dorsal-HC lesion" (Kandel 6e Ch 52 pp 1301–1302). **The project's 2026-05-14 mistake was precisely the one D.21 warns against: it tried transitive inference as stored-association chaining and got an artifact.**

### 2.2 The Tolman-Eichenbaum Machine — Whittington-Behrens 2020 (the decisive computational reframe)

TEM's central principle is **factorisation**: separate the *structural* code (the order/transition rules) from the *content* code (the items), then bind them. "Separating structural codes (the transition rules of the graph) from sensory codes allows generalization over environments sharing the same structure" (verified this gate). After learning, the structural basis units show **grid/band/border/object-vector** properties; the structure×content conjunction generalises — and "this factorization and conjunction approach is sufficient to build a relational memory system that generalizes structural knowledge… and accounts for… **transitive inference**." The payoff for *order*: the structural code IS a learned **ordinal axis** (a 1-D analogue of the grid metric), factored out of the items, over which inference is a geometric comparison. **TEM is the biological vindication of "learn an ordinal embedding + compare positions."**

### 2.3 Park et al. 2020 + the spatialized-time literature — the human/primate metric map

Park 2020 (verified): humans learned a 16-item structure from **pairwise comparisons along one ordinal axis at a time** (e.g. "popularity"), then achieved **93.6% on UNSEEN pairs**, with **map-like activity in vmPFC and entorhinal cortex sensitive to the ground-truth Euclidean distances** between items, and a hippocampal repetition-suppression signature at inference. "Representation of the inferred relationships in a map-like space" (HBM 2023) and "Spatialization of time in the entorhinal-hippocampal system" (Frontiers 2021) confirm the *same metric-map machinery* generalises from physical space to **abstract ordinal axes** — and that "a reasonably accurate spatial representation can be extracted from temporal context with as few as **eight cells**." Nieder's primate work shows **rank-tuned neurons** in dPFC/parietal that "encode the relative metrics of dimensions, such as quantities, numbers, and time" — i.e. a neural number-line with **distance-dependent tuning overlap** (the single-cell origin of the distance effect).

### 2.4 Why associative / RL models FAIL — the falsification anchor

This is the most useful literature result for the anti-cheat. From the Betasort paper (Jensen et al., PLoS Comput Biol 2015) and the value-transfer literature, verified this gate: **"Despite robust behavioral effects, reinforcement learning models reliant on reward prediction error or associative strength routinely fail to perform transitive inferences. Learning models that rely on only the expected values of stimuli fail to make the inference and respond at chance levels."** The models that DO work need either (i) a **biased ordinal update** ("update belief about one pair member but not the other" — Betasort; mirrored by *"asymmetric reinforcement learning facilitates human inference of transitive relations"*, Nat Hum Behav 2021) or (ii) a **map geometry** (TEM/Park). **Both produce the symbolic-distance effect; pure association does not.** This is the precise sense in which the 2026-05-14 spreading-activation result was *guaranteed* to be an artifact: a co-occurrence/association mechanism that "answers at chance" on real transitive inference can only have scored 90% by a bug — which is exactly what was found.

### 2.5 Point-neuron faithfulness — does this need the dendritic/grid substrate?

**No — and the catalog/literature is explicit.** Three independent reasons the ordinal-map form is point-neuron-faithful:
- TEM itself runs as a (rate-based, point-unit) RNN; the "structural basis" is learned weights, not dendritic computation. No spiking-TEM is published, but TEM's primitives (learned recurrent transition code + content binding) are all rate/point-level.
- The map need only be **1-D (an ordinal line)**, far simpler than the 2-D hexagonal grid (D.07, which *would* want path-integration/attractor machinery the project lacks). A 1-D ordinal axis is the easy case.
- The project's **PPMI/Hebbian online-stream cortex already learns a metric code geometry on the real spiking bridge** (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`: rate-Hebbian co-occurrence, population code 8–32 neurons/concept reaching 100% of host fidelity). The residual is to give that machinery an **ordinal training signal** (adjacent-pair comparisons) instead of a co-occurrence one — a change of *objective*, not of *substrate*. Park's "eight cells suffice" matches the project's population-code scale exactly.

The reframe verdict: **build a learned 1-D ordinal map (the easy, point-neuron-faithful case of the cognitive map) and compare positions with the existing spiking accumulator. Do NOT chain edges. Do NOT wait for a dendritic/grid substrate.**

---

## 3. RANK cheap-first options

Ranked by (substrate-fit × cheapness × biological-correctness × credibility-recovery on the burned front). All reuse-by-import; none requires the dendritic substrate. **Every option's decisive control is the SYMBOLIC-DISTANCE EFFECT** (§3.0), because that is the one signature a lookup/memorization/similarity artifact cannot fake.

### 3.0 The control that defines the whole gate — the symbolic-distance effect

For a learned ordinal map A<B<C<D<E<F<G, define **symbolic distance** = the number of rank-steps between the two queried items (B vs C = distance 1; B vs F = distance 4). The empirical, neurally-measured signatures (verified §literature) are:
- **accuracy INCREASES with symbolic distance** (adjacent pairs hardest, far pairs easiest);
- **the decision margin / cleanup confidence INCREASES with distance** (the read-out `_cleanup_all_scored` already returns this margin, `rf_phasor_composer.py:447` — read-only, no new mechanism);
- (optional, if a spiking accumulator is used) **commit latency DECREASES with distance**;
- **the hippocampal/map recruitment is GREATER for the harder, closer pairs** (Park; the PMC hippocampus-TI study) — the map is *worked harder* when positions are near.

**Why an artifact cannot fake this.** A lookup table / memorized edge-set has a *binary* truth value per pair (stored or not) — there is no graded "how far apart" for an **unstored** pair, so its accuracy-vs-distance curve is **flat** (or non-monotonic noise). A co-occurrence-similarity artifact (the 2026-05-14 mechanism) orders by raw activation overlap, which is *anti-correlated* or unrelated to ordinal distance — it cannot produce a clean monotonic rise. **A monotone-increasing accuracy/margin-vs-distance curve is a positive, falsifiable signature that a metric map exists and is being read.** This is the exact thing the retracted version lacked, and the reason this redemption can be *believed* where the original could not.

### 3.1 (a) ⭐ Learned 1-D ordinal map + spiking position-comparison — the recommended redemption (MEDIUM, ~2–3 weeks)

- **What:** train a 1-D ordinal embedding from **adjacent premise pairs only** (A>B, B>C, …) so items land at monotone positions on a line; answer an UNSTORED pair X vs Y by comparing map positions; report the symbolic-distance curve. Two sub-mechanisms, both point-neuron-faithful:
  - **the embedding:** an *ordinal training signal* into the existing co-occurrence/Hebbian machinery — adjacency drives the higher-ranked item's code one step "up" the axis (Betasort-style **biased update**: move one member, not both — this asymmetry is *itself* literature-validated, Nat Hum Behav 2021, and is what makes the order transitive rather than merely associative). Reuse: `_phaseB_onbridge_stream_cortex_derisk.py` rate-Hebbian + population code.
  - **the comparison:** feed the two items' map positions (population rates / accumulated drive) into the **existing spiking accumulator** (BG cascade / Wang-2002 commit, `g11_bg_runner.py`) → "X>Y" if X's position-evidence wins; the margin = the symbolic-distance read-out. Reuse: the GO spiking decision (`2026-06-19-spiking-decision-default-on-GO.md`).
- **Reusable machinery:** PPMI/Hebbian stream cortex (the metric code learner); the spiking WTA/accumulator (the comparator); `_cleanup_all_scored` (the margin/confidence read-out, `rf_phasor_composer.py:447`); the no-confab familiarity gate (`2026-06-11-familiarity-gate-v320-GO.md`); the anti-cheat tooling (`v16_compose_permuted_check.py`, the lesion/derangement hooks in `_phaseB_multihop_query_chain_derisk.py`).
- **Cost:** ~2–3 weeks; numpy de-risk first (CPU), then the comparison through the real spiking accumulator on ≥1 seed, then 6-seed GPU. NO `sim/` edit expected (reuse-by-import; the embedding objective lives in a new runner).
- **Payoff:** the **clean, biologically-correct redemption** of the project's most-burned capability, with the symbolic-distance effect as a positive falsifiable signature no artifact can fake. Directly realises catalog D.02/D.21 + TEM/Park. **Honest scope it buys:** transitive inference over a *single trained ordinal axis* of modest size (7–12 items — Park used 16; "8 cells suffice"). It does NOT yet claim cross-domain ordinal maps or a fully-self-organised TEM structural learner (those are §3.3).

### 3.2 (b) TEM-faithful factored structure×content embedding (HARDER, ~4–6 weeks; the "more biological" form)

- **What:** explicitly factor a *structural* code (the ordinal-axis basis, shared across domains) from a *content* code (the items), TEM-style, so the SAME learned ordinal structure transfers to a new set of items (learn "popularity" ordering, then order "competence" items faster). Reuse the structure across domains = genuine TEM generalisation.
- **Reusable machinery:** the composer's role/content factorisation (roles ARE a clean factored basis — `argstructure_composer.py`); the stream cortex for content; the ordinal objective from (a).
- **Cost:** ~4–6 weeks (factorisation + cross-domain transfer test). **Risk:** the structure-learning RNN that TEM uses is a genuine training problem; on point neurons the *structure* code is the open part. **Fire the research gate again before this** (it is a new mechanism class). Recommended only AFTER (a) GOes, as the generalisation upgrade.
- **Payoff:** cross-domain ordinal generalisation (the TEM hallmark) — high value, but a research bet, and gated on (a) succeeding first.

### 3.3 (c) Full relational structure learner / general transitive over arbitrary graphs (EXPENSIVE; the genuine months-frontier)

- **What:** a TEM-grade learner that infers arbitrary relational structure (not just a single line) from observations and supports transitive inference over any learned partial order, integrated with the conversational fact store.
- **Cost:** **months / research-grade** (this is the deep-relational-reasoning end-state, adjacent to the deep-knowledge build). NOT recommended now.
- **Payoff:** general relational reasoning — the strategic horizon, not a near-term de-risk.

### 3.4 (d) Edge-chaining / transitive-closure search at retrieval (CHEAP but WRONG — explicitly DO NOT build)

- **What:** answer B>D by chaining stored B>C, C>D via `query_chain`.
- **Why it is listed only to be rejected:** (i) it is the spreading-activation family that produced the 2026-05-14 artifact; (ii) it produces the **WRONG distance curve** — more hops for farther pairs → farther = *harder*, the exact opposite of the brain's symbolic-distance effect (it would **fail the mandatory anti-cheat by construction**, which is a useful negative-control but not a capability); (iii) it cannot answer when intermediate edges are themselves unstored. **This option is the trap; it is documented here so the build phase does not drift back into it.**

---

## 4. VERDICT — the single cheapest decisive de-risk + anti-cheats + falsification bar

### The recommendation

**Run option (a): the learned-1-D-ordinal-map + position-comparison de-risk, cheap-first on CPU.** It is the single cheapest test that decides whether transitive inference is unlockable on THIS point-neuron substrate without a months-arc, because it isolates the *one* thing in question (does a learned metric ordinal axis exist over which an unstored pair's order is read from positions) from the *one* expensive thing (a full self-organised TEM structural learner). It reuses validated machinery, needs no `sim/` edit, and its decisive control — the symbolic-distance effect — is the positive signature the burned version lacked.

### The de-risk protocol (read-only spec; build is the separate later step)

1. **Corpus:** a single ordinal chain of N=7 items, A>B>C>D>E>F>G. **Train ONLY the 6 adjacent premise pairs** (A>B, B>C, …, F>G). Never train, and never use to fit the embedding, any non-adjacent pair.
2. **Learn the embedding** via the biased-ordinal update into the Hebbian/population-code machinery (move the higher-ranked member's code one step up the axis per adjacent-pair presentation; Betasort/asymmetric-RL rule).
3. **Test on the held-out non-adjacent pairs** (B vs D, B vs E, …, A vs G — all 15 non-adjacent pairs). Answer each by comparing map positions through the spiking accumulator; record correctness AND the decision margin (`_cleanup_all_scored` confidence).
4. **Compute the symbolic-distance curve:** accuracy and margin as a function of |rank(X) − rank(Y)| over the held-out pairs.
5. Confirm the comparison **through the real spiking accumulator** (not just host arithmetic) on ≥1 seed, matching the numpy reference, before the 6-seed GPU run.

### Anti-cheats (ALL mandatory — this is the burned-capability bar; any failure → STOP, write the honest NEGATIVE, do not over-claim)

- **(i) ⭐ THE SYMBOLIC-DISTANCE EFFECT — the headline control.** Accuracy AND margin must **increase monotonically** with symbolic distance over the held-out pairs (a positive Spearman correlation between distance and accuracy/margin, significant across seeds). **Falsification: a flat or non-monotonic distance curve → NO-GO** (this is the signature the 2026-05-14 artifact could not produce; without it, any accuracy is presumed an artifact). This is the control the brief correctly identifies as the whole point.
- **(ii) held-out non-adjacent ≫ floor.** The scored pairs are NEVER trained and NEVER used to fit the embedding. Held-out accuracy must be **≫ chance (0.5 for a 2AFC order judgment)** AND ≫ a memorization baseline (a model trained only on the premise edges with no metric axis — i.e. a stored-edge lookup, which is at chance on non-adjacent pairs by construction). **Falsification: held-out ≤ chance → NO-GO** (this is exactly the corrected-architecture collapse of 2026-05-14).
- **(iii) permuted-order collapses (rank-1/N! discipline).** Shuffle the trained order (train a *random* set of "adjacent" pairs). The TRUE order must **uniquely** produce correct held-out inferences AND the symbolic-distance curve; permuted orders must collapse to chance with a flat distance curve. Reuse the `v16_compose_permuted_check.py` rank-1 discipline that *exposed* the 2026-05-14 artifact. **Falsification: permuted does not collapse / TRUE not uniquely best → NO-GO** (the exact failure mode the original lacked a control for).
- **(iv) lesion the map → collapse.** Sever the ordinal-embedding read-out (feed the comparator scrambled positions, or zero the learned axis) → held-out inference must drop to chance and the distance curve must flatten. **Falsification: lesion ≈ full → NO-GO** (proves the metric map is load-bearing, not a residual similarity cue).
- **(v) spreading-activation negative control.** Run the relation-blind co-occurrence/spreading baseline (`spreading_predict` in `_phaseB_multihop_query_chain_derisk.py`) on the same corpus. It must FAIL the held-out non-adjacent pairs (literature-guaranteed: associative models answer at chance) — confirming the corpus genuinely requires the map, not co-occurrence. **Falsification: spreading also "passes" → the corpus is not discriminating; redesign before any GO** (the self-policing the multi-hop probe already demonstrated).
- **(vi) no-confab moat 0-FA.** An item never placed on the map, or a comparison with an absent operand, must **abstain (None)**, zero false-accepts. **Falsification: any forced answer on an unmapped item → moat breach → NO-GO.**
- **(vii) 6-seed.** All of the above — especially the symbolic-distance correlation — must hold across 6 seeds (project standing rule) before any GO claim.

### The explicit falsification bar (one line)

> **GO only if: held-out non-adjacent accuracy ≫ chance AND a significant monotone symbolic-distance curve (accuracy & margin rise with distance) AND permuted-order collapses to a flat chance curve AND lesion collapses AND the spreading baseline fails AND the moat holds 0-FA — across 6 seeds. Absence of the monotone distance curve is by itself a NO-GO regardless of raw accuracy.**

### The plain verdict

- **Transitive inference via a learned 1-D ordinal map — UNLOCKABLE at MEDIUM cost (~2–3 weeks, reuse-by-import, no `sim/` edit expected), and it is the clean redemption of the most-burned retraction.** The substrate supports it (the residual is one learned ordinal axis + one comparator, both reusing GO machinery; the literature confirms a 1-D ordinal map is point-neuron-faithful and "8 cells suffice"). The de-risk above is *expected to GO* for a single trained ordinal axis of 7–12 items — IF and ONLY IF the symbolic-distance effect appears; that control is what converts the retraction into a believable result.
- **TEM-faithful cross-domain factored ordinal generalisation (option b) — a separate HARDER build (~4–6 weeks), gated on (a) succeeding and on re-firing the research gate** (it is a new structure-learning mechanism class). Recommended as the generalisation upgrade, not the first build.
- **General relational-structure learning over arbitrary graphs (option c) — GENUINE MONTHS-FRONTIER**, adjacent to the deep-knowledge / deep-relational-reasoning end-state. Not a near-term de-risk.
- **Edge-chaining / transitive-closure search (option d) — explicitly DO NOT build.** It is the retracted family; it fails the mandatory distance-effect control by construction (more hops → farther-is-harder, the wrong curve).

**This gate's bottom line:** the boundary is **not** the spiking substrate and **not** the binding algebra — it is **"does a metric ordinal map exist."** Build one (cheap, biology-grounded by D.02/D.21/TEM/Park, reusing the PPMI/Hebbian code-geometry machinery + the spiking accumulator), gate it on the **symbolic-distance effect**, and transitive inference is redeemable now. Try to do it by chaining stored edges again → it fails the distance-effect control by construction, which is exactly why the 2026-05-14 version was an artifact.

---

## 5. Reusable project machinery (what to build ON)

| Need | Existing machinery | Source (verified by read/grep) |
|---|---|---|
| Metric **code-geometry learner** (give it an ordinal objective) | PPMI / rate-Hebbian online-stream cortex + population code (8–32 neurons/concept → 100% host fidelity) | `_phaseB_onbridge_stream_cortex_derisk.py`; `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` |
| **Position comparator** (which map position is greater) | BG-cascade Wang-2002 / Lo-Wang spiking accumulator (GO, default-on) | `g11_bg_runner.py`; `2026-06-19-spiking-decision-default-on-GO.md` |
| **Margin / distance-effect read-out** (the symbolic-distance signal) | `_cleanup_all_scored` — per-decode mean-cosine confidence in [0,1], read-only of the matched filter | `rf_phasor_composer.py:447` |
| **Discrete position binding** (closest existing reuse; slots not a continuum) | `OrderedPositionWM` — items bound to position-*roles* pos0…pos6 (gamma slots, Lisman-Idiart) | `ordered_position_wm.py` |
| **No-confab abstention** (moat 0-FA on unmapped items) | learned Bogacz-Brown familiarity gate | `2026-06-11-familiarity-gate-v320-GO.md` |
| **Anti-cheat: permuted rank-1/N!** | the permuted-mapping discipline that exposed the artifact | `v16_compose_permuted_check.py` |
| **Anti-cheat: lesion + spreading baseline + derangement** | `query_chain(lesion_rng=…)`, `spreading_predict()`, the `_genfrontier_*_derisk.py` derangement pattern | `_phaseB_multihop_query_chain_derisk.py`; `_genfrontier_*_derisk.py` |
| **Cleanup / attractor re-discretization** | spiking NEF / RF phase-cosine cleanup (== numpy parity) | `rf_phasor_composer.py:381,423,447`; `one_brain_composer.py` |

**Explicitly NOT to reuse:** `compose_concept_chain_test.py` — the RETRACTED runner; mechanically it is **spreading-activation over a co-occurrence graph** (trains (A,B)+(B,C) engrams, queries A, ranks pools by **raw firing-rate overlap** with no role structure, no abstention, multiplicative error). It is the template of what NOT to do.

---

## 6. Honest hard walls (point-neuron substrate)

1. **The ordinal axis must be LEARNED with the right (biased) objective, not read off similarity codes.** Similarity is non-transitive; the PPMI codes order by relatedness, not rank. The Betasort/asymmetric-RL **biased update** (move one pair-member, not both) is what makes the learned axis transitive — and is itself literature-validated. If the embedding objective is symmetric/associative, expect the associative-model failure mode (chance on non-adjacent pairs). This is a *design* constraint, not a substrate wall.
2. **Graded-magnitude / rate-code SNR wall (documented, multiply-confirmed).** The *margin* read-out (the distance-effect signal) is a graded quantity on point neurons → expect **coarse** distance resolution (the curve may be monotone but step-like, with adjacent distances hard to separate). This is fine — the anti-cheat needs *monotonicity*, not fine precision — but characterise the resolution honestly; do not over-claim a smooth psychometric curve.
3. **Small-N first.** Park used 16 items, "8 cells suffice"; the project's population code is 8–32 neurons/concept. De-risk at N=7 (the classic 5–7 item TI ladder). Scaling N (and especially *2-D* maps / hexagonal grids, D.07) brings the path-integration / attractor machinery the project lacks — out of scope for this de-risk.
4. **Exact-inverse VSA algebra remains an idealization.** If the comparison routes any binding through the FHRR composer, it rides the clean-inverse algebra (`2026-06-06` known limitation) — legitimate but flag it. The cleaner design keeps the *comparison* in the spiking accumulator (rates/positions), not the composer.
5. **The TEM structural learner (option b/c) is the genuine open part.** Learning a *shared, transferable* structural code (vs a single fixed axis) on point neurons is unproven here and is a re-gate item. Do not let a single-axis GO be over-read as TEM-grade generalisation.

---

## 7. Sources

**Project (verified by file-read/grep):** `2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md` (the retraction: spreading-activation artifact, 90%→1/4 corrected, the monkey-patch root cause); `2026-06-27-analogy-representation-research-gate.md` (option (d) TEM ordinal map + the symbolic-distance framing; regime-A/B factored-relation distinction); `2026-06-27-conv-thinking-research-reasoning-thinking.md` (§2.4 transitive — "the biological mechanism (map geometry) is different and not yet built"; the reusable-machinery table; the hard walls); `2026-06-17-multihop-query-chain-GO.md` (the VALIDATED role-structured pointer-chase + its anti-cheats; the explicit "not order-inference, do not conflate" caveat); `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` + `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md` (the PPMI/Hebbian code-geometry learner, population code 8–32 neurons/concept → 100% host); `2026-06-19-spiking-decision-default-on-GO.md` (the Wang-2002/Lo-Wang spiking accumulator); `2026-06-11-familiarity-gate-v320-GO.md` (the no-confab moat); `rf_phasor_composer.py:208,381,423,447,681,721,765` (`_cleanup*`, `_cleanup_all_scored` margin read-out, `query_patient`, `query_chain`, `chain_of_thought`); `ordered_position_wm.py` (discrete gamma-slot position binding — the closest existing reuse); `compose_concept_chain_test.py` (the retracted spreading-activation runner — what NOT to do); `_phaseB_multihop_query_chain_derisk.py` (`spreading_predict`, lesion + permuted controls); `v16_compose_permuted_check.py` (rank-1/N! discipline). Grep confirmed: **no existing ordinal-scale / number-line / symbolic-distance / rank-embedding runner.**

**Catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`):** **D.02** relational binding / Eichenbaum-Cohen "memory space" — transitive inference, dorsal-HC configural deficit, O'Keefe-Nadel map-traversal supplement (pp 1098–1109, "place hypotheses tested without reactivating any stimulus"); **D.21** cognitive-map theory — O'Keefe-Nadel "transitive choices **drop out of the framework's geometry, not stored associations**" (pp 1041–1048); **D.07** grid cells / medial-EC metric (pp 1163+, flagged as the harder 2-D case needing path-integration the project lacks); **D.24** theta-paced sequence compression (the STDP-window mechanism for ordered codes). (`glossary.md` ABSENT from the catalog dir — only `feature-catalog.md`, `biology-buildout-roadmap.md`, `textbooks/`; substituted WebSearch per instructions.)

**Literature (web-verified this gate):**
- Whittington, Muller, Mark, Chen, Barry, Burgess & Behrens 2020, "The Tolman-Eichenbaum Machine," *Cell* — [cell.com](https://www.cell.com/cell/fulltext/S0092-8674(20)31388-X), [PMC7707106](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106/), [bioRxiv](https://www.biorxiv.org/content/10.1101/770495v2.full) (factorise structure from content → generalisation + transitive inference; structural basis = grid/band/border units).
- Park, Miller, Nili, Ranganath & Boorman 2020/2021, "Inferences on a multidimensional social hierarchy use a grid-like code" / map-like vmPFC+EC representations sensitive to ground-truth Euclidean distance; 93.6% on unseen pairs — and the follow-up [Representation of the inferred relationships in a map-like space, HBM 2023, PMC10203794](https://pmc.ncbi.nlm.nih.gov/articles/PMC10203794/).
- Jensen, Muñoz, Alkan, Ferrera & Terrace 2015, "Implicit Value Updating Explains Transitive Inference Performance: The Betasort Model," *PLoS Comput Biol* — [journal](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004523), [PMC4583549](https://pmc.ncbi.nlm.nih.gov/articles/PMC4583549/) (**"RL models reliant on RPE or associative strength routinely FAIL transitive inference… respond at chance"; the biased-ordinal update that works**).
- Ciranka et al. 2021/2022, "Asymmetric reinforcement learning facilitates human inference of transitive relations," *Nature Human Behaviour* — [nature.com](https://www.nature.com/articles/s41562-021-01263-w) (the biased/asymmetric update is what yields transitivity).
- "The role of the hippocampus in transitive inference," *Hippocampus*/[PMC2693094](https://pmc.ncbi.nlm.nih.gov/articles/PMC2693094/), [PubMed 19216061](https://pubmed.ncbi.nlm.nih.gov/19216061/) (**greater hippocampal activity for SMALLER symbolic distance** — the harder, closer pairs recruit the map more).
- "Inferior parietal cortex represents relational structures for explicit transitive inference," *Cerebral Cortex* 2024 — [Oxford](https://academic.oup.com/cercor/article/34/4/bhae137/7641204) (parietal representation modulated by symbolic distance AND serial position).
- "Spatialization of Time in the Entorhinal-Hippocampal System," *Front. Behav. Neurosci.* 2021 — [Frontiers](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2021.807197/full) ("eight cells suffice" to extract a spatial/ordinal representation from temporal context).
- Van der Helm 2002 / O'Reilly-Frank lineage, "Simulating symbolic distance effects in the transitive inference problem," *Neurocomputing* — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231201005124) (a connectionist precedent that explicitly reproduces the distance effect — a model to compare against).
- Nieder et al., rank/number-line neurons in dPFC/parietal — "Neuronal encoding of recognition memory for numerical quantities," *ScienceDirect* 2025 (distance-dependent tuning overlap = the single-cell origin of the symbolic-distance effect).
