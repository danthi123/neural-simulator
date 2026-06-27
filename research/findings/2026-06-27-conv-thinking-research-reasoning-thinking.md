# Deep-research findings — REASONING + THINKING capability cluster

**Date:** 2026-06-27
**Type:** Read-only deep research (standing practice: catalog + Kandel + literature review BEFORE building).
**Scope:** ONE capability cluster — *reasoning and thinking* — for the conversation/thinking roadmap.
**Author:** read-only research analyst. NO `sim/` edit, NO code. Pure research + gap analysis + ranked, anti-cheated build options.

> **Reader caveat — this project has been BURNED by over-claimed reasoning.** The 2026-05-14 "transitive inference 90% multi-seed" and "pool-firing readout 65%" results were **measurement artifacts** of an architecture-mismatch monkey-patch bug and were RETRACTED (`2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`). Every reasoning claim in this doc's recommendations therefore carries a **mandatory permuted/lesion control** in its anti-cheat block. A reasoning result is not believed until the TRUE relational mapping uniquely beats its permutations AND a lesion of the load-bearing pathway collapses it.

---

## 0. Sources consulted

1. **Catalog** `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (5640 lines, 323 entries). Grepped for reason/infer/prefrontal/hippocampus/replay/analogy/planning/default-mode. The "G cluster" (Kandel Ch 52–56, Memory/Cognition + Decision-Making) and the "D cluster" (hippocampus) + "N cluster" (sleep/replay) are the relevant homes.
2. **Kandel 6e** `…\textbooks\kandel-pns-6e\full-book.txt` (8.7 MB). Read Ch 34 (premotor rule-encoding, pp ~836), Ch 52/54/56 cited passages. Kandel is a *systems-neuro* text — it covers decision accumulators, working memory, and language well, but is **thin on analogy / rostrolateral-PFC / default-mode reasoning** (these are flagged via the catalog + primary literature instead).
3. **Glossary** — **ABSENT.** `E:\Documents\Projects\sim-catalog\references\glossary.md` does not exist (the catalog dir holds only `feature-catalog.md`, `biology-buildout-roadmap.md`, and `textbooks/`). Noted per instructions; substituted WebSearch + bio-research MCP + domain knowledge.
4. **Literature** — bio-research MCP (PubMed) + WebSearch. Key hits confirmed below with PMIDs/venues.

---

## 1. The sim's current reasoning machinery (ground truth for the gap analysis)

Read directly from the code, not the docs:

- **Concept codes:** learned from corpus co-occurrence (PPMI online-stream cortex; `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`). These codes *carry semantic similarity* and generalize across similar concepts (the generalization arc, CYCLE 88–96).
- **Facts:** stored as **bare (agent, verb, patient) SVO triples** in a vector-symbolic (FHRR / resonate-and-fire) composer — `rf_phasor_composer.py` (`RFPhasorComposer`), `one_brain_composer.py` (`OneBrainComposer`). Binding is the exact-inverse VSA algebra (a *principled idealization*, Eliasmith SPA — NOT a learned cortex; see `2026-06-06-composer-vsa-idealization-known-limitation.md`).
- **Multi-hop reasoning EXISTS** — `RFPhasorComposer.query_chain(cue, actions)` (line 721) / `BrainConversationalAgent.reason_chain`. **Exact mechanism, verbatim from the code:** `x = cue; for action in actions: x = self.query_patient(x, action); if x is None: return None`. It iterates `query_patient` (match the current concept as AGENT under the hop's ACTION, read the PATIENT) over an **explicit, caller-supplied list of actions**, with the no-confab moat abstaining the moment any hop misses. De-risked GO 3-seed × 3-D with leaky-spreading / permuted-relation / re-cue-lesion controls (`2026-06-17-multihop-query-chain-GO.md`). Cleanup re-discretizes between hops so error does not compound.
- **Content selection / "what to say":** a dlPFC spiking content-selection Control over an association graph built from the agent's own facts (`elaborate`).
- **No-confab moat:** learned Bogacz-Brown familiarity gate (validated alongside host abstention at V=320, zero breaches; `2026-06-11-familiarity-gate-v320-GO.md`).

**What this is:** a reliable, abstention-protected **role-structured pointer-chase over an explicit query plan**. **What it is NOT (the cluster-wide gap):** the system does not DECIDE which relations to chase, does not COMPUTE novel relations, and has **no analogy, no induction, no abduction, no commonsense, no mental simulation, no self-generated train of thought, no planning that searches.** Every "inference" today is the caller spelling out the hops.

---

## 2. Capability-by-capability analysis

For each: **(a)** human capability + concrete example; **(b)** biological mechanism (regions/circuits + catalog IDs + Kandel pages); **(c)** what OUR sim has/lacks; **(d)** cheap-first biology-grounded options, ranked, with reusable machinery + anti-cheat controls.

### 2.1 Deductive inference

**(a) Human capability.** Truth-preserving derivation from premises. *"All birds have feathers; a robin is a bird; therefore a robin has feathers."* In a conversational agent: answering "Is a robin warm-blooded?" from "robins are birds" + "birds are warm-blooded" without that fact ever being stored.

**(b) Biology.** Deduction is **not a dedicated brain organ** — humans are notoriously bad at formal logic and good at content-laden, schema-driven inference (Wason selection task). The substrate is **PFC-mediated rule application** over relational structure. Catalog **G.06 / G.08 PFC working memory — sustained delay-period activity** (Kandel 6e Ch 52 pp 1292–1294; Ch 34 pp 827–842) holds the premises online; **Kandel Ch 34 pp 835–836** (Wallis/Miller-style abstract-rule encoding: PFC/PMd neurons fire for "match rule" vs "nonmatch rule" independent of the cue) is the closest neural correlate of applying a rule to operands. The actual "all-X-are-Y → this-X-is-Y" step is **property inheritance over a taxonomy** — a hippocampal/cortical relational-memory operation (catalog **D.02**, below).

**(c) Our sim.** PFC-WM exists as a `partial` (the 60-neuron `dlpfc_wm` recurrent region). Rule-encoding is `missing`. Property inheritance is **structurally the same operation as multi-hop `query_chain`** (chase IS-A edges, then read the property) — so a *restricted* deductive capability (taxonomic inheritance) is **one query-plan away** IF IS-A facts are stored. General deduction is `missing` and is largely out of scope for a point-neuron substrate.

**(d) Options (ranked).**
1. **Taxonomic property inheritance via the existing pointer-chase** (CHEAPEST). Store IS-A and HAS-property triples; answer "is robin warm-blooded?" by `query_chain('robin', ['is_a', ... ]) → 'bird'` then `query_patient('bird','has') → 'warm_blooded'`. Reuses `query_chain` verbatim. **Anti-cheat:** (i) **held-out inheritance** — the answer fact (`robin has warm_blooded`) must NEVER be directly stored, only `robin is_a bird` + `bird has warm_blooded`; (ii) **permuted-IS-A control** — shuffle which animal is-a which class; the TRUE taxonomy must uniquely beat permutations; (iii) **lesion** the IS-A pathway → collapse to abstention (no-confab moat must hold, not confabulate). This directly re-tests the RETRACTED transitive claim under proper controls.
2. **Deciding *which* hops to chase** (the real gap) — see §2.6 (train of thought) and §2.10 (planning). General first-order deduction is **NOT recommended** as a build target: it is the wrong altitude for this substrate and humans don't do it natively.

---

### 2.2 Inductive inference (generalization from instances)

**(a) Human capability.** Infer a general rule from examples. *"Swan 1 white, swan 2 white, … → swans are white."* Conversationally: after hearing "dogs bark, dogs run, dogs eat", generalize a "dog" category that supports a never-heard prediction.

**(b) Biology.** Induction = **statistical learning + schema abstraction**. Cortex extracts regularities across episodes (slow, distributed); the **hippocampal–entorhinal–vmPFC circuit** builds **cognitive maps / schemas** that compress instances into relational structure (catalog **D.02 relational binding**, Kandel Ch 52 pp 1301–1302; Eichenbaum/Cohen 2014). vmPFC stores schemas that **accelerate** assimilation of schema-consistent new facts (Tse et al. 2007 — already the project's `V_SCHEMA` mechanism). The literature is explicit that the EC–HC–frontal circuit codes **abstract task structure** that generalizes (Park 2020/2021; Whittington TEM 2020; "A map of abstract relational knowledge in the human hippocampal–entorhinal cortex", eLife 2016, Garvert/Behrens).

**(c) Our sim.** **This is the project's STRONGEST reasoning-adjacent capability already.** The PPMI stream cortex IS inductive: it learns category structure from co-occurrence and generalizes to held-out concepts (held-out category accuracy 0.86–0.92, multi-seed, anti-cheated; generalization arc CYCLE 88–96, `2026-06-16-generalization-*`). What is `missing` is **explicit rule induction expressible as a new stored fact** ("dogs (in general) bark") and **few-shot induction in-conversation** (hear 3 instances → induce + state the generalization).

**(d) Options (ranked).**
1. **Category-level fact induction over the learned codes** (CHEAP, high-leverage, builds on a GO arc). When N instances share (category, relation, X), write a category-level triple. The category centroid already exists (the PPMI code geometry). **Anti-cheat:** (i) the induced generalization must predict a **held-out instance** never used to induce it; (ii) **derangement control** — shuffle category labels; true categories must uniquely win; (iii) lesion the convergence pathway → generalization collapses to chance (the validated `_genfrontier_onsubstrate_convergence` control pattern).
2. **Online few-shot induction in the dialogue loop** (MEDIUM) — accumulate instances in the `SpikingLoopContextBuffer` (multi-turn WM already exists), threshold, emit. Anti-cheat: empty-WM / single-instance must abstain (no premature induction).

---

### 2.3 Abductive inference (inference to the best explanation)

**(a) Human capability.** Given an observation, infer the most plausible cause. *"The grass is wet → it probably rained."* Conversationally: "Why is the dog barking?" → "because someone is at the door" (best stored/known explanation, with uncertainty).

**(b) Biology.** Abduction maps onto **Helmholtzian unconscious inference / predictive coding** — perception and cognition as *inference under a prior* (catalog **E.21 Constructive / inferential perception**, Helmholtz; Bayesian framing). Mechanistically: the brain inverts a generative model — top-down predictions vs bottom-up evidence, settling on the explanation that minimizes prediction error (hierarchical cortex; the project has a `predictive_coding.py` module). At the decision level, abduction is **accumulating evidence for competing causal hypotheses** — the **LIP/parietal accumulator** integrating *any* evidence weighted by reliability (catalog **G.18 — probabilistic reasoning from symbols, logLR accumulation in LIP**, Kandel Ch 56 pp 1404–1407, Fig 56-9: "LIP's accumulator is not specific to perceptual evidence… Foundation for inferential / Bayesian-like reasoning"). The "best" explanation is the hypothesis whose accumulated logLR first crosses threshold (G.16 drift-diffusion, Kandel Ch 56 pp 1399–1404).

**(c) Our sim.** Abduction is `missing`. BUT two reusable substrates exist: (i) the BG cascade is **functionally a bounded accumulator** (catalog G.16/G.17 `partial` — "cortex_X firing rate accumulates DA-modulated evidence; threshold at thalamus→motor"); (ii) a `predictive_coding.py` module exists. No symbol-with-learned-reliability primitive (G.18 `missing`), no causal/explanatory link type in the fact store.

**(d) Options (ranked).**
1. **Cause→effect relation in the fact store + reverse query for "why"** (CHEAP). Store `(rain, causes, wet_grass)`; "why wet grass?" = unbind the **agent** of facts whose patient matches the observation (the composer already supports `query_agent`). This is *single-best* abduction. **Anti-cheat:** permuted cause→effect mapping must lose; observation with no stored cause must abstain ("I don't know why").
2. **logLR accumulation over multiple competing causes** (MEDIUM, biologically the real thing — G.18). Each candidate cause contributes a learned-reliability vote; the BG-cascade accumulator (existing) picks the winner; report confidence from the margin. **Anti-cheat:** (i) reliability weights must be LEARNED, not hand-set (freeze-learning control → no preference); (ii) the winner must track the *manipulated* evidence (raise one cause's reliability → it wins). **Flag:** rate-coded logLR on point neurons inherits the documented graded-magnitude / rate-code SNR wall — accumulation works (the nav accumulator is GO) but fine logLR *graded precision* will be coarse. Honest-negative-characterize the precision ceiling.

---

### 2.4 Multi-hop & transitive reasoning

**(a) Human capability.** Chain relations across facts (multi-hop: "dog eats cat, cat eats mouse → dog→mouse via 2 hops") and infer un-stated orderings (transitive: trained A>B, B>C, C>D, D>E → infer B>D without ever seeing it).

**(b) Biology.** **Hippocampal relational memory** is the canonical substrate. Catalog **D.02** (Kandel Ch 52 pp 1301–1302; Eichenbaum/Cohen): "networks via overlapping events allowing flexible inference (e.g., transitive)"; **behavioral validation: transitive inference; selective deficit on configural learning after dorsal-HC lesion.** The mechanism is a **cognitive map** whose geometry encodes the latent order: items laid out on a 1-D (or N-D) manifold in EC–HC–vmPFC such that Euclidean distance on the map = inferred relational distance (Park et al. 2020/2021 *Neuron/Nature Neuro* — "map-like representations in vmPFC and EC sensitive to ground-truth Euclidean distances"; repetition-suppression in HC at inference time). Transitive inference does NOT require chaining at retrieval if the map already places B and D — it is a **lookup on a learned low-dimensional structure** (this is the key biological insight the project missed when it tried explicit chaining).

**(c) Our sim.** **Multi-hop is GO and PRODUCTION** (`query_chain`, anti-cheated). **Transitive inference is the RETRACTED capability** — and the retraction is *informative*: the project tried it as explicit chaining / spreading activation and got artifacts. The biological mechanism (map geometry) is **different and not yet built**: there is no learned 1-D/N-D ordering manifold; facts are flat triples with no embedded metric over a relation.

**(d) Options (ranked).**
1. **Transitive inference via a learned ordinal map** (MEDIUM, the biologically correct fix; directly redeems the retraction). Learn an embedding where trained adjacent pairs (A>B, B>C…) place items on a line (the PPMI/Hebbian co-occurrence machinery + a relational training signal); infer B vs D by **comparing map positions**, not chaining. Reuses: the stream-cortex code-geometry learning; the cleanup/attractor for read-out. **Anti-cheat (MANDATORY, this is the burned capability):** (i) **never train or test the inferred pair directly** (B>D held out); (ii) **permuted-order control** — shuffle the trained order; the TRUE order must uniquely produce correct held-out inferences (rank 1/N! exactly the `v16_compose_permuted_check` pattern that exposed the artifact); (iii) **symbolic-distance effect** — accuracy/latency should grade with map distance (|B−D| easier than |B−C|), a signature a lookup-artifact cannot fake; (iv) lesion the map → collapse.
2. **Keep `query_chain` for genuine multi-hop** (DONE) — but note it is *path-following*, not *order-inference*; do not conflate the two again.

---

### 2.5 Analogy & relational reasoning ⭐ (highest substrate-fit)

**(a) Human capability.** Map relational structure from a source to a target despite different surface features. *"Hand is to glove as foot is to ___ (sock)"*; Raven's Progressive Matrices; "an atom is like a solar system." The hallmark of fluid intelligence.

**(b) Biology.** **Rostrolateral prefrontal cortex (RLPFC / frontopolar, ~BA 10) is the analogy/relational-integration hub** — it integrates *relations between relations* (second-order relations), the step that distinguishes analogy from simple relation matching. Primary literature (confirmed via PubMed): **Bunge, Wendelken, Badre, Wagner** on rLPFC in relational reasoning and analogy (e.g., Wendelken et al.; Bunge et al.; PMIDs in the rostrolateral-PFC/relational-reasoning set incl. 18052787, 26663572, 27012301). The broader circuit is the **frontoparietal control network** (rLPFC + dorsolateral PFC + posterior parietal) supporting relational integration and the manipulation of structured representations. Kandel does NOT cover rLPFC analogy in depth (systems-neuro focus) — this is a primary-literature region. Catalog-adjacent: **G.06/G.08 PFC-WM** (holds the operands), **D.02** (relational structure).

**(c) Our sim.** Analogy is `missing` as a capability — BUT **the project already owns the exact computational primitive that a spiking analogy engine needs.** This is the most important finding in this doc:

> **Vector-Symbolic Architectures perform analogy by `unbind → average-transform → apply`** — and the project has a *production FHRR/VSA composer on the spiking substrate*. The literature is explicit and directly transferable:
> - **Eliasmith's Semantic Pointer Architecture (SPA) solves Raven's Progressive Matrices** in spiking neurons: build a VSA vector per cell, **unbind** each cell from the next to get a *transformation vector*, **average** the transformation vectors into a single *rule vector*, then **apply** the rule to the second-last cell to predict the answer (Rasmussen & Eliasmith 2011, *Topics in Cognitive Science*; Eliasmith 2013 *How to Build a Brain*).
> - **Komer & Stewart, "Analogical and Relational Reasoning with Spiking Neural Networks"** (arXiv 2010.06746; IJCNN 2020) implements proportional analogy (A:B::C:?) directly with semantic pointers in spiking neurons via the same unbind/transform machinery.
> - **Neuro-vector-symbolic architectures** now solve RPM at SOTA by exactly this bind/unbind/bundle algebra (Hersche et al. 2023, *Nature Machine Intelligence*).
>
> The project's `RFPhasorComposer` already exposes `bind`, `unbind`, `bundle`, and `cleanup` on resonate-and-fire phasor neurons + complex synapses. **A:B::C:? is computable today as `apply(bundle_of_transforms, C)` where each transform = `bind(A, inverse(B))`** — i.e., the SAME ops `query_chain` already calls, recomposed.

**(d) Options (ranked).**
1. **Proportional analogy A:B::C:? via the existing composer ops** (CHEAPEST and highest-leverage). Transform `t = bind(A, unbind-inverse(B))`; answer `= cleanup(apply(t, C))`. Multi-relation analogy: bundle several transforms then apply (the SPA-RPM recipe). Reuses `RFPhasorComposer` verbatim; NO `sim/` edit. **Anti-cheat (the burned-project bar):** (i) **held-out analogy pairs** — never store the (C, answer) relation; (ii) **permuted-relation control** — the TRUE relation transform must uniquely beat all permuted transforms (rank 1/k); (iii) **scramble the source** (random A,B) → must drop to chance; (iv) the no-confab moat must abstain when no clean cleanup target exists (no forced answer). This is *the* place to recover credibility on "reasoning" because the mechanism is published, spiking, and already half-built here.
2. **Second-order relational integration (analogy-OF-relations)** (MEDIUM) — bind relations themselves into higher-order structures, the rLPFC step. Reuses recursive-clause binding (already validated: `2026-06-18` clause parity). Anti-cheat: held-out second-order mappings; lesion the higher-order bind.
3. **Raven's Progressive Matrices smoke** (MEDIUM, a flagship demo) — run the SPA-RPM recipe on the composer as a capstone reasoning benchmark. Anti-cheat: the rule must be *induced from the first two rows* and applied to a held-out third row; permuted rule loses.

---

### 2.6 The sequential "train of thought" ⭐ (the structural heart of "thinking")

**(a) Human capability.** A self-sustaining sequence of internally-generated states, each cueing the next, not driven by external input. "Let me think… dog → barking → mailman → yesterday's package…". This is what makes the system *think* rather than *retrieve-and-render* (directly the owner's "communicable brain, not RAG" reframe).

**(b) Biology.** Two complementary substrates:
- **Hippocampal sequence generation** — theta-paced sequence read-out (catalog **D.24 theta-paced sequence compression**, Bz Cycle 11 pp 313–323) and **awake replay during behavioral pauses** (catalog **N.17** — "~50% of all SWRs occur during waking immobility… forward replay of candidate trajectories = deliberative planning; reverse replay after reward = credit assignment"; Foster & Wilson 2006; Pfeiffer & Foster 2013). N.17 explicitly frames awake replay as **online deliberation**, the closest biological substrate to a train of thought.
- **Default-mode network (DMN)** — self-generated, stimulus-independent thought: mind-wandering, prospection, autobiographical simulation (catalog **G.09 imagination/future simulation**, Kandel Ch 52 pp 1300–1302; Christoff et al. 2016 *Nat Rev Neurosci* "Mind-wandering as spontaneous thought: a dynamic framework"; Buckner et al. 2008 "internal train of thought"; the DMN core = mPFC, PCC/precuneus, retrosplenial, lateral parietal/temporal, HC). The DMN *generates* the sequence; the **frontoparietal control network** constrains/steers it (deliberate vs spontaneous thought, Christoff dynamic framework).

**(c) Our sim.** `missing`. The closest existing piece is the dlPFC `elaborate` content-selection over an association graph — but that is a **single-step ranked retrieval**, not a self-cueing chain. The `SpikingLoopContextBuffer` (multi-turn WM) and a *persistent on-bridge spiking loop* exist (the one-brain composer runs ops without host round-trips) — so the **substrate for a self-cueing loop is present but unconnected**.

**(d) Options (ranked).**
1. **Associative chain-of-thought via self-cued attractor hops** (CHEAP-MEDIUM, highest conceptual leverage for "thinking"). Seed the persistent loop with a concept; let the strongest learned association become the next state (cleanup → attractor → re-cue), for K steps, emitting each. This is literally `query_chain` with the **agent choosing the next relation by association strength instead of the caller supplying it** — the single change that turns retrieval into thought. Reuses: persistent one-brain loop, cleanup/attractor, the association graph from `elaborate`, the multi-turn buffer. **Anti-cheat:** (i) **lesion the recurrent re-cue** → the chain dies after one step (proves self-cueing is load-bearing, the N.17 awake-replay-disruption test); (ii) **permuted association weights** → the chain wanders to non-associated concepts (proves it follows *learned* structure); (iii) reset/empty-seed → no spurious chain; (iv) the chain must be **reproducible-given-seed but content-sensitive** (different seeds → different, on-topic chains), distinguishing structure from noise.
2. **Forward-replay deliberation before answering** (MEDIUM, biologically N.17) — at a "choice point" (ambiguous query) run a few forward-replay candidate chains and pick the one with best support. Anti-cheat: disabling awake-replay specifically impairs ambiguous-query handling but not direct lookup (N.17's decisive behavioral dissociation).

---

### 2.7 Mental simulation / imagination / prospection

**(a) Human capability.** Construct a novel scene/event never experienced — recombine known elements. "Imagine a purple dog on the moon." Prospection: simulate a future ("if I go north, then east, I'll reach the goal").

**(b) Biology.** Catalog **G.09** (Kandel Ch 52 pp 1300–1302): "Recombines stored elements to simulate future events. Same network active for *remember last beach trip* and *imagine next beach trip* (Schacter/Addis/Buckner). HC dysfunction degrades both episodic recall AND novel-scene imagination." The **constructive-episodic-simulation** hypothesis: imagination = the episodic memory system run **generatively**, recombining stored components, in the DMN + HC. For spatial prospection specifically, **forward hippocampal replay** simulates trajectories (N.17; Pfeiffer & Foster 2013). Behavioral validation: HC-amnesic patients fail "imagine a new picnic" with impoverished scene detail.

**(c) Our sim.** `missing` as constructive recombination of items into *novel hypotheticals*. Sleep-replay infra can re-run *experienced* trajectories but does not recombine. HOWEVER: the **compose-perceived-content arc is exactly novel-fact construction** — the agent binds a freshly-perceived object into a NEW held-out fact via the composer algebra (`2026-06-16-navigate-to-compose-then-answer.md`, 6-seed GO). That IS combinatorial construction of something not stored. What is missing is **deliberately generating a hypothetical** (vs perceiving one) and **forward simulation for planning**.

**(d) Options (ranked).**
1. **Hypothetical composition via the composer** (CHEAP) — bind arbitrary role-fillers to construct a novel proposition ("purple dog") and reason about it, *flagged as hypothetical not asserted* (a bound MODALITY/hypothetical tag, exactly like the existing AFFIRM/NEGATE polarity tag). Reuses the compose-perceived machinery. **Anti-cheat:** the hypothetical must NOT contaminate the fact store (query for it as a *fact* → abstain; query within the hypothetical context → retrieve). Lesion the modality tag → hypotheticals leak into assertions (moat breach = fail).
2. **Forward-replay spatial prospection** (MEDIUM) — see §2.10 planning; same machinery as the train-of-thought forward chain but over the nav state space. Anti-cheat: simulated trajectory must match the actual outcome better than a permuted-transition model; lesion → planning collapses to reactive.

---

### 2.8 Commonsense reasoning

**(a) Human capability.** Reason with vast implicit world knowledge: "If I drop a glass it breaks"; "you can't be in two places at once". The frame problem / default reasoning.

**(b) Biology.** No localized substrate — commonsense is **semantic memory + schemas** distributed across association cortex (the **anterior temporal lobe hub-and-spoke**, Patterson/Lambon-Ralph — already cited in the project's generalization arc) plus hippocampal relational structure (D.02) and predictive-coding priors (E.21). It is *acquired* (statistical learning over enormous experience), not computed by a rule engine. Kandel frames world-knowledge as distributed cortical semantic representation (Ch 52, semantic memory).

**(c) Our sim.** `missing` at scale, but the **PPMI stream cortex is the right substrate in miniature** — it learns implicit associations from corpus statistics and generalizes. Commonsense here = "the learned code geometry + stored schemas answer plausibility questions." The hard wall is **scale and grounding** (genuine commonsense needs a large grounded corpus — the owner's "deep knowledge" build), not a missing mechanism.

**(d) Options (ranked).**
1. **Plausibility judgment over the learned code geometry** (CHEAP probe; not a full capability) — "is X plausible with Y?" answered by code similarity + stored-schema consistency, *with calibrated abstention* on the long tail (where codes are noise — the documented frequent-concept-vs-rare-tail split). **Anti-cheat:** plausible vs implausible pairs must separate above chance; rare-tail must abstain (do not confabulate plausibility from noise codes — this is the known honest gap). Largely deferred to the corpus/deep-knowledge build, NOT a standalone reasoning target.

---

### 2.9 Planning (search over future states)

**(a) Human capability.** Construct a multi-step action sequence to reach a goal, evaluating consequences before acting (Tower of London; route planning).

**(b) Biology.** **PFC + posterior parietal + BG + hippocampus.** Catalog **G.05 PPC — spatial planning, reach intention** (Kandel Ch 34 pp 826–832: "encodes spatial goal… persists across delay periods (planning)"); **G.07 pre-SMA — internally generated sequences** (Kandel Ch 34 pp 822–828). The model-level account: **model-based RL** uses an internal model to simulate outcomes (PFC/HC), vs **model-free** cached values (striatum) — the System-1/System-2 dichotomy (§2.11). **Forward hippocampal replay is the simulation engine** (N.17; Mattar & Daw 2018, below). The decisive computational result: **Mattar & Daw 2018, "Prioritized memory access explains planning and hippocampal replay"** (*Nature Neuroscience* 21, PMID 30349103) — replay order is set by **utility = need × gain**; forward replay (planning) and reverse replay (credit assignment) fall out of one prioritization rule. This unifies replay-as-planning with replay-as-learning and is directly implementable.

**(c) Our sim.** `missing` as deliberative planning. The nav agent is **reactive** (BG cascade selects the next move from current perception; the spiking accumulator commits). Goal-cells + dlPFC-WM exist (`partial`). No forward model, no simulation-before-acting, no prioritized replay.

**(d) Options (ranked).**
1. **Forward-replay planning on the nav substrate** (MEDIUM, biologically N.17 + Mattar-Daw). At a choice point, run forward replay of candidate trajectories through a learned transition model; pick the action whose simulated rollout has highest value. Reuses: the nav cascade, the place/landmark code, the persistent loop, R-STDP for the transition model. **Anti-cheat:** (i) **lesion forward replay** → planning collapses to reactive (the decisive N.17 dissociation: choice-point performance drops, direct approach spared); (ii) **permuted transition model** → plans become random; (iii) the simulated trajectory must *predict* the actual path taken (decode-match), not be post-hoc. **Flag:** this is a genuine research build (≥2 mechanisms composed), and the transition-model learning on point neurons is the open risk — fire the research gate before building.
2. **Prioritized replay ordering (need × gain)** (MEDIUM-HARD) — implement Mattar-Daw utility to schedule which memories replay. Anti-cheat: the empirical forward/reverse-replay balance must emerge from the utility rule, not be imposed (the paper's own validation). High value but research-grade.

---

### 2.10 System-1 vs System-2 (the dual-process frame)

**(a) Human capability.** Fast/automatic/parallel/intuitive (System 1) vs slow/deliberate/serial/effortful (System 2). Answering "2+2" (S1) vs "17×24" (S2).

**(b) Biology.** Maps cleanly onto **model-free (striatal, cached, habitual) vs model-based (PFC/HC, simulated, goal-directed) control** — the dominant computational neuroscience framing (Daw, Niv, Dayan 2005 *Nat Neurosci*; Dolan & Dayan 2013 *Neuron*; arbitration by reliability/cost). System 1 = the **BG cascade + cached action-values** (the project's existing nav selection). System 2 = **forward simulation / replay-based planning + relational reasoning in PFC-HC** (§2.5, §2.6, §2.9). The brain *arbitrates* between them by cost/uncertainty (vmPFC/dlPFC). Catalog: G.16/G.17 accumulator (the commit mechanism shared by both), C/O reward clusters (cached values).

**(c) Our sim.** **System 1 is essentially what the sim IS** — fast spiking selection, cached associations, reactive nav, retrieve-and-render conversation. **System 2 is the entire gap of this cluster** (analogy, train of thought, planning, deliberation). The project even has the *arbitration substrate* latent (the accumulator margin / familiarity gate could signal "this needs deliberation").

**(d) Options (ranked).**
1. **Frame the build as "add a minimal System 2 and an arbiter"** (META — this is the organizing principle, not a single build). The train-of-thought loop (§2.6) + forward-replay planning (§2.9) + composer-analogy (§2.5) ARE System 2; gate them on a **deliberation trigger** (low familiarity / ambiguous query / tie at the accumulator → engage the slow loop; else answer fast). **Anti-cheat:** the arbiter must engage S2 *only* when S1 is uncertain (manipulate ambiguity → S2 engagement tracks it); S2-lesion → only hard/novel queries degrade, easy lookups spared (the canonical dual-process dissociation). This reframe also satisfies the owner's "communicable brain not RAG" directive: S1 = the RAG-feel retrieval, S2 = the actual thinking.

---

## 3. Reusable project machinery (what to build ON, not from scratch)

| Need | Existing machinery | Source |
|---|---|---|
| Relational ops (bind/unbind/bundle/cleanup) — **the analogy + multi-hop primitive** | `RFPhasorComposer`, `OneBrainComposer` (FHRR resonate-and-fire, on-bridge, spiking) | `rf_phasor_composer.py`, `one_brain_composer.py` |
| Multi-hop pointer-chase | `query_chain` / `reason_chain` (GO, anti-cheated) | `2026-06-17-multihop-query-chain-GO.md` |
| Inductive category structure (codes that generalize) | PPMI online-stream cortex; cross-modal convergence | generalization arc CYCLE 88–96, `2026-06-16-generalization-*` |
| Schema-accelerated assimilation | `V_SCHEMA` (Tse 2007) | `2026-05-12-V_SCHEMA-*` |
| No-confab abstention (the control that exposes confabulation) | learned Bogacz-Brown familiarity gate | `2026-06-11-familiarity-gate-v320-GO.md` |
| Bounded accumulator (abductive logLR, decision commit) | BG cascade + Wang-2002/Lo-Wang spiking accumulator (GO) | `2026-06-19-spiking-decision-default-on-GO.md` |
| Persistent self-cueing loop (train of thought) | one-brain persistent spiking loop (no host round-trips) | `2026-06-18-one-brain-*` |
| Multi-turn discourse WM (induction buffer, chain context) | `SpikingLoopContextBuffer` | `2026-06-17-multiturn-anaphora-derisk-GO.md` |
| Content selection ("what to think/say next") | dlPFC spiking content-selection over the agent's association graph (`elaborate`) | `brain_conversational_agent.py` |
| Modality/polarity tagging (hypothetical vs asserted) | bound AFFIRM/NEGATE polarity role (reuse for HYPOTHETICAL) | `rf_phasor_composer.py` `ask_yes_no` |
| Replay infra (forward/reverse) | NREM sleep-replay scaffolding (content is the bottleneck) | catalog D.19/N.07/N.17 |
| Anti-cheat tooling | permuted-mapping check, lesion harness, derangement controls | `v16_compose_permuted_check.py`, `_genfrontier_*_derisk.py` |

---

## 4. Honest hard walls on a point-neuron substrate

1. **Graded-magnitude / rate-code SNR wall (documented, multiply-confirmed).** Fine *graded* quantities (precise logLR for abduction §2.3, graded confidence, analog evidence weights) are physically limited on point neurons — biology computes them in the analog/dendritic pre-spike stage (Mikulasch-Priesemann). Accumulation *works* (the nav accumulator is GO), but expect coarse precision; **honest-negative-characterize** rather than chase.
2. **Exact-inverse VSA algebra is an idealization, not a learned cortex.** The analogy/multi-hop ops ride on the FHRR clean-inverse algebra (`2026-06-06` known limitation). It buys the no-confab moat + compositional reliability ~free, but it is NOT a functional cortical binder. A *learned* relational binder generalizes single-attribute bindings on spikes (GO) but multi-attribute bundling is **not learnable from scratch** on point neurons (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`). So: **reasoning will run on the fixed algebra binding learned codes** — legitimate (binding-by-coincidence is a structural neural primitive), but flag it.
3. **Train-of-thought stability is an attractor-dynamics problem.** A self-cueing loop can die (under-excitation) or run away (epileptiform) — the CA3-autoassociator stability issue (D.05). Mitigate with cleanup-between-hops (already proven to stop error compounding in `query_chain`) + homeostasis; expect tuning.
4. **Commonsense + deep deduction need SCALE, not a new mechanism.** These are gated on the grounded-corpus / deep-knowledge build (owner's 2026-06-26 direction), not on a clever circuit. Do not over-invest in mechanism here.
5. **Default-mode "mind-wandering" has no clean behavioral validation gate** — it is the hardest to anti-cheat (how do you prove a spontaneous chain is "thought" vs noise?). The §2.6 anti-cheats (lesion-recurrence, permuted-association, reproducible-but-content-sensitive) are the best available; treat any DMN claim with maximum skepticism.

---

## 5. TOP 3 highest-leverage build targets

Ranked by **(leverage toward genuine thinking) × (substrate fit / cheapness) × (recovers credibility on the burned "reasoning" front)**.

### 🥇 #1 — Spiking proportional analogy A:B::C:? on the existing FHRR composer (§2.5)
**Why #1.** (i) **The mechanism is published, spiking, and already half-built here** — Eliasmith SPA solves Raven's matrices, Komer-Stewart do A:B::C:? in spiking neurons, both via `unbind → average-transform → apply`, which are the exact ops `RFPhasorComposer` exposes. (ii) Analogy is *the* signature of fluid reasoning — it converts the system from "follows a query plan the caller writes" to "computes a novel relational mapping," the qualitative jump from retrieval to reasoning. (iii) **Cheapest possible** — recomposition of existing GO ops, NO `sim/` edit, runs on the production composer. (iv) It directly **redeems the retracted reasoning claim** under a published mechanism + the mandatory permuted/held-out/lesion controls. Anti-cheat: held-out analogy pairs + permuted-relation rank-1/k + scrambled-source-to-chance + moat-abstains-on-no-target.

### 🥈 #2 — Associative chain-of-thought: self-cued attractor hops (§2.6)
**Why #2.** This is the **structural heart of "thinking"** and the owner's explicit north-star ("a communicable brain, not a RAG retrieve→render→abstain system"). The single conceptual change — *the agent picks the next relation by learned association strength instead of the caller supplying the action list* — turns `query_chain` (retrieval) into a train of thought (cognition). Biologically grounded in **N.17 awake-replay-as-deliberation** + DMN self-generated thought. Reuses the persistent one-brain loop + cleanup + association graph + multi-turn buffer (all GO/partial). Slightly harder than #1 (attractor stability tuning, wall #3) but uniquely high-leverage. Anti-cheat: lesion-recurrence kills the chain (the decisive awake-replay-disruption test) + permuted-association wanders off-structure + reproducible-given-seed-but-content-sensitive.

### 🥉 #3 — Transitive inference via a learned ordinal map (§2.4) — *the redemption build*
**Why #3.** This **directly fixes the project's most-burned capability** by replacing the artifact-prone explicit-chaining approach with the **biologically correct mechanism** (a learned 1-D/N-D cognitive-map geometry where Euclidean distance = inferred relational distance; Park 2020, Garvert/Behrens eLife 2016, Eichenbaum D.02). It reuses the PPMI/Hebbian code-geometry learning the project already has. It is *the* test case for the new anti-cheat discipline: held-out inferred pairs + permuted-order rank-1/N! + the **symbolic-distance effect** (accuracy grades with map distance — a signature no lookup-artifact can fake) + lesion. Succeeding here, under these controls, converts the 2026-05-14 retraction into a clean GO and establishes credibility for the whole cluster.

**Honorable mention / sequencing note.** Forward-replay **planning** (§2.9, Mattar-Daw need×gain) and the **System-1/System-2 arbiter** (§2.10) are the natural *next* tier — but planning is a genuine research-grade build (fire the research gate; transition-model learning on point neurons is the open risk), and the arbiter only makes sense once ≥1 System-2 capability (#1 or #2) exists to arbitrate *to*. Build #1→#2→#3 first; they share the composer + loop machinery and each is independently anti-cheatable.

---

## 6. Citation index

**Catalog (`feature-catalog.md`):** D.01 (episodic, pp 1296–1302), **D.02** (relational binding / transitive inference, Ch 52 pp 1301–1302, Eichenbaum/Cohen 2014), D.03–D.05 (trisynaptic, CA3 autoassociator), D.19 (SWR replay, Ch 54 pp 1365–1366), **N.17** (awake replay = deliberation; Foster & Wilson 2006; Pfeiffer & Foster 2013), N.18 (NREM nesting), **G.05** (PPC planning, Ch 34 pp 826–832), G.06/G.08 (PFC-WM, Ch 52 pp 1292–1294 / Ch 34 pp 827–842), G.07 (pre-SMA sequences, Ch 34 pp 822–828), **G.09** (imagination/DMN, Ch 52 pp 1300–1302, Schacter/Addis/Buckner), G.16/G.17 (drift-diffusion + LIP accumulator, Ch 56 pp 1399–1404), **G.18** (logLR symbol reasoning in LIP, Ch 56 pp 1404–1407, Fig 56-9), E.21 (Helmholtz constructive inference).

**Kandel 6e:** Ch 34 pp 835–836 (abstract-rule encoding in PFC/PMd, Wallis-Miller-style); Ch 52 pp 1292–1302 (working memory, episodic, imagination); Ch 54 pp 1340–1366 (hippocampal circuit, replay); Ch 56 pp 1393–1413 (decision-making, accumulators, consciousness/global-workspace).

**Primary literature (confirmed PMID / venue):**
- Mattar & Daw 2018, "Prioritized memory access explains planning and hippocampal replay," *Nat Neurosci* 21 — **PMID 30349103** (need×gain replay; unifies planning + credit assignment).
- Bunge / Wendelken / Badre on rostrolateral-PFC relational reasoning & analogy — PMIDs incl. 18052787, 26663572, 27012301 (rLPFC = relational-integration / analogy hub).
- Christoff, Irving, Fox, Spreng & Andrews-Hanna 2016, "Mind-wandering as spontaneous thought: a dynamic framework," *Nat Rev Neurosci* (DMN self-generated thought; FPCN steering).
- Buckner, Andrews-Hanna & Schacter 2008 (DMN, "internal train of thought").
- Park et al. 2020/2021 (*Neuron / Nat Neurosci*), Garvert/Behrens et al. 2016 *eLife* "A map of abstract relational knowledge in the human hippocampal–entorhinal cortex," Whittington et al. 2020 TEM (cognitive-map geometry → transitive inference / generalization).
- Daw, Niv & Dayan 2005 *Nat Neurosci*; Dolan & Dayan 2013 *Neuron* (model-based vs model-free = System 2 vs System 1).
- **Rasmussen & Eliasmith 2011** *Topics in Cognitive Science* + **Eliasmith 2013** *How to Build a Brain* (SPA solves Raven's Progressive Matrices in spiking neurons via unbind/average/apply); **Komer & Stewart 2020** arXiv 2010.06746 / IJCNN "Analogical and Relational Reasoning with Spiking Neural Networks"; **Hersche et al. 2023** *Nat Mach Intell* (neuro-vector-symbolic RPM, SOTA via bind/unbind/bundle). — **directly transferable to the project's existing FHRR composer.**

*(Glossary `glossary.md` was specified as a source but is ABSENT from the catalog directory — only `feature-catalog.md`, `biology-buildout-roadmap.md`, and `textbooks/` are present. Flagged per instructions.)*
