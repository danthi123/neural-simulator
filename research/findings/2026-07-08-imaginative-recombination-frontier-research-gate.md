# Research gate — IMAGINATIVE RECOMBINATION (constructive/generative memory) on the spiking substrate, MOAT-preserving (2026-07-08)

**Read-only deep-research gate** (mandated before a NEW mechanism-class direction). This is the gate for the **LAST
unaddressed open-world-inference mechanism** — #5 of the prior open-domain gate
(`2026-07-08-open-domain-grounded-conversation-frontier-research-gate.md`), which ranked five mechanisms and flagged
imaginative recombination via replay (catalog G.09) as "the deepest / deferred." Mechanisms #1–#4 are now
done/reframed: (a) spreading-activation semantic completion (runner
`_realcorpus_spreading_activation_completion_derisk.py` built), (b) schema/HTM default-filling (the predictor exists),
(c) analogical transfer (validated sound on clean codes, NEGATIVE on production codes — a code-geometry boundary), and
(d) multi-referent discourse (specified WTA mechanism). This gate does the FOUR surpass moves on #5.

**BOTTOM LINE (verdict, expanded in §4).** Imaginative recombination is **NOT one unbuilt mechanism** — it is a
capability the project has ALREADY touched from two ends, and the honest news is that **the propositional core is
already GO and the genuine residual is small + precisely localized.**

- **The propositional constructive-recombination core is DONE (host loop): `2026-06-23-genfrontier-b2-generative-replay-derisk`
  is GO 6-seed** — a hippocampal generative-replay *proposer* resamples role-filler bindings from the learned PPMI
  co-occurrence graph to INVENT novel-but-plausible SVO triples (`duck eat soup`, `wolf eat deer`), **17× over a
  random-recombination baseline**, shuffled-graph collapses to the random floor (the learned structure is
  load-bearing), and the **moat is preserved-and-upgraded** (0 proposal→known-fact leaks, 0 negated-fact
  re-proposals, proposals honestly flagged as a HYPOTHESIS channel). This already refutes the measured-0.0
  novel-composition ceiling with a brain mechanism. It is the FHRR/graph analogue of Schacter-Addis "detail
  recombination" and Spens-Burgess "combine unique + schema elements."
- **The genuine residual has THREE faces**, in increasing depth: **(R-i) SCENARIO/MULTI-ELEMENT recombination** (b2
  invents single triples; a *coherent unstated whole* — a bound multi-role scene / short event sequence — is the
  next expressive step, and the substrate's own FHRR bind/bundle + resonator cleanup can do it CHEAPLY);
  **(R-ii) COUNTERFACTUAL / conditioned recombination** ("what if the dog were a bird?" — recombination steered by
  a query, not free sampling — a cheap conditioning of the same sampler); **(R-iii) the FULLY-SPIKING-ON-SUBSTRATE
  replay loop** (the b2 recombination bookkeeping is host code; the on-bridge SWR→cortex→decode realization is a
  DOCUMENTED NEGATIVE, `2026-05-24-c-generative-replay-decisive-NEGATIVE`, at chance — the honest deep/deferred
  piece, and it is a *realization* residual, not a capability gap).

The single cheapest, highest-value next de-risk is **(R-i) SCENARIO recombination via the composer's own
bind/bundle + resonator cleanup**: extend the GO b2 proposer from single triples to a *coherent multi-role
composite* (agent+action+patient+location, or a 2-event mini-sequence) assembled from stored parts, gated by the
same graph-plausibility + non-contradiction moat and flagged as imagined — reusing `rf_phasor_composer` (bind/bundle/
unbind/cleanup) + the b2 plausibility gate + the moat, **NO `sim/` edit**. It is the one move that turns "invent a
novel fact" into "imagine a novel coherent little scenario," which is what "imaginative recombination" actually adds
over everything else the brain already does.

---

## 1. ISOLATE + QUANTIFY the true residual — what does "imaginative recombination" ADD?

The brain already has FIVE things that look adjacent to imagination. Pinning down what recombination adds ON TOP of
each is the whole diagnosis:

| Existing capability | What it does | What it does NOT do (the gap recombination fills) |
|---|---|---|
| **Spreading-activation completion** (#1, runner built) | Given a novel/partial cue → propagate to nearest learned neighbours → hedge a plausible property ("X is near {frog,fish} → probably swims") | COMPLETES one missing attribute of a GIVEN concept. Does NOT assemble a NOVEL multi-element WHOLE (a scene/event) that was never a concept. |
| **Analogical transfer** (#3; sound on clean codes) | `A:B::C:?` — map a relation from one pair to another | Requires a CLEAN minable relational axis; transfers ONE relation. Does not freely COMBINE many stored parts into a new configuration. |
| **HTM next-state predictor** (EMERGE-14/24) | Predict the high-order next symbol/state in a learned sequence | Predicts the LIKELY continuation of a SEEN prefix. Does not SAMPLE a never-seen combination on its own initiative (it interpolates the learned manifold forward, it doesn't recombine off-trajectory). |
| **VSA composer** (`rf_phasor_composer`) bind/unbind/bundle | Store + recall + verify role-filler facts; multi-hop `query_chain` | Retrieval algebra: unbinds/cleans up what was BOUND. Has "no generative path — cannot SAMPLE a never-stored token sequence" (`2026-06-22-generation-novelty-categorical-gap-MEASURED`: novel-composition = 0.0). |
| **Replay / consolidation** (Phase 1.3 SWR, engram, lineage) | Re-run stored trajectories to consolidate into cortex (no forgetting) | RE-runs the STORED; the on-bridge loop that would resample FICTIVE recombinations is a documented NEGATIVE (`2026-05-24`). |

**So the precise addition of "imaginative recombination" is: the ability to ASSEMBLE stored elements (fillers,
roles, relations) into a NOVEL coherent configuration that was never stored or seen — a proposition, a scene, or a
short event sequence — on the brain's own initiative (or conditioned on a query), gated by learned plausibility, and
FLAGGED as imagined.** Three distinct sub-capabilities, and the project sits at very different places on each:

- **(A) Novel PROPOSITION by recombination (single triple).** *Status: **GO** (host loop), `2026-06-23-genfrontier-b2`.*
  Quantified: novel-composition **0.752 mean** (vs 0.0 retrieval baseline), plausibility advantage **17× random**,
  shuffled-graph collapses to floor, moat 0-leak. **This is done.** It is exactly G.09 constructive recombination at
  the proposition grain.
- **(B) Novel SCENARIO / multi-element coherent whole (R-i) + COUNTERFACTUAL conditioning (R-ii).** *Status: **OPEN,
  cheap.*** b2 invents `wolf eat deer`; it does NOT assemble `the hungry wolf chases the deer to the river` (a bound
  4–5-role composite) or a 2-event mini-episode (`wolf chases deer → deer runs to river`), nor does it condition the
  recombination on a query ("imagine the dog as a bird"). The substrate's FHRR bind/bundle already assembles
  multi-role composites (the whole ditransitive/PP-relational store — EMERGE-72/74/77, `2026-07-08-{ditransitive,pp}-
  relation-store-GO`) and the resonator/cleanup factors them back. **Recombination = sample the FILLERS from the
  plausibility graph instead of reading them from a stored fact, then bind/bundle them into a fresh composite.** This
  is the genuine, cheap residual — the expressive jump from "novel fact" to "novel coherent little scene."
- **(C) Fully-spiking on-substrate replay loop (R-iii).** *Status: **NEGATIVE / deferred**, `2026-05-24`.* The b2
  recombination LOOP (the resample-from-graph bookkeeping) is host code; the fully-on-bridge realization
  (encode PFC frame → SWR-trigger CA3 → capture cortex → decode continuation) ran at **chance** (5.78% vs 6.25%). The
  bottleneck was precisely localized: **the SWR trigger does not drive sequence-SPECIFIC cortical activity the decoder
  can read** (three candidate causes: SWR not reactivating the right CA3 ensemble; consolidation not carrying
  slot-position structure; decoder reading background dynamics). This is a *realization* residual (the capability is
  proven at the b2/host grain; the deepest biology-purity version of the LOOP is unbuilt), not a capability gap.

**Net quantification.** The propositional core (A) is GO. The genuine, cheap, high-value residual is (B): scenario +
counterfactual recombination — a coherent multi-element imagined whole, which is what recombination uniquely ADDS
over completion/analogy/prediction. The deepest residual (C) — the fully-spiking SWR loop — is a documented NEGATIVE
with a diagnosed bottleneck, and is genuinely deep/deferred. **Do B first (cheap, on-substrate-composable, high
expressive value); gate C separately if biology-purity of the loop is later prioritized.**

---

## 2. REFRAME — how real biology + the field actually do imaginative recombination

The load-bearing reframe (consistent with the prior gate's "same substrate, different mode"): **imagination and
memory are the SAME constructive process** — Schacter-Addis's foundational finding. Imagining a novel scene and
recalling a real one recruit the same hippocampal-DMN core; the difference is only in WHICH elements get bound and
HOW freely. This means the substrate does NOT need a separate "imagination engine": it needs its EXISTING
bind/bundle/store/replay run in a **free-sampling / conditioned-sampling mode** instead of a retrieval mode. Four
converging accounts:

### 2a. Constructive episodic simulation = detail RECOMBINATION (Schacter & Addis 2007; Kandel G.09)
Schacter-Addis: "episodic memory supports the construction of future experiences by providing access to episodic
details that can be **recombined into novel events**." The experimental paradigm is literally **random rearrangement
of episodic details (persons, places, objects) from one's own memories** — the person/place/object slots are
recombined. **"Detail recombination is critical to imagining coherent scenarios."** Catalog **G.09** (Kandel 6e Ch 52
pp 1300–1302): "recombines stored elements to simulate future events; same network for remember-last-trip and
imagine-next-trip; HC dysfunction degrades BOTH episodic recall AND novel-scene imagination." ⇒ **Reframe #1: this is
exactly slot-filler recombination — sample a filler for each ROLE (agent/action/patient/location) from stored
options, bind them into a coherent composite.** The project's role-filler FHRR store IS the person/place/object slot
structure; recombination = resample the fillers. **Maps to spikes** (bind/bundle already validated for multi-role
composites). The hippocampal role is the "intensive relational processing required for integrating disparate details
into a coherent simulation" — i.e. the binding + the plausibility check.

### 2b. Generative model trained by replay (Spens & Burgess 2024, *Nat Hum Behav*)
Spens & Burgess: **hippocampal replay (from an autoassociative net) TRAINS a neocortical generative model (VAEs in
EC/mPFC/ATL)**; memories are then **(re)constructed by combining unique hippocampal details with schema-based
neocortical predictions**, which is why memory shows schema distortions + boundary extension that GROW with
consolidation. Crucially, the SAME generative model yields **"semantic memory, imagination, episodic future thinking,
relational inference and schema-based distortions"** — one engine, five products. **The imagined scene = a sample from
the generative model** (the schema fills the unstated typical elements; the hippocampus supplies the specific ones).
⇒ **Reframe #2: imagination = SAMPLE from the learned generative model = combine (specific stored filler) + (schema-
typical default).** The project's PPMI co-occurrence cortex IS the learned schema/likelihood (which fillers plausibly
co-occur); the SVO store IS the specific detail. **b2 already does exactly this** (sample fillers weighted by
graph-plausibility). The residual is only the *grain* (single triple → multi-role scene) and the *realization*
(host loop → spikes). **Maps to spikes at the b2 grain; the full loop is R-iii.**

### 2c. Compositional replay = binding PRIMITIVES into never-experienced states (George/Barry/Behrens 2023; Stoianov-Maisto-Pezzulo 2022)
"Constructing future behaviour in the hippocampal formation through composition and replay" (bioRxiv 2023): "if
state-spaces are constructed compositionally from **primitives**, hippocampal responses are compositional memories
**binding primitives together** … replay can **construct states the agent has NEVER experienced**." Stoianov-Maisto-
Pezzulo 2022 (the hippocampus as a hierarchical generative model): generative replay resamples FICTIVE sequences
INCLUDING never-experienced recombinations. Empirically, hippocampal internally-generated sequences depict
**never-exploited shortcuts, optimized paths never taken, and preplay of novel environments** (the search-confirmed
rodent literature: Pfeiffer-Foster, Ólafsdóttir, Gupta). ⇒ **Reframe #3: recombination = BIND stored primitives in a
NEW configuration.** This is the composer's exact operation — bind/bundle over role phasors. **The composer's
"no generative path" limit (it can't SAMPLE) is dissolved by ADDING a sampler over the fillers** (b2's move), then
binding the sampled fillers (the composer's native op). **Maps to spikes.**

### 2d. What is intrinsically the (minimized) GENERATOR's job (not the spiking substrate's)
The reframe has a hard honest edge, exactly as the prior gate drew it for open generation: **the substrate can invent
the CONTENT (a novel proposition / bound scene = G2), but the FLUENT SURFACE FORM of that imagined scenario (a
multi-clause narrative describing it) is the minimized generator's job (G3).** A recombined composite `bind(agent:wolf,
action:chase, patient:deer, loc:river)` is a spiking-substrate product; "The hungry wolf chased the deer down to the
river's edge" is the renderer's. The b2 finding already states this ("produces propositions G2, not fluent discourse
G3 — a proposed composite FEEDS a renderer for the surface form, still gated/verified"). ⇒ **The spiking mechanism =
invent + bind + plausibility-gate + flag-as-imagined the COMPOSITE; the generator = word it fluently.** Do not
over-scope imaginative recombination into open narrative generation — that conflates R2 (substrate) with R1
(generator) and would repeat the falsified zero-transformer over-claim.

**Net of §2.** Biology + the field converge: imagination = the SAME constructive machinery (bind stored primitives /
sample the generative model) run in a free/conditioned mode. The project has every ingredient (role-filler FHRR
bind/bundle, the PPMI plausibility graph, the moat, replay infra) and has ALREADY validated the proposition grain
(b2). The residual is (i) the scene/counterfactual grain (cheap, on-substrate-composable) and (ii) the fully-spiking
loop (a documented NEGATIVE, deep). The surface-fluency wording of an imagined scene is the minimized generator's job.

---

## 3. RANKED cheap-first mechanisms (each cited + reuse-by-import de-risk + MOAT/flagged-as-imagined argument)

Ordered by (value × cheapness). Each: mechanism · citation · reusable machinery · single-variable anti-cheated de-risk
(6-seed 42/43/44/100/101/102) · the flagged-as-imagined MOAT argument · `sim/`-edit-or-not.

### (R-i) ★ CHEAPEST + HIGHEST-VALUE — SCENARIO / MULTI-ELEMENT RECOMBINATION (bind sampled fillers into a coherent composite)
- **Mechanism.** Extend the GO b2 proposer from a single SVO triple to a **coherent multi-role composite**: sample a
  filler for EACH role (agent, action, patient, and location/instrument — the ditransitive/PP slots already in the
  store) weighted by the learned PPMI plausibility graph, then **bind + bundle them into ONE composite phasor** on
  the composer's native ops, cleaned up by the resonator. Two grains: (1) a 4–5-role SCENE (`hungry wolf chases deer
  to river`); (2) a 2-event MINI-SEQUENCE (`wolf chases deer → deer runs to river`) chained via `query_chain`. The
  composite is a novel coherent WHOLE the brain was never told — the thing recombination uniquely adds.
- **Citation.** Schacter & Addis 2007 (detail recombination — persons/places/objects into coherent scenarios);
  George/Barry/Behrens 2023 (bind primitives into never-experienced states); catalog **G.09** (constructive
  recombination; Kandel Ch 52 pp 1300–1302); Frady-Kent-Sommer resonator networks + "Learning and generalization of
  compositional representations of visual scenes" (arXiv 2303.13691 — VSA scene composites + resonator factoring,
  generalize to novel factor combinations); the project's own `2026-06-23-genfrontier-b2` (the single-triple GO to
  extend).
- **Reusable machinery.** `rf_phasor_composer.py` bind/bundle/unbind/cleanup (already assembles + factors multi-role
  composites — the EMERGE-72/74/77 ditransitive/PP store proves ≥4 roles bind + read back cleanly); the b2 proposer
  + its PPMI plausibility gate + non-contradiction gate (`_genfrontier_b2_generative_replay_derisk.py`); the PPMI
  co-occurrence cortex (`option_c_real_cooccurrence_derisk` / the real-corpus codes); `query_chain` (for the
  mini-sequence chaining). **The recombination is the b2 sampler; the SCENE assembly + read-back is the composer's
  already-validated multi-role bind/factor.**
- **Cheap-first de-risk.** Store a small fact set with location/instrument roles; run the extended proposer to
  assemble K novel multi-role composites; verify each factors back to its sampled fillers via the resonator (the
  scene is COHERENT — recoverable, not superposition mush). **GO bar:** novel-scene rate ≫ random-recombination
  baseline AND every emitted scene factors back to its constituents (cleanup recovers all roles) AND it is flagged
  imagined. **Anti-cheats (all mandatory):** (1) **shuffled-plausibility-graph** → scene plausibility collapses to
  the random floor (learned structure load-bearing — the exact b2 control, extended); (2) **factor-recovery**
  (the resonator must recover ALL sampled roles; a composite that doesn't factor back is superposition noise, not a
  coherent scene — this is the R-i-specific gate that guards against "bundle mush"); (3) **role-count stress** (does
  coherence hold at 4/5 roles, or does the FHRR bundle SNR wall bite? — an honest capacity boundary to MEASURE, cf.
  the documented K=5 two-attribute bundle limit `2026-06-04`); (4) **MOAT** (a composite whose fillers are
  graph-implausible is NOT emitted; 0 emitted scene passes `query_patient` as a known fact); (5) 6-seed.
- **MOAT / flagged-as-imagined argument.** IDENTICAL to the b2 GO trade (owner-sanctioned,
  `feedback_moat_not_hard_lossy_memory_ok`), extended to scenes: a recombined scene is a SEPARATE, honestly-flagged
  HYPOTHESIS channel — it must NEVER pass the composer's KNOWN-fact retrieval (`query_patient`/`ask_yes_no` still
  abstain on it → 0 leaks), it must NOT contradict a stored fact (non-contradiction gate), and it is surfaced with an
  explicit hedge ("I imagine that … — I wasn't told this"). It can never assert a fabricated scene as a stored fact.
  The graph-plausibility gate + the factor-recovery gate together are the causal guards (lesion either → floods
  incoherent/implausible scenes).
- **`sim/` edit:** NONE (b2 sampler + composer native bind/bundle/cleanup + moat, all reuse-by-import).
- **Why first:** it is the ONE move that turns the GO single-triple proposer into genuine "imagine a novel coherent
  little scenario," reuses the two most-validated assets (the composer + the PPMI graph + the b2 gate), the
  factor-recovery anti-cheat is airtight, and it directly delivers what recombination adds over completion/analogy/
  prediction. Cheap (CPU, ~1–2 days), on-substrate-composable, moat-preserved.

### (R-ii) COUNTERFACTUAL / QUERY-CONDITIONED recombination (steer the sampler by a substitution)
- **Mechanism.** Instead of FREE sampling, CONDITION the recombination on a query substitution: "imagine the dog were
  a bird" = take a stored fact `bind(agent:dog, action:eat, patient:meat)`, unbind the agent, substitute a
  query-supplied filler (`bird`), re-bind, and check plausibility ("would a bird eat meat? — graph says no → imagine
  it eats {seed,worm} instead", re-sampling only the incoherent slots). This is imagination DIRECTED by a question,
  the conversational form of "what if …".
- **Citation.** Schacter-Addis (episodic simulation is goal/cue-directed, not purely random); Spens-Burgess (the
  generative model can be CONDITIONED on a partial cue → fills the rest); the composer's `unbind`→substitute→`bind`
  is the `2026-06-27-tier2.1-analogy` transform mechanism run in a substitution mode (validated sound on clean binds).
- **Reusable machinery.** `rf_phasor_composer` unbind/bind (the analogy-transform ops, proven faithful through the
  real RF `_bind`); the R-i sampler (to re-fill slots the substitution made implausible); the b2 plausibility + moat.
- **Cheap-first de-risk.** A "what if X were Y?" prompt → substitute → re-plausibility-check → emit a coherent
  counterfactual composite, flagged. **Anti-cheats:** substitution-lesion (skip the re-plausibility → emits incoherent
  counterfactuals — the check is load-bearing); shuffled-graph collapses; the moat abstains when NO plausible re-fill
  exists (a truly-nonsense counterfactual → "I can't imagine that"); 6-seed.
- **MOAT argument.** A counterfactual is doubly-flagged (hypothesis channel + "what-if" framing); it never enters the
  known-fact store; an un-re-fillable substitution abstains. Preserved.
- **`sim/` edit:** NONE. Do after R-i (it reuses R-i's re-sampler). Bounded scope: works where a plausible re-fill
  exists.

### (R-iii) FULLY-SPIKING ON-SUBSTRATE REPLAY LOOP (the documented NEGATIVE — deepest, research-gate before rebuild)
- **Mechanism.** Realize the recombination LOOP fully on the bridge: SWR-gated CA3 replay reactivates stored engrams →
  cortical activity captured → decoded into a recombined continuation, iterated (Schwartenbeck three-stage
  refinement). This is the biology-purest version (the b2 loop's recombination bookkeeping is host code).
- **Status: NEGATIVE, diagnosed.** `2026-05-24-c-generative-replay-decisive-NEGATIVE`: at chance (5.78% vs 6.25%),
  bottleneck = **the SWR trigger does not drive sequence-SPECIFIC cortical activity the decoder can read.** Three
  diagnosed candidates: (1) SWR-trigger doesn't reactivate the right CA3 ensemble (the loop's `trigger_swr_replay`
  omits the explicit CA3 drive the *validated* Phase 1.3 `run_concept_replay_phase` uses — the top-likelihood fix);
  (2) consolidation doesn't carry slot-position structure; (3) decoder reads background dynamics.
- **Citation.** Schwartenbeck et al. 2023 *Cell* (generative replay → compositional inference, three-stage
  refinement; `2026-05-24-Schwartenbeck-...` reference doc); catalog D.19 / N.07 (SWR replay); the project's NEGATIVE.
- **Reusable machinery.** The validated Phase 1.3 SWR consolidation (`consolidation_trainer.run_concept_replay_phase`,
  3/3 strict anti-cheat) — the diagnosed fix is to give the (c) loop the SAME explicit CA3 drive that the validated
  mechanism uses; engram tags (D.14); per-slot positional binding (D.01+D.02+D.11, already validated as substrate).
- **De-risk (the diagnosed NEXT step, ~30–45 min CPU, from the NEGATIVE doc):** the **SWR-reactivation probe** — load
  the consolidated substrate, run SWR for ONE tag, measure post-replay cortical similarity to (a) the correct engram,
  (b) other engrams, (c) random. HIGH-to-correct/LOW-to-others → the decoder is the fix; LOW-to-correct → the
  SWR-trigger is the fix (add the explicit CA3 drive). This localizes the fix BEFORE any rebuild.
- **MOAT argument.** Same flagged-hypothesis channel; the loop's output is gated + verified identically. No new moat
  risk (the moat lives in the gate, not the loop).
- **`sim/` edit:** likely NONE (the diagnosed fix — explicit CA3 drive in `trigger_swr_replay` — is a ~10-line RUNNER
  change; positional structure reuses validated substrate). Deep/deferred: research-gate the rebuild AFTER R-i/R-ii
  prove the capability at the composable grain; the fully-spiking loop is a biology-PURITY upgrade of an
  already-demonstrated capability, not a prerequisite for the conversational win.

### (R-iv) [SCAFFOLD, not a spiking mechanism] FLUENT NARRATION of an imagined scenario = the minimized generator
- **Verdict.** Wording a recombined composite as fluent multi-clause prose is R1 (the minimized generator's job), per
  §2d + the prior gate. The spiking mechanism produces + gates + flags the imagined COMPOSITE (R-i/R-ii); the
  generator renders it, still gated/verified. Not a new spiking mechanism; listed so it is not confused with R-i–R-iii.

---

## 4. VERDICT per mechanism — surpassable-cheaply vs genuinely-deep/deferred vs the generator's job

| # | Mechanism | Residual it closes | Verdict | `sim/` edit |
|---|---|---|---|---|
| (A) | Novel PROPOSITION recombination (single triple) | Invent a novel plausible fact | **DONE — GO 6-seed** (`2026-06-23-b2`), host loop, moat-preserved | NONE |
| (R-i) | SCENARIO / multi-element recombination (bind sampled fillers) | Imagine a novel COHERENT whole (scene / mini-sequence) | **SURPASSABLE-CHEAPLY (spiking-composable); do FIRST** | NONE |
| (R-ii) | COUNTERFACTUAL / query-conditioned recombination | Directed "what if …" imagination | **SURPASSABLE-CHEAPLY**; after R-i; bounded (needs a plausible re-fill) | NONE |
| (R-iii) | Fully-spiking on-substrate SWR replay loop | Biology-PURITY of the loop | **DOCUMENTED NEGATIVE, DIAGNOSED**; deep/deferred; diagnosed fix is cheap (probe first) | likely NONE (runner) |
| (R-iv) | Fluent NARRATION of an imagined scenario | Open prose describing the scene | **THE MINIMIZED GENERATOR'S JOB** — scaffold, not a spiking-circuit gap | (generator) |

**The single cheapest, highest-value next de-risk: (R-i) SCENARIO / MULTI-ELEMENT RECOMBINATION.**

> **Runner:** new `research/runners/_imaginative_scenario_recombination_derisk.py` — reuse-by-import: the b2 proposer
> + PPMI plausibility gate + non-contradiction gate (`_genfrontier_b2_generative_replay_derisk.py`); the composer
> multi-role bind/bundle/unbind/resonator-cleanup (`rf_phasor_composer.py`; the EMERGE-72/74/77 ditransitive/PP store
> proves ≥4-role composites bind + factor back); `query_chain` for the mini-sequence grain; the PPMI real-corpus
> co-occurrence cortex. **NO `sim/` edit. CPU (`SIM_BACKEND=numpy`), ~1–2 days.**
>
> **Setup.** Store a small fact set with agent/action/patient + location/instrument roles. Run the extended proposer:
> for each of K attempts, SAMPLE a filler per role weighted by the learned PPMI graph (b2's sampler), BIND+BUNDLE them
> into one composite phasor (composer native ops), and accept iff NOVEL (never stored), PLAUSIBLE (each adjacent
> role-pair graph-related), NON-CONTRADICTORY (not an explicitly-negated fact), and COHERENT (the resonator factors
> the composite back to ALL sampled fillers). Emit accepted composites as a flagged HYPOTHESIS channel ("I imagine …").
>
> **GO bar.** Novel-coherent-scene rate ≫ random-recombination baseline (the b2 17× ratio, extended to multi-role),
> factor-recovery = 1.0 on accepted scenes, clearly flagged imagined, moat 0-leak.
> **Anti-cheats (all mandatory):** (1) SHUFFLED-plausibility-graph → scene rate collapses to random floor (learned
> structure load-bearing); (2) FACTOR-RECOVERY (resonator recovers all roles — guards against bundle mush; the
> R-i-specific gate); (3) ROLE-COUNT stress (measure coherence at 3/4/5 roles — the honest FHRR-bundle capacity
> boundary, cf. K=5 two-attr limit); (4) MOAT (implausible-filler composite NOT emitted; 0 scene passes known-fact
> retrieval); (5) 6-seed (42/43/44/100/101/102).
>
> **What GO shows.** The FIRST time the brain imagines a novel COHERENT SCENARIO (not just a novel fact) from stored
> parts — the capability recombination uniquely adds over completion/analogy/prediction — on its own composer's
> spiking bind/bundle ops, with the moat preserved-and-upgraded (a flagged hypothesis channel that never asserts the
> imagined as stored). Reuse-by-import, NO `sim/` edit.

**Chain after (R-i) GO** (all reuse-by-import): (R-ii) counterfactual conditioning → then, IF biology-purity of the
loop is prioritized, gate (R-iii) starting with the cheap SWR-reactivation probe (~30–45 min) that the NEGATIVE doc
already specified. The imagined composites from R-i/R-ii FEED the minimized generator (R-iv) for fluent narration,
still gated/verified — orthogonal track.

**Honest scope / where the walls are real.**
- **(i) The propositional core is already GO** — do NOT re-derive it. This gate's value is isolating that the residual
  is the SCENE/COUNTERFACTUAL grain + the spiking-loop realization, not the whole mechanism.
- **(ii) The fully-spiking SWR-replay LOOP is a documented NEGATIVE** (`2026-05-24`), diagnosed to the SWR-trigger not
  driving sequence-specific cortical activity. Do NOT rebuild it blind — run the diagnosed reactivation probe FIRST.
  Claiming a fully-spiking recombination loop without that probe would repeat the "compose the components and hope"
  pattern the NEGATIVE already refuted.
- **(iii) The FHRR-bundle capacity is a real, measured boundary** — bundling many fillers into one composite hits the
  superposition-SNR wall (the K=5 two-attribute limit, `2026-06-04`; the b2 "conjunctive plausibility makes absolute
  fractions small"). R-i's role-count stress MEASURES this honestly; a 5-role scene may need the resonator's O(M)
  factoring or a 2-composite chaining rather than one giant bundle. Report the coherent-scene role-count ceiling, do
  not force it.
- **(iv) Fluent open narration of an imagined scene is the minimized generator's job** (R-iv / R1) — the spiking
  substrate invents + binds + gates + flags the COMPOSITE; the generator words it. Conflating the two would repeat the
  falsified zero-transformer over-claim.
- **(v) The moat is preserved-and-upgraded, not weakened** — every recombined item is a flagged hypothesis channel
  that never passes known-fact retrieval and never contradicts a stored fact (the b2 guarantee, extended). This is the
  owner-sanctioned reconstructive-memory trade (`feedback_moat_not_hard_lossy_memory_ok`), biologically correct (real
  memory confabulates *plausibly*), NOT a loosening of the no-confab guarantee below "no unsupported claim asserted as
  fact."

---

## Artifacts / key citations

**Project (reusable machinery + prior findings — the decisive ones):**
`research/runners/_genfrontier_b2_generative_replay_derisk.py` + `2026-06-23-genfrontier-b2-generative-replay-derisk.md`
(**the GO single-triple constructive-recombination core to extend** — 17× over random, shuffled-graph collapses, moat
0-leak); `2026-05-24-c-generative-replay-decisive-NEGATIVE-…md` (**the fully-spiking SWR-loop NEGATIVE + the diagnosed
reactivation-probe fix** — R-iii); `2026-05-24-Schwartenbeck-2023-biology-reference-…md` (three-stage iterative
refinement for R-iii); `rf_phasor_composer.py` (bind/bundle/unbind/resonator-cleanup — the multi-role composite
assembly + factor-back); `2026-07-08-{ditransitive-ternary,pp-spatial}-relation-store-GO` (≥4-role composites bind +
read back cleanly — the substrate basis for R-i); `_realcorpus_spreading_activation_completion_derisk.py` +
`factored_relation_analogy.py` + `2026-06-27-tier2.1-analogy-NEGATIVE.md` (the adjacent #1/#3 mechanisms + the
unbind→substitute→bind transform R-ii reuses, and the code-geometry boundary); `option_c_real_cooccurrence_derisk` /
the real-corpus PPMI codes (the learned plausibility graph); `consolidation_trainer.run_concept_replay_phase`
(the validated explicit-CA3-drive SWR replay — the diagnosed R-iii fix source);
`2026-06-22-generation-novelty-categorical-gap-MEASURED.md` (the 0.0 baseline b2 refuted);
`2026-06-04` (the K=5 FHRR-bundle capacity boundary R-i must measure);
`2026-07-08-open-domain-grounded-conversation-frontier-research-gate.md` (the parent gate; #1–#4).

**Catalog (Kandel 6e / O'Keefe-Nadel):** **G.09 imagination / future simulation as constructive memory
(Ch 52 pp 1300–1302 — "recombines stored elements; same network for remember + imagine; missing")**;
D.02 relational binding / transitive inference (Ch 52 pp 1301–1302); D.13 CA3 pattern completion (Marr 1971;
Ch 54 pp 1342, 1360–1361); D.14 engram tagging (Ch 54); D.19 sharp-wave ripples / replay (Ch 54 pp 1365–1366);
N.07 hippocampal SWRs — NREM replay events (⭐, "replay quality not quantity is the bottleneck"); D.01+D.02+D.11
positional/relational binding (validated substrate — R-iii slot structure); J.34 schemas / gist (Ch 52 pp 1306–1308);
N.14 systems consolidation (Ch 52 p 1299, Ch 54 p 1366).

**Literature (current, cited):** Schacter & Addis 2007 (*Phil Trans R Soc B* — constructive episodic simulation
hypothesis: detail recombination into novel coherent scenarios); Schacter et al. 2008 (*Ann NY Acad Sci* — episodic
simulation of future events); Addis et al. 2007 (*Neuropsychologia* / PubMed 18157862 — hippocampal engagement scales
with detail recombination); **Spens & Burgess 2024 (*Nat Hum Behav* 8, s41562-023-01799-z — replay trains a
neocortical generative model; combine unique + schema elements → semantic memory, imagination, episodic future
thinking, relational inference, boundary extension)**; Stoianov, Maisto & Pezzulo 2022 (*Prog Neurobiol* — hippocampus
as a hierarchical generative model; generative replay resamples fictive/never-experienced sequences);
George, Barry, Behrens et al. 2023 (bioRxiv 2023.04.07.536053 — "Constructing future behaviour through composition and
replay"; bind primitives → construct never-experienced states); Barry & Love 2023 (*Nat Hum Behav* — generative memory
construction); Schwartenbeck et al. 2023 (*Cell* PMC10914680 — generative replay → compositional inference, three-stage
refinement); Frady, Kent, Olshausen & Sommer 2020 (resonator networks — factor a composite VSA vector, O(M));
"Learning and generalization of compositional representations of visual scenes" (arXiv 2303.13691 — VSA scene
composites generalize to novel factor combinations); van de Ven et al. 2020 (*Nat Commun* — brain-inspired generative
replay prevents catastrophic forgetting); Aslam et al. 2025 (*Int J Intell Syst* — continual-learning-inspired-by-brain
survey); Eldan & Li 2023 (*TinyStories* — the minimized-generator scaffold for R-iv narration).
