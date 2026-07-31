---
type: plan
status: live
date: 2026-06-12
---

# Phase 1 of "step 3 true cortex" — COMPOSER ARCHITECTURE at production scale (2,048 concepts) + the A-vs-B de-risk that decides it

> **Status:** present-before-build. READ-ONLY design (no `sim/` edit, no GPU run, no bridge built). The single
> deliverable is this doc + one commit. This is the FIRST decision of Phase 1 of the production build: at
> 2,048 concepts (= 32 bridges × 64), does the FHRR composer stay **one scaled union composer (route B)** or split
> into **per-bridge composers + a cross-bridge identity layer (route A)**? It is the project's standing
> "design + cheap-first de-risk BEFORE building" opening move — the cheapest run (at 8 bridges = 512 concepts, NOT
> 32) that decides the architecture before the multi-week 32-bridge spend. **Date:** 2026-06-12.
> **Author role:** read-only design subagent. Every load-bearing claim is cited to a file read in full.

---

## 0. Terms (defined once)

- **FHRR** — Fourier Holographic Reduced Representation: the vector-symbolic algebra the composer uses. Each
  concept/role is a unit-magnitude *phasor* vector (phases in `[0,1)^D`). **bind** = element-wise complex product
  (role ⊗ filler), **unbind** = multiply by the conjugate, **bundle** = sum of phasors, **cleanup** = nearest
  codebook entry by phase-cosine argmax. Implemented in `research/runners/rf_phasor_composer.py`
  (`_bind` lines 117–126, `_bundle` 128–136, `_unbind_phases` 159–167, `_cleanup` 247–252).
- **D** — the composer's phasor dimension (`RFPhasorComposer.D`, line 62). The capacity/SNR knob.
- **Composer** — one `RFPhasorComposer` instance: a vocabulary of phasor codes + a knowledge base `kb` of bound
  composites + the bind/unbind/cleanup/abstention ops (`rf_phasor_composer.py:61`).
- **Bridge / shard** — one of the N small spiking pools (64 concepts each) that the learned-graded cortex is
  sharded into, to dodge the single-pool quadratic memory wall
  (`docs/plans/2026-06-11-semantically-structured-cortex-BUILD-PLAN.md` §"Scaling path", lines 78–101). The
  corpus is **sharded by semantic cluster** (animals together, foods together) so within-bridge graded similarity
  (cat≈dog) is meaningful (build plan §"Genuine open questions" item 1, line 94).
- **Within-bridge fact** — an SVO fact (agent–action–patient) whose content concepts all live in ONE shard
  (e.g. `cat eats meat`, cat and meat both in `animals`… or in a shard where both are co-located).
- **Cross-bridge fact** — an SVO fact whose content concepts span two shards (e.g. `dog eats meat`, dog∈animals,
  meat∈foods).
- **V-tag layer** — the validated cross-bridge composition mechanism: a Tonegawa engram-tag (catalog D.14) named
  `"<cue>__<target>"` imprinted in BOTH the cue's bridge and the target's bridge over each bridge's spiking `pool`
  region; recall stimulates the tag and reads per-concept firing in the target bridge (the "V-tag" key→value
  store). `GradedBridge` + `cross_bridge_eval` in `research/runners/multibridge_graded_derisk.py:446` and `:551`.
  This is **identity recall** (which concept), NOT generative role-filler binding.
- **Generative VSA binding** — composing a NEW bound structure with the FHRR algebra (bind/bundle), recoverable
  by unbind/cleanup. The thing the composer does; the thing the V-tag layer does NOT do.
- **The moat** — the no-confab abstention: the composer returns `None` / `"unknown"` when no stored fact matches
  a query (`rf_phasor_composer.py:310, 323, 332, 347`), and the learned `RelationalFamiliarityGate` validated
  alongside it (`research/runners/familiarity_gate_v320_validation.py`, V=320 zero breaches).

---

## 1. Where this sits (what is already decided)

The dual / complementary-learning-systems (CLS) learned-graded cortex is **de-risked end-to-end** — mechanism AND
capability — at small scale:

| Piece | Validation (file read in full) | Status |
|---|---|---|
| Cross-bridge composition + the moat **survive on correlated graded codes** | `research/findings/2026-06-12-multibridge-graded-derisk-GO.md` (3 bridges × 64, 3 seeds) | GO |
| The full conversational matrix on the learned cortex + **generalization-in-conversation** + moat | `research/findings/2026-06-12-cortex-conversation-capability-GO.md` (1 shard × 64, 3 seeds) | GO |
| The 3-bridge ENSEMBLE de-risk (matrix spanning bridges + within-bridge generalization + cross-bridge + moat) | `research/runners/cortex_conversation_ensemble_derisk.py` (the just-finishing run, GO at D=512) | GO |
| The integration architecture (cortex → DG-decorrelate → composer → reinstatement → moat → dialogue) | `docs/plans/2026-06-12-cortex-conversation-integration-design.md` | DESIGNED |
| The build plan + its scaling path (single-pool wall → 32-bridge multi-bridge) | `docs/plans/2026-06-11-semantically-structured-cortex-BUILD-PLAN.md` | APPROVED-PENDING-BUILD |

**What is NOT yet decided — and is THIS doc's job.** The integration design (§1.3) and the ensemble runner both
assume **ONE union `RFPhasorComposer` over the union vocabulary** (`EnsembleCortexAgent.__init__`,
`cortex_conversation_ensemble_derisk.py:202`: `composer = RFPhasorComposer(... vocab=union_words ...)`). That was
correct for 3–8 bridges (union ≤ 512 concepts). **At 32 bridges = 2,048 concepts a single union composer's
dimension must scale with the union vocabulary** — which is the learning that triggered this design.

### 1.1 The learning that triggers this design (from the ensemble de-risk)

The 3-bridge ensemble de-risk's conversational matrix (Gate A) **failed its `clause` cell at composer dimension
D=128 over the 192-concept union vocabulary, and passed at D=512.** This is the concrete observation that the
FHRR composer dimension must scale with the union-vocabulary size, at a rate of roughly **D/concept ≈ 512/192 ≈
2.7** in that run. The mechanism is FHRR's well-known capacity law: bundling L bound terms and cleaning up against
a V-entry codebook needs D large enough that the recovered phasor's signal clears the cross-talk from the other
bound terms AND the other V−1 codebook entries. As the union vocabulary V grows, the cleanup's argmax over V
entries needs more D to stay reliable — and the `clause` cell is the worst case (a clause binds a nested SVO, so
its bundle has the most terms and the deepest unbind, `rf_phasor_composer.py:139–146, 148–157`).

**Extrapolated to 2,048 concepts that is D ≈ 2,048 × 2.7 ≈ 5,000–6,000.** A single union composer at D ≈ 5.5k is
the route-B object this design must size and weigh against the per-bridge alternative.

### 1.2 What the ensemble used (and why it does not settle the question)

The ensemble runner built **one union composer** with **every shard's DG-decorrelated phase codes merged into one
`grounded_codes` dict** (`EnsembleCortexAgent.__init__`, lines 191–204), so the composer can bind BOTH
within-shard and cross-shard SVO facts generatively. The matrix in the ensemble (`_make_union_codebook_for_matrix`,
lines 541–578) deliberately **interleaves the shards' words** so the matrix's SVO roles are drawn from DIFFERENT
bridges — and the `clause` cell (`gate_A_matrix`, `cortex_conversation_capability_derisk.py:426, 458–462`) binds
`Clause(words[5], words[1], words[2])` where, under the interleave, those three words come from THREE different
shards. **So the ensemble's passing matrix used route B with a CROSS-BRIDGE generative clause.** That tells us
route B *works at 512 concepts*; it does NOT tell us whether the capability *requires* the cross-bridge clause to
be generative, nor whether route B's D ≈ 5.5k holds the matrix at 2,048. Both are this design's crux + de-risk.

---

## 2. The two candidate architectures (+ a hybrid)

### 2.1 Route A — per-bridge composers + cross-bridge V-tag identity layer

**Shape.** Each of the 32 bridges owns its OWN small `RFPhasorComposer` (D ≈ 128–512) over its **64 concepts +
shared auxiliary vocab** (the action verbs / polarity tags / property words — `MATRIX_ACTIONS`,
`rf_phasor_composer.py:91` `pol_words`). WITHIN-bridge facts, clauses, and attributes bind generatively in that
bridge's composer (full FHRR, but over a 64-concept codebook). CROSS-bridge facts go through the validated V-tag
**identity** layer (`GradedBridge.encode_tag` / `cross_bridge_eval`, `multibridge_graded_derisk.py:480, 551`):
store `dog eats meat` as the shared tag `animals.dog__foods.meat` in both bridges; recall by stimulating the tag
and reading the target bridge's firing.

**Component mapping (what each piece becomes).**

| Function | Route-A realization | Source class |
|---|---|---|
| within-bridge bind / unbind / cleanup / clause / attribute | per-bridge `RFPhasorComposer` over 64 concepts | `rf_phasor_composer.py:61` (verbatim, 32 instances) |
| within-bridge generalization (cat≈dog) | per-bridge graded cortex codes, read directly | `CortexCodebook` + `similarity_vote_infer` (integration design §1.3) |
| within-bridge moat | per-bridge composer abstention + per-bridge `RelationalFamiliarityGate` | `familiarity_gate_v320_validation.py` (one gate per bridge) |
| cross-bridge fact store/recall | V-tag engram-tag layer over the spiking `pool` regions | `GradedBridge` + `cross_bridge_eval` (`multibridge_graded_derisk.py:446, 551`) |
| cross-bridge moat | the V-tag M4 moat over cross-facts (already validated) | `moat_eval` (`multibridge_graded_derisk.py:644`) |
| routing a query to the right bridge | a `word → shard` map (already built) | `EnsembleCortexAgent.word_to_shard` (`cortex_conversation_ensemble_derisk.py:188–194`) |

**What changes vs the ensemble runner.** The ensemble's ONE union composer is replaced by a `{shard:
RFPhasorComposer}` dict + a query router. The `EnsembleCortexAgent` already holds the `{shard: CortexCodebook}`
dict and the `word_to_shard` map — route A just additionally holds a `{shard: RFPhasorComposer}` and dispatches
`store_fact` / `what_does` / `who_does` to the agent-word's shard's composer (within-bridge), falling back to the
V-tag layer when the fact's roles span shards. This is **new glue in the agent wrapper, no `sim/` edit, no new
mechanism** (every piece is validated; only the dispatch is new).

**Cost (analytic, computed below in §4.2).** 32 composers each at D≈128–512 → codebook 2.4–9.4 MB total; each
bind is a 2D≈256–1024-neuron RF op; each cleanup is argmax over **64** concepts. The V-tag layer adds 2 engram
tags per cross-fact (top_k≈150 indices), NO D-scaling. **Scalable: every composer's cost is independent of total
vocabulary** (D set by the 64-concept shard, not the 2,048-concept union).

**The honest limitation.** Cross-bridge composition is **identity recall, NOT generative VSA binding across
bridges.** A cross-bridge fact `dog eats meat` is stored as a key→value tag and recalled as "the target is meat" —
the agent cannot *generatively* bind a NEW cross-bridge structure it never stored (e.g. answer a cross-bridge
**clause** "the dog that eats meat runs", with dog∈animals and meat∈foods, by composing it on the fly). Whether
the capability needs that is §3.

### 2.2 Route B — one scaled union composer

**Shape.** A single `RFPhasorComposer` over all 2,048 concepts at D ≈ 5,000–6,000 (§1.1). Within- AND cross-bridge
facts/clauses all bind generatively in the union composer — full FHRR across all 2,048 concepts. This is the
ensemble runner's `EnsembleCortexAgent` extended to 32 shards (`cortex_conversation_ensemble_derisk.py:170–215`),
verbatim except the union vocab is 2,048 and D ≈ 5.5k.

**Component mapping.** Exactly the ensemble runner today: one `RFPhasorComposer(vocab=union_2048, D≈5500,
grounded_codes=all_32_shards_phases)`; per-bridge `CortexCodebook`s for the within-bridge generalization read; one
union `RelationalFamiliarityGate` for the moat. The cross-bridge facts are stored as relational SVO in the union
composer (`gate_X_conv`, `cortex_conversation_ensemble_derisk.py:336–388`) — generative, not V-tag.

**What changes vs the ensemble runner.** Only the scale (3→32 shards, union 192→2,048, D 512→~5.5k). No code-path
change. This is the path of least *code* change.

**Cost (analytic, §4.1).** Codebook ≈ 90 MB (2,048 × D≈5.5k float64). **The load-bearing cost is NOT the
codebook** — it is (i) every bind builds a **2D ≈ 11,000-neuron** RF bridge and every bundle a **(L+1)D ≈
22,000-neuron** bridge (`_bind`/`_bundle`/`_resonate`, `rf_phasor_composer.py:101–136`); (ii) every cleanup is an
**argmax over 2,048 concepts** each O(D) ≈ 11 M float-ops, ~45 M per multi-unbind query (§4.1); (iii) **the open
FHRR-SNR question at V=2,048 / D≈5.5k** (§5).

**The honest advantage.** Cross-bridge facts AND clauses bind generatively (full VSA across all concepts) — the
maximally-expressive composer. **The honest risk.** D≈5.5k is a large composer whose per-op RF bridges are 11–22k
neurons, and whose binding/cleanup SNR at 2,048-concept cleanup is **not validated** (the K=5 two-attribute
boundary and the 320-concept correctness GO bound what IS known — §4.3).

### 2.3 Hybrid — per-bridge composers + ONE SMALL cross-bridge composer

**Shape.** Route A's per-bridge composers for within-bridge work, PLUS one **small** union-ish composer used ONLY
for the relatively-few cross-bridge facts that genuinely need generative binding (e.g. cross-bridge clauses /
attribute transfer). Its vocabulary is NOT all 2,048 concepts — it is only the concepts that actually participate
in cross-bridge generative structures (a small, curated subset, or the per-fact participants loaded on demand), so
its D stays modest (the cross-bridge generative load is the few facts' worth of terms, not the whole vocabulary).

**Component mapping.** Route A's `{shard: RFPhasorComposer}` + V-tag layer + `word_to_shard`, AND one extra
`RFPhasorComposer(vocab = the cross-bridge-participant concepts, D = sized to that small set)` that holds the
cross-bridge SVO facts that need generative binding (the `gate_X_conv` path, but over a small composer).

**When the hybrid is warranted.** ONLY if §3's crux finds the capability needs *some* cross-bridge generative
binding but it is RARE (a small fraction of facts), so paying route B's D≈5.5k for the whole vocabulary is wasteful
when a small dedicated composer covers the few cross-bridge generative facts. If the crux finds cross-bridge
generative binding is NOT needed → route A (no extra composer). If it finds cross-bridge generative binding is
PERVASIVE → route B (the small composer would itself grow to the union and you have route B with extra plumbing).

---

## 3. THE CRUX — does the validated conversational matrix NEED cross-bridge generative VSA binding?

**The decision pivots entirely on this question.** If the validated matrix only needs **within-bridge generative
VSA** (clauses/attributes inside a bridge) + **cross-bridge IDENTITY composition** (fact recall: "dog eats meat"
→ recall meat), then route A wins on scalability. If it needs **cross-bridge GENERATIVE binding** (a cross-bridge
clause, cross-bridge attribute transfer), route B (or the hybrid) is required.

### 3.1 What the validated gates actually tested — read directly from the runner

I read the ensemble runner's gates and the matrix construction they call. Mapping each cell to within- vs
cross-bridge, and identity vs generative:

| Gate / cell | What it binds | Within or cross bridge? | Identity or generative? | Source |
|---|---|---|---|---|
| **Gate A** `what_does` / `who_does` | a stored SVO fact, roles drawn from the **interleaved union** | the ROLES span shards, but each is ONE concept; the FACT is one bound composite in the union composer | generative (bind+bundle+unbind) | `gate_A_matrix:436–441`, interleave `_make_union_codebook_for_matrix:541–578` |
| Gate A `abstention` | a never-stored (agent, action) | n/a (returns None) | the moat | `gate_A_matrix:441, 464–473` |
| Gate A `negation` | a stored SVO + AFFIRM/NEGATE polarity tag | roles span shards | generative | `gate_A_matrix:444–450` |
| Gate A `one_attribute` | `(adj, noun)` patient — adj and noun from the union | adj and noun can be different shards | generative | `gate_A_matrix:452–456` |
| **Gate A `clause`** | `Clause(words[5], words[1], words[2])` — a nested SVO patient | under the interleave, the THREE clause words are THREE different shards | **generative, CROSS-bridge** | `gate_A_matrix:426, 458–462` |
| **Gate B** generalization | held-out graded-neighbour, read from cortex codes | **strictly WITHIN one bridge** (the fallback restricts candidates to the agent-word's shard) | generative (binding) + a direct graded read | `EnsembleCortexAgent._graded_fallback_patient:223–252` (`word_to_shard` restriction l.236) |
| **Gate X** cross-bridge | `dog eats meat`, dog∈animals, meat∈foods | **strictly CROSS bridge** | **IDENTITY** (V-tag) + an X-conv identity-recall realization | `gate_X_vtag:391–416`, `gate_X_conv:336–388` (fallback DISABLED, l.344–345) |

### 3.2 The crux finding

**The ensemble's PASSING matrix DID exercise a cross-bridge GENERATIVE clause** — `gate_A_matrix`'s `clause` cell,
under the interleaved union, binds three words from three shards into ONE composite in the union composer and
unbinds the nested SVO back (`gate_A_matrix:458–462`; the interleave `_make_union_codebook_for_matrix:541–578`).
**However — and this is the load-bearing nuance — that the ensemble *implemented* the clause cross-bridge does NOT
prove the capability *requires* it cross-bridge.** The matrix design explicitly notes (ensemble runner docstring
lines 88–92) that Gate A "tests binding + the moat, not generalization," and that interleaving the union "makes the
stored facts genuinely cross-bridge" as a STRESS choice. The same `clause` capability is independently validated
**within a single bridge** in the single-shard capability de-risk (`cortex_conversation_capability_GO.md`: matrix
6/6 on a 1-shard × 64 composer, where every clause word is in the one shard).

**So the honest reading of the crux is a FORK that the de-risk (§4.4) must resolve empirically, because the runner
proves route B *can* do the cross-bridge clause but not that route A *cannot* satisfy the capability:**

- **The capability the project actually cares about** (from the integration design §1.3 and the build-plan piece
  iii, lines 51–56) is: (1) the conversational matrix WITHIN a bridge (who/what/abstention/negation/attribute/
  clause — the single-shard GO), (2) WITHIN-bridge graded generalization (Gate B — strictly within-bridge by
  construction), and (3) CROSS-bridge FACT composition (Gate X — explicitly IDENTITY, build plan §"Genuine open
  questions" item 1, line 94: *"cross-bridge relationships go through the existing composition/binding layer, not a
  shared embedding"*). **Nowhere does a validated finding state that a cross-bridge CLAUSE or cross-bridge
  attribute-transfer is a required capability.** The build plan's own framing is that cross-bridge is identity
  composition; within-bridge is where generative binding + generalization live.

- **Therefore the leading hypothesis is: the matrix needs within-bridge generative VSA + cross-bridge identity,
  and route A is sufficient.** The ensemble's cross-bridge clause is an artifact of the interleave STRESS test, not
  a stated capability requirement. **But this must be PROVEN, not assumed** — the de-risk (§4) runs route A's
  matrix with the clause forced WITHIN a bridge and route B's matrix with the clause CROSS-bridge, at 512 concepts,
  and compares. If route A's within-bridge matrix passes (it should, per the single-shard GO) AND the only thing
  route A loses is the cross-bridge clause (which no requirement demands), route A wins. If a downstream
  conversational need for cross-bridge clauses/attribute-transfer surfaces, the hybrid (§2.3) adds it cheaply.

**Crux verdict (to be confirmed by §4.4): the validated matrix does NOT require cross-bridge generative binding —
within-bridge generative VSA + cross-bridge identity is sufficient — so route A is the recommended route, with the
hybrid as the cheap escape hatch if a cross-bridge generative need is later demonstrated.**

---

## 4. Quantitative estimates (computed CPU-only, no bridge built)

All numbers below are from a CPU-only analytic script (counting array sizes + op counts from the actual class
code) — no `SimulationBridge`, no GPU.

### 4.1 Route B — one union composer at 2,048 concepts

- **Dimension.** D ≈ 2,048 × 2.7 ≈ **5,460** (the ensemble's D/concept ≈ 512/192 ≈ 2.67 extrapolated; §1.1).
  Band **5,000–6,000**.
- **Codebook memory.** 2,048 concepts × D≈5,460 × 8 bytes (float64 `concepts` dict, `rf_phasor_composer.py:80`) ≈
  **90 MB** (+ roles/polarity, negligible). Per-concept code = 43.7 kB. **The codebook is NOT the problem.**
- **Per-op RF bridge size (the real cost).** `_bind` builds a **2D = 10,922-neuron** RF bridge per bind
  (`rf_phasor_composer.py:117–126`); `_bundle` of a 3-role fact builds a **(L+1)D = 21,844-neuron** bridge
  (lines 128–136). The spiking-cleanup matched-filter bridge is **D+V = 7,509 neurons** + an Izhikevich WTA bank
  of V=2,048 (`_spiking_cleanup`, lines 197–245). These are per-op (the bridge cache reuses by neuron count,
  l.106–108), but every distinct op size allocates a multi-thousand-neuron RF bridge.
- **Cleanup cost.** `_cleanup` (line 251) is an argmax cos over the WHOLE 2,048-entry codebook, each O(D):
  ≈ **11 M float-ops per cleanup**; a relational query does ~2–4 unbind+cleanups → **~45 M float-ops/query**
  (vs ~1 M/cleanup at the ensemble's V=192). The cleanup cost scales with the union vocabulary — the thing route A
  avoids.
- **FHRR SNR at this scale — the open risk (§5).** Whether bind/bundle/unbind+cleanup stays reliable at V=2,048 /
  D≈5,460 is NOT validated; see §4.3 for what IS known.

### 4.2 Route A — 32 per-bridge composers + V-tag

- **Per-composer dimension.** Each composer's vocab is 64 concepts + ~8 aux. The cap de-risk passed Gate A at
  **D=128 over 64 concepts** (`cortex_conversation_capability_GO.md`, 1 shard × 64). 64 × 2.67 ≈ 171, so D≈192–256
  is ample headroom; D=512 is generous.
- **Codebook memory (all 32).** D=128 → **2.4 MB** total; D=256 → 4.7 MB; D=512 → 9.4 MB. Trivial.
- **Per-op RF bridge size.** bind = 2D = **256–1,024 neurons**; bundle (3 roles) = 4D = 512–2,048 neurons —
  **~10–20× smaller per-op than route B.**
- **Cleanup cost.** argmax over **64** concepts each O(D) ≈ 8–33 k float-ops/cleanup — **~300× cheaper per cleanup
  than route B** (and independent of total vocabulary).
- **Cross-bridge V-tag cost.** Per cross-fact = 2 engram tags (top_k≈150 neuron indices each) on the existing
  2,400-neuron spiking `pool` regions (`GradedBridge.encode_tag`, `multibridge_graded_derisk.py:480–507`). **NO
  D-scaling.** The V-tag layer is already GO at 3 bridges (M3 signal/floor 17–24×, M7 collapses;
  `multibridge_graded_derisk-GO.md`).
- **Co-residence.** Route A's spiking bridges are the SAME per-bridge graded pools route B also needs (both routes
  shard the cortex; the difference is the *composer*, not the cortex). The per-bridge graded pool is ~3.46M
  synapses (`multibridge_graded_derisk-GO.md` line 21); 32 co-resident ≈ 110M synapses — within the 24 GB GPU's
  reach in the multi-bridge regime (the whole reason for sharding, build plan §"Scaling path" line 101).

### 4.3 What FHRR capacity/SNR is KNOWN (the load-bearing citations)

- **The K=5 two-attribute boundary.** The ±1 Hadamard scheme provably could not invertibly bind two concept codes
  (adj⊗noun) — a documented K=5-load boundary (CLAUDE.md "ONE-BRIDGE UNIFICATION" + the FHRR pivot notes). FHRR
  was adopted PRECISELY to lift this (`rf_phasor_composer.py:266–268` notes the 2-attribute "K=5 boundary — does
  FHRR lift it?"). **This bounds within-fact binding LOAD, not vocabulary size** — relevant to both routes' clause/
  two-attribute cells, NOT specifically to route B's vocabulary scaling.
- **The 320-concept correctness GO.** The FHRR composer reproduced the full capability matrix at **320 concepts**,
  correctness GO 8/8/8 (CLAUDE.md "OPPONENCY ESCAPED"). This is the **highest validated single-composer vocabulary**
  for the FHRR composer. **Route B at 2,048 is 6.4× beyond this validated ceiling** — the cleanup argmax over
  2,048 entries at D≈5.5k is in untested territory; whether D≈5.5k is *enough* (or whether SNR degrades and D must
  go even higher, compounding the per-op bridge cost) is **the route-B scaling unknown the de-risk must measure.**
- **The ensemble's D=128→512 clause failure→pass at V=192.** Direct evidence that the cleanup over the union
  vocabulary is the binding constraint, and that D must track V (§1.1). At V=2,048 the required D is an
  extrapolation, not a measurement — §4.4 measures it at V=512 (8 bridges) where it is cheap.

### 4.4 The cost asymmetry, stated plainly

| Quantity | Route A (32 × D≈256) | Route B (1 × D≈5,460) |
|---|---|---|
| codebook memory | ~4.7 MB | ~90 MB |
| per-bind RF bridge | ~512 neurons | ~10,922 neurons |
| per-bundle RF bridge | ~1,024 neurons | ~21,844 neurons |
| cleanup ops/query | ~0.1 M (over 64) | ~45 M (over 2,048) |
| cleanup scales with total vocab? | **NO** (per-shard) | **YES** (per-union) |
| cross-bridge composition | identity (V-tag, validated) | generative (in-composer, untested at 2,048) |
| FHRR SNR validated at this scale? | yes (64 ≪ 320 GO) | **NO** (2,048 = 6.4× the 320 GO) |
| code change vs ensemble runner | new dispatch glue | scale constants only |

**Route A is dramatically cheaper per-op AND its per-op cost is vocabulary-independent; route B is the
least-code-change path but carries the unvalidated FHRR-at-2,048 SNR risk and 10–20× larger per-op bridges.**

---

## 5. The cheap-first de-risk that DECIDES it (512 concepts = 8 bridges, NOT 32)

**Goal.** The smallest run that measures, for BOTH route A and route B, the full conversational matrix (incl. the
cross-bridge clause where the route supports it) + within-bridge generalization + cross-bridge composition + the
moat + the COST (dimension, memory, wall-clock), at **8 bridges × 64 = 512 concepts** — the scale where route B's
D is still tractable to *measure* (D≈512×2.7≈1,400, not 5,460) yet large enough to exercise the cross-bridge
fan-out beyond the 3-bridge mechanism de-risk, and where route A's per-bridge cost is identical to its 32-bridge
cost (per-shard, vocabulary-independent). 8 bridges is the build plan's stated "more fan-out, not a new code path"
(integration design §2, lines 217–219; CLAUDE.md "holds to 8-bridge fan-out").

**Why 512 and not 32.** At 512 concepts the A-vs-B comparison is **decidable**: route A passes/fails the matrix at
its true per-shard cost; route B's required D is *measured* (sweep D until the clause cell passes) and its per-op
cost is observed — and the D/concept ratio at V=512 (8 bridges) extrapolates to V=2,048 far more safely than the
V=192 (3-bridge) ratio does. 32 bridges is the production build; this de-risk is the gate for which architecture
that build uses.

### 5.1 The runner — extend `cortex_conversation_ensemble_derisk.py` with a per-bridge-composer mode

`cortex_conversation_ensemble_derisk.py` already runs route B verbatim (one union `RFPhasorComposer`,
`EnsembleCortexAgent`). It needs **one new mode: `--composer per-bridge`** (default `union` = the current
behavior), which:

1. Builds a `{shard: RFPhasorComposer}` dict (each over its 64 concepts + aux) instead of one union composer
   (a `PerBridgeCortexAgent` subclass of `EnsembleCortexAgent` overriding `__init__`'s composer construction,
   lines 200–204, and `store_fact` / `what_does` / `who_does` to dispatch by `word_to_shard`).
2. Routes within-bridge facts to the agent-word's shard's composer; routes cross-bridge facts to the V-tag layer
   (the existing `gate_X_vtag` path, `cortex_conversation_ensemble_derisk.py:391–416`) — and, for route A's matrix,
   builds the `clause` cell WITHIN a single bridge (override `_make_union_codebook_for_matrix` so the clause's
   three words come from ONE shard, NOT interleaved) so route A's matrix is the within-bridge generative clause.
3. Logs **cost**: each composer's D, the codebook bytes, the per-op RF bridge sizes, and wall-clock per gate.

For **route B** the runner runs AS-IS at `--n-bridges 8` with a **`--D-sweep`** over the union composer
(e.g. 512, 1024, 1536, 2048) to find the smallest D at which Gate A's cross-bridge `clause` cell passes — i.e.
*measure* route B's D/concept at V=512, plus its per-op bridge sizes and wall-clock.

### 5.2 The exact commands the controller runs

```bash
# ---- Route B (current union composer), 8 bridges = 512 concepts, sweep D to find the clause-cell threshold ----
SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_ensemble_derisk \
    --mode full --seeds 42,43,44 --cortex learned \
    --composer union \
    --n-bridges 8 --concepts-per-bridge 64 \
    --D-sweep 512,1024,1536,2048 \
    --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
    --out research/findings/raw/_phase1_composer_routeB_512.json

# ---- Route A (NEW per-bridge-composer mode), 8 bridges = 512 concepts ----
SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_ensemble_derisk \
    --mode full --seeds 42,43,44 --cortex learned \
    --composer per-bridge --per-bridge-D 256 \
    --n-bridges 8 --concepts-per-bridge 64 \
    --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
    --out research/findings/raw/_phase1_composer_routeA_512.json
```

(Tiny CPU plumbing smoke first, both modes: append `--smoke --seeds 42` with `SIM_BACKEND=numpy` — the ensemble
runner's existing `--smoke` path, lines 740–753.)

### 5.3 What each run measures (the same gate suite the ensemble already implements)

For BOTH routes (reusing the ensemble runner's Gates A/B/X + anti-cheats C1/C4/Cx/C3, `run_seed`/`aggregate`,
lines 435–671), multi-seed 42/43/44:

- **Gate A** — the conversational matrix (who/what, abstention, negation, one-attribute, clause). Route B: clause
  CROSS-bridge (interleaved). Route A: clause WITHIN-bridge. Both must hit ≥ 5/6 cells + zero abstention breach.
- **Gate B** — within-bridge graded generalization (B1 ≥ 0.7 ≈ 4× chance; B2 moat zero false-accepts). **Identical
  for both routes** (generalization is within-bridge, reads cortex codes directly — neither composer touches it).
- **Gate X** — cross-bridge composition. Route B: generative (X-conv, in the union composer). Route A: identity
  (V-tag, X-vtag). Both must retrieve the target top-2 above the noise floor with the Cx anti-cheat collapsing.
- **The anti-cheats** — C1 (permuted similarity → B1 collapses), C4 (random shard → B1 collapses), Cx (permuted
  cross-bridge → Gate X collapses), C3 (moat alongside host, zero breaches + lesion collapses). All mandatory.
- **COST** (the new logging) — per route: composer D(s), codebook MB, per-op RF bridge neuron counts, wall-clock
  per gate and total, peak GPU memory.

### 5.4 The GO / decision criteria

- **Route A wins (the expected outcome)** if: route A passes Gate A (within-bridge clause) ≥ 5/6 + moat holds,
  Gate B GO, Gate X (V-tag identity) GO, all anti-cheats collapse, multi-seed — AND its cost is per-shard-bounded
  (D≈256, cleanup over 64, per-op bridges ≈ hundreds of neurons), i.e. **flat to 32 bridges.** The only capability
  route A "loses" is the cross-bridge generative clause — which §3 argues no requirement demands. **⇒ carry route A
  to 32 bridges.**
- **Route B is required** if: route A's matrix FAILS specifically because a cross-bridge generative structure
  (clause / attribute transfer) is genuinely needed AND the V-tag identity layer cannot stand in — i.e. a gate
  that route A cannot pass but route B can — AND route B's measured D at V=512 extrapolates to a D at V=2,048 whose
  per-op bridge + cleanup cost is acceptable on the 24 GB GPU (no OOM, wall-clock tractable). **⇒ carry route B to
  32 bridges** (accepting the FHRR-at-2,048 SNR risk, characterized at V=512).
- **Hybrid is chosen** if: route A passes the matrix BUT a small, identifiable set of cross-bridge facts needs
  generative binding (route A fails ONLY those, route B passes them, and they are a small fraction) → route A +
  one small cross-bridge composer over just those participants.
- **NEGATIVE (blocks the build)** if: a moat breach on either route (fatal), OR neither route passes the matrix at
  512 concepts, OR route B's required D at V=512 already extrapolates to an infeasible 2,048-concept composer
  (per-op bridge OOM) AND route A cannot deliver a needed cross-bridge generative capability. A NEGATIVE here is the
  scientific deliverable — it maps the composer scaling boundary before the multi-week spend.

**Estimated cost of the de-risk:** hours, not days (the ensemble runner's graded learn + V-tag are the small
spiking ops it already runs at 3 bridges; 8 bridges is 2.7× the pools; the matrix/generalization/moat reads are
numpy). Route B's `--D-sweep` adds a few union-composer rebuilds at increasing D (the dominant new cost, still
tractable at V=512).

---

## 6. Anti-cheats + honest risks

**Anti-cheats (carried verbatim from the ensemble runner; the GO is void without them).**
- C1 permuted-similarity → within-bridge generalization (B1) collapses to chance (`anticheat_C1_permuted`).
- C4 random-shard → B1 collapses (`anticheat_C4_random_shard`).
- Cx permuted cross-bridge mapping → Gate X collapses (`cross_bridge_eval(..., permuted=True)`, the FIXED M7).
- C3 the moat validated ALONGSIDE the host → zero host-abstain/gate-accept breaches + lesion collapses
  (`anticheat_C3_moat`).
- **NEW for this de-risk:** an **abstention battery on BOTH composer routes** (the matrix's `abstention_battery`,
  `gate_A_matrix:464–473`) — route A's per-bridge composers each have their own moat, so the de-risk must verify a
  cross-bridge query whose target bridge has NO matching fact ALSO abstains (the V-tag layer must not falsely
  recall; `cross_bridge_eval`'s Cx already tests the wrong-target case, but the *abstention* — no tag at all —
  must be explicitly checked per-route).

**Honest risks.**

- **RISK 1 — Route B: does D≈5,460 actually hold the matrix at 2,048, or does FHRR SNR degrade? (THE real
  scaling risk.)** The 320-concept correctness GO is the highest validated FHRR single-composer vocabulary;
  2,048 is 6.4× beyond it (§4.3). If SNR degrades faster than D≈2.7/concept compensates, route B needs an even
  larger D — compounding the 11–22k-neuron per-op bridges and the 45 M-op cleanup, possibly to infeasibility on
  24 GB. **The de-risk's `--D-sweep` at V=512 measures the D/concept the matrix actually needs at 8 bridges**
  (a far safer extrapolation base than the 3-bridge V=192 ratio), turning this from an extrapolation into a
  measurement before the build commits.
- **RISK 2 — Route A: does cross-bridge IDENTITY recall suffice for the matrix, or do cross-bridge clauses /
  attribute-transfer silently fail? (THE crux risk.)** §3 argues no validated requirement demands cross-bridge
  generative binding (the build plan frames cross-bridge as identity composition). But a downstream conversational
  need — e.g. "the dog that eats meat runs" with dog and meat in different shards — would need generative
  cross-bridge binding route A lacks. **Mitigation:** the de-risk builds route A's matrix with the clause WITHIN a
  bridge and ALSO probes a cross-bridge clause to see whether the V-tag identity layer + within-bridge composition
  can stand in (or whether it genuinely fails) — and the hybrid (§2.3) is the cheap escape hatch if a real
  cross-bridge generative need surfaces. The honest framing: route A bets that conversation's cross-bridge needs
  are FACTUAL ("dog eats meat") not deeply COMPOSITIONAL ("the dog that eats meat that the cat saw…"), which the
  build plan's own framing supports — but the bet must be checked, not assumed.
- **RISK 3 — Route A: per-bridge moats vs one union moat.** 32 separate `RelationalFamiliarityGate`s (one per
  bridge) is 32 thresholds to keep zero-breach, vs route B's single union gate. More moat surface area = more
  places a breach could hide. The de-risk's per-route abstention battery (above) checks this at 8 bridges; the
  build must re-assert it at 32 (the moat is non-negotiable, build plan honest-risks line 74).
- **RISK 4 — the D/concept ratio is itself uncertain.** 2.67 comes from a SINGLE observation (clause fail@128 /
  pass@512 over V=192). The real ratio could be sub-linear (cleanup capacity often scales better than linear in V
  for fixed crosstalk) or super-linear. **This is exactly why the de-risk sweeps D at V=512 rather than trusting
  the extrapolation** — and why route A (whose cost does NOT depend on the ratio at all) is the safer default.
- **RISK 5 — semantic-cluster sharding (shared by both routes).** Within-bridge generalization requires similar
  concepts co-located (C4 confirms random sharding collapses it). The production sharding (co-occurrence-graph
  clustering of the 2,048 concepts into 32 bridges) is a build-time design choice both routes inherit; the de-risk
  uses the curated stand-in shards (`SHARD_NAMES`, animals/foods/vehicles/…, `multibridge_graded_derisk.py:147`).
  Flag for the build, not a blocker for the de-risk. (Route A is MORE sensitive to sharding quality than route B,
  because route A's cross-bridge facts can't be re-bound generatively — a mis-shard that splits a tightly-related
  pair across bridges costs route A a generative within-bridge clause it would otherwise have.)

---

## 7. Recommended route + the controller sequence

**Recommended: route A (per-bridge composers + cross-bridge V-tag identity layer), with the hybrid as the
documented escape hatch.** Three reasons: (1) **scalability** — route A's per-op cost (D≈256, cleanup over 64,
per-op RF bridges of hundreds of neurons) is per-shard and **vocabulary-independent**, so it is flat from 8 to 32
bridges, whereas route B's D≈5.5k, 11–22k-neuron per-op bridges, and 45 M-op cleanup all grow with the union and
sit 6.4× beyond the highest validated FHRR vocabulary (the 320-concept GO). (2) **The crux** — the validated
conversational capability needs within-bridge generative VSA (the single-shard matrix GO) + within-bridge graded
generalization (Gate B, within-bridge by construction) + cross-bridge IDENTITY composition (Gate X, explicitly
identity per the build plan), and **no validated finding requires cross-bridge GENERATIVE binding** (the ensemble's
cross-bridge clause is an interleave STRESS artifact, not a stated requirement). (3) **Risk** — route A carries no
unvalidated-SNR risk (every per-bridge composer is at 64 ≪ the 320 GO), and the hybrid adds cross-bridge generative
binding cheaply IF a real need surfaces. The cost of being wrong is small (add the hybrid's one small composer); the
cost of route B being wrong is large (re-architect after discovering D≈5.5k OOMs or under-performs at 2,048).

**This recommendation is CONTINGENT on the §5 de-risk confirming it** — specifically that route A passes the full
matrix (within-bridge clause) + Gate X (V-tag identity) at 512 concepts at flat per-shard cost, with no moat breach
and no demonstrated cross-bridge generative need that route A cannot meet.

**Controller sequence:**
1. **This design** (done) — present before building; owner reviews (trust-but-verify the crux reading: that no
   validated requirement demands cross-bridge generative binding).
2. **The 512-concept A-vs-B de-risk (§5), multi-seed 42/43/44.** Extend `cortex_conversation_ensemble_derisk.py`
   with the `--composer per-bridge` mode + the route-B `--D-sweep`; run both commands at `--n-bridges 8`. Decide
   per §5.4. Cost: hours.
   - On **route A wins** → carry route A to 32 bridges.
   - On **route B required** → carry route B (with the V=512-measured D/concept + the characterized SNR risk).
   - On **hybrid** → route A + one small cross-bridge composer.
   - On **NEGATIVE** → STOP; the NEGATIVE (moat breach, or neither route passing, or route B infeasible at 2,048
     with a real cross-bridge generative need) is the scientific deliverable and reshapes Phase 1 before the spend.
3. **ONLY THEN the winning route to 32 bridges = 2,048 concepts** — the production semantic-cluster sharding + the
   full conversational matrix + generalization-in-conversation + the moat at 32-bridge fan-out (build plan piece
   iii, the ~2–4 week sustained push). The composer architecture decided here is the foundation that build sits on.

**Why this ordering is the standing opening move:** every prior decisive pivot (the whitening reframe, the
missing-accumulator fix, the ventral-vs-dorsal nav root-cause, the dual-CLS resolution) came from a cheap
read-only/de-risk proof BEFORE committing build/GPU resources. The composer-architecture decision is the cheapest
thing that can be settled before the 32-bridge spend, and it is load-bearing for the whole build — so it gets its
own design + its own A-vs-B de-risk gate.

---

## Summary (the four things the controller needs)

**Recommended architecture (route A, with hybrid escape hatch) + why.** Use **per-bridge composers + the
cross-bridge V-tag identity layer (route A)**: each of the 32 bridges runs its own small `RFPhasorComposer`
(D≈256 over 64 concepts) for within-bridge generative binding (clauses/attributes/who-what), within-bridge graded
generalization reads the cortex codes directly, and cross-bridge facts go through the validated V-tag identity
recall — so the composer cost is **per-shard and vocabulary-independent (flat from 8 to 32 bridges)**, carrying
none of route B's unvalidated FHRR-SNR risk at 2,048 concepts (6.4× the highest validated 320-concept composer
GO). Route B (one union composer at D≈5,000–6,000, ~90 MB codebook but 11–22k-neuron per-op RF bridges and a 45 M-op
cleanup over 2,048 concepts) is the least-code-change path but bets the build on an untested binding-SNR scale; the
hybrid (route A + one small cross-bridge composer) is the cheap fix if a cross-bridge GENERATIVE need ever surfaces.

**The crux finding (does the matrix need cross-bridge generative binding?).** **No — within-bridge generative VSA
+ cross-bridge IDENTITY composition is sufficient for the validated conversational matrix.** Reading the ensemble
runner directly: Gate B generalization is strictly within-bridge (the fallback restricts candidates to the
agent-word's shard, `_graded_fallback_patient:236`), Gate X cross-bridge is explicitly IDENTITY (V-tag,
`gate_X_vtag`/`gate_X_conv`, fallback disabled), and the only cross-bridge GENERATIVE thing in the passing matrix
is Gate A's `clause` cell under the interleaved union — which the runner's own docstring frames as a binding STRESS
choice, not a stated capability (the build plan explicitly says cross-bridge relationships go through composition,
not a shared embedding). The same clause capability is independently GO WITHIN one bridge. So the cross-bridge
generative clause is not a requirement; route A is sufficient — to be confirmed by the de-risk forcing route A's
clause within-bridge and checking nothing required is lost.

**The exact 512-concept A-vs-B de-risk.** Extend `cortex_conversation_ensemble_derisk.py` with a `--composer
per-bridge` mode (default `union`) + a route-B `--D-sweep`, then run both, multi-seed 42/43/44, at **8 bridges ×
64 = 512 concepts**:
```bash
SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_ensemble_derisk --mode full \
  --seeds 42,43,44 --cortex learned --composer union --n-bridges 8 --concepts-per-bridge 64 \
  --D-sweep 512,1024,1536,2048 --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
  --out research/findings/raw/_phase1_composer_routeB_512.json
SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_ensemble_derisk --mode full \
  --seeds 42,43,44 --cortex learned --composer per-bridge --per-bridge-D 256 --n-bridges 8 \
  --concepts-per-bridge 64 --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
  --out research/findings/raw/_phase1_composer_routeA_512.json
```
GO = the route that passes the full matrix + Gate B + Gate X + all anti-cheats at acceptable, vocabulary-scalable
cost carries to 32 bridges (expected: route A, flat per-shard cost; route B only if a cross-bridge generative need
route A cannot meet appears AND its V=512-measured D extrapolates feasibly to 2,048).

**The single deepest risk.** Route B's FHRR binding/cleanup SNR is validated only to 320 concepts; at 2,048 (6.4×)
the required D≈5.5k is an extrapolation from a single V=192 data point, and if SNR degrades faster than D≈2.7/concept
compensates, route B's 11–22k-neuron per-op bridges + 45 M-op cleanup grow to possible infeasibility on the 24 GB
GPU — which is exactly why the de-risk SWEEPS D at V=512 (a safe extrapolation base) and why route A (whose
per-shard cost does not depend on the union vocabulary at all) is the recommended default.
