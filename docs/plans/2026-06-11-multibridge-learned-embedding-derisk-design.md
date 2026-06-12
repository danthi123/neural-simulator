# Multi-bridge learned-embedding: cheap-first de-risk DESIGN (the large-vocabulary route)

> **Status:** present-before-build (DESIGN ONLY). This is the standing "deep research + design BEFORE
> building" opening move for a new direction (the large-vocabulary frontier). It is read-only:
> nothing here builds a bridge or runs a probe — it specifies the smallest GPU run the controller
> would run *next* to falsify the load-bearing risk, with explicit GO / BOUNDARY / NEGATIVE criteria.
> No `sim/` edits are proposed; every step reuses existing runners.

**The question this doc answers.** The dual / complementary-learning-systems ("CLS") learned
graded-similarity cortex is validated **single-pool** up to V = 320 concepts, but a single pool OOMs
(runs out of GPU memory) by ~V = 320–450 on a 24 GB RTX 3090 (the synapse install exhausts pinned
memory). The only route to large vocabulary (e.g. 2,048 concepts) is **multi-bridge**: many small
bridges, each running the per-bridge recipe. The project already has a validated multi-bridge
architecture at 320 concepts (5 bridges × 64). So the design problem is: **how do the per-bridge
learned graded embedding and the existing cross-bridge layer fit together to reach 2,048 concepts
(32 bridges × 64), and what is the smallest run that falsifies it before any large build?**

Term definitions used once here:
- **Bridge** — one `SimulationBridge` (a brain). A "pool" is the bridge's concept-holding neuron
  population.
- **Learned graded embedding** — the validated dual/CLS piece: a spiking-Hebbian co-occurrence
  recurrent (`LearnedAssocGraph`) with a biological homeostatic normalization (default Oja
  incoming-L2 renorm) + a brain-based divisive-normalization read-out. Its output is concept codes
  where **related concepts cluster** (cat ≈ dog), so the agent generalizes across similar concepts.
- **Cross-bridge composition** — the existing mechanism for relating concepts that live on different
  bridges. The project has **two** validated variants (detailed in §1).
- **The no-confab moat** — the agent abstains ("I don't know") when no stored fact matches, instead
  of confabulating. Validated as a host check and as a learned neural familiarity gate.
- **Sharding** — deciding which concepts live on which bridge.

---

## 0. Sources this design is grounded in (all read)

| Claim | File |
|---|---|
| Single-pool recipe GO to V=320; quadratic synapse wall; multi-bridge is the only large-V route; the 3 open multi-bridge questions | `docs/plans/2026-06-11-semantically-structured-cortex-BUILD-PLAN.md` §"Scaling path" |
| Single-pool OOMs at ~V=320–450 (354M synapses @ V=640 exhausts pinned memory); recipe validated V=160→V=320, improving | `research/findings/2026-06-11-V640-single-pool-memory-wall.md` |
| The homeostatic learner recipe (Oja default, set-point, divnorm read-out), cycle-independent, multi-seed 3/3 | `research/findings/2026-06-11-learned-graded-embedding-homeostasis-GO.md`; `research/runners/learned_graded_embedding_homeostasis_probe.py` |
| V=160 production recipe is a clean GO near host ceiling (Oja t=40 → Pearson +0.977, gen 1.000) | `research/findings/2026-06-11-build-piece-ii-V160-scale-check.md` |
| Existing multi-bridge **engram-tag** cross-bridge mechanism (`apple_big` spanning bridges; tag-name search + firing aggregation) | `research/runners/g20_multibridge.py`; `research/runners/shared_pool_chat.py`; `research/findings/2026-05-15-G20-sparse-ensemble-160concept-end-to-end-SHIPPED.md` |
| 320-concept engram-tag ensemble SHIPPED (98.4%/bridge, cross-bridge retrieval validated seed 42) | `research/findings/2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md`; `research/runners/g20_sparse_5bridge_chain_320.ps1` |
| Existing multi-bridge **VSA/coincidence flat-distinct** cross-bridge composition (5 distinct-seed bridges → 320 distinct flat codes; SVO 1.000, any-bank 0.992 6-seed, abstention) | `research/findings/2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md` |
| The no-confab familiarity gate at V=320 (zero moat-breaches, multi-seed) | `research/findings/2026-06-11-familiarity-gate-v320-GO.md` |
| The architecture proof (round-trip preserves graded similarity, GO) that justifies the whole build | `research/findings/2026-06-11-dual-CLS-architecture-proof-GO.md` |
| The learned-assoc-graph learner itself | `research/runners/learned_assoc_graph.py` |

---

## 1. Diagnosis — how the pieces fit, and where they DON'T

### 1.1 The two layers are orthogonal, and that is the key insight

The dual/CLS work and the multi-bridge work operate at **two different layers**, and they have never
been combined:

- **The per-bridge graded embedding is a WITHIN-pool property.** The recipe learns a recurrent
  weight matrix `W` over the **patterns of concepts that co-fire in the SAME pool**, then reads graded
  codes from it. "cat ≈ dog" is recoverable only because cat's pattern and dog's pattern are both in
  the same pool and co-fire with shared context there. There is no mechanism by which a concept in
  bridge A acquires graded similarity to a concept in bridge B — they never share a recurrent.
  (Verified in `learned_assoc_graph.py`: `graph()` reads `cp_connections[pool_base][:, pool_base]` —
  a single pool's recurrent; and `homeostasis_probe.py`: the homeostatic rule is applied to the
  `pool↔pool` mask of one bridge.)

- **The existing cross-bridge layer relates concept CODES/PATTERNS, not graded embeddings.** It binds
  *identities*, not *similarities*. There are two validated variants, and **both are
  graded-embedding-agnostic** — they would work identically whether the per-bridge code is graded or
  orthogonal:

  - **(V-tag) Engram-tag cross-bridge** (`g20_multibridge.py --sparse`). To store "apple is big" when
    apple ∈ bridge A and big ∈ bridge C, the system records a tag **named `apple_big` in BOTH
    bridges** — over apple's sparse pattern in A and big's sparse pattern in C (`encode_partial` /
    `encode_partial_pair_engram_sparse` in `shared_pool_chat.py`). Recall searches **tag names**
    across all bridges for the cue word, stimulates each matching tag, and aggregates per-pattern
    firing (`query_concept` → `recall_rates` → `stim_recall_sparse_rates`). This is the SHIPPED
    320-concept ensemble. It is fundamentally a **distributed key-value store keyed by tag name**: the
    "cross-bridge" link is the shared string `apple_big`, plus each bridge independently recalling its
    own concept from its own pattern.

  - **(V-vsa) VSA / coincidence flat-distinct composition** (`2026-06-02-flat-distinct...`). Each
    bridge is trained with a **distinct seed** so its sparse patterns — and therefore the flat codes
    captured from it — differ; the 320 codes are then composed at a **single** vector-symbolic binding
    level (bind/unbind by spiking coincidence + cosine cleanup over all 320). Structured
    subject-verb-object facts hit 1.000; any-role 0.992 (6-seed); absent-cue abstains. This is the
    "brain-analogue composition" path (the owner's preferred framing over static retrieval/ranking).

  **Neither variant carries graded similarity across bridges.** V-tag aggregates firing rates of
  *identity* patterns; V-vsa deliberately makes the codes **maximally distinct** (between-cosine mean
  0.045) — the opposite of graded.

### 1.2 The consequence — the corpus must be sharded by semantic cluster

Because graded generalization is within-pool only, **for the new capability to be useful, similar
concepts must live in the SAME bridge.** If cat is in bridge A and dog is in bridge B, the agent
cannot infer "cat is like a dog" — that inference requires them to share a recurrent. So:

> The 2,048-concept corpus must be **sharded by semantic cluster**: each bridge holds one cluster of
> ~64 mutually-similar concepts (e.g. an "animals" bridge, a "vehicles" bridge, a "foods" bridge).
> Within-bridge → graded generalization. Cross-bridge → the existing composition layer (identity
> binding), with **no graded similarity** between bridges.

This is internally consistent with the existing 320 ensemble, which is *already* sharded by category
(bridgeA_nouns / bridgeB_verbs / bridgeC_adj / bridgeD_spatial / bridgeE_functional) — but sharded by
**part-of-speech**, not by **semantic neighbourhood**. The new requirement is a *finer* sharding:
within a part-of-speech, group by similarity (animals together, not animals-mixed-with-tools).

### 1.3 The honest framing of what cross-bridge similarity costs

The agent will be able to:
- generalize **within** a cluster (cat ≈ dog, if both are in the animals bridge);
- **compose** any two concepts across clusters into a fact (apple is big; dog chases cat) via the
  existing layer;
- **abstain** on unknown facts.

The agent will **not** be able to generalize **across** clusters (cat ≈ dog is fine; cat ≈
some-unrelated-cross-bridge-concept has no representation — but that is *correct*, those concepts are
dissimilar by construction of the shard). The only genuine loss is **cross-cluster near-neighbours
that the sharding split apart** (a concept on a cluster boundary whose nearest neighbour landed in the
adjacent bridge). §3 ranks how to handle this; the leading answer is that for the conversational
matrix it does not need handling (within-cluster graded + cross-cluster composition suffices), which
the cheap-first run is designed to test.

---

## 2. Reusable machinery (the build assembles validated parts)

| Need | Reuse | Notes |
|---|---|---|
| Per-bridge graded learn | `research/runners/learned_assoc_graph.py` `LearnedAssocGraph` + the homeostatic subclass `HomeostaticAssocGraph` in `research/runners/learned_graded_embedding_homeostasis_probe.py` | Default = Oja incoming-L2 renorm, per-post-neuron, on the `pool↔pool` mask; read-out = brain-based divnorm (`learned_graded_embedding_divnorm_readout_probe.divnorm_spreading_readout`, recipe `ch`/interleave/steps2/σ0.001/exp2). Runner-side only, NO `sim/` edits. |
| The graded gate suite (G1 structure recovery + 2nd-order margin, G2 generalization + controls, G5 permuted-co-occurrence) | the `run_seed` / `measure_point` / gate-reconfirm harness in `learned_graded_embedding_homeostasis_probe.py` | Reuse VERBATIM per-bridge. (Apply the §5 G5 fix: gate on a margin/Pearson threshold, not the brittle `is_graded` boolean.) |
| Multi-bridge ensemble + cross-bridge encode/recall | `research/runners/g20_multibridge.py --sparse` + `research/runners/shared_pool_chat.py` (`encode_pair_engram_sparse`, `encode_partial_pair_engram_sparse`, `stim_recall_sparse_rates`) | The V-tag cross-bridge layer, shipped + tested (16 CPU reproducibility tests). |
| Per-bridge sparse pool builder + trainer | `research/runners/concept_pool_sparse_distributed.py` (`build_sparse_pool_bridge`, `generate_sparse_patterns`, the `--save-bridge` trainer) | Used by the 320 chain `g20_sparse_5bridge_chain_320.ps1`. |
| VSA flat-distinct cross-bridge composition (the alternative cross-bridge layer) | `research/findings/raw/_insubstrate_flatdistinct320_test.py`, `_insubstrate_flatdist320_anybank_test.py`, `research/runners/compose_flatdist320_conversation_demo.py` | The distinct-seed-per-bridge route; SVO 1.000, any-bank 0.992. |
| The no-confab moat | `research/runners/familiarity_gate_v320_validation.py` (learned gate) + the host `if` abstention in `rf_phasor_composer.query_agent` | Gate validated at V=320; keep host as belt-and-suspenders. |
| The conversational capability matrix tests | `tests/test_brain_conversational_agent.py`, `tests/test_core_sim_composition.py` | The who/what Q&A + abstention + negation + clauses + dialogue suite. |
| Synapse/feasibility sizing | analytic (this doc §6) | A per-bridge 64-concept graded recurrent at the validated ~37.5 neurons/concept density is ~3.6M synapses — ~25× below the V=320 single-pool wall (88.6M), ~100× below the V=640 OOM (354M). 2–3 such bridges are comfortably feasible. |

---

## 3. Ranked design options — sharding + cross-bridge similarity

### Question A — how is the corpus sharded into per-cluster bridges?

**A1 (RECOMMENDED for the build; not needed for the cheap-first run) — semantic-cluster sharding via
the co-occurrence graph the build already produces.** The build's learn consumes a co-occurrence
corpus (the agent's own subject-verb-object knowledge base + optional Tiny Shakespeare). Run a single
global pass of the **same `LearnedAssocGraph` learn over a coarse, low-resolution pool of ALL
concepts** purely to get the concept-concept co-occurrence/similarity matrix, then **cluster it**
(e.g. spectral / agglomerative clustering into 32 balanced clusters of ~64). Each cluster → one
bridge. This is self-consistent (the same statistics that make within-bridge graded structure also
decide the shards) and reuses the learn machinery.
*Risk (honest, see §7):* this is partly circular — deciding the shards needs the very similarity
structure the per-bridge learn is meant to produce. Mitigation: the coarse global pass only needs to
be good enough to **cluster** (a much weaker requirement than producing clean per-bridge graded
codes), and the cluster pool can be small/cheap because it is discarded after sharding. If even the
coarse clustering is unreliable, fall back to A2.

**A2 (BASELINE; what the cheap-first run uses) — curated category sharding.** Shard by a
hand-authored taxonomy (animals / vehicles / foods / …), exactly as the existing 320 ensemble shards
by part-of-speech. Zero dependence on the learned structure; deterministic; reproducible. For the
cheap-first falsification this is the right choice (it removes one confound — we want to test the
graded + composition mechanisms, not the clustering algorithm). For the full build A1 is preferred
because hand-curation does not scale cleanly to 2,048 concepts.

**A3 (fallback) — random sharding.** Concepts assigned to bridges at random. This deliberately
**destroys** within-bridge graded structure (a cluster's members scatter across bridges). It is NOT a
production option — it is valuable only as the **sharding ANTI-CHEAT control** (§4): under random
sharding, within-bridge graded generalization must COLLAPSE to chance, proving the A1/A2 graded result
is real semantic co-location and not an artifact.

### Question B — is cross-bridge graded similarity needed at all?

**B1 (RECOMMENDED — the working hypothesis the cheap-first run tests) — NO; within-bridge graded +
cross-bridge composition suffices for the conversational matrix.** The conversational matrix
(who/what Q&A, abstention, negation, one/two-attribute, clauses, dialogue) consumes **identity
binding** (compose a fact, retrieve its roles, abstain) — which the existing cross-bridge layer
already delivers at 320. The genuinely-new capability (graded generalization) is consumed **within** a
cluster (cat ≈ dog inference). So the hypothesis is: **no shared cross-bridge embedding is required.**
The cheap-first run falsifies this directly (if the conversational matrix needs cross-bridge similarity
it will fail on cross-bridge attribute/clause facts).

**B2 (if B1 is falsified) — a coarse cross-bridge "which-cluster" router, not a shared embedding.**
Represent only **cluster-level** similarity: a small router that, given a cue, returns the *bridge*
(cluster) most likely to hold related concepts, then does within-bridge graded inference there. This
gives "which neighbourhood" without a 2,048-wide shared embedding (which is exactly the OOM-ing object
we are avoiding). Cheap to add because clusters are few (32). Biologically ≈ a coarse
prefrontal/hippocampal index over cortical modules.

**B3 (most expensive; only if B1 AND B2 fail) — a two-level hierarchical embedding.** A learned
**graded embedding over the 32 cluster-centroids** (32 ≪ the OOM wall), composed with the within-cluster
embedding. This reintroduces a second binding level — and the project has a **hard prior** that a
second binding level is dangerous: the hierarchical-320 shortcut scored a catastrophic 0.000 on
structured facts precisely because of "the nesting/multi-hop SNR wall from stacking a 2nd binding
level" (`2026-06-02-flat-distinct...`). So B3 is explicitly **deprioritized**; it is listed only for
completeness, and would itself need its own de-risk against that known failure mode.

### Question C — which cross-bridge composition layer (V-tag vs V-vsa)?

**C1 (RECOMMENDED for the cheap-first run) — V-tag (`g20_multibridge --sparse`).** It is the SHIPPED,
tested 320 ensemble, with a working chat surface, and it is graded-embedding-agnostic, so it drops in
on graded-coded bridges unchanged (the cross-bridge link is the tag name + per-bridge identity
recall). Lowest integration risk; the fastest path to a PASS/FAIL signal.

**C2 (the brain-analogue target for the build) — V-vsa (flat-distinct composition).** This is the
mechanism the owner's "brain-analogue, not retrieval-ranking" standard prefers. For the full build,
the integration target is to feed the **graded** per-bridge codes (not the maximally-distinct
distinct-seed codes) into the single-level VSA bind/unbind. Open sub-question (honest): VSA cleanup
wants **distinct** codes, but within a cluster the graded codes are **deliberately similar** — so
cross-cluster composition (distinct clusters) is fine, but **within-cluster** composition of two
similar concepts may have lower cleanup SNR. This is a real interaction to measure (the cheap-first
run measures the cross-cluster case; a within-cluster-composition micro-check is a cheap follow-on).
For the cheap-first falsification, use C1; flag C2's within-cluster-cleanup question as the next
de-risk if C1 passes.

---

## 4. THE CHEAP-FIRST FALSIFICATION

### 4.1 Design rationale (what is and is not at risk)

The recipe is already GO single-pool at V = 160 (> 64), so **within-bridge graded generalization at
64 concepts/bridge is NOT the risk** — it is expected to pass and serves as a **precondition check**
(if it fails at 64, something is wrong with the integration, not the science). The genuine
load-bearing risks the small run must hit are:

1. **Does the existing cross-bridge composition layer still work when each bridge carries a learned
   GRADED code instead of the orthogonal sparse code it was validated on?** (The 320 ensemble used
   orthogonal/sparse codes; graded codes are *correlated within a bridge* — higher within-bridge
   spurious co-activation, a potential SNR hit for tag recall.)
2. **Does the no-confab moat survive** on graded-coded, multi-bridge facts (abstain on absent
   cross-bridge facts)?
3. **Does within-bridge graded generalization coexist with cross-bridge composition on the SAME
   bridges** without one breaking the other?

A **2–3 bridge** run exercises every cross-bridge code path (routing, partial-pair encode in two
bridges, tag-name aggregation, abstention) at full fidelity — 32 bridges is **more of the same fan-out**
(the §7 scaling risk), not a new code path. So 2–3 bridges falsifies the mechanism cheaply; 32 tests
only the fan-out.

### 4.2 What to build (controller's task after approving this design)

1. **Shard:** pick **3 small semantic clusters** of 64 concepts each (A2 curated sharding), e.g.
   `animals` / `foods` / `vehicles`. Each cluster's concepts are mutually similar (so within-bridge
   graded generalization is meaningful) and the three clusters are mutually dissimilar (so cross-bridge
   is pure identity composition). Author a small co-occurrence corpus per cluster (subject-verb-object
   facts within the cluster, mirroring the homeostasis-probe corpus shape: hub + member + bridge
   facts) **plus** a handful of **cross-bridge facts** (e.g. `dog eats meat` with dog ∈ animals, meat
   ∈ foods; `car carries dog`).
2. **Train 3 graded bridges** with the validated homeostatic learn (Oja, set-point calibrated to the
   per-bridge pool — at 64 concepts/~2.4k pool the homeostasis-probe calibration applies; see §6),
   read out the graded codes, **save each bridge**. This is a new thin runner — call it
   `multibridge_graded_derisk.py` — that wraps `HomeostaticAssocGraph` per cluster and writes a
   bridge per cluster, reusing the gate harness.
3. **Run the per-bridge graded gates** (G1/G2/G5) on each of the 3 bridges (precondition check).
4. **Run cross-bridge composition** over the 3 bridges via the existing V-tag layer
   (`g20_multibridge --sparse` adapted to load the graded bridges, OR the thin runner calls
   `encode_pair_engram_sparse` / `encode_partial_pair_engram_sparse` + `stim_recall_sparse_rates`
   directly). Store the within- and cross-bridge facts; query them back.
5. **Run the abstention battery** (known cross-bridge facts → answer; absent cross-bridge facts →
   abstain) with the familiarity gate validated alongside the host check.
6. **Run the conversational matrix subset** (who/what Q&A, one-attribute, yes-no/negation, a 2-hop
   clause) on the 3-bridge ensemble.

### 4.3 What to measure + GO / BOUNDARY / NEGATIVE criteria

| # | Measurement | GO | BOUNDARY | NEGATIVE |
|---|---|---|---|---|
| **M1** precondition: per-bridge within-bridge generalization (held-out-neighbour, A1) on all 3 bridges; orthogonal (A2) + permuted-property (A3) controls collapse | ≥ 0.7 all 3 bridges, controls ≤ ~1.5× chance | 0.5–0.7 on a bridge | < 0.5 on a bridge (recipe broke at 64/integration) |
| **M2** within-bridge structure-recovery Pearson(sim,S_true) + 2nd-order cat~dog margin per bridge | Pearson ≥ 0.7 **or** (margin ≥ +0.10 **and** gen ≥ 0.7) all 3 | margin ≥ +0.10 but Pearson 0.33–0.7 (the V=160 read-out-degradation pattern; gen still passes) | margin < +0.10 |
| **M3** **cross-bridge composition recall** (store N cross-bridge facts e.g. `dog eats meat`; query cue → target retrieved as top-1/top-2 across bridges, signal clears the noise floor) | ≥ 80% of cross-bridge facts recall target in top-2, signal decisively > noise floor (≳ 1.5×, as in the 320 demo `882 vs 518`) | 50–80% | < 50% (graded codes' within-bridge correlation collapses cross-bridge tag SNR) |
| **M4** **no-confab moat** (abstention-floor: absent cross-bridge facts) | zero false-accepts; learned gate agrees with host on every cue; lesion collapses separation | gate margin shrinks but zero breaches | any moat-breach (host-abstain / accept) on graded codes |
| **M5** **conversational matrix subset** on the 3-bridge ensemble | who/what + one-attribute + yes-no/negation + a clause all pass, no regression vs the single-bridge agent | one capability degrades | a core capability (Q&A or abstention) fails on the merged graded ensemble |
| **M6** **sharding anti-cheat (A3 random shard)** | random-shard within-bridge generalization COLLAPSES to chance (proving M1 is real co-location) | partial collapse | random shard still "generalizes" (M1 was an artifact, not semantics) |
| **M7** **cross-bridge permuted-mapping anti-cheat** | shuffling which target each cue is bound to COLLAPSES cross-bridge recall to chance (proving M3 is the stored binding, not structural bias) | partial | permuted recall ≈ true recall (M3 was an artifact) |

**Overall verdict:** **GO** = M1+M2 pass (precondition holds) AND M3+M4+M5 pass (cross-bridge
composition + moat + conversational matrix all survive graded codes) AND M6+M7 anti-cheats clean.
**BOUNDARY** = the capability is real but one quantity sits in its BOUNDARY band (e.g. cross-bridge
recall 50–80%, or the V=160-style Pearson degradation) — characterize precisely; it becomes a
documented build constraint (e.g. "graded codes need a higher tag teacher current / `top_k` to clear
the noise floor", a tuning knob, not a wall). **NEGATIVE** = M3 or M4 fails — i.e. graded codes break
cross-bridge composition or the moat — which would mean the within-bridge graded gain is **incompatible**
with the multi-bridge layer as-is, and the build needs a different cross-bridge representation (B2
router) before scaling. An honest NEGATIVE here is the deliverable: it maps exactly what the substrate
can and cannot do, before any 32-bridge spend.

### 4.4 The exact run the controller would execute

Sizing (per §6): 3 bridges × 64 concepts × ~2.4k-pool graded recurrent ≈ 3.6M synapses each — far below
the single-pool wall; all three fit with headroom. Each per-bridge graded learn is ~minutes (the
homeostasis probe's 30-concept learn is ~30–120 s/cycle-count; 64 concepts × cycles≈10 is ~single-digit
minutes), cross-bridge encode/recall is seconds/fact. **Total: a few hours, not days.** Use
`SIM_BACKEND=cupy` (GPU); the moat sub-check (tiny bridges + numpy linear algebra) can run on
`SIM_BACKEND=numpy` exactly as the V=320 familiarity-gate validation did.

```bash
# (1) Author 3 cluster vocabs + corpora (controller; small text files):
#     research/findings/raw/g11_bg/graded_derisk/{animals,foods,vehicles}_vocab64.txt
#     + a per-cluster within-cluster SVO corpus + a shared cross-bridge fact list.

# (2) Train 3 GRADED bridges with the validated homeostatic learn + run the per-bridge
#     graded gates (M1/M2). New thin runner wrapping HomeostaticAssocGraph (NO sim/ edits):
SIM_BACKEND=cupy python -m research.runners.multibridge_graded_derisk \
    --mode train-and-gate \
    --clusters animals,foods,vehicles \
    --vocab-dir research/findings/raw/g11_bg/graded_derisk \
    --n-concepts-per-bridge 64 --n-pool 2400 --pattern-size 100 \
    --homeo oja --homeo-target 40 --cycles 10 \
    --readout-divnorm ch --readout-order interleave \
    --readout-sigma 0.001 --readout-exponent 2.0 --diffusion-steps 2 \
    --seeds 42,43,44 \
    --save-bridge-dir research/findings/raw/g11_bg/graded_derisk/bridges \
    --out research/findings/raw/_multibridge_graded_derisk_pergate.json

# (3) Cross-bridge composition + moat + conversational subset (M3/M4/M5) over the 3 graded bridges,
#     reusing the V-tag layer (encode_pair_engram_sparse / encode_partial_pair_engram_sparse /
#     stim_recall_sparse_rates from shared_pool_chat.py) + the familiarity gate alongside the host check:
SIM_BACKEND=cupy python -m research.runners.multibridge_graded_derisk \
    --mode crossbridge-eval \
    --load-bridge-dir research/findings/raw/g11_bg/graded_derisk/bridges \
    --clusters animals,foods,vehicles \
    --within-facts research/findings/raw/g11_bg/graded_derisk/within_facts.txt \
    --cross-facts  research/findings/raw/g11_bg/graded_derisk/cross_facts.txt \
    --abstention-floor research/findings/raw/g11_bg/graded_derisk/absent_facts.txt \
    --familiarity-gate --keep-host-moat \
    --seeds 42,43,44 \
    --out research/findings/raw/_multibridge_graded_derisk_crossbridge.json

# (4) Anti-cheats (M6 random-shard collapse + M7 permuted cross-bridge mapping collapse):
SIM_BACKEND=cupy python -m research.runners.multibridge_graded_derisk \
    --mode anticheat \
    --vocab-dir research/findings/raw/g11_bg/graded_derisk \
    --n-concepts-per-bridge 64 --n-pool 2400 --pattern-size 100 \
    --homeo oja --homeo-target 40 --cycles 10 \
    --random-shard --permuted-crossbridge-mapping \
    --seeds 42,43,44 \
    --out research/findings/raw/_multibridge_graded_derisk_anticheat.json
```

> The thin runner `multibridge_graded_derisk.py` does **not** exist yet — building it is the
> controller's first step after approving this design. It is ~a few hundred lines that compose
> `HomeostaticAssocGraph` (train + gate, per cluster), `shared_pool_chat` sparse encode/recall (the
> V-tag cross-bridge layer), and `familiarity_gate_v320_validation` (the moat), with the three `--mode`
> branches above. NO `sim/` edits (everything is runner-side, exactly as every cited probe is).
> Calibration note: `--homeo-target 40` is the V=160 Oja set-point; at the smaller ~2.4k pool the
> set-point should be **re-bracketed cheaply** (sweep {20, 40, 80}) on bridge 1 before committing all
> three — the homeostasis findings show the Oja L2 set-point is robust but pool-size-dependent.

### 4.5 If the cheap-first run reveals the V-tag layer is the wrong cross-bridge mechanism

The fallback is **not** a redesign — it is to swap the cross-bridge layer to **V-vsa** (the
flat-distinct VSA composition, `_insubstrate_flatdist320_anybank_test.py` machinery) and re-run M3–M5,
since V-vsa is the project's strongest validated cross-bridge composition (any-bank 0.992) and is the
brain-analogue target anyway. This is queued as the immediate next de-risk if C1 (V-tag) hits BOUNDARY
on M3, and carries the §3-C2 within-cluster-cleanup question.

---

## 5. Anti-cheat controls (mandatory, beyond M6/M7 above)

- **Sharding control (M6, A3 random shard):** the decisive control that within-bridge graded
  generalization is **real semantic co-location**. Under random sharding the per-bridge clusters are
  destroyed, so M1 generalization MUST collapse to chance. If it does not, M1 was measuring something
  other than learned similarity. This is the multi-bridge analogue of the single-pool A2 orthogonal
  contrast.
- **Cross-bridge permuted-mapping (M7):** shuffle which target each cue's cross-bridge fact binds to;
  recall MUST collapse to chance. Mirrors the project's standing permuted-label / permuted-co-occurrence
  controls (e.g. the engram-composition 24-permutation anti-cheat) — guards against the historical
  "structural bias masquerading as learning" failure (2026-05-03 permuted-label NEGATIVE).
- **Moat-preserving validation (M4):** the familiarity gate is validated **alongside** the host
  abstention check (never replacing it), exactly as in `familiarity_gate_v320_validation.py` — the gate
  may accept only where the host accepts; the load-bearing cell (host-abstain / gate-accept) must be 0.
  This keeps the moat un-weakened even while testing its neural form on graded codes.
- **Graded-vs-orthogonal cross-bridge contrast:** run M3 cross-bridge recall **also** on the existing
  orthogonal-coded 320 bridges (already trained) as the reference — if graded-coded M3 is materially
  worse, that quantifies the SNR cost of within-bridge correlation on the cross-bridge layer (the
  expected BOUNDARY mechanism), distinguishing "graded broke it" from "the harness is wrong".
- **G5 control-criterion fix (carried from the homeostasis finding):** when reusing the gate harness,
  gate the permuted-co-occurrence anti-cheat on a **margin/Pearson threshold** (permuted 2nd-order
  margin < +0.10), NOT the bare `is_graded` boolean — the boolean is a coin-flip on a structureless
  permuted matrix and produced a spurious seed-43 BOUNDARY label in the homeostasis run
  (`2026-06-11-learned-graded-embedding-homeostasis-GO.md` §4).

---

## 6. Feasibility / sizing (analytic, CPU-checked)

The learned recurrent's `pool↔pool` synapses scale ~quadratically with pool size; the two measured
anchors imply a consistent internal density d ≈ 0.62 (V=160 pool 7,000 → 30.7M; V=320 pool 12,000 →
88.6M). At the validated ~37.5 neurons/concept density, a per-bridge **64-concept** graded pool is
~2,400 neurons → **~3.6M recurrent synapses**:

| Object | synapses | feasible? |
|---|---|---|
| per-bridge graded 64-concept recurrent (~2.4k pool) | ~3.6M | **yes (huge headroom)** |
| single-pool V=320 | 88.6M | yes (near the wall) |
| single-pool V=640 | 354M | **no — OOM at synapse install** |

So 2–3 graded bridges co-resident are ~7–11M synapses total — ~8–12× **below** the single feasible
V=320 pool, and ~30–50× below the V=640 OOM. The cheap-first run is comfortably feasible; **32 bridges
× 64** at ~3.6M each is ~115M synapses if co-resident, but the production architecture loads bridges
**sequentially / on demand** (the existing 320 ensemble loads 5 bridges; the 2,048 build would page
bridges, not hold all 32 in VRAM), so the full build's memory is bounded by the few bridges live at
once — which is exactly why multi-bridge dodges the single-pool wall.

---

## 7. Honest risk list (what could make this NEGATIVE)

1. **Cross-bridge composition SNR collapses on graded codes (the primary risk, M3).** The V-tag layer
   was validated on **orthogonal** sparse codes (between-cosine ≈ 0); graded codes are **correlated
   within a bridge**, raising within-bridge spurious co-activation and potentially drowning the
   cross-bridge tag-recall signal in the noise floor. The 320 demo already noted "higher noise floor at
   64-vs-32 concepts" — graded codes push that further. *If M3 is NEGATIVE, graded + V-tag are
   incompatible and the build must move to V-vsa or the B2 router.* (This is the single most important
   thing the cheap-first run tests, and the reason 64-concept graded bridges — not orthogonal — must be
   used in the run.)
2. **Sharding can't be decided without the graded structure we're building (the circularity, A1).**
   Semantic-cluster sharding needs a concept-similarity matrix, which is downstream of the very learn
   that produces graded codes. Mitigated for the cheap-first run by using curated sharding (A2, no
   circularity); a real risk only for the **full** build's A1 step, where the coarse global clustering
   pass must be good enough to cluster (weaker than producing clean codes). If even coarse clustering is
   unreliable at 2,048 concepts, the build falls back to curated/taxonomic sharding (A2) — usable but
   less principled.
3. **The 32-bridge fan-out the cheap-first run does NOT test.** 2–3 bridges exercise every code path but
   not the **scale** of cross-bridge search (32 bridges × ~66 tags each = thousands of tags to search
   per query) or the abstention-floor at 2,048 distractors. The moat finding shows the gate margin holds
   to V=1280 with zero breaches (4× the validated scale) — encouraging — but 2,048 cross-bridge facts is
   the genuine open scaling question that only a (later, gated) larger run answers. The cheap-first run
   de-risks the *mechanism*; the fan-out is a separate, explicitly-deferred gate.
4. **Within-cluster VSA cleanup (if the build uses V-vsa, C2).** VSA cleanup wants distinct codes;
   within a cluster the graded codes are deliberately similar, so composing two **similar** concepts
   from the same bridge may have low cleanup SNR. Not tested by the cross-bridge cheap-first run (which
   composes across dissimilar clusters); flagged as the C2 follow-on micro-check.
5. **Read-out global-Pearson degradation at the per-bridge scale (M2 BOUNDARY band).** The V=160
   scale-check showed the brain-based divnorm read-out's *global* Pearson can degrade while
   *generalization* still passes; the fix was the production homeostatic learn (not the de-saturation
   stand-in). The cheap-first run must use the **homeostatic** learn (Oja), which recovered Pearson to
   near-ceiling at V=160 — but if the smaller per-bridge pool re-introduces the degradation, M2 lands in
   BOUNDARY (gen still passes), a documented constraint, not a wall.
6. **Integration interactions not seen when pieces were validated separately.** The architecture proof,
   the homeostatic learn, the cross-bridge layer, and the moat were each validated in isolation. The
   cheap-first run is the first time the graded learn and the cross-bridge layer touch — interaction
   bugs (e.g. the graded read-out's code format vs what the sparse encode/recall expects) are possible;
   front-loading them in a 3-bridge run (not a 32-bridge build) is the whole point.

---

## 8. Recommendation (one paragraph)

Shard the corpus by **semantic cluster** (curated A2 for the cheap-first run, co-occurrence-graph
clustering A1 for the full build), keep cross-bridge relationships in the **existing composition
layer** (V-tag for the cheap-first run because it is shipped + tested; V-vsa as the brain-analogue
build target), and **do not** build a cross-bridge shared graded embedding (working hypothesis B1:
within-bridge graded + cross-bridge identity composition suffices for the conversational matrix). The
single cheap run that falsifies this — **3 graded bridges × 64 concepts**, the homeostatic Oja learn
per bridge, the V-tag cross-bridge layer, the familiarity-gate moat, plus the random-shard and
permuted-mapping anti-cheats — costs a few hours on the GPU and answers the only load-bearing question
that 2–3 bridges can answer that 32 cannot any better: **does the within-bridge graded gain coexist
with cross-bridge composition and the no-confab moat, or does graded coding break the existing
cross-bridge layer?** GO → scope the 32-bridge build (gated, sequential-load); NEGATIVE on cross-bridge
recall or the moat → an honest map of the substrate boundary and a pivot to V-vsa / a coarse cluster
router before any large spend.
