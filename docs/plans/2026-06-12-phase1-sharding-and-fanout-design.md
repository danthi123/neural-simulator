---
type: plan
status: live
date: 2026-06-12
---

# Phase 1 of "step 3 true cortex" — SHARDING the 2,048 concepts + the 32-BRIDGE FAN-OUT de-risk + the production TRAIN/GATE plan

> **Status:** present-before-build. READ-ONLY design (no `sim/` edit, no GPU run, no bridge built — only tiny
> CPU-only analytic checks). This is the SECOND half of Phase 1 of the production build, downstream of the
> composer-architecture decision (`docs/plans/2026-06-12-phase1-composer-architecture-design.md`, route A = per-bridge
> composers + cross-bridge identity layer, recommended, de-risk in flight). It is the project's standing
> "design + cheap-first de-risk BEFORE building" opening move applied to the three remaining Phase-1 pieces:
> **(1) how the 2,048 production concepts are sharded into 32 bridges of 64; (2) the 32-bridge fan-out de-risk
> (cross-bridge composition + the no-confab moat at 4× the validated 8-bridge fan-out); (3) the production-scale
> train + gate plan.** **Date:** 2026-06-12. **Author role:** read-only design subagent. Every load-bearing claim
> is cited to a file read in full; every number is from a CPU-only analytic check shown in §0.4.

---

## 0. Terms, scope, and the analytic checks this rests on

### 0.1 Terms (defined once)

- **Bridge / shard** — one small spiking pool of 64 concepts. The learned-graded cortex is sharded into 32 bridges
  to dodge the single-pool quadratic memory wall (`docs/plans/2026-06-11-semantically-structured-cortex-BUILD-PLAN.md`
  §"Scaling path", lines 78–101). I use "bridge" and "shard" interchangeably (the code does: `SHARD_NAMES`,
  `n_bridges`, `concepts_per_bridge` in `research/runners/multibridge_graded_derisk.py:147, 968–969`).
- **Within-bridge generalization** — answering a query about a held-out concept via a *similar* known concept in
  the SAME bridge (cat ≈ dog → "what does a dog eat?" → meat). It works ONLY because the bridge's 64 concepts carry
  a learned *graded* code (correlated by similarity), and ONLY across concepts in the same bridge
  (`cortex_conversation_capability_GO.md`, B1 = 0.99–1.00; `multibridge_graded_derisk-GO.md`, M1 GO).
- **Cross-bridge composition** — storing and recalling a fact whose concepts span two bridges (`dog eats meat`,
  dog∈animals, meat∈foods). Realized by the **V-tag identity layer**: a Tonegawa engram-tag (catalog D.14) named
  `"<cue>__<target>"` imprinted in BOTH bridges over each bridge's spiking `pool` region; recall stimulates the tag
  and reads per-concept firing in the target bridge (`GradedBridge.encode_tag` / `cross_bridge_eval`,
  `multibridge_graded_derisk.py:480, 551`). This is **identity recall** (which concept), not generative binding.
- **The moat** — the no-confab abstention: the agent returns `None`/abstains when no stored fact matches, validated
  by the learned `RelationalFamiliarityGate` ALONGSIDE the host abstention check, with zero breaches
  (`familiarity_gate_v320_validation.py`; `moat_eval`, `multibridge_graded_derisk.py:644`).
- **Fan-out** — the number of co-resident bridges. The cross-bridge layer + moat are validated to **8-bridge
  fan-out** (`cortex_conversation_capability_GO.md` line 5: "holds to 8-bridge fan-out"; the mechanism de-risk ran 3,
  the composer-architecture de-risk runs 8). 32 bridges is **4× the validated fan-out** — the load-bearing scaling
  risk this design's de-risk measures.
- **POS-sharding vs semantic-cluster-sharding** — TWO different ways to assign concepts to bridges, and the crux of
  §1. The existing g20 ensemble shards by **part of speech** (one bridge of nouns, one of verbs, …;
  `research/runners/g20_vocab_spec.py:18–119`). The build plan REQUIRES sharding by **semantic cluster** (mutually
  *similar* concepts together — animals together, foods together) for within-bridge generalization to be meaningful
  (build plan §"Genuine open questions" item 1, line 94). These are NOT the same and the difference is load-bearing.

### 0.2 Where this sits (what is already decided / de-risked)

| Piece | Validation (file read in full) | Status |
|---|---|---|
| Cross-bridge composition + the moat **survive on correlated graded codes** | `2026-06-12-multibridge-graded-derisk-GO.md` (3 bridges × 64, 3 seeds) | GO |
| The conversational matrix on the learned cortex + **generalization-in-conversation** + moat | `2026-06-12-cortex-conversation-capability-GO.md` (1 shard × 64, 3 seeds) | GO |
| The 3-bridge ENSEMBLE de-risk (matrix spanning bridges + generalization + cross-bridge + moat) | `2026-06-12-cortex-conversation-3bridge-ensemble-GO.md` (D=512) | GO |
| The composer architecture (route A per-bridge composers + cross-bridge V-tag) | `docs/plans/2026-06-12-phase1-composer-architecture-design.md` | DESIGNED (de-risk in flight) |
| Cycle-independent homeostatic Oja learn at V=320 single-pool | build plan piece (ii), line 44 ("V=320 multi-seed = CLEAN GO 3/3") | GO |
| The build plan + its scaling path (single-pool wall → 32-bridge multi-bridge) | `docs/plans/2026-06-11-semantically-structured-cortex-BUILD-PLAN.md` | APPROVED-PENDING-BUILD |

**What is NOT yet decided — and is THIS doc's job.** (1) The concrete source of the 2,048 concepts and HOW they are
assigned to 32 bridges so within-bridge generalization is meaningful — the existing g20-320 spec shards by POS, NOT
by semantic cluster, which is a discrepancy the build plan's own requirement exposes. (2) The 32-bridge fan-out
de-risk — extending the validated 8-bridge mechanism to 32 (memory feasibility + does SNR + the moat hold). (3) The
production train + gate plan + its honest wall-clock cost.

### 0.3 The composer-architecture dependency (and why this doc is robust to its outcome)

The composer-architecture de-risk (the prior design) is in flight and recommends **route A** (per-bridge composers +
cross-bridge V-tag identity layer). **This sharding/fan-out design assumes route A** (per the recommendation) but is
LARGELY ROUTE-INDEPENDENT for pieces (1) and (2):

- **Sharding (piece 1)** is needed by BOTH routes identically — both shard the cortex into 32 bridges; the difference
  is only the *composer* object, not the *cortex sharding* (composer-architecture design §4.2: "both routes shard
  the cortex; the difference is the composer"). So piece (1) stands regardless.
- **The 32-bridge fan-out de-risk (piece 2)** measures the cross-bridge V-tag layer + the moat at 32 bridges. The
  V-tag identity layer is route A's cross-bridge mechanism AND is also present in route B's ensemble (the X-vtag gate
  runs in `cortex_conversation_ensemble_derisk.py` regardless of composer mode). So the V-tag fan-out de-risk is
  needed either way; under route A it is THE cross-bridge mechanism, under route B it is the spiking cross-bridge
  realization alongside the generative X-conv path.
- **Only the per-bridge composer COST** (piece 3's train plan) is route-specific: route A builds 32 small composers
  (D≈256, vocabulary-independent cost; composer-architecture design §4.2), route B one D≈5.5k union composer. The
  train plan §3 notes the route-A numbers and flags the route-B delta.

**If the composer de-risk returns route B or the hybrid instead of route A**, pieces (1) and (2) are unchanged; only
§3's per-bridge-composer build step changes to "build the union composer" (route B) or "+ one small cross-bridge
composer" (hybrid). This doc flags those branch points inline.

### 0.4 The CPU-only analytic checks this design rests on (no bridge, no GPU)

All numbers below are from `python3 -c` arithmetic over the actual measured constants in the cited findings — no
`SimulationBridge` was constructed, no GPU touched.

| Quantity | Value | Source of the constant |
|---|---|---|
| Synapses per 64-concept graded bridge (n_pool=2400) | **3.46M** | `2026-06-12-multibridge-graded-derisk-GO.md` line 21 |
| 32 co-resident bridges, synapses | 32 × 3.46M = **110.7M** (0.111B) | arithmetic |
| Bytes per synapse (from the build plan's V=320 datum) | 7.7 GB / 88.6M = **86.9 B/syn** | build plan line 85 (V=320 pool 12000 → 88.6M syn = 7.7 GB) |
| **32 co-resident bridges, synapse memory** | 110.7M × 86.9 B = **9.6 GB** | arithmetic |
| Per-bridge state arrays (~12 × 2700 neurons × 8 B) | **0.26 MB** (negligible) | arithmetic (pool 2400 + ~10–20% fs) |
| GPU budget (RTX 3090) | **24 GB** | CLAUDE.md / build plan |
| Headroom after the 9.6 GB of synapses | **~14.4 GB** | arithmetic (before CuPy pool, per-op RF bridges, fragmentation) |
| Production vocab | **2,048 = 32 × 64** | build plan §"Scaling path" line 91 |
| Existing g20 spec vocab | **320 = 5 × 64** | `g20_vocab_spec_320.py:155` (`TOTAL_VOCAB_64 = 320`) |
| Extension needed | **+1,728 concepts = +27 bridges × 64** | arithmetic |

**The single most important analytic result: 32 co-resident graded bridges need ~9.6 GB of synapse memory, which
FITS the 24 GB GPU with ~14 GB of headroom** — and the pinned-memory-pool wall that caused the V=640 single-pool OOM
(`build plan line 86`: "pinned-memory pool exhausted transferring 354M synapses host→device") does NOT apply, because
the 110.7M synapses are installed as **32 small ~3.46M-synapse installs one bridge at a time** (each install is tiny),
not one 354M-synapse transfer. **⇒ co-residence is feasible; a sequential build-and-keep is the default, with a
build-and-evict mitigation held in reserve (§2.4).** This is the de-risk's central feasibility finding, computed before
any GPU spend.

---

## 1. Sharding 2,048 concepts → 32 bridges × 64 (the semantic-cluster requirement)

### 1.1 The load-bearing constraint, restated precisely

Within-bridge generalization (the whole point of the learned-graded cortex — the capability the idealized composer
algebra could NOT deliver, `cortex_conversation_capability_GO.md` line 5) requires that the 64 concepts in a bridge be
**mutually similar**, so that the graded code places them close and a held-out concept inherits a neighbour's facts.
The C4 anti-cheat PROVES this is not optional: **random sharding collapses within-bridge generalization to chance**
(`cortex_conversation_capability_GO.md` line 24: "C4 random-shard (destroy semantic co-location) → B1 collapses.
Co-location is load-bearing"; the same as M6 in `multibridge_graded_derisk-GO.md` line 19). So the production sharding
must guarantee within-bridge semantic coherence — it is a correctness requirement, not a nicety.

**The discrepancy the build plan's own requirement exposes (load-bearing):** the existing g20 ensemble shards by
**part of speech** — `bridgeA_nouns`, `bridgeB_verbs`, `bridgeC_adj`, `bridgeD_spatial`, `bridgeE_functional`
(`g20_vocab_spec.py:113–119`). A POS bridge is NOT a semantic cluster: the noun bridge mixes `apple, river, dog, cat,
bird, … tree, flower, … ball, key, book, … hand, foot, … person, baby, … house, road, … water, fire, sun, moon`
(`g20_vocab_spec.py:18–33`) — animals, plants, body parts, places, and substances all in one bridge. `dog` and `cat`
are similar; `dog` and `moon` are not. So a POS-sharded noun bridge does NOT satisfy the "64 mutually-similar concepts"
requirement — within-bridge generalization would work for the `dog≈cat` sub-block and FAIL across the bridge's
unrelated sub-blocks. **The g20-320 spec is a retrieval-tier sharding (each bridge a 64-concept Kanerva-SDM sparse
store, validated for IDENTITY recall at 98.4%, `g20_vocab_spec_320.py:11–14`), NOT a graded-generalization sharding.**
For the dual/CLS learned-graded cortex, sharding must be by semantic cluster.

**The crux for the de-risk's stand-in:** the cheap-first de-risks (3/8-bridge) used CURATED semantic super-clusters
(`SHARD_NAMES = ["animals", "foods", "vehicles", "tools", "clothes", "furniture", "plants", "weather"]`,
`multibridge_graded_derisk.py:147`), each an internally-structured set of mutually-similar members built by
`build_bridge_corpus` (hub-mediated sub-clusters, `multibridge_graded_derisk.py:169–199`). That is the CORRECT
sharding type — but it is 8 SYNTHETIC clusters, not 32 real ones over a real 2,048-concept vocabulary. The production
sharding must produce 32 REAL semantic clusters of 64 mutually-similar concepts each.

### 1.2 The options, ranked

**Option (a) — extend the existing curated g20 sharding to 2,048. ✗ NOT RECOMMENDED as-is (wrong sharding axis).**
The g20-320 spec (`g20_vocab_spec_320.py`) is the validated 5-bridge × 64 sharding, and one's first instinct is "just
extend it 320→2,048." But §1.1 shows the g20 axis is **part of speech**, which does NOT guarantee within-bridge
similarity. Extending the POS axis to 2,048 (e.g. 8 noun bridges, 8 verb bridges, …) would put `dog` and `airplane` in
different bridges (good) but `dog`, `oak-tree`, and `hammer` could land in the same "nouns-3" bridge (bad — they are
not mutually similar). **So option (a) is reusable for VOCABULARY (the curated word lists are a high-quality,
duplicate-checked, hand-validated word source — `g20_vocab_spec_320.py:138–157` asserts global uniqueness) but NOT for
the BRIDGE ASSIGNMENT.** The right move is to reuse the g20 WORDS and RE-CLUSTER them semantically (option c).

**Option (b) — co-occurrence-graph clustering of the agent's own KB. ✗ CIRCULAR at build time.** The build plan
names the corpus source as "the agent's own SVO-fact KB co-occurrence (on-substrate, no download)" (build plan piece
ii, line 42). One could cluster concepts by how often they co-occur in the KB and assign clusters to bridges. **But
this is circular for the FIRST build:** the graded codes are LEARNED FROM the co-occurrence corpus, and the corpus is
what we are trying to shard — there is no learned similarity to cluster on until after the learn, and the learn needs
the sharding to be in place. So option (b) needs an EXTERNAL similarity to bootstrap the clusters (a pre-existing
semantic resource), which is exactly option (c). Option (b) becomes viable as a SECOND-PASS refinement (re-shard using
the learned codes' similarity after a first build) but cannot be the initial sharding. **Defer to a refinement, not
the initial source.** (Honest caveat on "no download": a curated taxonomy is authored once, by hand or from a public
word-category list, then frozen into a Python spec like `g20_vocab_spec_320.py` — it is a static project artifact, not
a runtime download, matching the project's existing practice.)

**Option (c) — a curated semantic taxonomy: 32 super-categories of 64 mutually-similar concepts. ✓ RECOMMENDED.**
Author a `g20_vocab_spec_2048.py` (mirroring `g20_vocab_spec_320.py`'s additive structure + global-uniqueness assert,
`g20_vocab_spec_320.py:138–157`) that defines **32 semantic super-clusters**, each a hand-curated list of 64 mutually-
similar concepts. This is the SAME KIND of object the de-risk's `SHARD_NAMES` already are (animals / foods / vehicles /
… — internally-coherent semantic clusters), just (i) over a real 2,048-word vocabulary, (ii) 32 clusters instead of 8,
and (iii) authored as a frozen spec with the uniqueness assert as the correctness net. Concretely, the 32 clusters are
a taxonomy like: `mammals, birds, fish_reptiles, insects, fruits, vegetables, prepared_foods, drinks, land_vehicles,
air_water_vehicles, hand_tools, machines, clothing, furniture, buildings, body_parts, plants_trees, weather_nature,
kinship_people, motion_verbs, perception_verbs, communication_verbs, manipulation_verbs, emotion_states,
size_shape_adj, color_adj, texture_material_adj, time_words, spatial_words, quantity_number_words, question_discourse,
abstract_relations` (32 clusters; exact membership is the curation work). Each cluster's 64 members are chosen to be
mutually similar (all mammals; all colors), so within-bridge generalization is meaningful by construction and the C4
anti-cheat is satisfied for the right reason.

### 1.3 How within-bridge similarity is GUARANTEED (the correctness argument)

The within-bridge generalization requires the 64 concepts of a bridge to be mutually similar in the LEARNED graded
code, which the learn produces from CO-OCCURRENCE (`learned_assoc_graph.py`, the spiking-Hebbian co-occurrence learner;
build plan piece i). So "mutually similar" reduces to "mutually co-occurring in the corpus." The taxonomy guarantees
this in two layers:

1. **Curation guarantees taxonomic similarity** — all 64 members of `mammals` are mammals, so they share semantic
   features and naturally co-occur with overlapping contexts (all mammals "run", "eat", "breathe").
2. **The corpus must MAKE them co-occur** — the learned-graded recipe needs the within-bridge corpus to contain the
   facts that bind the cluster (the de-risk's `build_toy_cooccurrence` builds hub-mediated sub-clusters precisely so
   members co-occur via shared hubs, `multibridge_graded_derisk.py:181–185`; build plan piece ii: the corpus is "the
   agent's own SVO-fact KB co-occurrence"). **So the production corpus per bridge must be authored/generated to give
   each cluster's 64 members the within-cluster co-occurrence structure the learn consumes** — the same hub/sub-cluster
   structure the de-risk used, but over the real cluster words. This is the corpus-generation work of piece (ii)
   (build plan line 42–44), and the gate that confirms it is **G1 structure recovery + G2 generalization** (build plan
   piece ii, lines 46–49) re-run per bridge at production scale.

**The honest checkpoint:** curation gives a cluster whose members SHOULD co-occur; whether the GENERATED corpus
actually induces the graded structure is measured per bridge by G1/G2 (Pearson(sim, S_true) ≥ 0.7 + the within>between
margin + held-out-neighbour generalization ≥ 0.7). A bridge whose curated members don't achieve G1/G2 is a mis-curated
cluster (a member that doesn't belong, or insufficient within-cluster facts) — caught by the gate, fixed by re-curating
that cluster, NOT a silent failure.

### 1.4 The concrete sharding source + method (the recommendation)

**Source:** a new frozen Python spec `research/runners/g20_vocab_spec_2048.py`, authored mirroring
`g20_vocab_spec_320.py` (additive over the validated 320 base + a global-uniqueness assert as the correctness net,
`g20_vocab_spec_320.py:138–157`). **Method:** 32 hand-curated SEMANTIC super-clusters of 64 mutually-similar concepts
each (option c), RE-CLUSTERING the high-quality g20 word lists (reuse the 320 words as a vetted, duplicate-free word
source) plus ~1,728 new curated words, assigned by taxonomy NOT part of speech. The within-bridge similarity is
guaranteed by curation (taxonomic coherence) + a per-bridge corpus that induces within-cluster co-occurrence, and
CONFIRMED by re-running the G1/G2 structure-recovery + generalization gates per bridge at production scale; a bridge
failing G1/G2 is a mis-curated cluster, caught and re-curated. Co-occurrence-graph re-clustering (option b) is deferred
to a second-pass refinement (re-shard using the learned codes' similarity AFTER the first build), since it is circular
for the initial sharding.

**Implementation cost:** the spec is a static authored artifact (no GPU, no `sim/` edit) — the dominant cost is the
CURATION of 2,048 words into 32 coherent clusters of 64, which is hand-labor (a day or two of authoring + the
uniqueness assert catches collisions). The corpus generation per cluster reuses `build_toy_cooccurrence`'s structure
(or the agent's KB co-occurrence) and is part of piece (ii).

---

## 2. The 32-bridge FAN-OUT de-risk (cross-bridge composition + the moat at 4× the validated fan-out)

### 2.1 The single load-bearing question

The cross-bridge V-tag layer + the no-confab moat are validated to **8-bridge fan-out**
(`cortex_conversation_capability_GO.md` line 5; the composer-architecture de-risk runs 8). 32 bridges is **4× that
fan-out**. The build plan names this as the remaining open question (build plan §"Genuine open questions" item 2, line
95: "Cross-bridge composition at 32 bridges. The project validated cross-bridge composition at 5 bridges; 32 is 6.4×
more — whether composition + the no-confab moat hold at that fan-out is the open question"). **The de-risk asks: at 32
co-resident bridges, does (i) cross-bridge identity recall still retrieve the true target over a NOISE FLOOR that now
includes 31 other bridges' worth of concepts, (ii) the FIXED M7/Cx anti-cheat still collapse (a cue must not retrieve a
WRONG target), and (iii) the moat stay zero-breach when an unknown cross-cue could match any of 2,048 concepts?**

**Why fan-out is a genuine risk, not a formality.** Two distinct SNR pressures grow with fan-out:
- **Cross-bridge recall floor.** The V-tag recall reads per-concept firing in the TARGET bridge and ranks the target
  over that bridge's other 63 concepts (`cross_bridge_eval`, `multibridge_graded_derisk.py:598–615`). The recall is
  scored WITHIN the target bridge (64 concepts), so the per-recall floor is NOT directly 32× worse. BUT the ROUTING —
  finding which tags contain the cue across all 32 bridges (`cross_bridge_eval:595`) — and the moat's abstention
  surface (an unknown cue could spuriously match a tag in any of 32 bridges) DO grow with fan-out. The de-risk
  measures whether the per-recall signal/floor (validated 17–24× at 3 bridges, `multibridge_graded_derisk-GO.md` line
  16) degrades when 32 bridges' tags coexist.
- **The moat's false-accept surface.** The moat must abstain on a cross-cue that matches NO stored fact; at 32 bridges
  there are 2,048 concepts an unknown cue could be confused with, vs 192 at 3 bridges. More candidates = more chances
  for a spurious familiarity. The de-risk re-asserts zero false-accepts at the 32-bridge surface (build plan honest-
  risk 3, line 74: "Moat preservation … non-negotiable").

### 2.2 What changes vs the existing runner (the minimum to reach 32 bridges)

The existing `cortex_conversation_ensemble_derisk.py` ALREADY parametrizes the fan-out via `--n-bridges`
(`cortex_conversation_ensemble_derisk.py:692`) and runs every cross-bridge gate (Gate X V-tag + X-conv, Cx, C3 moat).
The blockers to running it at `--n-bridges 32` are three concrete things:

1. **`SHARD_NAMES` has only 8 entries** (`multibridge_graded_derisk.py:147`). For 32 bridges it needs **32 shard
   names** — the 32 super-cluster names from the §1 taxonomy (`mammals, birds, …`). This is a one-line list extension
   in `multibridge_graded_derisk.py` (or a parameter pointing at `g20_vocab_spec_2048.py`'s 32 cluster names). The
   runner already slices `SHARD_NAMES[:args.n_bridges]` (`:785`), so it just needs ≥32 names available.
2. **The cross-fact corpus must span 32 bridges.** `author_cross_facts` (`multibridge_graded_derisk.py:755–774`)
   already authors (cue, target) pairs spanning two RANDOM different bridges from whatever shards exist, so it scales to
   32 automatically — but `--n-cross-facts` (default 12, `:996`) should be raised so the cross-fact set actually
   EXERCISES many of the 32 bridges as both cue-source and target-source (a 12-fact set touches at most 24 bridge-
   slots; to stress all 32 as targets, raise to ~64–96 cross-facts so each bridge is a target a few times).
3. **Memory management for 32 co-resident bridges.** §0.4 shows 32 co-resident graded bridges need ~9.6 GB (fits 24
   GB). The runner's Gate X builds ALL bridges co-resident (`gate_X_vtag`, `cortex_conversation_ensemble_derisk.py:398–404`
   loops over `all_corpora` building a `GradedBridge` each, all held in `graded_bridges`) — which at 32 is ~9.6 GB +
   per-op overhead. **This is feasible (§0.4, §2.4) but must be VERIFIED**, with the sequential-build-and-evict
   mitigation (§2.4) coded as a fallback BEFORE the run, so an OOM does not waste the multi-hour GPU spend.

There is NO new mechanism and NO `sim/` edit — only (1) 32 shard names, (2) more cross-facts, (3) the memory-
verification + the eviction fallback. The de-risk is the existing ensemble runner at `--n-bridges 32`.

### 2.3 The exact command + config the controller runs

**Primary (32-bridge fan-out, the build-gate run), multi-seed 42/43/44:**

```bash
SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_ensemble_derisk \
    --mode cross --seeds 42,43,44 --cortex learned \
    --n-bridges 32 --concepts-per-bridge 64 \
    --n-cross-facts 96 \
    --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
    --coresidence-strategy keep \
    --out research/findings/raw/_phase1_fanout32_cross.json
```

This runs Gate X (cross-bridge composition — X-vtag spiking + X-conv numpy), the Cx anti-cheat (FIXED M7), and the C3
moat at 32 bridges. `--mode cross` (not `full`) is the right scope for the FAN-OUT de-risk specifically: Gate B
(within-bridge generalization) and Gate A (the matrix) are fan-out-INDEPENDENT (they are per-bridge / within a bridge —
Gate B reads one bridge's codes; the matrix's binding capacity is the composer-architecture question, not the fan-out
question), so the fan-out de-risk's load-bearing measurement is Gate X + the moat. (The FULL matrix-at-32-bridges run
is the production gate confirmation, §3.3.)

**TWO required additions to the runner before the run (small, runner-side, no `sim/` edit):**

- **`--coresidence-strategy {keep,evict}`** (default `keep`) — `keep` builds all 32 bridges co-resident (the §0.4
  feasible path); `evict` is the §2.4 sequential mitigation (build → encode the bridge's tags → free the recurrent,
  keeping only the tag indices) used if `keep` OOMs. Coding `evict` BEFORE the run is the cheap insurance.
- **`SHARD_NAMES` extended to 32** (or a `--shard-names-from g20_vocab_spec_2048` hook) so `--n-bridges 32` has 32
  cluster names.

**Tiny CPU plumbing smoke first** (proves it RUNS end-to-end at 32 bridges with tiny pools, NOT the science; the
runner's existing `--smoke` path, `cortex_conversation_ensemble_derisk.py:740–753`, but with `--n-bridges 32`
overridden — note the smoke caps `n_bridges` to 3, so the smoke for THIS de-risk must lift that cap or use a dedicated
tiny-32 config):

```bash
SIM_BACKEND=numpy python -u -m research.runners.cortex_conversation_ensemble_derisk \
    --mode cross --seeds 42 --cortex synthetic --skip-vtag \
    --n-bridges 32 --concepts-per-bridge 8 --target-per-sub 4 \
    --n-pool 200 --pattern-size 30 --cycles 2 --n-cross-facts 32 \
    --out research/findings/raw/_phase1_fanout32_smoke.json
```

(This exercises the 32-bridge ROUTING + cross-fact authoring + X-conv + the moat on CPU in seconds, before the GPU run
— it verifies the 32-shard plumbing without the multi-hour spiking cost. `--skip-vtag` because the live V-tag layer
needs GPU bridges.)

### 2.4 Does 32 co-resident fit 24 GB? — the answer + the mitigation

**Yes — 32 co-resident graded bridges need ~9.6 GB of synapse memory (§0.4), within the 24 GB GPU with ~14 GB of
headroom, and the pinned-pool wall that killed V=640 single-pool does NOT apply** (32 small sequential installs, not
one 354M-synapse transfer). So `--coresidence-strategy keep` is the expected path.

**The mitigation, coded in reserve (`--coresidence-strategy evict`):** if the live run OOMs — because the per-op RF
bridges, the CuPy memory pool's fragmentation, or the engram-tag drive arrays push past 24 GB during the V-tag
recall — the cross-bridge V-tag layer does NOT actually need all 32 LEARNED RECURRENTS resident at recall time. The
V-tag layer stores a fact as engram-tag NEURON INDICES (`commit_engram_tag(top_k=150, region_filter=["pool"])`,
`multibridge_graded_derisk.py:503`), and recall stimulates those indices and reads firing. So the sequential strategy
is: (i) build a bridge, run its co-occurrence learn, encode every cross-fact tag that touches it, then (ii) **free the
bridge's learned recurrent** (keeping a thin bridge with only the tag indices + the sparse patterns) before building
the next. At recall, a cross-fact needs only the CUE bridge and the TARGET bridge live simultaneously (a pair), not all
32 — so recall pages in pairs. This trades wall-clock (rebuild/repage) for memory and is the documented fallback. **It
is coded before the run so an OOM is a one-flag pivot, not a re-architecture.** Given §0.4's 9.6 GB ≪ 24 GB, `evict`
is insurance, not the plan.

### 2.5 What the de-risk measures + the GO / BOUNDARY / NEGATIVE criteria

Reusing the ensemble runner's Gate X + Cx + C3 (the FIXED `cross_bridge_eval` + `moat_eval`,
`cortex_conversation_ensemble_derisk.py:503–536`), multi-seed 42/43/44, at 32 bridges:

- **Gate X (cross-bridge composition).** X-vtag M3: target top-2 over the target bridge's concepts with signal/floor ≥
  1.5 (the band `cortex_conversation_ensemble_derisk.py:409–414`). X-conv: who/what exact identity recall ≥ 0.7
  (`--x-bar`, `:732`) with zero abstention breaches. **The fan-out-specific question: does the 3-bridge signal/floor of
  17–24× hold (or degrade gracefully but stay ≥ 1.5) at 32 bridges?**
- **Cx (the FIXED M7 anti-cheat).** A cue scored against a RANDOM WRONG target in the same target bridge must rank
  ~median → Gate X collapses under permutation (`cross_bridge_eval(..., permuted=True)`,
  `multibridge_graded_derisk.py:551–564`). Mandatory — a non-collapsing Cx means the recall is an artifact, not a
  specific link.
- **C3 (the moat alongside the host).** Over the 96 cross-facts at 32-bridge fan-out: agreement with the host, **zero
  host-abstain/gate-accept breaches, zero abstention-floor false-accepts**, lesion collapses (`moat_eval`,
  `multibridge_graded_derisk.py:738–748`). **Non-negotiable — any breach is FATAL.**
- **COST logging (new):** peak GPU memory (to confirm the §0.4 9.6 GB estimate or trigger `evict`), wall-clock for the
  32-bridge build + the V-tag encode/recall, per-bridge build time.

**Decision criteria:**

- **GO (the expected outcome):** Gate X GO (X-vtag M3 top-2 ≥ 0.80 + signal/floor ≥ 1.5; X-conv ≥ 0.7, zero abstention
  breach), Cx collapses, C3 moat intact (zero breaches + lesion collapses), all multi-seed — AND peak memory fits 24 GB
  (or `evict` made it fit). **⇒ the cross-bridge layer + moat hold at 32-bridge fan-out → proceed to the production
  train + gate.**
- **BOUNDARY:** Gate X recall real but in the 0.50–0.80 top-2 band (signal/floor degraded by fan-out but still > floor)
  with Cx + C3 clean. A real-but-weakened cross-bridge link at 32 bridges — characterize it (which bridges' targets
  fail; is it a routing-noise issue fixable by higher top_k or stronger drive?) before committing the full train. A
  BOUNDARY is a legitimate stopping point to tune the V-tag drive/top_k, not a NEGATIVE.
- **NEGATIVE (blocks the build, is the deliverable):** ANY moat breach (C3 host-abstain/gate-accept > 0 or floor
  false-accept > 0, or X-conv abstention breach) — FATAL; OR Gate X top-2 below 0.50 (cross-bridge recall fails at
  fan-out); OR Cx fails to collapse (the recall is a fan-out artifact); OR the run OOMs even under `evict` (32 bridges
  infeasible on this GPU). A NEGATIVE here maps the cross-bridge / moat scaling boundary BEFORE the multi-week train —
  the scientific deliverable (it would say "the cross-bridge layer scales to 8 but not 32 bridges" or "the moat's
  false-accept surface breaks at 2,048-concept fan-out," reshaping Phase 1).

### 2.6 Anti-cheats (the GO is void without them)

Carried VERBATIM from the validated ensemble runner (the same controls the 3- and 8-bridge GOs used):

- **Cx** — the FIXED M7 cross-bridge permuted control (score a WRONG target → must rank ~median → Gate X collapses).
  This is the corrected control (`multibridge_graded_derisk.py:551–564`); the original M7 was mis-implemented (it
  stored AND scored the permuted mapping, so it could not collapse by construction —
  `multibridge_graded_derisk-GO.md` §"The anti-cheat fix"). The de-risk uses the FIXED version.
- **C3** — the moat validated ALONGSIDE the host (the gate may ACCEPT only where the host accepts; the host-abstain/
  gate-accept cell must be 0; lesion collapses the separation).
- **The abstention surface at fan-out (the new emphasis for 32 bridges):** because an unknown cross-cue could now match
  any of 2,048 concepts, the C3 moat-floor (`--moat-floor`, default 20, `:729`) should be raised so the abstention test
  covers a representative sample of the larger candidate surface (e.g. `--moat-floor 64`), confirming zero false-
  accepts against the bigger confusion set.

---

## 3. The production TRAIN + GATE plan (2,048 concepts, multi-seed)

### 3.1 Train the 32 cortex bridges (the homeostatic Oja recipe)

**The recipe (per bridge, already validated):** the homeostatic recurrent learn — `learn_W_homeostatic` /
`HomeostaticAssocGraph` with **Oja's incoming-L2 set-point** (build plan piece i: "⭐ DEFAULT TO OJA … transfers
GRACEFULLY across scales"; `multibridge_graded_derisk.py:217–219`, `--homeo oja --homeo-target 40`), over the bridge's
within-cluster co-occurrence corpus, then the brain-based divnorm read-out for the graded codes
(`divnorm_spreading_readout`, `multibridge_graded_derisk.py:221–225`). Per bridge: ~3.46M synapses, n_pool=2400. This
is EXACTLY the per-bridge recipe the 3- and 8-bridge de-risks ran — 32 is the same recipe 32 times.

**Set-point calibration:** Oja's set-point transfers across scales (build plan piece i: t=15@pool1000 → 40@7000 →
50@12000), and at n_pool=2400 the de-risk used t=40. The production train re-brackets the set-point on the FIRST bridge
(the runner's `--bracket-setpoint` over {20, 40, 80}, `multibridge_graded_derisk.py:979–981`) and reuses the winner for
the rest — a cheap per-scale one-line calibration, not 32 separate tunings.

**Incremental / resumable training applies.** The single-pool sparse-distributed trainer supports
`--resume-from <checkpoint>` (`concept_pool_sparse_distributed.py:381, 445–448`: "loaded {resume}; accumulating
+{n_train_events} events/concept on top"), so a 32-bridge train that must chunk across breaks can checkpoint and
resume. **Critically, the 32 bridges are INDEPENDENT** — each bridge's learn is a standalone job, so the natural
incrementality is "train bridge k, save it, train bridge k+1" (embarrassingly parallel across bridges; resumable at the
bridge granularity), and `--resume-from` covers the within-bridge event accumulation. A bridge that fails its gate is
re-trained alone without touching the other 31.

### 3.2 Wall-clock estimate (and the honest cost flag)

From §0.4's analytic bracket (anchored on the 3-bridge full run = 7.3 h total,
`2026-06-12-cortex-conversation-3bridge-ensemble-GO.md` line 17):

| Scenario | Per-bridge | 32 bridges × 1 seed | 32 bridges × 3 seeds |
|---|---|---|---|
| **A. de-risk-style (learn + full gate suite per bridge)** | ~2.4 h | **~78 h (~3.2 days)** | **~234 h (~9.7 days)** |
| **B. production (learn-only per bridge; gates run ONCE at the end, §3.3)** | ~0.5–1.0 h | **~16–32 h (~0.7–1.3 days)** | **~48–96 h (~2–4 days)** |

**The honest cost flag:** the production train is **scenario B** — the per-bridge LEARN is the recurring cost (gates
run once on the assembled ensemble, §3.3, not re-run per bridge). At ~0.5–1.0 h/bridge that is **~16–32 h for a single
seed** and **~2–4 days for the 3-seed re-confirm** of the per-bridge LEARN alone. **The dominant cost is the 32× fan-
out, not any single bridge** — and it is GPU-bound, sequential by default (one GPU). This matches the build plan's
"~2–4 week" envelope for the whole of piece (iii) (build plan §"Cost", line 107–112: "the multi-seed re-confirm … is
~1 day of GPU" for the learn + "integration iteration + multi-seed conversational-matrix compute" for the rest). The
build plan's own caveat applies: "My implementation-time estimates run ~2–3× high (treat as a ceiling); compute
estimates are reliable" (build plan line 110). **Treat ~2–4 days of GPU for the multi-seed per-bridge learn + the
matrix/gate runs as the reliable compute cost, inside the ~2–4 week integration window.**

**Branch on the composer route (the only route-specific cost):** under **route A** the per-bridge composers are 32
small D≈256 numpy objects built from each bridge's codebook (vocabulary-independent cost; composer-architecture design
§4.2) — negligible vs the spiking learn. Under **route B** the train additionally builds ONE D≈5.5k union composer
(~90 MB codebook + 11–22k-neuron per-op RF bridges, composer-architecture design §4.1) — a per-op cost the route-B
gate run pays, and the FHRR-SNR-at-2,048 risk that design flagged. This train plan's wall-clock above is the SPIKING
LEARN cost (route-independent); the composer cost rides on whichever route the prior de-risk picks.

### 3.3 Re-confirm the gates at 2,048-concept production scale (multi-seed)

After the 32 bridges are trained, run the FULL gate suite ONCE on the assembled ensemble (not per bridge), multi-seed
42/43/44 — the build plan's acceptance matrix (build plan pieces ii–iii, lines 45–55) at production scale:

1. **G1–G4 per bridge (structure + generalization):** Pearson(sim, S_true) ≥ 0.7 + the within>between margin
   (`graded`); held-out-neighbour generalization ≥ 0.7 (chance 0.25); cortex-channel round-trip; spiking strong-encode
   repro 1.0 + decorrelation. These are per-bridge reads over the trained bridges (cheap, numpy + a little GPU) — the
   per-bridge science the de-risk already validated, re-asserted at production scale. **A bridge failing G1/G2 is a
   mis-curated cluster (§1.3) — re-curate + re-train that bridge alone.**
2. **The full conversational matrix at 32-bridge fan-out (Gate A):** who/what Q&A, abstention, negation/yes-no, one/two-
   attribute, clauses, dialogue — the `cortex_conversation_ensemble_derisk.py --mode full --n-bridges 32` run, must NOT
   regress (≥ 5/6 cells + zero abstention breach, `:599–602`). NOTE: the matrix's BINDING CAPACITY is the composer-
   architecture question (route A: within-bridge clause; route B: D≈5.5k union); this gate inherits whatever the
   composer de-risk decided. The fan-out-specific part is that the matrix's SVO roles + cross-bridge facts span 32
   bridges.
3. **Within-bridge generalization-in-conversation (Gate B), every bridge, co-resident:** B1 ≥ 0.7 (≈4× chance) per
   bridge + B2 moat zero false-accepts. The genuinely-NEW capability, at production scale, every bridge.
4. **Cross-bridge composition + the moat at 32-bridge fan-out (Gate X + Cx + C3):** the §2 fan-out de-risk's gates —
   re-run on the FULLY-TRAINED production bridges (the §2 de-risk runs them on the de-risk's synthetic-cluster stand-in
   bridges; this re-runs them on the real 2,048-concept ensemble).
5. **All anti-cheats collapse (C1 permuted-similarity, C4 random-shard, Cx permuted-cross-bridge, C3 moat alongside
   host):** every one mandatory, multi-seed.

**The production GO:** the full matrix does not regress + gains graded semantic generalization (Gate B) with the no-
confab moat intact (zero abstention-floor false-accepts) + cross-bridge composition holds at 32-bridge fan-out + all
anti-cheats collapse, multi-seed 42/43/44 (build plan piece iii gate, line 55). This is the "step 3 true cortex"
delivered at the production 2,048-concept scale.

---

## 4. Honest risks + the cheap-first ordering (which de-risk gates which build step)

### 4.1 The cheap-first ordering (each gate de-risks the NEXT spend)

The ordering is strictly cheapest-first; each step is a gate for the next, so an honest NEGATIVE stops the spend early:

| # | Step | Cost | Gates | An honest NEGATIVE here means |
|---|---|---|---|---|
| 0 | Composer-architecture A-vs-B de-risk (PRIOR design, in flight) | hours | the composer object for steps 3–4 | re-architect the composer before sharding-scale spend |
| 1 | **Author `g20_vocab_spec_2048.py`** (32 semantic clusters × 64) | hand-labor, CPU, no GPU | the sharding for everything downstream | the 2,048-concept taxonomy can't be coherently clustered into 32 — re-scope the vocab or cluster count |
| 2 | **32-bridge fan-out de-risk (§2)** — `--mode cross --n-bridges 32`, on the de-risk's synthetic-cluster stand-in bridges, multi-seed | hours–~1 day GPU | the cross-bridge layer + moat AT 32 bridges, before the full train | the cross-bridge layer or moat breaks at 32-bridge fan-out (or OOMs even under `evict`) — the cross-bridge mechanism scales to 8 but not 32; reshapes Phase 1 BEFORE the multi-day train |
| 3 | **Production train (§3.1)** — 32 real bridges, Oja recipe, 1 seed first | ~16–32 h GPU (1 seed) | the per-bridge learn at production scale on the REAL clusters | a real cluster fails G1/G2 — a mis-curated cluster (re-curate that bridge); OR the brain-based learn is too weak at production scale (the build plan's risk 2, line 73 — an honest BOUNDARY) |
| 4 | **Production gate re-confirm (§3.3)** — full matrix + Gate B + Gate X + moat + anti-cheats, multi-seed | ~2–4 days GPU (3 seeds) | the "step 3 true cortex" capability at 2,048 | the integrated system regresses, or generalization/cross-bridge/moat fails at production scale — the honest characterization of the production boundary |

**Why this ordering:** step 2 (the 32-bridge fan-out de-risk on cheap synthetic-cluster stand-in bridges) is the
load-bearing cheap-first gate — it answers "does the cross-bridge layer + moat survive 4× the validated fan-out" for
HOURS of GPU, BEFORE the multi-day production train (step 3) commits to building 32 real bridges. This is the project's
standing pattern (every decisive pivot — the whitening reframe, the missing-accumulator fix, the multibridge mechanism
de-risk — came from a cheap proof before the expensive build; `2026-06-12-multibridge-graded-derisk-GO.md` §"Why this
ran"). The §2 de-risk's stand-in is the de-risk's synthetic semantic clusters (`build_bridge_corpus` over 32
`SHARD_NAMES`); the production gate (step 4) re-runs the same gates on the real clusters — so the fan-out risk is
retired cheaply before the real-cluster spend, then confirmed on the real clusters.

### 4.2 Honest risks

- **RISK 1 — the curation effort + quality (piece 1).** 2,048 words into 32 coherent clusters of 64 is hand-labor, and
  a mis-curated cluster (a member that doesn't belong, or two clusters that should be one) silently weakens that
  bridge's within-bridge generalization. **Mitigation:** the per-bridge G1/G2 gate (§3.3) CATCHES a mis-curated cluster
  (it fails the structure/generalization threshold); the global-uniqueness assert (`g20_vocab_spec_320.py:150–153`)
  catches duplicates at import. The cost of being wrong is bounded (re-curate + re-train ONE of 32 independent
  bridges). **Not a blocker, but the largest hand-labor item.**
- **RISK 2 — the 32-bridge fan-out SNR (piece 2, THE scaling risk).** Cross-bridge recall + the moat are validated to
  8 bridges; 32 is 4× (the build plan's stated 6.4× vs the 5-bridge retrieval tier). If the V-tag recall's signal/floor
  degrades below the 1.5× band at 32-bridge fan-out, or the moat's false-accept surface breaks against 2,048 candidates,
  Gate X / C3 fail. **Mitigation:** the §2 de-risk MEASURES this at 32 bridges on cheap stand-in bridges BEFORE the
  train (the cheapest thing that answers it); a BOUNDARY is tunable (higher top_k / stronger V-tag drive / larger moat
  floor), a NEGATIVE is the deliverable. **This is the single deepest risk (§5).**
- **RISK 3 — memory at 32 co-resident (piece 2).** §0.4 shows 9.6 GB fits 24 GB, and the pinned-pool wall does not
  apply — but the live run also allocates per-op RF bridges + the CuPy pool + tag-drive arrays + fragmentation. If the
  realized peak exceeds 24 GB, `keep` OOMs. **Mitigation:** the `evict` sequential-pair strategy (§2.4) is coded BEFORE
  the run as a one-flag pivot; given the 14 GB headroom it is insurance, not the plan.
- **RISK 4 — the brain-based-learn strength gap at production scale (piece 3).** The build plan's standing risk (line
  73): the spiking-Hebbian learn is weaker than backprop, so graded structure could be coarser at 2,048 than at toy
  scale. **Mitigation:** the learn already reaches the host ceiling at V=320 single-pool (build plan piece ii GO,
  Pearson +0.99); per-bridge at 64 concepts is FAR below that, so the per-bridge learn is on solid validated ground —
  the gap, if any, would show as a per-bridge G1/G2 BOUNDARY (caught + characterized, an honest deliverable), not a
  silent failure.
- **RISK 5 — the moat surface multiplies with both fan-out AND (under route A) per-bridge composers.** At 32 bridges
  the abstention surface is 2,048 candidates; under route A there are also 32 per-bridge composer moats (composer-
  architecture design risk 3). **Mitigation:** C3 (the moat alongside the host) is re-asserted at the 32-bridge surface
  with a raised floor (§2.6), and the production gate (§3.3) re-asserts zero breaches on the integrated system — the
  moat is the non-negotiable gate (build plan line 74).
- **RISK 6 — the composer route is not yet decided (the §0.3 dependency).** This design assumes route A; if the prior
  de-risk returns route B / hybrid, §3.2's per-bridge-composer cost changes (and route B inherits the FHRR-SNR-at-2,048
  risk). **Mitigation:** pieces (1) and (2) are route-independent (§0.3); only §3.2's composer cost branches, flagged
  inline. **Not a blocker — the routes share the cortex sharding + the V-tag fan-out.**

---

## 5. Summary (the things the controller needs)

**Recommended sharding source + method (3–5 sentences).** Author a new frozen `research/runners/g20_vocab_spec_2048.py`
(mirroring `g20_vocab_spec_320.py`'s additive structure + global-uniqueness assert) that defines **32 hand-curated
SEMANTIC super-clusters of 64 mutually-similar concepts each** — RE-CLUSTERING the high-quality, duplicate-checked g20
word lists (a vetted word source) plus ~1,728 new curated words, assigned by TAXONOMY not part of speech. This is
load-bearing because the existing g20-320 spec shards by **part of speech** (`g20_vocab_spec.py:113–119`), which does
NOT guarantee within-bridge similarity, and the C4/M6 anti-cheat PROVES random/incoherent sharding collapses within-
bridge generalization (`cortex_conversation_capability_GO.md` line 24) — so semantic co-location is a CORRECTNESS
requirement. Within-bridge similarity is guaranteed by curation (taxonomic coherence) + a per-bridge corpus that
induces within-cluster co-occurrence, and CONFIRMED per bridge by the G1/G2 structure-recovery + generalization gates
(a failing bridge = a mis-curated cluster, caught + re-curated, never silent). Co-occurrence-graph re-clustering
(option b) is deferred to a second-pass refinement because it is circular for the initial sharding (the codes are
learned FROM the corpus being sharded).

**The exact 32-bridge fan-out de-risk command + whether 32 co-resident fit 24 GB.** Extend the existing
`cortex_conversation_ensemble_derisk.py` (it already parametrizes `--n-bridges`) with three small runner-side changes
— 32 shard names (`SHARD_NAMES`/a `g20_vocab_spec_2048` hook), `--n-cross-facts 96`, and a `--coresidence-strategy
{keep,evict}` flag — then run multi-seed:
```bash
SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_ensemble_derisk \
  --mode cross --seeds 42,43,44 --cortex learned --n-bridges 32 --concepts-per-bridge 64 \
  --n-cross-facts 96 --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
  --coresidence-strategy keep --out research/findings/raw/_phase1_fanout32_cross.json
```
**Yes — 32 co-resident graded bridges fit:** ~9.6 GB of synapse memory (32 × 3.46M syn × 86.9 B/syn, §0.4), within the
24 GB GPU with ~14 GB of headroom, and the pinned-memory-pool wall that caused the V=640 single-pool OOM does NOT apply
(the 110.7M synapses install as 32 small ~3.46M sequential installs, not one 354M transfer). The `--coresidence-strategy
evict` sequential-pair mitigation (build → encode the bridge's cross-tags → free its recurrent, keeping only tag indices;
recall pages in cue/target pairs) is coded BEFORE the run as a one-flag pivot in case the live per-op + CuPy-pool
overhead pushes past 24 GB. GO = Gate X (cross-bridge) holds at 32-bridge fan-out (V-tag top-2 ≥ 0.80, signal/floor ≥
1.5; X-conv ≥ 0.7) with Cx collapsing + C3 moat zero-breach, multi-seed.

**The production-train wall-clock estimate.** The per-bridge homeostatic Oja LEARN at production scale is ~0.5–1.0 h/
bridge (bracketed from the 3-bridge full run = 7.3 h, §0.4), so the production train of all 32 bridges is **~16–32 h
GPU for a single seed and ~2–4 days for the 3-seed re-confirm of the per-bridge learn** (gates run ONCE on the assembled
ensemble, not per bridge). The dominant cost is the 32× fan-out, GPU-bound and sequential; the 32 bridges are
INDEPENDENT (embarrassingly parallel, resumable at bridge granularity), and the single-pool `concept_pool_sparse_distributed
--resume-from` covers within-bridge event accumulation across breaks. This sits inside the build plan's reliable ~2–4
day compute envelope within its ~2–4 week integration window (the build plan notes its implementation-time estimates
run ~2–3× high but compute estimates are reliable).

**The single deepest risk.** The **32-bridge fan-out SNR for cross-bridge composition + the no-confab moat** — they are
validated only to 8-bridge fan-out, and 32 is 4× that, with the moat's abstention surface growing to 2,048 candidate
concepts. If the V-tag recall's signal/floor degrades below the band or the moat's false-accept surface breaks at fan-
out, Gate X / C3 fail. This is exactly why the cheap-first ordering runs the §2 fan-out de-risk (hours of GPU on cheap
synthetic-cluster stand-in bridges) BEFORE the multi-day production train — turning the deepest risk into a measurement
that gates the spend, with an honest NEGATIVE ("the cross-bridge layer + moat scale to 8 but not 32 bridges") being the
scientific deliverable that reshapes Phase 1 before the build commits.
