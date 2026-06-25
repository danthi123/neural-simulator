# Foundational training curriculum — scaling the stream cortex to base worldly knowledge: read-only scoping

**Date:** 2026-06-25
**Type:** RESEARCH-GATE scoping (read-only; NO edits/runs/webapp). Standing-practice deep-research + catalog review BEFORE committing build/GPU.
**Trigger:** NEW STRATEGIC DIRECTION (owner): give the sim brain a FOUNDATIONAL TRAINING CURRICULUM — an LLM-pretraining-style corpus (smaller than GPT-scale) to seed BASE WORLDLY KNOWLEDGE, instead of only slow incremental conversation-stream growth. The fix for "LLM-feel richness is gated by the brain's knowledge".
**Framing (load-bearing):** the corpus = the **linguistic ENVIRONMENT** (the legit environment boundary, like the sensory world). The BRAIN learns from it via its OWN stream cortex (online Hebbian / PPMI co-occurrence). The LLM stays fluency-only — NEVER does the cognition. Biologically this is **developmental exposure** (cf. BabyLM). The real question is **SCALE**.

---

## TL;DR verdict

**FEASIBLE, local-first, with one genuine scaling boundary that is engineering-shaped not substrate-shaped.** The single most important technical fact, verified in the code: the stream cortex's learned co-occurrence object is **`M[Nt, n_hub]`** — **LINEAR in vocabulary `Nt`, NOT quadratic `V×V`** — because the context dimension `n_hub` is a small FIXED pool of the most-frequent words (~200–500), exactly the biology-faithful "context neurons" choice (CYCLE 94). So the feared `V²` co-occurrence-matrix blow-up **does not exist** in this architecture. The learning compute, the matrix memory, and the read-out normalization are all `O(V·n_hub)` = linear.

The real bottlenecks are two, and both are characterized:
1. **Per-substrate discrimination capacity** — sparse-distributed concept codes hold **64 @ 100%, 128 @ 84%, 320 @ 98.4% per bridge**; the production scaling route is the **multi-bridge ensemble (linear in bridge count)**. A 2K–5K foundational vocab = ~7–16 bridges; ~30K = ~94 bridges. VRAM tiling, not a wall.
2. **Wall-clock to stream a real corpus** — streaming ~150K windows learns 320 concepts in ~84 min on a 3090. A foundational corpus is more windows; this is the develop-loop's job (run over simulated days/weeks), the owner already accepts long local runs with an ETA. NOT a VRAM wall.

The recommended path is **staged**: a cheap-first de-risk that streams a REAL small corpus (TinyStories-class, ~1.5K core words) into the stream cortex to ~1–2K concepts on a 2–3-bridge ensemble, with the no-confab moat + generalization held — measuring VRAM + wall-clock — BEFORE committing to a full BabyLM-10M foundational corpus.

---

## 1. THE CORPUS — ranked options for "base worldly knowledge" at a tractable scale

The decisive sizing context from the literature (the BabyLM challenge made this canonical): **children hear 2M–7M words/year; by age 12 that's 24M–84M words → rounded to "100M words" as the developmentally-plausible cap; "10M words" ≈ the first 2–5 years.** This is the right yardstick: we want *developmental* exposure, not GPT-scale (trillions). A "base worldly knowledge" corpus is **10M–100M words**, with a **core content vocabulary of a few thousand word-types** (the long tail is rare and low-information for co-occurrence learning).

Ranked options (best-fit first):

| # | Corpus | Word count | Core / full vocab | What "base worldly knowledge" it buys | Fit |
|---|---|---|---|---|---|
| **1** | **TinyStories** (already the wired corpus) | ~476M tokens | **~1,500 core content words**; ~10K–29K full token vocab | Concrete everyday world: people/animals/objects/actions/places/feelings, simple causality, narrative structure. The exact 3–4-yr-old experiential domain — clean, no rare-word noise. | **BEST cheap-first** — already the `_load_token_stream()` target; controlled vocab = clean co-occurrence; matches the validated 320-concept taxonomy's grain. |
| **2** | **BabyLM Strict-Small (10M words)** | 10M words | ~5K–15K content types | Broader world: CHILDES child-directed speech + children's books + simple Wikipedia + subtitles + dialogue. Real (not LLM-generated) developmental input. THE canonical developmentally-plausible target. | **BEST foundational target** — the principled "base worldly knowledge" scale; multi-source so the brain hears facts, dialogue, and narration. |
| 3 | **Simple English Wikipedia** | ~10s of M words (full dump) | ~5K–20K controlled-vocab types | Encyclopedic world facts in deliberately simple language: what things ARE, categories, properties — the "worldly knowledge" the owner names most directly. | Strong **fact-density** complement to TinyStories' narrative; pairs well as a second source in a curriculum. |
| 4 | **CHILDES (child-directed speech)** | ~29M words (English subset); ~6.5M words child-directed 0–60mo | ~3K–8K types | The most biology-faithful input (literally what caregivers say to children). Concrete, repetitive, high-frequency core. | Use as the **earliest-developmental** slice (week-1 of the develop loop) — maximally faithful, smallest vocab. |
| 5 | **BabyLM Strict (100M words)** | 100M words | ~15K–30K types | Full age-12 developmental exposure: approaches an adult's ~27K word-family vocabulary. | The **full foundational** ceiling — only after the 10M tier is GO; ~10× the wall-clock. |

**Sizing the brain's target vocabulary.** Adult native speakers know ~20K–35K **word families** (mean ~27K). But for *base worldly knowledge as a baseline for interaction*, the high-value target is the **core 2K–8K content words** (these carry almost all the world-knowledge co-occurrence signal; the rare tail is Zipf-thin). So the staged vocab targets are: **320 (validated) → ~1–2K (next step, real corpus) → ~5–8K (foundational core) → ~15–30K (full BabyLM, far tier)**.

**Recommendation:** **TinyStories for the cheap-first de-risk** (it's already wired, controlled vocab, clean), then **BabyLM-10M (Strict-Small) as the foundational target** (the principled developmental scale, real multi-source input), with **Simple Wikipedia layered in as the fact-density source**. The full BabyLM-100M is a far tier gated on the 10M GO.

---

## 2. THE SCALE-FEASIBILITY ANALYSIS (the crux) — does the stream cortex + sparse-distributed substrate scale 320 → 5K–30K?

### 2a. The decisive fact: the co-occurrence object is LINEAR in vocab, not V²

The owner's question frames a feared `V×V` co-occurrence matrix. **In this codebase that object does not exist.** Verified in `research/runners/_longitudinal_develop_loop_gpu.py` (the `StreamCortex` class) and `_phaseB_online_stream_cortex_derisk.py`:

- The learned co-occurrence is **`M[Nt, n_hub]`** (`_longitudinal_develop_loop_gpu.py:191`, `:254-256`): `Nt` = number of target concepts (the growing vocab), `n_hub` = a **FIXED** pool of context "hub" neurons (default `n_hub=200`, the online-derisk used 300–500). The hubs are the *most-frequent context words*, chosen once by global frequency — explicitly noted as the biology-faithful "context neurons" choice (`2026-06-15-...-online-stream-cortex...:41`: "a cortex *does* have context neurons for frequent words").
- So the matrix is **`Nt × n_hub` = O(V · 300)** — **linear in V**. A 30K-vocab matrix is `30000 × 300 ≈ 9M` float entries ≈ **72 MB** dense float64 (≈36 MB float32). Trivial. There is **no `V²` term anywhere**.
- The PPMI / normalization is `double_center(log1p(M·100))` (`:257`) — a per-row/per-column mean subtraction = **`O(V·n_hub)`** = linear. The on-brain version (per-hub spike-frequency adaptation + per-concept feedforward inhibition, `2026-06-16-harden-320-...:58-72`) is also `O(V·n_hub)`. **The normalization is not a bottleneck.**

**⇒ The co-occurrence learning + normalization scales LINEARLY in vocab.** This is the single most important feasibility result: the architecture sidesteps the classic distributional-semantics `V²` wall by construction (fixed hub dimension), and it was validated as biology-faithful (online Hebbian co-occurrence ≈ batch PPMI, `+0.513` vs `+0.502`, 3 seeds).

### 2b. The genuine bottleneck #1 — sparse-distributed per-substrate DISCRIMINATION capacity

This is where the real ceiling lives. The downstream representation that the conversation pipeline reads is the **sparse-distributed K-of-N code** (`concept_pool_sparse_distributed.py`, `generate_sparse_patterns(n_concepts, n_pool, pattern_size, seed)` — each concept = `K=pattern_size` random neuron indices in an `N=n_pool` shared pool). Measured capacity (`2026-05-15-sparse-distributed-capacity-curve.md`):

| concepts / bridge | pool N | pattern K | top-1 discrimination | pool+FS neurons | ~VRAM/bridge |
|---|---|---|---|---|---|
| **64** | 2000 | 100 | **100.0%** | 2300 | ~1.6 GB |
| 128 | 3000 | 100 | 84.4% | 3300 | ~2.1 GB |
| 256 | 5000 | 100 | (training-bound; ~) | 5300 | ~3.2 GB |
| **320** (5×64 multi-bridge) | 2000/bridge | 100 | **98.4%/bridge** | 11,500 total | ~8 GB total |

**The production scaling route is multi-bridge, and it is LINEAR in bridge count** (each bridge holds an independent disjoint set of concepts; `g20_multibridge.py --sparse`; validated 160 (5×32 @ 100%) and 320 (5×64 @ 98.4%) end-to-end). Keep each bridge ≤320 concepts (where discrimination is high) and add bridges:

| foundational vocab | bridges (@320/bridge) | total pool+FS neurons | rough VRAM (untiered) |
|---|---|---|---|
| 1,280 | 4 | ~46K | ~13 GB |
| 2,048 | ~7 | ~80K | ~22 GB (near the 24 GB line) |
| 5,000 | ~16 | ~184K | ~52 GB (NEEDS synapse tiering) |
| 30,000 | ~94 | ~1.1M | ~300 GB (heavy tiering) |

**VRAM crossover:** a single 3090 (24 GB) comfortably holds **~2K concepts (~7 bridges)** resident at once. Beyond that, the project's existing **synapse tiering** (`sim/synapse_storage.py`, `TieredSynapseStore`, idle/pressure eviction) is exactly the mechanism — bridges are independent so only the queried bridge needs to be resident. **This is the only place a >24 GB number appears, and it is solved by tiering (page bridges in/out), NOT by needing a bigger GPU**, because the bridges never interact during learning (each learns its own disjoint concepts). So even 5K–30K stays local-feasible with tiering; cloud is not required for VRAM.

**Honest caveat (the 2026-05-22 NEGATIVE):** the activity-grounded composition pipeline read **near-silent** activity off a *freshly-built, UNtrained* sparse bridge (0.015% mean activity vs the 7.5% the pipeline needs). Diagnosis (`2026-05-22-...-too-sparse...:46-63`): a **setup gap (untrained substrate), not a fundamental limit** — a *trained* sparse bridge discriminates at 100%, and the two fixes (capture from a trained bridge, or ground the symbol in the clean K-of-N pattern itself) are both available. **Relevance to scaling: the de-risk MUST ground codes from the trained/stream-learned substrate, not a fresh one** (the stream cortex already does this — it reads `M` from the learned synapses, `:253`).

### 2c. The genuine bottleneck #2 — composition DEPTH at large per-codebook vocab (NOT a learning bottleneck)

A separate, already-mapped ceiling: at a *single* codebook, **multi-step composition** degrades as distractors grow (`2026-06-04-vocab-scaling-capacity-curve-refines-cost-model.md`): flat/1-attr/who/abstain hold to ≥4× vocab at fixed D, but **two-attribute (F=3 resonator, needs `D ∝ M²`) and recursive clauses degrade at 640+ concepts**. This is **why the multi-bridge route exists** — keep per-bridge ≤320 (full capability) and scale by adding bridges. So: **the foundational-knowledge KNOWLEDGE-base scales free (linear); the deep COMPOSITION over thousands of distractors in one codebook does not — but the production design already routes around it (per-bridge ≤320 + GPU resonator for two-attribute + sparse block codes as the far lever).** For "base worldly knowledge" (recall, who/what, abstain, single-attribute, multi-hop chains), this bottleneck does **not** bind.

### 2d. Wall-clock estimates (local 3090 vs the >24 GB cloud line)

- **Stream-learning rate:** ~150K windows → 320 concepts ≈ **84 min on a 3090** (`2026-06-16-harden-320-...:74`). The develop-loop GPU smoke: per-day ~2.2 min at 24-concept scale, WAKE/stream ≈ 15 s/day (`2026-06-23-longitudinal-develop-loop-GPU-GO.md`).
- **Window budget scales with vocab** (need enough windows per concept to learn its row of `M`): if a concept needs ~500 windows of context, then 2K concepts ≈ 1M windows ≈ **~9 GPU-hr**; 5K ≈ ~22 GPU-hr; 30K ≈ a few overnight runs. Compare the develop loop's own estimate: a "compressed year" (365 days) ≈ **13.5 hr** (`2026-06-23-...GO.md`) — i.e. a multi-day local run is exactly the intended operating mode (the owner accepts long local runs with an ETA).
- **VRAM during learning:** the stream cortex itself is tiny (`2700` neurons at 24 concepts; the hub→target CSR is ~5M synapses ≈ 2–3 GB). Learning a large vocab one bridge at a time keeps VRAM ~2–4 GB. **The >24 GB line is only hit if you try to hold all bridges resident simultaneously — which learning never requires.**

**⇒ Cloud is NOT needed for a VRAM wall.** Per the owner's standing directive (cloud only for a genuine >24 GB VRAM wall, not wall-clock): the foundational curriculum is **wall-clock-bound, not VRAM-bound**, so it runs **local** with an ETA; cloud (H100) would only *speed* it ~3–5× (cut a 22-hr run to ~5 hr), not enable it — and is justified only if the owner wants a faster turnaround on the full 100M tier.

---

## 3. THE MECHANISM — how the brain INGESTS the corpus (reuse vs new)

The develop loop **already is** the ingestion engine. Per simulated day (`_longitudinal_develop_loop_gpu.develop_gpu`, `:287-449`): **WAKE** = `cortex.hear_day(day_concepts, max_windows)` streams corpus windows containing the day's concepts, online-Hebbian into `M` (`:218-247`) → **CONVERSE** = `MultiTurnAgent` stores facts on the learned grounded codes → **SLEEP** = self-replay consolidation (no catastrophic forgetting) → **GROWTH** = `TierPromoter` (`sim/auto_growth.py`) promotes the architecture tier at mastery → **PERSIST** = `BridgeLineage` atomic HDF5 + JSON metadata, resumes next day (`sim/lineage.py`).

**Reuse (already built, validated):**
- The online-Hebbian stream cortex (`StreamCortex`, reads `M` from learned synapses; `corr(M,C)` learning-fidelity check) — `2026-06-15` + `2026-06-16` GO.
- The sparse-distributed multi-bridge ensemble (`g20_multibridge --sparse`, `generate_sparse_patterns`, deterministic-from-seed pattern regen) — 160/320 validated.
- The no-confab moat (learned Bogacz-Brown familiarity gate, `--moat learned`) + on-brain read-out normalization (`--readout-norm neural`) — `2026-06-16` GO, fully brain-based.
- Lineage persistence + per-day bundles + the webapp `/api/brains` picker (the watch-and-talk-to-a-developing-brain capstone) — `2026-06-24-week1-...`.
- TierPromoter growth + weight-transfer (`sim/auto_growth.py`, `TIER_LADDER`).
- Synapse tiering (`sim/synapse_storage.py`) for paging bridges past 24 GB.

**New (small, engineering-shaped — NOT research frontiers):**
1. **A streaming corpus loader.** `StreamCortex._load_token_stream()` (`:197-202`) is **hardcoded** to read ONE file `data/corpus/tinystories.txt`, loads the WHOLE file into memory, and **that file is not in the repo**. New: (a) a `--corpus-path` arg; (b) a *streaming* generator (read line-by-line / sharded, don't load a 10M–100M-word corpus fully into RAM); (c) the corpus file(s) themselves (download TinyStories / BabyLM). This is the #1 prerequisite and is pure engineering.
2. **A corpus-derived curriculum.** The day-by-day vocab (`_GPU_SYLLABUS`, `:85-132`) is a hardcoded 4-day Python list. New: derive each day's NEW concepts **from the corpus by frequency** (introduce high-frequency content words first = the natural developmental order: age-of-acquisition tracks frequency). Generalize `GradedCurriculum` to read a frequency-ranked word list from the corpus rather than a literal list. Engineering, not research.
3. **Hub-pool selection at scale.** `n_hub` (the fixed context pool) is currently small (200–500). For a large vocab the hub pool should be the top-`n_hub` most-frequent words of the *foundational* corpus (still a fixed, modest dimension — keeping `M` linear). A one-pass global frequency rank (already how it's done; noted as a defensible one-pass global statistic, `2026-06-15-...:41`). Trivial.
4. **Multi-bridge sharding of a large vocab.** Assign the foundational vocab to bridges (≤320/bridge), by semantic cluster (the `g20_vocab_spec_2048.py` 32-cluster taxonomy is *already designed* for exactly this — 32 clusters × 64 = 2048 — data-complete, not yet trained at scale). Reuse the spec; wire the stream cortex to learn per-cluster.

**The honest scope line:** ingestion = ~95% reuse + 4 small engineering pieces (corpus loader, corpus-derived curriculum, hub-pool-at-scale, multi-bridge sharding). **No new neural mechanism is required** to scale the *knowledge base*. (The deep-composition ceiling in §2c is the only research-shaped item, and the production design already routes around it for base-knowledge tasks.)

---

## 4. ANTI-CHEATS — the corpus is the ENVIRONMENT, not LLM-cognition

The owner's framing is the load-bearing guard. The de-risk MUST assert all four:

1. **The corpus is the environment; the BRAIN learns the codes (provenance).** The stream cortex learns `M` in its OWN synapses from co-activation windows (`hear_day` → `_present_window` drives `cp_external_input_current` → `_run_one_simulation_step()` → Hebbian; `M` read from `cp_connections`, `:253`). **No LLM ever supplies a code, a co-occurrence count, or a fact.** Anti-cheat: a **frozen-brain control** (`plasticity_on=False`, `:295-297`) — the brain HEARS the stream but learning is gated off → competence must NOT rise. Already built into the loop. Also: the LLM (if used at all, e.g. the off-bridge renderer) is **fluency-only** — it never reads `M`, never answers a knowledge query; the grounded-language faculty already enforces gate→constrain→verify so the LLM cannot inject knowledge (`project_grounded_language_faculty`).
2. **The no-confab moat holds at scale.** The learned familiarity gate (`--moat learned`) must keep **0 false-accepts** at the foundational vocab (it did at 320, 3-seed, closing the seed-43 host-threshold leak; `2026-06-16-harden-320-...`). De-risk asserts: query the brain about never-stored facts → abstain; lesion the learned projector → margin collapses to ~0 (decision rides learned synapses, not a host rule). NB the owner's standing relaxation: the moat is a **plus, not a hard gate** — graded confidence is acceptable if it buys scaling, as long as it never *fabricates*.
3. **The learned knowledge GENERALIZES at scale.** Not just memorized rows of `M` — the codes must place similar concepts near each other (cat≈dog) so the brain answers about novel-but-related concepts. Measured at 320: stream-cortex generalization **0.91** (`2026-06-15-...:28`); the cross-modal/PPMI generalization arc is GO. De-risk asserts held-out category-generalization above chance at the larger vocab, with a **category-derangement control** (shuffle category labels → generalization collapses).
4. **Honest-negative if the substrate can't scale.** If, at e.g. 2K concepts, discrimination drops below a usable bar OR the moat breaks OR generalization collapses to chance OR wall-clock is intractable even staged — that is the **scientific deliverable** (maps the substrate's foundational-knowledge ceiling), per the BRAIN-BASED-ONLY standard. Specifically: the 128-concept/single-bridge 84% discrimination is the known per-bridge soft-edge — the de-risk tests whether the *multi-bridge* route preserves per-bridge ≤320 @ ~98% at the larger total, or whether cross-bridge routing/interference degrades it.

---

## 5. CHEAP-FIRST DE-RISK + GO bars

**The question to answer cheaply:** *does the mechanism scale a MEANINGFUL step beyond 320 — to ~1–2K concepts from a REAL corpus — with the moat + generalization intact, at a measured VRAM/wall-clock — before committing to the full foundational corpus?*

### De-risk ladder (cheapest first; each gates the next)

**Step 0 — corpus-loader smoke (CPU, minutes).** Drop a REAL TinyStories shard into `data/corpus/`, add `--corpus-path` + a streaming reader, confirm `StreamCortex` streams windows from it and `corr(M,C) > 0.85` at the existing 24–64-concept scale (i.e. the loader change didn't break learning). GO bar: loader works, learning-fidelity unchanged. *Pure plumbing; no science risk.*

**Step 1 — single-bridge real-corpus scale-step (1 GPU, ~1–3 hr).** Stream a REAL TinyStories (or BabyLM-10M) shard to learn the **~320 most-frequent content words** on ONE bridge (reproduce the validated tier, but with concepts/vocab *derived from the corpus by frequency*, not the hand-curated taxonomy). GO bars (3 seeds): who/what recall ≥ 0.95; **moat 0 false-accepts**; generalization (held-out category) ≥ 0.80 with derangement-control collapse; **measure VRAM + wall-clock**. This proves the *corpus-derived* (not hand-curated) pipeline works at the validated scale.

**Step 2 — the meaningful scale-step: ~1,280 concepts, 4-bridge ensemble (1 GPU, staged ~6–12 GPU-hr).** Stream the corpus to ~1,280 corpus-frequency concepts across 4 sparse bridges (≤320/bridge), via `g20_multibridge --sparse` + the `g20_vocab_spec_2048` cluster sharding. **THIS is the decisive de-risk.** GO bars (3 seeds, **measured VRAM + wall-clock reported**):
- per-bridge discrimination ≥ 95% (the multi-bridge route preserves the ≤320 capability);
- cross-bridge who/what recall ≥ 0.90 (the documented cross-bridge associative retrieval holds 4×);
- **moat 0 false-accepts** across the 1,280-concept space (no fabrication at 4× vocab);
- generalization ≥ 0.80 with derangement collapse;
- frozen-brain control: competence does NOT rise (provenance = brain learned it);
- VRAM ≤ 24 GB resident (4 bridges fit; confirms local-feasible without tiling at this tier).

**GO ⇒** commit to the full **BabyLM-10M foundational corpus** at ~5K–8K core concepts (16–25 bridges, with synapse tiering for VRAM), run over the develop loop's simulated days/weeks (the per-day-bundle capstone makes it watchable). Then the 100M / ~15–30K far tier as a separate gated arc (cloud only if the owner wants the full-tier turnaround faster — wall-clock, not VRAM).

**NO-GO at Step 2 ⇒** honest-negative: the substrate's foundational-knowledge ceiling is below the multi-bridge promise (e.g. cross-bridge routing degrades discrimination, or the moat breaks at scale, or generalization collapses). That maps the real boundary and reframes (e.g. the deep-composition / learned-cortex frontier becomes load-bearing for richness, not just scale).

### Why this is the right cheap-first

Step 2 is the smallest run that tests the **actual** scaling claim (multi-bridge linear-in-count beyond 320, from a REAL corpus, moat + generalization intact, VRAM/wall-clock measured) — it is ~6–12 GPU-hr on local hardware, reuses validated machinery, and its result flips the decision: GO → full BabyLM; NO-GO → a characterized boundary + a research pivot. It avoids committing days of GPU to a 10M corpus before the 4× scale-step is proven.

---

## 6. VERDICT

**FEASIBLE — and local-first — via the multi-bridge stream-cortex ensemble, with a staged path. The scaling is engineering-bound (corpus loader + wall-clock + VRAM tiling), NOT substrate-bound, for the KNOWLEDGE base.**

- **The feared `V²` co-occurrence wall does not exist:** the stream cortex learns `M[V, n_hub]` (linear, `n_hub` fixed ~200–500), validated biology-faithful. Learning + PPMI normalization are `O(V·n_hub)`.
- **Per-substrate discrimination is the real ceiling, and the multi-bridge route (linear in bridge count) is the validated answer:** 64@100%, 320@98.4%/bridge; 2K ≈ 7 bridges (~22 GB, near the 24 GB line resident); ≥5K needs the existing synapse tiering (page bridges in/out — they never interact during learning, so VRAM is not a true wall). **Cloud is NOT needed for VRAM; the run is wall-clock-bound** (a few overnight local runs for the foundational tier, per the owner's accepted long-local-run mode + ETA).
- **The deep-composition ceiling (`D∝M²` two-attribute, recursive clauses at large per-codebook vocab) is already mapped and routed-around** (per-bridge ≤320 + GPU resonator + sparse-block far lever); it does not bind for base-knowledge tasks (recall/who-what/abstain/single-attr/multi-hop).
- **The mechanism is ~95% reuse** (develop loop + stream cortex + multi-bridge + lineage + tiering + TierPromoter + learned moat + neural read-out); the 4 new pieces (streaming corpus loader, corpus-derived curriculum, hub-pool-at-scale, multi-bridge sharding) are engineering, not research.

**Recommended staged path:**
1. **Now:** corpus-loader smoke (Step 0) + single-bridge real-corpus 320 (Step 1) — proves the *corpus-derived* pipeline at the validated scale.
2. **The decisive de-risk:** ~1,280 concepts on a 4-bridge ensemble from a REAL TinyStories/BabyLM-10M shard (Step 2) — GO bars on discrimination + cross-bridge recall + moat + generalization + measured VRAM/wall-clock.
3. **On GO:** full **BabyLM-10M** foundational corpus (~5K–8K core concepts, 16–25 bridges + synapse tiering) over the develop loop's simulated days/weeks (watchable via the per-day-bundle console capstone).
4. **Far tier (separate gated arc):** BabyLM-100M / ~15–30K vocab; cloud-justifiable only for faster full-tier turnaround (wall-clock), never for VRAM.

**The genuine open risk to watch (the honest-negative trigger):** whether the **multi-bridge route preserves per-bridge ≤320 discrimination + the moat + generalization** at 4× total vocab from a real (noisier-than-curated) corpus, and whether **cross-bridge routing** (which bridge holds a queried concept, cross-bridge associative retrieval) degrades at thousands of concepts. Step 2 is designed to surface exactly this.

---

## Sources / artifacts

**In-repo (load-bearing, verified):**
- `research/runners/_longitudinal_develop_loop_gpu.py` (`StreamCortex`: `M[Nt,n_hub]` at `:191`/`:254-256`; hardcoded corpus loader `:197-202`; `hear_day` `:218-247`; `develop_gpu` 5-stage loop `:287-449`; frozen-brain anti-cheat `:295-297`; `_GPU_SYLLABUS` hardcoded curriculum `:85-132`)
- `research/runners/_phaseB_online_stream_cortex_derisk.py` (numpy reference: online Hebbian `M`, `double_center` normalization)
- `research/runners/concept_pool_sparse_distributed.py` (`generate_sparse_patterns(n_concepts,n_pool,pattern_size,seed)` `:137-152`; `build_sparse_pool_bridge` `:53-134`)
- `research/runners/g20_multibridge.py` (`--sparse` multi-bridge loader); `g20_vocab_spec_2048.py` (32-cluster × 64 = 2048 spec, data-complete, NOT yet trained at scale); `g20_subtaxonomy_2048.py`
- `sim/auto_growth.py` (`TierPromoter`, `TIER_LADDER`, weight-transfer); `sim/lineage.py` (atomic HDF5+JSON persistence); `sim/synapse_storage.py` (`TieredSynapseStore`)
- Findings: `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md` (+0.513, gen 0.91); `2026-06-16-harden-320-stream-cortex-learned-moat-GO.md` (fully brain-based 320, learned moat, neural read-out, ~84 min GPU); `2026-05-15-sparse-distributed-capacity-curve.md` (64@100%/128@84%); `2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md` (320 @ 98.4%/bridge); `2026-05-22-vocabulary-scaling-64concept-NEGATIVE-...-too-sparse...md` (untrained-substrate setup gap, not a limit); `2026-06-04-vocab-scaling-capacity-curve-refines-cost-model.md` + `2026-06-04-capacity-curve-scaling-cost-model.md` (D∝M² composition ceiling, per-bridge≤320 route); `2026-06-23-longitudinal-develop-loop-GPU-GO.md` (per-day ~2.2 min, year ≈13.5 hr); `2026-06-24-week1-develop-loop-console-capstone.md` (per-day bundles + console)

**External (corpus scale):**
- [Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora](https://arxiv.org/abs/2412.05149) — children hear 2M–7M words/yr; 10M (Strict-Small, ~age 2–5) and 100M (Strict, ~age 12) word tracks; corpus = CHILDES + children's books + Gutenberg + subtitles + Wikipedia + dialogue.
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) — ~1,500 core content words; ~476M tokens; 3–4-yr-old experiential domain.
- [Evaluating lexical coverage in Simple English Wikipedia](https://files.eric.ed.gov/fulltext/ED578271.pdf) / [Sketch Engine English Wikipedia corpus](https://www.sketchengine.eu/english-wikipedia-corpus/) — Simple Wikipedia controlled-vocab encyclopedic facts.
- [Is Child-Directed Speech Effective Training Data for LMs?](https://arxiv.org/pdf/2408.03617) / CHILDES — English subset ~29M words; ~6.5M words child-directed 0–60mo.
- [Average Vocabulary Size by Age](https://vocabulary-test.com/average-vocabulary-size/) — adult native speakers ~20K–35K word families (mean ~27K); active contemporary English ~100K–150K.
