---
type: plan
status: live
date: 2026-05-11
---

# Strategic re-evaluation — what we have, what's missing, three paths forward

**Date:** 2026-05-11 02:30 EDT
**Status:** STRATEGIC — stop-and-think doc before sinking more autonomous-arc
effort into architectural scaling that may not address the real gap.
**Trigger:** User (2026-05-11) — "even just the first use with an end user
is expected to be hundreds of words, not to mention that the user would
expect the sim pretrained with knowledge/capabilities comparable to at
least a tiny SOTA LLM."

---

## Executive summary

We have built a **biology-grounded continual-learning substrate** that works
at small scale: 64-word vocabulary, ~50M synapses, sub-2-sec inference at
the validated 16-word arch, retention validated across catastrophic-
forgetting protocols (Phase 1.4), and memory consolidation validated via
sleep replay (Phase 1.3). Tonight we shipped persistent state (lineage)
and auto-growth scaffolding so the sim "lives" between sessions.

We have **not** built anything that approximates a tiny SOTA LLM.
The current architecture is a *motor-binding* substrate (word → N/E/S/W
direction) with no compositional grammar, no multi-entity world knowledge,
no abstract concepts beyond direction primitives, and no reasoning chains.
Scaling parameter count alone won't bridge that gap — the architecture
itself needs to change. We are 10-76× off in raw parameters and
structurally incomplete for language modeling.

This doc lays out three paths forward, each with realistic risk/cost/
timeline, and proposes a decision conversation before more autonomous
scaling effort is spent.

## What current capability actually delivers

Honest inventory of what's validated, what's shipped, and what's claimed
but unproven.

### Validated science (multi-seed GO)

1. **Tier 1 binding** — 4-word vocabulary (north/east/south/west), 5/6 W→A
   + 6/6 A→W aligned across 6 seeds. Foundation: embodied Hebbian co-
   firing + STDP + topographic prior + FS lateral inhibition.
2. **Tier 2.1 synonym binding** — 8-word vocabulary (4 primary + 4 synonyms),
   5/6 W→A + 6/6 A→W aligned, A→W mean 63.7%. Scaled arch (n_lang=4096,
   n_motor=1000). Capacity hypothesis validated.
3. **Phase 1.4 BRANCH A retention** — 5/6 seeds PASS at ≥80% primary
   retention when training NEW synonym vocab on top of existing primaries.
   No catastrophic forgetting at synonym scale.
4. **Phase 1.3 consolidation** — 3/3 seeds GO on combined Phase 1.3+Tier 2.1
   eval. Hippocampus-OFF retention ≥80% after sleep replay — memory truly
   consolidated to cortex. Anti-cheat 3-seed (`--strict-silence`) confirmed.
5. **STP default flip** — STP off during training = 3.28× faster + higher
   accuracy across 3 seeds. STP reversible at inference time (re-enable
   for biology realism without retraining).
6. **Encoding-axis scale-up** — n_lang=8192 unlocks 64-word vocabulary at
   3-seed GO. The discovery that broke the prior "~333 motor neurons per
   sub-pop" capacity rule.

### Validated negative results (real findings)

1. **96-word XL NEGATIVE (2026-05-11)** — n_lang=16384 doesn't fix the
   96-word retention wall; accuracy converges to primary-direction floor
   (~25%) across 350 trials. Encoding-axis scale-up alone insufficient;
   motor-pool capacity is the next bottleneck.
2. **Cluster-stacking falsified (2026-04-30)** — 8 attempts to stack
   cluster B/C/D/F on A+E baseline, all NEUTRAL or NEGATIVE. A+E remains
   the robust operational ceiling for cheat-5 multi-goal navigation.
3. **Three-factor learning fails at W→A scale (2026-05-05)** — Both classical
   sign-only DA and magnitude-graded DA fall below dendritic-learning
   decision gate at biological scale. Global scalar feedback insufficient.
4. **Phase 2.3a next-char features don't transfer (2026-05-07)** — Pretrained
   cortex (Phase 2.2 next-char SNN) used as adapter feature extractor:
   22% W→A vs 28% random init. Char-level next-char features at 134K-
   param scale ~4 orders of magnitude too small for direction-word
   embeddings to be useful.
5. **Cheat-5 v1/v2/v3.1/v4 all NEGATIVE** — cross-projections aren't
   fundamentally broken, but are under-constrained without the biology
   buildout (D1/D2 asymmetry, FSIs, TANs, structural plasticity, etc.)

### Shipped infrastructure (2026-05 sprint)

1. **Bridge Lineage Manager** (commits 3030517 → 0c5f81b, this arc)
   — persistent training state, atomic save/load, history snapshots,
   fork/rollback, growth-log markdown, webapp endpoints, frontend tab,
   CLI subcommands. 83 tests across the subsystem. The sim "lives"
   between sessions.
2. **Auto-growth Phase A scaffold** (commit aef098a)
   — TierPromoter + TierLadder + transfer-weights pure-Python logic.
   25 tests. Substrate for tier-promotion via checkpoint reload.
3. **Inference benchmark suite** (commit 8e2aa77)
   — 8 vocab tiers × :speak latency measured. Identifies n_lang as
   dominant cost driver, three clean arch bands (1.3 / 1.7 / 6.5 sec).
4. **Text I/O infrastructure** — Tier 2.1 BREAKTHROUGH + Phase 2.2
   surrogate-grad BPTT (path-f-hybrid branch) + char tokenizer.
5. **Webapp** — Bridges + Lineages + Findings + Plans + Brain 3D viz
   + Experiment launcher + capability status. 50+ tests.

### What we don't have (the real gap)

1. **Compositional grammar** — no subject + verb + object handling.
   "north" works; "the bird flies north" doesn't even parse.
2. **Multi-entity world knowledge** — no concept of objects, properties,
   relations beyond the 4 direction primitives.
3. **Abstract concepts** — no categorization, no inheritance, no
   metaphor, no analogy.
4. **Reasoning chains** — no multi-step inference. Even simple "if A
   then B; A; therefore B" is out of scope.
5. **Discourse memory** — the :history command tracks turns, but the
   sim itself has no narrative memory beyond the bound primaries.
6. **Procedural memory** — no skill learning beyond the motor-pool
   binding for direction primitives.
7. **Cross-domain transfer** — text-to-action works at 4-8 words; no
   evidence it transfers to action-to-text generation at larger scales,
   text-to-vision, or any other cross-modal capability.

## What "hundreds of words" actually requires

Two distinct gaps that get conflated in scaling conversations:

### Quantitative gap (raw parameters)

| Target | Params | Multiplier over our 64w ceiling |
|---|---|---|
| Our 64w validated arch | 50M | 1× |
| Our 96w XL (NEGATIVE) | 112M | 2× |
| Qwen2.5-0.5B | 500M | 10× |
| TinyLlama 1.1B | 1.1B | 22× |
| Phi-3-mini 3.8B | 3.8B | 76× |
| Llama 3.2 1B | 1B | 20× |

Closing this gap requires either:
- (a) Cloud-anchored training on A100/H100 clusters (Phase 2 of master plan,
  $1K-50K, 6-18 months)
- (b) CPU/RAM/SSD tiering to fit bigger sims on commodity hardware (less
  GPU-bound; foundation for everything else; 1-2 months for the substrate)
- (c) Architectural compression (sparse activations, mixture-of-experts,
  pruning) — research bet; uncertain.

### Qualitative gap (architecture)

Closing this CANNOT be done by scaling parameters in the current
Phase 1.4 BRANCH A architecture. The architecture is *motor-binding-
shaped*: 4 motor pools modeling directional actions, language_input
driving them via Hebbian co-firing. A 5B-synapse version of this would
be a very big motor-binding network, not a language model.

To support compositional language at hundreds-of-words scale we need
*new architectural primitives* the current sim doesn't have:

1. **Hierarchical cortex** — multiple cortical layers with feedforward
   + feedback connectivity, not the current single-layer language → motor
   topology. Models: Felleman & Van Essen 1991 cortex hierarchy; Mountcastle
   columns.
2. **Compositional binding** — variable binding (this entity ↔ that
   property), role-filler representations. Biology: prefrontal cortex
   working memory; thalamic synchronization. Models: Smolensky tensor
   products; binding-via-synchrony.
3. **Attention / gating** — top-down control over which sensory features
   reach motor / language pathways. Biology: cholinergic modulation,
   pulvinar gating, basal-ganglia disinhibition.
4. **Episodic + semantic memory hierarchy** — declarative knowledge
   separate from motor binding. Biology: medial temporal lobe / temporal
   cortex; cortico-hippocampal complementary learning systems (we have
   partial: Phase 1.3 consolidation, but only for direction primitives).
5. **Sequence prediction** — predict next token given context. Either
   biological (cortex predicts next sensory input; Predictive Coding)
   or hybrid (transformer module).

We have working biological infrastructure for (1)-(4) at small scale, but
nothing scaled to support hundreds-of-words discourse. (5) is the place
where the Path F hybrid was exploring surrogate-grad BPTT — that scaled
to a 4-layer SNN on Tiny Shakespeare but Phase 2.3a showed the small-
scale features don't transfer to the biology-grounded W→A task. The
Project Nord reference (1.088B params, FineWeb-Edu corpus) is ~4 orders
of magnitude above our tested scale.

## Three realistic paths forward

Each path has different risk profile, capability ceiling, timeline,
and what gets thrown away vs preserved from the current sim.

### Path 1 — Biology-grounded scale-up (current trajectory, ambitious)

**Thesis:** The biology-grounded SNN approach scales to LLM-class
capability. We add hierarchical cortex, compositional binding,
attention via thalamocortical loops, episodic memory beyond direction
primitives, and predictive-coding-style sequence prediction. We
validate at cloud-anchored scale (Phase 2 master plan: $1K-50K, 6-18
months) and ship a working biology-grounded LLM equivalent.

**What survives:** all of tonight's work. Lineage, retention,
consolidation, the validated 4-64 word arc, the auto-growth scaffold.
All foundational.

**What's added:**
- Multi-cortical-area hierarchy (sim/cortex_hierarchy.py — new)
- Compositional binding primitives (variable binding via gamma synchrony
  or tensor products)
- Attention gating (cholinergic + pulvinar + BG disinhibition modules)
- Episodic memory beyond direction (hippocampus → temporal cortex
  pipeline at scale)
- Predictive-coding sequence layer (cortex predicts next sensory frame)
- 100-1000× more synapses (cloud-anchored training)

**Cost:** $1K-50K cloud (per strategic addendum). 12-24 months focused
work. Multi-disciplinary team if doing this seriously.

**Risk:** **Highest of the three paths.** Nobody has built a biology-
grounded LLM at scale. Project Nord (1B+ params, SOTA biology-inspired)
exists but their results aren't competitive with LLM SOTA at the same
parameter budget. We'd be inventing the playbook, not following it.
Could land somewhere in the "interesting research artifact" zone rather
than "useful product."

**What it delivers:** a research artifact demonstrating biological
plausibility at scale + a working chat sim that learns continually.
The unique value prop ("an AI that learns like a brain, not like
gradient descent") is preserved. Potentially publishable.

**Falsification criteria (would tell us to abandon Path 1):**
- We can't get hierarchical cortex stable past 4 layers at 1M-synapse
  scale within 6 months of focused work.
- Compositional binding evaluation (e.g. "the bird flies north" decomposed
  into entity + action + direction) fails to beat random across 3 seeds.
- Cloud-anchored training cost exceeds $50K without crossing the
  100-word vocabulary threshold.

### Path 2 — Hybrid SNN + transformer (pragmatic, moderate risk)

**Thesis:** The biology-grounded sim is *the right substrate* for embodied
/ motor / continual-learning capabilities (where we have validated wins).
Transformers are the right substrate for compositional language and world
knowledge. Combine them: SNN as embodied substrate + transformer as
language processor + biology-grounded plasticity for the SNN side +
LoRA-style continual learning for the transformer side.

Path F-hybrid (branch state as of 2026-05-07) was an early sketch — it
used surrogate-grad BPTT for cortex pretraining + biology-grounded
plasticity for the Phase 1.4 BRANCH A arch. The 2.3a negative result
showed naive feature transfer doesn't work at toy scale; the lesson
is that the SNN and transformer should be **complementary** not
**stacked**.

**What survives:** all of tonight's lineage + retention + consolidation
+ the validated 4-64 word arc as the *embodied substrate*. Plus the
Path F infrastructure (BPTT, char tokenizer, surrogate-grad).

**What's added:**
- Transformer language module — locally runnable (Phi-3-mini scale OR
  bespoke small transformer trained from scratch). Handles tokenization,
  grammar, multi-entity discourse.
- Bridge interface layer — SNN ↔ transformer translator. Transformer
  outputs concept tokens; SNN binds them to motor/sensory pathways.
- Biology-grounded continual learning ON the transformer — LoRA / DoRA /
  similar weight-delta methods plus a sleep-replay consolidation loop.
- Unified lineage — both the SNN state AND the transformer's LoRA deltas
  saved/loaded together. The "brain state" includes both substrates.

**Cost:** $1K-5K cloud for any transformer pretraining; mostly local
work otherwise. 6-12 months. Single-developer feasible.

**Risk:** **Moderate.** Hybrid architectures are known to work (e.g.
SpikeGPT, NeuroPilot research). Integration complexity is the main
risk — two substrates means two failure modes. But the components
individually are well-understood.

**What it delivers:** a chat sim that handles "hundreds of words"
naturally via the transformer, *and* learns continually via the SNN's
embodied / motor / memory pathways. The "tiny SOTA LLM equivalent" target
is delivered by the transformer side; the biology-grounded learning is
the distinctive differentiator that vanilla LLM chat doesn't have.

**Falsification criteria:**
- We can't make the SNN ↔ transformer interface lossy-enough for the
  transformer's vocabulary to project into SNN motor space at >50%
  retention.
- LoRA continual learning + sleep replay produces catastrophic
  forgetting (the SNN side already validated this works; question is
  whether it transfers to a transformer).
- Combined system is slower than running a transformer alone, with no
  capability gain.

### Path 3 — LLM with biology-inspired memory subsystem (most pragmatic)

**Thesis:** A real, locally-runnable LLM (Phi-3-mini, Llama 3.2 1B,
Qwen2.5-0.5B, or similar) handles all the language/cognition. The
biology-grounded sim becomes the **memory subsystem** — a continually-
learning knowledge graph that the LLM queries between turns. The
distinctive thing isn't the chat substrate; it's the *memory* substrate.

Tonight's lineage + retention + consolidation work becomes the core
product feature: "this is an LLM with a memory that grows over weeks
and months without catastrophic forgetting, because the memory is
stored in a biology-grounded continual-learning sim."

**What survives:**
- Lineage system (this is now the *product*, not just infrastructure)
- Phase 1.3 consolidation (this is the *memory consolidation*, biology-
  grounded)
- Phase 1.4 retention (this is the *no-catastrophic-forgetting* guarantee)
- Auto-growth (this is the *memory grows with use*)
- 78-test lineage subsystem and all the operational infrastructure

**What changes:**
- Project framing — "a brain-inspired AI sim" → "an LLM with a
  brain-inspired memory subsystem"
- Architecture — LLM as the chat layer; sim as a memory subsystem
  callable via tool-use or in-context retrieval
- The whole "scale the SNN to handle hundreds of words" line of work
  is OUT of scope. SNN handles direction-primitive embodied tasks
  only. Everything else is the LLM.

**What's added:**
- Tool-use bridge between LLM and sim (e.g. "store this fact"
  → SNN binds; "recall what we discussed about X" → SNN retrieves)
- LLM hosting (vLLM / llama.cpp / ollama; locally runnable)
- A clean product story

**Cost:** essentially zero cloud spend. Local LLM runs on the same
RTX 3090. 3-6 months of focused product/integration work.

**Risk:** **Lowest of the three paths.** Both halves work; integration
is straightforward. The risk is **project identity** — we'd be moving
the centerpiece from the SNN to the LLM. The biology work becomes a
supporting feature, not the headline.

**What it delivers:** a usable chat product within 3-6 months. The
distinctive feature is the lineage-backed memory (vs vanilla RAG, which
forgets across sessions and overwrites). The biology research continues
as a supporting track.

**Falsification criteria:**
- LLM tool-use can't reliably write to / read from the SNN-backed
  memory (latency, reliability, hallucinated memories).
- The lineage-backed memory provides no measurable advantage over
  vanilla RAG with a vector DB.

## Comparison matrix

| Dimension | Path 1: Biology scale-up | Path 2: Hybrid | Path 3: LLM + bio memory |
|-----------|--------------------------|----------------|--------------------------|
| Risk | High | Moderate | Low |
| Capability ceiling | Unknown (potentially highest) | Tiny-SOTA-LLM level | LLM level |
| Timeline | 12-24 months | 6-12 months | 3-6 months |
| Cloud cost | $10K-50K | $1K-5K | ~$0 |
| Local hardware | RTX 3090 + cloud bursts | RTX 3090 | RTX 3090 |
| Throws away | Nothing | Nothing | "SNN is the chat substrate" framing |
| Distinctive value | Biology-grounded AGI substrate | Biology-grounded embodied + transformer cognition | Biology-grounded continual memory |
| Publishable | Yes — novel architecture | Yes — hybrid integration | Maybe — memory subsystem |
| Useful product timeline | 12+ months | 6-9 months | 3 months |
| Probability of working | 20-40% | 60-80% | 90%+ |

## Decision criteria — questions to answer before committing

1. **What's the actual goal?**
   - "Publish a paper on biology-grounded AGI" → Path 1
   - "Ship a useful chat product within 12 months" → Path 2 or 3
   - "Demonstrate continual learning in a real-world setting" → Path 2 or 3

2. **What's the time + money budget?**
   - >12 months focused work + $10K+ cloud → Path 1 viable
   - 6-12 months + ~$5K → Path 2 viable
   - <6 months, near-zero cloud → Path 3 only

3. **What's the team?**
   - Solo developer → Path 3 (Path 2 stretch; Path 1 likely not feasible)
   - 2-3 developers + ML researcher → Path 2 viable
   - Research team → Path 1 viable

4. **What's the user's primary value-prop priority?**
   - "An AI that learns biologically" → Path 1 strongest
   - "A useful chat sim that learns continuously" → Path 2 strongest
   - "An LLM that remembers across sessions" → Path 3 strongest

5. **What's the falsification appetite?**
   - High (willing to spend 6 months and abandon if no progress) → Path 1
   - Moderate (want to ship something useful within a year) → Path 2
   - Low (need product traction immediately) → Path 3

## Recommended next steps regardless of path

### Foundational (do regardless)

1. **CPU/RAM/SSD tiering** — `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`
   (shipped this session as companion to this doc). Hardware-independence;
   unlocks bigger sims; foundation for whichever path we pick.

2. **NumPy reference backend** — part of the tiering work. CI without
   GPU, algorithmic verification, Mac M-series compatibility.

3. **Lineage workflow stress-test** — once we have lineage in production
   use across 5+ sessions, run a "memory across sessions" demo. Useful
   for all three paths.

### Path-specific (depends on decision)

4a. **(Path 1)** Multi-cortical-area hierarchy design doc + scaffold.
    Compositional binding research. Cloud-anchored training plan.

4b. **(Path 2)** SNN ↔ transformer interface design. Local transformer
    selection (Phi-3-mini vs bespoke). LoRA continual-learning protocol.

4c. **(Path 3)** Tool-use bridge design (LLM → sim API). Local LLM
    hosting decision (vLLM / llama.cpp / ollama). Product-framing
    documentation update (README, capability_status.json).

### What to STOP doing (regardless of path)

- **Stop scaling synonym binding for its own sake.** 4 → 8 → 16 → 64 →
  96 was the right exploration; we now know the ceiling. Don't run 32-word,
  128-word, or 256-word synonym smokes "to see what happens." We know
  what happens.
- **Stop treating small-word success as predictive of large-word success.**
  The 64-word arch is GO; the 96-word arch is NEGATIVE. The arch is the
  binding factor, not vocab size. Cluster-stacking falsification
  (2026-04-30) should have already taught us this; the 96-word XL
  NEGATIVE (tonight) reinforces it.
- **Stop chasing kernel-level speedups in isolation.** Phase 1
  optimization design doc targets 3-5× — useful, but doesn't address the
  architecture gap. Do it as part of a path-specific plan, not as a
  standalone effort.
- **Stop pretending capability=parameter count.** A 5B-synapse SNN trained
  on direction primitives is not a tiny SOTA LLM. Even if VRAM and compute
  weren't constraints, the capability gap would persist.

## Honest read on the project's distinctive value

Throughout all three paths, **the lineage + Phase 1.3 + Phase 1.4 + retention
+ consolidation work shipped to date is the most distinctive thing in
the project**. No mainstream LLM has biology-grounded continual learning
with the retention guarantees we've validated. Nobody else has lineage
as a first-class concept — you can fork an LLM, but you can't "fork the
brain state and explore a branch."

That should be the **product positioning** regardless of path:

- Path 1: "an AI that learns biologically, including its memory"
- Path 2: "a chat sim with biology-grounded embodied + memory substrates"
- Path 3: "an LLM with biology-grounded continual memory"

The biology is the moat. The chat substrate is replaceable. We've been
spending effort on the wrong layer.

## Provenance + open questions

- This doc: `docs/plans/2026-05-11-strategic-reevaluation.md`
- Companion: `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`
- Master plan: `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`
- Strategic addendum: `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
- Phase 1 optimization: `docs/plans/2026-05-10-phase1-local-optimization-design.md`
- Auto-growth: `docs/plans/2026-05-10-auto-growth-design.md`
- Bridge lineage: `docs/plans/2026-05-10-bridge-lineage-design.md`

**Open questions for the user / strategic conversation:**

1. Which path most aligns with your goal? (Or is there a Path 4 we're missing?)
2. What's the realistic time + budget?
3. Do you want a publishable artifact or a usable product or both?
4. How much of the SNN work are you emotionally / technically attached to
   keeping at the center of the project vs willing to move to a supporting
   role?
5. Is the "tiny SOTA LLM equivalent" target a hard requirement or a North
   Star? If hard requirement, Path 1 is the only one that delivers it
   *as an SNN*; Path 2/3 deliver it via integration.

The next autonomous-arc effort should be either (a) implementing the CPU/
RAM/SSD tiering (foundational regardless of path) or (b) starting the
chosen path's design work after this strategic conversation lands.
