---
type: plan
status: live
date: 2026-05-10
---

# Master plan strategic addendum — 3-phase commitment + auto-growth

**Date:** 2026-05-10 23:30 EDT
**Trigger:** User strategic question on whether to scale toward LLM
capability or optimize for local-first continual-learning differentiator.
**Status:** Approved direction; 3 strategic phases + auto-growth pipeline.

---

## The strategic question (paraphrased from user 2026-05-10)

"This sim could be groundbreaking in continual learning + biology-grounded
growth, but matching SOTA LLM raw capability looks compute-prohibitive.
Should we invest heavy compute toward LLM-parity, or focus on optimization
that makes day-to-day continual learning sustainable on local hardware?
And can the sim auto-grow as it learns, instead of pre-sizing to max
expected scale?"

## Strategic decision

**Don't try to match Qwen3-class SOTA LLM raw capability.** The biology-
grounded SNN compute cost at 50K+ word vocab is multi-million-dollar
distributed-cluster territory. Compounding factors: time-loop (1000+
sim steps per inference), bigger-than-feedforward synapse count, slower
construction time, custom kernels can't fully match Tensor Core efficiency.

**DO push for the differentiator: continual learning + episodic memory +
multimodal grounding at the largest vocab tier that's sustainably usable
on local hardware.** A user living with a 500-word continuously-learning
agent that never forgets is more interesting than a 50K-word static LLM.

**Auto-growth is a key piece of the value prop:** sim should grow with
the user, not be pre-sized for max expected scale.

## Three-phase commitment

### Phase 1 (next 1-3 months, local + spot cloud)

Push aggressive local optimization, target 5-10× more speed on RTX 3090.

- Per-pathway STP gates (biology + selective optimization)
- Bridge construction profiling + speedup (current 5-44 min build is the
  bottleneck beyond n_lang=16384)
- Sparser cross-region density (0.1 → 0.05 with revalidation)
- FP16 throughout (membrane V, conductances) where validated safe
- Multi-stream GPU execution
- Fused kernel extension

Target: **256-512 word vocab usable on local RTX 3090**, with day-to-day
continual learning sustainable (~1-2 hr/day GPU at this tier).

Cost: $0-200 (spot pricing for big smokes), ~100-200 commits, weeks of
focused work.

**Phase 1 decision point: does continual learning work meaningfully at
this scale? Can a user "live with" a 256-512 word agent for a week and
see it accumulate knowledge?**

### Phase 2 (3-9 months, cloud-anchored)

If Phase 1 demo'd, scale to 1000-2500 word vocab via cloud training,
keep local for day-to-day inference.

- Cloud training: A100/H100 spot pricing, periodic 12-48 hour multi-seed
  runs
- Multi-modal binding: vision + language co-firing (Cluster K v2 visual
  cortex integration with chat language pathways)
- Auto-growth Phase A + B implementation (see below)
- Sleep replay scaling: longer NREM consolidation phases for richer
  vocabulary
- Hippocampus + cortex consolidation rigorous testing

Target: **1000-2500 word vocab + multimodal + continual** validated
multi-seed.

Cost: $1K-10K cloud compute total.

**Phase 2 decision point: is the continual-learning differentiator
real and measurable? Does the sim outperform LLM fine-tuning for
"learn this new fact and retain it" workflows?**

### Phase 3 (9-24 months, mixed local+cloud)

Conditional fork:

- **Path A (biology-grounded standalone)**: If Phase 2 shows continual
  learning is genuinely competitive, scale to 5000-10K word vocab on
  cloud (multi-H100 nodes). User interaction migrates to cloud API
  endpoint. Cost: $30K-500K.

- **Path B (hybrid LLM + sim)**: If Phase 2 reveals continual learning
  alone isn't enough to compete, pivot. LLM handles syntax/parsing/large
  vocabulary; sim handles continual learning + episodic memory + biology.
  Shared embedding space. Cost: lower than Path A (sim stays smaller).

## Auto-growth pipeline (4 sub-phases A-D)

The sim should grow autonomously as the user interacts with it,
matching the biological reality of childhood neural development.

### Growth Phase A: tier promotion via checkpoint reload (~1 week)

The "starter feature" — discrete tier jumps but feels like growth from
user perspective.

1. Train at current tier
2. Eval: when accuracy > threshold for K consecutive epochs, trigger
   promotion
3. Save bridge state to checkpoint
4. Build next-tier bridge (bigger n_lang, n_motor, vocab)
5. Transfer weights from old motor pools to corresponding sub-pops in
   new pool. Random-initialize the new sub-pops added by tier expansion.
6. Continue training in new arch
7. UX: "Your agent grew from 8-word to 16-word vocabulary"

**Status today:** All pieces exist (checkpoint save/load, tier configs).
Wiring needed: auto-promotion trigger + cross-tier weight transfer
function.

**Effort:** ~1 week of focused work.

### Growth Phase B: within-tier structural plasticity (~2-3 weeks)

Synapses form and prune within fixed pool capacity based on
co-activation. Biology-real but enabled but disabled for chat configs.

1. Enable `cfg.enable_structural_plasticity = True` in chat configs
   (currently False to save compute)
2. Activity-dependent synaptogenesis: high-co-firing pairs grow new
   synapses if they don't already have one
3. Pruning: low-utility synapses removed (biology: ~50% pruning during
   childhood)
4. Validation: does this improve binding quality or just add cost?

**Status today:** Infrastructure exists in `bridge.update_pruning()`,
`cfg.enable_structural_plasticity`. Needs:
- Re-enable in chat config
- Tune `struct_plast_activity_bias` parameter for current arch
- Multi-seed validation that it improves (vs hurts) binding accuracy

**Effort:** ~2-3 weeks (validation is the slow part).

### Growth Phase C: online neurogenesis (~1-2 months)

True online growth — pool size itself grows as needed.

1. Detect: user introduces new word, no spare sub-pop available
2. Allocate: extend motor pool by N neurons (e.g. 125 per new sub-pop)
3. Wire: trigger structural plasticity from lang_input to new neurons
4. Persist: save updated bridge to checkpoint
5. CuPy CSR matrix re-allocation handled via the existing
   `_synapse_capacity` growth_factor pre-allocation OR explicit
   re-build at growth events

**Status today:** Infrastructure for capacity growth exists (capacity =
N × growth_factor pre-allocation). Needs:
- Detection logic in chat REPL
- Dynamic pool expansion at runtime
- CSR matrix re-build on capacity exceeded (slow but rare)
- Region manager dynamic re-init support

**Effort:** ~1-2 months. Significant engineering, performance edge cases.

### Growth Phase D: hardware migration orchestration (~2 weeks)

When local hardware saturates, auto-migrate to cloud.

1. Detect: VRAM > 80% of local capacity, OR steps/sec < 5
2. Snapshot bridge state to local checkpoint
3. Upload checkpoint to cloud storage
4. Spin up cloud H100 instance (Vast.ai, RunPod, Lambda, AWS spot)
5. Resume training in cloud
6. UX: "Your agent has grown beyond your local hardware. Migrate to
   cloud?"

**Status today:** Checkpoint save/load works. Cloud orchestration is
its own project: API integration with cloud providers, billing setup,
state sync (download trained checkpoint back).

**Effort:** ~2 weeks for the orchestration infra. Defer until Phase C
demonstrates need.

## Recommended Phase 1 work breakdown

Critical-path order, ~100-200 commits:

1. **Continue current arc**: finish 96-word XL smoke + Tier 2.3 retest
2. **Local-side perf push**: items 4-8 from `2026-05-10-perf-optimization-audit.md`
   - FP16 throughout (membrane V) — biggest single potential win, needs
     spike-train divergence analysis
   - Sparser density (0.1 → 0.05) — full revalidation needed
   - Multi-stream GPU
   - Fused kernel extension
   - Bridge construction profiling + targeted optimization (the 44-min
     build at n_lang=32768 is the wall)
3. **Per-pathway STP gates**: biology + selective optimization
4. **Growth Phase A (tier promotion)**: ~1 week
5. **Growth Phase B (within-tier structural plasticity)**: ~2-3 weeks
6. **Webapp chat UI WebSocket + bridges library "load" workflow**: 1-2 days
   (deferred so far)
7. **Inference benchmark suite**: verify predicted latency chart across
   all validated vocab tiers
8. **256-word smoke (post-optimization)**: revisit with optimized stack;
   the original 256-word at-chance result may improve with bigger
   encoding + optimizations
9. **512-word smoke**: stretch goal for end of Phase 1
10. **Continual-learning life-test**: simulate a "user lives with the
    agent for a week" scenario; track knowledge accumulation, forgetting,
    consolidation

## Recommended Phase 2 work breakdown (sketch)

- Cloud orchestration infrastructure (Phase D pieces)
- Auto-growth Phase C (online neurogenesis)
- Vision + language multimodal binding
- 1000-2500 word vocab on cloud H100, multi-seed
- "Live with agent for a month" longitudinal test
- Comparison to LLM fine-tuning for continual-learning workflows

## Reference: today's optimization stack baseline

For context, the optimizations we've shipped today and where they apply:

| Optimization | Speedup | Status |
|--------------|---------|--------|
| `dt=1.0` (Izhikevich) | 2× | DEFAULT |
| `cfg.fast_spike_reset` | 1.2-1.5× | DEFAULT True |
| Three-factor GPU port | 2× | DEFAULT |
| STP disabled in chat training | **3.28×** | DEFAULT FLIPPED 2026-05-10 |
| `fp16_synapse_state` | 1.135× | opt-in, validated |
| `freeze_plasticity_during_reset` | TBD | opt-in, untested |
| Encoding-axis architecture | **~35×** at 64-word | recommended for vocab≥32 |
| **Cumulative current** | **~5-10× over original** | shipped today |

Additional Phase 1 optimization potential: ~5-10× more (FP16 throughout,
sparser density, kernel fusion, faster construction).

**Realistic best-case Phase 1 cumulative: ~25-50× over baseline.** At
this point, 256-512 word vocab is locally usable with day-to-day
continual learning.

## Provenance

- This addendum: `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
- Original master plan: `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`
- Perf optimization audit: `research/findings/2026-05-10-perf-optimization-audit.md`
- Encoding-axis findings: `research/findings/2026-05-10-encoding-axis-64word-3SEED-GO.md`
- VRAM ceiling probe: `research/findings/2026-05-10-vram-ceiling-probe-results.md`
- STP discovery: `research/findings/2026-05-10-stp-default-flip.md`
- STP reversibility: `research/findings/2026-05-10-stp-reversibility-3SEED.md`
