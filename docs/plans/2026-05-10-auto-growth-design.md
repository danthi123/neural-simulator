# Auto-growth design — sim grows autonomously with the user

**Date:** 2026-05-10
**Status:** DESIGN — Phase A is the practical starter, others are
research-engineering scope
**Trigger:** User (2026-05-10) — "is there a way to run the sim
starting small then growing autonomously as it learns, instead of
pre-sizing to max expected scale?"

---

## Vision

A user starts with a tiny 4-word agent on a basic machine. As they
interact with it, the agent learns new words, grows new neurons, adds
new motor sub-pools, and eventually outgrows local hardware and
migrates to the cloud — automatically. Matches the biological reality
of childhood neural development.

The differentiator vs static LLMs: **the sim's structure itself
grows with usage. There is no "training cutoff."**

## Four sub-phases (A through D)

Each sub-phase is independently shippable. Phase A is the practical
starter; D is the long-tail "infrastructure ready when needed."

---

### Growth Phase A: tier promotion via checkpoint reload

**Scope:** ~1 week of focused work.
**Status today:** All pieces exist; needs wiring.
**Value:** Immediate UX of "agent grows over time"; discrete tier
jumps but feels organic.

#### Implementation

```python
class TierPromoter:
    """Monitors training accuracy and promotes the bridge to the next
    vocabulary tier when mastery is achieved.

    Tier ladder: 4 -> 8 -> 12 -> 16 -> 24 -> 32 -> 48 -> 64 -> 96 ->
                 128 -> 256 (each shipped today; see text_eval.py)
    """
    TIERS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]

    def __init__(self, threshold=0.90, consecutive_required=3):
        self.current_tier = 4
        self.threshold = threshold
        self.consecutive_required = consecutive_required
        self.consecutive_passes = 0
        self.next_tier_arch = self._derive_next_arch()

    def step(self, eval_accuracy: float, bridge: SimulationBridge) -> bool:
        """Called after each eval. Returns True if promotion triggered."""
        if eval_accuracy >= self.threshold:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
        if self.consecutive_passes >= self.consecutive_required:
            self._promote(bridge)
            return True
        return False

    def _promote(self, bridge):
        """Save current bridge, build next-tier, transfer weights."""
        old_tier = self.current_tier
        new_tier_idx = self.TIERS.index(old_tier) + 1
        if new_tier_idx >= len(self.TIERS):
            return  # at max tier
        new_tier = self.TIERS[new_tier_idx]
        ckpt = f"bridges/auto_promote_{old_tier}_to_{new_tier}.simstate.h5"
        bridge.save_checkpoint(ckpt)
        new_arch = self._arch_for_tier(new_tier)
        new_bridge = self._build_bridge(new_arch)
        self._transfer_weights(old_bridge=bridge, new_bridge=new_bridge,
                                 old_tier=old_tier, new_tier=new_tier)
        self.current_tier = new_tier
        # caller swaps bridge reference

    @staticmethod
    def _arch_for_tier(tier: int) -> dict:
        """Per the encoding-axis discovery (2026-05-10), pick optimal arch
        for each tier."""
        return {
            4:   {"n_lang": 2048,  "n_motor": 500,  "n_motor_fs": 60},
            8:   {"n_lang": 4096,  "n_motor": 1000, "n_motor_fs": 120},
            12:  {"n_lang": 4096,  "n_motor": 2000, "n_motor_fs": 240},
            16:  {"n_lang": 4096,  "n_motor": 2000, "n_motor_fs": 240},
            24:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
            32:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
            48:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
            64:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
            96:  {"n_lang": 16384, "n_motor": 2000, "n_motor_fs": 240},
            128: {"n_lang": 16384, "n_motor": 2000, "n_motor_fs": 240},
            256: {"n_lang": 16384, "n_motor": 4000, "n_motor_fs": 480},
        }[tier]

    def _transfer_weights(self, old_bridge, new_bridge,
                            old_tier, new_tier):
        """Map weights from old motor pools to new (larger) pools.

        Algorithm (refined 2026-05-11 after surveying bridge.py +
        regions.py):

        At the architecture level, synonyms are NOT distinct sub-
        regions — they are patterns within a continuous motor pool
        that emerge during STDP/embodied-Hebbian training. The
        promotion algorithm is therefore "grow the pool, copy what's
        trained, random-init the rest":

        For each pathway involving a growing region (e.g. motor_N at
        tier1 has 500 neurons -> tier2.1 has 1000):

        1. Get old + new region indices via region_manager:
              pre_old  = old_bridge.region_manager.indices(pre_region)
              post_old = old_bridge.region_manager.indices(post_region)
              pre_new  = new_bridge.region_manager.indices(pre_region)
              post_new = new_bridge.region_manager.indices(post_region)
              n_pre_old, n_pre_new = len(pre_old), len(pre_new)
              n_post_old, n_post_new = len(post_old), len(post_new)

        2. Read old weights into dense block via CSR slicing:
              old_W = old_bridge.cp_connections[
                  post_old[:, None], pre_old[None, :]
              ].toarray()  # shape (n_post_old, n_pre_old)

        3. Map first n_pre_old / n_post_old new neurons 1:1 to old:
              new_W[0:n_post_old, 0:n_pre_old] = old_W (copy trained)
              new_W[n_post_old:, :] = random_init  (new post-neurons)
              new_W[:, n_pre_old:] = random_init  (new pre-neurons)

        4. Install via set_pathway_weights:
              new_bridge.set_pathway_weights(
                  pathway_name,
                  pre_indices=expanded_pre_array,
                  post_indices=expanded_post_array,
                  weights=new_W.flatten(),
                  add_missing=True,  # CSR may not have all edges yet
              )

        Pathways to transfer (per Phase 1.4 BRANCH A arch):
            - language_input -> motor_{N,E,S,W}
            - motor_{N,E,S,W} -> language_output
            - motor_{N,E,S,W} -> motor_FS_{N,E,S,W} (FS lateral inhib)
            - motor_FS_{N,E,S,W} -> motor_{N,E,S,W}
            - (hippo pathways if enabled)

        Biology rationale: real cortex grows by adding pyramidals with
        random initial connections, while existing pyramidals keep
        their trained patterns (Lichtman 2014; Holtmaat 2009 review).
        Sleep replay (Phase 1.3) consolidates the new neurons over a
        few REPL sessions.

        Random init params (match original arch's prior):
            weight_mean = old pathway's weight_mean
            weight_jitter = old pathway's weight_jitter
            density = old pathway's density (use rng with new seed)
        """
        ...
```

#### Trigger logic

Insert into the training loop (in `bio_three_factor.run_three_factor`):
```python
if tier_promoter and (event_idx % eval_interval == 0):
    eval_acc = evaluate_w_to_a_baseline_synonym(bridge)["accuracy"]
    if tier_promoter.step(eval_acc, bridge):
        print(f"\n🎉 PROMOTED to tier {tier_promoter.current_tier}!", flush=True)
        # Caller is responsible for swapping the bridge reference
        bridge = tier_promoter._build_next_bridge()
```

#### UX
- User runs `chat_repl --auto-grow` or starts with low tier
- Periodic console output: "Mastered 8-word vocab; promoting to 16-word"
- 3D viz could highlight new sub-pops being added
- Bridge auto-saved at each tier promotion (allows rollback if next
  tier struggles)

**Testing plan:**
1. Unit tests on `TierPromoter._transfer_weights()` (deterministic
   weight-copy logic)
2. Integration test: small dummy bridge promotes 4-word → 8-word →
   16-word; verify weights flow correctly
3. Smoke test on real bridge: start at 4-word, train 100 events, eval,
   confirm promotion fires

---

### Growth Phase B: within-tier structural plasticity

**Scope:** ~2-3 weeks (mostly validation, infrastructure exists).
**Status today:** `cfg.enable_structural_plasticity` exists but is
disabled in chat configs.
**Value:** Biology-realism + connection refinement.

#### What it does

Real synapses form and prune over a lifetime. The sim's
`update_pruning()` step handles this when enabled:
- Activity-dependent synaptogenesis: high-co-firing pairs grow new
  synapses
- Survival score: each synapse has a "fitness" updated by usage
- Low-survival synapses pruned (set to alive=False, weight=0)
- Re-allocation: new synapses fill those pruned slots

#### Why disabled today

`bio_three_factor.run_three_factor` sets:
```python
cfg.enable_structural_plasticity = False
```

To save compute during the focused binding-task training. The plasticity
gate doesn't help that specific test; it adds per-step cost without
improving the final metric.

#### Phase B work

1. **Re-enable** in chat configs (opt-in flag first)
2. **Tune parameters**:
   - `cfg.struct_plast_activity_bias` (0.0 to 1.0, currently 0.0)
   - Pruning threshold
3. **Validate**: does enabling improve final binding quality? Hurt it?
   Multi-seed at 8-word and 16-word with structural plasticity ON vs OFF
4. **Decision rule**: if structural plasticity improves or no-ops on
   binding, default to ON for biology realism + future-proofing. If it
   hurts, keep off and document as deferred.

#### Implementation

```python
# In bio_three_factor.run_three_factor:
cfg.enable_structural_plasticity = enable_structural_plasticity  # new arg
cfg.struct_plast_activity_bias = struct_plast_activity_bias
```

CLI: `chat_speak_synonym_demo --structural-plasticity`.

---

### Growth Phase C: online neurogenesis / pool expansion

**Scope:** ~1-2 months of focused engineering.
**Status today:** Infrastructure has hooks but they're not used.
**Value:** True online growth — pool size grows with vocabulary.

#### What it does

User says: "Learn the word 'porpoise' meaning N."
- Current state: capacity used (8 words bound to 4 actions; pool full)
- New behavior:
  1. Detect capacity exhausted
  2. Allocate N=125 new neurons in motor_N
  3. Wire them to language_input via structural plasticity
  4. Trigger embodied-Hebbian co-firing for "porpoise" + motor_N
  5. New sub-pop forms organically

#### Why this is hard

The bridge uses **CSR sparse matrices** for connectivity. CSR is
**not appendable** efficiently. Growing the pool means:
- Re-build CSR with new entries → O(nnz) per growth event
- OR pre-allocate with `growth_factor` headroom → caps growth

Current sim uses pre-allocation via `gpu_config.synapse_capacity_growth_factor`
(default 1.5). So capacity = nnz × 1.5 is allocated at init. This works
for moderate growth; beyond that, full re-build needed.

#### Phase C work breakdown

1. **Detection**: REPL-level "no spare sub-pop" check
2. **Pool expansion**: allocate new neurons in existing region
3. **Connectivity expansion**: pre-allocated slots + structural plasticity
   fills them with relevant connections
4. **CSR re-build trigger**: when pre-allocated capacity exceeded, do
   a periodic re-build (slow but rare)
5. **Index management**: region indices must stay consistent through
   growth events

#### Performance edge cases

- During CSR re-build: ~30-60 sec stall in training/inference
- Should happen rarely (every Nth vocab word, e.g. every 50th word)
- User-visible: brief "growing" pause

---

### Growth Phase D: hardware migration orchestration

**Scope:** ~2 weeks for the orchestration infra.
**Status today:** Checkpoint save/load works.
**Value:** No manual intervention when sim outgrows local hardware.

#### What it does

1. Bridge monitors VRAM utilization + steps/sec
2. When VRAM > 80% threshold AND steps/sec < 5 (slow + tight):
   - Snapshot bridge state to local checkpoint
   - UX prompt: "Your agent has grown beyond your local hardware.
     Migrate to cloud H100? [Y/n]"
3. On approval:
   - Upload checkpoint to cloud storage (S3 / R2)
   - Spin up H100 instance (Vast.ai, RunPod, Lambda)
   - Resume training in cloud
   - User interacts via cloud API endpoint
4. Optional: nightly sync back of trained checkpoint for local resume

#### Defer until Phase C demonstrates need

Cloud orchestration is its own project (provider APIs, billing,
networking). Best built **after** Phase A+B+C demonstrate the
sim's growth trajectory genuinely warrants cloud migration.

---

## Recommended build order (REVISED 2026-05-10 per user direction)

User confirmed (2026-05-10): focus on growth capabilities urgent;
defer Phase C+D until we actually need them. We're staying local
for a while; cloud migration can wait until we hit actual saturation.

**Build order:**

0. **NEW PRIORITY: Bridge Lineage Manager** (~1 week) — persistent
   continuous-learning state. Foundation for everything else; see
   `docs/plans/2026-05-10-bridge-lineage-design.md`. Pairs naturally
   with Phase A.

1. **Now (immediate value)**: Phase A — tier promotion via checkpoint
   reload. ~1 week. Gives "agent grows over time" UX immediately.
   Integrates with lineage (each promotion is a lineage growth event).

2. **Next**: Phase B — within-tier structural plasticity. ~2-3 weeks
   (validation-bound).

3. **Defer significantly**: Phase C — online neurogenesis. ~1-2 months.
   Needs Phase A+B running smoothly first. Per user (2026-05-10):
   "we intend to run fully locally for a while; many pending tests/
   work/implementations are higher priority."

4. **Defer significantly**: Phase D — hardware migration. ~2 weeks.
   Per user: only build when we actually need it. Local-first
   commitment means this is far-future.

## Biological grounding notes

- **Synapse formation**: real cortex grows synapses continuously, peaks
  in childhood (~700/sec at age 2), declines but never stops
- **Pruning**: ~50% of synapses pruned during adolescence — keeps
  high-utility connections, drops noise
- **Neurogenesis**: limited to specific regions in adult brain (dentate
  gyrus, olfactory bulb) — our sim's neurogenesis is more aggressive
  than biology but still grounded in the same principle
- **Lifelong learning**: real brains never freeze. The auto-growth
  pipeline matches this directly.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Tier promotion at wrong time (incomplete mastery) | High threshold (90%) + K consecutive passes (3) |
| Cross-tier weight transfer loses information | Save checkpoint before promotion; allow rollback |
| Structural plasticity destabilizes binding | Validate via multi-seed before enabling default |
| Online neurogenesis CSR rebuild stalls training | Pre-allocate with `growth_factor` headroom; rebuild rarely |
| Hardware migration disrupts user session | Snapshot + resume; user sees "migrating..." progress bar |

## Open questions for future arcs

- Should tier promotion accumulate KNOWLEDGE (full vocab history) or
  just adapt arch (reset some learned weights)?
- How does episodic memory (hippocampus) interact with growth events?
- What's the role of sleep replay across tier promotions?
- Multi-modal growth: does adding vision require new neurons or
  re-purposing existing motor pool?

## Provenance

- This design: `docs/plans/2026-05-10-auto-growth-design.md`
- Master plan addendum: `docs/plans/2026-05-10-MASTER-PLAN-strategic-addendum.md`
- Existing infra:
  - `sim/config.py:enable_structural_plasticity`
  - `sim/config.py:struct_plast_activity_bias`
  - `sim/bridge.py:update_pruning`
  - `sim/bridge.py:_synapse_capacity` (growth_factor)
  - `research/runners/chat_repl.py` (--save-bridge / --load-bridge)
  - `text_eval.get_synonym_groups` (4-256 word tiers)
