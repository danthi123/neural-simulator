# Warm-start across vocab tiers: raw trained-weight transplant is NOT viable (connectivity diverges 88% at n_lang scaling); the clean path is grow-by-append, which is real engineering

**Date:** 2026-05-28 ~00:30 EDT
**Status:** Cheap-first falsification NEGATIVE on the simple approach; corrected path identified. Pure engineering finding (no bar, no science verdict). Owner-prompted design question ("can completed V=320 run warm-start V=640?").

## The idea (owner, 2026-05-28)

Since each vocab tier is a strict superset of the previous (D8 V=640 = D7 V=320 + 64 new words per category), reuse the completed lower-tier trained bridges as the starting point for the next tier instead of cold-training every concept from scratch. Potential ~40-50%% training-time saving per tier (only train the NEW concepts), plus it doubles as a continual-learning capability test.

## What transfers cleanly (verified)

The per-concept INPUT ENCODING is byte-identical across tiers. The scale params were (by design) chosen to hold n_active=61 and stride=64 constant across D6 V=160 -> D7 V=320 -> D8 V=640. Verified: for all 64 shared concepts, orthogonal_drive_pattern places the active lang_input neurons at IDENTICAL positions in D7 (n_lang=4096, V=64) and D8 (n_lang=8192, V=128). 0/64 mismatches at both production and smoke scale.

Bridge construction is also deterministic: same builder + same seed twice -> byte-identical connectivity (3054 == 3054 edges).

## What does NOT transfer (the obstacle)

Raw trained-weight transplant requires the sparse lang_input->pool CONNECTIVITY (which specific edges exist) to match across tiers. It does not:

| | D7 V=64 (n_lang=4096) | D8 V=128 (n_lang=8192) |
|---|---|---|
| APPLE pool lang_input->pool edges | 3054 | 6282 |
| Jaccard overlap | -- | **0.116 (12%)** |

Each concept pool connects to ~30%% of ALL lang_input neurons (the text_input_to_motor_density=0.30 pathway). When lang_input doubles (4096->8192), both the edge COUNT doubles AND the specific random draw of which neurons diverges (the connectivity generator consumes RNG proportional to neuron count, so by the time any given pool's connectivity is drawn, the RNG state differs between the two tiers even at the same base seed). Copying D7's trained weights would land 88%% of them on edges that don't exist in D8.

Probe: research/findings/raw/direction_8_warm_start_connectivity_probe.py (CPU-only).

## Why the obvious fixes are blocked

1. **Per-pool deterministic connectivity** (seed each pool's connectivity by (bridge_seed, pool_name), independent of n_lang) WOULD make shared pools identical across tiers. But it requires modifying build_biological_brain_regions, the PROTECTED builder reused byte-unchanged across pillars n=98/n=105/n=108/n=109. Modifying it risks the entire validated pillar chain. Not acceptable under the reuse-byte-unchanged discipline.

2. **Constant n_lang across tiers** does not help on its own: even at fixed n_lang, adding pools changes total RNG consumption during construction, so per-pool connectivity still diverges.

3. **Replay-based warm-start** (imprint shared concepts via teacher drive instead of weight copy) works through the normal training pathway so it sidesteps connectivity matching, but it IS training -- no obvious time saving over cold-training the shared concepts.

## The clean path: grow-by-append (real engineering)

The correct warm-start is to GROW the D7 bridge into D8 rather than build a fresh larger bridge:
1. Load the completed D7 bridge (its 64 shared pools + connectivity unchanged -- it literally IS the trained D7 state).
2. Append 64 new concept pools + extend lang_input by 4096 new neurons [4096:8192].
3. The shared concepts keep their exact D7 encoding ([0:4096]) AND their exact D7 connectivity (untouched). Only new structure is added.
4. Train only the 64 new concepts.

This sidesteps the connectivity-divergence problem entirely because the shared structure is preserved byte-for-byte (loaded, not rebuilt). It is exactly the "start small, grow as it learns" pattern in the project's auto-growth roadmap (docs/plans/2026-05-10-auto-growth-design.md), and sim/auto_growth.py already has a TierLadder + TierPromoter + weight-transfer scaffold for the v14/v16 motor-pool arch.

BUT it is genuine engineering, not a quick smoke:
- Bridge array-resize (all cp_* GPU state arrays must grow) + CSR extension + region-table append on an already-initialized SimulationBridge. Delicate; touches bridge internals.
- auto_growth.py targets the Phase 1.4 BRANCH A motor-pool arch, not the D-arc cross-bridge dedicated-pool arch -- it needs adaptation.
- Needs its own validation: does a grown bridge reproduce a cold-built bridge for the shared concepts (no perturbation of shared pools by the append)? does the grown-then-trained result match cold-trained within noise? Without that, a warm-started pillar could be inflated/deflated vs the honest cold-start baseline.

## Honest recommendation

- The simple "save snapshot + transplant weights" version the owner suggested does NOT work with the current protected builder (this finding).
- The correct version (grow-by-append, adapting auto_growth.py to the cross-bridge arch) is worth building for the project's growth goal, but it is a multi-step engineering task with its own validation arc, NOT a side-experiment that fits in the D7-production wait window.
- Cold-start should remain the canonical pillar validation regardless (independent replication per tier is what makes each pillar trustworthy; warm-start across tiers would compound any artifact).
- Cheapest near-term speed lever remains the per-event compute reduction already built (D8 --use-fp16 + --stim-steps-per-event 50), which needs no builder changes and no warm-start.

## Files

- Probe: research/findings/raw/direction_8_warm_start_connectivity_probe.py
- Auto-growth scaffold (existing, needs adaptation): sim/auto_growth.py + sim/lineage.py
- Auto-growth design doc: docs/plans/2026-05-10-auto-growth-design.md

## Discipline

No bar; engineering finding. No protected/frozen/moat module touched. CPU-only probe; no GPU contention with the in-flight D7 production. Cheap-first falsification saved a multi-day build of a transplant that would have silently corrupted 88%% of transferred weights.

## Addendum (2026-05-28): cross-pool connectivity sharpens the grow-by-append tradeoff

Owner asked whether grow-by-append precludes connections between old (previous-tier) and new pools. Architecture check (text_minimal_isolation.py) clarifies:

**Two kinds of cross-pool connection in the base cross-bridge arch:**
1. **Inhibitory winner-take-most (EXISTS, required):** each concept pool's FS interneuron inhibits all OTHER pools in its category (line ~897, "each pool's FS inhibits OTHER pools"). This lateral inhibition is what makes concepts discriminable.
2. **Excitatory pool->pool (does NOT exist in base arch):** direct concept->concept excitatory pathways are opt-in only (v16 --enable-direct-verb-to-motor); D8 does not enable them. Cross-bridge composition is computed downstream (FHRR/probe), not through synapses.

**Implication for grow-by-append:** it does NOT preclude old<->new connections -- but the REQUIRED WTA inhibition is itself an old<->new connection that must be added. To discriminate a new concept from an old one, new pools' FS must inhibit old pools and vice versa (all-pairs WTA). The moment that old<->new inhibition is wired, the OLD pools receive new inhibitory input -> their dynamics shift -> they are no longer byte-for-byte frozen. So "freeze old, train only new" is not clean; some re-equilibration training is needed, eroding the time saving. This is the stability-plasticity dilemma in miniature and is the honest reason cold-start (full all-pairs WTA from the start) stays the trustworthy baseline.

**What survives grow-by-append regardless:** the project's VALIDATED associative capability (90%% multitag cue retrieval, engram stim-recall) works through engram tags (stored sets of co-firing neuron indices), NOT plastic pool->pool synapses. Tags span old + new pools freely at the activity level -- "apple (old) relates to horse (new)" is reachable via the engram mechanism without new synaptic pathways.

**Where it bites (future):** a future arch wanting LEARNED excitatory concept->concept pathways would need old<->new edges + training, which makes old pools participate in new learning -> catastrophic-forgetting risk -> exactly the problem the project's hippocampal consolidation (Phase 1.3, validated no-catastrophic-forgetting) is the biological answer to.

Net: grow-by-append is biologically correct (real cortex doesn't add concepts in isolation -- new concepts must join the WTA competition and can associate with old ones via fast hippocampal binding) and remains worth building as its own arc with CLS/consolidation, NOT a quick freeze-and-extend optimization. Owner elected to keep D7/D8 on the cold-start path for now.
