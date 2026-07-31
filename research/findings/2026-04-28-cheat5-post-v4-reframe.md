---
type: finding
status: live
date: 2026-04-28
mechanism: bg-cross-projections
---

# Cheat #5 reframe — ON HOLD pending biology buildout

**Date:** 2026-04-28 (afternoon, post-option-1 NO-GO + patch-matrix sanity check)

**Supersedes:** the "closed by design" framing in [`2026-04-28-cheat5-v4-results.md`](2026-04-28-cheat5-v4-results.md). The v4 results stand; the conclusion drawn from them was too quick.

## Summary

After option 1 (structural plasticity) NO-GO and patch-matrix (option 2) showing high-variance partial signal under proper multi-goal evaluation, cheat #5 is **ON HOLD pending a multi-cluster biology buildout**, not closed. Cross-projections aren't fundamentally broken — they're under-constrained. Real BG carves them via structural plasticity + closed-loop teaching + D1/D2 asymmetry + cholinergic plasticity gating + thalamo-cortical feedback. Our reduced model is missing all of this scaffolding.

## What changed today

### Multi-goal eval (the methodological correction)

All prior cheat-5 NO-GOs (v1, v2, v3.1, v4) used a single goal change at step 300, then 1500 stable steps. The user observed that this is a "static adult after one transition" test — not the "dynamic adult with rapid action-pattern switching" scenario that cross-projections would actually be useful for. Switched to `--goal-schedule multi` (4 phases × 450 steps, 3 transitions).

Under multi-goal:
- v3 baseline (no cross-projections): **7.08 ± 0.12** (n=3) — solid, low-variance
- Option 1 (cross + structural pruning): **22.46 ± 4.84** (n=2, seed 42 hung) — catastrophic
- Patch-matrix (density 0.25, no pretraining): **8.76 ± 2.54** (n=3) — high variance, seed 44 actually beat baseline at 5.88

### What the patch-matrix variance pattern reveals

Per-phase std for patch-matrix:
- Phase 0 (goal (6,6)): std 0.45
- Phase 1 (goal (1,6)): std 0.46
- **Phase 2 (goal (1,1)): std 2.09** ← the "topology luck" signal
- Phase 3 (goal (6,1)): std 0.22

Phase 2 corresponds to the (1,6)→(1,1) transition, which needs cross-action couplings the random topology seed=0 includes for some eval seeds and misses for others. This is *exactly* the pattern predicted by the "cross-projections need scaffolding" hypothesis: a sparse random topology gets lucky on some action-transitions and unlucky on others, with no mechanism to refine the choice.

### What "scaffolding" means

Real BG cross-action structure is shaped by multiple interacting biological mechanisms our current substrate doesn't have:

- **Structural plasticity** (Cluster E, partly tried in option 1): axon pruning + synaptogenesis carve which cross-pairs survive based on experience. We added pruning; alone it's insufficient.
- **D1/D2 plasticity asymmetry** (Cluster B): D1 LTPs under +DA / LTDs under −DA; D2 inverts. Lets cross-projections to D1 vs D2 encode complementary "do X" / "don't do Y" signals.
- **Striatal FSIs** (Cluster B): millisecond-scale broadcast inhibition; sharper WTA than MSN-MSN lateral.
- **Cholinergic interneurons / TANs** (Cluster B): ACh-gated plasticity windows. Real BG only consolidates synapses when ACh says "now's the time."
- **Closed thalamo-cortical loop** (Cluster A): the missing teaching signal — without feedback, cross-projections can't be told "your contribution helped/hurt."
- **Compartmentalized DA** (Cluster C): per-action DA pulses instead of global scalar. Lets each cross-projection learn target-specific weights.
- **Sequence-aware learning via hippo/PFC → striatum** (Cluster D): episodic context for action transitions.

## Reframe

**Cheat #5 is ON HOLD, not closed.** The honest framing:

> Cross-projections (cortex_X → str_Y for X ≠ Y) require a complete striatal microcircuit, a closed BG loop, and a properly-structured DA system to behaviorally pay off. Our reduced model has none of these. We've validated that connectivity-only changes (option 1: dynamic structural plasticity; option 2: sparse static topology) are insufficient on their own — both produce high-variance behavior, with option 2 occasionally beating baseline (one seed at 5.88) but no consistent improvement.
>
> Closing cheat #5 requires building out the surrounding biology systematically. This is a multi-month research program, organized as cluster-by-cluster biology additions per the [cheat-5 real-options survey](../../docs/plans/2026-04-28-cheat5-real-options-survey.md).

## What ships from this batch

- **No code reverts.** Patch-matrix support (`--cross-projection-density`, `--cross-projection-topology-seed`) and structural pruning (`--enable-structural-pruning` plus `bridge.update_pruning`) remain as opt-in infrastructure. Both will be re-tested under each cluster as it lands.
- **Recommended flagship config unchanged.** Still v3 lateral inhibition + perception arc + curriculum, no cross-projections. Will only flip when a cluster combination gets cross-projection cheat-5 closure consistently.
- **Cheat #5 status flipped from "CLOSED" to "ON HOLD pending biology buildout"** in CLAUDE.md, SCIENCE_ROADMAP, INDEX, CHANGELOG, memory. The closure-by-design framing was too conservative; cross-projections deserve a full biology stack before we declare them off-axis.

## Updated decision policy

Each cluster (A through G+) gets:
1. **Plan + design doc** (citation-grounded once textbook catalog lands).
2. **Implementation** behind opt-in flags — flagship behavior unchanged when off.
3. **Tier 1-3 validation** for the cluster's own biological correctness (does the cascade behave more like real BG on standalone microcircuit benchmarks?).
4. **Cheat-5 re-eval** under multi-goal, both with and without cross-projections, to see whether the cluster's additions shift the cross-projection failure mode.

We accumulate clusters until either:
- Cross-projections perform consistently better than v3 baseline under multi-goal across a 6-seed validation → cheat #5 closes for real.
- We've added every reasonable cluster (A-G+) and cross-projections still don't pay off → cheat #5 is genuinely off-axis at this fidelity level; closure-by-design becomes the final answer (with overwhelming evidence rather than the premature 3-attempt closure).

Either outcome is a valid scientific result.

## Next steps

1. **Cluster B design doc** (D1/D2 asymmetry as the smallest first step, plus striatal FSIs and TANs). [`2026-04-28-cluster-b-striatal-microcircuit-design.md`](../../docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md) — TBD.
2. **Wait for textbook catalog** (parallel session) to ground design docs in citations.
3. **Implement Cluster B** under TDD per writing-plans.
4. **Re-run patch-matrix + Cluster B together** under multi-goal as the first cluster validation.
5. Loop on next cluster.

## Files

- v3 baseline multi-goal: `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{163ab1,4ba3a9,516136}.json`
- Option 1 multi-goal: `research/findings/raw/g11_bg/g11_seed{43,44}_flagship_{2aec01,180852}.json` (seed 42 hung — `f95e36` log only)
- Patch-matrix multi-goal: `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{686be3,805227,7f1376}.json`
- Aggregator: `scripts/analyze_cheat5_v4.py` (handles v4-style runs; new aggregator for the cluster era TBD)
- Survey: [`docs/plans/2026-04-28-cheat5-real-options-survey.md`](../../docs/plans/2026-04-28-cheat5-real-options-survey.md)
- This reframe: this doc.
