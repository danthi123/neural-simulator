---
status: live
type: finding
lane: variable-binding working memory (fresh-slot allocation — the multi-slot WM's host free-counter residual)
date: 2026-08-11
---

# Emergent neural-WTA fresh-slot ALLOCATION replaces the multi-slot WM's host free-counter — 6-seed GO

**Date:** 2026-08-11
**Runner:** `research/runners/_var_bind_emergent_wta_allocation_derisk.py` (reuse-by-import of the D3 slow-NMDA slot + the sparse barcode mint; numpy-CPU; **NO `sim/` edit**)
**Raw:** `research/findings/raw/_emergent_wta_alloc/emergent_wta_alloc_6seed.json` (+ `seed_{42,43,44,100,101,102}.json`)
**Status:** GO (6 seeds) on the de-risk gate — an emergent, entity-conditioned neural WTA allocates fresh slots with the host free-counter + host teaching-clamp REMOVED, matching the host reference on downstream recall. Honest scope below.

## Why (the residual BOTH banked lanes named)
The variable-binding WM is spiking (D3 slow-NMDA hold + Hebbian STP bind) and multi-slot (banked GOs). Its biggest remaining host shortcut is SLOT ALLOCATION: `_multi_slot_binding_derisk.py` uses a host `HebbianBinder` to assign each entity a stable local slot (a free-counter) and `write(reg, local)` to HOST-drive exactly that pool (the teaching-clamp). RUNG6e (`2026-07-13-...-freshslot-allocation-is-the-subproblem.md`) isolated the exact open sub-problem: with equal barcode drive "a winner emerges but it is NOISE-picked (not entity-specific)"; the delicate WTA "can't yet deliver a clean, entity-specific, high-rate winner" (measured blur ~0.31). Reproduced here: a graded multi-pool drive on `build_persistent_slot` gives winner-fraction ~0.31 (many pools fire, no clean winner).

## The mechanism (all runner-side host math on the substrate's own pools; NO `sim/` edit)
One bank of K=8 D3 slow-NMDA attractor pools sharing ONE FS (the multi-slot substrate, R=1; the allocation residual is WITHIN a bank). A NEW entity's barcode drives ALL pools through a fixed developmental-random projection P (K×64) → graded entity-conditioned external current. Two SELF-CALIBRATING competition mechanisms resolve the RUNG6e blur into a clean fresh-slot winner; neither names a pool:
- **(A) adaptive competition threshold** — a DOWN-RAMP (release-of-inhibition): a pooled subtractive inhibition common to all pools starts HIGH (every pool silent) and is released step-by-step; the FIRST pool to escape (highest total drive, shaped by the recurrent + heterogeneous substrate) is the winner, and the ramp STOPS the instant exactly one pool is active. The right release level DEPENDS on the per-entity drive margin, so no single hand-set cut works — the adaptive ramp tracks the operating point. The controller reads only the COUNT of active pools (a population statistic; Carandini-Heeger / the lever-3 feedback-inhibition motif), never a pool.
- **(B) adaptive occupancy excitability** — HTM boosting / Turrigiano homeostatic intrinsic plasticity: a per-pool `-boost_beta·used_k` depresses a pool that just latched, steering the next novel entity to a FREE pool. Occupancy lives in a neural excitability trace, NOT a host free-counter.

BIND/retrieve = a content-agnostic Hebbian fast weight `W[winner] += barcode/||barcode||` (the banked RUNG6c mechanism); re-presenting drives the pools with `W @ barcode` → the bound pool wins the SAME self-calibrating WTA → retrieve. The winner is READ FROM SPIKES (the pool the attractor LATCHED after the drive is removed), never a host argmax over the drive logits; the controller sets only a scalar inhibition. **No host free-counter, no host teaching-clamp picks the slot.**

## Result — 6-seed GO (seeds 42 43 44 100 101 102; K=8 pools, N=6 entities; slot-chance 0.125)

| arm | same-entity→same-slot (retrieve) | collision (distinct→distinct) | downstream recall | winner spike-fraction |
|---|---|---|---|---|
| **emergent_wta (candidate)** | **1.000** [min 1.000] | **0.000** [max 0.000] | **1.000** | 0.7593 |
| host_free_counter (ceiling) | 1.000 | 0.000 | 1.000 | 1.000 |
| lesion_selfcalib (fixed hand-set cut) | 1.000* | 0.4722 | **0.5278** | 0.4882 |
| noise-picked null (no entity-cond) | — | 0.000 | **0.16667 (=verb-chance)** | — |

Per-seed emergent recall = 1.000 and collision = 0.000 on ALL 6 seeds (6 distinct slots for 6 entities every seed); winner spike-fraction 0.63–0.90. `*` the lesion's retrieve is degenerate (Hebbian retrieve recovers the bound slot even when the allocation collided), so the lesion is read on RECALL, not retrieve.

Note (discriminating power): emergent + host recall are at CEILING (1.000) — at this task size (6 entities, 8 slots, near-orthogonal barcodes) a working allocator recalls perfectly, so recall alone cannot separate a good allocator from a lucky one. The DISCRIMINATION is carried by the collapsing controls (lesion recall 0.53, null recall at verb-chance) and by the metrics that DO vary across seeds (collision, winner spike-fraction) — recall=1.000 is meaningful only BECAUSE the lesion/null on the same task do not reach it. Pushing N toward K (the capacity edge) is where recall would drop and re-separate; that is the named harder regime, not this gate.

**Load-bearing proofs.** <!--derived--> (1) The self-calibration is load-bearing: the fixed-hand-set-threshold lesion collapses downstream recall 1.000→0.528 and drives collision 0.000→0.472 (mean) — a used-agnostic fixed cut lets distinct entities pile onto a favoured pool (RUNG6e's distinct=False). It is UNRELIABLE, not uniformly dead: it happens to mostly-work on seed 102 (recall 0.833) and collapses on the other five — which IS the argument for self-calibration (no single hand-set cut is reliable across seeds/entities). (2) The barcode-conditioning is load-bearing: the no-entity-conditioning null keeps the competition (distinct slots, collision 0.000) but the pools fill by intrinsic excitability, not identity → downstream recall falls to verb-chance (≈1/6). So the emergent allocation is ENTITY-conditioned, and the neural competition is what makes it distinct + clean.

## Honest scope (what is neural, what is host — the accepted de-risk boundary)
- **Host-computed (the named next rung, NOT this de-risk):** the barcode→pool projection is host math injected as external current (the SAME residual `write()` already uses), and the release-of-inhibition / occupancy controller are host math on the pools (the SAME accepted scope as the banked lever-3 competitive stabilizer). This is NOT self-organized (the projection is host-designed random weights) and NOT fully spiking (host-computed drive + a host scalar controller + a host-argmax-over-spikes read-out). The on-substrate spiking lateral-inhibitory / DA-gated realisation is the named next rung (RUNG6e's hard region-framework WTA engineering).
- **Neural (what the de-risk shows):** the WTA SELECTION (which pool wins) emerges from the spiking attractor + FS competition under the self-calibrated inhibition — the controller sets only a scalar from a population active-COUNT, never a pool — and the read is of the LATCHED spikes.
- **Residual 1 — winner spike-fraction 0.76 (not pristine one-of-K).** The latch is dominant but leaks ~24% to a runner-up; the read is argmax over post-competition spikes. A true lateral-inhibition WTA (the next rung) would sharpen it. Reported, not hidden.
- **Residual 2 — the EXACT slot index is noise-invariant on 4/6 seeds (noise-stability 1.000) but noise-co-determined on seeds 44/101 (0.167 / 0.667).** <!--derived--> On those seeds the barcode preferences are closer, so the specific assignment reshuffles across noise — yet it remains a valid DISTINCT, retrievable bijection every run (retrieve/recall 1.000). So the allocation is entity-conditioned + distinct + retrievable on every seed; a pure barcode→index function holds on 4/6. A stronger barcode→pool code would tighten the other two.

## Verdict
Emergent neural-WTA fresh-slot allocation **replaces the host free-counter + teaching-clamp** entity-specifically: distinct slots for distinct entities (collision 0.000/6 seeds), a stable retrievable address (retrieve 1.000), matching the host reference on downstream recall (1.000 vs 1.000), with the self-calibrating competition threshold AND the barcode-conditioning both load-bearing (lesion → recall 0.528 + collision 0.472; null → recall at verb-chance). <!--derived--> The self-calibration lesion re-collapses toward RUNG6e's distinct=False (it does NOT stay a clean winner — it collides), so RUNG6e's noise-picked wall is surpassed by the self-calibrating competition, within the accepted host-drive scope. Honest-negative was first-class and did not fire.

## Reproduce
```
for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
  research.runners._var_bind_emergent_wta_allocation_derisk --seeds $s \
  --out research/findings/raw/_emergent_wta_alloc/seed_$s.json & done ; wait
SIM_BACKEND=numpy python -m research.runners._var_bind_emergent_wta_allocation_derisk \
  --merge-from research/findings/raw/_emergent_wta_alloc/seed_*.json \
  --out research/findings/raw/_emergent_wta_alloc/emergent_wta_alloc_6seed.json
```
`cfg.seed` seeds the substrate (verified: build twice at one seed → identical firing thresholds; seed 42≠43 differ). Reuse-by-import; NO `sim/` edit.
