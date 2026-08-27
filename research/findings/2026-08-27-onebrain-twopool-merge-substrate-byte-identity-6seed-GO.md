---
type: finding
status: live
lane: onebrain-merge
date: 2026-08-27
mechanism: onebrain-twopool-merge
---

# One-brain merge — production POOL #1 + POOL #2 (the FOUR core cortical organs) share ONE bridge byte-identically at the substrate-init level (6-seed GO)

**Date:** 2026-08-27 · **Runner:** `research/runners/_onebrain_twopool_merge_derisk.py` · **Artifact:**
`research/findings/raw/_onebrain_twopool_merge_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`).
**NO `sim/` edit** (reuse-by-import + existing per-region seams). De-risk, NOT a production flip.

## The gap this closes on the substrate axis

Production runs TWO SEPARATE merged pools: pool #1 (`onebrain_merge_production.MergedSubstrate`,
`BRAIN_ONEBRAIN_MERGE`) = D2 SURPRISE + E2 WORLD-MODEL (`enable_parameter_heterogeneity=False`, homeostasis-based);
pool #2 (`onebrain_merge_production2.MergedSubstrate2`, `BRAIN_ONEBRAIN_MERGE2`) = E1 METACOG + D-PRAGMATIC
(`enable_parameter_heterogeneity=True`, frozen, NMDA-on). `2026-08-13-onebrain-second-pool-SCOPED.md` scoped pool #2
as a SEPARATE pool from pool #1 precisely because their GLOBAL configs conflict (param-het ON vs OFF), so "one
substrate" for the four core cortical organs was still CO-RESIDENCY of two pools. This runner de-risks the merge of
both pools onto ONE `SimulationBridge` (N=2034 = surprise 1056 + world-model 528 + metacog 290 + pragmatic 160).

## The reconciliation — the param-het ON/OFF conflict is a PER-REGION MASK, not a wall

The five pool #1/pool #2 config deltas each map to a per-region seam already on `main` (no engine change):

| conflict | pool #1 | pool #2 | reconciled by |
|---|---|---|---|
| param-het | global OFF | global ON | global OFF + per-region `BrainRegion.enable_heterogeneity=True` on metacog/pragmatic ONLY (mask); `per_region_parameter_heterogeneity=True` (name-keyed) |
| homeostasis | organs ON | frozen OFF | global OFF + per-region `BrainRegion.enable_homeostasis=True` on surprise/world-model (diffbuilder pattern) |
| NMDA | off | metacog ON | global ON + per-region `BrainRegion.enable_nmda` → only metacog workspace/meta_schema carry it |
| GABA_B | inert (max=0) | off | `enable_gabab=True, gabab_conductance_max=0.0` (inert for all) |
| wiring / thresholds | global | name-keyed | `per_region_wiring_seed=True` + `per_region_threshold_heterogeneity=True` (both name-keyed = co-residence-ORDER-invariant) |

The load-bearing insight: `_overwrite_region_scoped_parameter_heterogeneity` (`bridge.py:3477`) already supports a
per-region MASK — with the global flag OFF it overwrites ONLY the masked (`enable_heterogeneity=True`) slices with a
name-keyed draw and leaves every unmasked slice at the non-jittered preset. So a param-het-ON organ (metacog) and a
param-het-OFF organ (surprise) coexist on ONE bridge with each byte-identical to its standalone self. The
production2 scoping named this conflict as the blocker; the mask dissolves it.

## Result — 6/6 GO (substrate-init byte-identity)

| criterion | 6 seeds | verdict |
|---|---|---|
| ONE shared pool (all 16 regions, one `cp_membrane_potential_v`, N=2034 = Σorgans) | 6/6 | GO |
| determinism (build merged twice at one seed → all init arrays identical) | 6/6 (maxerr 0.0) | GO |
| **per-organ INIT byte-identity, merged (all 4) vs co-resident (organ alone, same superset cfg)** | **6/6 (max delta 0.0)** | **GO** |
| param-het MASK load-bearing (metacog/pragmatic jitter, surprise/world-model untouched) | 6/6 | GO |
| legacy DISCRIMINATOR (seams OFF → merged-vs-coresident diverges) | 6/6 (89–134) | GO |

- **INIT byte-identity is EXACT 0.0** for BOTH pool-1 and BOTH pool-2 organs across all 13 per-neuron init arrays
  (`cp_neuron_firing_thresholds`, `cp_membrane_potential_v`, `cp_recovery_variable_u`, `cp_izh_a/b/C/c_reset/
  d_increment/vpeak/vt/vr`, and the two per-region gate masks), every region, every seed. Each organ's substrate is
  invariant to its three co-residents.
- **The param-het mask is DOING WORK** (anti-vacuity): forcing the mask off shifts metacog/pragmatic Izhikevich
  params by 63–87 while leaving surprise/world-model at 0.0 — the mask genuinely jitters the pool-2 organs and
  correctly leaves the pool-1 organs at the preset.
- **The byte-identity is NOT vacuous** (discriminator): with the per-region seams OFF (legacy global param-het +
  global wiring) the SAME merged-vs-coresident compare DIVERGES 89–134 on every organ — the seams are what close it.

## Honest scope (what is and is NOT claimed)

- **CLAIM: the four-organ SUPERSET config is internally byte-consistent** — each organ's substrate INIT is
  co-residence-invariant on ONE reconciled bridge (the STRUCTURAL merge gate, same gate the 2-organ pool used in
  `2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md`).
- **NOT claimed:** (1) the organ READ pipelines (`SurpriseProductionOrgan.judge`, `WorldModelProductionOrgan`,
  `MetacogProductionOrgan`, `PragmaticProductionOrgan`) run byte-identically on the merged pool — this runner does
  not exercise the reads or the POST-BUILD topographic wiring (surprise block-diagonal, metacog assembly loops).
  (2) answer-preservation vs the two CURRENT production pools — the merged superset config differs from each pool's
  native config (homeostasis approach, RNG stream), so raw rates may differ while classifications are preserved (the
  characterized cost of a shared config, per pool #1's flip finding). Both are the named FOLLOW-ON rung.
- **Not a production flip.** No `BRAIN_*` flag change; production keeps two pools. Per `docs/TERMS.md`, "closed"
  requires production integration — this is a 6-seed GO de-risk, the substrate half of merging the two pools.
- **Functional read-outs only**; no phenomenal claim.

## The follow-on (parallelizable next rung)

Full organ-READ byte-identity + answer-preservation on the four-organ pool: reuse each production organ's `shared=`
read path (as pool #1's `MergedSubstrate` / pool #2's `MergedSubstrate2` already do), install the post-build
topographic wiring (block-diagonal / assembly loops) with per-synapse gain-0 protecting the frozen pool-2 edges from
surprise's Hebbian `hebbian_max_weight=45` clip (the primitive `2026-08-14-onebrain-parser-on-pool-GO.md` used), and
add per-organ read isolation (full-snapshot-restore). That is a larger build (belongs on the pool/GPU), gated by the
same byte-identity + faculty-alive + answer-preservation panel the pool #1/#2 flips used.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_twopool_merge_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_twopool_merge_6seed.json
```
