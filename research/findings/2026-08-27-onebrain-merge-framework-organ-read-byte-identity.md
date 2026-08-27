---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# One-brain merge framework — GROUP A ORGAN-READ byte-identity (self_schema + d6, 6/6 GO)

Status: GO for the ORGAN-READ co-residence-invariance gate on the two tractable Group-A organs. This extends the
framework from SUBSTRATE-INIT byte-identity (the prior rung,
`2026-08-27-onebrain-merge-framework-groupA-batch-migration.md`) to running each organ's REAL production
read/judge pipeline on the merged pool and proving its output is byte-identical to running that organ
co-resident-alone on the same superset config — the pattern `_onebrain_twopool_organread_verify.py` used for the
4 core organs, now generalized into the declarative registry.

Artifacts (both carry a `tools.verdict.Verdict` `preconditions` block):
`research/findings/raw/_onebrain_merge_organread_all7_6seed.json` (the full `--keys all` 6-seed sweep) and
`research/findings/raw/_onebrain_merge_organread_groupA_6seed.json` (the focused 2-organ sweep).

## What now passes (6 seeds = 42,43,44,100,101,102)

- SUBSTRATE-INIT byte-identity: 7/7 Group-A organs, 42/42 organ-seeds (no regression from the prior rung).
- ORGAN-READ byte-identity: `self_schema` + `d6_multiref_wm`, both `read_maxerr == 0.0` on every seed AND
  `answer_same == True` on every seed (12/12 read-organ-seeds), verified co-resident with ALL 7 Group-A organs on
  ONE `SimulationBridge` (N=4968) — not just the two alone.
- Legacy discriminator diverges 6/6 (the seam-OFF pool's slices diverge merged-vs-coresident, so the byte-identity
  is NOT a vacuous all-zero compare).
- Param-het reconciliation load-bearing 24/24 (clearing the per-region param-het mask changes each param-het
  organ's Izhikevich params — the reconciliation is doing real work, not a vacuous all-zero het).
- Byte-identical-when-`shared=None`: an exact diff of both organs' standalone reads (this branch) vs the original
  main-checkout code is IDENTICAL over all 6 seeds — the `shared=` kwarg is a purely ADDITIVE change.

## The shared= pattern (additive, byte-identical when None)

Each organ class gained a `shared=None` kwarg. `None` -> the organ builds its own bridge EXACTLY as today. A
`MergedPool` -> the organ uses the injected substrate + its region slice: it reads `shared.bridge/.cfg/.xp`,
takes its dev-index map from the pool's region slices (`shared.idx(key)` -> the descriptor's `idx_fn`), uses the
pool's settle-to-rest snapshot (`shared.snap`), and wraps each read in `shared.read_isolation(key)` so a
co-resident organ's slice is restored after. `self_schema` is a bare-bridge read (`_run_trial`); `d6` threads
`shared=` through its `MultiSlotHold` core so the LOAD/HOLD/READ protocol runs on the pool slice.

`MergedPool` gained a `wire=True` mode: after the base build it rebuilds `cp_connections` from ONE
`build_wiring_plan(per_region_seed=True)` (every edge keyed on its endpoints' NAMES -> co-residence + order
INVARIANT) UNIONed with each descriptor's `explicit_wiring_fn` (self_schema's K assembly loops + member->attend),
runs each `post_inject_fn` (the frozen WS loop gate), then settles to a quiescent rest and snapshots (`self.snap`).
Both the MERGED and the CORESIDENT pools take this identical path, so a slice's WEIGHTS (not just its init arrays)
are byte-identical, which is the precondition the reads need.

## Two config reconciliations the reads forced (declared, not hidden)

1. Global noise off. `enable_conductance_noise` + `enable_ou_process` draw from a SINGLE global RNG stream in
   neuron-index order, so a neuron's noise depends on its ABSOLUTE index — inherently co-residence-DEPENDENT (an
   organ built second sits at a different offset). Making it invariant needs a per-neuron-seeded noise stream (a
   `sim/` engine edit, out of scope), so the frozen migration pool runs the noise OFF — a per-pool config decision,
   like the per-region threshold-het seam. Both organs declare the same value (no MergeConflict).

2. d6 under the param-het seam + explicit NMDA-mask membership. Two co-residence couplings had to be neutralized,
   both traced to a global engine behavior a non-het / non-per-region-nmda co-resident inherits ASYMMETRICALLY:
   (a) `nmda_ratio`/parameter-heterogeneity — once self_schema opts a region into per-region het, the engine
   applies params per-region, so a non-het d6 read the default ratio in the merged arm but a global override in the
   alone arm; making d6 `param_het=True` (the SAME name-keyed seam) makes the het state symmetric; (b) the
   per-neuron NMDA mask — the engine builds it the moment ANY region sets `enable_nmda=True` (self_schema's
   `workspace` does), after which regular NMDA applies ONLY to masked neurons and SILENTLY excludes d6; marking
   every d6 region `enable_nmda=True` (`region_flags`) restores d6's faithful GLOBAL-NMDA operating point AND makes
   its mask membership identical merged-vs-coresident. Both are declared reconciliations that leave d6's read
   MEANINGFUL (every referent still recovered, `all_recovered=True`), not tuning to force a number.

## Honest boundary — this is still the MIGRATION gate, not INTEGRATION

Byte-identity-in-ISOLATION forbids the cross-region interaction that IS the one-brain goal: a pool with zero
cross-synapses is MIGRATED, not INTEGRATED. Functional integration (cross-region edges under the F-gate) is the
named next rung. And the co-resident-alone baseline is the pool-built-alone organ on the superset config, NOT the
organ's pre-migration standalone — the claim is co-residence-invariance (exactly as the twopool proof compared
merged-4 vs the two production pools), not identity to the un-merged organ.

## Organ-read deferrals (5/7 — substrate-init GO, read needs a named seam; NONE blocked the batch)

Registered as data in `onebrain_merge_framework.GROUP_A_ORGANREAD_DEFERRED`:

- `curiosity` — neuromodulator-subsystem + plasticity seam (from_novelty->ASK excitability + a spiking-SNc RPE
  critic: enable_stdp + enable_reward_modulation + gabab CONFLICT with the frozen pool).
- `source_provenance` — neuromodulator-context-line seam (ctx_perceived/ctx_generated gate zero-init Hebbian
  traces; the `ProvenanceBrain` wrapper must accept an injected bridge; hebbian-encode conflicts with the frozen
  pool).
- `comprehension` — wrapper + operating-point seam (the read is tied to the `SpikingRoleCompetition` wrapper; its
  merge operating point wants dt=1.0 + homeostasis ON, which conflicts with the pool's global enable_homeostasis
  OFF).
- `prospective_memory` — stateful-wrapper seam (a `SFANmdaProspectiveMemory` homeostatic hierarchy with SFA +
  dendritic plateau + one-shot Hebbian FORMATION, read across MULTIPLE turns).
- `causal_whatif` — live-composer grounding + DA/STDP seam (the read enumerates events + moat-confirms answers
  against a live RFPhasorComposer, and the forward model trains temporal-order STDP + phasic-DA at build).

## Files changed

- `research/runners/onebrain_merge_framework.py` — `MergedPool.wire`/`.snap` + `_install_organ_read_wiring` +
  settle/snapshot; `OrganDescriptor.explicit_wiring_fn`/`.post_inject_fn`; self_schema + d6 read plumbing + the two
  config reconciliations; `GROUP_A_ORGANREAD_DEFERRED`.
- `research/runners/onebrain_merge_verify.py` — organ-read on the `wire=True` pools + an organ-READ `require` +
  the organ-read deferral print.
- `research/runners/self_schema_production_organ.py`, `.../d6_multiref_wm_production_organ.py`,
  `.../_multi_slot_binding_derisk.py` — the additive `shared=` kwarg (byte-identical when None).

NO `sim/` edit. All runs are tiny numpy nets (N<=4968) on the CPU.
