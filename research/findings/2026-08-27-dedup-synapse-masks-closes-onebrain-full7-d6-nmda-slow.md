---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# The one-brain full-7 `--keys all` wall is a duplicate-edge per-synapse-MASK MISALIGNMENT in `inject_explicit_wiring` (NOT a co-residence-dependent nmda_slow RNG draw) — `cfg.dedup_synapse_masks` closes d6; pmem is a separate seam

Status: GO for d6_multiref_wm (the measurable target), honest PARTIAL for the pool. A guarded, DEFAULT-OFF
`sim/` edit (`cfg.dedup_synapse_masks`) makes the per-synapse routing masks co-residence-invariant. With it ON,
`d6_multiref_wm`'s slow-NMDA (`nmda_slow`) receptor tagging is byte-identical merged-vs-coresident and d6's
organ-read CLOSES (`onebrain_merge_verify --keys all`: d6 flips NO-GO -> GO, 6/7 organs GO). Flag OFF is
byte-identical to today (full-state SHA256 unchanged + determinism suite 9/9). `prospective_memory` remains
NO-GO for a SEPARATE reason, proven independent of this fix below.

NO-EXTERNAL-NEEDED: this is not a biology-mechanism wall — it is an internal engine bug, localized by direct
reproduction to `inject_explicit_wiring` building per-synapse masks from an un-deduped `keyed` list while
`cp_connections` is deduplicated by scipy's `csr` canonical format (`sum_duplicates` merges coincident
`(pre,post)` coordinates). No external neuroscience literature applies; the grounding fact is scipy's documented
sparse canonical-format behavior (logged to the integration-lane external-search record this session).

## The prior hypothesis was REFUTED by reproduction (read the substrate, don't theorize)

`2026-08-27-deterministic-conductance-matvec-hardening-and-the-real-onebrain-full7-wall.md` localized the wall
to d6's `nmda_slow` self-excitation being drawn at `density=0.9` from a co-residence-DEPENDENT global RNG (a
"different subset tagged nmda_slow"). Reproduced directly, that is NOT the mechanism: with the plan's
per-region-seamed pathway draw UNCHANGED, **d6 built ALONE has a PERFECTLY aligned mask (0 misaligned of 10819
`nmda_slow`), and only the MERGED pool's mask is wrong (5148 of 10819 land on non-self edges — including
`w_k->fs` density-0.6 AMPA edges tagged `nmda_slow`).** A receptor-draw difference cannot tag an AMPA
`w_k->fs` edge as `nmda_slow`; a MASK MISALIGNMENT can. The draw is fine; the per-synapse mask ordering is the
bug.

## The real cause — `keyed` (pre-dedup) vs `cp_connections` (post-`sum_duplicates`)

`inject_explicit_wiring` builds `cp_connections` via `coo->tocsr()+sum_duplicates()`, which MERGES duplicate
`(pre,post)` edges, so `nnz = unique coords`. But it builds every per-synapse mask/gate (nmda_slow, gaba_b,
coincidence, graded, stp_disabled, plastic, and the plasticity/transmission gate index maps) from a `keyed`
list sorted by `(pre,post)` whose length is the UN-deduped plan-edge count, via `np.fromiter(..., count=nnz)` /
positional alignment. When the plan has `D` duplicate coords, `len(keyed) = nnz + D`, so from the first
duplicate coord on, `keyed[i]` addresses a DIFFERENT synapse than `cp_connections.data[i]` — every mask entry
after it is shifted.

This is co-residence-DEPENDENT: `d6` ALONE has **0 duplicate edges** (mask aligned); the merged pool has
**1415** (`len(keyed)=253532` vs `nnz=252117`), all from OTHER organs — `prospective_memory`'s `c2d`/`d2c`
attractor-loop `explicit_wiring_fn` overlapping the base `pathway_cortex_ctx_to_dlpfc_wm`/`..._to_cortex_ctx`
RegionPathways (468 + 459), and `comprehension`'s `cue_monitor` overlapping `pathway_cortex_ctx_to_rel_A/B`
(256 + 232). Those duplicates shift d6's `nmda_slow` mask, so d6's AMPA-suppression suppresses a DIFFERENT
subset of synapses merged-vs-coresident — the full-7 wall. It is the SAME class the framework fixed for the OU
noise + threshold/param-het draws, but at the per-synapse ROUTING layer, and it is a MASK-ALIGNMENT bug, not an
RNG-draw bug.

## The fix (additive, DEFAULT-OFF, byte-identical-when-off, guarded)

- `sim/config.py` — `dedup_synapse_masks: bool = False` (a new opt-in flag).
- `sim/bridge.py` `inject_explicit_wiring` — when ON, COLLAPSE the `(pre,post)`-sorted `keyed` list's
  adjacent-duplicate runs to ONE entry BEFORE any mask is built (weights are already summed in
  `cp_connections`; only the routing attributes are aggregated — OR of the boolean flags, routing-dominant
  receptors `gaba_b`/`nmda_slow`, first-non-empty gate names). Then `len(keyed) == nnz`, so every downstream
  mask aligns synapse-for-synapse with `cp_connections.data` regardless of co-residence. Aggregation is
  order-stable, so a duplicate coord resolves identically alone vs co-resident.
- `research/runners/onebrain_merge_framework.py` `_base_config` — `cfg.dedup_synapse_masks = not legacy` (ON
  for the real seams, OFF for the legacy discriminator), mirroring `per_region_threshold_heterogeneity` +
  `deterministic_transpose_matvec`.

When OFF (default) the collapse block is never entered ⇒ byte-identical to today (a plan with no duplicate
edges is already `len(keyed)==nnz`, so the collapse is a no-op even when ON).

## Proofs (all measured; numpy, small)

1. FLAG-OFF byte-identity (the critical safety property). A net exercising `inject_explicit_wiring` WITH
   duplicate `(pre,post)` edges + every routing mask (nmda_slow + gaba_b + coincidence + graded + stp + plastic
   + gates), flag OFF, 40 steps, full-state SHA256 (membrane / recovery / firing / all conductances / all masks
   / `cp_connections` data+indices+indptr): `ea5a3991…32b0` IDENTICAL edited (worktree) vs original (main
   checkout, flag absent). `tests/test_determinism.py`: 9 passed, 2 skipped (cupy-only).
2. Mask alignment (flag ON). d6's `nmda_slow` tags AGREE 10819/10819 merged-vs-core (0 merged-only, 0
   core-only); within-pool 0 misaligned in BOTH arms (was 5148 misaligned in the merged arm).
3. `--keys all` closure, 6-seed (42,43,44,100,101,102). `d6_multiref_wm` GO 6/6 (read_byte 6/6,
   substrate_byte 6/6, het-loadbearing 6/6) — flips from NO-GO. The other five (causal_whatif, comprehension,
   self_schema, source_provenance, curiosity) stay GO 6/6 (NO regression). Legacy discriminator diverges 6/6
   (the seam is non-vacuous). 6/7 organs GO; the seed-42 verdict reproduces an earlier INDEPENDENT-process run
   bit-for-bit (d6 GO, pmem X). Artifact: `research/findings/raw/_onebrain_merge_full7_dedup_6seed.json`.

Evidence: `research/findings/raw/2026-08-27-dedup-synapse-masks-onebrain-full7-evidence.json`.

## `prospective_memory` residual — a SEPARATE seam, proven independent of this fix

pmem stays NO-GO (read_maxerr 0.06667 on `fire_min`; the rendered answer flips (T,T,T) alone -> (F,F,F)
merged). It is NOT the mask this fix corrects: the misaligned d6 mask marks ONLY d6-pre synapses (verified), so
no pmem synapse is ever tagged `nmda_slow`. DECISIVE test: forcing `cp_nmda_recurrent_synapse_mask=None` on the
merged pool (disabling the AMPA-suppression rebuild entirely) leaves pmem's read_maxerr UNCHANGED at 0.06667.
So pmem's divergence is independent of the `nmda_slow` routing — a separate co-residence seam (d6's region
presence / total-N; pmem runs a dt=0.5 long spiking integration where a sub-ULP per-step delta amplifies over
hundreds of steps). The prior finding itself flagged that pmem's exact in-step coupling was "NOT fully
isolated." This finding does NOT close pmem and does NOT flip any production default.

## Honest boundary — unchanged: the MIGRATION gate, not INTEGRATION

Byte-identity-in-isolation forbids the cross-region interaction that IS the one-brain goal; a pool with zero
cross-synapses is MIGRATED, not INTEGRATED. This is a co-residence-invariance fix for the per-synapse routing
masks at the shared pool operating point, not the functional-integration phase.

## Files changed

- `sim/config.py` — `dedup_synapse_masks: bool = False` (additive flag).
- `sim/bridge.py` — the guarded `keyed`-collapse in `inject_explicit_wiring` (default OFF ⇒ byte-identical).
- `research/runners/onebrain_merge_framework.py` — `_base_config` opt-in (`= not legacy`).
