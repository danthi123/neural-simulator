---
status: live
type: finding
lane: one-brain/migration
date: 2026-08-27
---

# `prospective_memory`'s full-7 residual was a SIXTH co-residence seam, not FP accumulation — `cfg.per_region_inhibitory_seed` closes it, 7/7 GO

Status: GO. A guarded, DEFAULT-OFF `sim/` edit (`cfg.per_region_inhibitory_seed`) name-keys each region's
inhibitory-cell selection, closing `prospective_memory`'s organ-read residual. `onebrain_merge_verify --keys
all --seeds 42,43,44,100,101,102` now reads **7/7 organs GO, 6/6 seeds each** (pmem `read_maxerr` flips
0.06667 -> **0.0**, `answer_same=True` on every seed) — the full-7 one-brain migration gate CLOSES. <!--derived--> Flag OFF
is byte-identical to today (full-state SHA256 unchanged edited-vs-original + `tests/test_determinism.py`
11/11 passed). Artifact: `research/findings/raw/_onebrain_merge_full7_inhibseed_6seed.json`.

## The prior hypothesis (sub-ULP FP accumulation) was REFUTED by direct instrumentation

The 2026-08-27 dedup finding characterized pmem's residual as "believed to be a floating-point accumulation
seam." Read the substrate instead of theorizing further (CLAUDE.md's own rule): hooking `bridge.
_run_one_simulation_step` on both a pmem-ALONE pool and the full-7 MERGED pool and diffing every per-neuron
array of pmem's slice step-by-step found the first divergence at step 7 of the very first calibration probe
(cue-alone drive, before ANY synaptic input reaches the driven population) — `cp_conductance_g_e` on
`dlpfc_wm`, core=0.002 vs merged=0.0005, a clean **4x**, not a sub-ULP jitter. <!--derived--> Three follow-up instruments
each RULED OUT floating-point-order as the cause: (1) `cue_A`'s own V/u/firing trajectory, driven by pure
external current with zero synaptic input, was **bit-identical** merged-vs-alone over 30 steps (both arms
fire the identical 40/40 neurons on the identical periodic schedule); (2) the 27 pre-neurons wired into the
diverging `dlpfc_wm` target neuron carry **identical weights** and (3) **identical per-step firing** in both
arms at every traced step — yet the matvec's contribution to that neuron's `g_e` still differed 4x.

## The real cause — `RegionManager.initialize()`'s inhibitory-cell draw is NOT name-keyed

Same firing, same weights, same edges, different `g_e` -> the matvec must be routing SOME of those 27
pre-neurons to a different conductance channel (`g_e` vs `g_i`) in each arm. `bridge.cp_traits` (the array
the main synaptic matvec's E/I split reads, `cp_traits == cfg.inhibitory_trait_index`) is reassigned by
`inject_explicit_wiring`'s `output_inhibitory_indices` argument from `region_manager.inhibitory_indices()`.
Counting the SAME 27 pre-neurons' region-based inhibitory membership directly: **5 inhibitory (core) vs 8
inhibitory (merged)** — a mismatch among identically-firing, identically-wired neurons. `RegionManager.
initialize()` (`sim/regions.py`) draws `rng.sample(idx_list, n_inh)` per region from ONE shared
`random.Random(seed)` threaded through every region **in region-list order** — the same "one shared stream"
defect `build_wiring_plan`'s `per_region_wiring_seed` fixed for connectivity/weights, but never applied to
THIS draw. A region's inhibitory subset therefore depends on how much RNG the regions BEFORE it in the
(co-residence-dependent) list consumed. This is invisible to substrate-init byte-identity — thresholds, V, u,
and the Izhikevich a/b/C/d parameters are untouched — it only shows once a step actually runs, because it
changes which conductance CHANNEL a firing neuron's output routes through, not any per-neuron init value.

## The fix (additive, DEFAULT-OFF, byte-identical-when-off, guarded) — the SIXTH merge seam

- `sim/regions.py` — a new stride constant `_INHIBITORY_SEED_STRIDE` (a third, distinct large prime,
  alongside the existing wiring-region/pathway strides) and a `per_region_seed: bool = False` parameter on
  `RegionManager.initialize()`. When True, each region's inhibitory-cell sample uses `_wiring_substream`
  (the SAME stable-crc32-of-region-NAME substream mechanism `build_wiring_plan` already uses) instead of the
  shared stream, so the SAME subset of a region's neurons is chosen inhibitory regardless of co-residence.
  When False (default) the legacy shared-stream draw is unchanged bit-for-bit.
- `sim/config.py` — `per_region_inhibitory_seed: bool = False` (a new opt-in flag).
- `sim/bridge.py` `_initialize_region_manager` — threads `per_region_seed=getattr(cfg,
  "per_region_inhibitory_seed", False)` into `region_manager.initialize(...)`.
- `research/runners/onebrain_merge_framework.py` `_base_config` — `cfg.per_region_inhibitory_seed = not
  legacy` (ON for the real seams, OFF for the legacy discriminator), mirroring the other four seams.

## Proofs (all measured)

1. **FLAG-OFF byte-identity.** A multi-region net (three regions, `exc_fraction` 0.8/0.7/1.0, param-het,
   cross-region wiring) stepped 40 times with a random external drive, `cfg.per_region_inhibitory_seed` left
   at its default (unset), full-state SHA256 (V/u/thresholds/izh-params/traits/het-mask/conductances/firing +
   `cp_connections` data+indices+indptr): **identical** between the edited worktree and the same script run
   against the pre-edit code (`git stash`/`git stash pop` around the same three files) —
   `0f49f4ea06bfe1095901b81da203b22d8f226262da85628a3d70d3a23c2908f9` both times.
   `tests/test_determinism.py`: **11 passed** (a real CUDA GPU is present in this environment, so the
   GPU-only tests ran rather than skipped), exit code 0, 713.6s.
2. **Mechanism isolation (flag ON).** The same 27 pre-neurons feeding pmem's diverging `dlpfc_wm` target now
   show the SAME region-based inhibitory count merged-vs-core (verified via `region_manager.
   inhibitory_indices`), and pmem's `cp_conductance_g_e` trajectory at that neuron is bit-identical
   core-vs-merged from step 0 onward (was a clean 4x split at step 7 with the flag off).
3. **`--keys all` closure, 6-seed (42,43,44,100,101,102).** `prospective_memory` flips NO-GO -> **GO 6/6**
   (`read_maxerr=0.0` and `answer_same=True` on every seed — was 0.06667/mixed <!--derived-->). The other six organs
   (causal_whatif, comprehension, self_schema, source_provenance, curiosity, d6_multiref_wm) stay **GO 6/6**
   (no regression). Legacy discriminator diverges 6/6 (the seam is non-vacuous: `legacy_maxerr=118.6` on
   pmem's seed-42 slice with the seam forced off). **7/7 organs GO, `all_go: true`.** Artifact:
   `research/findings/raw/_onebrain_merge_full7_inhibseed_6seed.json`.

## Honest boundary — unchanged: the MIGRATION gate, not INTEGRATION

Byte-identity-in-isolation forbids the cross-region interaction that IS the one-brain goal; a pool with zero
cross-synapses is MIGRATED, not INTEGRATED (DESIGN §4, `onebrain_merge_verify`'s own disabled-scope line).
This closes the 7-organ MIGRATION gate cleanly (all seven Group-A organs now byte-identical merged-vs-
coresident, substrate-init AND organ-read); the functional-integration phase is the named next rung.

## A latent sibling (not fixed here, flagged for the record)

`RegionManager.initialize()`'s topographic-coordinate assignment (`_assign_coords`) uses the SAME
single-shared-stream pattern (`coord_rng`, threaded region-by-region in list order) for regions with
`coordinate_dim > 0`. No Group-A organ registered here uses topographic coordinates, so it did not surface
in this arc and was left untouched (minimal-fix scope) — but it is the same class of co-residence seam and
will need the identical `_wiring_substream` treatment the day a topographic organ joins a merged pool.

## Files changed

- `sim/regions.py` — `_INHIBITORY_SEED_STRIDE` + `RegionManager.initialize(..., per_region_seed=False)`.
- `sim/config.py` — `per_region_inhibitory_seed: bool = False` (additive flag).
- `sim/bridge.py` — `_initialize_region_manager` threads the flag into `region_manager.initialize(...)`.
- `research/runners/onebrain_merge_framework.py` — `_base_config` opt-in (`= not legacy`) + the legacy
  discriminator's explicit `cfg.per_region_inhibitory_seed = False`.
