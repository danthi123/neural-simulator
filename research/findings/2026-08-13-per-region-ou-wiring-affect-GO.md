---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# Per-region OU-noise + wiring seed (guarded `sim/` edits) — AFFECT now merges onto the shared spiking pool BYTE-IDENTICALLY (GO, 6/6); the rung-2b affect boundary was the OU seam alone (neuromod measured NOT to diverge)

**Date:** 2026-08-13 · **`sim/` edits:** `cfg.per_region_ou_seed` + `cfg.per_region_wiring_seed` (both default-off,
additive) · **Runners:** `research/runners/_per_region_ou_wiring_engine_verify.py` (substrate proof),
`research/runners/_per_region_ou_affect_merge_verify.py` (real-organ read proof) · **Artifacts:**
`research/findings/raw/_per_region_ou_wiring_engine_6seed.json`,
`research/findings/raw/_per_region_ou_affect_merge_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`
so cp == numpy → bit-exact). Builds on rung 2b
(`2026-08-13-per-region-param-het-cluster-GO.md`), which NAMED these two seams as the next rungs.

## What this lands

Rung 2b migrated the metacog + pragmatic production organs onto the shared spiking pool byte-identically via
`per_region_parameter_heterogeneity`, but AFFECT stayed an honest partial (BOUNDARY): its mood-ladder read runs
with `enable_ou_process=True` (OU noise is a `size=n` per-step GLOBAL `cp.random.randn(n)` draw → a region's noise
slice is position-shifted) AND drives the global neuromodulator subsystem. Rung 2b MEASURED the OU seam (OU-on
co-resident delta ~1.5e2 vs OU-off 0.0) and NAMED per-region OU + per-region neuromod scoping as the next rungs. It
also named a THIRD position source for a fully-wired same-`region_manager` merge: `build_wiring_plan`'s shared-RNG
sparse-pathway ORDER dependence. This arc BUILT both guarded engine features, PROVED them byte-identical-when-off,
and MIGRATED the REAL affect production organ onto a co-resident pool byte-identically. This is an engine feature +
a de-risk of the affect merge — NOT a production flip (the flags are default-off; the organ is not yet built merged
in production / wired to `/api/brain-chat`).

## The `sim/` edits (additive, guarded, byte-identical-when-off)

- `sim/config.py`: `per_region_ou_seed: bool = False`, `per_region_wiring_seed: bool = False`.
- `sim/bridge.py` (OU): `_build_region_ou_streams` builds one PERSISTENT host RNG stream per brain region at
  `_initialize_ou_process_state` (name-keyed via a STABLE `zlib.crc32` hash + a per-region stride, keyed on
  `cfg.ou_seed` else `cfg.seed`); `_draw_ou_noise_samples` runs the legacy global `cp.random.randn(n)` draw FIRST
  (so global-RNG consumption is preserved bit-for-bit and any non-region neuron keeps its legacy value), then
  OVERWRITES each region's slice from its own stream. Streams persist across steps so the OU temporal correlations
  are preserved. A region's per-step OU realization is therefore invariant to its co-residents / absolute pool
  position. All three OU step sites (the two Izhikevich fused paths + the Python HH path) route through the helper.
- `sim/regions.py` (wiring): `build_wiring_plan(..., per_region_seed=False)`. When ON, each region's internal
  connectivity draw AND each pathway's draw use their OWN `random.Random` seeded from a stable `zlib.crc32` hash of
  the region / pathway name (keyed on `seed`), so synapse placement is invariant to co-residence ORDER; OFF keeps
  the single shared-stream order-dependent draw. `sim/bridge.py` threads
  `per_region_seed=getattr(cfg, "per_region_wiring_seed", False)` into the framework wiring build.
- **Why default-off is byte-identical:** the OU per-region streams are built only under
  `getattr(cfg, "per_region_ou_seed", False)` (else None → the legacy global draw stands bit-for-bit); the wiring
  guard defaults `per_region_seed=False` → the single shared `random.Random(seed)` is threaded exactly as today.

## GO — the engine features (substrate proof, 6/6 each)

`_per_region_ou_wiring_engine_verify.py`:

| check (6 seeds) | result | verdict |
|---|---|---|
| OU: region R's `cp_ou_current` trajectory BYTE-IDENTICAL alone-vs-co-resident (behind a spacer), flag ON | 6/6 | **GO** (position-invariant) |
| OU: same trajectory DIFFERS with flag OFF (worst off-delta 2.03e2) | 6/6 | confirms the OU seam |
| OU: determinism, co-resident pool built+stepped twice at one seed → identical | 6/6 | **GO** |
| WIRE: region-internal + pathway A→B placement BYTE-IDENTICAL regardless of co-residence order, flag ON | 6/6 | **GO** (order-invariant) |
| WIRE: same placements DIFFER with flag OFF | 6/6 | confirms the order seam |
| WIRE: determinism, plan built twice at one seed → identical | 6/6 | **GO** |
| OFF-path substrate (OU-stepped) + wiring-plan hash BYTE-IDENTICAL to HEAD (git-stash both `sim/` edits) | 6/6 | **GO** (zero regression) |

## GO — the REAL affect organ read BYTE-IDENTICAL on a co-resident pool (6/6)

`_per_region_ou_affect_merge_verify.py` builds the production `AffectProductionOrgan` (a full one-brain bridge with
the co-resident graded-affect ladder) STANDALONE vs with INERT (density-0, unwired) co-resident regions PREPENDED —
the whole brain (rf + faculties + ladder) shifts to a non-zero offset, the exact perturbation a shared pool
introduces, while the pads consume NO `build_wiring_plan` RNG (wiring byte-identical) and add NO cross-synapse. The
read is the organ's REAL production logic `read_differential` (the sign-aware neural ladder differential
rate(aff_pos_readout) − rate(aff_neg_readout) through the `affect_out` gate), run with `enable_ou_process=True` and
`per_region_ou_seed` + `per_region_parameter_heterogeneity` + `per_region_threshold_heterogeneity` ON.

| check (6 seeds) | flag ON | flag OFF | verdict |
|---|---|---|---|
| affect read max delta merged-vs-co-resident (positive + negative appraisal) | **0.0** 6/6 | 5.6e-4 … 5.0e-3 (diverges) 6/6 | **GO** (byte-id 6/6) |
| faculty alive (merged): positive appraisal holds +differential, negative holds −differential | 6/6 | — | **GO** |

The flag-ON read is byte-identical merged-vs-co-resident (max delta 0.0) for BOTH signs, all 6 seeds; the flag-OFF
read DIVERGES, confirming the OU (+ init) seams were load-bearing. On the merged pool the graded staggered-bistable
ladder still represents signed valence (positive ≈ +0.05, negative ≈ −0.05, all seeds).

## The rung-2b affect boundary was the OU seam ALONE — neuromod measured NOT to diverge (honest refinement)

Rung 2b named TWO open affect seams: OU noise AND the global neuromodulator subsystem. This arc MEASURES that the
neuromodulator subsystem is NOT a divergence source: `NeuromodulatorManager.step` reads region firing
position-independently (`rm.indices`) and sets concentrations by name, so given byte-identical firing (which
per-region OU + param/threshold deliver) the neuromod effects are byte-identical too. The affect organ's read —
which DOES drive the appraisal neuromodulators — is byte-identical 6/6 with only the OU seam closed (plus the
already-landed param/threshold seams). So affect closes on the OU seam alone; no per-region neuromod scoping was
needed. This REFINES the rung-2b boundary to a GO (measured, not asserted).

## No regression (flags OFF = today)

- `pytest tests/test_determinism.py -q` → **9 passed**.
- OFF-path substrate hash (OU-stepped `cp_ou_current` + membrane + thresholds) AND the wiring plan BYTE-IDENTICAL to
  HEAD, all 6 seeds (git-stash `sim/bridge.py`+`sim/config.py`+`sim/regions.py`, rerun `--mode off`; empty diff).
- `brain_chat_tui --smoke` JSON **byte-identical** to a stashed pre-change baseline (raw diff empty).
- `build_one_brain`'s DEFAULT path (no `coresident_regions` / new flags) is unchanged: the additive kwargs
  (`coresident_regions`/`per_region_param_het`/`per_region_thresh`/`per_region_ou`, defaults None/False) are the
  merge instrument, default-preserving (mirrors the metacog/pragmatic builders).
- Rung-1 + metacog/pragmatic regression guards stay 0.0 (they set neither new flag, so this edit is inert for them).

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._per_region_ou_wiring_engine_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_per_region_ou_wiring_engine_6seed.json
SIM_BACKEND=numpy python -m research.runners._per_region_ou_affect_merge_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_per_region_ou_affect_merge_6seed.json
```

## Honest scope / non-claims

- `on_shared_substrate: the OU + wiring engine seams named by rung 2b are RESOLVED behind guarded flags; the AFFECT
  production read is byte-identical under co-residence on one co-stepped pool (the 3rd cluster organ) / wired (to
  /api/brain-chat): NO / on_by_default: NO (the flags are opt-in, default-off; the organ is not yet flipped to
  build merged in production) / scaffold_retired: none.` Functional read-outs only; no phenomenal claim.
- The co-resident affect organ's regions are present + co-stepped at a non-zero offset but the pads are INERT and
  add NO cross-synapse — this proves the affect read is INVARIANT to co-residence on one pool (the same
  no-cross-synapse scope rungs 1/2b established), NOT a genuine cross-region affect synapse (a later step).
- `per_region_wiring_seed` is proven at the substrate level (a region's / pathway's placement is order-invariant)
  and is the clean fix for the fully-wired same-`region_manager` both-organ merge ORDER seam rung 2b named; it is
  NOT yet exercised end-to-end in a two-fully-wired-organ production merge (the next rung).
- The appraisal injection (per-word valence → neuromodulator concentration) is a declared host scaffold, unchanged;
  the load-bearing spiking part is the ladder read through `affect_out`.
