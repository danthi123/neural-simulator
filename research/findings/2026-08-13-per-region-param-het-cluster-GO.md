---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# Per-region parameter-heterogeneity (guarded `sim/` edit) — the metacog + pragmatic production organs now merge onto the shared spiking pool BYTE-IDENTICALLY (GO, 6/6 each); affect is a MEASURED BOUNDARY (an OU-noise + neuromodulator seam param-het does not close)

**Date:** 2026-08-13 · **`sim/` edit:** `cfg.per_region_parameter_heterogeneity` (default-off, additive) +
`SimulationBridge._overwrite_region_scoped_parameter_heterogeneity` · **Runners:**
`research/runners/_per_region_param_het_engine_verify.py` (substrate proof),
`research/runners/_per_region_param_het_cluster_verify.py` (real-organ read proof) · **Artifacts:**
`research/findings/raw/_per_region_param_het_engine_6seed.json`,
`research/findings/raw/_per_region_param_het_cluster_6seed.json` (6 seeds 42/43/44/100/101/102,
`SIM_BACKEND=numpy` so cp == numpy → bit-exact). Builds on rung 2
(`2026-08-13-merge-production-rung2-BOUNDARY.md`), which NAMED this engine feature as the next rung.

## What this lands

Rung 2 mapped that the metacog / pragmatic / affect production organs cannot join the shared spiking pool
byte-identically because their graded rate code REQUIRES `enable_parameter_heterogeneity`, whose per-neuron
Izhikevich jitter is drawn as ONE `size=n` sample per parameter from the GLOBAL RNG over the whole pool → a
co-resident organ's slice is position-shifted (a valid but DIFFERENT seeded param-het than standalone). The named
fix is a per-region param substream EXACTLY mirroring the landed `per_region_threshold_heterogeneity` (`fb27b610`).
This arc BUILT it, PROVED it byte-identical-when-off (opt-in, default-off), and MIGRATED the two cluster organs
whose ONLY divergent global flag was param-het. Affect has a second + third global-per-step seam (OU noise +
neuromodulators), measured here, so it remains an honest partial. This is an engine feature + a de-risk of the
cluster merge — NOT a production flip: the flag is default-off and the organs are not yet built merged in
production (not `wired` to `/api/brain-chat`).

## The `sim/` edit (additive, guarded, byte-identical-when-off)

- `sim/config.py`: `per_region_parameter_heterogeneity: bool = False`.
- `sim/bridge.py`: after the legacy global `_apply_parameter_heterogeneity` draw runs (so global-RNG consumption
  is preserved bit-for-bit), when the flag is ON, `_overwrite_region_scoped_parameter_heterogeneity` OVERWRITES
  each brain region's slice of the jittered Izhikevich/HH parameter arrays with a draw from a REGION-SCOPED host
  RNG keyed on a STABLE `zlib.crc32` hash of the region name (process-independent) + a per-parameter stride, each
  substream reset to position 0. A region's param-het is therefore invariant to its co-residents / its absolute
  pool position. It reproduces the backend-neutral draw+clip math exactly and always uses the host RNG (like the
  threshold sibling), so region-scoped values are identical on numpy and cupy.
- **Why default-off is byte-identical:** the overwrite call site is gated on
  `getattr(cfg, "per_region_parameter_heterogeneity", False)` (default False) AND `region_manager is not None`, so
  the block is SKIPPED entirely and the legacy global draw stands bit-for-bit. Neurons NOT owned by a region keep
  their legacy value even when ON.

## GO — the engine feature (substrate proof, 6/6)

`_per_region_param_het_engine_verify.py` (region R alone at offset 0 vs behind a spacer X at offset 30 — the exact
perturbation a shared pool introduces):

| check (6 seeds) | result | verdict |
|---|---|---|
| R's Izhikevich param-het slice (a/b/d/C) BYTE-IDENTICAL alone-vs-co-resident, flag ON | 6/6 | **GO** (position-invariant) |
| same slice DIFFERS with flag OFF (the bug the flag fixes) | 6/6 | confirms position-dependence |
| determinism: build the co-resident pool twice at one seed → identical | 6/6 | **GO** |
| OFF-path full param+threshold hash BYTE-IDENTICAL to HEAD (git-stash `sim/` edit) | 6/6 | **GO** (zero regression) |

## GO — metacog + pragmatic reads BYTE-IDENTICAL on a co-resident pool (6/6 each)

`_per_region_param_het_cluster_verify.py` builds each organ STANDALONE vs with the OTHER cluster organs' regions
PREPENDED as INERT (density-0, unwired, no-pathway, exc-only) co-residents on ONE co-stepped pool — shifting the
organ to a non-zero offset while consuming NO `build_wiring_plan` RNG (so its own connectivity stays byte-identical)
and adding NO cross-synapse. Reads use the organs' real production logic (metacog: the balance-of-evidence margin
via `_run_trial`; pragmatic: the graded RSA L1 belief via `_rsa_recursion`).

| organ (6 seeds) | read max delta, flag ON | read max delta, flag OFF | faculty alive (merged) | verdict |
|---|---|---|---|---|
| metacog (balance-of-evidence confidence) | **0.0** 6/6 | 7.5e-4 … 2.4e-3 (diverges) 6/6 | 5/6 | **GO** (byte-id 6/6) |
| pragmatic (scalar-implicature RSA belief) | **0.0** 6/6 | 4.4e-3 … 5.3e-2 (diverges) 6/6 | 6/6 | **GO** |

The flag-ON read is byte-identical merged-vs-co-resident (max delta 0.0) for BOTH organs, all 6 seeds; the flag-OFF
read DIVERGES, confirming param-het was the load-bearing seam. Faculty-alive on the merged pool: metacog's
high-evidence margin exceeds its low-evidence margin (5/6 — the one miss, seed 43, is a pre-existing
narrow-dynamic-range property of the balance read, NOT a merge effect: byte-id 6/6 means the standalone read is
identical, so it would miss standalone too); pragmatic represents the some→not-all implicature 6/6.

## BOUNDARY — affect (measured, 6/6): an OU-noise + neuromodulator seam param-het does not close

Affect's mood-ladder read runs with `enable_ou_process=True` (bridge.py: `cp.random.randn(n)` is a size-n
per-step GLOBAL draw, so a region's noise slice is position-shifted) AND drives the global neuromodulator
subsystem. Even with `per_region_parameter_heterogeneity` ON, region R's OU-driven trajectory DIFFERS alone
vs co-resident, while the OU-OFF control at the same offset is byte-identical — isolating OU as the open seam:

| check (6 seeds) | result |
|---|---|
| R's OU-driven trajectory delta, co-resident, param-het ON (OU on) | 1.46e2 … 2.03e2 (>0, position-dependent) 6/6 |
| OU-OFF control delta at the same offset, param-het ON | **0.0** 6/6 (param-het closes the INIT seam) |
| affect organ uses the global neuromodulator subsystem | True 6/6 |

So affect cannot join the shared pool byte-identically on param-het alone → honest partial (BOUNDARY). It needs a
per-region OU draw + per-region neuromodulator scoping (distinct engine features), the mapped next rungs.

## No regression (flag OFF = today)

- `pytest tests/test_determinism.py -q` → **9 passed**.
- OFF-path substrate hash BYTE-IDENTICAL to HEAD, all 6 seeds (git-stash `sim/bridge.py`+`sim/config.py`, rerun
  `_per_region_param_het_engine_verify --mode off`; the two `OFF_HASHES` lines are identical).
- `brain_chat_tui --smoke` JSON **byte-identical** to a stashed pre-change baseline (raw diff empty).
- The two de-risk builders' DEFAULT path (`build_metacog_bridge` / `build_rsa_bridge` with no new kwargs) is
  byte-identical to HEAD (substrate arrays + production reads hashed, git-stash comparison — empty diff). The
  additive kwargs (`coresident_regions`/`per_region_param_het`/`per_region_thresh`, defaults None/False) are the
  merge instrument, default-preserving.
- Rung-1 regression guard STILL 0.0: the surprise + world-model pair stays byte-identical merged-vs-co-resident
  (`_onebrain_merge_rung1_verify.py`; they set `enable_parameter_heterogeneity=False`, so this edit is inert for
  them, and the new flag defaults off).

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._per_region_param_het_engine_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_per_region_param_het_engine_6seed.json
SIM_BACKEND=numpy python -m research.runners._per_region_param_het_cluster_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_per_region_param_het_cluster_6seed.json
```

## Honest scope / non-claims

- `on_shared_substrate: the param-het engine seam rung 2 named as the cluster blocker is RESOLVED behind a guarded
  flag; metacog + pragmatic production reads are byte-identical under co-residence on one co-stepped pool (2 of the
  3 cluster organs) / wired (to /api/brain-chat): NO / on_by_default: NO (the flag is opt-in, default-off; the
  organs are not yet flipped to build merged in production) / scaffold_retired: none.` Functional read-outs only;
  no phenomenal claim.
- **The co-resident organs' regions are present + co-stepped but NOT simultaneously independently wired-and-read.**
  Wiring a second organ on the SAME `region_manager` adds a THIRD, cross-synapse-free position source:
  `build_wiring_plan` (regions.py) samples its sparse pathways from ONE shared `random.Random(seed)` in
  region-then-pathway ORDER, so a second wired organ shifts the first's pathway sampling. The clean fix is a
  per-organ-plan-then-remap combine (build each organ's plan on its own layout, remap indices to the shared
  offset, inject once) OR a per-region wiring-plan seed — the mapped next rung. This rung proves the PARAM-HET
  seam (the one rung 2 named as the cluster blocker) is closed, and each organ's read is invariant to co-residence
  on one pool — the same no-cross-synapse scope rung 1 established.
- **affect** is a real BOUNDARY here (OU + neuromod), not deferred: it is measured and its exact open seams named.
  **causal / curiosity** (stdp+reward+neuromod) and **comprehension** (per-region `dt`, impossible byte-exact)
  remain separate arcs, unchanged from rung 2.
