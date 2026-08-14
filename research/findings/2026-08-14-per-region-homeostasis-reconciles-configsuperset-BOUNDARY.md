---
type: finding
status: live
lane: gap#5
date: 2026-08-14
mechanism: one-brain-merge
---

# Per-region homeostasis RESOLVES the config-superset homeostasis conflict (comprehension 6/6) — and the mapped `sim/` primitive ALREADY EXISTED; a SECOND surprise-read residual blocks full-cell GO

**Date:** 2026-08-14 · **Verdict:** Stage-1 GO (the per-region-homeostasis primitive) · Stage-2 BOUNDARY-REFINED
(the homeostasis conflict is reconciled — comprehension 6/6 — but the merge is 1/6, blocked by an independent
surprise-read robustness residual, not by homeostasis) · **Backend:** numpy (bit-exact CPU) · 6 seeds
(42,43,44,100,101,102) × 5 cells.
**Refines** `research/findings/2026-08-14-onebrain-configsuperset-production-merge-BOUNDARY.md` (closes its
homeostasis axis; re-maps its residual).

**Runner:** `research/runners/_one_brain_merge_configsuperset_production_derisk.py` (new `0.5:PR` cell) ·
**Artifact:** `research/findings/raw/_one_brain_merge_configsuperset_perregion_6seed.json` ·
**Reproduce:**
```
SIM_BACKEND=numpy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m \
  research.runners._one_brain_merge_configsuperset_production_derisk \
  --seeds 42,43,44,100,101,102 --cells 0.5:True,0.5:False,1.0:True,1.0:False,0.5:PR \
  --out research/findings/raw/_one_brain_merge_configsuperset_perregion_6seed.json
```

## Headline 1: the mapped engine feature was already in the substrate — NO `sim/` edit was needed

The BOUNDARY mapped its unblock as a `sim/` feature to BUILD: "a per-region `enable_homeostasis` — each
`BrainRegion` opts in/out of intrinsic-homeostatic threshold adaptation." **That primitive already exists.**
`BrainRegion.enable_homeostasis` (`sim/regions.py:171`, built 2026-06-08 for a deterministic-nav MSN-D1 critic) is
EXACTLY it: when True a region's neurons use the adapted `cp_neuron_firing_thresholds` as their spike threshold AND
adapt them every step EVEN WHEN global `cfg.enable_homeostasis` is False; when False (default) they use the fixed
`cp_izh_vpeak`. The whole path is already wired: the mask build (`bridge.py:2079-2094`, mirrors the per-region NMDA
mask), the 3-branch spike-threshold selection (`bridge.py:9221-9228`), the update gate (`bridge.py:10717-10776`).
The BOUNDARY conflated this with the DIFFERENT `per_region_homeostasis_isolation` flag (which only freezes idle
regions) and missed the actual opt-in primitive. **The surpass is done WITHOUT touching the protected engine:**
`git diff sim/` is empty; the merge is reconciled by USING the existing primitive (global `enable_homeostasis=False`;
surprise `_S` regions opt IN; role regions keep the default OFF). Runner-only change, two additive default-preserving
edits.

## Headline 2: per-region homeostasis reconciles the COMPREHENSION side fully — the homeostasis conflict is resolved

`go_by_cell` = {dt0.5_True: 0, dt0.5_False: 0, dt1.0_True: 0, dt1.0_False: 0, **dt0.5_PR: 1**}/6.

| cell (dt, homeo) | GO/6 | comp byte-id | comp AUC≥.80 | comp ans | surp byte-id | surp ans | cross LB |
|---|---|---|---|---|---|---|---|
| 0.5, True  | 0 | 6 | 2 | 0 | 4 | 2 | 5 |
| 0.5, False | 0 | 6 | 6 | 6 | 6 | 0 (SILENT) | 0 |
| 1.0, True  | 0 | 5 | 0 | 0 | 4 | 2 | 5 |
| 1.0, False | 0 | 6 | 6 | 6 | 6 | 0 (SILENT) | 0 |
| **0.5, PR** | **1** | **6** | **6** | **6** | 4 | 2 | 2 |

The four global cells reproduce the BOUNDARY exactly. The PR cell gives comprehension the BEST of both regimes at
once: **AUC 1.000 (6/6), comp byte-id (6/6), comp answer-preserved (6/6)** — where global-homeo-ON degraded it to
AUC≥.80 2/6 and answer-preserved 0/6. The merge stays 6/6 on one-pool, determinism, GABA_B+NMDA-coexist, and
read-isolation. **The single binding conflict the BOUNDARY named — the global `enable_homeostasis` — is resolved.**

## Headline 3: a SECOND, independent residual — the surprise read, NOT homeostasis — blocks the full-cell GO

The PR cell is 1/6 (not ≥5/6) because two surprise-side axes fail, both orthogonal to homeostasis:

- **`surp_answer_preserved` = 2/6, and it is 2/6 IDENTICALLY in every surprise-active cell** (dt0.5_True 2/6,
  dt1.0_True 2/6, dt0.5_PR 2/6). Per-region homeostasis neither helped nor hurt it: the merged-pool surprise read
  flips 1-2 of 24 borderline facts vs the standalone native REGARDLESS of dt (2/6 at both 0.5 and 1.0) or
  homeostasis mode. This is a co-residence surprise-READ robustness residual (a per-fact hair-trigger threshold on
  the merged pool), not the homeostasis conflict and not a dt artifact. Per-seed PR surp-ans: 24,23,22,21,19,24 /24.
- **`cross_load_bearing` = 2/6, DOWN from 5/6 at global-homeo-ON** — a genuine NEW interaction that per-region
  homeostasis EXPOSES. The `surprise_S→sel_agent` cross bias is measured by a brief untrained probe. With role
  homeostasis OFF (the setting comprehension REQUIRES), `sel_agent` uses the fixed, higher `cp_izh_vpeak`, so the
  cross drive clears threshold only on seeds with strong surprise_S probe firing (42,100: +25 Hz) and reads +0 on the
  rest. The very role-OFF that cleans comprehension's graded margin removes the excitability that let the cross drive
  `sel_agent`. (`surp_byte_id` 4/6 is the same borderline-fact brittleness, merged-vs-decoupled.)

**This corrects one BOUNDARY claim with data:** the BOUNDARY said "dt_ms is NOT the binding constant … each organ
tolerates BOTH dt". At the strict per-fact answer-preservation bar the surprise read is brittle at BOTH dt values
(2/6 each) — the residual is not dt but per-fact-threshold sensitivity of the surprise read on the merged pool. Its
other claim — that homeostasis-ON is "a COUPLING channel" via "global homeostatic threshold normalization" — does not
hold: `fused_homeostasis_update` (`kernels.py:1314`) is a per-neuron elementwise EMA→threshold rule with NO
cross-neuron term. The global-homeo-ON coupling was the load-bearing cross synapse amplified by `sel_agent`'s OWN
adaptive threshold; with role on fixed `vpeak` that amplification is gone (which is exactly why the cross bias
weakens).

## Mapped next levers (named, not deferred)

1. **A robust surprise answer read** — margin / rate-window based rather than a per-fact hair-trigger threshold — to
   remove the borderline-fact brittleness (closes `surp_answer_preserved` and `surp_byte_id`). The BOUNDARY guessed a
   "dt-invariant" read; the data refines the target to threshold-robustness, since the brittleness is dt-independent.
2. **Restore the surprise→role drive under role-homeostasis-OFF** — the newly-exposed interaction. Either scope
   homeostasis MORE narrowly (adapt `sel_agent` excitability without corrupting the graded WTA margin — a
   sel-pool-only opt-in, testable with the SAME existing `BrainRegion.enable_homeostasis` primitive), or raise the
   cross gain, to recover the `cross_load_bearing` the whole-role-OFF removed. This is a genuinely NEW lever the
   per-region reconciliation surfaced, not a re-derivation.

## Stage-1 verification of the primitive (determinism + byte-identical-when-off + per-region masking)

- **No engine edit → byte-identical by construction:** `git diff sim/` is empty. The default build (no region opts
  in) leaves `cp_homeostasis_neuron_mask = None` (asserted), so the spike-threshold selection and update gate take
  their legacy branches unchanged.
- **Determinism 9/9:** `pytest tests/test_determinism.py -q` → 9 passed (incl. `TestSubstrateActuallySeeded`); the
  merged bridge builds byte-identically twice (hash of `cp_membrane_potential_v` + `cp_neuron_firing_thresholds`),
  6/6 all cells (in the artifact).
- **Per-region masking works (direct data asserts):** in the PR build the mask covers EXACTLY the 1056 surprise
  neurons and NONE of the 1032 role neurons with the global flag forced OFF; under 80 driven steps the surprise-slice
  thresholds ADAPT (max|Δ|≈9e-4) while the role slice's firing trace is BIT-IDENTICAL (16464 == 16464 spikes) to a
  fully-homeostasis-OFF bridge — role genuinely uses fixed `vpeak`, unaffected by the computed-but-unused homeostatic
  update. (`tools`-free check: scratch `verify_perregion_mask.py`, all asserts pass.)

## Anti-cheats (verified, inherited from the BOUNDARY runner + the new masking asserts)

- Genuinely one pool (2088 = 1056 surprise + 1032 role), determinism (build-twice byte-id), GABA_B+NMDA coexist
  (per-region NMDA mask = 48 = sel_agent+sel_patient) — 6/6 all cells.
- No weight transport / host gradient / reward leak: role cue→role weights are the comprehension organ's OWN frozen
  learned synapses (installed BY NAME, gates frozen); `enable_stdp=False`, `enable_reward_modulation=False`;
  `current_reward_signal==reward_baseline==0.0` (asserted).
- Brain-based reads only: comprehension margin is `cp_firing_states` off sel_agent/sel_patient with the host
  `_semantic_contrast` dot-product replaced by a raising tripwire (never called); surprise is windowed `surprise_S`
  firing; the cross is a real `cp_connections` synapse.
- Read-isolation 6/6 both organs.

## Biology grounding (cites that RESOLVE)

- **Intrinsic homeostatic plasticity is cell-autonomous and per-region** (Desai 1999 / Turrigiano): a neuron adapts
  its OWN threshold toward a target rate — correctly a per-neuron elementwise update; scoping it per-region is the
  faithful choice. In-repo anchor: `research/findings/2026-06-08-navfaithful-derisk-FAIL-homeostasis-confound.md`
  (the finding that motivated `BrainRegion.enable_homeostasis`).
- **GABA_B/GIRK + NMDA coexist as independent per-neuron conductances** summing only at the membrane
  (`research/findings/2026-06-08-gabab-girk-conductance-design.md`; Wong & Wang 2006 for the NMDA-slow WTA).

## What stays a residual / the follow-on

The homeostasis conflict is closed for the comprehension side (production role read fully reconciled on one shared
pool). The config-superset merge does NOT yet reach a full-cell GO: the surprise organ's per-fact answer read is
brittle on the merged pool, and role-homeostasis-OFF weakens the surprise→role cross. Both have named levers (above),
both testable with existing machinery (no new `sim/` risk). This is a DE-RISK; the PRODUCTION `shared=` wiring of
`ComprehensionProductionOrgan` + DEFAULT flip remains the gated follow-on, now unblocked on the comprehension axis.

## Files

- Runner extension (additive, default-preserving): `_one_brain_merge_configsuperset_production_derisk.py` gains a
  `0.5:PR` cell (per-region homeostasis: surprise ON, role OFF); `_one_brain_merge_Norgan_derisk.py`
  `build_merged_diffbuilder` gains `per_region_homeo=False` (default-OFF → byte-identical to the legacy global path).
- **NO `sim/` edit** — reconciliation uses the existing `BrainRegion.enable_homeostasis` primitive.
- Artifact: `research/findings/raw/_one_brain_merge_configsuperset_perregion_6seed.json`.
