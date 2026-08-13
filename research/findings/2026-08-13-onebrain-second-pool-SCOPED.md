---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
integration_faculty: onebrain-merge-organs-pool2
---

# One-substrate SECOND pool (SCOPED): the metacog + pragmatic production organs share ONE spiking bridge BYTE-IDENTICALLY end-to-end (the first two-FULLY-WIRED-organ merge, `per_region_wiring_seed` exercised) — pragmatic is answer-preserving, metacog's narrow-margin confidence is a MEASURED answer-preservation boundary, affect is a STRUCTURAL exclusion → the production-default flip is WITHHELD (pool default-OFF, opt-in)

**Date:** 2026-08-13 · **Build:** `research/runners/onebrain_merge_production2.py` (`MergedSubstrate2`,
`merge2_enabled`, `get_merged_substrate2`; `_MERGE2_DEFAULT_ON=False`, escape/opt-in `BRAIN_ONEBRAIN_MERGE2`) ·
**Wired:** `metacog_production_organ.get_organ` / `pragmatic_production_organ.get_organ` inject the process-shared
`MergedSubstrate2` when `merge2_enabled()` · **Verify:** `research/runners/_onebrain_production_flip2_verify.py` ·
**Artifact:** `research/findings/raw/_onebrain_production_flip2_6seed.json` (6 seeds 42/43/44/100/101/102,
`SIM_BACKEND=numpy` → bit-exact) · **Ledger:** row `onebrain-merge-organs` updated + the pool-#2 anchor.

## What this lands

Pool #1 (`onebrain_merge_production.py`, DEFAULT-ON) put the D2 surprise + E2 world-model organs
(`enable_parameter_heterogeneity=False`) on ONE shared bridge. Its named next rung
(`2026-08-13-onebrain-production-default-flip-SCOPED.md`): a SECOND pool for the three param-het-ON organs
(metacog / pragmatic / affect), which need a DIFFERENT global config and so cannot join pool #1. This arc BUILT that
second pool for the two compatible CORTICAL-MICROCIRCUIT organs — the E1 metacog balance-of-evidence confidence
monitor (`build_metacog_bridge`, confidence_read="balance") and the D-pragmatics scalar-implicature RSA belief organ
(`build_rsa_bridge` + `_rsa_recursion`) — on ONE `SimulationBridge` (one `cp_membrane_potential_v`, N=450), each organ
reading its own region slice. This is the FIRST TWO-FULLY-WIRED-ORGAN merge: it exercises `per_region_wiring_seed`
end-to-end — the seam named "proven at the substrate level but NOT yet exercised in a two-fully-wired-organ production
merge" by `2026-08-13-per-region-ou-wiring-affect-GO.md`.

## Result — the genuine merge is GO 6/6; the production-default FLIP is SCOPED

Every read is through the REAL production organ APIs the `/api/brain-chat` handler calls — metacog `judge(evidence)`,
pragmatic `interpret(utterance)`. Three build variants per seed: TODAY (`shared=None`, == the pre-flip separate-bridge
production == the escape), MERGED (both organs on ONE shared bridge, the opt-in path), CORESIDENT (each organ on its OWN
bridge with the THREE merge seams ON — the apples-to-apples merge baseline).

| axis (6 seeds; metacog evidence sweep ×8, pragmatic {none,some,all}) | result | verdict |
|---|---|---|
| A. ONE shared pool (metacog.bridge IS pragmatic's shared bridge IS the substrate bridge; one `cp_membrane_potential_v`, N=450) | 6/6 | GO |
| B. MERGED == CORESIDENT byte-identical (metacog balance margin + pragmatic belief deltas 0.0 — the genuine merge byte-identity, end-to-end through the real read APIs) | 6/6 | GO |
| **MERGE-GO (A + B — a genuine, byte-identical one-pool merge of two fully-wired organs)** | **6/6** | **GO** |
| C. answer preserved vs TODAY — PRAGMATIC (`implicature_represented` + `enriched_interpretation` identical) | 6/6 | GO |
| C. answer preserved vs TODAY — METACOG (`confident` bool identical across the evidence sweep) | 1/6 | **BOUNDARY** |
| D. numeric residual vs TODAY (reported, NOT gated): metacog balance ≤ 0.0022, pragmatic belief ≤ 0.0658 | — | documented |
| **FULL FLIP-GO (A + B + C both organs)** | **1/6** | **SCOPED → flip WITHHELD** |

## Why the flip is WITHHELD, not forced — the two honest blockers

**METACOG — a MEASURED answer-preservation boundary (its narrow dynamic range, NOT the merge).** The merge REQUIRES
the three region-scoped seams (`per_region_parameter_heterogeneity`, `per_region_threshold_heterogeneity`,
`per_region_wiring_seed`) so each organ's slice is invariant to co-residence — that is what makes B == 0.0. But those
seams RE-DRAW metacog's workspace init + competition wiring name-keyed instead of from the global-RNG order, and
metacog's balance-of-evidence confidence has a DECLARED NARROW DYNAMIC RANGE (`metacog_production_organ` residual;
`2026-08-13-per-region-param-het-cluster-GO` already flagged the seed-43 fragility as "a pre-existing
narrow-dynamic-range property, NOT a merge effect"). The small margin shifts enough under the re-draw to flip the
confident/uncertain decision at mid-range evidence (measured: even with param+threshold only and metacog's STANDALONE
wiring, the per-evidence `confident` bool matches TODAY only 4–5/8 on seeds 42/43; `per_region_wiring_seed` compresses
the margin further, flattening the evidence-tracking). So metacog cannot be flipped to production-default
answer-preservingly. This is a boundary on the CONFIDENCE READ (make it robust to the re-draw — a wider-dynamic-range
comparator / more integration), NOT on the merge: the co-residence itself is byte-identical (B 6/6).

**AFFECT — a STRUCTURAL exclusion (measured, not deferred).** The affect production organ builds a WHOLE co-resident
brain (`_stageA_full_integration_derisk.build_one_brain(with_faculties=True, co_resident_affect_ladder=True)`) whose
honesty relay defines regions NAMED `workspace` / `workspace_fs` / `meta_schema` — a HARD NAME COLLISION with metacog on
one `region_manager`. Renaming is impossible without breaking byte-identity: every merge seam keys its
position-invariant RNG on the region NAME (`zlib.crc32`), so a rename changes the slice's init + wiring. Affect ALSO
needs a GLOBAL `enable_ou_process=True` + the neuromodulator subsystem, which the OU-off / neuromod-off microcircuits
do not share in ONE cfg. So affect's merge target is its OWN pool / the recall-composer bridge (it is itself a "one
brain"), a distinct rung — the same "flip the clean subset, map the rest honestly" rule pool #1 followed.

**PRAGMATIC — answer-preserving (6/6).** Its graded RSA implicature belief self-normalizes: `some` → `some but not all`
(SBNA ~0.73 preferred, `all` ~0.27-possible), `implicature_represented` + `enriched_interpretation` identical
merged-vs-TODAY on every seed, byte-identical merged-vs-coresident. Pragmatic would flip cleanly — but a shared pool
needs ≥2 organs, and its only compatible partner (metacog) is not answer-preserving, so the pool ships default-OFF.

## No regression (default-OFF = the pre-flip production)

- `BRAIN_ONEBRAIN_MERGE2` absent/0 → `merge2_enabled()` False → each organ builds its own bridge (`shared=None`) — the
  pre-flip separate-bridge path, byte-identical. Opt-in `BRAIN_ONEBRAIN_MERGE2=1` → metacog + pragmatic share ONE
  `cp_membrane_potential_v` (N=450, verified).
- `pytest tests/test_determinism.py -q` → **9 passed**.
- `brain_chat_tui --smoke --stub-renderer` JSON **byte-identical** to clean HEAD under the identical command
  (git-stash the two organ edits, rerun, exact JSON compare — empty diff; the pool-#2 organs are not invoked on the
  smoke's rf/tiny-demo path, and the default-OFF flag makes the edits a no-op).
- Pool #1 regression guard holds: pool #1's `BRAIN_ONEBRAIN_MERGE` + its surprise/world-model organs are untouched;
  the new flag + module are independent.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_production_flip2_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_production_flip2_6seed.json
```

## Honest scope / non-claims

- `wired: metacog + pragmatic get_organ inject a process-shared MergedSubstrate2 when BRAIN_ONEBRAIN_MERGE2=1 (one
  cp_membrane_potential_v, N=450) / on_by_default: NO (default-OFF, opt-in; the flip is SCOPED-withheld on metacog's
  answer-preservation boundary + affect's structural exclusion) / scaffold_retired: none (the flip is withheld).`
  Functional read-outs only; no phenomenal claim.
- The genuine one-pool MERGE (A + B, byte-identical two-fully-wired-organ co-residence, `per_region_wiring_seed`
  exercised end-to-end) is GO 6/6 — a real, load-bearing result and the named rung. The production-default FLIP is
  what is withheld.
- No cross-organ synapse is added; the load-bearing claim is one shared pool + byte-identity to the merge baseline. A
  genuine cross-region synapse and merging onto the RECALL composer bridge are later rungs.
- Organs sharing a substrate by DEFAULT: pool #1 (surprise + world-model) = 2 (default-ON). Pool #2 (metacog +
  pragmatic) = built + byte-identical + wired but DEFAULT-OFF (flip scoped) → still 2 organs on a shared substrate by
  default, project-wide. Remaining cluster residuals: metacog confidence-read robustness (the flip blocker), affect on
  its own pool / the composer bridge, comprehension (per-region `dt`, impossible byte-exact), causal/curiosity
  (stdp+reward+neuromod).
