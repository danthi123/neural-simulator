---
type: finding
status: live
date: 2026-08-13
mechanism: onebrain-merge-organs-pool2-flip
integration_faculty: onebrain-merge-organs
---

# One-brain pool #2 flip (GO): metacog + pragmatic are now wired into production on ONE shared spiking bridge, default-ON — 4 organs share a substrate by default

**Date:** 2026-08-13 · **Flips:** `research/runners/onebrain_merge_production2.py` `_MERGE2_DEFAULT_ON = True`
(escape `BRAIN_ONEBRAIN_MERGE2=0`) + `research/runners/metacog_production_organ.py` adopts the divisive-normalized
NMDA-conductance confidence read as its DEFAULT (`nmda_norm`; escape `BRAIN_METACOG_READ=balance`). ·
**Gate:** `research/runners/_onebrain_production_flip2_verify.py` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`) ·
**Artifact:** `research/findings/raw/_onebrain_production_flip2_6seed.json` · **Unblocked by:**
`2026-08-13-metacog-robust-confidence-GO.md` · **Resolves:** `2026-08-13-onebrain-second-pool-SCOPED.md` (the withheld flip).

## What flipped

Pool #2 (the metacog E1 balance-of-evidence confidence monitor + the D-pragmatics W4 scalar-implicature RSA belief
organ) now builds on ONE shared `MergedSubstrate2` `SimulationBridge` (one `cp_membrane_potential_v`, N=450) BY
DEFAULT, exactly as pool #1 (surprise D2 + world-model E2) has since 2026-08-13. Both production `get_organ`s inject
the process-shared `get_merged_substrate2(seed)` when `merge2_enabled()`, so the `/api/brain-chat` handler reads
metacog `judge()` and pragmatic `interpret()` through the shared singletons. Verified in-process: metacog `_shared`
IS pragmatic `_shared` IS the `get_merged_substrate2` singleton; pool #1 and pool #2 are distinct substrates.

## Why it was blocked, and the correction that unblocked it

The merge was already byte-identical (MERGE-GO A+B 6/6) and pragmatic was answer-preserving (6/6), but metacog was
NOT (1/6). Its confidence read was the ABSOLUTE spike-rate margin `|rate(asm1)-rate(asm0)|` off `cp_firing_states`,
which sits at the workspace's ~0.1%-firing NOISE FLOOR (near-random monotonicity ~0.5); the three region-scoped merge
seams (`per_region_parameter_heterogeneity` / `_threshold_heterogeneity` / `per_region_wiring_seed`) reshuffle that
noise and flip the confident/uncertain decision at mid-range evidence. The fix (`2026-08-13-metacog-robust-confidence-GO.md`)
reads the balance off the assemblies' slow-NMDA RECURRENT CONDUCTANCE instead —
`|g_nmda(asm1)-g_nmda(asm0)| / (g_nmda(asm1)+g_nmda(asm0)+eps)` (Carandini & Heeger divisive normalization; Wang
persistent-NMDA accumulator). This session ADOPTED that read as the metacog production default (canonical
`metacog_production_organ.nmda_norm_margin`; the de-risk's `RobustMetacogProductionOrgan` now reuses it by import),
then flipped the pool #2 default.

## Result — FLIP-GO 6/6 (up from the SCOPED 1/6)

Every read is through the real production organ APIs over the pool-#2 panel (metacog evidence sweep MC_EVID×8;
pragmatic {none, some, all}). TODAY = each organ on its own bridge (== `BRAIN_ONEBRAIN_MERGE2=0`, the escape == the
pre-flip production); MERGED = both on ONE shared bridge (the default-ON path); CORESIDENT = each on its own bridge
with the three merge seams ON (the apples-to-apples merge baseline).

| axis (6 seeds) | result |
|---|---|
| A. ONE shared pool (metacog.bridge IS pragmatic.bridge IS substrate, N=450) | 6/6 |
| B. MERGED == CORESIDENT byte-identical (Δ==0.0, both organs' reads) | 6/6 |
| **MERGE-GO (A + B)** | **6/6** |
| C. answer preserved MERGED-vs-TODAY — PRAGMATIC (implicature + enriched) | 6/6 |
| C. answer preserved MERGED-vs-TODAY — **METACOG** (`confident` bool, nmda_norm read) | **6/6** |
| **FULL FLIP-GO (A + B + C both organs)** | **6/6** |

D (documented residual, reported not gated): the merged-vs-today numeric deltas — metacog balance ≤ 0.0204, pragmatic belief ≤ 0.0658 (rounded aggregate maxes over the 6 seeds; exact fields residual_metacog_balance_max / residual_pragmatic_belief_max in the cited _onebrain_production_flip2_6seed.json) <!--derived--> — the honest cost of one shared pool (one global RNG cannot reproduce both organs' standalone threshold draws); NO classification crosses a threshold, so the answer is unchanged.

## No regression

- `pytest tests/test_determinism.py -q` -> 9/9 PASS (substrate seeding unaffected).
- `brain_chat_tui --smoke` BYTE-IDENTICAL default-vs-escape (`BRAIN_ONEBRAIN_MERGE2=0`) — the smoke does not invoke
  the pool-#2 organs, so the flip is a genuine no-op for it (the smoke module imports none of the changed modules).
- Escape reverts: `merge2_enabled()` = True default, False under `=0`, True under `=1`; `default_confidence_read()`
  = `nmda_norm` default, `balance` under `BRAIN_METACOG_READ=balance`.
- Pool #1 (surprise + world-model) UNTOUCHED and still default-ON (`_MERGE_DEFAULT_ON = True`,
  `merge_enabled()` = True); it remains a distinct shared substrate.
- The de-risk `_metacog_robust_confidence_derisk` still reproduces the baseline vs GO after its refactor (single-seed:
  `balance` read metacog-answer-preserved 0/1 = the 1/6 blocker; `nmda_norm` read 1/1, mono/non-degen/tracks 1/1).

## The metacog read-swap delta (honest — it is a correction, not merely a merge)

Adopting `nmda_norm` as the metacog default changes some STANDALONE confidence calls vs the pre-2026-08-13 absolute
read: full-sweep agreement ~4/6 seeds, clear-extreme agreement ~5/6 (per `2026-08-13-metacog-robust-confidence-GO.md`).
Every change is a DE-NOISING correction: the absolute read is at the noise floor and non-monotone (on seed 102 it
called ZERO-evidence "confident" and evid 0.75 "uncertain"), while the NMDA read yields the clean monotone pattern
`[F,F,F,F,T,T,T,T]` (confident boundary at evidence ~0.5) on all 6 seeds, both standalone and merged. So the swap
stabilizes the standalone organ AND makes the merged decision invariant. Functional read-out only; no phenomenal claim.

## One-brain honesty (the residual, not deferred)

`onebrain-merge-organs`: pool #1 (surprise + world-model) 2 default-ON + pool #2 (metacog + pragmatic) 2 default-ON =
**4 organs sharing a substrate by default**, across TWO distinct shared pools. `scaffold_retired` stays PARTIAL:
affect (structural region-name collision `workspace`/`workspace_fs`/`meta_schema` with metacog + a global OU/neuromod
cfg), comprehension (per-region `dt`), and the causal/curiosity plasticity/neuromod organs remain separate arcs; and
neither pool is yet merged with the recall-composer bridge (`one-brain-substrate`). Those are the named next rungs.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_production_flip2_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_production_flip2_6seed.json
```

## Scope / non-claims

- `wired: YES (metacog + pragmatic build on the shared MergedSubstrate2 by default; /api/brain-chat reads them via the
  shared singletons) / on_by_default: YES (`_MERGE2_DEFAULT_ON=True`, `BRAIN_ONEBRAIN_MERGE2=0` escape) /
  scaffold_retired: PARTIAL (the separate-bridge co-residency for the metacog+pragmatic PAIR is retired; affect /
  comprehension / causal / curiosity remain separate; neither pool merged with the recall composer yet).`
- NO `sim/` edit; the three region-scoped seams already exist on `main` (guarded, default-off; the merge sets them per
  pool). Reuse-by-import; process backend (numpy here -> bit-exact; cupy in production — the reads are deterministic
  given the substrate).
- The GO is the pool-#2 answer-preservation flip. It does not re-assert the E1 type-2 SDT / meta-d' faculty gate (a
  separate metric on a separate runner). Functional read-out only; no phenomenal claim.
