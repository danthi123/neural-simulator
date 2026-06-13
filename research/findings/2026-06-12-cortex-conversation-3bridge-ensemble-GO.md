# 3-bridge conversational ensemble de-risk (build step 1, option b): GO — the two validated de-risks INTEGRATE; within-bridge generalization + cross-bridge composition coexist with the moat intact

**Date:** 2026-06-12. **Runner:** `research/runners/cortex_conversation_ensemble_derisk.py` (commit 5e6f6952). **Backend:** `SIM_BACKEND=cupy` (GPU). **Raw:** `research/findings/raw/_cortex_conversation_ensemble_D512_full.json` + `.log`. **Scope:** 3 shards × 64 = 192 concepts (animals/foods/vehicles), 3 seeds (42/43/44), composer **D=512**, learned cortex. Owner chose this intermediate (option b) before the 32-bridge build.

> **Verdict: GO (3 seeds).** Tonight's two validated de-risks — within-bridge generalization-in-conversation (the capability) and cross-bridge composition + moat (the mechanism) — **integrate cleanly on a 3-bridge ensemble.** The conversational matrix passes, within-bridge generalization works per bridge, cross-bridge composition works (both conversational identity recall and the spiking V-tag layer), the no-confab moat is intact throughout, and all anti-cheats collapse — all multi-seed. One wrinkle surfaced and was diagnosed + fixed (the composer dimension must scale with the union vocabulary), yielding a load-bearing learning for the 32-bridge build.

## Results (3 bridges × 64 concepts, D=512, seeds 42/43/44)

| Gate | Result |
|---|---|
| **A — conversational matrix on the ensemble** (who/what, abstention, negation, one-attribute, clause; SVO roles span the bridges) | **6/6 cells, moat holds (0 abstention breaches)**, all 3 seeds |
| **B — within-bridge generalization in conversation** (per bridge, co-resident) | **0.975–1.000** (≈4× chance) on all 3 bridges × 3 seeds; B1-conv (through the `what_does` fallback) 0.95–1.00; B2 moat 0 false-accepts on 64-cue floors |
| **X — cross-bridge composition** (conversational X-conv + spiking V-tag) | X-conv `what`/`who` = **1.0/1.0**, 0 abstention breaches; X-vtag M3 top2=1.00, **signal/floor 16.85–23.96×**; Cx PERMUTED (fixed anti-cheat) collapses (top2 0.00–0.08); `x_vtag_recall_ok` + `x_vtag_band_ok` True |
| **moat over cross-bridge facts** (C3) | agreement 1.000, host-abstain/gate-accept=0, floor-false-accepts=0, lesion collapses → intact, all 3 seeds |
| **anti-cheats** | C1 permuted-similarity (per bridge) collapses; C4 random-shard collapses; all 3 seeds |

`COMBINED VERDICT: GO`. Total elapsed ~7.3 h (the per-shard anti-cheat re-learns + the live-spiking V-tag recall are the cost).

## The clause / dimension wrinkle (diagnosed + fixed) — the build learning
The first run (composer **D=128**) gave Gate A = **5/6**: every cell passed except `clause` (recursive composition). Diagnosis: D=128 over the **192-concept union vocabulary** is under-provisioned — the single-shard capability de-risk used D=128 for only 64 concepts, and FHRR binding capacity scales with dimension, so the highest-capacity cell (recursive clauses) breaks first at 3× the vocabulary. A 10-minute matrix-only check at **D=512** confirmed: **clause passes, 6/6** (`_cortex_ensemble_matrix_D512.json`). So the failure was purely the composer-dimension capacity knob, NOT a fundamental integration break. The full D=512 run is the GO above.

**⇒ Load-bearing build learning: the FHRR composer dimension must scale with the union-vocabulary size** (~2.7 D/concept in this run). At 2,048 concepts a single union composer needs D ≈ 5,500 — which is why **Phase 1's first decision is the composer architecture** (per-bridge composers, whose cost is vocabulary-independent, vs one scaled union composer; design `docs/plans/2026-06-12-phase1-composer-architecture-design.md`, de-risk in flight).

## Honest scope
- **3 bridges, D=512.** The 8-bridge fan-out + the per-bridge-vs-union composer decision are Phase 1's de-risks; the 2,048-concept / 32-bridge production system is the build.
- The parser is bypassed (composer path; separately validated) — the full `hear()` loop is Phase 2.
- Gate B generalization is strictly within-bridge (the fallback is shard-restricted); cross-bridge is identity composition (Gate X), not graded generalization — by design (graded similarity is a within-bridge property).

## Conclusion
Option (b) is a GO: the learned-graded cortex's two capabilities (within-bridge generalization, cross-bridge composition) coexist with the no-confab moat on a multi-bridge conversational ensemble. The integration is proven at small scale. Next: Phase 1 — the composer-architecture decision (route A per-bridge composers, de-risk in flight), then sharding 2,048 concepts + the 32-bridge fan-out + production-scale gate confirmation. No `sim/` edits. No banking — the clause wrinkle was diagnosed + fixed + re-run before the GO.
