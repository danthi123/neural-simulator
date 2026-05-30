# Phase-factored Task 4 adversarial review: CLEAR -- decisive run (Task 6) is trustworthy

Date: 2026-05-30. Independent adversarial reviewer, 9 scrutiny items + recomputation.

## Verdict: CLEAR -- the wiring is sound; the decisive run is trustworthy.

1. Genuinely TWO-PHASE in order: Phase 1 online order-preserving (binding index == presentation index; engram bind) BEFORE Phase 2 offline shuffled (run_concept_replay_phase randomize_order=True). Phase-ordering pins PASS. SOUND.
2. Avoids the prior encode-order conflict: order lives in the engram tag (sparse IDENTITY index, immune to representational drift = the two_phase_pointer regime); selectivity built by Phase 2 shuffled replay in cortex (cannot rewrite the tag index). full keeps ep=1.0, no_hippo_store collapses ep=0.0 -> order genuinely survives consolidation. GENUINELY DECOUPLED.
3. Lesion fidelity: _make_pairs is the SOLE per-trial RNG consumer; byte-identical pairs+state across full + 7 lesions + v1 (no per-lesion re-seed). no_shared_clock genuinely desyncs 16/16 steps (recomputed), partitioned SHARED. SOUND.
4. Cannot score a false PASS: controller defines no own bars; output flows through frozen integrated_loop_verdict unchanged; bars byte-unchanged. A wm=0.0-everywhere full-scale input -> GATE=VOID (recomputed). Tiny path = GATE=TINY/NOT-PROPAGATED, bypasses the real verdict. CANNOT.
5. (HIGHEST RISK) tiny-synth wm=0.0 = documented VOID-by-construction SCALE ARTIFACT, not a bug: gate threshold 650 vs tiny theoretical max 6*2=12 (always abstains); wm readout is a faithful structural copy of the validated parked path reading the correct noun_pool_F* region; full scale 320*48=15360 (~24x over gate, ~2.4x headroom on a bound pool per the parked measured 0.10 active fraction). EP readout proves the pipeline is alive+discriminating even at tiny scale. Worst case = honest VOID, never silent false PASS. NOT A MASKED BUG.
6. Substrate-caveat insurance wired: Phase 2 run_concept_replay_phase drives CA3 ensembles -> STDP at ca3->ca1->cortex auto-consolidates (validated Phase-1.3 ca1->concept path) under sleep gates -> consolidation UPDATES the index. WIRED.
7. Genuine reuse: git diff protected set byte-empty; all subsystems + SharedThetaGamma reused by import (reuse pins PASS; SharedThetaGamma not redefined). BYTE-EMPTY.
8. Cheap probe honest: single_pass_best [0.78,0.78,0.79] < 0.90 reproduced by real min(wm,ep) sweep (not hardcoded); residual-coupling a REAL measurement (commit 23ae76e) that contradicted the original closed form (coupling_demonstrated=False at D=64; reappears at D=4); RESOLVES honestly caveated with the substrate caveat. NOT circular.
9. No autograd / no protected modification: no torch/.backward anywhere; only the 4 expected files new since c1e79b7. CLEAN.

## Non-blocking item for the decisive smell-test
EP readout is taken POST-consolidation. The smell-test (Task 6.3) should explicitly verify ep survives Phase 2 on the REAL substrate (real reps have large common-mode -> consolidation moves them substantially). The frozen v1 ep>=0.90 soundness gate already enforces this (VOIDs if order doesn't survive); the consolidation-updates-index insurance is the mitigation. Check-don't-fix.
