# R-iii CA3 formation, Rung 1 (the research-gate root-cause fix): adding the MISSING CA3 feedback inhibition (a `ca3_pv_basket` FS pool, runner-side, NO `sim/` edit) SPARSIFIES the CA3 code (0.43 → 0.21) — confirming the gate's diagnosis that CA3 had ZERO feedback inhibition wired — but GLOBAL feedback inhibition is NON-SELECTIVE: it suppresses the ENSEMBLE MEMBERS too, so the within-ensemble potentiation stays weak (7.05) and the within/silent ratio does NOT improve (1.16×, no better than the distributed baseline). The sparsification saturates at 0.21 (byte-identical at inhibition weight 120 vs 250 — the FS pool fires maximally). NEXT: Rung 2, the mossy-detonator (strong sparse dg->ca3 so a few CA3 cells fire HARD despite the inhibition → a sparse + strongly-firing selective ensemble) — the gate's ranked refinement, still runner-side (NO `sim/` edit).

**Date:** 2026-07-09
**Method note:** this was the FIRST arc to lead with the new gate rule (a0) — READ OUR OWN SUBSTRATE FIRST. Reading `text_minimal_isolation.py:712-719` confirmed the root cause (`ca3` region `internal_density=0.0` → its 15% inhibitory cells unconnected; every X->ca3 pathway excitatory) BEFORE building, and I read the partial result off my own numbers (within-ensemble weakened → members suppressed) rather than after another blind sweep. See memory `feedback_read_own_substrate_before_theorizing`.
**Runner:** `research/runners/_riii_ca3_attractor_diag.py --ca3-fb-inhib W` + `_riii_ca3_coincidence_completion_derisk.py::_build` (appends a `ca3_pv_basket` FS region + `ca3->ca3_pv_basket` + `ca3_pv_basket->ca3` feedback pathways, mirroring the validated `dg_pv_basket` FFi wiring but as a FEEDBACK loop). GPU. NO `sim/` edit.

## Result
```
config                                    sparsity   within-ens   member->silent   ratio
baseline (no CA3 inhibition)              0.43-0.47  8.36 (lr50)  8.02             1.44x (the plateau)
+ ca3_pv_basket feedback inhib w=8        0.38       --           --               --
+ feedback inhib w=16/30/60/120 (hebb_sym) 0.19->0.15 --          --               -- (sparsity saturates ~0.15)
+ feedback inhib w=120 + rate-window lr5  0.21       7.05         6.09             1.16x
+ feedback inhib w=250 + rate-window lr5  0.21       7.05         6.09             1.16x  (byte-identical -> saturated)
```

## What the numbers say (read from my own data, not theory)
1. **The root-cause fix is real:** adding the missing CA3 feedback inhibition drops sparsity 0.43 → 0.15-0.21 (a 2-3× sparsification). The gate's diagnosis (CA3 had NO feedback inhibition → uncapped recurrent excitation → distributed code) is confirmed — the `ca3_pv_basket` is a genuine reusable sparsifier.
2. **But GLOBAL inhibition is non-selective:** member->silent is now at init (6.09) — non-members are TRULY silenced (good, they stop getting potentiated) — BUT within-ensemble is only 7.05 (vs 8.36 no-inhib): the global inhibition suppresses the MEMBERS too, so they co-fire weakly and potentiate weakly. The ratio is now limited by WEAK MEMBER FIRING, not by silent rising. This is exactly the gate's cited caveat (PMC12244581: "global inhibition fails; assembly-SELECTIVE feedback inhibition enforces the bindable sparse code").
3. **Saturating:** doubling the inhibition weight (120 → 250) is byte-identical (0.21 / 7.05 / 1.16×) — the FS pool already fires maximally; more weight can't sparsify further or select the ensemble.

## The reframe for Rung 2 (mechanism-motivated, not a blind next-sweep)
The bottleneck is now clear + specific: the members must fire STRONGLY (co-fire densely) to potentiate, WHILE the non-members are suppressed. Global inhibition suppresses everyone. The biological answer (gate Rung 2, Kandel Ch 54 mossy fibers): the DG->CA3 MOSSY synapses are "detonators" — a few strong sparse synapses make their target CA3 cells fire HARD from a single DG input. Currently `dg->ca3` is density 0.10, weight 8.0 (`text_minimal_isolation.py:1110-1115`). Strengthening the mossy weight makes the DG-selected CA3 cells DETONATE (fire hard = the ensemble) while the feedback inhibition suppresses the rest → a SPARSE + STRONGLY-FIRING SELECTIVE ensemble → strong within-ensemble co-activity → strong potentiation, with truly-silent non-members → a high ratio. Test: mossy weight sweep + feedback inhibition + rate-window → does the ratio pass ~3×? Runner-side (modify the returned dg->ca3 pathway weight, like the coincidence flip); NO `sim/` edit.

## R-iii arc status (honest)
- Completion half: SOLVED (CYCLE 1068, dendritic dAP, 6-seed).
- Formation half: rule solved (symmetric + rate-window, byte-safe `sim/` primitives); root cause found + Rung-1 sparsifier built (this cycle, no `sim/` edit); the OPEN piece is a SELECTIVE sparse ensemble (Rung 2 mossy-detonator, then Rung 3 theta-gamma synchronization if needed).

## Files
`research/runners/_riii_ca3_coincidence_completion_derisk.py` (`ca3_fb_inhib`/`ca3_fb_n` -> the ca3_pv_basket append), `_riii_ca3_attractor_diag.py`. Research gate: `2026-07-09-riii-sparse-synchronous-ca3-ensemble-research-gate.md`. Prior: `2026-07-09-riii-formation-rules-saturate-ensemble-dynamics-is-the-blocker.md` (1070/1071). Biology: Kandel 6e Ch 54 pp 1357-1361 (DG sparse code, mossy detonators, CA3 inhibition); PMC12244581 (assembly-selective inhibition); Kopsick 2024 (sparse synchronous CA3 assemblies).
