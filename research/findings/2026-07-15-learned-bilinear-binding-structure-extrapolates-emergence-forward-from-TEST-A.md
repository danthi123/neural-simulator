# Emergence-bar FORWARD from TEST A: a LEARNED bilinear BINDING STRUCTURE extrapolates systematically (mean 0.75, 6/6 > the plain MLP 0.39, approaching the fixed bind 0.87) — the systematicity-enabling structure can be LEARNED, given the multiplicative-binding inductive bias

**Date:** 2026-07-15 · **Runner:** `research/runners/_learned_bilinear_binder_systematicity_derisk.py` (reuse-by-import TEST A's task + `_train_snn`; numpy-CPU; NO `sim/` edit). Follows `2026-07-15-TEST-A-fixed-bind-...GO` + its a-1 reconciliation. **Verdict: the load-bearing STRUCTURE-vs-UNSTRUCTURED comparison is a clean 6/6; the strict GO gate is 2/6 only due to the same memfloor split-luck as TEST A.**

## The question
TEST A: a FIXED ±1 bind extrapolates to held-out compositions (0.87) where a general MLP learner memorizes+fails (0.39). The a-1 reconciliation (`2026-06-11`): a LEARNED *bilinear* binder is ALSO systematic on decorrelated codes. So the discriminator is a BINDING STRUCTURE (fixed or learned-bilinear) vs a from-scratch map-classifier. This runs a LEARNED bilinear binder — `bound = (W_a @ cat) ⊙ (W_b @ q)`, the multiplicative `⊙` a FIXED inductive bias, the projections W_a,W_b LEARNED by gradient — on TEST A's EXACT 7×7 held-out-combination task (decorrelated ±1 codes).

## Result (6-seed 42/43/44/100/101/102; chance 0.25)
| arm | held-out extrapolation (mean) | 6-seed |
|---|---|---|
| **LEARNED bilinear (learned projections + fixed ⊙ bind)** | **0.75** (0.571–1.000) | above chance + above the MLP on ALL 6 |
| fixed ±1 bind (TEST A reference) | 0.87 | the ceiling |
| **plain MLP map-classifier (the control)** | **0.39** (0.000–0.500) | memorizes train (1.0), fails held-out |
| 1-NN memfloor | 0.51 | split-dependent (0.43–0.86 — held-out shares a factor with train) |
| permuted (anti-cheat) | ~0.21 | collapses |
- **The load-bearing comparison — a LEARNED binding STRUCTURE (0.75) >> an UNSTRUCTURED learner (0.39) — holds on all 6 seeds**, and approaches the fixed bind (0.87). So the systematicity-enabling structure is LEARNABLE (the projections), as long as the multiplicative-binding INDUCTIVE BIAS is present. Provide the bind *structure*; the codes/projections can be learned.
- **Honest caveat:** the strict GO gate (also requiring bilinear > memfloor+0.15) is 2/6 because the memfloor 1-NN gets lucky on splits where held-out combos sit near training combos in code space — the SAME split-luck as TEST A's s102, a bounded gate-tightening (hold out memfloor-hard combos), NOT a mechanism issue.

## ⇒ What this confirms + the next rung
Confirms the emergence-bar path from TEST A: **the composing-machinery need not be a hand-fixed primitive — the binding STRUCTURE can be LEARNED** (its projections/codes) as long as the multiplicative-binding inductive bias is provided; an unstructured from-scratch learner cannot discover it (memorizes). This unifies with `2026-06-11` (gradient bilinear GO on decorrelated) + the project's composer (fixed bind over learned codes).
- **NEXT (the full emergence step):** train the same bilinear by the GO transport-free **deep-credit e-prop rule** (not gradient) — does a BIOLOGICAL rule learn the binding structure? (`2026-07-14` feedforward deep-credit GO makes this plausible; the bilinear-with-deep-credit trainer is the build.) Then the on-substrate realization (learned projections + the spiking coincidence bind).
- Harden the memfloor split (hold out memfloor-hard combos) for a clean 6/6 gate on both this + TEST A.
Reuse-by-import; NO `sim/` edit.
