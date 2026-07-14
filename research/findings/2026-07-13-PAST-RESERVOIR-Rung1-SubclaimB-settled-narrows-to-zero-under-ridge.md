# Rung-1 Sub-claim B SETTLED (6-seed, honest): under a closed-form ridge read-out the Sigma-Pi product advantage over a generic strong nonlinear basis NARROWS TO ~0 — the CONJUNCTION is the missing ingredient (Sub-claim A, robust), but the SPECIFIC product form is not privileged over any strong nonlinearity

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_conjunction_subclaimB_ridge_derisk.py` (reuse-by-import of the Rung-1 conjunction runner; numpy; NO `sim/` edit).
**Status:** Honest settling of the qualification the adversarial-verify flagged (recommended follow-on: RFF-vs-product under ridge at 6 seeds).

## What was tested

The Rung-1 adversarial-verify (4/4 survived) separated two sub-claims and flagged that Sub-claim B was readout-dependent:
- **Sub-claim A** — "a LINEAR read-out over a fixed reservoir CANNOT extract the conjunction" (product ≫ linear) — ROBUST/decisive (representational limit; gap persists to 4× reservoir; +0.28 on a non-adjacent conjunction).
- **Sub-claim B** — "it's the CONJUNCTION STRUCTURE, not generic nonlinear capacity" (product > a param-matched strong nonlinear basis) — comfortable under the delta rule (+0.09 vs RFF) but narrowing to +0.02–0.04 under ridge at 1–2 seeds. The verify recommended a 6-seed ridge check.

This runs it: product vs random-Fourier-features (RFF, a strong universal nonlinear basis, param-matched to N_PROD) under a **closed-form ridge** read-out (zero epoch/lr dependence), 6 seeds, on both the adjacent (`rule[prev2,prev1]`) and non-adjacent (`rule[prev3,prev1]`) conjunction tasks.

## Result — product ≈ RFF under ridge (Sub-claim B does NOT hold under regularization)

| task | product − RFF (per seed) | product>RFF+0.05 |
|---|---|---|
| adjacent | +0.007, +0.016, −0.006, −0.003, +0.006, +0.001 (mean **+0.004**) | **0/6** |
| non-adjacent | +0.024, +0.000, +0.030, +0.021, +0.024, +0.006 (mean **+0.018**) | **0/6** |

Under closed-form ridge, product (~0.94) ≈ RFF (~0.94) on both tasks — the Sigma-Pi conjunction features and random Fourier features (both strong nonlinear bases) perform EQUALLY. **Sub-claim B is settled as NOT robust under a regularized read-out:** the specific product/conjunction form is not privileged over a generic strong nonlinearity; the headline delta-rule advantage over RFF was the delta rule's difficulty fitting a dense random nonlinear expansion, not conjunction-specificity.

## ⇒ what this means (and does NOT change)

- **Sub-claim A stands, and it is what Rung 2+ rest on.** The load-bearing claim — a LINEAR read-out over a fixed reservoir provably cannot recover the input×input interaction, no matter the reservoir size — is unaffected. ANY strong nonlinear mechanism (a Sigma-Pi product, RFF, OR the selective SSM's input-dependent MULTIPLICATIVE gate) can recover it; a linear read-out cannot. The reservoir-LM long-range bound IS a missing-nonlinear-conjunction problem.
- **Rung 2–4b are unaffected.** The selective diagonal SSM's power is that its multiplicative gate is a *learned, recurrent, per-neuron* nonlinear mechanism that carries the conjunction across distance and is trainable transport-free — validated on the gated-conjunction task (Rung 2), on real text (Rung 3, beats the fixed reservoir + bigram, and the advantage GROWS with scale), and on the spiking substrate (Rung 4b). Its value never depended on the SPECIFIC Sigma-Pi product form being privileged over other nonlinearities.
- **The honest refinement:** "conjunctions are the missing ingredient" is correct and robust; "the *specific* product basis beats a generic nonlinear basis" is a small, readout-dependent effect that does not survive regularization. Recorded so the Rung-1 finding is not overclaimed.

## Files
- `research/runners/_reslm_conjunction_subclaimB_ridge_derisk.py`. Settles the qualification in `2026-07-13-PAST-RESERVOIR-conjunction-readout-...` (the adversarial-verify's recommended follow-on).
