# Controlled-lag recurrent e-prop — the copy task's horizon-extension is a MEMORY-TIMESCALE ARTIFACT (the anti-cheats caught it on a controlled task); the discriminator must require RECURRENT COMPUTATION

**Date:** 2026-07-14 · **Status:** de-risk BUILT + eligibility grad-check-validated; the COPY-task result is an honest artifact (not genuine credit-driven learning); the recurrent-computation task is the pinned next step. numpy CPU; NO `sim/` edit.

## Context

The emergence-engine recurrent-language-cortex frontier: get a spiking recurrent cortex to LEARN long-range structure from a stream via a biological LOCAL rule. The prior attempt was adversarially REFUTED (`2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED`): the WikiText "deep-context" win was a credit-direction-INDEPENDENT memory-timescale artifact. The deep-research gate (Bellec-2020 e-prop, read in depth) diagnosed: the eligibility was already correct (faithful ALIF 2-component form); the failure was the TASK (a gameable next-token-CE metric with no controlled long-range dependency) + running the decisive controls on the short-horizon LIF arm. The gate's fix: a **delayed cued-recall (copy) task with a controlled dependency length T** on the **ALIF** arm, with genuine-learning anti-cheats.

## What was built

`research/runners/_reslm_controlled_lag_eprop_derisk.py` — the copy task `[STORE, x, f_1..f_T, RECALL, x]` (score predicting x at the RECALL position; provably beats no n-gram — the last n<T tokens are fillers/RECALL, independent of x) + an ALIF e-prop trainer carrying the **finite-difference-validated** Bellec-2020 2-component eligibility (`grad_check_alif` PASS: eps_h/eps_a vs local FD exact to ~1e-9; the e-prop total-grad vs full-FD residual is the expected off-diagonal-truncation, converging at tight spectral radius) + all credit-direction arms + the RECALL-position metric + a T-sweep. Reuse-by-import of the validated `RateReservoir` (ALIF).

## The result (seed 42)

- **T=5:** fixed=0.950 ≈ plastic=1.000 — the fixed ALIF reservoir's intrinsic adaptation memory ALREADY holds x for 5 steps → no discrimination (trivially within the fixed horizon). n-gram 0.190 ≈ chance 0.167 ✓; cue_scramble 0.115 (collapses) ✓ — the task is sound.
- **T=15 (fixed partial 0.365), lr_rec=0.01:** plastic=0.140 (WORSE than fixed, below chance) — the e-prop W_rec updates DESTABILIZE the good reservoir dynamics at the default lr.
- **T=15, lr_rec=0.0002 (tuned):** plastic=**0.730** EXCEEDS fixed=0.365 — BUT the genuine-learning **anti-cheats FAIL**: symmetric (TRUE gradient)=**0.140** HURTS (should be the ceiling ≥ plastic), sign_flip=**0.520** does NOT collapse to chance. ⇒ the SAME memory-timescale-artifact signature as the WikiText refutation, now on a controlled task.

## Diagnosis (honest)

The single-cue COPY task is solved by the ALIF **adaptation state directly** (the read-out reads `[h, a]`, and `a` holds x), so a learned W_rec is UNNECESSARY within the fixed horizon and only PERTURBS the good dynamics — a bidirectional memory-timescale nudge (any structured W_rec change helps a bit; the random-FA direction happens to help more, its negation helps less, the true gradient over-optimizes the shallow loss and HURTS recall). This is exactly the artifact class the anti-cheats were built to catch — and they caught it. **The de-risk WORKED: a single-cue hold task does NOT elicit genuine credit-driven recurrent learning; W_rec learning is not load-bearing for it.**

## The pinned next step (the real discriminator)

The task must REQUIRE recurrent computation a memory-timescale nudge cannot fake: the answer must be a **nonlinear function of temporally-separated cues** that the learned recurrence MUST compute — e.g. **delayed-XOR** (STORE x1 … x2 … RECALL → x1⊕x2) or **evidence-accumulation** (multiple weak cues → majority). On such a task the fixed reservoir cannot solve it by holding (no single value to hold), so genuine plastic learning (if it works) yields the true signature: **plastic > fixed AND symmetric ≥ plastic AND sign_flip collapses AND cue_scramble collapses**. This is Bellec's actual store-recall/evidence-accumulation regime (which e-prop+ALIF solves) — vs the copy task, which the ALIF alone solves. Build the XOR/accumulation task mode into the runner + re-run the arms.

## Bottom line

The controlled-lag de-risk is built + its eligibility is grad-check-validated + its anti-cheats are proven (they caught the memory-timescale artifact on a controlled copy task, confirming the WikiText refutation's lesson). The honest finding: **plastic e-prop's apparent horizon-extension on a single-cue copy task is a memory-timescale artifact, not genuine credit-driven learning** — the discriminator must require recurrent computation (delayed-XOR / evidence-accumulation), the pinned next step on this genuinely-hard frontier (the field: long-range from a local rule is partially-solved at seconds-scale, open at LRA scale).
