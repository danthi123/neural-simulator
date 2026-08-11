---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes — LEARNED FEEDBACK (Kolen-Pollack): does transport-free *learned* feedback reach the 3rd hidden layer where FIXED-random DFA could not
lane: gap#4 / deep-credit
verdict: 6-VALID-SEED GO (CONFIRMED). Transport-free KP-LEARNED feedback REACHES the 3rd hidden layer — over 6/15 ceiling-holding seeds it closes 66% of the BP-depth-2→BP-depth-3 FIT gap (>=50% bar) where fixed-DFA stays at the mean-predictor floor (-85%); freezing the feedback (kp-lr=0) collapses it to -40% (the win is DUE TO learning G). This SURPASSES the prior fixed-feedback wall at de-risk level. Residual: KP reaches but does not yet MATCH the oracle (~forward optimization).
seeds: [42, 43, 44, 104, and 2 more of the 6/15 testable pool]
artifacts:
  - research/findings/raw/_gap4_learned_feedback_smoke.json
  - research/findings/raw/_gap4_learned_feedback_6valid.json
runner: research/runners/_gap4_learned_feedback_derisk.py
instrument: the layer-3-credit-fidelity harness of 2026-08-11, extended with (1) LEARNED transport-free feedback (Kolen-Pollack: each hidden feedback matrix G_l is updated by the SAME Adam step as W_l, transposed, so G_l co-adapts toward W_l^T by a LOCAL rule — never copied), and (2) PER-SEED ceiling-gating (a seed is scored only if BP-depth-3 fits AND BP-depth-2 underfits). Arms: BP-depth-3 (ceiling), BP-depth-2 (separation), KP (learned), fixed-DFA (prior baseline), frozen-KP=fixed-FA (the lever endpoint), permuted-KP. SIM_BACKEND=numpy.
---

# gap#4 LEARNED FEEDBACK — Kolen-Pollack transport-free feedback REACHES the 3rd hidden layer where fixed DFA could not (smoke GO)

The prior finding (`2026-08-11-gap4-layer3-credit-fidelity-transport-free-DFA-does-NOT-reach-the-3rd-layer-...`) banked
the KNOWN fixed-feedback deep-layer limit: on a tent^3 FIT target where backprop-depth-3 fits and backprop-depth-2
underfits, transport-free DFA with a FIXED random feedback matrix sits at the mean-predictor / depth-2 floor (closes
~0% of the fit gap) — error does NOT reach the 3rd hidden layer. The NAMED surpass was LEARNED feedback. This de-risk
builds it and the mechanism WORKS.

## The surpass — Kolen-Pollack learned feedback (Kolen & Pollack 1994; Akrout et al. 2019, "Deep Learning without Weight Transport")

<!--derived-->
The credit runs a SEQUENTIAL backward pass in which each hidden feedback matrix G_l replaces W_l^T. Under KP, G_l is
updated by the SAME (Adam) step as the forward weight W_l, transposed, so W_l^T and G_l receive identical increments
and their DIFFERENCE stays at its random-init value while the accumulated matched updates grow to dominate it —
cos(G_l, W_l^T) → 1 EMERGES from co-adaptation. This is TRANSPORT-FREE: G_l is never read from W_l (no `G = W.T`
copy); the credit path computes `delta @ G`, never a forward W^T; the KP update uses only the layer's presynaptic
activity and local error (the same quantity that trains W_l). Init cos(G,W^T) ~ 0 (separate random stream, max |.| =
0.29 across seeds, deep-layer a1 = -0.002), rising to 0.93 at the deep layer through training — learned, not
transported.

## Smoke result (3/3 testable seeds; `research/findings/raw/_gap4_learned_feedback_smoke.json`)

<!--derived-->
Width-16 tanh/ReLU 3-hidden net, tent^3 FIT target, 8000 epochs, seeds 42/43/44 (all ceiling-holding +
depth-separating). Testable-seed-mean MSE: BP-depth-3 oracle 2.1e-05, BP-depth-2 0.0329, mean-predictor 0.0419;
**KP-learned 0.0140, fixed-DFA 0.0369, frozen-KP(=fixed-FA) 0.0400**. As a fraction of the BP2→BP3 fit gap closed
(1.0 = reaches the oracle): **KP = 54.7% (per-seed 62/37/65%), fixed-DFA = -16.9%, frozen-KP = -30.0%**. So
transport-free LEARNED feedback crosses decisively to the depth-3-oracle side of the depth-2 floor, while both
fixed-feedback baselines stay at (or below) the mean-predictor. The deep-layer (a1) DFA-vs-BP credit-direction
alignment goes fixed-DFA -0.26 → KP +0.47 — the learned feedback recovers the backprop credit DIRECTION at the deep
layer, which fixed feedback anti-aligns. All 8 Verdict preconditions hold → GO. (Backend numpy/cpu.)

## ⭐ 6-VALID-SEED CONFIRMATION (coordinator-run) — GO (`research/findings/raw/_gap4_learned_feedback_6valid.json`)

<!--derived-->
The robust-ceiling run (15-seed pool, per-seed ceiling-gated, `--min-testable 6`, epochs=8000) yields **6/15 testable
(ceiling-holding) seeds and status GO**. Over those 6 valid seeds: **KP-learned feedback closes 66% of the BP-depth-2→
BP-depth-3 fit gap** (KP loss 0.0092 vs bp3 1.1e-05 / bp2 0.024 / mean-predictor 0.042), where **fixed-DFA closes only
−85%** (stays at the mean-predictor). The `kp_beats_dfa`, `go_kp`, `dfa_fails`, `perm_ok`, `kp_moved_all`, and
`fa_frozen_all` checks ALL hold. **Freezing the feedback (kp-lr=0 → fixed-FA) collapses the gap-close to −40%** — the
win is DUE TO learning the feedback G. Transport-free confirmed: `cos(G,Wᵀ)` init 0.253 → deep-layer final **0.826**
(co-adapted through training, never copied — cos starts far from 1); deep (a1) DFA-vs-BP credit alignment −0.236 (fixed)
→ **+0.696** (KP). **This CONFIRMS the smoke at 6 valid seeds: transport-free learned feedback reaches the 3rd hidden
layer where fixed-feedback DFA could not — the gap#4 deep-credit surpass, at de-risk level.**

## The lever + anti-cheats (all executed, not asserted in prose)

<!--derived-->
- **Lever (kp-lr=0):** freezing G (feedback-learning OFF → fixed-FA) collapses the fit gap-close from +55% to -30%
  (down to the fixed-feedback floor). The win is DUE TO learning the feedback, not the sequential path alone —
  `lever()` confirms KP moved G every step while frozen-KP left it unchanged.
- **fixed-DFA baseline fails:** reproduces the prior wall (-17% gap-close) — the thing KP beats. `control()` confirms
  the KP fit differs from the fixed-DFA fit (|sep| = 0.023).
- **permuted-target KP:** per-step target reshuffle → no fit (loss stays at the floor).
- **transport-free:** init cos(G,W^T) not a copy (max |.| = 0.29 « 1.0), rising only through training.
- **per-seed ceiling-gating (the instrument fix):** the prior 6-seed was UNDEFINED because the width-8 ceiling was
  seed-fragile. Here each seed is scored only if BP-depth-3 fits (loss ≤ 2% of var) AND BP-depth-2 underfits (gap >
  5% of var); the verdict is computed on TESTABLE seeds and N_testable/N_total is reported. A wider net (width 16)
  makes the BP-depth-3 ceiling robust; per-seed gating handles the residual fragility (at width 16 depth-2 sometimes
  ALSO fits, so ~1/3–1/2 of seeds are testable — the honest resolution is to gate, not to assume).

## Scope / next (per THE LAW — the capability is now OPEN, headline pending)

<!--derived-->
- **Smoke, not the headline.** 3 valid seeds (all testable). The 6-valid-seed validation is the caller-launched run
  over a wider seed pool (`--seeds 42,43,44,45,46,47,48,49,50,100,101,102,103,104,105` → 6 testable verified:
  42,43,44,46,50,104). GO must survive that before the wall-ledger flips.
- **KP under-trains at 3000 epochs (a false-negative trap).** Feedback-alignment converges slower than backprop;
  seed-42 KP gap-close is -6% at 3000 epochs but +62% at 8000. The mechanism was there; the training budget was not.
- **A residual, honestly.** KP closes ~55% of the gap, not ~100% — it REACHES the deep layer (crosses the midpoint)
  but does not yet MATCH the oracle. Next levers: more epochs, weight-mirror (Akrout noise-driven alignment) as an
  alternative learned-feedback rule, or the φ′-vanishing fix. Matched weight-decay (kp_wd>0) drives cos(G,W^T)→~1.0
  but HURTS the fit here, so kp_wd=0 was kept — a clue the residual is the forward optimisation, not the feedback
  alignment.
- **Data-only, NO `sim/` edit.** Additive, default-off runner. The BP oracle gradient is computed ONLY for the
  REPORTED alignment read-out, never applied to the KP net — the training credit is fully transport-free.
