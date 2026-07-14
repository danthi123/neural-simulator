# Validated-scale test (batched infra): a FIXED (untrained) selective channel does NOT lift margin-over-bag — an honest negative that pins the requirement: the LEARNED gate is what the selective mechanism needs at scale

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_batched_scale_selssm_derisk.py` (reuse-by-import of the batched reservoir-scale infra + a fixed selective-channel augmentation). numpy; NO `sim/` edit.
**Status:** HONEST NEGATIVE for the FIXED-gate cheap-first shortcut → the decisive test is the TRAINED gate at scale.

## What ran + why

Per the a-1 null-discriminator finding, the decisive validated-scale metric is `margin_over_BAG` (does the reservoir's dynamics beat a memoryless bag-of-prefix, and does that margin GROW with data — the reservoir-scale run CLOSED the FIXED reservoir as Ueda-bounded / bag-matches-reservoir at scale). The cheap-first question: does adding a **FIXED** (untrained, input-dependent `λ=σ(w·E[tok]+forget_bias)`, `w` random) selective channel to the read-out lift `margin_over_BAG`? (The Rung-3 `detached` control — a fixed input-dependent gate — still beat the fixed reservoir, so a fixed selective was a plausible cheap first look.)

## Result (np=300, n_train=2800, V=200, TinyStories) — the FIXED selective HURTS

`margin_over_BAG`: **res-only +0.140** (res_ce 3.197, bag 3.337 — the reservoir dynamics ARE load-bearing at this scale) · **res+sel +0.064** (sel_ce 3.272) · **sel_lift = −0.076** (res+sel is WORSE than res-only).

The fixed selective channel does NOT lift the scale margin — it slightly HURTS: an untrained input-dependent leaky integral of random projections adds read-out capacity/noise that dilutes a strong (np=300) reservoir's read-out, and the memoryless bag control gets no worse. So a fixed HOLD is not the ingredient at scale.

## ⇒ honest read (not a mechanism wall — a requirement)

This is consistent with the whole ladder: Rung 3's `detached` (untrained gate) < `selective` (trained gate); the frozen coupling's `fix` (input-independent) < `sel`. A FIXED/untrained selective channel is mostly noise; **the LEARNED input-dependent gate is what carries the value.** So the cheap-first fixed-gate shortcut is too cheap to evaluate the mechanism at scale — the decisive validated-scale test must use the TRAINED gate. (The trained gate helps at tractable scale, joint GO 6/6; the open question is whether its margin over the bag/bigram GROWS with data.)

The batched reservoir-scale infra trains only the READ-OUT over fixed features; it cannot train the selective gate (which needs the per-token online eligibility loop). ⇒ the decisive TRAINED-selective scale test uses the joint runner (which trains the gate) at increasing `n_train` (a tractable sequential run) OR a batched-gate-gradient build (the engineering follow-on for full validated scale). The trained-selective scale sweep (joint runner, nt 2800→5600) is the immediate next.

## Files
- `research/runners/_reslm_batched_scale_selssm_derisk.py`; raw `research/findings/raw/_reslm_scale_selssm_smoke.json`.
