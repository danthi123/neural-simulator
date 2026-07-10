# D3 — the transition on a biological credit rule: **PARTIAL**, and the residual is that I implemented feedback alignment, not the somatic nudge

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_delta_cleanerror_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** PARTIAL. The deep-credit channel works without weight transport and every anti-cheat collapses; the accuracy
reaches ~55% of the backprop reference. The cause is identified and is a **fidelity gap in my implementation of the rule**,
not a property of the register.

## What was attempted
The register's last host learning machinery is the transition δ's **backprop**: `d_lc @ Wc` transports the forward weight
into the backward pass, which biology cannot do. Replace it with the credit channel this project already validated on
spikes (`2026-07-07-D1-microcircuit-...clears-bar-on-spikes.md`, 0.964 held-out, batch-robust), keeping the one-step
(no-BPTT) schedule and the replay-taught write gate.

The network has **no agent label anywhere**: only the emission is observed, so only the top layer has target access.
`clause-code + a_curr + a_prev + patient → h → AGENT → EMISSION`.

## Result (6-seed, held-out-deeper; gated on `a_prev` / RETURN, never next-emission)

| arm | a_prev | RETURN | a_curr | emission | hidden moved |
|---|---|---|---|---|---|
| backprop reference (weight transport) | **0.654** | 0.724 | 0.823 | 0.667 | 0.466 |
| **clean-error, fixed-random feedback** | **0.340** | 0.331 | **0.454** | 0.510 | 0.512 |
| feedback lesion (descending error zeroed) | 0.206 | 0.190 | **0.188** | 0.228 | 0.000 |
| wrong sign (the teacher lies) | 0.256 | 0.174 | 0.190 | **0.038** | 0.167 |
| no-teaching null | 0.195 | 0.175 | 0.181 | 0.146 | **0.000000** |
| shuffled replay target | 0.141 | 0.195 | 0.440 | 0.494 | 0.519 |
| no replay | 0.141 | 0.193 | 0.440 | 0.494 | 0.513 |

*(Markov next-emission floor 0.390.)*

## What genuinely works
- **Deep credit reaches the never-taught agent layer with no weight transport.** `a_curr` 0.454 vs a feedback-lesion's
  0.188; emission 0.510 vs 0.228, above the Markov floor. The agent representation is learned without ever being taught.
- **Every anti-cheat collapses.** Wrong-sign anti-learns *coherently* (emission 0.038 — far below chance, so the credit
  is signed and aligned, not noise). The no-teaching null moves the hidden weights by **exactly 0.0** (the P₀ moat).
  Shuffled/no replay kill the held slot (0.141) while leaving `a_curr` intact (0.440) — the two learning problems are
  cleanly separable, as the previous rung claimed.
- **No weight transport, asserted per seed.**

## The residual, named exactly
Clean-error reaches 0.454 `a_curr` against backprop's 0.823, and beats it on only 1/6 seeds (seed 42, which the earlier
per-seed data show is the weak seed for the reference).

**Cause: I implemented plain feedback alignment on the clean error, which is not the rule D1's result rests on.** That
finding's adversarial audit attributes its accuracy to the **Urbanczik–Senn M2.6 somatic-rate feedforward rule** —

```
dW = eta * ( phi(u_P) - phi(v_basal) ) * r_pre     # the apical error NUDGES THE SOMA;
                                                    # the FF weights follow the NUDGED RATE
```

— and *not* to `dW ∝ e · phi' · r_pre`, which is what this runner computes. The two differ in what the weight follows: a
rate difference between a nudged and an unnudged soma, versus a linearised error times a derivative. I approximated the
source, and the approximation is what underperforms. This is the same discipline failure as reading a headline verdict
without its correction block — **an approximation of a cited rule is not the cited rule.**

## Honest reporting
- The backprop reference in this runner reproduces `train_pushpop(truncate=True, replay_gamma=1)` **to the digit**
  (a_prev 0.314 / RETURN 0.161 / a_curr 0.699 / pop-sep +0.042 on seed 42), so the harness is faithful and the gap is the
  rule, not the scaffolding.
- The first smoke ran on seed 42, which is the reference's *weak* seed — clean-error appeared to beat backprop there
  (0.345 vs 0.314). It does not, across seeds. **Smoking on one seed can invert a comparison.**
- Applying updates inside the clause loop rather than accumulating per minibatch was a step-size artifact worth ≈0.01,
  not the cause (checked, then fixed anyway).

## ⇒ Next
Implement the **actual M2.6 somatic nudge**: run the agent layer's soma twice per step — once with basal drive only, once
with the apical clean error added — and let the feedforward weights follow the **rate difference**. Gate on `a_prev` /
RETURN against the backprop reference (a_prev 0.654, RETURN 0.724, a_curr 0.823), keeping the controls that already
discriminate: feedback lesion, wrong sign, the no-teaching moat (hidden-moved must be exactly 0.0), shuffled replay, and
the no-weight-transport assertion. Read `_gnw_d1_spiking_bdsp_derisk.MicrocircuitBDSPNet.train_step` first — the exact
M2.6 form is already implemented there. If it closes, port via the committed `enable_bdsp` ∧ `enable_bdsp_microcircuit`
`sim/` path (additive, byte-identical-when-off).

## Files
`research/runners/_d3_delta_cleanerror_derisk.py`; raw `research/findings/raw/_d3_cleanerr_seed*.json`.
The rule's source of record: `2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`;
`sim/kernels.py: fused_bdsp_update`.
