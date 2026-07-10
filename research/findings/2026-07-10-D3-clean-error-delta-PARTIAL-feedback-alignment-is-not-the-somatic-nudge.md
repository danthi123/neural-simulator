# D3 — the transition on a biological credit rule: **PARTIAL** (corrected) — the residual is batch sensitivity, not the learning rule

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_delta_cleanerror_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** PARTIAL. The deep-credit channel works without weight transport, every anti-cheat collapses, and at batch
32-64 it **matches the host delta on next-emission (0.62)**. The held slot still trails the backprop reference. The cause is
**batch sensitivity of alignment-based credit** (predicted by this project's own D1 control probe) -- NOT, as this document
first claimed, a wrong learning rule; that attribution was refuted by reading the source and is corrected below.

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

## The residual — my first attribution was WRONG, and I refuted it by reading the source

**What I first wrote (and committed):** that clean-error underperforms because I implemented plain feedback alignment
rather than the Urbanczik-Senn M2.6 somatic-rate rule.

**That is false.** Reading `_gnw_d1_spiking_bdsp_derisk.py: MicrocircuitBDSPNet.train_step` — which I should have done
*before* writing the attribution — the M2.6 rule is implemented there as

```python
soma_err = (E * (1.0 - E)) * v_api    # phi'(u^P) * apical error = the M2.6 somatic delta
```

i.e. **in the small-signal linearisation the somatic nudge IS a derivative times the fixed-random clean error** — exactly
what this runner computes. That file states outright that its plain-FA arm "computes the SAME numeric credit the
MicrocircuitBDSPNet computes at the rate level." I asserted a distinction the source explicitly denies.

**Two hypotheses tested, both refuted (6-seed):**

| | a_prev | RETURN | a_curr | emission |
|---|---|---|---|---|
| `somatic_nudge` (elementwise phi', the exact M2.6 form) | 0.350 | 0.319 | 0.438 | 0.495 |
| `clean_error` (softmax Jacobian at the agent layer) | 0.340 | 0.331 | 0.454 | 0.510 |

Neither the rule's form nor the narrow 6-unit softmax Jacobian is the residual.

## The residual, actually: batch sensitivity (predicted by this project's own control probe)

D1's control probe already reported that alignment-based credit is **batch-sensitive** — Burstprop scores 0.924 at batch
32 and 0.513 at full batch, *degrading* as per-update averaging increases. This runner trained at **batch 256**.

**3 dev seeds, `clean_error`:**

| | batch 256 | batch 64 | batch 32 |
|---|---|---|---|
| a_prev | 0.340 | 0.414 | **0.482** |
| a_curr | 0.454 | 0.516 | 0.513 |
| **next-emission** | 0.510 | **0.621** | **0.616** |

At batch 32–64 the biologically-plausible credit channel **reaches the host delta's next-emission accuracy (0.619)** —
with **no weight transport and no backprop through time**. The held slot improves (0.34 → 0.48) but still trails the
backprop reference (0.65): a genuine, quantified, *open* residual, not an implementation error.

**The process lesson, for the second time today:** I named a cause from a headline rather than from the code, and the code
refuted it in ten minutes. `feedback_read_own_substrate_before_theorizing` applies to my own findings' internals, not just
to `sim/`.

## Honest reporting
- The backprop reference in this runner reproduces `train_pushpop(truncate=True, replay_gamma=1)` **to the digit**
  (a_prev 0.314 / RETURN 0.161 / a_curr 0.699 / pop-sep +0.042 on seed 42), so the harness is faithful and the gap is the
  rule, not the scaffolding.
- The first smoke ran on seed 42, which is the reference's *weak* seed — clean-error appeared to beat backprop there
  (0.345 vs 0.314). It does not, across seeds. **Smoking on one seed can invert a comparison.**
- Applying updates inside the clause loop rather than accumulating per minibatch was a step-size artifact worth ≈0.01,
  not the cause (checked, then fixed anyway).

## The 6-seed matrix at batch 32, and momentum (both hypotheses run to ground)

| arm (6-seed, batch 32) | a_prev | RETURN | a_curr | next-emission |
|---|---|---|---|---|
| backprop reference | 0.645 | 0.597 | 0.681 | 0.613 |
| **clean-error** | **0.438** | 0.406 | **0.502** | **0.586** |
| clean-error + momentum 0.9 | 0.456 | 0.416 | 0.525 | 0.510 |
| feedback lesion | 0.156 | 0.191 | 0.191 | 0.222 |
| shuffled replay | 0.310 | 0.268 | 0.490 | 0.580 |
| no-teaching null | 0.195 | 0.175 | 0.181 | 0.146 (hidden moved **0.000000**) |

**Momentum is not the missing piece either.** Copying `MicrocircuitBDSPNet`'s exact optimizer (heavy-ball, 0.9) lifts the
held slot (0.438 to 0.456) but *costs* next-emission (0.586 to 0.510), and degrades the backprop arm identically
(0.613 to 0.513). It over-steps at this learning rate and trades one metric for the other rather than closing the gap.

## Where this rung actually lands

With a biologically-plausible credit rule -- **no weight transport, no backprop through time**, and the write gate taught
by replay -- the register reaches:

* **next-emission 0.586 against the host delta's 0.619** (Markov floor 0.390): **95% of the reference**, with the credit
  demonstrably load-bearing (feedback lesion 0.222).
* **held slot a_prev 0.438-0.456 against backprop's 0.645**: **~70%**, with shuffled-replay at 0.310 and no-replay at 0.201.
* every mechanism control intact: the no-teaching null moves hidden weights by **exactly 0.0**; wrong-sign anti-learns
  coherently (emission 0.038); no weight transport, asserted per seed.

**The transition transfers to a biological rule almost completely; the held slot transfers to ~70%.** That asymmetry is
the honest, quantified deliverable of this rung, and it is consistent with the whole arc: the held slot is the hard part,
because nothing in the present rewards holding it.

## Corrected next steps
1. ~~Re-run the 6-seed matrix at batch 32~~ -- **done, above.**
2. ~~Momentum~~ -- **done, above: it trades a_prev for emission; it does not close the gap.**
3. The remaining candidate is an **optimizer-footing mismatch**: `MicrocircuitBDSPNet` normalises `upd/m` at `lr=0.3`,
   while this runner divides the top error by `B` at `lr=0.05`. Sweep `(lr, momentum)` jointly for the clean-error arm
   **before** concluding anything intrinsic.
4. If that does not close `a_prev`, the genuinely interesting question deserves a research gate rather than an assumption:
   **does alignment-based credit degrade on a recurrent, gated state variable (this register) relative to a static readout
   (D1's task)?**
5. Then port via the committed `enable_bdsp` and `enable_bdsp_microcircuit` `sim/` path (additive, byte-identical-when-off).

## OLD next (superseded)
1. **Re-run the full 6-seed matrix at batch 32** (the rule's regime), with all controls, and gate on `a_prev` / RETURN.
   Next-emission is already at the host reference there; the question is whether the held slot follows.
2. If `a_prev` still trails, the next candidate is **momentum + the per-layer homeostatic magnitude control**, both of
   which `MicrocircuitBDSPNet` has and this runner does not (`_MOMENTUM`, `_homeo_scale`) — read them before implementing.
3. Then port via the committed `enable_bdsp` ∧ `enable_bdsp_microcircuit` `sim/` path (additive, byte-identical-when-off)
   and re-run the register end-to-end.

## Files
`research/runners/_d3_delta_cleanerror_derisk.py`; raw `research/findings/raw/_d3_cleanerr_seed*.json`.
The rule's source of record: `2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`;
`sim/kernels.py: fused_bdsp_update`.
