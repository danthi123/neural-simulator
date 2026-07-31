---
type: finding
status: corrected
date: 2026-07-02
mechanism: microcircuit-credit
---

# EMERGE-5c — GO: the Sacramento–Senn microcircuit's ACTIVE CANCELLATION is dramatically more noise-robust than Burstprop's raw burst-rate estimation (carry the microcircuit to the substrate)

**2026-07-02 (autonomous; substrate ladder rung 2 — the decided lever from EMERGE-5b).** Runner `research/runners/_emerge5c_microcircuit_noise_derisk.py`; result `research/findings/raw/_emerge5c_microcircuit_noise.json`. Reuse-by-import (EMERGE-1b + EMERGE-3 + EMERGE-5 + EMERGE-5b machinery); NO `sim/` edit; CPU; run capped at 4 workers (owner gaming — light contention).

## Why this ran
EMERGE-5 found Burstprop's rate→spike credit degrades under finite-sample burst noise; EMERGE-5b's credit-vs-readout diagnostic isolated it to **credit quality** (a clean readout on the frozen spiking-Burstprop rep recovers only ~0.622 vs the rate ceiling ~0.796) and decided the next lever: the microcircuit's **active interneuron cancellation** (structurally different — does it survive the same noise?). Two prior levers were eliminated first (width — oracle won't scale to 768; naive population-averaging — mathematically identical to the already-tested S-sweep).

## The head-to-head (width-384, same depth-2 task/seeds, same S=300 finite-sample budget, 3 seeds)
Noise model = `Binomial(S, phi(u^P))/S` on the credit-carrying firing rates — the SAME injection point as EMERGE-5's Burstprop. Output error target-exact in both (clean logits; the noise is in the HIDDEN credit channel). Eval/probe read the clean analytic rep.

| arm | own acc (mean) | clean-readout-on-rep (mean) |
|---|---|---|
| **spiking microcircuit** | **0.971** (0.994/0.947/0.972) | **0.975** (0.989/0.969/0.967) |
| spiking Burstprop (EMERGE-5 ref) | 0.505 | 0.622 |
| microcircuit rate ceiling (sanity) | 0.978 | — |
| Burstprop rate ceiling | 0.796 | — |
| untrained (random-features floor) | — | 0.488 |
| apical-feedback lesion (anti-cheat) | — | 0.488 (= floor ✓) |
| no-teaching null (anti-cheat) | — | 0.488 (= floor ✓) |

## Verdict: GO
Under the same finite-sample noise where Burstprop's representation degrades to 0.622, the spiking microcircuit builds a rep that reads out at **0.975 — essentially its own noise-free rate ceiling (0.978)**. Multi-seed, with both anti-cheats load-bearing (killing the apical feedback OR the teaching signal collapses it to the random-features floor). **Active interneuron cancellation is the noise-robust rung-2 credit rule — carry the microcircuit (not Burstprop) toward the `sim/` two-compartment port.**

**Mechanistic reading (why):** Burstprop must *locally estimate* the top-down credit as a burst fraction (`Binomial(k,p)/k`) — inherently high-variance, and the noise IS the credit. The microcircuit *receives* the descending error through a fixed feedback pathway (the interneuron cancels the top-down, leaving the error driven by the exact output nudge); finite-sample rate noise only *modulates* that credit (via the `phi'(r)` factor + noisy pre-synaptic rates), it does not constitute it. So the microcircuit's credit magnitude comes from the exact top signal propagated down, not from a noisy local estimate — a genuine, biologically-sensible structural advantage.

## Honest caveats (record, do not gloss)
1. **The comparison slightly under-noises the microcircuit's credit.** Noise was injected on the firing rates (matching EMERGE-5's injection point), but the descending apical error `e_upper` carried the exact output nudge down (only `phi'(r_obs)`-modulated); the interneuron cancellation *difference* (`r_upper − r_int`) was not itself re-estimated from noisy spike counts at each descent step. A **stricter test** should inject finite-sample noise into that cancellation difference too. The structural argument predicts the microcircuit still wins (its error comes from the exact top nudge, robust to per-step rate noise), but that is TO BE TESTED, not assumed — this is the immediate verification before committing to the months-scale substrate build.
2. **Cosmetic numerical overflow** (`RuntimeWarning: overflow in matmul`) in the *reused* rate-microcircuit's interneuron-maintenance rule (`_emerge3_microcircuit_derisk.py:229`) at width-384 (EMERGE-3's default was width-64). It is in the maintenance loop that does NOT feed the credit (read in the self-predicting form), the rate ceiling came out valid (0.978, not NaN), and the spiking-microcircuit arm drops the maintenance loop entirely (unaffected). Worth a numerical guard if the rate microcircuit is run wide again, but it does not affect this result.

## STRICT-NOISE UPDATE (caveat #1 closed) — 2026-07-02
Ran the stricter test flagged above: `SpikingMicrocircuitMLP(cancel_noise=True)` injects additive finite-sample noise `sd=sqrt(2·r·(1−r)/S)` into the descending apical error itself (the interneuron cancellation difference `r_upper − r_int`, a difference of two independent S-sample spike-rate estimates) at the output AND each descent step — so the microcircuit's distinctive credit channel is noised, not just the `phi'` modulation. Result (width-384, S=300, 3 seeds, 4-worker light-contention run):

| arm | clean-readout (mean) |
|---|---|
| **microcircuit STRICT-noise** | **0.981** (0.983/0.969/0.992) |
| microcircuit rate-noise-only | 0.975 |
| microcircuit rate ceiling | 0.978 |
| spiking Burstprop | 0.622 |
| strict-lesion / strict-null (anti-cheats) | 0.488 / 0.479 (= floor ✓) |

**GO (STRICT) — the caveat is closed.** Strict-noise clean-readout (0.981) ≈ rate-noise-only (0.975) ≈ the noise-free ceiling (0.978), ≫ Burstprop (0.622); lesion + null collapse to floor. The cancellation-difference noise barely dents the microcircuit because the credit magnitude is anchored by the exact top-down output nudge propagated through the fixed feedback pathway — a genuine structural robustness, not an artifact of under-noising. Caveat #2 (the width-384 W_PI-maintenance overflow) remains cosmetic (unused-for-credit maintenance loop; the rate ceiling is valid; the spiking arm drops that loop) — a numerical guard is deferred, not load-bearing.

## State of rung 2 (updated)
- Rate→spike credit: Burstprop degrades under finite-sample noise (EMERGE-5/5b); **the microcircuit does NOT** (this doc) — active cancellation is the noise-robust mechanism.
- Immediate next: the stricter cancellation-difference-noise test (verify the GO survives injecting noise into `r_upper − r_int`). If it survives → the microcircuit is confirmed as the credit rule for the `sim/` two-compartment port (rung 4); scope that build (research-gated).
- Deferred: fix the width-384 W_PI-maintenance overflow guard if the rate microcircuit is run wide.

## ⚠️ ATTRIBUTION CORRECTION (2026-07-07, from the D1-microcircuit adversarial-verify wjn6hxyuu)
The verdict "ACTIVE CANCELLATION is MORE noise-robust than raw burst-rate estimation" carries an **attribution error**: the noise-robustness is the **clean-error credit CHANNEL** (the M2.6 somatic-rate feedforward rule descending the interneuron-cancelled clean apical error `φ′(E)·(Yᵀ@e)` = clean-error feedback alignment), **NOT an active dynamical cancellation loop**. This finding's own HONEST_NOTE already says interneuron maintenance "does not affect the within-step FF update"; the D1 controls confirmed it on the spiking-substrate reference — a clean-error-FA net with NO interneuron reproduces the microcircuit accuracy byte-identically, and killing the burst signal (`beta=0`) leaves it unchanged (burst is inert to the microcircuit's weight update). Re-read this finding as: **clean-error credit > noisy burst-fraction credit** (real, reproduced, batch-robust); the interneuron cancellation is the biological *realization* of the clean channel, load-bearing on the substrate for the burst READOUT — its accuracy causality at DEPTH is the open D2 test. See `2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`.

## Artifacts
`research/runners/_emerge5c_microcircuit_noise_derisk.py` (+ `SpikingMicrocircuitMLP`, `--max-workers` cap for light-contention runs), `research/findings/raw/_emerge5c_microcircuit_noise.json`. Prior: `2026-07-02-emerge5b-credit-vs-readout-PARTIAL.md`, `2026-07-01-emerge5-noise-driven-self-organization-discovery.md`, `2026-07-01-emerge3-microcircuit-GO.md`.
