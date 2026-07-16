# DECISIVE NEGATIVE (dose-response + multi-seed + 4-lens adversarial-verify): the off-diagonal MDGL is rate-validated but does NOT cleanly spike-realize on point neurons — population averaging reveals it adds MAGNITUDE, not SIGN-CORRECT directional credit. The biological path past this boundary is REPLAY (SWR-replaces-BPTT), which the project already holds. On-bridge realization is NOT warranted.

**Date:** 2026-07-16 · **Status:** DECISIVE NEGATIVE (adversarially verified). Completes the off-diagonal recurrent-credit arc. Answers the owner's "what mechanism lets a spiking cortex learn deep recurrent representations that we're missing?" at the mechanism level.
**Runners:** `_mdgl_replica_popcoded_spiking_derisk.py` (population coding done right: K noisy replicas, ensemble-averaged credit, dynamics preserved) · `_mdgl_verify_workflow.js` (6-seed + 4 adversarial skeptics + synthesis, run `wf_ed8dbff4-f2f`). numpy-CPU; NO `sim/` edit.

## The question this closes

The off-diagonal cross-neuron temporal credit (the term e-prop's diagonal RTRL zeroes) is the missing mechanism for training recurrent weights; MDGL (Liu 2021 cell-type one-hop neuromodulation) is its biological realization. It is **rate-VALIDATED clean-directional** (+48–64% of the diagonal→BPTT gap, sign-flip HURTS, `_mdgl_offdiagonal_credit_derisk.py`). The spiking port DEGRADED (single-neuron marginal, first pop-coded attempt magnitude-confounded). The named surpass — **population coding** (the lever that closed FEEDFORWARD spiking credit this session) — was tested cleanly here: K independent NOISY replicas of the SAME non-degenerate net, ENSEMBLE-AVERAGE the credit, forward dynamics untouched.

## The result — two independent tests + a 4-lens adversarial verify, all converging on NEGATIVE

**Test 1 — DOSE-RESPONSE (seed 42, gain 0.2, fixed eval_n=32 across all arms → read-out denoising identical, isolating the training-credit effect).** A real population-averaging-recovers-directionality effect requires the sign-flip collapse `(e−f)` to grow *more positive* with N. It does NOT:

| N | offdiag (MDGL−eprop) | collapse (eprop−signflip) |
|---|---|---|
| 8 | +0.470 | **−0.525** (flip wins) |
| 12 | +0.100 | **−0.340** (flip wins) |
| 16 | +0.110 | +0.155 (clean) |
| 24 | +0.250 | **−0.275** (flip wins) |
| 32 | +0.155 | **−0.380** (flip wins) |

**No monotonic dose-response** — the N=16 clean point is an isolated (N, gain) fluke; the sign-flip wins at N=8/12/24/32.

**Test 2 — MULTI-SEED at N=16 (full arm set: BPTT ceiling, e-prop, MDGL×5 gains, sign-flip, zero-Γ, permuted).** GO 2/4 seeds — and the two "GO" seeds don't even share a clean gain (seed 42 @ gain 0.2; seed 43 @ gain 0.15; seeds 101/102 have NO clean gain at any gain). On seeds 101/102 the sign-FLIPPED (wrong-direction) control BEATS the correct-sign MDGL at most/all gains (s101 F 0.95/0.98 > M 0.45/0.36 at every non-chance gain).

**The 4-lens adversarial-verify (the decisive part):**
- **gain-cherry-pick → REFUTED (0.92):** no single gain is clean across a majority of seeds; `best_clean_gain` is cherry-picked per seed.
- **margin-vs-noise → REFUTED (0.85):** the sign-flip control that DEFINES "clean directional" is itself pure seed-noise (F at g0.15 is 0.34/0.95/0.65 across seeds — a range >10× the e-prop band); the "F collapses" event does not replicate.
- **eval-denoising → not-refuted:** the fixed-eval_n design + the zeroG control (which sits at e-prop every seed while MDGL rises to 0.46–0.67) prove the effect is NOT a read-out artifact — the off-diagonal TERM genuinely adds information beyond diagonal e-prop.
- **ceiling-gap → not-refuted:** BPTT 0.995–1.000 vs e-prop 0.30–0.37 (≈chance) — a real ~0.65 gap the off-diagonal *could* close; permuted at chance (no leak).

## The honest verdict (adversarial synthesis)

**The off-diagonal term adds MAGNITUDE/variance, not SIGN-CORRECT structure, on point-neuron spikes.** The load-bearing evidence: a *wrong-sign* copy of the Γ term reproduces (or beats) the correct-sign lift at most operating points — so whatever population averaging surfaces is not *directional* credit. "Directional" requires the sign-flip to collapse to e-prop; instead the wrong sign is frequently as good or better. It closes only ~1/3 of the e-prop→BPTT gap, non-directionally, at a cherry-picked seed/gain. **NOT a pure null** (the off-diagonal term is real — zeroG collapses to e-prop, not a read-out artifact — the task genuinely needs cross-neuron credit), but a NEGATIVE for the property that matters: sign-correct directional credit on spikes.

**⇒ On-bridge realization is NOT warranted** — building a spiking implementation would harden an effect that fails its own directionality control and doesn't replicate (GO 2/4, no shared clean gain, no dose-response). Keep it off the bridge.

## What this LAUNCHES (a boundary is an undiscovered mechanism, not an endpoint)

The rate mechanism is clean; point-neuron spikes degrade the *direction*. The missing ingredient is a credit whose SIGN/structure is load-bearing on spikes. The roadmap's own biology prior (`2026-07-15` deep-credit plan) names the biological answer: **cortex likely does NOT compute the off-diagonal Jacobian ONLINE — it approximates via REPLAY** (sharp-wave-ripple replay / SWR-replaces-BPTT), long NMDA-plateau within-dendrite eligibility, and neuromodulator volume-transmission. And the project has ALREADY partially validated SWR-replaces-BPTT (the D3 discourse event-register, `2026-07-10`: replay replaced backprop-through-time, 6-seed GO). ⇒ the online one-hop MDGL is a rate-idealization; the SPIKING off-diagonal credit is done OFFLINE by replay — a mechanism the project holds. IF the mission later needs trainable recurrent credit on spikes, the replay path (not an online one-hop) is the next de-risk.

## The complete off-diagonal arc conclusion (for the owner's directive)

The owner steered this session to "find + implement the learned-cortex mechanism we're missing." The arc concluded it decisively:
1. **FOUND:** the missing mechanism = recurrent off-diagonal cross-neuron temporal credit (e-prop's diagonal zeroes it; delayed-XOR: diagonal at chance, BPTT 1.0).
2. **RATE-VALIDATED:** MDGL closes it cleanly at rate (+48–64%, sign-flip hurts).
3. **SPIKING NEGATIVE (this finding):** it does NOT cleanly spike-realize on point neurons — population averaging reveals magnitude, not sign-correct direction. Decisive (dose-response + multi-seed + 4-lens adversarial).
4. **BIOLOGICAL PATH past it:** REPLAY (offline consolidation), roadmap-biology-prior-named + project-partially-held.
5. **OFF-CRITICAL-PATH for fluency regardless** (ROADMAP §12: the fluency lever is the learned INPUT representation + composing machinery + SCALE, not trainable recurrent weights — which the roadmap found *counterproductive* for language).

**⇒ the mechanism hunt the owner directed reached an honest, complete conclusion.** The rate-vs-spike degradation is the project's recurring theme (the point-neuron limit, Mikulasch-Priesemann family); the biological escape (replay/offline) is known and held. The convergent evidence (this + the input-representation gate + the roadmap) is that the composing-machinery + memory + credit mechanisms are largely in place on spikes, and the remaining fluency gap is SCALE/DATA for the learned representations — the owner-reserved invest-wallclock decision.

NO `sim/` edit anywhere in the arc. Both remotes.
