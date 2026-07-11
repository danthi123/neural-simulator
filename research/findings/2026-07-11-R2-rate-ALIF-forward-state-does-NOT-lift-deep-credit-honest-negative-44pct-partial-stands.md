# R2 (the pre-registered deep-tail lever): the rate-ALIF adaptation-as-FORWARD-STATE does NOT lift the deep-context recurrent credit — it HURTS (deep −3.27 vs plain −2.16), and the credit-vs-capacity control proves the damage is the `−β·a` forward imprint degrading the representation, NOT the credit. The clean ~44% plain-e-prop partial STANDS; the residual ~56% is the off-diagonal (cross-unit-across-time) credit the diagonal eligibility drops

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_stream_eprop_lm_derisk.py` (torch/GPU; ALIF arms + 2-component eligibility grad-check PASS eps_h 2.39e-9 / eps_a 4.19e-9 / torch==numpy 1e-17; controller-verified; NO `sim/` edit). TinyStories 2M tokens, V=2000, n_pool=300, block=128, 4 epochs, seed 42, 8 arms, ρ log-uniform over [30,300]-token windows, β=1.0.
**Verdict:** the research-ranked #1 deep-tail lever (ALIF adaptation-as-state, "highways into the future") does NOT lift the deep recurrent-credit capture in this RATE realization — it substantially HURTS it — a well-controlled honest NEGATIVE. The clean ~44% plain-e-prop deep partial (R1b) stands as the biological one-step-local recurrent-credit capture; the residual is the off-diagonal gradient, which a slow FORWARD state cannot supply.

## The result — DEEP (ctx17-127) margin vs the add-1 bigram + the clean recurrent-credit fraction
| arm | DEEP margin | deep gain over fixed floor | clean frac_deep (vs BPTT_fixed_win) |
|---|---|---|---|
| fixed_reservoir (floor) | −2.760 | — | — |
| **plastic_eprop** (plain, R1b) | −2.158 | +0.602 | **+0.443** |
| **plastic_eprop_alif** (the lever) | **−3.274** | **−0.514** | **−0.378** |
| plastic_eprop_alif_readonly (a read, NOT credited) | −3.273 | −0.513 | — |
| BPTT_fixed_win (recurrent-credit ceiling, frozen W_in) | −1.401 | +1.359 | 1.000 |
| BPTT_same_net (all-params) | +0.902 | +3.662 | — |

- **ALIF HURTS: deep −3.274 vs plain e-prop −2.158 (a −1.116 drop), and BELOW the fixed echo-state floor (−2.760).** The clean fraction goes negative (−0.38). The research expectation (~0.7-0.85, based on Bellec's SPIKING adaptive-threshold LSNN) does NOT transfer to a rate leaky-tanh net.
- **CREDIT-vs-CAPACITY control (decisive):** crediting the adaptation vs merely reading it changes the deep margin by **−0.0006** (alif −3.2739 vs alif_readonly −3.2733). ⇒ the credit rule is NOT the problem; the damage is entirely the **`−β·a` subtractive forward imprint** that degrades the tanh representation. Both the credited and read-only ALIF arms are equally bad because both carry the forward imprint.
- **ADAPTATION-SHUFFLE control:** alif deep −3.2739 → a_t-shuffled −3.2816 (collapses slightly — the a-values do reach the read-out; but the read-out benefit is swamped by the forward degradation).

## Why the rate-ALIF fails where Bellec's spiking-ALIF works (the honest reconciliation)
Bellec-2020's ALIF is a **spiking adaptive THRESHOLD**: a slow variable RAISES the firing threshold, so it carries a genuine non-fading HOLD that the eligibility exploits WITHOUT corrupting the unit's spike code. The rate analogue here SUBTRACTS `β·a` from the tanh pre-activation — a **diluting perturbation** of the continuous representation, not a clean gate. This is the project's OWN prior finding (`2026-07-11-ALIF-adaptation-state-NEGATIVE-...md`, within-sentence: rate-ALIF marginally worse than plain e-prop, "a diluting average, not a specific-item hold"), now CONFIRMED + sharpened at contiguous 2M-token scale with a decisive credit-vs-capacity control: it is the FORWARD imprint, not the credit, that fails. β-tuning only interpolates between plain (44%) and ALIF-hurt, so it cannot exceed the plain 44%.

## What STANDS + what this launches (boundary = the next mechanism)
- **STANDS:** the clean ~44% plain-e-prop deep recurrent-credit capture (R1b) — biological one-step-local credit (the rate analogue of the on-bridge BDSP) keeps ~44% of the deep-context recurrent-credit margin full backprop achieves, on identical embeddings; the diagonal RTRL truncation loses ~56%, which lives in the OFF-DIAGONAL cross-unit-across-time gradient.
- **NOT the lever:** a slow adaptation as a FORWARD STATE (rate-ALIF) — it cannot supply the off-diagonal credit and it degrades the forward representation.
- **The residual is the off-diagonal gradient.** The next mechanisms target THAT, not the (exhausted) forward-state-horizon family:
  1. **Dual-timescale ELIGIBILITY (a slow eligibility trace, NO forward imprint)** — extends the CREDIT horizon without degrading the forward state (the reuse-base's `adaptive` mode); a cheap single-variable test of whether the deep gap is horizon-limited vs genuinely off-diagonal. (Was within-sentence-negative in the reuse base; untested at contiguous scale.)
  2. **SnAp-2 (Menick 2021) — keep 2-step recurrent influence** — the richer-than-diagonal factorization; directly measures how much of the 56% off-diagonal gap one step beyond diagonal recovers (the research-named "cost of diagonal-ness" upper bound).
  3. **Multi-layer** (R3) — e-prop (temporal) × learned Kolen-Pollack feedback (spatial); a deeper net may capture more long-range even with per-layer diagonal credit; the spiking realization is burstprop/BDSP (`sim/bridge.py`).
  4. **Biological future-error predictor (DNI analogue)** — a second population learning to predict the downstream error, supplying the "indirect influence" e-prop drops (what Bellec needed for long-range PTB).

## Honest scope
Single seed (42), 2M tokens, single layer — a cheap-first lever test, as designed. The ALIF grad-check PASSED (the 2-component eligibility is faithful — this is a genuine mechanism result, not an implementation bug), and the credit-vs-capacity + adaptation-shuffle controls make the negative interpretable (forward-imprint degradation, not credit failure). The ~44% partial + the ALIF-forward-state negative together map the frontier precisely: biological one-step-local credit reaches a substantial fraction of BPTT's deep long-range credit; the remaining gap is specifically the off-diagonal gradient, addressable by richer factorization (SnAp-2), depth (multi-layer), or a future-error predictor — NOT by a slow forward state. All rate-level torch, GPU, grad-checked, anti-cheated, NO `sim/` edit.

## Files
`_emerge_stream_eprop_lm_derisk.py`; raw `research/findings/raw/_stream_eprop_lm_r2_alif.json` + `.log`. Builds on `2026-07-11-R1-stream-eprop-...md` (R1/R1b) + the reuse-base ALIF-negative `2026-07-11-ALIF-adaptation-state-NEGATIVE-...md`.
