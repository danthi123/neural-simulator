# R2b/R2c (GO, single-seed anti-cheat-confirmed): a slow DUAL-TIMESCALE ELIGIBILITY trace lifts biological one-step-local recurrent credit from ~44% → ~81% of the deep-context (long-range) recurrent-credit margin full backprop achieves — the deep tail was own-unit-HORIZON-limited, and a spread of eligibility timescales (NO forward-state change) recovers it where the ALIF forward-imprint FAILED; the dualtc-SHUFFLE anti-cheat confirms the lift is genuine credit-structure, not magnitude

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_stream_eprop_lm_derisk.py` (torch/GPU; eligibility grad-check PASS; controller-verified; NO `sim/` edit). TinyStories 2M tokens, V=2000, n_pool=300, block=128, 4 epochs, seed 42, a_slow=0.01 (~100-token credit horizon).
**Verdict:** **GO (single-seed, anti-cheat-confirmed).** The dual-timescale ELIGIBILITY (a slow eligibility trace alongside the fast one; credit = e_fast + e_slow; NO forward-dynamics change) lifts the clean deep recurrent-credit fraction from ~44% (plain e-prop) to **~81%** — squarely in the research-predicted 0.7-0.85 band — and the dualtc-shuffle anti-cheat confirms the lift is genuine credit-structure. Pending: 6-seed + an a_slow robustness sweep + adversarial verification.

## The result — DEEP (ctx17-127) margin vs the add-1 bigram + the clean recurrent-credit fraction
| arm | DEEP margin | deep gain over fixed floor | clean frac_deep (vs BPTT_fixed_win) |
|---|---|---|---|
| fixed_reservoir (floor) | −2.760 | — | — |
| plastic_eprop (plain, R1b) | −2.158 | +0.602 | 0.443 (**44%**) |
| **plastic_eprop_dualtc** (slow eligibility) | **−1.661** | **+1.099** | **0.808 (~81%)** |
| **plastic_eprop_dualtc_shuffle** (magnitude kept, structure broken) | **−2.790** | −0.030 | ~0 (collapses BELOW plain AND fixed) |
| BPTT_fixed_win (recurrent-credit ceiling, frozen W_in) | −1.401 | +1.359 | 1.000 |

- **The lift: 44% → 81%.** Dual-timescale eligibility recovers most of the deep recurrent-credit gap the plain diagonal eligibility left (the plain rule kept 44%; adding a slow-decay trace of the SAME diagonal sensitivity brings it to 81% of the matched full-backprop recurrent-credit ceiling on identical frozen embeddings).
- **DEEP-SPECIFIC (the credit-horizon-extension signature):** dualtc improves ctx17-127 by +0.497 over plain e-prop while SHALLOW is ~unchanged (−3.282 vs −3.281). Per-depth: dualtc pulls ahead only from ctx9-16 (−3.055 vs plain −3.167) and dominates at ctx17-127 (−1.661 vs −2.158) — the slow trace reaches the deep dependencies the ~1/alpha≈3-token fast eligibility cannot.
- **DUALTC-SHUFFLE anti-cheat (decisive):** permuting the combined (e_fast + e_slow) eligibility before each W_rec update — which KEEPS the update MAGNITUDE identical but BREAKS the credit STRUCTURE — collapses the deep margin −1.661 → **−2.790** (below plain e-prop −2.158 AND below the fixed floor −2.760). ⇒ **the 81% is genuine credit-structure, NOT a magnitude/capacity artifact** (a magnitude effect would survive the shuffle; it does not). This rules out the load-bearing confound (the slow trace ~doubles the update magnitude).

## Why this succeeds where the ALIF forward-state FAILED (the honest mechanism story)
- **ALIF (R2, NEGATIVE):** the slow adaptation was a FORWARD STATE (`pre −= β·a`) — the `−β·a` subtractive imprint DEGRADED the tanh representation (credit-vs-capacity control: crediting added ~0; the damage was the forward imprint). Deep went to −3.27 (worse than plain).
- **Dual-timescale eligibility (R2b, GO):** the slow trace lives ONLY in the ELIGIBILITY (the credit horizon), with the forward state UNCHANGED. So it extends how far back the correct diagonal sensitivity `ψ·h_prev` is credited WITHOUT corrupting the representation. Same "slow-timescale" idea, opposite outcome — the lesson: for a rate net the long-range lever is a slow CREDIT trace, not a slow FORWARD state.
- **Biological grounding:** a spread of eligibility-trace time constants is realistic three-factor-learning biology (multiple synaptic-tag / calcium / CaMKII time constants; Gerstner-Lehmann-Liakoni-Corneil-Brea eligibility-trace review; the "e_slow" is a slow synaptic tag). It is the rate analogue of what the on-bridge BDSP/burstprop eligibility could carry with a slow component — i.e. this ports to the spiking substrate as a slow eligibility, no forward-state hack.

## What this means for the frontier
- **Biological one-step-local recurrent credit is NOT deep-credit-limited to a small fraction** — with a slow eligibility it reaches ~81% of full-BPTT's deep recurrent-credit margin on this contiguous long-range task. The pessimistic literature prior (Bellec e-prop needed synthetic-gradient DNI for long-range PTB; Murray RFLO short-horizon-only) was for the PLAIN fast-only eligibility; a dual-timescale eligibility substantially closes it. The residual ~19% is the genuinely off-diagonal cross-unit gradient (SnAp-2 / multi-layer territory), now a much SMALLER gap.
- **Honest scope + pending verification (before a firm generalization claim):** single seed (42), single layer, single a_slow (0.01), 2M tokens. Required next: (1) **6-seed** (42/43/44/100/101/102) — the standing rule; (2) **a_slow robustness sweep** (is ~81% robust across horizons 0.005-0.05, or is 0.01 favorable?); (3) **adversarial verification** (independent skeptics probing for a residual confound). All rate-level torch, GPU, grad-checked, anti-cheated, NO `sim/` edit. Then: the multi-layer rung (temporal dual-timescale e-prop × spatial learned-KP feedback) and the spiking `enable_bdsp` realization with a slow eligibility.

## Files
`_emerge_stream_eprop_lm_derisk.py`; raw `research/findings/raw/_stream_eprop_lm_r2b_dualtc.json` (R2b) + `_stream_eprop_lm_r2c_shufctl.json` (R2c anti-cheat) + `.log`s. Builds on `2026-07-11-R1-...md` (44% clean) + `2026-07-11-R2-rate-ALIF-...md` (forward-state negative).
