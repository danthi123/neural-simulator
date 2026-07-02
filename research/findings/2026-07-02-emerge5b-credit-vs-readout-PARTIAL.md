# EMERGE-5b — credit-vs-readout diagnostic: the spiking accuracy gap is BOTH (partly), and the credit signal is the higher-leverage limit → microcircuit arm warranted

**2026-07-02 (autonomous; substrate ladder rung 2 follow-up).** Runner `research/runners/_emerge5b_credit_vs_readout_derisk.py`; result `research/findings/raw/_emerge5b_credit_vs_readout.json`. Reuse-by-import (EMERGE-1b + EMERGE-5 machinery); NO `sim/` edit; CPU.

## Why this ran
EMERGE-5 (rate→spike Burstprop) at the realistic p0=0.03 found the hidden representation partially emerges under spike noise (XOR-latent probe ~0.70) but task accuracy sits at chance (~0.50 vs rate ceiling ~0.79). Two candidate diagnoses point to different next levers: **(A) readout bottleneck** (good rep, noisy spike-count output → cheap fix, credit is fine) vs **(B) credit bottleneck** (impaired rep from noisy credit → need a better/more-noise-robust credit rule = the microcircuit). Two prior levers were eliminated first: **width** (the oracle won't scale to width 768 regardless of lr — a confound, not a result), and **naive population-averaging** (mathematically identical to raising the sample budget S — pooling M independent Poisson/Binomial copies = `Poisson(M·e·S)` then `Binomial` — which EMERGE-5's S-sweep already tested and which failed to recover accuracy at p0=0.03).

## The test
Train the spiking Burstprop net at the **healthy width-384 config** (oracle ~1.0, so width-scaling is not a confound), FREEZE its hidden weights, then train ONLY a fresh softmax readout on its (analytic, noise-free) hidden activations with a clean full-batch gradient. Compare against the rate ceiling, a clean-readout-on-rate upper bound, and a clean-readout-on-untrained random-features floor.

## Result (3-seed means; verdict = PARTIAL)
| measure | mean | per-seed (42/43/44) |
|---|---|---|
| own_spiking (end-to-end) | 0.505 | 0.460 / 0.532 / 0.524 |
| **clean_readout_on_spiking** | **0.622** | 0.585 / 0.593 / 0.688 |
| clean_readout_on_untrained (floor) | 0.488 | 0.451 / 0.538 / 0.476 |
| rate_ceiling | 0.796 | 0.766 / 0.858 / 0.763 |
| clean_readout_on_rate (upper bound / method sanity) | 0.834 | 0.905 / 0.891 / 0.705 |

- **Method is sound:** clean-readout-on-rate (0.834) ≥ the rate ceiling, and clean-readout-on-untrained (0.488) ≈ chance — the probe recovers structure where it exists and reports floor where it doesn't.
- **A clean readout DOES help** (0.622 > own 0.505 by +0.12) — the noisy spike-count readout was *part* of the accuracy gap.
- **But the spiking rep is genuinely impaired** (clean-on-spiking 0.622 ≪ clean-on-rate 0.834, and ≪ ceiling 0.796) — the noisy burst credit built a representation materially worse than the rate model's, and no readout can fully rescue it.
- **The credit signal built real structure** (0.622 > 0.488 random floor by +0.13) — it is not nothing; it is degraded.

## Verdict + decision
**Both contribute, and credit quality is the higher-leverage limit.** This rules out "just fix the readout" as a complete solution. The decision (per the standing ladder + the shortlist's Urbanczik-Senn note that the real population mechanism is a population-*feedback factor*, not naive averaging): pursue a **more noise-robust credit rule** — the **Sacramento–Senn dendritic microcircuit's active cancellation** (EMERGE-3 was the numpy microcircuit; the arm is to add the same finite-sample spike-count noise model EMERGE-5 uses and test whether active interneuron cancellation builds a *cleaner* representation under noise than raw burst-rate estimation). A cleaner/averaged output readout remains a cheap *partial* complement, not the primary fix.

## State of rung 2 (honest)
- Rate→spike burst credit: representation partially survives (this doc + EMERGE-5); task accuracy limited by degraded credit under finite-sample noise.
- Levers tested + eliminated: width (oracle-scaling confound), variance-reduction via S / naive population-averaging (already-shown-insufficient, and mathematically the same thing).
- Next lever: the microcircuit active-cancellation arm under spike noise. Genuinely different mechanism (not variance-reduction); the honest expectation is it *may* be more noise-robust (active cancellation vs raw estimation) or may hit the same finite-sample wall — either is build-informative.

## Artifacts
`research/runners/_emerge5b_credit_vs_readout_derisk.py`, `research/findings/raw/_emerge5b_credit_vs_readout.json`. Prior: `2026-07-01-emerge5-noise-driven-self-organization-discovery.md` (the noise-variance-self-organization discovery + p0 dose-response).
