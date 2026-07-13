# On-bridge NP de-risk: the sim's STDP-TIMING eligibility is the WRONG eligibility for node perturbation (fragile, 2/6 seeds, inconsistent sign) — NP needs a RATE-COINCIDENCE eligibility (the emerge1-validated form). Scopes the on-bridge realization.

**Date:** 2026-07-13
**Runner:** `research/runners/_np_onbridge_eligibility_probe.py` (minimal 2-region pre→post bridge, `enable_stdp` + `enable_reward_modulation`; NO permanent `sim/` edit). Cheap-first de-risk of the riskiest assumption of the on-bridge NP realization.

## The question
The on-bridge NP realization plan: perturb the hidden/post region with intrinsic-noise current ξ → the sim's STDP builds a ξ-correlated eligibility on the pre→post pathway → a global reward = −dL scales it (the committed `enable_reward_modulation` three-factor update) → the sim's OWN plasticity does the NP step, no host weight-write. The riskiest assumption: does `cp_eligibility_trace` actually correlate with the injected perturbation ξ?

## Result — the STDP-timing eligibility route is UNRELIABLE
Drive pre; run a settle with +ξ vs −ξ current on post; read the pre→post eligibility for each; `corr(elig(+ξ)−elig(−ξ), per-post ξ)`, 6 seeds (post_bias 400, fwd 20, post firing ~0.04–0.08):
- seed 42: corr **−0.514** (elig populates) · seed 100: corr **+0.258** (opposite sign!) · **seeds 43/44/101/102: corr 0.000, eligibility FLAT (d_std=0, no accumulation at all).**
- So the eligibility populates on only **2/6 seeds** and with **inconsistent sign**. The STDP-timing eligibility does NOT robustly encode the perturbation.

## Why (the mechanistic diagnosis) + the next mechanism
The sim's eligibility is `cp_eligibility_trace += STDP weight_change` — a **spike-TIMING** quantity (LTP if pre-before-post, LTD if post-before-pre). At the sparse firing needed to keep the perturbation graded (post ~0.05), pre→post coincidences WITHIN the STDP window are rare + highly seed-sensitive → the eligibility is fragile (flat on most seeds). And where it does fire, the ξ current drives post DIRECTLY (bypassing the synapse), so post fires BEFORE the pre input → LTD → a negative, timing-artifact sign.
**Node perturbation's natural eligibility is a RATE COINCIDENCE, not spike-timing:** the credit for synapse (pre_i→post_j) is `ξ_j × (pre_i activity)` — how much post_j was perturbed × how active pre_i was. This is exactly what the emerge1 numpy NP used (node credit `× outer(ξ, pre-rate)`), and it is robust (no timing coincidence required). ⇒ **the on-bridge NP realization must use a rate-coincidence eligibility, NOT the sim's STDP-timing eligibility.** Two routes: (a) read the post rate-difference under ±ξ × the pre rate directly (a rate three-factor); (b) a `sim/` eligibility variant that accumulates pre-rate×post-rate coincidence (a rate eligibility) rather than STDP timing. This is the scoped next step of the on-bridge arc.

## Scope
This is the honest first de-risk of the on-bridge NP realization: the plan's assumed eligibility (STDP timing) is the wrong one; the right one (rate coincidence) is identified + emerge1-validated. The on-bridge realization is a focused next arc (build the rate-coincidence eligibility loop). NO permanent `sim/` edit. Ties to `2026-07-13-fresh-deep-credit-class-NODE-PERTURBATION-*` (NP's proven feedforward deep credit — the thing being ported on-bridge).
