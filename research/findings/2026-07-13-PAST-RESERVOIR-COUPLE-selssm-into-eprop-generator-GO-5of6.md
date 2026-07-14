# MISSION-CENTRAL COUPLING (GO, 6-seed): adding the learned selective-SSM context channel to the EMERGENT e-prop generator carries the DEEP context the (e-prop-trained) reservoir alone loses — and pulls its deep CE BELOW the bigram floor

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_couple_selssm_into_eprop_generator_derisk.py` · CI `tests/test_reslm_couple_selssm_into_eprop_generator.py` · raw `research/findings/raw/_couplessm/`. numpy; NO `sim/` edit.
**Status:** ✅ GO 5/6 (the robust sub-claims 6/6). The isolated selective-SSM ladder now COUPLES into the actual emergent conversational cortex.

## The question (the mission-central next step)

The isolated selective-SSM ladder proved the point-by-point mechanism: a learned per-neuron input-dependent SELECTIVE gate beats a FIXED reservoir at deep context on real text (Rung 3), runs+learns on the SPIKING substrate (Rung 4b, byte-equivalent to numpy), and the advantage GROWS with scale. This step brings it into the ACTUAL emergent generator — the **e-prop-trained rate reservoir LM** (`_emerge_reservoir_lm_eprop_recurrent`, the rate analogue of on-bridge BDSP: `W_rec` LEARNS by one-step-local eligibility, NO BPTT/transport). Question: does adding the learned selective channel to the *e-prop-trained* generator's read-out carry the deep context the (now-trained, not merely fixed) reservoir still loses?

## Setup (single variable = the selective channel; the reservoir is IDENTICAL across arms)

1. Build + e-prop-TRAIN the reservoir ONCE per seed (`mode='plastic'` → the emergent generator's learned cortex). FREEZE it.
2. Precompute the frozen reservoir's `h_t` sequences ONCE (shared across arms).
3. Per arm, add (or not) a per-neuron selective-SSM context channel and train a read-out over the arm's feature:
   `c_{t,i}=λ_{t,i}c_{t-1,i}+(1−λ_{t,i})inj_i`, `λ_{t,i}=σ(w_i·E[tok_t]+b_i)`, trained by the Zucchet one-step-local eligibility (survives input-dependent selectivity; NO transport). Forget-bias `b=2.5`.

Arms: **eprop** (read-out over `h_t` only = the generator as-is) · **eprop_sel** (`[h_t,c_t]`, gate trained) · **eprop_sel_rand** (`[h_t,c_t]` but the gate reads a RANDOM token/step = broken current-token selectivity) · **eprop_sel_fix** (`[h_t,c_t]` but `λ` FIXED = a slow LINEAR integrator, ~ALIF). Metric: per-context-depth held-out CE, deep aggregate d≥4 (TinyStories, V=200, n_sent=4000).

## Result — 6-seed (deep-context CE, LOWER = better; gain = eprop − arm)

| seed | eprop | sel | rand | fix | bigram | sel_gain | rand_gain | fix_gain | GO |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 4.030 | 3.331 | 3.586 | 3.690 | 3.384 | +0.699 | +0.444 | +0.340 | GO |
| 43 | 3.958 | 3.378 | 3.638 | 3.702 | 3.498 | +0.579 | +0.320 | +0.255 | GO |
| 44 | 3.892 | 3.223 | 3.479 | 3.585 | 3.358 | +0.669 | +0.412 | +0.307 | GO |
| 100 | 4.023 | 3.332 | 3.650 | 3.714 | 3.444 | +0.692 | +0.373 | +0.309 | GO |
| 101 | 4.028 | 3.611 | 3.610 | 3.725 | 3.452 | +0.417 | +0.418 | +0.303 | no |
| 102 | 3.891 | 3.385 | 3.531 | 3.603 | 3.428 | +0.506 | +0.360 | +0.288 | GO |

- **sel_gain > 0 on 6/6** (mean **+0.594**, min +0.417) — the coupling ALWAYS lowers the emergent generator's deep-context CE.
- **sel > fix on 6/6** (mean sel +0.594 vs fix +0.300, ~2×) — the gate being INPUT-DRIVEN, not a constant leak, is load-bearing; the gain is not merely an extra slow linear memory channel.
- **sel > rand on 5/6** (the strict GO gate; seed 101 a thin tie +0.417≈+0.418).
- **sel deep CE − bigram = −0.051 (mean)** — the coupled channel pulls the generator's deep context BELOW the bigram floor, whereas the reservoir-only generator is **+0.543 ABOVE** it. The coupling doesn't just beat the reservoir baseline; it clears the n-gram floor at V=200.

## ⇒ interpretation + the honest scale nuance

The emergent generator's deep-context is carried BETTER by the learned selective gate than by the (even e-prop-trained) fading reservoir — 6/6 for the coupling helping, 6/6 for the input-driven selectivity beating a fixed integrator, 5/6 for the strict current-token-selectivity anti-cheat, and it clears the bigram floor. This is the rate-level realization of the mission-central path: **fluent long-range context in the conversational cortex, carried by a learned, transport-free selective gate that is byte-equivalent on the spiking substrate (Rung 4b-iii-a).**

**Honest scale-dependence of the `sel>rand` anti-cheat (recorded, consistent with the Rung-1 Sub-claim A/B settling).** The random-token-gate control is NOT null — a random-token-driven gate still adds generic input-varying nonlinear capacity + some memory (it just doesn't condition on the CURRENT token). So:
- **The ROBUST claims** (sel > eprop; sel > fix) — a nonlinear input-DRIVEN context channel beats the reservoir-only generator AND beats a fixed slow integrator — hold across seeds and stabilize by ~n_sent 2500/V 160 (the Sub-claim-A analogue: any input-driven nonlinear gate beats the fixed/linear alternative).
- **The HARDER claim** (sel > rand: the SPECIFIC current-token selectivity beats a GENERIC input-varying gate) is scale-dependent — it fails at intermediate scale (2000–2500) where the deep signal is thin, and holds 5/6 only at the full 4000/200 scale (the Sub-claim-B analogue: the specific structure beats generic nonlinear capacity only with enough data/vocab). This is the same lesson as the ridge-settled Rung-1 Sub-claim B, and it predicts the current-token advantage keeps widening with scale (the fluency direction).

## Scope / next
- numpy rate de-risk (the emergent generator is a rate reservoir; the selective channel is byte-equivalent on the spiking substrate, Rung 4b-iii-a, so it transfers). The frozen-reservoir coupling isolates the selective add-on cleanly.
- CI guard asserts the robust invariants (sel>eprop, sel>fix) at 3000/180; the scale-dependent sel>rand is documented here.
- NEXT: (a) scale (bigger V/data — does the current-token sel>rand advantage widen, and the sel−bigram margin grow, per the Rung-3 trajectory?); (b) the on-bridge realization — thread the coupled selective channel into the spiking generator's read-out end-to-end (the channel + its eligibility are already on-substrate-validated); (c) jointly e-prop-train the reservoir WITH the selective channel present (vs the frozen-reservoir coupling here).

## Files
- `research/runners/_reslm_couple_selssm_into_eprop_generator_derisk.py`, `tests/test_reslm_couple_selssm_into_eprop_generator.py`, raw `research/findings/raw/_couplessm/seed*.json`. Follows the PAST-RESERVOIR Rung 1–4b + scale-trajectory arc.
