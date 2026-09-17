---
type: finding
status: verified
date: 2026-09-17
mechanism: the own-voice spiking mouth's broad-domain fluency bend-test scaled to n2M (2M sentences) at fixed model
  size — hold the eval corpus fixed, scale the training-token supply, and read whether the deep-context language skill
  (WKV-deep NLL) keeps descending (more-data-is-the-lever) or plateaus (capacity-saturated). Run across two corpora
  (WikiText-103, FineWeb-Edu) and two model sizes (d_model 96, 192), 3 seeds each on the GPU keystone
integration_faculty: language (own-voice mouth — broad-domain fluency)
lane: mouth / own-voice fluency (the keystone, ~49 blocked retirements)
seeds: [100, 101, 102]
verdict: NO-GO on the token-supply lever at these SMALL model sizes — CAPACITY-SATURATED. At d_model 96 and 192, the
  broad-domain deep-context NLL plateaus ABOVE the fluency band and scaling training tokens to n2M gives diminishing
  returns: FineWeb-d192 and WikiText-d96 are capacity-saturated (adding tokens does not improve NLL, not still
  descending at the top point), and FineWeb-d96 still gains modestly from tokens (PARTIAL) but never reaches the band.
  The models are HEAVILY OVER-TOKENED relative to compute-optimal scaling (80-174 tokens per active parameter vs the
  ~20 Chinchilla-optimal), so at this scale the BINDING lever is MODEL CAPACITY (parameters), NOT more tokens. This
  REFINES the prior "the wall is token/data-bound" framing: the path is compute-optimal (scale tokens AND parameters
  together), and having now over-scaled tokens at small sizes, the parameter side is what binds. Next lever: scale
  d_model up with Chinchilla-matched (~20 tok/param) token budgets, on the GPU keystone. A wall defers a METHOD
  (pure token-scaling at fixed small size), never the capability (broad-domain own-voice fluency).
runner: research/runners/_gen_cortex_token_supply_scaling_derisk.py
artifacts:
  - research/findings/raw/_gencortex_scaling/fineweb_d192_n2M_s100.json
  - research/findings/raw/_gencortex_scaling/fineweb_d96_n2M_s101.json
  - research/findings/raw/_gencortex_scaling/wt103_d96_n2M_s100.json
external: EXTERNAL-DONE — the compute-optimal token/parameter scaling law (Hoffmann et al. 2022, "Training
  Compute-Optimal Large Language Models" / Chinchilla, ~20 tokens/parameter) is the direct framing; this bend-test
  measures where the mouth sits against it (heavily over-tokened at small sizes). Identifier in the body.
builds_on:
  - research/findings/raw/_gencortex_scaling/fineweb_d192_n2M_s100.json
---

# Mouth token-supply bend-test at n2M — CAPACITY-SATURATED at small sizes; the lever is parameters, not tokens

The own-voice spiking mouth is the keystone (~49 blocked retirements wait on it). The open question for broad-domain
fluency: is more READING (training tokens) the lever, or has it saturated? This bend-test holds the eval corpus fixed
and scales the training-token supply to n2M (2M sentences) at fixed model size, across two corpora and two model sizes.

## Result (from research/findings/raw/_gencortex_scaling/*_n2M_*.json)
<!--derived-->
Per config (3 seeds each; fluency band = deep NLL 3.0-3.69; lower is better):
- FineWeb-Edu d_model=192: CAPACITY-SATURATED. delta-NLL over the token range ~0.07 nats (tiny), NOT still descending
  at the top, top deep-NLL ~3.91 (residual ~+0.22 above the band), ~80 tokens/active-param.
- FineWeb-Edu d_model=96: PARTIAL. delta-NLL ~0.19 nats (still gains from tokens — uses_tokens), but NOT still
  descending at the top, top deep-NLL ~3.95 (residual ~+0.26), ~174 tokens/active-param.
- WikiText-103 d_model=96: CAPACITY-SATURATED. delta-NLL ~0.076 nats, top deep-NLL ~3.75 (residual ~+0.06 — the
  CLOSEST to the band), ~87 tokens/active-param.
- Common to all: beats the trigram at the top point, but NONE reach the fluency band, and margin does not grow with
  tokens. Compute-optimal (Chinchilla, Hoffmann et al. 2022, arXiv:2203.15556) is ~20 tokens/param; every config here
  is 4-9x over-tokened.

## What it means, and the next lever

At these small model sizes the mouth is heavily over-tokened, so scaling training tokens further gives diminishing
returns (d192 and WikiText-d96 flat; FineWeb-d96 only modestly gaining) and the broad-domain deep-context NLL plateaus
~3.75-3.95, above the fluency band. So the binding lever at this scale is MODEL CAPACITY (more active parameters), not
more tokens — the parameter side of the compute-optimal law. This refines, not contradicts, the prior token/data-bound
reading: the path is to scale tokens AND parameters together, and having over-scaled tokens at small sizes, parameters
now bind. The next GPU-keystone lever is a capacity sweep — scale d_model up (e.g. 384+) with Chinchilla-matched (~20
tok/param) token budgets — to test whether more capacity, properly tokened, descends into the fluency band. WikiText-d96
sitting only +0.06 above the band is the encouraging anchor. A wall defers a METHOD (fixed-small-size token scaling),
never the capability.
