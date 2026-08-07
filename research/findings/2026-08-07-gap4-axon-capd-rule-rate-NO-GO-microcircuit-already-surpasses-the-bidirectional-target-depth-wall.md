---
type: finding
status: no-go
date: 2026-08-07
mechanism: gap4-axon-capd-rule-vs-microcircuit-rate
backend: numpy
runner: research/runners/_gap4_axon_capd_rule_derisk.py
artifacts:
  - research/findings/raw/_gap4_axon_capd_rule.json
---

# gap#4 adoption D1: Astera Axon's CaP−CaD rule does NOT beat our microcircuit at rate — because our SST-microcircuit ALREADY surpasses the feedback-alignment depth wall that Axon's bidirectional target hits

## Verdict
<!--derived-->
All numbers in this doc are quoted from the runner's own `verdict`/`per_seed` fields in
`research/findings/raw/_gap4_axon_capd_rule.json` (which now carries a `preconditions` block: window-informative
oracle 0.966 ≫ reservoir 0.504, no-weight-transport, 6-seed).
**NO-GO (6-seed, rate/numpy).** Owner-directed adoption test of the landscape-survey "goldmine" (Axon = the closest
external work to our deep-credit-on-spikes wall). Axon CaP−CaD held-out **0.476** did NOT match our microcircuit
**0.942**, is below the 0.75 bar, and did not clear the reservoir floor 0.504 (window IS informative: oracle 0.966 ≫
reservoir 0.504). **Do NOT port to spikes; the rate rule must match the microcircuit first, and it doesn't.**

## Which factor (precise diagnosis — the valuable part)
<!--derived-->
1. **PRIMARY limiter — the 2-phase bidirectional (GeneRec/CHL) target's credit DECAYS THROUGH DEPTH.** CHL
   cos-to-true-grad per layer [deep-hidden..output] = [0.272, 0.501, 0.458]: the output is aligned but the deepest
   hidden is ~0. Even *learned* bidirectional return weights don't rescue it (held-out 0.717). This is exactly the
   **feedback-alignment depth wall** — and it is the wall OUR Sacramento-Senn / SST-microcircuit ALREADY SURPASSES
   (held-out 0.961) via INTERNEURON ERROR-CANCELLATION (the interneuron cancels the predictable feedback so the apical
   carries ERROR, not raw teaching). So Axon's target-delivery is WEAKER at depth than what we already have.
2. **SECONDARY limiter — the CaP−CaD calcium temporal-derivative read** degrades even the output credit CHL gets right
   (CaP−CaD cos per layer = [−0.122, 0.015, 0.644]): the end-of-plus-phase temporal-derivative does not faithfully
   recover the two-phase difference at rate. (Trace factor slightly hurts here: no_tr 0.513 vs full 0.476.)

## Consequence for the adoption plan (honest update)
- **The Axon LEARNING RULE is NOT the fix for gap#4** — our microcircuit is the better rate rule, and its interneuron
  error-cancellation is the load-bearing ingredient Axon's bidirectional-reverberation target lacks (the full
  CT-predict/pulvinar version is ALSO a bidirectional-reverberation mechanism, so it likely hits the same depth wall).
  This VALIDATES our existing microcircuit choice and rules out a wholesale Axon-rule swap.
- **The gap#4 spiking-port wall** ("the local rule doesn't enter the learning regime on real spikes") is therefore
  NOT solved by Axon — it needs our own scoped **RESIDUAL-A** (the spiking BDSP port: a `fused_bdsp` kernel + a
  fixed-random apical RegionPathway porting the microcircuit to spikes), per `2026-07-07-deep-lever-research-gate`.
- **Still open from the survey (NOT ruled out by this de-risk):** Rubicon's DELAYED-CREDIT machinery (maintained-goal
  temporal bridge + VSPatch reward-timing) for the vocal BG credit / crux family — a different payload, untested here.
  And the Axon calcium-derivative *read* could in principle combine with OUR interneuron-cancellation target — but
  that's a refinement, not the depth-wall fix, which we already have.

Discipline note: this is the RAG-grounded, anti-cheat-gated de-risk the owner asked for — it converts the "goldmine"
claim into a precise, honest result (our microcircuit is ahead of Axon on the one axis that matters for depth) rather
than an adopted overclaim.
