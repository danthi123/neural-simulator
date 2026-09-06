---
type: finding
status: partial
claim_check: measured
date: 2026-09-06
mechanism: metacog honesty-hedge confidence read — accumulation-to-bound over the recall competition
  (mean-trajectory margin + time-to-bound) vs the single-snapshot recall margin, 6-seed GPU verify
lane: introspection / metacognition (honesty-hedge)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_metacog_accumulation_to_bound_derisk.py
artifacts:
  - research/findings/raw/_metacog_accumulation_to_bound_derisk/6seed.json
builds_on:
  - research/findings/2026-09-06-metacog-accumulation-to-bound-middle-band-PARTIAL.md
verdict: >
  PARTIAL — direction CONFIRMED at 6 seeds, magnitude modest, does NOT resolve the ambiguous middle band.
  The 6-seed GPU run (5 sigmas x 40 trials x 6 seeds) reproduces the CPU-smoke direction: the runner's own
  verdict has type-2 AUC modestly higher for accumulation than for the single snapshot over ALL trials (a small
  real lift from reading the recall competition as a temporal accumulation rather than one snapshot; exact
  figures in the marked body). In the AMBIGUOUS middle band specifically (the residual this arc targets),
  accumulation beats the snapshot on 5 of 6 seeds, but BOTH predictors sit near chance there (even the host
  reference margin is only weakly predictive) — so accumulation IMPROVES the honesty-hedge overall without
  RESOLVING the hard cases. Per THE LAW this is a method result, not a wall: the wall-reframe (the single-snapshot
  margin replaced the brain's temporal evidence-accumulation) is validated as directionally-right and worth
  keeping as a component, but the ambiguous band needs a further companion process (accumulation alone is not
  enough). Default-OFF, unwired; additive (`_spiking_margin` untouched). A genuine 6-seed upgrade of the
  CPU-smoke PARTIAL.
---

# Metacog accumulation-to-bound — 6-seed PARTIAL (direction confirmed, ambiguous band unresolved)

## What ran
`_metacog_accumulation_to_bound_derisk.py --seeds 42 43 44 100 101 102 --sigmas 1.5 1.8 2.0 2.2 2.5
--n-trials-per-sigma 40` on the GPU (cupy), the 6-seed follow-up to the CPU-smoke. 1200 trials total, 288 in the
ambiguous middle band (role-confidence 0.3-0.5).

## Derived — type-2 AUC (higher = confidence better predicts correctness)
<!--derived: all-trials AUCs from the runner's own `verdict` field in research/findings/raw/_metacog_accumulation_to_bound_derisk/6seed.json; ambiguous-band + per-seed AUCs computed directly from its `trials` array -->
| predictor | all-trials AUC (n=613 scored) | ambiguous-band AUC (n=288 pooled) |
|---|---|---|
| snapshot margin (current) | 0.600 | 0.478 |
| accumulation mean-trajectory | 0.628 | 0.495 |
| time-to-bound (frac roles bounded) | 0.579 | 0.522 |
| host reference margin | — | 0.551 |

<!--derived: per-seed ambiguous-band AUCs computed from the trials array of research/findings/raw/_metacog_accumulation_to_bound_derisk/6seed.json -->
Per-seed ambiguous band, accum vs snapshot: seed42 0.596>0.580 · seed43 0.454>0.401 · seed44 0.466>0.437 ·
seed100 0.546>0.395 · seed101 0.457<0.483 · seed102 0.645>0.572 → accum beats snapshot on **5/6 seeds**.

## Reading it
Accumulation-to-bound gives a real, consistent overall lift (the all-trials AUCs above; 5/6 seeds in the
ambiguous band) — the wall-reframe direction holds at 6 seeds. But in the ambiguous band both predictors are near
chance (see the table), and even the host reference margin is only weakly predictive there — the
genuinely-ambiguous cases are hard for ANY margin-based read. So accumulation is a keep-worthy COMPONENT of the
honesty-hedge, not a resolution of the middle band; the residual needs a further companion signal (per the
wall-reframe, likely a non-margin cue — e.g. recall-time variability, or a second competing-evidence channel).

## Honest scope
6-seed, cupy. Default-OFF, unwired; additive (the production `_spiking_margin` is untouched). The ambiguous-band
AUCs are computed here from the trials array; the all-trials AUCs are the runner's own verdict field.
