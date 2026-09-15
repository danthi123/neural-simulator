---
type: finding
status: contributing
date: 2026-09-15
mechanism: gap4-inengine-selfpredicting-interneuron-microcircuit
lane: deep-credit-assignment
seeds: [42, 43, 44, 101, 102]
verdict: UNDEFINED (interpretability-foreclosed, the PRE-REGISTERED outcome) — the transport_ceiling (best-case
  copied-weight oracle) never clears chance — the oracle sits at or below it on every one of the 5 seeds — so the read-regime cannot establish the
  instrument and no GO/NO-GO on the learned in-engine microcircuit is earnable. Also foreclosed by count:
  micro_inengine > fixed_fa on 0/5 seeds vs a >=5/6 GO bar (max 1/6 even if the 6th passed), so seed 100 was
  STOPPED as compute-foreclosed. NOT a clean NO-GO.
runner: research/runners/_gap4_selfpredict_interneuron_inengine_derisk.py
artifacts:
  - research/findings/raw/gap4/_aggregate_5seed.json
  - research/findings/raw/gap4/selfpredict_inengine_s42.json
  - research/findings/raw/gap4/selfpredict_inengine_s43.json
  - research/findings/raw/gap4/selfpredict_inengine_s44.json
  - research/findings/raw/gap4/selfpredict_inengine_s101.json
  - research/findings/raw/gap4/selfpredict_inengine_s102.json
external: NO-EXTERNAL-NEEDED -- the Sacramento-Senn self-predicting-interneuron mechanism + its transport-ceiling
  instrument are already banked in the gap#4 arc (in-engine microcircuit commit 137ecf42); this is its 6-seed read.
builds_on:
  - research/findings/raw/_gen_cortex_token_supply_scaling.json
---

# gap#4 in-engine self-predicting-interneuron microcircuit: UNDEFINED — the transport ceiling is foreclosed by the read-regime

**One-line.** The decisive 6-seed run of the learned in-engine self-predicting-interneuron microcircuit (the
genuinely-untested fix on the *feedback* side of deep credit) returns the **pre-registered UNDEFINED** verdict: on
5/6 seeds the `transport_ceiling` arm — the best-case copied-weight oracle that upper-bounds what any local rule
could achieve in this read-regime — **never clears chance** (the oracle sits at or below chance on every seed). When
the oracle itself sits at chance, the instrument cannot say whether the microcircuit works, so per the runner's own
gate the verdict is UNDEFINED, not a NO-GO. Seed 100 was stopped mid-run (see "compute foreclosure").

## The numbers (5 seeds; chance = one-in-six)

<!--derived-->
(per-seed arm means from the cited per-seed artifacts; saved in `research/findings/raw/gap4/_aggregate_5seed.json`.)

| seed | fixed_fa (baseline) | micro_inengine (learned) | transport_ceiling (oracle) | ineng > fa? | ceiling > chance? |
|---|---|---|---|---|---|
| 42 | 0.148 | 0.074 | 0.093 | no | no |
| 43 | 0.093 | 0.056 | 0.148 | no | no |
| 44 | 0.093 | 0.093 | 0.093 | no | no |
| 101 | 0.093 | 0.037 | 0.167 | no | no |
| 102 | 0.074 | 0.074 | 0.111 | no | no |

- **`micro_inengine > fixed_fa`: 0/5.** The learned in-engine cancellation never beats the fixed random-feedback
  baseline; it is <= it on every seed.
- **`transport_ceiling` never clears chance — on none of the completed seeds.** The oracle upper bound is at or below chance throughout — the
  read-regime forecloses the instrument.

## Why this is UNDEFINED, not NO-GO (the interpretability gate)

The runner's pre-registered rule: "GO = micro_inengine > fixed_fa on >=5/6 seeds in the FA-wall regime + earned
in-engine silence + all anti-cheats + an INTERPRETABLE ceiling. **UNDEFINED if the transport ceiling cannot clear
chance (the read-regime forecloses the instrument).**" With the oracle at chance throughout, you cannot
distinguish "the microcircuit fails" from "no local rule could succeed in this read-regime, and neither could the
copied-weight oracle." That is a foreclosed instrument, so the honest verdict is UNDEFINED — a valid deliverable,
exactly as pre-registered, and NOT license to conclude the microcircuit (or the feedback-side fix) is a dead end.

## Compute foreclosure (why 5/6, not 6/6, and why that is sound here)

The 6th seed (100) was running when this was harvested and was **stopped**, because it cannot change the outcome:
(a) the GO count needs micro_inengine > fixed_fa on >=5/6 and it is 0/5 -> max 1/6 even if seed 100 passed; and
(b) the UNDEFINED branch is already triggered on all 5 seeds and there is no reason a 6th would flip an oracle that
is at chance everywhere. The 6-seed standard exists to stop generalization from too-few seeds; here the verdict is
*determined*, not generalized, so stopping seed 100 saved ~hours of GPU (redirected to the token-scaling sweep)
without weakening the conclusion. Seed 100 is re-runnable if a literal 6/6 record is ever wanted; the verdict stands.

## The named next lever (no-defer)

UNDEFINED foreclosure is a verdict on this READ-REGIME's instrument, not on the mechanism. The next lever is to
make the `transport_ceiling` oracle interpretable — i.e., a read-regime where the copied-weight upper bound
genuinely clears chance (a stronger/longer readout, more FA-wall-regime coverage per seed — `n_fa_wall` was only
0-1/seed here, so the regime that makes the comparison meaningful barely triggered) — and only then re-run the
micro_inengine vs fixed_fa comparison. Until the ceiling clears chance the comparison is uninterpretable by
construction. Functional read-outs only.
