---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
---

# Open-ended generation on the PRODUCTION turn: the draw lesion changes the reply distribution (seed 42), and BRAIN_OPEN_ENDED mode bypassed the draw entirely (2026-09-23)

seed-waiver: seed 42 was run in-session. Seeds 43/44/100/101/102 are staged (see "Staged"), so this is a labelled
single-seed probe until the 6-seed aggregates land.

Lane `research/open-ended-production-turn-lb` (charter D1). Pre-registration:
[`docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md`](../../docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md)
(committed `4edf6fc10`; amendment 1 `94b71d511`, both before the runs they govern). Instrument:
`research/runners/_lbf_open_ended_production_turn_probe.py`.

## What was missing

Earlier rulers both missed the production turn. The single-turn field-diff (finding
2026-09-21-open-ended-generation-single-turn-not-load-bearing-spiking-plausibility-gate-masks-draw) reads one
draw. The distributional ruler (`LB_OPEN_ENDED_DISTRIB_PROBE`) samples a synthetic taxonomy world and never runs
`webapp.server.brain_chat`. This probe drives the real `brain_chat` and teaches a graded chase KB through 13 chat
turns. It then asks "what might a dog chase" 40 times in the same session and records each reply. The arms are
intact, intact_rebuild and lesion (`BRAIN_SPIKING_DRAW_LESION=1`), each a fresh process at `BRAIN_CHAT_SEED=42`.
The statistic is the TV distance between the reply histograms, tested against a label-permutation null (B=10000).

## Result, seed 42 (numpy backend, K=40)

| mode | verdict | intact replies | lesion replies | TV | perm p | null TV q95 | D | draws intact / lesion |
|---|---|---|---|---|---|---|---|---|
| `default` (production default turn) | LOAD-BEARING | deer 39, rabbit 1 | rabbit 39, deer 1 | 0.95 | 1e-4 (floor) | 0.20 | +0.95 | 41 / 432 |
| `oe_unfixed` (BRAIN_OPEN_ENDED=1) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | 0 | 1.0 | 0 | - | **0 / 0** |
| `oe_routed` (+ route fix) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | 0 | 1.0 | 0 | - | 16000 / 16000 |
| `oe_unfixed_taught` (amendment 1) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | 0 | 1.0 | 0 | - | **0 / 0** |
| `oe_routed_taught` (amendment 1) | LOAD-BEARING | deer 39, rabbit 1 | rabbit 39, deer 1 | 0.95 | 1e-4 (floor) | 0.20 | +0.95 | 41 / 432 |

Artifacts: `research/findings/raw/_load_bearing/_oe_production_turn/<mode>/<mode>_s42_verdict.json`, for example
research/findings/raw/_load_bearing/_oe_production_turn/default/default_s42_verdict.json,
research/findings/raw/_load_bearing/_oe_production_turn/oe_routed_taught/oe_routed_taught_s42_verdict.json,
research/findings/raw/_load_bearing/_oe_production_turn/oe_unfixed_taught/oe_unfixed_taught_s42_verdict.json and
research/findings/raw/_load_bearing/_oe_production_turn/oe_routed/oe_routed_s42_verdict.json. Each has three
per-arm worker JSONs beside it holding every reply, the stored facts, the draw counter and the likelihood weights.
In every mode the intact and intact_rebuild reply sequences are identical, an exact list compare.

Other details:
- The lesion held at measurement: 432 of 432 lesion-arm draws went through the ablated sampler, and 0 intact draws
  did.
- `attributable_fraction` (TV vs the null-median TV of 0.05) = 0.9473684210526315.
- The brain's own likelihood weights for (dog, chase, ·): deer 3, rabbit 2, beetle 1, minnow 1. Cat (2) is a
  stored fact and is excluded as not novel.
- The intact draw volunteers the likelihood peak (deer). The lesioned draw, with uniform drive, volunteers rabbit
  instead. That choice comes from the bank's own neuron heterogeneity, not from the association graph.
- D = mean likelihood weight volunteered, intact minus lesion = 2.975 − 2.025.

## The defect found: the BRAIN_OPEN_ENDED reply never reached the generative draw

With `BRAIN_OPEN_ENDED=1` every turn is answered by `webapp/open_ended_chat.answer_turn`, and that function never
calls `chat.gate`. `extract_topic("what might a dog chase")` returns the whole prompt, retrieval finds no facts,
and the reply is the fixed "I'm not sure about ..." abstain. The data confirm this: the draw count is **0 in both
arms**, and that holds even when the brain knows 13 facts (`oe_unfixed_taught`). In the mode the D4 conversation
battery targets, the open-ended reply was lesion-invariant to the spiking draw by construction.

**Fix (default-OFF): `BRAIN_OPEN_ENDED_GENERATE_ROUTE`**, in `webapp/server.py::_open_ended_generate_route`. An
explicit generation prompt skips the free-talk block and falls through to the ordinary pipeline's GENERATE channel.
The prompt test is the ChatBrain's own `_parse_open_ended`, the same conservative pattern set gate() uses. With the
fix, `oe_routed_taught` reaches the draw (41 draws). Its replies are identical, as an exact list compare, to the
`default` mode's replies, and it reads LOAD-BEARING on seed 42.

Flag-off behaviour: the function returns before touching `chat`. With BRAIN_OPEN_ENDED off it is never called,
because the AND short-circuits. `tests/test_open_ended_generate_route.py` pins both cases. Byte-identity of the
flag-off reply is NOT asserted in data; it was inferred from code, so it counts as unverified under docs/TERMS.md.

## A second bypass, not fixed here: open-ended mode does not learn from assertions

With BRAIN_OPEN_ENDED=1 the 13 teach assertions also went to the free-talk path. The replies were of the form
"I'm not sure about wolf chase the rabbit…", in-loop acquisition never ran, and `stored_facts` stayed at the 5
build-time facts. As a result, `oe_routed` exercises the draw (16000 draws) but has nothing novel to volunteer, so
it is UNDEFINED. Amendment 1 isolates the ASK path: the `*_taught` modes teach through the ordinary path. The
teach-path bypass is a separate open defect: with BRAIN_OPEN_ENDED=1 the brain does not learn from being told.
Next method: route `_maybe_acquire`-eligible SVO assertions the same way, default-OFF, and measure it with this
probe's `oe_routed` mode.

## What is NOT claimed

- No 6-seed claim. Seed 42 only; the gate for "load-bearing on the production turn" is 6/6 per mode.
- Nothing about a single turn. This is a distributional claim over 40 asks.
- The lesioned distribution is not "uniform". It is set by the bank's intrinsic heterogeneity. A seed whose
  heterogeneity favourite coincides with the likelihood peak would read a small TV; the 6-seed run tests this
  honestly.
- No production default was flipped.

## Declared host shortcuts

- The KB, the prompt and the permutation statistic are world and instrument.
- In the oe modes the warm Qwen faculty is a stub, with BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK=1. FORM is not measured
  here.
- The draw is a `cp_firing_states` read on an Izhikevich WTA bank.
- The plausibility gate is the production default spiking associative read.

## Staged (6-seed, detached, memcapped, local numpy)

`bash research/runners/_lbf_open_ended_production_turn_stage.sh <mode> <parallel>` was launched for these modes:
- `default` (parallel 2)
- `oe_routed_taught` (parallel 2)
- `oe_unfixed_taught` (parallel 1)

Each run covers seeds 43, 44, 100, 101 and 102. Per-mode aggregates land at
`research/findings/raw/_load_bearing/_oe_production_turn/<mode>_6seed_aggregate.json`, and logs in `logs/`. The
pool was not usable: `pool_queue.sh` validates modules against `~/derisk-pool/sim`, which lacks this runner, and
pool42 provisioning timed out.
