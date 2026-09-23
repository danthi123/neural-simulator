---
type: plan
status: live
lane: load-bearing
date: 2026-09-23
---

# Pre-registration: is the spiking generative draw load-bearing on the PRODUCTION open-ended reply? (2026-09-23)

Lane `research/open-ended-production-turn-lb`, charter D1 (docs/plans/2026-09-23-autonomous-charter.md).
Committed BEFORE any run it governs. Instrument: `research/runners/_lbf_open_ended_production_turn_probe.py`.

## The question

Earlier rulers each miss the production turn. The single-turn field-diff reads one draw, and one draw is
dominated by OU noise. The distributional ruler (`LB_OPEN_ENDED_DISTRIB_PROBE`) draws in a synthetic world and never
builds the webapp brain. So the question stays open: does lesioning the spiking draw (`BRAIN_SPIKING_DRAW_LESION=1`,
the likelihood drive into the soft-WTA bank replaced by a uniform drive) change the DISTRIBUTION of replies that the
REAL `webapp.server.brain_chat` gives to an open-ended prompt?

## Code-read diagnosis (before any run)

Two production reply paths exist:
1. **Default turn** (`BRAIN_OPEN_ENDED` unset). `chat.gate()` / `gate_extract()` enters `_generate_hypothesis`, which
   draws each patient through the spiking soft-WTA (`draw_from_weights`). The draw is reached.
2. **Open-ended free-talk turn** (`BRAIN_OPEN_ENDED=1`, the path D4 targets). `webapp/open_ended_chat.answer_turn`
   replaces the whole turn and never calls `chat.gate`. For "what might a dog chase", `extract_topic` returns the
   whole prompt, retrieval finds nothing, and the reply is the fixed abstain string. The draw is NEVER reached, so
   the reply cannot depend on it. This is the bypass defect.

Fix (default-OFF, `BRAIN_OPEN_ENDED_GENERATE_ROUTE`): `webapp/server.py::_open_ended_generate_route`. Under
BRAIN_OPEN_ENDED=1, an explicit generation prompt (the same conservative `_parse_open_ended` patterns gate() uses)
skips the free-talk block and falls through to the ordinary pipeline's GENERATE channel. With the flag off the
function returns before touching `chat`, and the default path never calls it.

## Protocol (fixed now)

- World: 13 teach turns (the `TEACH` list in the runner): rabbit chased by 4 predators, deer by 3, mouse by 2,
  beetle and minnow by 1 each, and the dog chased by 2 (coyote, bear). Then the ask "what might a dog chase" is
  sent K = 40 times, with rich=True, in the same session.
- Arms: each is a fresh subprocess at the same `BRAIN_CHAT_SEED`: `intact`, `intact_rebuild` (determinism) and
  `lesion` (`BRAIN_SPIKING_DRAW_LESION=1`). Every other env var is identical across the three arms.
- Modes: `default`, `oe_unfixed` (BRAIN_OPEN_ENDED=1, route OFF) and `oe_routed` (BRAIN_OPEN_ENDED=1, route ON).
  Both oe modes also set BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK=1 and stub the warm faculty. That is a declared host
  stub: the mouth's FORM is not the quantity measured here.
- Seeds: 42, 43, 44, 100, 101, 102.
- Reply outcome: the volunteered patient of the hypothesis SVO, or ABSTAIN.
- Statistic T: the total-variation distance between the intact and lesion outcome histograms.
- Null: a label-permutation distribution of T over the pooled 80 replies, with B = 10000 permutations seeded by
  the seed, and p = (1 + #{T_null >= T}) / (B + 1).
- Direction D: the mean brain likelihood weight `_weight_partner((dog, chase), p)` of the intact volunteered
  patients minus the lesion's mean.

Per-seed decision rule (implemented in `score_seed`, selftest-covered, each branch able to fail):
- ARM-FAILED: any arm missing, or any reply errored.
- NONDETERMINISTIC: the intact and intact_rebuild reply sequences (or stored facts) are not exactly equal.
- UNDEFINED, never a pass: the draw count is 0 in either arm, the lesion did not reach the draw, or intact
  volunteered nothing.
- LOAD-BEARING: p < 0.05 and D > 0. WRONG-DIRECTION: p < 0.05 and D <= 0. Otherwise NOT-LOAD-BEARING.

Headline, reported per mode: the count of LOAD-BEARING seeds out of 6, reported Option-C beside the thin-probe
count. A claim of "load-bearing on the production turn" requires 6/6 in the named mode. Predicted: `oe_unfixed`
UNDEFINED on every seed (the bypass). No threshold, K, KB or operating point is changed after the first governed
run. A shortfall is reported as measured, and the next method is banked.

## Pipeline smoke run before this commit (no lesion arm, no statistic)

Before committing this file I ran one intact-only worker (seed 42, K=4, `default` mode) to check the pipeline and
its cost. It produced no verdict. It showed three things:
- The brain's comprehension DECLINES some teach turns ("role-binding didn't resolve the PATIENT"). The world stays
  fixed; what the brain learns from it belongs to the brain and is recorded per seed as `stored_facts`.
- The default spiking plausibility gate admitted candidates on this KB.
- A teach phase costs about 500 s on numpy, and each ask about 18 s.

The protocol above is unchanged by the smoke.

## Declared host shortcuts

The KB, the prompt, and the permutation statistic are world and instrument. The plausibility gate stays at its
production default (the spiking associative read); it is not forced to host. The draw itself is a
`cp_firing_states` read on an Izhikevich WTA bank.
