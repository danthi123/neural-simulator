---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
---

# Open-ended generation on the PRODUCTION turn: removing the host likelihood vector from the spiking draw changes the reply (seed 42, one observation), and BRAIN_OPEN_ENDED mode bypassed the draw entirely (2026-09-23)

seed-waiver: seed 42 was run in-session and is ONE observation. Seeds 43/44/100/101/102 are staged (see "Staged").
No multi-seed claim is made here; under the amended rule a single seed cannot reach significance.

Lane `research/open-ended-production-turn-lb` (charter D1). Pre-registration:
[`docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md`](../../docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md)
(committed `4edf6fc10`; amendment 1 `94b71d511`; amendment 2 `4f9cc6b3e`, each before the runs it governs).
Instrument: `research/runners/_lbf_open_ended_production_turn_probe.py`.

## Correction after adversarial review (fix round, 2026-09-23)

The first version of this finding overstated four things. Each is corrected below and pinned by a test that fails
on the pre-fix runner `b34be3f2f` and passes after (11 of the 14 tests in `tests/test_lbf_open_ended_production_turn_probe.py`
and `tests/test_open_ended_generate_route.py` fail on `b34be3f2f`; all 14 pass at `c07cb09d7`).

1. **The p-value is withdrawn.** It treated the 40 asks of one session as 40 exchangeable samples and reported
   p = 1e-4. The asks are serially dependent, and each arm is a deterministic function of the seed (intact and
   rebuild give the same 40-reply sequence). Seed 42 is ONE observation. Amendment 2 makes the seed the unit: an
   exact sign-flip test over the seeds' D_s, whose smallest value with one seed is 0.5.
2. **What the lesion removes is a HOST vector.** `BRAIN_SPIKING_DRAW_LESION=1` replaces
   w = `_weight_partner((dog, chase), patients)` with np.ones. w is a sum over the host co-occurrence matrix P that
   `ChatBrain._build_generation_proposer` builds from the stored facts; it is not "the brain's own likelihood". The
   supportable claim is: that host likelihood vector, transmitted through the spiking WTA, changes the reply.
3. **The draw is near-argmax here, not a sampler.** The intact bank volunteered the peak (deer) 39/40 times. Drawing
   p ∝ w over the admissible set would give deer about 3/7. The "distributional sampling" narrative is withdrawn.
4. **`oe_routed_taught` is not evidence about open-ended mode.** Its teach phase ran with BRAIN_OPEN_ENDED=0 and
   its routed ask falls through to the default pipeline, so its replies equal `default`'s by construction.

## Result, seed 42, amendment-2 scoring (numpy, K=40)

Artifact: research/findings/raw/_load_bearing/_oe_production_turn/a2/default/default_s42_verdict.json (worker
JSONs beside it; the original-scorer verdicts stay at `_oe_production_turn/<mode>/<mode>_s42_verdict.json`).

| mode | per-seed verdict | intact replies | lesion replies | D_s | spiking draws intact / lesion |
|---|---|---|---|---|---|
| `default` (production default turn) | DEFINED, CHANGED-TOWARD-LIKELIHOOD | deer 39, rabbit 1 | rabbit 39, deer 1 | +0.95 | 41 / 432 |
| `oe_unfixed` (BRAIN_OPEN_ENDED=1) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | - | **0 / 0** |
| `oe_routed` (+ generate route) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | - | 16000 / 16000 |
| `oe_unfixed_taught` (amendment 1) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | - | **0 / 0** |
| `oe_routed_taught` (amendment 1; NOT independent) | as `default` | deer 39, rabbit 1 | rabbit 39, deer 1 | +0.95 | 41 / 432 |

- Host weight vector for (dog, chase, ·): deer 3, rabbit 2, cat 2 (a stored fact, not novel), beetle 1,
  minnow 1. Approximate admissible set: beetle, deer, minnow, rabbit.
- Chance base rate for this seed: a uniformly random lesion favourite would differ from the intact modal with
  probability 0.75, and sit below the peak (D_s > 0) with probability 0.75. So one seed with D_s > 0 is weak
  evidence on its own; the 6-seed sign-flip is the test.
- The lesion held at measurement: 432 of 432 lesion-arm spiking draws were ablated, 0 intact draws were.
- The user-visible reply changes ("perhaps the dog chases the deer" vs "... the rabbit").
- The lesioned arm's favourite (rabbit) is set by the draw bank's own neuron heterogeneity under uniform drive.
- The oe_unfixed_taught seed-43 control also reads UNDEFINED with 0 draws
  (research/findings/raw/_load_bearing/_oe_production_turn/a2/oe_unfixed_taught/oe_unfixed_taught_s43_verdict.json).

## What the spiking part contributes

The reply path is: host co-occurrence matrix P → host weight vector w → host affine map to drive
(base 110 pA + gain 160 pA × w/peak) → Izhikevich soft-WTA bank with OU membrane noise → host argmax over firing
counts → host plausibility / non-contradiction / moat gates. The lesion removes w, so it cannot isolate the
spiking step. The staged `host_oracle` arm (`BRAIN_SPIKING_DRAW=0`: the same w drawn by host np.random.choice)
does. Seed-42 prediction from the numbers above: the spiking bank sharpens toward the host argmax (deer 0.975
under spiking vs about 0.43 predicted under p ∝ w). If that holds, the spiking bank's contribution at this operating
point is a near-deterministic selection of the host-likelihood peak, and the loss of generative diversity is a
defect to record, not a sampling property to credit.

## The defect found: the BRAIN_OPEN_ENDED reply never reached the generative draw

With `BRAIN_OPEN_ENDED=1` every turn is answered by `webapp/open_ended_chat.answer_turn`, which never calls
`chat.gate`. For "what might a dog chase", `extract_topic` returns the whole prompt, retrieval finds nothing, and the
reply is the fixed abstain. The draw count is **0 in both arms**, even with 13 facts known (`oe_unfixed_taught`).

**Fix 1 (default-OFF): `BRAIN_OPEN_ENDED_GENERATE_ROUTE`** (`webapp/server.py::_open_ended_generate_route`). An
explicit generation prompt (the ChatBrain's own `_parse_open_ended`) skips the free-talk block and reaches the
ordinary GENERATE channel.

**Second bypass: open-ended mode did not learn from being told.** The teach assertions also went to the free-talk
path, so `stored_facts` stayed at the 5 build-time facts and `oe_routed` had nothing novel to volunteer.

**Fix 2 (default-OFF, this fix round): `BRAIN_OPEN_ENDED_ACQUIRE_ROUTE`**
(`webapp/server.py::_open_ended_acquire_route`). A told SVO assertion reaches in-loop acquisition. The test is
`ChatBrain._is_acquisition_candidate`, a side-effect-free mirror of `_maybe_acquire`'s accept predicate, pinned equal
to it on 14 probe inputs with the B3 organ on and off. The caller now reads
`BRAIN_OPEN_ENDED and not _open_ended_brain_route(chat, msg)`; with both route flags off, `chat` is never touched.

**The true open-ended configuration is staged:** mode `oe_routed_full` teaches AND asks under BRAIN_OPEN_ENDED=1 with
both routes. Every turn of this protocol then runs the ordinary pipeline, so its replies are expected to equal
`default`'s; the harvest checks that exactly (`equals_default_replies`). If equal, the honest wording is: open-ended
mode reaches the same default GENERATE path through the routes. That is not a second, independent mechanism.

Flag-off identity: NOT yet asserted in data. The staged identity lane compares every full response body of a fixed
9-turn chat at the pinned pre-change SHA `4c141b8e8` against the fix commit `8c5d7b03a`, env sets `default` and
`oe_off`, with a pre-vs-pre control. Until it lands the wording is "expected unchanged (unverified)".

## What is NOT claimed

- No multi-seed claim, and no significance: seed 42 is one observation (sign-flip p over one seed = 0.5).
- Not that the spiking part is load-bearing: the lesioned input is a host vector.
- Not distributional sampling: at this operating point the draw is near-argmax.
- Nothing about BRAIN_OPEN_ENDED free-talk turns: only explicit generation prompts are routed.
- No production default was flipped; no sim/ edit.

## Declared host shortcuts

- Teach KB + ask prompt (world); histogram / TV / sign-flip statistic (instrument).
- Co-occurrence matrix P and the weight vector w (the lesioned input), and the affine drive map.
- Argmax over the bank's firing counts (host read-out of the spiking winner).
- Hypothesis role induction, the SVO template, the RF-composer moat verify.
- Prompt routers: `_parse_open_ended` and `_is_acquisition_candidate` (host regex / token rules).
- Warm Qwen faculty stubbed in the oe_* modes (FORM not measured).
- Spiking: the Izhikevich + OU-noise WTA bank whose firing decides the winner; the production-default spiking
  plausibility read.

## Staged (amendment 2; local, numpy, memcapped, each lane started only when tools/mem_ok.sh passes)

`bash research/runners/_lbf_open_ended_production_turn_stage_a2.sh <pre_tree> <post_tree>`, launched from worktree
`/home/dant123/Projects/sim/.claude/worktrees/wf_a686cbcd-9ff-2`. Lanes: flag-off identity; `default` host_oracle arm
(6 seeds); `oe_routed_full` (6 seeds, two lanes). The `default` and `oe_unfixed_taught` arms for seeds 43–102 come
from the original staging in worktree `/home/dant123/Projects/sim/.claude/worktrees/wf_6cf1082d-b06-2` (unchanged
worker protocol; the 5-seed `oe_routed_taught` run there was stopped as non-independent). Harvest (idempotent):
`bash research/runners/_lbf_open_ended_production_turn_harvest_a2.sh`, writing
`research/findings/raw/_load_bearing/_oe_production_turn/a2/<mode>_aggregate.json`. GO per mode: 6 seeds, all
DEFINED, sign-flip p < 0.05, with the held-out 5-seed p reported beside it (seed 42 is in-sample for amendment 2).
