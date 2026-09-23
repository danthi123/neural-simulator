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

## Amendment 1 (committed before any run it governs)

The seed-42 `oe_*` workers exposed a second bypass (their artifacts are in
`research/findings/raw/_load_bearing/_oe_production_turn/`). With BRAIN_OPEN_ENDED=1 the TEACH assertions also go
to the free-talk path, so in-loop acquisition never runs and `stored_facts` stays at the 5 build-time facts. The
ask then has nothing novel to volunteer, so `oe_routed` cannot exercise the draw under the original protocol. The
original `oe_unfixed` and `oe_routed` rows are kept and reported as run.

Two new modes isolate the ASK path from this teach-path bypass:
- `oe_unfixed_taught` and `oe_routed_taught` use the same env as `oe_unfixed` and `oe_routed`.
- The only difference: they run the 13 TEACH turns with BRAIN_OPEN_ENDED=0 (the ordinary chat path) in the same
  session and the same ChatBrain, then restore the arm env before the 40 asks.

The statistic, null, alpha, K, KB and decision rule are unchanged, and so are the seeds. The teach-path bypass is
a separate defect; this lane records it as a residual and does not fix it.

## Declared host shortcuts

The KB, the prompt, and the permutation statistic are world and instrument. The plausibility gate stays at its
production default (the spiking associative read); it is not forced to host. The draw itself is a
`cp_firing_states` read on an Izhikevich WTA bank.

## Amendment 2 — fix round after adversarial review (committed before any run it governs)

### Amendment log: what I had seen when writing this

- All seed-42 artifacts of all five modes (verdicts, histograms, worker JSONs). Seed 42 is therefore IN-SAMPLE for
  this amendment's design, and is declared so.
- The adversarial review of the lane (fix-required, not safe to merge).
- NOT seen: any seed 43/44/100/101/102 output. The `default` run for those seeds (launched 13:20 from worktree
  `wf_6cf1082d-b06-2` under the original rule) had produced no worker file when this was written. The
  `oe_unfixed_taught` s43 files exist there; I listed their names and did not open them.
- The `oe_routed_taught` 5-seed controller was stopped at ~13:44, before it wrote any s43+ file (reason below).

### What changes, and why

1. **The unit is the seed, not the ask.** The 40 asks of one session are serially dependent, and each arm is a
   deterministic function of the seed (intact and rebuild give the same sequence). The original label-permutation
   p over 80 pooled replies treated them as exchangeable and so reported p = 1e-4 for ONE observation. It is
   withdrawn. Per seed the scorer records D_s (the mean host-likelihood weight of the intact volunteered patients
   minus the lesion's), the modal patient of each arm, and a direction label. The within-session TV is kept as a
   descriptive effect size only.
2. **Seed-level test.** The exact one-sided sign-flip randomization p over the seeds' D_s (H0: intact and lesion
   labels exchangeable within a seed). With 6 seeds the smallest attainable p is 1/64; one seed can never be
   significant (p = 0.5). **GO for a mode: 6 seeds, all DEFINED (no UNDEFINED / NONDETERMINISTIC / ARM-FAILED),
   and sign-flip p < 0.05.** Because seed 42 is in-sample, the same p over the 5 held-out seeds (43, 44, 100, 101,
   102; smallest p = 1/32) is reported beside it; a GO that holds only with seed 42 included is reported as such.
3. **Chance base rate.** Per seed, with A = the approximate admissible set (positive host weight, not a stored
   (dog, chase, p) fact, plus anything an arm volunteered): if the lesion's reply were a uniformly random member of
   A, P(modal changes) = 1 − 1/|A| and P(D_s > 0 | intact at the peak) = (#A below the peak)/|A|. The expected
   counts over the seeds are reported next to the observed count of seeds whose modal reply changed. They are
   context for the count, not a second test.
4. **The lesioned edge is a HOST vector, and is declared.** `BRAIN_SPIKING_DRAW_LESION=1` replaces
   w = `_weight_partner` (a sum over the HOST co-occurrence matrix P) with np.ones before the host affine map into
   the spiking bank's drive. The claim this lesion can support is: the host likelihood vector, transmitted through
   the spiking WTA, is load-bearing on the reply. It cannot show that the spiking part is load-bearing.
5. **New arm `host_oracle`** (`BRAIN_SPIKING_DRAW=0`: the same host w drawn by host np.random.choice). Run for the
   `default` mode on all 6 seeds. It measures what the spiking WTA contributes: the fraction of asks on the intact
   modal patient under the spiking draw minus that under the host sampler ("sharpening"), and their TV.
   Descriptive, with a stated prediction from seed 42: sharpening > 0 (the bank is near-argmax, not a sampler).
6. **The sampling narrative is withdrawn.** Seed 42's intact bank volunteered the likelihood peak 39/40 times where
   p ∝ w gives it about 3/7. At this operating point the WTA behaves as a near-argmax. That is recorded as an
   observation (generative diversity is lost), not as distributional sampling.
7. **The true open-ended configuration: new mode `oe_routed_full`.** Teach AND ask both under BRAIN_OPEN_ENDED=1,
   with both default-OFF routes: `BRAIN_OPEN_ENDED_GENERATE_ROUTE=1` and the new `BRAIN_OPEN_ENDED_ACQUIRE_ROUTE=1`
   (a told SVO assertion reaches in-loop acquisition, closing the teach-path bypass). With both routes on, every
   turn of this protocol leaves the free-talk block, so the replies are EXPECTED to equal `default`'s. That is
   checked in data (`equals_default_replies`, exact compare of reply texts per seed and arm). If they are equal, the
   finding states that open-ended mode reaches the SAME default GENERATE path through the routes; it is not a
   second, independent piece of evidence. Same GO rule as item 2.
8. **`oe_routed_taught` is not independent evidence** (its teach phase ran with BRAIN_OPEN_ENDED=0 and its routed
   ask falls through to the default pipeline). Its seed-42 row is kept and labelled; it is not run on more seeds.
9. **Flag-off identity, in data.** Pinned pre-change SHA `4c141b8e8` (origin/main; has neither route) against the
   lane code commit `8c5d7b03a`, each a clean `git archive`, same fixed 9-turn chat script, env sets `default` and
   `oe_off` (BRAIN_OPEN_ENDED=1, route flags unset). Exact compare of every full response body. A pre-vs-pre rerun
   of `default` is the nondeterminism control. Required: IDENTICAL on both env sets and on the control.

Unchanged: the world (TEACH, ASK), K = 40, seeds, arms intact / intact_rebuild / lesion, the UNDEFINED conditions,
alpha = 0.05. No constant is fit on any seed. The `oe_unfixed_taught` control continues (predicted UNDEFINED, 0
draws, on every seed). The in-flight `default` worker JSONs for seeds 43–102 are scored by the amendment-2 scorer;
the worker protocol that produces them is unchanged.
