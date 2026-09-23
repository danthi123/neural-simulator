---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
---

# Open-ended generation on the PRODUCTION turn: on seed 42 removing the host likelihood vector from the spiking draw changes the reply, but that seed's draw noise was frozen across asks, so the multi-seed test was redesigned around independent noise-stream sessions (staged on the pool); BRAIN_OPEN_ENDED mode bypassed the draw entirely (2026-09-23)

seed-waiver: seed 42 was run in-session and is ONE observation, under a statistic now withdrawn as a GO rule. The
6-seed amendment-3 run is staged on the pool (see "Staged"). No multi-seed claim is made here.

Lane `research/open-ended-production-turn-lb` (charter D1). Pre-registration:
[`docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md`](../../docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md)
(committed `4edf6fc10`; amendment 1 `94b71d511`; amendment 2 `4f9cc6b3e`; amendment 3 `eefdd666a`; each before the
runs it governs). Instrument: `research/runners/_lbf_open_ended_production_turn_probe.py`.

## Correction 3, after the round-5 re-review of the amendment-3 power simulation (2026-09-23)

1. **"No sim/ or webapp/ edit in this round" ("What is NOT claimed" below) is FALSE and withdrawn as written.**
   The original line (`2a989c367`) read "no sim/ edit" -- true at the time, since only `webapp/server.py` had
   changed. The `8c5d7b03a` commit widened the wording to "no sim/ or webapp/ edit in this round" WHILE, in that
   SAME commit, editing `webapp/server.py` to add the default-OFF `BRAIN_OPEN_ENDED_ACQUIRE_ROUTE`. The bullet
   below is corrected in place; the true, checked claim is: no `sim/` edit anywhere in this lane, and every
   `webapp/server.py` edit (`BRAIN_OPEN_ENDED_GENERATE_ROUTE` in `dfcc72ce4`, `BRAIN_OPEN_ENDED_ACQUIRE_ROUTE` in
   `8c5d7b03a`) is default-OFF and gated by `BRAIN_OPEN_ENDED` being truthy through a short-circuit `and` -- so no
   PRODUCTION DEFAULT was flipped, which is the part of the original claim that stayed true throughout.
2. **The amendment-3 power simulation (correction 1, item 1 in Correction 2's PREREG counterpart) was itself
   biased toward the claim.** See the PREREG's "Amendment-log correction 2 (round-5 review)" for the full
   disclosure: `draw_until_admissible` tested admissibility against the arm's own drive weights, which gave the
   uniform-drive lesion arm a free pass the intact arm did not get. The symmetric, production-faithful fix (fixed
   `n_cand_max=96` too) gives Delta 0.094-0.302, mean 0.193, 6/6 positive -- not the retracted 0.302-0.698, mean
   0.517. This does not change the a3 governed design (M=4, K=8, floor=0.10) already run against the 55 staged
   AWS sessions; see the PREREG for why it is disclosed rather than changed.

## Correction 2, after the re-review of `c807f869b` (amendment 3)

1. **The amendment-2 GO statistic is withdrawn.** Each arm is a deterministic function of the seed. Under H0 the
   per-seed D_s is exactly 0, not a distribution, so the 6-seed sign-flip p = 1/64 would only have said "the modal
   reply changed on six seeds". The "toward likelihood" label also followed by construction. Amendment 3 replaces it.
   Per seed, 4 intact and 4 lesion sessions run as fresh processes, and each draws on its OWN noise stream. The null
   is then a real distribution, and the scorer asserts in data that the streams vary the reply (`noise_live`). A
   seed whose arms are deterministic reads UNDEFINED, never a pass. The GO test is an exact sign test over the 6
   seeds plus an effect floor. Code `36205a4f9`; the tests are in `tests/test_lbf_oe_production_turn_a3.py`, and in
   them an A/A null world GOes at or below alpha.
2. **The "near-argmax / sharpening" reading is withdrawn.** On seed 42 the intact arm gave one patient on 39 of 40
   asks. So did the lesion arm, whose drive is UNIFORM (rabbit 39/40). A uniform drive cannot be an argmax of w, so
   the concentration is not sharpening toward the likelihood. The draw noise was effectively frozen from one ask to
   the next. Amendment 3's per-session stream is not reset between asks, so it also measures within-session diversity.
3. **The identity criterion FAILED as pre-registered for `oe_off`.** Amendment 2 item 9 required IDENTICAL on both env
   sets. The raw compare differs at `.body.open_ended.gen_seconds`, a wall-clock duration. The content verdict with
   that key stripped is IDENTICAL, but the exclusion was added after the raw compare was seen.
   `verdict_oe_off.json` now leads with the raw DIFFERENT and carries the content verdict beside it as post hoc.
4. **The renderer stub is declared for every mode.** `default` too runs with `BRAIN_CHAT_RENDERER=stub` and
   `SIM_DISABLE_LLM=1`. "Production turn" here means the production `brain_chat` pipeline down to the GENERATE
   channel, with the reply renderer stubbed.
5. **The amendment-2 multi-seed runs never produced evidence.** The controllers for `default` seeds 43–102 died
   before writing any worker file, and the parent session killed the remaining local lanes to free RAM. The first
   version of this finding listed them as running. The local staging scripts (`_stage.sh`, `_stage_a2.sh`,
   `_harvest_a2.sh`) now refuse to run. The amendment-3 run is staged on the pool, one full brain per node.

## Correction 1, after the first adversarial review (fix round, 2026-09-23)

The first version of this finding overstated four things. Items 1 and 3 are superseded by Correction 2.

1. The within-session permutation p = 1e-4 was withdrawn: the 40 asks of one session are serially dependent and
   deterministic given the seed, so seed 42 is ONE observation. (Its replacement, the sign-flip over D_s, is
   itself withdrawn above.)
2. **What the lesion removes is a HOST vector.** `BRAIN_SPIKING_DRAW_LESION=1` replaces
   w = `_weight_partner((dog, chase), patients)` with np.ones. w is a sum over the host co-occurrence matrix P that
   `ChatBrain._build_generation_proposer` builds from the stored facts; it is not "the brain's own likelihood". The
   supportable claim is: that host likelihood vector, transmitted through the spiking WTA, changes the reply.
3. The "distributional sampling" narrative was withdrawn. (The "near-argmax" reading that replaced it is itself
   withdrawn above.)
4. **`oe_routed_taught` is not evidence about open-ended mode.** Its teach phase ran with BRAIN_OPEN_ENDED=0 and
   its routed ask falls through to the default pipeline, so its replies equal `default`'s by construction.

## Result, seed 42 (numpy, K=40; description, not a test)

Artifact: research/findings/raw/_load_bearing/_oe_production_turn/a2/default/default_s42_verdict.json (worker
JSONs beside it; the original-scorer verdicts stay at `_oe_production_turn/<mode>/<mode>_s42_verdict.json`).

| mode | amendment-2 per-seed record | intact replies | lesion replies | D_s | spiking draws intact / lesion |
|---|---|---|---|---|---|
| `default` (production pipeline, renderer stub) | DEFINED (label by construction, see Correction 2) | deer 39, rabbit 1 | rabbit 39, deer 1 | +0.95 | 41 / 432 |
| `oe_unfixed` (BRAIN_OPEN_ENDED=1) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | - | **0 / 0** |
| `oe_routed` (+ generate route) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | - | 16000 / 16000 |
| `oe_unfixed_taught` (amendment 1) | UNDEFINED | ABSTAIN 40 | ABSTAIN 40 | - | **0 / 0** |
| `oe_routed_taught` (amendment 1; NOT independent) | as `default` | deer 39, rabbit 1 | rabbit 39, deer 1 | +0.95 | 41 / 432 |

- Host weight vector for (dog, chase, ·): deer 3, rabbit 2, cat 2 (a stored fact, not novel), beetle 1,
  minnow 1.
- The lesion held at measurement: 432 of 432 lesion-arm spiking draws were ablated, 0 intact draws were.
- The user-visible reply changes ("perhaps the dog chases the deer" vs "... the rabbit").
- Both arms repeat one patient from the second ask on. Under the production global RNG the draw noise did not vary
  across asks, so the 40 asks of an arm carry about one draw's worth of information.
- The oe_unfixed_taught seed-43 control also reads UNDEFINED with 0 draws
  (research/findings/raw/_load_bearing/_oe_production_turn/a2/oe_unfixed_taught/oe_unfixed_taught_s43_verdict.json).

## What the spiking part contributes: not determined

The reply path is: host co-occurrence matrix P → host weight vector w → host affine map to drive
(base 110 pA + gain 160 pA × w/peak) → Izhikevich soft-WTA bank with OU membrane noise → host argmax over firing
counts → host plausibility / non-contradiction / moat gates. The lesion removes w, so it cannot isolate the
spiking step. The `host_oracle` arm (`BRAIN_SPIKING_DRAW=0`: the same w drawn by host np.random.choice) was meant to.

Seed 42, host_oracle arm (same stored facts as intact; 76 host draws, 0 spiking draws):

| draw | deer | rabbit | minnow | beetle |
|---|---|---|---|---|
| spiking WTA (intact) | 39 | 1 | 0 | 0 |
| spiking WTA, uniform drive (lesion) | 1 | 39 | 0 | 0 |
| host sampler, same w (host_oracle) | 20 | 16 | 4 | 0 |
| p ∝ w over the admissible set (predicted mass) | 0.43 | 0.29 | 0.14 | 0.14 |

Artifact: research/findings/raw/_load_bearing/_oe_production_turn/a2/default/default_s42_verdict.json. The host
sampler draws from its own generator (`prop.rng`), which advances between asks. The spiking bank's noise came from
the global RNG, which appears not to have varied between asks: both spiking arms repeat one patient. The cause (for
example a per-turn reseed of the global RNG elsewhere in the turn) is not traced. So the 39/40 against 20/40
difference most likely compares a varying sampler with a frozen one. It does not show that the spiking step sharpens the likelihood: the uniform-drive lesion is just as
concentrated. What the spiking step contributes is not determined by this seed. Amendment 3's per-session stream is
the instrument that can measure it.

## The defect found: the BRAIN_OPEN_ENDED reply never reached the generative draw

With `BRAIN_OPEN_ENDED=1` every turn is answered by `webapp/open_ended_chat.answer_turn`, which never calls
`chat.gate`. For "what might a dog chase", `extract_topic` returns the whole prompt, retrieval finds nothing, and the
reply is the fixed abstain. The draw count is **0 in both arms**, even with 13 facts known (`oe_unfixed_taught`).

**Fix 1 (default-OFF): `BRAIN_OPEN_ENDED_GENERATE_ROUTE`** (`webapp/server.py::_open_ended_generate_route`). An
explicit generation prompt (the ChatBrain's own `_parse_open_ended`) skips the free-talk block and reaches the
ordinary GENERATE channel.

**Second bypass: open-ended mode did not learn from being told.** The teach assertions also went to the free-talk
path, so `stored_facts` stayed at the 5 build-time facts and `oe_routed` had nothing novel to volunteer.

**Fix 2 (default-OFF): `BRAIN_OPEN_ENDED_ACQUIRE_ROUTE`**
(`webapp/server.py::_open_ended_acquire_route`). A told SVO assertion reaches in-loop acquisition. The test is
`ChatBrain._is_acquisition_candidate`, a side-effect-free mirror of `_maybe_acquire`'s accept predicate, pinned equal
to it on 14 probe inputs with the B3 organ on and off. The caller now reads
`BRAIN_OPEN_ENDED and not _open_ended_brain_route(chat, msg)`; with both route flags off, `chat` is never touched.

**The true open-ended configuration** is mode `oe_routed_full`: it teaches AND asks under BRAIN_OPEN_ENDED=1 with
both routes. Every turn of this protocol then runs the ordinary pipeline, so its replies are expected to equal
`default`'s. If equal, the honest wording is: open-ended mode reaches the same default GENERATE path through the
routes. That is not a second, independent mechanism. It is NOT staged in the amendment-3 round.

**Flag-off identity, asserted in data** (`research/runners/_lbf_oe_route_flag_off_identity.py`). Each side is a clean
`git archive` (with the untracked data/ corpus linked in) run through the real `brain_chat` on a fixed 9-turn script
(4 assertions, 2 generation prompts, a recall, a free-talk turn, a late assertion), numpy, BRAIN_CHAT_SEED=42.
Pinned pre-change SHA `4c141b8e8` (no route code) vs fix commit `8c5d7b03a` (the server / ChatBrain code is unchanged
from there to this branch head). Verdicts in research/findings/raw/_load_bearing/_oe_production_turn/a2/flag_off_identity/
(the compare re-run at `36205a4f9` on the same dumps):

| env set | pre-registered verdict (raw, exact) | post-hoc content verdict (wall-clock key stripped) | differing paths |
|---|---|---|---|
| `default` (no BRAIN_OPEN_ENDED*) | IDENTICAL | IDENTICAL | none |
| `oe_off` (BRAIN_OPEN_ENDED=1, route flags unset) | **DIFFERENT (criterion FAILED)** | IDENTICAL | `.body.open_ended.gen_seconds` only |
| control: pre vs pre, `default` | IDENTICAL | IDENTICAL | none |

So with the flags off, the default turn is byte-identical to the pre-change code (every full response body, exact
compare). In open-ended mode the pre-registered exact compare FAILED on one field, `gen_seconds`: a `time.time()`
duration from the mouth (webapp/open_ended_chat.py). Every other field is identical. The key was excluded only after
the raw compare had been seen. The control does not cover `gen_seconds` (the default path has no such field), so its
wall-clock nature rests on the source line.

## What is NOT claimed

- No multi-seed claim and no significance. Seed 42 is one observation, and its arms were deterministic.
- Not that the lesion shows the spiking part is load-bearing: the lesioned input is a host vector.
- Not that the spiking bank sharpens or samples the likelihood: on seed 42 its noise was frozen across asks.
- Nothing about BRAIN_OPEN_ENDED free-talk turns: only explicit generation prompts are routed.
- Not that the open-ended-mode flag-off turn is byte-identical: it passes only after a post-hoc exclusion.
- No production default was flipped (see Correction 3, item 1: this bullet previously and incorrectly also said
  "no sim/ or webapp/ edit" -- `webapp/server.py` WAS edited, twice, both times default-OFF; no `sim/` edit).

## Declared host shortcuts

- Teach KB + ask prompt (world); histogram / TV / sign-test statistic (instrument).
- The amendment-3 per-session noise stream (instrument): it chooses which OU-noise realization the bank sees; it
  computes no draw.
- Co-occurrence matrix P and the weight vector w (the lesioned input), and the affine drive map.
- Argmax over the bank's firing counts (host read-out of the spiking winner).
- Hypothesis role induction, the SVO template, the RF-composer moat verify.
- Prompt routers: `_parse_open_ended` and `_is_acquisition_candidate` (host regex / token rules).
- Reply renderer stubbed in every mode; warm Qwen faculty stubbed in the oe_* modes (FORM not measured).
- Spiking: the Izhikevich + OU-noise WTA bank whose firing decides the winner; the production-default spiking
  plausibility read.

## Staged (amendment 3; mini-PC pool, numpy, one full brain per node)

Pool revision: `eefdd666a4f24174cb017cecd6ecd3c9b4394f15` (the amendment-3 prereg commit), provisioned with
`bash tools/pool_provision.sh --isolated --revision eefdd666a pool41 pool42`. Queued with
`bash research/runners/_lbf_open_ended_production_turn_stage_a3_pool.sh <full sha>`: 54 jobs, one fresh-process
session each (6 seeds × (4 intact + 4 lesion + 1 rebuild)), each holding node-level full-brain `flock`s. Outputs land
on the node at `~/derisk-pool/revisions/<sha>/research/findings/raw/_load_bearing/_oe_production_turn/a3/default/`.
Harvest and score (idempotent; any seed without all 9 sessions reads ARM-FAILED, never a pass):
`bash research/runners/_lbf_open_ended_production_turn_harvest_a3.sh <full sha>`, writing
`research/findings/raw/_load_bearing/_oe_production_turn/a3/default_a3_aggregate.json`. GO: 6 seeds DEFINED, exact
sign test p < 0.05 (6/6 positive), and mean Delta ≥ 0.10.

- A non-governed smoke session (seed 7, intact, n0, into `a3_smoke/`) was queued ahead of the 54 governed jobs.
  It checks on a node that the noise stream is engaged (`noise_stream_competes` > 0) and reports the session cost.
  Harvest it with `bash research/runners/_lbf_open_ended_production_turn_harvest_a3.sh <full sha> smoke`.
- Both provisioned nodes passed the provisioner's remote brain build (8382 neurons, 9 bridges, numpy) with the
  corpus and the LTM bundles synced. pool40 was unreachable. Every a3 session runs in this one pool environment,
  on identical i5-10500T nodes, so no session is compared against a local-box session.
- When queued, the pool queue held about 38 jobs from other lanes and both nodes were CPU-saturated (load 23–26 on
  12 cores). With one full brain per node, expect many hours before all 54 sessions are back. Nothing has been
  scored yet: no a3 result exists at the time of writing.
