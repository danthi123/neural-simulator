---
type: finding
status: live
date: 2026-09-23
lane: D6-learn-and-grow
mechanism: D6 gate v3 PRE-REGISTRATION -- the capability gate (does the reply to a later recall question depend on the synaptic write?), primary contrast USE_H vs FREEZE_H with identical input and host code, plus a post-hoc ablation arm (ABL_H) and a host-record inertness control (NOREC_H); scored by research/runners/d6_learn_through_use_lb.py --variant capability
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only (filed before any `capability` arm existed). No result is claimed here.
runner: research/runners/d6_learn_through_use_lb.py
artifacts:
  - research/findings/raw/_d6_learn_through_use/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use/s42_USE_H.json
  - research/findings/raw/_d6_learn_through_use_engram/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use_engram/s42_USE_D.json
  - research/findings/raw/_d6_learn_through_use_prune/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use_prune/s42_SHUF_H.json
  - research/findings/raw/_d6_learn_through_use_prune/d6_ltu_prune_s42_rescored_original_C4.json
---

# D6 learn-through-use, gate v3: PRE-REGISTRATION (filed before any `capability` run)

**Filed 2026-09-23 (fix round 3), in its own commit, before any `--variant capability` arm was built.** Seeds 42 43
44 100 101 102. Arm dir `research/findings/raw/_d6_learn_through_use_v3`.

## What was SEEN when this was written

- base s42 (all 5 arms): NO-GO on C3 (`research/findings/raw/_d6_learn_through_use/d6_ltu_s42_smoke.json`).
- engram s42 (all 5 arms): NO-GO on C3 and C4
  (`research/findings/raw/_d6_learn_through_use_engram/d6_ltu_engram_s42_smoke.json`).
- prune s42 (all 5 arms): NO-GO on C3 and C4 under the original C4; banked as an invalid instrument
  (`research/findings/raw/_d6_learn_through_use_prune/d6_ltu_prune_s42_rescored_original_C4.json`).
- base-variant pool arm files for s43 and s100 (USE_H, USE_H_REP, SHUF_H, and s43 FREEZE_H), still running.
- No `readtime` arm and no `capability` arm exists. The six `readtime` pool lines were never dispatched.

The v3 design below was therefore written with every earlier s42 result in view. It is not blind to them. What makes it
a registration is that no v3 arm existed when it was committed, and that each criterion is stated with its predicted
outcome, so a reader can see which ones were expected to pass and which one is the open question.

## Why gate v2 is superseded

Gate v2's own registration predicted it would fail:
- C4 on the teach-turn ack text;
- C3 and C3b on the DA-mode suffix, which comes from habituation to the heard word 'wolf'.

So a v2 run could only return a known NO-GO, and neither failure is about the capability. Two parts of v2 carry over
unchanged: the ABL_H ablation arm, and reading the lever at probe time. Two parts are removed:
- the requirement that the frozen arm's full reply equal the SHUF_H control's. SHUF_H hears different words, so this
  compared learning with word exposure;
- the requirement that the teach-turn ack be identical (see K6).

## The question

**Does the reply to a later recall question depend on the synaptic write?** The lesion freezes only the synaptic
update: eta is 0 inside the Hebbian encode. The input, the flags and the host code path are identical between
USE_H and FREEZE_H.

## Arms

All arms run with `BRAIN_D6_HEBBIAN_STORE=1 BRAIN_D6_ENGRAM_VOCAB=1 BRAIN_D6_ENGRAM_READTIME=1`, except USE_D.

| arm | teach turn | difference from USE_H |
|---|---|---|
| USE_H | "the wolf hunts the deer" | (treatment) |
| USE_H_REP | same | rebuilt (null) |
| SHUF_H | "the fox eats the berry" | different content |
| FREEZE_H | same | `BRAIN_D6_HEBBIAN_FREEZE=1`: eta=0 for in-conversation writes (THE lesion) |
| ABL_H | same | after the teach turn, `ablate_block` zeroes the taught block's synapses (kb record kept) |
| NOREC_H | same | FREEZE_H, then after the teach turn `remove_block_record` removes the frozen fact's kb RECORD (it refuses unless all the block's weights are exactly 0) |
| USE_D | same | production direct copy, all D6 flags 0 (secondary, non-scoring) |

Turns: teach, d1 "what does the cat eat", d2 "what does the dog chase", probe "what does the wolf hunt", xprobe "what
does the fox eat". Decision fields per turn: `abstained`, `recalled_svo`, `answer`.

## Criteria (per seed; ALL of K1..K7)

- **K1 LEARNS.** USE_H.probe recalls 'deer': it is not abstained, and 'deer' is in recalled_svo.
- **K2 USE-SPECIFIC.** Every part of the double dissociation holds:
  - SHUF_H.probe does not recall 'deer';
  - USE_H.probe decision != SHUF_H.probe decision;
  - SHUF_H.xprobe recalls 'berry';
  - USE_H.xprobe does not recall 'berry'.
- **K3 THE REPLY DEPENDS ON THE WRITE.** This is the capability. All of the following:
  - FREEZE_H.probe does not recall 'deer';
  - the word 'deer' does not appear in FREEZE_H.probe's answer text (word-boundary match);
  - FREEZE_H.probe decision != USE_H.probe decision.
- **K4 ... READ OFF THE SYNAPSE AT PROBE TIME.** ABL_H.probe does not recall 'deer', and 'deer' is not in its answer
  text.
- **K5 THE HOST RECORD IS INERT.** NOREC_H equals FREEZE_H on the decision fields of every turn, answer text
  included. The teach-turn part of this is a determinism check, because the removal happens after the teach turn. K5
  replaces the earlier wording that "every kb reader asks the engram". Only four readers do (`ENGRAM_READTIME_ROUTED`).
  K5 measures whether the others (`ENGRAM_READTIME_NOT_ROUTED`) carry the record of an unheld fact into any reply.
- **K6 THE LESION IS WRITE-ONLY.** All of the following:
  - lever, read at the probe turn: the taught block's mean |w| is > 0.5 in USE_H and exactly 0 in FREEZE_H and ABL_H;
  - the write counter is installed in FREEZE_H, ABL_H and NOREC_H, and each records zero store writes after the teach turn;
  - the encode episode is the same: FREEZE_H's `d6_last_encode` has the same `steps`, `phase_lock_steps` and
    `block` as USE_H's, with `frozen` True vs False;
  - the teach-turn parse (`abstained`, `recalled_svo`) is the same in FREEZE_H and USE_H;
  - d1 and d2 are the same, on every decision field, in FREEZE_H and ABL_H as in USE_H.
- **K7 NULL.** USE_H equals USE_H_REP on every turn. ABL_H.teach equals USE_H.teach, which is a determinism check,
  not lesion evidence.

**The teach-turn ACK text is NOT scored.** It is reported as `teach_ack_same_FREEZE_vs_USE`. It is rendered after the
write, in the same turn, and its render-verify reads the known-word sets. Under the read-time view those sets come from
the engram that was just written, so the ack is itself a reply that depends on the write. The evidence:
- in the base variant, where the sets come from the host list, the FREEZE ack equalled the USE ack (s42 C4 held);
- in engram s42 and prune s42 the FREEZE ack differed, while USE_D's ack equalled USE_H's.

**VOID / UNDEFINED.** Any of the following makes that arm VOID, and a seed with a VOID scored arm is UNDEFINED (never
a pass, never a NO-GO):
- a scored arm is missing or has an errored turn;
- ABL_H's ablation was not applied;
- NOREC_H's record was not removed;
- a lesion arm's write counter was not installed.

USE_D is secondary: if it is void, only the secondary C7 no-regression reads UNDEFINED.

**Aggregate GO** requires all 6 seeds to be defined and GO.

**Reported, never scored:** secondary C7 (USE_D == USE_H on teach/d2/probe/xprobe), the teach-ack equality, the
homeostatic-scaling call count, the d6_ops integrity counters, and `attributable_to_write`.

## Predictions, written down before any v3 arm

| criterion | prediction | basis |
|---|---|---|
| K1 | pass | C1 held at s42 in base, engram and prune |
| K2 | pass | C2 and C5 held at s42 in all three |
| K3 | pass | FREEZE_H.probe abstained, with no 'deer' in the answer, in all three s42 smokes |
| K4 | pass | the unit test shows the read-time view drops an ablated block and recall abstains |
| **K5** | **open** | the substantive unknown. `_ensure_sequencer` builds the seq fabric's fact list from `kb[:K]`, which includes the frozen 'wolf' fact in FREEZE_H and not in NOREC_H; whether that reaches any reply is not known |
| K6 | pass | d6_last_encode steps/phase_lock_steps/block were equal at s42 in all three; d1/d2 were equal |
| K7 | pass | C6 held at s42 in all three |

This gate is not registered to fail. Its one open criterion is K5, and a K5 failure would name a host reader that
still carries a fact the synapses do not hold. That is a real finding, and it has a named next method: retire the kb
list as a membership source, per d6_hebbian_store shortcut (g).

## Run order (registered)

1. Run s42 locally first, all 7 arms, memcapped. It counts as seed 42 of the 6-seed set.
2. If s42 is GO, stage the other five seeds on the pool.
3. If s42 is NO-GO, do NOT stage the pool run. The pool is RAM-tight, and a failing seed means the design or the
   mechanism needs fixing first. Bank the NO-GO with its next method.
4. If s42 is UNDEFINED, fix the instrument and re-run s42.

Scoring command, verbatim:

```
.venv/bin/python -m research.runners.d6_learn_through_use_lb --score-only --variant capability \
    --arm-dir research/findings/raw/_d6_learn_through_use_v3 --seeds 42 43 44 100 101 102 \
    --json <ARM_DIR>/d6_ltu_v3_6seed_verdict.json
```

(`<ARM_DIR>` is the `--arm-dir` above; the verdict file does not exist until the run.)

The selftest (`--selftest`) must print SELFTEST PASS. Its `gate_v3` block covers:
- every K criterion failing in its failing direction;
- a framing-only leak in NOREC_H failing K5;
- an ack-only teach difference NOT failing;
- each VOID condition reading UNDEFINED.

## Scope, and what this does not claim

- One learning pathway: in-conversation declarative fact acquisition on the tiny-demo brain.
- The instructive pattern is still the composer's FHRR bind/bundle.
- The host shortcuts (a)-(j) declared in `research/runners/d6_hebbian_store.py` are declared, not replaced. They
  include:
  - the host-wired instructive pathway;
  - the host phase-lock loop;
  - the W_MAX clamp;
  - the held-threshold;
  - the partial read-time view;
  - the block-to-words map.
- A v3 GO would show that this reply depends on this synaptic write, on this protocol. It would not show open-ended
  learning, and it would not flip a default.

## ADDENDUM A6 — filed fix round 4, after the round-3 re-review of the s42 capability result above

**EXPO_H was dropped from this gate without a word.** ADDENDUM A4 in the v2 prereg
(`research/findings/2026-09-23-d6-learn-through-use-v2-PREREGISTRATION-readtime-view-and-engram-ablation.md`),
filed before any `readtime` arm existed, committed: "the next registered gate (v3) would then use the
exposure-matched control as the primary comparison, stated in advance." This v3 registration removes the
FREEZE_H==SHUF_H equality for exactly the exposure reason A4 gave (see "Why gate v2 is superseded" above), but it
never mentions EXPO_H, and the capability variant built no EXPO_H arm. K3's weaker replacement check (FREEZE_H does
not recall 'deer', 'deer' is absent from its answer text, FREEZE_H != USE_H) can, with eta=0 forcing the weights to
exactly 0, pass through a host leak of the fact text outside `composer.kb` that an EXPO_H comparison exists
specifically to catch. **Seen when this addendum was filed:** the s42 result above (K1-K7 all pass, INCOMPLETE
1/6) was already banked, and the round-3 re-review of the fix-round-3 commit named the omission.

**Fix.** `research/runners/d6_learn_through_use_lb.py` AMENDMENT A6 restores EXPO_H into the `capability` variant's
`arms_for` / `score_seed_v3`, on A4's own terms: SECONDARY, NON-SCORING. Two new checks are REPORTED, never scored
(cannot move K1-K7 or `go`):
- **K3e**: FREEZE_H.probe == EXPO_H.probe, and FREEZE_H does not recall 'deer';
- **K4e**: ABL_H.probe == EXPO_H.probe, and ABL_H does not recall 'deer'.

**Prediction, written before any capability-variant EXPO_H arm exists:** K3e and K4e PASS. The already-banked s42
FREEZE_H/ABL_H probe reply ("I don't know about that. My curiosity is piqued -- I haven't learned about wolf yet:
what can you tell me about wolf?") already reads as the unfamiliar-word reply an exposure-matched no-write control
is expected to produce, unlike the base variant's pre-read-time-view reply to the same probe ("Setting the held
thread aside -- On wolf, then -- I don't know about that."), which DID carry a familiarity trace. If K3e/K4e FAIL
instead, that would mean K1-K7's PASS partly rode on word exposure rather than the write -- a real finding the
scored gate does not currently detect on its own.

**Reporting the gate BOTH ways, per this fix round:**
- **Scored (K1-K7), unaffected:** the s42 verdict above still stands unchanged -- all seven criteria pass. EXPO_H
  cannot move this verdict by construction (SECONDARY, NON-SCORING).
- **Secondary (K3e/K4e), status:** wired and selftest-verified (`--selftest`:
  `v3_missing_EXPO_is_secondary_only`, `v3_expo_secondary_can_fail_without_touching_go`; both use synthetic arms,
  no live brain required), but **the s42 EXPO_H arm itself has not yet been run.** The local box was RAM-bound at
  fix-round-4 time (under 1 GB free, 16+ GB already swapped) when this addendum was filed, and a new ~10-25
  min/12 GB-memcapped local build was judged unsafe to stage alongside concurrent lanes on that box. This is the
  next action on this lane, not a claimed result: read the file `s42_EXPO_H.json`, once it exists, under
  `research/findings/raw/_d6_learn_through_use_v3/` for the measured K3e/K4e outcome, and treat any such reading
  dated before that file exists as unfounded.

**ADDENDUM A7 — fanout gap, filed fix round 5, after the round-4 re-review of this addendum.** The 5-seed
`capability` pool fanout (s43/s44/s100/s101/s102) was staged before this A6 landed, pinned to a revision with 7
arms per seed (USE_H, USE_H_REP, SHUF_H, FREEZE_H, ABL_H, NOREC_H, USE_D) and no EXPO_H line. When that fanout
returns, K3e/K4e will read UNDEFINED on all five of those seeds -- the s42 measurement above is the only seed that
can carry the secondary check until a post-A6-revision EXPO_H pool line is staged for the other five. This is
recorded here rather than fixed by editing the live pool queue, which is running infrastructure state outside this
finding's or this fix round's scope to rewrite. Any future report of the v3 aggregate GO must say the EXPO_H
secondary is s42-only (1/6 seeds) unless and until those five EXPO_H lines have been staged, run, and harvested --
it must not imply a 6-seed secondary read from the 6-seed K1-K7 fanout alone.
