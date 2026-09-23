---
type: finding
status: live
date: 2026-09-23
lane: D6-learn-and-grow
mechanism: D6 gate v2 PRE-REGISTRATION -- read-time engram view (BRAIN_D6_ENGRAM_READTIME, no host retraction) + a post-hoc engram-ablation arm (ABL_H), scored by research/runners/d6_learn_through_use_lb.py --variant readtime
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only (filed before any readtime result). No result is claimed here.
runner: research/runners/d6_learn_through_use_lb.py
artifacts:
  - research/findings/raw/_d6_learn_through_use_engram/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use_engram/s42_SHUF_H.json
  - research/findings/raw/_d6_learn_through_use/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use_prune/d6_ltu_prune_s42_rescored_original_C4.json
  - research/findings/raw/_d6_learn_through_use_prune/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use_prune/s42_SHUF_H.json
---

# D6 learn-through-use, gate v2: PRE-REGISTRATION (filed before any `readtime` run)

**Filed 2026-09-23 ~16:05 UTC**, in its own commit, after the mechanism commit `f233f68f6` and before any
`--variant readtime` arm was built. The results SEEN when this was written are listed in the runner's AMENDMENT LOG
(`research/runners/d6_learn_through_use_lb.py`): base s42 (NO-GO on C3), engram s42 (NO-GO on C3 and C4), and two
prune s42 arm files with no prune verdict. Seeds 42 43 44 100 101 102.

## Why a v2 gate

The adversarial review of `387d96a5b` found that the prune variant could not answer the D6 question.
- Its lesion arm ran a host step, `retract_unencoded_last`, that the treatment arm never runs, because every unfrozen
  write saturates and is therefore always "held". A C3 pass could come from that host deletion, not from the synapses.
- The engram check ran once, at write time, so a later loss of the engram would never reach the reply.
- Its C4 relaxation was written after the engram-variant C4 failure had been seen. That is reverted (AMENDMENT A1).

The v2 design removes the host deletion and adds a lesion that the write-time check cannot see.
- `BRAIN_D6_ENGRAM_READTIME=1`: every reader of "which facts does the brain hold" asks the substrate at read time.
  This covers `gnw_thought_swap._known_concepts`, `gnw_multistep_deliberation._all_concepts`, the episodic content
  lookup in `brain_reply`, and `ChatBrain._refresh_facts`, which now re-runs at the start of every turn. No kb record
  is ever deleted. The same host code runs in every arm; only the synapses differ.
  [ERRATUM, fix round 3: both sentences above overclaim. "Every reader" is false: only these four readers are
  routed, and `d6_hebbian_store.ENGRAM_READTIME_NOT_ROUTED` lists the host-kb readers that are not. "Only the synapses
  differ" is false for ABL_H, which also runs the experimenter's cache invalidation and a re-read of every block. The
  gate v2 criteria are unchanged. Gate v2 is superseded by gate v3
  (`research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md`), which MEASURES
  host-record inertness in its NOREC_H arm.]
- **ABL_H**: identical to USE_H. After the teach turn, the experimenter zeroes the taught block's synapses and leaves
  the kb record intact. If any reader still takes familiarity from the host list, ABL_H's probe reply keeps a use-trace
  that SHUF_H does not have.

## Arms (each a fresh tiny-demo brain at `BRAIN_CHAT_SEED=seed`, numpy, stub renderer, no LLM)

Turns: teach, d1 "what does the cat eat", d2 "what does the dog chase", probe "what does the wolf hunt",
xprobe "what does the fox eat".

| arm | flags | teach |
|---|---|---|
| USE_H | HEBBIAN_STORE=1, FREEZE=0, ENGRAM_VOCAB=1, ENGRAM_READTIME=1 | the wolf hunts the deer |
| USE_H_REP | same as USE_H, rebuilt | same |
| SHUF_H | same as USE_H | the fox eats the berry |
| FREEZE_H | USE_H + FREEZE=1 (eta=0 for in-conversation writes only) | the wolf hunts the deer |
| ABL_H | USE_H + taught block's synapses zeroed after the teach turn | the wolf hunts the deer |
| USE_D | every D6 flag explicitly 0 (production direct copy) | the wolf hunts the deer |

## Gate v2 (per seed, ALL must hold)

Decision fields per turn are `abstained`, `recalled_svo` and `answer`. "Recalls X" means not abstained and X is in
`recalled_svo`.
- **C1 learns**: USE_H.probe recalls deer.
- **C2 use changes the reply**: USE_H.probe decision differs from SHUF_H.probe, and SHUF_H.probe does not recall deer.
- **C3 freeze removes**: FREEZE_H.probe does not recall deer, and its full decision (all three fields) equals SHUF_H.probe.
- **C3b ablation removes**: ABL_H.probe does not recall deer, and its full decision equals SHUF_H.probe.
- **C4 write-only**, the ORIGINAL C4 plus lesion persistence. All of the following must hold.
  - FREEZE_H.teach equals USE_H.teach on all three fields, ack text included.
  - FREEZE_H.d2 equals USE_H.d2.
  - The taught block's mean |w|, read at the PROBE turn, is above 0.5 in USE_H and exactly 0 in FREEZE_H and ABL_H.
  - ABL_H.teach and ABL_H.d2 equal USE_H's.
  - No store write occurs after the teach turn in FREEZE_H or ABL_H (the `_write_block` counter is empty).
- **C5 specific**: SHUF_H.xprobe recalls berry, and USE_H.xprobe does not.
- **C6 deterministic**: USE_H equals USE_H_REP on every turn.
- **C7 no-regression**: USE_D equals USE_H on teach, d2, probe and xprobe.

**Aggregate GO** requires all 6 seeds to be defined and GO. An arm that fails to build or run, or an ABL_H arm whose
ablation did not apply, is VOID. A seed with a VOID arm is UNDEFINED, never a pass and never a fail.

**Reported, never scored:**
- `C4_parse_posthoc`: the parse-only teach check. It was written after the engram C4 failure had been seen.
- Integrity smokes (pass by construction): no retraction in any arm, and the start-of-turn re-read ran once per turn
  in every Hebbian arm.

Scorer selftest: `.venv/bin/python -m research.runners.d6_learn_through_use_lb --selftest` must print SELFTEST PASS.
Every v2 failing direction is covered: ablation not removing the recall, a framing-only leak in ABL_H, the ablation
lever not moving, a store write after teach in a lesion arm, a lever that is 0 at teach but non-zero at probe, and a
missing ABL_H arm scoring UNDEFINED.

## Predictions, written down before the run

- **C4 is predicted to FAIL on the teach acknowledgement.** In the engram variant at s42 the frozen arm acknowledged
  "The wolf hunts deer." against "the wolf hunts the deer"
  (`research/findings/raw/_d6_learn_through_use_engram/s42_FREEZE_H.json`), because
  the ack is rendered after the write from the engram-derived known sets. The read-time view keeps that path.
  - If this happens, the v2 verdict is NO-GO on C4. It will not be re-scored under the parse-only check.
  - That outcome would mean the frozen write changes the SAME turn's reply as well as the later one. It is the
    next thing to explain, not a pass.
- **The load-bearing question is C3 and C3b.** They are predicted to pass if `_known_concepts` was the last kb-direct
  reader on this protocol's path; the engram s42 FREEZE_H probe diff shows only the thread-swap lead and the DA-mode
  suffix remaining (`research/findings/raw/_d6_learn_through_use_engram/s42_FREEZE_H.json` vs
  `research/findings/raw/_d6_learn_through_use_engram/s42_SHUF_H.json`).
  - If C3 passes and C3b fails, some reader still takes familiarity from a record that outlives the synapses. It
    must be named and moved onto the read-time view.
  - If both fail, the read-time view does not cover the reply path. The next method is to route the swap and
    common-ground topic through the substrate's own familiarity read, not a host list.

## The run and the scoring command

Pool, one queue line per seed, each variant in its own arm directory:
`--variant readtime --seeds <S> --arm-dir research/findings/raw/_d6_learn_through_use_readtime`.
Scoring, after the per-arm files from every node are pulled into one directory:

```
.venv/bin/python -m research.runners.d6_learn_through_use_lb --score-only --variant readtime \
    --arm-dir research/findings/raw/_d6_learn_through_use_readtime --seeds 42 43 44 100 101 102 \
    --json <ARM_DIR>/d6_ltu_readtime_6seed_verdict.json      # ARM_DIR = the --arm-dir above
```

## Declared host shortcuts (not closed by this gate)

These are listed in `research/runners/d6_hebbian_store.py`, "DECLARED HOST SHORTCUTS", each with its next method.
- (a) The one-to-one acc->readout instructive pathway is host-wired for each write.
- (b) The phase-lock before the context cell fires is a host loop.
- (c) The W_MAX clamp means only the phase is learned, not a graded strength.
- (d) The banked prune retraction.
- (e) The held/not-held decision is a host threshold on a neural read.
- (f) The DA-gain side effect is unmeasured.
- The composer's FHRR bind/bundle, the trigger-slot assignment and the rule's evaluation in runner code (not a `sim/`
  kernel) also remain.
- Nothing is flipped on by default.

## ADDENDUM A4 — filed ~17:30 UTC, before any `readtime` arm existed

**Seen at this time:** the prune variant's full s42 smoke, all 5 arms. Re-scored under the original C4 it is NO-GO on
C3 and C4 (`research/findings/raw/_d6_learn_through_use_prune/d6_ltu_prune_s42_rescored_original_C4.json`). The
frozen encode WAS retracted, and the thread-swap lead ("Setting the held thread aside — On wolf") is gone from the
FREEZE_H probe. The probe still differs from SHUF_H in one place: SHUF_H ends with the DA-mode suffix
" — worth going further here.", and FREEZE_H does not.
- DA mode reads `neutral` in FREEZE_H and `focus` in SHUF_H.
- The spiking novelty organ's per-word freshness for `wolf` is 0.84 in FREEZE_H, which heard "wolf" at the teach
  turn, and 1.0 in SHUF_H, which never heard it
  (`research/findings/raw/_d6_learn_through_use_prune/s42_FREEZE_H.json` vs
  `research/findings/raw/_d6_learn_through_use_prune/s42_SHUF_H.json`).

**What this means for the design.** SHUF_H does not match WORD EXPOSURE. C3 and C3b therefore mix up two use-traces:
the fact-write engram, and habituation to heard words. The write freeze does not touch habituation, and should not.
**Updated prediction for the registered gate v2 (unchanged):** C3 and C3b are expected to FAIL on this DA-mode
suffix, as is C4, on the teach ack.

**Added (secondary, non-scoring, cannot confer GO):** arm EXPO_H. It has USE_H's flags, and its teach turn is
"the wolf and the deer": the same content words, but not an SVO assertion, so no acquisition and no write. It is
checked by two secondary tests:
- C3e: FREEZE_H.probe equals EXPO_H.probe, and FREEZE_H does not recall deer;
- C3be: ABL_H.probe equals EXPO_H.probe, and ABL_H does not recall deer.

If C3e and C3be hold while C3 and C3b fail, the fact-write engram carries the fact-specific change, and word
exposure carries the rest. The next registered gate (v3) would then use the exposure-matched control as the primary
comparison, stated in advance.
