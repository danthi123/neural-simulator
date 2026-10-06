# s2-c4-build: the brain names the emotion channel its affect ladder holds (Stage-2 C4)

*Claude, 2026-10-06, branch `claude-review/s2-c4-build` (cut from main b6711066e). Owner 2026-10-06: "Can we build C4
and C5 before November?" This packet is C4. Gate: Stage-2 exit, window 2026-11-03..11-10, prereg
`research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` (PREREG_SHA 24dac499e) as amended by
`docs/drafts/stage2-exit/AMENDMENT-A3.md` (on `claude-review/stage2-c4c5` @ bd5c95bb4). Hermes steps: `HANDOFF.md`.*

## 1. The criterion and the gap

C4 `emo` after A3: after each of four situations (joy, fear, anger, sadness; each told twice) the reply to `how do you
feel` NAMES the situation's category (frozen `EMOTION_LEXICON`, exactly one category's words), four distinct, on 6/6;
the ladder lesion (`BRAIN_AFFECT_LESION=1`, the `affect_out` gate) changes the NAMED category on all four feel turns,
6/6, reproduced; the affect-word-swapped arm is tracked 6/6.

Today (dev seed 7, `_stage2_exit/dev/s7/emo__intact_a.json`): no feel reply names a category. The production affect
organ is a signed scalar ladder (valence; the arousal band is the host EMA of the words' norms): fear and anger read the
same, and in the DR-2 learned map the sadness words carry high arousal too (sad 5.75, cry 6.20), so no read-out of
today's state separates the three negative categories. A category needs more affect state, not a better read-out.

## 2. What the record already had (verify-first, 2026-10-06)

`git branch -a`, `git log --all --grep` (emotion / category / affect / ladder / feel / name), `ls research/findings`,
the ledger, `before_you_build.sh "name an emotion category from affect state"` and the RAG (workstation `.venv-rag`,
corpus all). No existing C4 build or branch. Reused, not rebuilt:

- `2026-08-13-spiking-appraisal-discrete-emotion-reappraisal-derisk.md` (6-seed GO, runner-only): appraisal dimension
  pools onto four category pools in a WTA. Its residual 2 is the gap here: the appraisal dimensions were set by the
  environment, and its winner is an argmax. This packet supplies the dimensions from heard words and decides by a commit.
- `2026-08-28/09-01 affect-marker-spiking-wta` (a lateral-inhibition WTA over a continuous mood; lesion and shuffle
  anti-cheats): the shuffle control is copied (`BRAIN_EMOTION_CATEGORY_SHUFFLE`).
- `docs/drafts/spiking-commit/PLAN.md` and `word_role_parser_organ.py` (COMMIT_GRID row 0): the commit stage (gate,
  exclusion, bistable latch, threshold-invariant read) with the same weights; the Hebbian word -> ensemble learning.
- the production valence ladder's staircase motif (staggered rungs) and the active-clear quench
  (`2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-...`).
- the W5 empathy lead (fires only on a named third party; kept out of these turns by the prereg).

## 3. Methods tried first and banked (no-defer: a method verdict, not a capability verdict)

`research/runners/_emotion_category_corpus_probe.py` -> `research/findings/raw/_s2_c4_build/corpus_probe.json`.

- **A. Corpus label propagation** (the DR-2 construction that sources the production valence map: PPMI over hubs,
  cosine kNN, seed-clamped propagation, leave-one-out), four category channels, 108 configurations. The selection
  rule written without reading the Stage-2 items (best macro LOO accuracy over non-Stage-2 seed words, 0.875) picks a
  configuration that would name 6 of the 8 registered situations (intact + permuted), min margin -0.12. 45 of 108
  configurations get all 8 right, best min margin 0.16, so using this route means choosing it on the gate's own items.
- **B. Trigger conditioning in text** (each affect word's co-occurrence with the innate-trigger words, per-channel
  base-rate gain): `happy` reads fear as much as joy and `angry` reads fear, in TinyStories (scares end happily);
  permuted margins down to -0.15.

Banked: TinyStories co-occurrence does not separate fear, anger and sadness words reliably. A child does not learn
these words from co-occurrence statistics either; it learns them from a caregiver labelling its own state.

## 4. The mechanism (`research/runners/emotion_category_organ.py`, default OFF)

One bridge, Izhikevich neurons, explicit wiring, ~920 neurons, numpy ~0.5 s per read.

- **Four affect channels** joy / fear / anger / sadness (Panksepp's SEEKING-PLAY, FEAR, RAGE, PANIC-GRIEF as separate
  populations: Namburi et al. 2015 Nature 520:675; Lin et al. 2011 Nature 470:221; Silva et al. 2013 Cell 155:1596).
  Appraisal ensembles A_k with lateral inhibition (a mixed message resolves to one channel).
- **Innate triggers** (wired, fixed): predator words -> FEAR, violation words -> RAGE, loss words -> PANIC-GRIEF,
  reward/play words -> SEEKING. Declared seed, the analogue of the Warriner norms.
- **Development** (learned, the engine's Hebbian co-firing rule): the teacher narrates labelling episodes, a trigger and
  an emotion word together ("monster ... scared"). The trigger fires its channel through the innate synapse; the emotion
  word's synapses onto that channel potentiate (evaluative conditioning, Staats & Staats 1957; emotion words learned by
  labelling the child's state, Widen & Russell 2008; Hoemann, Xu & Barrett 2019 Dev Psychol 55:1830). No teacher current
  enters the brain. Frozen after development. Learned weight taught vs untaught: 9.5-11.0 vs 0.05 (dev seeds).
- **The ladder** per channel: three rungs of 20 neurons, AMPA self-recurrence (bistable), driven at staggered weights
  (Koulakov et al. 2002; the production ladder's motif). Rungs hold after the message ends: the held state between
  turns, kept per session as the bridge's own state. A new appraisal quenches the other channels' rungs.
- **Readout** O_k through the gate named `affect_out` (the production ladder's readout gate name and semantics;
  `BRAIN_AFFECT_LESION=1` clamps it).
- **Commit** (spiking-commit PLAN, COMMIT_GRID row 0 weights): O_k -> gate G_k (exclusive) -> latch S_k. The feel query
  opens the O -> G window for 60 steps, a 100-step zero-input hold follows, and the category is WHICH latch is on,
  re-read at x0.5 and x2. One latch -> that category; none or two -> no category. No host comparison between channels.
- **Reply**: the production feel read-out, unchanged, plus one sentence: "The emotion channel that latched on my affect
  ladder is fear." or "No single emotion channel latched on my affect ladder." A functional read of which population
  latched; never a claim of felt experience (honesty boundary). The old marker word stays retired: this sentence exists
  only on an explicit feel query and names a state, it is not a tone marker on other replies.

**Switch.** `webapp/server.py`: `_EMOTION_CATEGORY_DEFAULT_ON = False`, `_emotion_category_on()` (env
`BRAIN_EMOTION_CATEGORY` overrides). Off: the organ is never imported and the feel response dict is the same object
contents as before (pinned by `tests/test_emotion_category_organ.py::test_server_uses_the_organ_only_under_the_flag`).
On: the affect block folds each message into the session's held channels; the feel branch appends the clause and
records `emotion_category` (latch counts, read variants, route, op, error). An organ error is recorded, never raised.

**Who owns the computation (prereg section 5 vocabulary).** Candidate scoring: learned synapses (spiking; the
curriculum and the innate map are host-written seeds). Hold: spiking. Commitment: spiking (commit latch, invariant
read). Composition: host (the fixed sentence). Persistence across turns: the bridge's own state, stored per session by
the host while the process waits (no host EMA on this path).

**Declared residuals.** The curriculum's emotion labels overlap the Stage-2 lexicon and situations by design: C4 tests
naming the held channel with words learned this way, not naming with never-heard words. The situations' frame words
(win, gift, wolf, attacking, someone, cheats, alone) are absent from development (pinned), so the frames drive nothing
and the permuted arm reads the emotion words. Word form -> assembly and the stop-word strip are host. The read window
is host timing. The organ is its own bridge (one-brain debt, like the valence ladder). The valence ladder and these
channels are two bridges that the same lesion flag cuts.

## 5. Dev results (organ level, workstation numpy, dev seeds 1-8; gate seeds never run)

`research/runners/emotion_category_dev.py`, ten cases per seed (intact; permuted; affect, word, latch and commit
lesions; shuffle build; teach-swap build; ladder-only drive; baseline). Selection rule: the first GRID row whose
battery passes every case on all 8 dev seeds. Row 0 was set after a dev-seed 1-4 smoke (LR 0.1 / W_AL 14 left the
two-word fear situation below the latch on 2 seeds); the order of rows 1-5 is frozen in the module.

- `research/findings/raw/_s2_c4_build/dev_select.json`: row 0 selected, 8/8 seeds pass all ten cases. Every named
  read: the latched slot count is 74 or more (read level 18 spikes; x2 = 36); every lesion read: 0 slot spikes.
- The ladder-only probe drives A_k at 300 pA, calibrated before the battery to the word-driven firing (at 1500 pA, the
  word assemblies' own drive, the ensemble saturates and every channel latches; recorded in the module).
- Bar-sanity of this dev bar: `docs/drafts/stage2-c4-build/bar_sanity_record_dev.json` (n 1000, seed 0): perfect
  1.000; host lexicon lookup 0.000; ladder-gated lookup 0.000; lesion-inert 0.000; no-hold 0.000; host argmax (no
  commit) 0.000; random 0.000; info one-flaky-case 0.710. The lookup brains fail on the shuffle, teach-swap and
  ladder-only cases, which the Stage-2 reply bar cannot see (A3 "what these clauses do NOT establish").

bar-sanity: docs/drafts/stage2-c4-build/bar_sanity_record_dev.json

Handler level, through the Stage-2 runner's own worker with the flag on: intact_a, lesion_affect and permuted at dev seed 7 (shards in `research/findings/raw/_s2_c4_build/handler_smoke/on/s7/`, runner bd5c95bb4's file over this branch, workstation numpy, memcap 10, 846 s per arm). Read with the Stage-2 scorer (`score_emo`, `named_category_changed`): intact names joy, fear, anger, sadness (`all` true); the lesion arm names none on all four (`category_changed` true on every turn); the permuted arm names sadness, anger, fear, joy (`all` true); 4/4 feel reads parse in every arm. Every non-feel reply is identical to the committed flag-off dev shard (`_stage2_exit/dev/s7/emo__intact_a.json`), every feel reply is that reply plus the one clause. One dev seed, one process each: no reproduction (intact_b) and no other seed; that is H1's job.

## 6. Path to the gate (dates are deadlines)

| step | who | what | by |
|---|---|---|---|
| A | Claude (orchestrator) | merge this branch to main | 10-07 |
| B | Hermes | sync `research/stage2-exit-gate` with main; organ battery re-run at the synced tip (CPU) | 10-09 |
| C | Hermes, pool | handler dev H1 (emo, 4 arms, dev seeds 1-8, flag on) + H2 (flip specificity) | queued 10-10, landed 10-14 |
| D | Hermes | score: Stage-2 aggregate on H1 (C4 at dev) + `--flip-check` on H2 | 10-15 |
| E | Claude | verify-go on D; flip `_EMOTION_CATEGORY_DEFAULT_ON = True`, ledger row, tests | 10-20 |
| F | Hermes | re-run the five dev-seed-7 `intact_a` arms at the flip tip (the gate's dev smoke) | 10-24 |
| G | Claude | pin GATE_SHA at a revision with the flip, A3, the runner and the tests | 10-31 |

FLIP-HURTS-ANSWERS: `emotion_category_dev --flip-check` classifies every turn of every on/off shard pair with
`tools/answer_rate.classify` and also requires every non-feel reply to be identical and every feel reply to be the
flag-off reply plus one clause. Any flag -> NEEDS, the flip waits (RULING flip-hurts-answers).

## 7. What this will not show

That the category generalises to emotion words never heard in development (none are tested). That a channel is felt.
That the channels are learned from scratch (the innate trigger map and the curriculum are written). That the
category is decided by the one shared substrate (separate bridge). The host-share instrument (prereg section 5,
NEEDS-CLAUDE #10) is where the scoring/commitment split above is measured, not this packet.
