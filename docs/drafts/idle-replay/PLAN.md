# Idle replay: the idle turn voices a fact from the brain's own store (Stage-2 C5 build)

*Claude, 2026-10-06, branch `claude-review/s2-c5-build`. Owner 2026-10-06: "Can we build C4 and C5 before November?"
This is the C5 half. Phase: prove who owns the computation. Answers NEEDS-CLAUDE #15 (C5 private store). Hermes steps:
`HANDOFF.md` in this folder.*

## 1. The defect

Stage-2 exit criterion C5 `inner` (prereg `2026-09-30-stage2-exit-gate-PREREGISTRATION.md` section 4.5, amended by
`docs/drafts/stage2-exit/AMENDMENT-A3.md` on `research/stage2-exit-gate`) asks the idle turn (`say something`) to
voice a fact the brain HOLDS, and, after A3, one SELECTED by its state: the fact voiced after the permuted exposure must
differ from the one voiced after the intact exposure. Dev seed 7 voices "bird see seed", which comes from the
self-initiation organ's own four-fact demo lexicon (`_self_initiated_utterance_derisk.AGENTS/VERBS/PATIENTS`), never
from the brain's store, so `private_voices_stored_fact` is already False. A fixed fact fails A3 by design.

## 2. What the record already has (verify-first, 2026-10-06)

`git branch -a`, `git log --all --grep` (self-init, idle, inner, spontaneous, awake replay, DMN, wander, private, C5),
`ls research/findings`, `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`, `bash tools/before_you_build.sh "idle turn voices
a self-selected stored fact"` and the RAG (workstation). Found, read, and what is reused:

| record | what it is | used here |
|---|---|---|
| self-initiated utterance (findings 2026-08-13 x3, 2026-08-18 wired GO; ledger `self-initiated-utterance`, default ON) | idle-turn trigger + remark frame; the CA3 multibasin wander (cupy) or, on numpy, the curiosity-top of its OWN lexicon | trigger, frame, lesion flag and fallback line reused unchanged; its selection is what this replaces on the idle turn |
| DMN per-basin equalization (2026-08-17/18 GO) | all four of the organ's basins ignite (cupy write path) | principle only (every encoded engram must be able to win) |
| continuous wander + mull (2026-08-20/26, default ON) | between-turn wander, biased toward the just-discussed topic | not touched; it still draws from the demo lexicon (follow-up, section 8) |
| awake replay capture / completion (2026-09-25 NO-GO arc, 2026-09-29 arcc GO 6/6, default OFF) | quiet-rest SWR reactivation keeps a day-old trace capturable | a different job (consolidation of a trace, not voicing a fact); its spiking item competition is the same idea |
| spiking commit A1 (`docs/drafts/spiking-commit/PLAN.md`, on main) | exclusive commit gate + bistable latch, read by persistence | the stage itself; this is its second user, in `research/runners/spiking_commit.py` (PLAN section 3 "Reuse") |
| DA-gated encoding (ledger `da-gated-encoding`, default ON) | dopamine at store time scales an engram's write strength | the salience term, read off the composer's own engram |
| D5 episodic, generative wander, sleep replay | consolidation and novelty rungs | not used |

No existing build selects the idle fact from the brain's own store. The CA3 multibasin wander is not reused directly:
its basins are bound to its own lexicon, it runs only on cupy (minutes on numpy; the gate is numpy), and its numpy path
is a host curiosity ranking. This organ keeps its principle (noise-driven competition among bistable assemblies under
a bias) and its lesion contract, on the brain's store, at a numpy-sized circuit (8 facts: 504 neurons, well under a second per idle turn).

## 3. The mechanism (full text: `research/runners/idle_replay_organ.py` docstring)

- One engram assembly per stored fact (16 excitatory neurons, recurrent), lateral inhibition between engrams.
- Each engram's excitability tag `x = BASE + s * exp(-age / TAU_TURNS)`: `age` = turns since the fact was (re)encoded
  (recency, CREB-like allocation excitability: Yiu et al. 2014; Cai et al. 2016), `s` = its encoding strength read off
  the composer's engram (the DA-gated write, `_measure_block_readout / _homeo_setpoint`; salience: Singer & Frank 2009,
  McNamara et al. 2014). The tag is a constant bias current during the rest window.
- Rest: independent Ornstein-Uhlenbeck background current to every engram neuron, the same for every engram. Whichever
  engram ignites first wins (awake rest reactivation favouring recent and salient experience: Karlsson & Frank 2009;
  Kudrimoti et al. 1999).
- The commit stage (`spiking_commit.py`): engram -> exclusive gate -> bistable latch; an occupancy pool driven by every
  latch inhibits all gates and engrams once one memory is held. Read: exactly one latch ON after a zero-input hold,
  re-read at x0.5 and x2 (threshold invariance).
- The voice: the winning block is decoded on the composer's own bridge (`OneBrainComposer._read_one_block`); the
  utterance is the decoded triple in the existing frame "Something's been on my mind -- a v p." A missing role abstains.
- Lesions: `BRAIN_SELF_INITIATE_LESION` (the gate's `lesion_selfinit` arm) = no-encode: no engram recurrence and no tag,
  nothing ignites, the neutral idle line. `BRAIN_IDLE_REPLAY_TAG_LESION` = equal excitability (noise decides).
  `BRAIN_IDLE_REPLAY_COMMIT_LESION` = engram -> gate closed (nothing commits).
- Server: `webapp/server.py` `brain_reply`: a flag-gated `note_turn` on every turn (bookkeeping: turn clock + which facts
  entered the store), and the idle block calls the organ instead of the demo-lexicon organ when the flag is on; the
  session reset drops it. Flag `BRAIN_IDLE_REPLAY`, default OFF (`_IDLE_REPLAY_DEFAULT_ON = False`): off, only the flag
  is read and every reply is byte-identical (pinned statically by `tests/test_idle_replay_organ.py`; at runtime by the
  HANDOFF's seed-7 flag on/off pair).

## 4. Dev bar (organ level) and its sanity

`research/runners/idle_replay_battery.py`, dev seeds 1-8 x 10 background-noise draws, on the Stage-2 C5 store shape
(5 build facts at turn 0, a 3-fact chain at turns 1-3, the idle turn at 4). PASS iff on every dev seed: the newest fact
wins >= 7/10 draws after each exposure and the two exposures differ >= 7/10; no-encode and commit lesions latch nothing;
under the tag lesion the newest wins <= 5/10 and >= 4 fewer than intact; every read is threshold-invariant.

bar-sanity: docs/drafts/idle-replay/bar_sanity_record_c5_organ.json

(n 1000 per brain, seed 0: perfect 0.995, perfect_noisy 0.974; null fixed fact, host FIFO, lesion-aware host FIFO,
graded no-latch and random 0.000. Reported: a host FIFO that also randomizes under the tag lesion passes 0.998; a
behavioural bar cannot tell a host that mimics every lesion from the circuit, so ownership stays the host-share
instrument's question, prereg section 5.)

Result (workstation numpy, `docs/drafts/idle-replay/dev_organ_battery.json`): every REPLAY_GRID row passes 8/8; the
frozen rule selects row 0. Row 0: newest wins 78/80 intact and 78/80 permuted (the two misses, dev seed 2, are the
second-newest fact winning the race), the exposures differ 80/80, no-encode and commit lesions latch nothing in 24/24,
tag lesion newest 11/80 (chance 10/80), first ignition median 129 ms. Near tie (the second-newest fact encoded at
strength 2 against the newest at 1): 48/32 of 80, so the race, not a sort, resolves it.

## 5. Reply level (Hermes, pool; the gate's own scorer)

At a revision containing both this organ and the Stage-2 runner (research/stage2-exit-gate synced with a main that has
this branch), the runner's `inner` group with `BRAIN_IDLE_REPLAY=1` on dev seeds 1-8: intact_a, permuted and
lesion_selfinit (teach turns + the idle turn), plus intact_a flag-off per seed and the full seed-7 intact_a flag
on/off pair. `--score-reply` scores them with `stage2_exit_gate.score_inner` and `private_selected_by_state` (A3) and
reports `tools/answer_rate.report` on the idle turn, on vs off. Pass: per seed both exposures voice a stored fact, they
differ, the lesion gives the neutral idle line, the trace route is `idle_replay`; seed 7 full: only `p_private`
differs between on and off.

Workstation smoke, dev seed 7 (2026-10-06, `dev_s7_reply_smoke.json`): intact voices "beetle chase mite", permuted
"vole chase beetle", lesion_selfinit the neutral idle line; the gate's scorer reads `private_voices_stored_fact`
True/True and `private_selected_by_state` True. Idle turn 1.0-1.2 s, peak RSS 6.2 GB. One seed, so a smoke, not a
result; seeds 1-8 are HANDOFF section 2.

## 6. Path to default-on before 11-03

1. Hermes runs the reply-level dev lines (HANDOFF section 3) and the regression battery
   (`onebrain_regression_battery.py --flag BRAIN_IDLE_REPLAY`, HANDOFF section 4).
2. Flip dev check (RULING flip-hurts-answers): `--score-reply` prints the idle-turn answer-rate report; a
   `FLIP-HURTS-ANSWERS` flag is a NEEDS line. Expected: none (off answers with the demo-lexicon remark, on answers with
   a stored fact). If it fires because the race latched nothing on a seed, that is a defect to rule on, not to tune.
3. Claude reads the results and writes the flip: `_IDLE_REPLAY_DEFAULT_ON = True`, a `default_anchor` sub-rung
   (`research/runners/idle_replay_organ.py` `_IDLE_REPLAY_DEFAULT_ON`) plus a sub-rung note on the ledger row
   `self-initiated-utterance` (deliberately not added while default OFF: the ledger changes with the status, together
   with the board sync), and the summary docs.
4. The gate picks it up through its own HANDOFF step 3 (merge main into research/stage2-exit-gate, re-pin GATE_SHA;
   "a merge that changes a production default the gate reads" is read by Claude at the next inspection). No prereg
   change: the idle turn, its frame, the lesion flag and the scorer are unchanged; only the selection behind it moves.
   The prereg's section 4.5 host-audit sentence about the numpy curiosity-top ranking becomes stale and the result
   finding must say so.

What the flip retires (the pivot rule): the idle turn's demo-lexicon content and the numpy light path's host
curiosity-top ranking of it. The between-turn wander (continuous engine) still uses the demo lexicon (section 8).

## 7. Declared residuals and what this does not show

Host: the idle trigger and the remark frame; the candidate list transcribed from the composer's fact index; the turn
clock and the per-turn advance of the excitability trace (a compression of an hours-long window onto turns); the
relay of the encoding strength into a bias current (the organ is its own bridge: one-brain debt); the K_MAX pre-filter
(never reached at Stage-2 sizes). Wired, not learned: every weight and the tag's time constant. Not shown: that the
trigger is autonomous (it is HTTP), that the reply is inner speech in any experiential sense (it is a functional
read-out of which engram reactivated), or that a host could not mimic it (section 4's info brain).

## 8. Follow-ups (named, not built)

- The continuous engine's between-turn wander (`recent_wander`, "(I'd been mulling over X.)") still draws from the
  demo lexicon; point it at this organ (same store, same race) in a separate packet.
- Recall re-tagging: a fact recalled on a turn should regain excitability (reconsolidation); the D5/mull recall hook
  is the place. Not needed by C5.
- One-brain debt: move the engram assemblies onto the composer's bridge.
