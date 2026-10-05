---
type: finding
status: live
date: 2026-10-05
integration_faculty: w5-affective-tom
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
prereg:
  - research/findings/2026-09-30-iw2-false-belief-chat-flip-PREREGISTRATION.md
  - research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_iw2/verdicts/573a943/D0.json
  - research/findings/raw/_iw2/verdicts/573a943/D_false-belief-chat.json
  - research/findings/raw/_iw2/verdicts/573a943/R_fb.json
  - research/findings/raw/_iw2/verdicts/1fb805e/B.json
  - research/findings/raw/_iw2/verdicts/1fb805e/A.json
  - research/findings/raw/_iw2/verdicts/1fb805e/F2.json
  - research/findings/raw/_iw2/verdicts/1fb805e/F3.json
  - research/findings/raw/_iw2/identity/573a943/default.json
  - research/findings/raw/_iw2/identity/1fb805e/default.json
  - research/findings/raw/_load_bearing/_shards/iw2-flip-1fb805e/s42/tom-false-belief/intact_a_tom_fb.json
---

# iw2 false-belief-chat: default-on flip validated (PROVISIONAL, 2026-10-05)

P7 = 573a9438f075239069c8966dd62f1da4d981e158, F7 = 1fb805e9e0bedcce684a47388d95848daf1008a2. Six
seeds (42/43/44/100/101/102). Every number below is read from the cited verdict JSONs. Status: merged with caveats by cloud Claude session 7 (see the review section at the end).

## 1. Result

- **R (step R): PASS**, preconditions 2/2. Capability 6/6: intact "Sally will look in the basket for
  the marble." vs lesion "Sally will look in the box for the marble." — the pair is identical on all
  six seeds. Control `tom-true-belief-ctl` 6/6 (intact = lesion on every seed).
- **D0: IDENTICAL** (8 preconditions, gate-checked). **D_false-belief-chat: DIFFERS-IN-SCOPE**, turn 4
  the only differing turn; the step-S scope check is inside that verdict.
- **B: FAIL** with exactly one regression — `affect-marker-spiking-wta` at s2, 4 -> 0 load-bearing
  seeds; `undefined` is empty.
- **A: PASS**, preconditions 3/3 — the regression is attributed `excused`: the off-arm reproduces the
  same drop (off_run s0 = 4 / s2 = 0 against flip s0 = 4 / s2 = 0). Decision-table row 1: the flips
  stand.
- **F1: 6/6** — the false-belief-chat row is reply-level load-bearing on every seed at F (6 regressed
  cells, 6 new rows at s2).
- **F2: IDENTICAL** — arms `default` vs `on:false-belief-chat+transitive-chat+neural-selfid`, differing
  [] (the default now behaves like all three flags explicitly on).
- **F3: IDENTICAL** — arms `off` vs `default`, differing [] (the explicit-off escape still reproduces
  the pre-flip behaviour).

## 2. What changed for the user

Identity arm turn 4, "Where will Sally look for the marble?"
(`research/findings/raw/_iw2/identity/573a943/default.json` vs `research/findings/raw/_iw2/identity/1fb805e/default.json`):

- **default@P:** "I don't know about that. My curiosity is piqued — I haven't learned about sally yet:
  what can you tell me about sally? — worth going further here." (abstained, 24.41 s)
- **default@F:** "Sally will look in the basket for the marble." (answered, 5.33 s)

## 3. Declared residuals

Copied from `research/findings/2026-09-30-iw2-false-belief-chat-flip-PREREGISTRATION.md`:

> Witnessing and presence ("X leaves the room", "X returns") are a host parse of the narration; the
> belief-location read is a host argmax over the store's late-window rates; the reply is a host
> template. One scenario per session; a PLACE naming a new agent starts a new scenario. The open-ended
> and rich paths are not wired. Default-on means any narration that matches the PLACE/MOVE/LEAVE/RETURN
> grammar updates the per-session scenario silently (the reply to a narration turn is unchanged: a
> write-only fold) and a matching "where will X look for the Y?" is answered by the register.

## 4. Deviations

Four s101 flip-battery cells deviated once (`research/packets/integration-wave2/reruns.tsv`):
`episodic-memory` and `discourse-register` died rc=137 (OOM kill) on wspool; `self-initiated-utterance`
and `vision-identity-spiking-hmax` vanished with no exit record. Each attempt-1 output was moved aside
to `_attempt1_s101_<faculty>` and the cell re-run once (section 8 rule); the scorer ran once, at
section 7. No other deviation from the prereg.

## 5. Honesty

This is a functional read-out of a belief store: a narration fold and a register answer whose reply
changes when the substrate path is lesioned. It is not a claim of understanding or experience
(non-negotiable 5). The state at F: **on by default, reply-level load-bearing 6/6, with the load-bearing input being the host witnessing decision (session-7 review below)** (docs/TERMS.md).

## Host-ownership bound (RULING `iw2-verify-go`, required sentence)

The spiking belief store carries the belief that the reply reads. Who saw the move is decided by the
host witnessing parse, and that decision gates the write. The lesion forces the gate open.

Per RULING `iw2-stage1-credit`: counts toward Stage 1 as **PARTIAL (host decision: the witnessing
`present` parse)** until T18 passes O and R and T21 keeps the path default. It stays a sub-rung of
`w5-affective-tom`; no cognitive-ToM row while the witnessing decision is host-owned.

## Claude review (2026-10-05, cloud session 7): merged, NARROWED

verify-go: three independent skeptics (control-integrity, gate-cheat and instrument-trust lenses), plus my own
recomputation of the identity compares. None refuted the result; each narrowed what it shows.

What holds, re-derived rather than read from the verdict labels:

- **F2 and F3, recomputed from the identity files.** The compare hashes each turn's full response body (sha256 of the
  sorted-key JSON; the volatile-key filter dropped 0 keys on every turn of every arm), so IDENTICAL here is an exact
  compare in the data (docs/TERMS.md `byte-identical`). default@F equals on@F on all 16 turns, and off@F equals
  default@P on all 16 turns. default@F differs from default@P on turns 4, 9, 14 and 15 only, which is exactly the
  union of the three single-flag arms at P (4, 9 and 14-15). Dev seed 7, one seed: an identity check, not a
  generalization. Each arm's `cwd` names its revision directory (the `git_sha` field is null in the pool export).
- **The excused regression is legitimate under the prereg's own B1 rule.** affect-marker-spiking-wta reads 4
  load-bearing seeds at S2 in B2b (`b2b0924-base` at a308f1e09) and 0 at F. The all-off arm at F (all three iw2 flags
  `=0`, six cells) reads S0 4 and S2 0, the same as the flip, so no iw2 flag causes the drop. The drop predates F; it
  is not this wave's and is reported here so it is not lost (handed to Hermes as its own item).
- **The reply-level row holds on six seeds.** intact_a = intact_b ("Sally will look in the basket for the marble."),
  lesion "... box ...", and the lesion repeat reproduces, on every seed; `tom-true-belief-ctl` answers "box" in both
  arms on every seed. The scorer's degenerate-intact guard holds: the answered slot reads a positive belief rate
  (0.333) and every other slot 0.0, so the store did carry the belief the reply reads. <!--derived-->

What is narrowed:

- **The lesion does not cut a spiking path.** `BRAIN_FALSE_BELIEF_LESION` sets the `witness_other` gate to 1.0 at
  write and at query (`research/runners/tom_false_belief_chat_organ.py`, `_write_event` and `_query`), replacing the
  host's witnessed boolean, which the presence regexes in `webapp/false_belief_chat.py` decide. It overrides a host
  decision input; no arm silences the store. A host dictionary that keeps the first location unless the agent is
  present would give the same replies on the probe and on the control. So the row is reply-level load-bearing per
  docs/TERMS.md, and the load-bearing input is the host witnessing decision. Whether the spiking store is causally
  necessary is not tested. This earns no level-3 "spiking on-by-default" credit (docs/TERMS.md, `integrated`), and the
  Stage-1 credit stays PARTIAL (RULING `iw2-stage1-credit`).
- **The six seeds are one deterministic template.** One fixed story; the belief rates and the replies are identical on
  all six seeds. The seeds show the result does not depend on the bridge-build seed; they are not six independent
  samples of a behaviour.

Owed: a store lesion (the belief store's weights cut with the witnessing gate left intact; expected: abstain or a
chance answer). It belongs with T18, whose spiking witness population replaces the host decision this lesion
overrides. The same gate-versus-module question arises in modular networks: ablating a module and re-routing around it
are different interventions (MiCRo, https://arxiv.org/abs/2506.13331). <!--derived-->
