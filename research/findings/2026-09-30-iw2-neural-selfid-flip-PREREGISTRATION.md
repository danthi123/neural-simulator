---
type: preregistration
status: preregistered
date: 2026-09-30
lane: integration (wave 2, packet neural-selfid; ledger row content-selection, recorded as a sub-rung)
mechanism: self/identity comprehension on the brain -- a question naming the brain itself ("what do you use?") is
  comprehended by the on-brain BridgeParser (ChatBrain._neural_question_parse, parser.role_of on Izhikevich neurons)
  and answered by the substrate recall, and a bare identity question by the substrate recall's candidate-relation
  retry, instead of the host QuestionRouter (research/runners/brain_chat_tui.py BRAIN_NEURAL_SELFID, the existing
  anchor `_NEURAL_SELFID_DEFAULT_ON`, flipped ALONE: BRAIN_NEURAL_ANAPHORA_ABSTAIN stays off); measured at reply level
  for the first time through the handler (LBF row neural-selfid)
seeds: [42, 43, 44, 100, 101, 102]
backend: numpy (CPU pool)
verdict: PRE-REGISTRATION only. Filed before any gate-seed job exists. No result is claimed here.
---

# Packet neural-selfid: flip self/identity comprehension on the brain default-on (pre-registration)

Shared steps (selection and ranking, D, S, B, F, decision table, UNDEFINED rule):
`2026-09-30-iw2-wave-battery-PREREGISTRATION.md`.

## Why this faculty

- **Evidence.** The rank-13 de-risk is a six-seed GO on both production combiners (the plain gate and the installed
  GNW bus): self-factual and bare-identity questions answer correctly with zero calls to the host router, and the
  parser lesion collapses the self-factual answer on 6/6 seeds while the recall reflex survives
  (`2026-09-05-rank13-selfid-anaphora-scaffold-derisk-GO-6of6.md`). The 2026-09-05 production flip of the PAIR
  (SELFID + ANAPHORA_ABSTAIN) was NO-GO twice; the corrected diagnosis attributes the surviving seed-44 regression to
  ANAPHORA_ABSTAIN (it removed the router's rescue of a wrong discourse referent) and fixed SELFID's own seed-43
  over-fire (`2026-09-05-rank13-selfid-anaphora-production-flip-CORRECTED-DIAGNOSIS.md`; the NO-GO finding's
  root-cause section is partially retracted in docs/RETRACTED.md, its verdict stands). This packet flips SELFID
  alone; ANAPHORA_ABSTAIN keeps its False anchor (pinned by a test).
- **Reply impact.** Intact (flag on): "what do you use?" is answered from the substrate recall of (brain, use,
  spikes); with the content-selection row's own parse lesion (`BRAIN_SUBSTRATE_PARSE_LESION=1`) the parse declines
  and the turn abstains. With the flag off the router answers regardless of the lesion: the flip is what makes this
  decision the brain's.
- **Stage and phase.** Weak on the Stage 1/2 exit criteria (the self as a topic of grounded conversation); strong on
  the current phase: it retires one of the two host-router classes the `content-selection` row still declares.
- **Checked.** Not retracted (the de-risk); not on by default (`_NEURAL_SELFID_DEFAULT_ON = False` at main f5e3732fe).

## Step R: reply level, capability + permuted control, at P

Row `neural-selfid` (`research/runners/lbf_rows/neural_selfid.py`, new): one fresh-session turn "what do you use?".
Intact = `BRAIN_NEURAL_SELFID=1`; lesion adds `BRAIN_SUBSTRATE_PARSE_LESION=1`. Compared fields: `answer`,
`abstained`, `recalled_svo`.

Control `selfid-nofact-ctl` (`research/runners/lbf_rows/iw2_controls.py`): "what do you drink?" -- no stored self
fact, so the substrate route must abstain (the no-confabulation moat; the 2026-09-05 seed-43 defect was a self-alias
over-fire that produced a self fact for the wrong question). Tag `iw2-R-si-<P7>`, six gate seeds, `--repeats 2`,
`LB_IW2_CONTROLS=1`.

- **Capability cell counts** iff DEFINED, S2 `regressed`, the intact reply is not degenerate and names `spikes`.
- **Control cell holds** iff DEFINED, the intact reply abstains, and the lesion leaves it (S2 `pass`).
- **R PASS:** capability 6/6 and control 6/6. **R FAIL:** any DEFINED miss. **UNDEFINED:** any cell not DEFINED after
  one re-run.
- **Prediction:** PASS. Through the handler at dev seed 7 (a scratch feasibility run before this prereg was
  committed, repeated as the recorded dev run in the HANDOFF): flag on, "what do you use?" -> "the brain uses the
  spikes", and with the parse lesion "I don't know about that."; flag off, the router answers the same text with or
  without the lesion; "what do you drink?" abstains in every arm.

## A known behaviour change, recorded before any gate data (not a pass/fail criterion; NEEDS-CLAUDE before merge)

At dev seed 7 the bare identity questions change with the flag: "what are you?" and "who are you?" read "the brain
uses the spikes" with the flag off (the router's keyword match: the question's only content word resolves to 'brain'
and the router returns the first brain fact) and "I don't know about that." with the flag on (the substrate route
finds no (brain, isa, X) fact and its candidate-relation retry does not answer). The rank-13 de-risk's class (b)
success was measured on its own fixture and does not reproduce through the handler here. The flip therefore trades a
keyword answer to a different question for an honest abstain on these two questions. Step D reports it
(`reply_differing_turns`); whether it is acceptable in production is Claude's call at merge (packet HANDOFF).

## Packet-specific production-default criteria (in addition to the shared B1, B2, F2, F3)

- **F1:** in `iw2-flip-<F7>` the row `neural-selfid` (REQUIRED_ENV lifted by the flip) is reply-level load-bearing on
  6/6 seeds with no BRAIN_* flag set, read with step R's guards.
- B1 is where the 2026-09-05 anaphora regression would reappear if SELFID carried any of it (the rows
  `content-selection`, `semantic-recall`, `moat-verify`, `spiking-anaphor`, `wm-binding-advanced`); the shared
  attribution and bisect rows apply unchanged.

## Declared residuals

Class (b), the bare identity question, is not the neural parser: it is the same host regex comprehension helper
'what is X?' already uses, with the recall itself on the substrate. The self-alias resolution ('you' -> 'brain')
mirrors the router's own `_resolve_self`. The noisy-anaphora fallback stays host-owned (ANAPHORA_ABSTAIN off).
Default-on, 'it' is also a self-alias: a bare "it" that the discourse WM does not substitute resolves to the brain
itself on the substrate path (the router did the same on the host path); step S plays one ordinary anaphora turn to
check that a held referent still wins.
