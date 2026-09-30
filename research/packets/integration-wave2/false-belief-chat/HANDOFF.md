# HANDOFF packet: false-belief-chat (integration wave 2)

Run the shared flow in `../HANDOFF.md`; this file holds what is specific to this packet. Prereg:
`research/findings/2026-09-30-iw2-false-belief-chat-flip-PREREGISTRATION.md`.

**What flips.** The new anchor `_FALSE_BELIEF_CHAT_DEFAULT_ON` in `webapp/false_belief_chat.py` turns on
`BRAIN_FALSE_BELIEF_CHAT`: a narrated Sally-Anne scenario (PLACE / LEAVE / RETURN / MOVE sentences, in one message or
across turns) is folded onto the W3 agent-keyed, witnessing-gated false-belief register, and "where will <agent> look
for the <object>?" is answered off its belief store. Recorded as a sub-rung of ledger row `w5-affective-tom`. Escape
after the flip: `BRAIN_FALSE_BELIEF_CHAT=0`.

**Why it was chosen.** Rank 1 of the wave (shared prereg): the reply names the location the register believes, so the
lesion changes the reply by construction; the wire, its LBF row and its byte-identity-off check already existed.

## Step R for this packet (shared HANDOFF sections 3-5)

Job file `jobs/R_fb_<P7>.txt` (12 lines), tag `iw2-R-fb-<P7>`: row `tom-false-belief` (the story with Sally leaving;
lesion `BRAIN_FALSE_BELIEF_LESION=1`) and control `tom-true-belief-ctl` (the same story, Sally never leaves). Score:
`tools/iw2_score.py R --packet false-belief-chat ...`.

| R verdict | per-seed detail (`R_fb.json`) | do |
|---|---|---|
| PASS | capability 6/6 `regressed` (intact names basket), control 6/6 `control-ok` | into the flip list (if D0 and D_false-belief-chat read PASS / DIFFERS-IN-SCOPE) |
| FAIL | a capability seed `pass` or `intact-wrong`, or a control `control-violated` | not flipped. Draft `research/findings/<date>-iw2-false-belief-chat-reply-level-NO-GO-<n>of6.md` (provisional): per seed, the intact and lesion answers of both rows (`detail.intact_answer`, `detail.lesion_answer`), and which rule failed |
| UNDEFINED | a cell not DEFINED (`state` not DEFINED, or `UNDEFINED:*`) | shared re-run rule once; still UNDEFINED: not flipped, NEEDS-CLAUDE "iw2 fb R UNDEFINED: <seed>: <fails>" |

## Production default (shared steps B and F)

F1: in `iw2-flip-<F7>` the row `tom-false-belief` must read reply-level load-bearing on 6/6 (step R's guards).

## Finding template (after the shared section 7 reads PASS)

`research/findings/<date>-iw2-false-belief-chat-default-on-flip-validated-PROVISIONAL.md`, frontmatter:
`type: finding`, `status: provisional`, `date`, `integration_faculty: w5-affective-tom`, `lane: integration`,
`seeds: [42, 43, 44, 100, 101, 102]`, `prereg:` both preregs, `artifacts:` the verdict JSONs. Sections, numbers only
from the verdict JSONs: (1) Result: R (6/6 capability, 6/6 control, the per-seed answers), D0/D/S, B (regressions:
none or the attribution), F1, F2, F3. (2) What changed for the user: one before/after pair from the identity arms
(`default@P` vs `default@F`, turn 4). (3) Declared residuals, copied from the prereg. (4) Deviations: none, or each
re-run from `reruns.tsv`. (5) Honesty: a functional read-out of a belief store; no claim of understanding or
experience. Never write "integrated", "closed" or "works" (docs/TERMS.md): write "on by default, reply-level
load-bearing 6/6".

## Dev seed 7 (Claude, 2026-09-30)

See the shared HANDOFF section 9.

## NEEDS CLAUDE (packet-specific)

1. Ledger placement (shared NEEDS-CLAUDE 2).
