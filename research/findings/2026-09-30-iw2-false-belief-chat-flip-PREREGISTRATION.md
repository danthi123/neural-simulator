---
type: preregistration
status: preregistered
date: 2026-09-30
lane: integration (wave 2, packet false-belief-chat; ledger row w5-affective-tom, recorded as a sub-rung)
mechanism: theory of mind, false belief in live chat -- a narrated change-of-location scenario is folded onto the W3
  agent-keyed, witnessing-gated false-belief register (research/runners/tom_false_belief_chat_organ.py, reuse-by-import
  of research/runners/_false_belief_register_derisk.py) and "where will <agent> look for the <object>?" is answered off
  its belief store (webapp/false_belief_chat.py, BRAIN_FALSE_BELIEF_CHAT, anchor `_FALSE_BELIEF_CHAT_DEFAULT_ON`, new
  in this wave); measured at reply level for the first time at six seeds (LBF row tom-false-belief)
seeds: [42, 43, 44, 100, 101, 102]
backend: numpy (CPU pool)
verdict: PRE-REGISTRATION only. Filed before any gate-seed job exists. No result is claimed here.
---

# Packet false-belief-chat: flip the false-belief chat wire default-on (pre-registration)

Shared steps (selection and ranking, D, S, B, F, decision table, UNDEFINED rule):
`2026-09-30-iw2-wave-battery-PREREGISTRATION.md`.

## Why this faculty

- **Evidence.** The W3 register is a six-seed GO with every anti-cheat (false belief, reality baseline fails, true
  belief updates, other-lesion collapses, scrambled witnessing collapses, self/other dissociation;
  `2026-07-24-W3-false-belief-register-6seed-GO-adversarially-verified-immunity-claim-corrected.md`). The chat wire
  (`2026-09-24-tom-false-belief-chat-wire-PREREGISTRATION.md`) passed its seed-7 organ smoke and its byte-identity-off
  check twice, the second time as a clean git_archive pool run at the post-review commit, whose code main runs today
  (`2026-09-25-tom-false-belief-chat-wire-seed7-smoke-GO.md`). Its six-seed reply-level gate was deferred; this is it.
- **Reply impact.** The organ's belief read decides the answer's content: "Sally will look in the basket for the
  marble." (intact) vs "... in the box ..." (witnessing gate forced open). The flag-off reply to the same story is the
  default path's abstain-plus-curiosity text (dev seed 7).
- **Stage.** False belief is a Stage 3 exit criterion (master roadmap, STAGE 3); inference about another agent's
  belief is the kind of held-out inference Stage 2 asks for. Ranked first by (a) and (b) in the shared prereg.
- **Checked.** Not retracted; not on by default (no anchor before this wave; `false_belief_chat_enabled()` reads the
  flag unset -> False). The ledger has no cognitive-ToM row; this flip is recorded as a sub-rung of the ToM row
  `w5-affective-tom` (NEEDS-CLAUDE: whether to split a row at merge).

## Step R: reply level, capability + permuted control, at P

Row `tom-false-belief` (`research/runners/lbf_rows/tom_false_belief_chat.py`, compared fields now the primary reply
fields `answer`, `abstained`, `recalled_svo`): one fresh-session turn narrating PLACE (Sally, basket), LEAVE (Sally),
MOVE (Anne, box), QUERY ("Where will Sally look for the marble?"). Intact = `BRAIN_FALSE_BELIEF_CHAT=1`; lesion adds
`BRAIN_FALSE_BELIEF_LESION=1` (the witnessing gate forced open at write and query).

Control `tom-true-belief-ctl` (`research/runners/lbf_rows/iw2_controls.py`): the same story WITHOUT the LEAVE sentence
(Sally sees the move). Tag `iw2-R-fb-<P7>`, six gate seeds, `--repeats 2`, `LB_IW2_CONTROLS=1`.

- **Capability cell counts** iff DEFINED, S2 `regressed`, the intact reply is not degenerate and names `basket` and
  not `box` (`tools/iw2_score.py` PACKETS["false-belief-chat"]).
- **Control cell holds** iff DEFINED, the intact reply names `box` and not `basket`, and the lesion leaves it (S2
  `pass`). A control violation means the answer does not come from tracked witnessing.
- **R PASS:** capability 6/6 and control 6/6. **R FAIL:** any DEFINED miss (a capability seed not regressed or
  `intact-wrong`, a violated control). **UNDEFINED:** any cell not DEFINED after one re-run.
- **Prediction:** PASS. The W3 register read false-belief accuracy 1.0 on all six seeds (chance 0.25) and the reply
  is a template over it. The risk is the seed-dependent operating point of the chat organ's own bridge (it reuses the derisk's DRIVE_STEPS 50 / helper 5000
  fix); a seed that fails to ignite would read `intact-wrong` or degenerate and is a real finding either way.

## Packet-specific production-default criteria (in addition to the shared B1, B2, F2, F3)

- **F1:** in `iw2-flip-<F7>` the row `tom-false-belief` (REQUIRED_ENV lifted by the flip) is reply-level load-bearing
  on 6/6 seeds with no BRAIN_* flag set, read with step R's guards.

## Declared residuals (unchanged from the wire's prereg)

Witnessing and presence ("X leaves the room", "X returns") are a host parse of the narration; the belief-location
read is a host argmax over the store's late-window rates; the reply is a host template. One scenario per session;
a PLACE naming a new agent starts a new scenario. The open-ended and rich paths are not wired. Default-on means any
narration that matches the PLACE/MOVE/LEAVE/RETURN grammar updates the per-session scenario silently (the reply to a
narration turn is unchanged: a write-only fold) and a matching "where will X look for the Y?" is answered by the
register.

## Amendment 1 (2026-10-01): lesion-content check

Step R's capability-cell rule (and F1) gains the shared prereg's Amendment 1, written before any step-R cell was
read: the lesion reply must not state the false belief. A lesion reply that is not degenerate, names `basket` and
does not name `box` makes the cell DEFINED `lesion-kept`, an R FAIL. The prediction (lesion names `box`) is unchanged.
