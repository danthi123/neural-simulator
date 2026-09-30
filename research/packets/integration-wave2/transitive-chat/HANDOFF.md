# HANDOFF packet: transitive-chat (integration wave 2)

Run the shared flow in `../HANDOFF.md`; this file holds what is specific to this packet. Prereg:
`research/findings/2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md`.

**What flips.** The new anchor `_TRANSITIVE_CHAT_DEFAULT_ON` in `webapp/reasoning_transitive_chat.py` turns on
`BRAIN_TRANSITIVE_CHAT`: a yes/no question of the shape "does <A> <relation> <B>?" (three bare words) about a pair that was never told
directly is answered by the keystone re-entrant chase (`webapp.gnw_multistep_deliberation.multistep_chase`, the same
chase ledger row `gnw-multistep-deliberation` runs by default), whose hop count is the GNW workspace's own ignition
read; the answer states the derivation, and a chase that never reaches B abstains. Recorded as a sub-rung of
`gnw-multistep-deliberation`. Escape after the flip: `BRAIN_TRANSITIVE_CHAT=0`.

**Why it was chosen.** Rank 2 of the wave (shared prereg), the only candidate that bears directly on a Stage 2 exit
criterion (inference on held-out material: transitive inference is named). It has never been run through the handler
at six seeds; step R is its first reply-level test and can fail honestly.

## Step R for this packet (shared HANDOFF sections 3-5)

Job file `jobs/R_tr_<P7>.txt` (12 lines), tag `iw2-R-tr-<P7>`: row `reasoning-transitive-chat` (teach cat -> ball ->
worm -> river with 'precede', ask "does cat precede river?"; lesion `BRAIN_TRANSITIVE_LESION=1`, the keystone's own cut) and
control `transitive-reversed-ctl` (the chain taught in reverse; the same question must abstain). Score:
`tools/iw2_score.py R --packet transitive-chat ...`.

| R verdict | per-seed detail (`R_tr.json`) | do |
|---|---|---|
| PASS | capability 6/6 `regressed` (the intact reply names river, says "reasoned" and does not abstain), control 6/6 `control-ok` | into the flip list (if D0 and D_transitive-chat read PASS / DIFFERS-IN-SCOPE) |
| FAIL | a capability seed `pass` (the lesion did not change the reply) or `intact-wrong`, or a control `control-violated` (a "yes" to a reversed chain) | not flipped. Draft `research/findings/<date>-iw2-transitive-chat-reply-level-NO-GO-<n>of6.md` (provisional): per seed, the intact and lesion answers of both rows, and which rule failed |
| UNDEFINED | a cell not DEFINED; `UNDEFINED:degenerate-intact` means the intact chase ABSTAINED on the forward chain while the lesion said something else | shared re-run rule once; still UNDEFINED: not flipped, NEEDS-CLAUDE "iw2 tr R UNDEFINED: <seed>: <fails>" |

A seed whose intact reply abstains AND whose lesion also abstains reads `pass`, so R is FAIL (the capability is absent
on that seed): record it in the NO-GO draft as "the chase did not reach river at seed <s>" (or, if the arm files show a teach turn answered with a
clarification question, "the chain was not stored at seed <s>").

## Production default (shared steps B and F)

F1: in `iw2-flip-<F7>` the row `reasoning-transitive-chat` must read reply-level load-bearing on 6/6 (step R's guards).

## Finding template (after the shared section 7 reads PASS)

`research/findings/<date>-iw2-transitive-chat-default-on-flip-validated-PROVISIONAL.md`, frontmatter as in the
false-belief-chat template with `integration_faculty: gnw-multistep-deliberation`. Sections: (1) Result: R (6/6
capability, 6/6 control, per-seed answers and `derived_from` hops from the arm files), D0/D/S, B, F1, F2, F3. (2) The
reply before/after (identity turn 9, `default@P` vs `default@F`). (3) Per-turn cost of the chase turn
(`turn_wall_s`, identity arms) -- reported, not gated. (4) Declared residuals from the prereg. (5) Deviations.
(6) Honesty: the derivation is the substrate's hop sequence; the question-shape regex, the direct-fact check and the
yes/no template are host. Never write "reasons" or "works" (docs/TERMS.md); write "on by default, reply-level
load-bearing 6/6".

## Dev seed 7 (Claude, 2026-09-30)

See the shared HANDOFF section 9.

## NEEDS CLAUDE (packet-specific)

1. The question-shape regex matches any "does <w1> <w2> <w3>?" (three single tokens). Default-on, an ordinary yes/no
   question of that shape whose first token is not a determiner is taken by the route (the scope turn "does the cat
   sleep?" is excluded by the guard described in the prereg). The answer is then the route's direct-fact check or an
   honest abstain, never the ordinary path's answer to a different question. Review at merge whether that scope is
   acceptable for production.
