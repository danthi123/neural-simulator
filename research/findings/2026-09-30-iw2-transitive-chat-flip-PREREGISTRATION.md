---
type: preregistration
status: preregistered
date: 2026-09-30
lane: integration (wave 2, packet transitive-chat; ledger row gnw-multistep-deliberation, recorded as a sub-rung)
mechanism: held-out transitive inference in live chat -- a yes/no question about a never-told non-adjacent pair
  ("does cat precede river?") is answered by the keystone re-entrant chase the gnw-multistep-deliberation row already runs
  by default (webapp/gnw_multistep_deliberation.multistep_chase: the hop count is the GNW workspace's own n_ignited
  read), through webapp/reasoning_transitive_chat.py (BRAIN_TRANSITIVE_CHAT, anchor `_TRANSITIVE_CHAT_DEFAULT_ON`, new
  in this wave); measured at reply level for the first time (LBF row reasoning-transitive-chat)
seeds: [42, 43, 44, 100, 101, 102]
backend: numpy (CPU pool)
verdict: PRE-REGISTRATION only. Filed before any gate-seed job exists. No result is claimed here.
---

# Packet transitive-chat: flip the transitive yes/no chat route default-on (pre-registration)

Shared steps (selection and ranking, D, S, B, F, decision table, UNDEFINED rule):
`2026-09-30-iw2-wave-battery-PREREGISTRATION.md`.

## Why this faculty

- **Evidence.** The mechanism is the keystone chase, a six-seed GO with its anti-cheats
  (`2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md`, `2026-08-19-gnw-multistep-reentrant-
  deliberation-GO.md`) and on by default for "chase to the end" questions since 2026-08-19. The route
  (`2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md`) adds a new question shape on top of it; its logic is
  pinned by `tests/test_reasoning_transitive_chat_wirein.py`, but its real-handler seed-7 run was stopped before it
  finished, so this packet's dev seed 7 (shared HANDOFF section 9) is its first handler-level evidence and step R its
  first six-seed test.
- **Reply impact.** Intact: a derived answer that states its hops; lesion (the keystone's own cut, the workspace built
  with its assembly self-recurrence zeroed): the chase cannot sustain ignition past the first hop and the route
  abstains honestly ("I don't know about that.").
- **Stage.** Stage 2's exit names "inference on held-out" and lists transitive inference among the inferences that
  come online; this is the only screened candidate that bears on it directly (shared prereg, ranking).
- **Checked.** Not retracted; not on by default (no anchor before this wave; the reader reads unset -> False).

## A scope guard added before any gate data (a deviation from the 2026-09-24 route prereg)

The route's question regex matches any "does <w1> <w2> <w3>?" of three single tokens, and a matched question never
falls through to the ordinary path (by design: the ordinary parser would answer a different question). Default-on,
that would capture ordinary yes/no questions whose first token is a determiner or pronoun ("does the cat sleep?",
"does it rain now?"), answering them with the route's abstain instead of the ordinary reply. Before any gate job, the
route now declines (returns None, falls through, byte-identical to the flag-off path) when the parsed agent or target
is a closed-class word (determiners, pronouns, wh-words: `_TRANSITIVE_CLOSED_CLASS` in
webapp/reasoning_transitive_chat.py). Bare-noun questions ("does cat precede river?", "does dog chase cat?")
are unchanged. Pinned by `tests/test_integration_wave2_anchors.py`; step S tests it through the handler ("does the cat
sleep?" must read identical with the flag on and off).

## Step R: reply level, capability + permuted control, at P

Row `reasoning-transitive-chat` (`research/runners/lbf_rows/reasoning_transitive_chat.py`, no longer thin: its turns
are a literal EXTRA_TURNS list): one session teaches "the cat precedes the ball", "the ball precedes the worm", "the
worm precedes the river", then asks "does cat precede river?" (three hops, never told). Intact =
`BRAIN_TRANSITIVE_CHAT=1`; lesion adds `BRAIN_TRANSITIVE_LESION=1`. Compared fields: `answer`, `abstained`,
`recalled_svo`.

**A probe change made before any gate data (a deviation from the 2026-09-24 route prereg's World A).** Its tokens
e0..e4 are not words to the production acquisition path: through the handler at dev seed 7 every teach turn read "I
followed the shape of that, but I don't know the words 'e' or 'e' yet" (the digits are stripped) and nothing was
stored, so every probe abstained in every arm, flag on or off, intact or lesioned. The chain is now made of words the
tiny-demo brain has, alternating animate and inanimate nouns: a chain of animals ("the cat chases the bird") is held
back by the comprehension monitor's other-repair ("which of them is the 'chase' done to?") and never stored (dev seed
7). 'precede' keeps the relation genuinely transitive, as the route's prereg intended.

Control `transitive-reversed-ctl` (`research/runners/lbf_rows/iw2_controls.py`): the chain taught in reverse ("the
river precedes the worm", "the worm precedes the ball", "the ball precedes the cat"), the same question. Not derivable: a
route that said "yes" to the question's shape would fail here. Tag `iw2-R-tr-<P7>`, six gate seeds, `--repeats 2`, `LB_IW2_CONTROLS=1`.

- **Capability cell counts** iff DEFINED, S2 `regressed`, the intact reply is not degenerate (in particular: not an
  abstention), names `river`, and carries the route's own-inference framing (`reasoned`: "I believe the cat precedes
  the river, but I reasoned that myself rather than being told it directly.").
- **Control cell holds** iff DEFINED, the intact reply abstains, and the lesion leaves it (S2 `pass`).
- **R PASS:** capability 6/6 and control 6/6. **R FAIL:** any DEFINED miss. **UNDEFINED:** any cell not DEFINED after
  one re-run.
- **Prediction:** PASS, with less margin than the other two packets. Through the handler at dev seed 7 (a scratch
  feasibility run before this prereg was committed, repeated as the recorded dev run in the HANDOFF), all three teach
  turns were stored, "does cat precede river?" read "I believe the cat precedes the river, but I reasoned that myself
  rather than being told it directly." with the three hops in `derived_from`, the lesion read "I don't know about
  that.", a direct pair ("does cat precede ball?") answered the stored fact under the lesion too, and the reverse
  question abstained. Two risks on the gate seeds: the comprehension monitor could hold back a teach turn on some seed
  (the reply then abstains in both arms: a FAIL, "the chain was not stored"), and the keystone's workspace could fail
  to sustain ignition to the third hop on some seed (the same FAIL, "the chase did not reach river").

## Packet-specific production-default criteria (in addition to the shared B1, B2, F2, F3)

- **F1:** in `iw2-flip-<F7>` the row `reasoning-transitive-chat` (REQUIRED_ENV lifted by the flip) is reply-level
  load-bearing on 6/6 seeds with no BRAIN_* flag set, read with step R's guards.

## Declared residuals (from the route's prereg, unchanged)

The question-shape detection is a host regex (now with the closed-class guard above); the direct-fact short-circuit
is `composer.query_patient`; the yes/no framing is a host template; only the positive case is answered (a chase-
confirmed "no" is reported as the same abstain as a genuine unknown); the rich path is not wired. The load-bearing
part is the workspace's re-entrant ignition and its n_ignited read deciding ADVANCE / COMMIT / ABSTAIN.

## Cost (reported, not gated)

The chase runs a multi-cycle spiking simulation per hop on a small workspace bridge; the identity arms record the
chase turn's `turn_wall_s` (non-negotiable 4: speed is secondary).
