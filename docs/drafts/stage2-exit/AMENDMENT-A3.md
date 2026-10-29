# Stage-2 exit gate, Amendment A3: C4 and C5 must show the answer depends on the brain's state (Claude, 2026-10-06)

Amends `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` (kept byte-identical, so `PREREG_SHA` is
unchanged). Filed BEFORE any gate-seed job was queued and before `GATE_SHA` exists. No Stage-2 gate data exist (the
window is 2026-11-03..11-10), so the change is outcome-blind. Criteria C1, C2 and C3, every arm, turn and seed, the
runner's worker, the group verdict rule and the stage verdict rule are unchanged. Two registered clauses change.

## Why

The bar-sanity sweep over the six Stage-2 bars (`bar_sanity_record_s2_*.json`) found each bar passable by a perfect
brain and failed by every registered null. Two are weak against a host shortcut, which matters in the phase "prove who
owns the computation" and under the rule that everything between sensation and action is neurons:

- C4 `emo`. A brain whose emotion-category word is a lookup of the heard affect words, independent of the affect ladder,
  passed 1.000 (`info_word_lookup_ladder_independent`, the old C4 record). The lesion clause was met merely because the
  printed valence number moved when the ladder was clamped; the named category survived the lesion untouched.
- C5 `inner`. A brain voicing one fixed stored fact on every idle turn passed 1.000 (`info_constant_idle_fact`, the old
  C5 record). The bar checked that the voiced fact is stored, never that the brain's own state selected it.

## A3.1 C4: the lesion must change the NAMED category

Replaces the lesion-effect clause of prereg 4.4 / section 7 for group `emo` only. Per lesion seed, the lesion arm
(`lesion_affect`) now counts as having an effect iff BOTH hold:

1. (unchanged) a primary reply field differs from `intact_a` on at least one scored feel turn; and
2. (new) on EVERY one of the four scored feel turns (`feel_joy`, `feel_fear`, `feel_anger`, `feel_sadness`) the discrete
   category the reply names (`parse_feel(...)["category"]`, the frozen `EMOTION_LEXICON`) differs from the category
   `intact_a` names on that turn. A turn naming no category under both arms counts as not changed.

This is the prereg's own sentence ("the ladder lesion changes every feel reply", section 4.4 anti-cheat 2), made to
bind the category and not only the number. `reproduced`, the permuted-tracks clause, correctness and the lesion arm's
construction are unchanged. The aggregate records `reply_effect` (clause 1) and `category_changed` (per turn) beside the
`effect`, and a "not load-bearing" reason prints how many seeds had the number move versus the category. Code:
`named_category_changed` and the `emo` branch of the lesion loop in `research/runners/stage2_exit_gate.py`.

## A3.2 C5: the voiced idle fact must follow the exposure

Extends the permuted-tracks clause of prereg 4.5 / section 7 for group `inner` only. The permuted-exposure arm is the
second registered idle context: the same `say something` turn after different teaching. `tracks` is now true iff

1. (unchanged) the permuted arm's scored replies follow the permuted chain (`score_inner(permuted=True)["all"]`, which
   already requires the idle utterance to be a stored fact of that arm); and
2. (new) the triple voiced on the idle turn in the permuted arm DIFFERS from the triple voiced in `intact_a`
   (`private_selected_by_state`: both present, both stored, not equal).

A brain voicing one fixed fact on every idle turn now reads FAIL "not exposure-driven" with the sub-clause
`private_selected_by_state` at 0/6 while `private_voices_stored_fact` stays 6/6, so the report says which clause failed.
Nothing new is recorded by the worker: the voiced utterance is the reply text the runner already stores
(`private_utterance`). Code: `private_selected_by_state` and the `inner` branch of the permuted block in the runner.

Stated risk, not tuned around: a brain whose idle thought is a fixed built-in fact under both exposures fails this
clause even though that fact is stored. That is the intended reading ("the idle thought depends on what the brain has
learned"); the next integration slot for C5 is already named (NEEDS-CLAUDE #15: select from the brain's own store).

## What these clauses do NOT establish

Both are necessary conditions for "the brain's state decides", not proof of ownership. A category lookup that prints its
word only while the ladder reads non-neutral, and a host first-taught-fact FIFO over the store, still satisfy the
hardened bars; the bar-sanity specs carry them as `info_word_lookup_gated_by_ladder` and `info_first_taught_fact`
(reported, never thresholded; both pass 1.000). Who owns the category and the selection is the host-share instrument's
question (prereg section 5, NEEDS-CLAUDE #10), not this bar's.

## Bar-sanity (n = 1000 per brain, seed 0; `tools/bar_sanity.py`)

New null generators (promoted from `info_*`): C4 `null_word_lookup_ladder_independent` (host word lookup, ladder
lesion moves the number and not the word) 1.000 -> 0.000; C5 `null_constant_idle_fact` (one stored fact every idle turn)
1.000 -> 0.000. Every other null stays 0.000 and every perfect brain stays 1.000 (the C5 perfect brain now voices the most
recently taught fact, a deterministic function of its store, since a random pick is neither a state function nor
null-clean). C1, C2, C3 and the stage record were re-measured at the amended scorer and read OK unchanged.

bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c4_emo_A3.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c5_inner_A3.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_stage_A3.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c1_conv_A3.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c2_infer_A3.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c3_ask_A3.json

The records they replace (old scorer; superseded, stale by sha, never to be cited as the Stage-2 bar-sanity):

bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c4_emo.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c5_inner.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_stage.json

## Dev seed 7 under the amended scorer (reported, not tuned to)

The branch carries dev-seed-7 shards (`research/findings/raw/_stage2_exit/dev/s7/`). They govern nothing (prereg
section 11). Read through the amended scorer, the dev brain would NOT pass either hardened bar, as it did not pass the
old ones. C4: no feel read names a category (capability absent, prediction FAIL stands), so `category_changed` is False
on all four turns for `lesion_affect`. C5: the idle turn voices the self-initiation organ's fixed demo lexicon ("bird
see seed"), so `private_voices_stored_fact` is already False (NEEDS-CLAUDE #15); the new clause is not evaluable at dev
(no permuted dev shard) and would add nothing. The predictions in prereg section 9 stand as written.

## Consequences for the packet

- `PREREG_SHA` is unchanged. `GATE_SHA` (not yet written) must be pinned at a revision that contains this file, the
  amended runner and the amended tests (the branch `claude-review/stage2-c4c5` tip or a descendant); the HANDOFF's
  step-3 prereg-byte-identical check is unaffected.
- No dev line is re-run: the worker, arms, turns, lesions and shard format are unchanged, so every existing dev shard
  re-scores under the new code (the pool revision `f2aaabc8a` produced shards the new aggregate reads identically).
- Pool lines are emitted per revision (`--emit-pool-jobs <GATE_SHA>`); nothing in a line depends on the scorer change.
