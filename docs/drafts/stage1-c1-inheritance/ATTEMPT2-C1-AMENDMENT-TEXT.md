# Stage-1 attempt 2: the C1 section of the attempt-2 amendment (draft text, Claude review 2026-10-06)

Fold this section into the Stage-1 attempt-2 amendment (with the C4 control of RULING stage1-sync), committed before
any attempt-2 gate data, in the amendment order the `prereg_amendment_order` gate checks. Not registered by itself.

bar-sanity: N/A the C1 bar is unchanged; this section substitutes which lesion arm feeds the existing lesion clause

## C1: the counted lesion is the responsible faculty's (cat_disc; cat_told diagnostic)

**Why.** Prereg section 3 asks for "a lesion arm per responsible faculty". At attempt 1 no C1 route answered, so the
registered `lesion_draw` (the generative DRAW) was the only candidate. At attempt 2 the C1 answer comes from the
category-inheritance organ on the strict route's recall miss (`webapp/category_inherit_chat.py`, flipped default on
after its dev check; `research/findings/2026-10-06-stage1-c1-category-inheritance-RESULT.md`). `lesion_draw` does not
engage on that route: on the wiring's six gate seeds it changed no probe and left no engagement record. Under
Amendment A1's rule (a no-effect lesion with no engagement record is UNDEFINED) it would make C1 UNDEFINED, which
measures the instrument, not the brain.

**Change.** For `cat_disc` and `cat_told`, the counted lesion arm is `lesion_inherit` with its `.rep0` repeat:
`BRAIN_CATEGORY_INHERIT_LESION=1` (both learned organ pathways silenced at read time, restored after). It is
ENGAGED only if the direct probe's `category_inherit.read` records `lesion: "all"` and `lesion_held: true`; a lesion
arm without that record makes the seed UNDEFINED for C1. `lesion_draw` stays as an informational arm (reported, never
counted). Turns, the permuted arm, the null, the correctness rule and the bar (correct 6/6, lesion effect 6/6
reproduced, permuted tracks 6/6) are unchanged.

**Runner change it needs** (`research/runners/stage1_exit_gate.py`, Claude, in the amendment's commit): add the arm
to both groups' `lesions` (env above, `patch` None), move `lesion_draw` to `informational`, and make the C1 lesion
effect require the engagement record (a scorer test with a lesion arm lacking `lesion_held` reading UNDEFINED).
Attempt 2 runs at the flip revision, where the wiring is on by default; every arm records the flag's effective value.
