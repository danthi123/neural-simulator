# Stage-1 attempt 2: the C1 section of Amendment A2 (draft text, Claude review 2026-10-06; RULING stage1-c1-lesion)

Fold this section into Amendment A2 (with the C4 control of RULING stage1-sync), committed before any attempt-2 gate
data, in the amendment order the `prereg_amendment_order` gate checks. Not registered by itself.

bar-sanity: docs/drafts/stage1-exit/bar_sanity_record_s1_c1_fixed_pathway_lesion.json

(Re-run that record in A2's own commit: A2 edits `research/runners/stage1_exit_gate.py`, one of its scorer files, so
the record made before A2 goes stale. Command: `python tools/bar_sanity.py
docs/drafts/stage1-exit/bar_sanity_spec_s1_c1_fixed_pathway_lesion.py --n 1000 --seed 0 --out <same path>`.)

## C1: the counted lesion cuts the route that answers (cat_disc; cat_told diagnostic)

**Why.** Prereg section 3 asks for "a lesion arm per responsible faculty". At attempt 1 no C1 route answered, so the
registered `lesion_draw` (the generative DRAW) was the only candidate. At attempt 2 the C1 answer comes from the
category-inheritance organ on the strict route's recall miss (`webapp/category_inherit_chat.py`, flipped default on
after its dev check; `research/findings/2026-10-06-stage1-c1-category-inheritance-RESULT.md`). `lesion_draw` does
not engage on that route: on the wiring's six gate seeds it changed no probe and left no engagement record. Under
Amendment A1's rule (a no-effect lesion with no engagement record is UNDEFINED) every seed would read UNDEFINED, so
the registered bar is UNREACHABLE for this route (`docs/drafts/stage1-exit/bar_sanity_record_s1_c1_registered.json`).

**Change.** For `cat_disc` and `cat_told`, the counted lesion arm is `lesion_inherit_pathway` with its `.rep0` repeat
(one name for the arm and the in-process patch it installs, `research/runners/stage1_exit_gate.py::install_patch`).
It runs every organ read with both learned pathways (member->frame, frame->frame) silenced and restores them after,
and records `calls`. (The wiring gate's env lesion `lesion_inherit`, `BRAIN_CATEGORY_INHERIT_LESION=1`, is not used by
Stage 1.) The arm is APPLIED only if `calls > 0`; the direct probe's `category_inherit.read` also records `lesion:
"all"` and `lesion_held: true`, reported beside it. `lesion_draw` stays as an informational arm (reported, never
counted). Turns, the permuted arm, the null, the correctness rule and the bar (correct 6/6, lesion effect 6/6
reproduced, permuted tracks 6/6) are unchanged.

**Runner change in A2's commit:** in `GROUPS["cat_disc"]` and `GROUPS["cat_told"]`, `lesions` becomes
`{"lesion_inherit_pathway": {"env": {}, "patch": "lesion_inherit_pathway", "cuts": ...}}` and `lesion_draw` moves to
`informational`. The same commit renames the arm key `lesion_pathway` to `lesion_inherit_pathway` in
`docs/drafts/stage1-exit/bar_sanity_spec_s1_c1_fixed_pathway_lesion.py` (and the `PATCH_OF` / role name in
`bar_sanity_s1_c123_common.py`) before re-running the records that hash them, and makes the C1 engagement check
read the hop record described below. The patch itself is already on main with a CPU test
(`tests/test_stage1_exit_gate.py::test_lesion_inherit_pathway_patch_cuts_the_organ_read_and_counts_calls`). Attempt 2
runs at the flip revision, where the wiring is on by default.

**The attempt-2 C1 configuration includes the hop fix** (RULING c1-hop-first; `claude-review/c1-hop`;
`research/findings/2026-10-06-c1-hop-attenuation-bridging-member.md`): the flip revision has the category-inheritance
wiring ON and its hop attenuation ON by default (`BRAIN_CATEGORY_INHERIT_HOP` unset = on after the flip). Every
attempt-2 arm records the effective values of both (`BRAIN_CATEGORY_INHERIT`, `BRAIN_CATEGORY_INHERIT_HOP`), and the
direct probe's `category_inherit.read` must record `hop: true` in every flag-on C1 arm, else the seed is UNDEFINED for
C1. The C1 evidence A2 rests on is the gate re-measure WITH the fix (prereg 2026-09-30 section 13, amendment
2026-10-06, `research/findings/raw/_stage1_c1_category/gate_hop/`), not the 09-30 PASS measured without it.
