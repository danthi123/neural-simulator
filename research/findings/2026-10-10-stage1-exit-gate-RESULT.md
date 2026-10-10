---
type: finding
status: live
date: 2026-10-10
lane: integration-first (stage exits on the calendar)
mechanism: Stage-1 (INFANT) exit gate on the integrated production brain, six seeds, scored by the registered scorer
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage1_exit_gate.py
artifacts:
  - research/findings/raw/_stage1_exit/gate/aggregate.json
---

# Stage-1 (INFANT) exit gate: result (2026-10-10)

Prereg: research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md (gate revision c238ff325a38e40a68b1f9751822630ff729377b). Scored only by `python -m research.runners.stage1_exit_gate --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage1_exit/gate/aggregate.json.

**Stage-1 exit verdict: UNDEFINED (INCOMPLETE): unmet: six seeds defined for every primary criterion (measured=2 expect=0)** (primary criteria: C1 FAIL, C2 PASS, C3 INCOMPLETE, C4 INCOMPLETE)

| group | criterion | role | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|---|
| cat_disc | C1 | primary | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| cat_told | C1 | diagnostic | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| word | C2 | primary | PASS | 6/6 | lesion_recruit 6/6 | lesion_recruit 6/6 | 6/6 | - |
| lived | C3 | primary | INCOMPLETE | 0/0 | lesion_recall 0/0, lesion_episodic 0/0 | lesion_recall 0/0, lesion_episodic 0/0 | 0/0 | 0 of 6 seeds defined |
| val_heldout | C4 | primary | INCOMPLETE | 0/0 | lesion_affect 0/0 | lesion_affect 0/0 | 0/0 | 0 of 6 seeds defined |
| val_novel | C4 | diagnostic | FAIL | 0/6 | lesion_affect 0/6 | lesion_affect 6/6 | 0/6 | capability absent on 6 of 6 seeds |

Per-seed valence r (reply-visible differential vs Warriner norm, 24 held-out words):
- seed 42: r = 0.938 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.938; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 43: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 44: r = 0.951 (n = 24, sign agreement 0.88); lesion r = None; permuted-map r = 0.951; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 100: r = 0.947 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.947; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 101: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 102: r = 0.944 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.944; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)


## Per-criterion reading (fixed sentences from HANDOFF section 8)

cat_disc (C1): FAIL, capability absent on 6 of 6 seeds. The integrated brain does not show this Stage-1 ability at production defaults; the gap named in prereg section 8 applies.

word (C2): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced; the permuted exposure was tracked on 6 of 6.

lived (C3): UNDEFINED, 0 of 6 seeds defined (reasons in aggregate.json).

val_heldout (C4): UNDEFINED, 0 of 6 seeds defined (reasons in aggregate.json).

cat_told (diagnostic, not counted): FAIL, correct 0/6.

val_novel (diagnostic, not counted): FAIL, correct 0/6.

val_heldout per-seed r lines (verbatim from the report):

- seed 42: r = 0.938 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.938; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 43: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 44: r = 0.951 (n = 24, sign agreement 0.88); lesion r = None; permuted-map r = 0.951; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 100: r = 0.947 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.947; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 101: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)
- seed 102: r = 0.944 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.944; permuted map deranged 0 of 24 held-out words (the learned map was not loaded)

## Section-9 STOP (invoked)

Section 9: "Any UNDEFINED reason in section 8 other than missing shards." Both undefined reasons qualify: C3 all
six seeds `lesion_recall: an env-flag lesion with no reply change and no engagement record (inert lesion and
non-load-bearing path are indistinguishable)`; C4 all six seeds `permuted map deranged 0 of 24 held-out words
(the learned map was not loaded)`. The line of work stops here: no requeue, no re-run, no scorer/arm edit.
Judgment items written into the packet WORKLOG for Claude (NEEDS-CLAUDE filed on main).
