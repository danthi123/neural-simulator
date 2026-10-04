---
type: finding
status: live
date: 2026-10-04
lane: integration-first (stage exits on the calendar)
mechanism: Stage-1 (INFANT) exit gate on the integrated production brain, six seeds, scored by the registered scorer
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage1_exit_gate.py
artifacts:
  - research/findings/raw/_stage1_exit/gate/aggregate.json
---

# Stage-1 (INFANT) exit gate: result (2026-10-04)

Prereg: research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md (gate revision c238ff325a38e40a68b1f9751822630ff729377b). Scored only by `python -m research.runners.stage1_exit_gate --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage1_exit/gate/aggregate.json.

**Stage-1 exit verdict: FAIL** (primary criteria: C1 FAIL, C2 PASS, C3 FAIL, C4 FAIL)

| group | criterion | role | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|---|
| cat_disc | C1 | primary | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| cat_told | C1 | diagnostic | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| word | C2 | primary | PASS | 6/6 | lesion_recruit 6/6 | lesion_recruit 6/6 | 6/6 | - |
| lived | C3 | primary | FAIL | 6/6 | lesion_recall 0/6, lesion_episodic 6/6 | lesion_recall 6/6, lesion_episodic 6/6 | 6/6 | lesion_recall: reply-level effect on 0 of 6 seeds (not load-bearing) |
| val_heldout | C4 | primary | FAIL | 6/6 | lesion_affect 6/6 | lesion_affect 6/6 | 0/6 | permuted exposure tracked on 0 of 6 seeds (not exposure-driven) |
| val_novel | C4 | diagnostic | FAIL | 0/6 | lesion_affect 0/6 | lesion_affect 6/6 | 0/6 | capability absent on 6 of 6 seeds |

Per-seed valence r (reply-visible differential vs Warriner norm, 24 held-out words):
- seed 42: r = 0.938 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.938; defined
- seed 43: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; defined
- seed 44: r = 0.951 (n = 24, sign agreement 0.88); lesion r = None; permuted-map r = 0.951; defined
- seed 100: r = 0.947 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.947; defined
- seed 101: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; defined
- seed 102: r = 0.944 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.944; defined


## Per-criterion reading (fixed sentences from HANDOFF section 8)

C1 (`cat_disc`): FAIL, capability absent on 6 of 6 seeds. The integrated brain does not show this Stage-1 ability at production defaults; the gap named in prereg section 8 applies.

C2 (`word`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced; the permuted exposure was tracked on 6 of 6.

C3 (`lived`): FAIL, not load-bearing: the reply was correct but the lesion_recall arm changed it on only 0 of 6 seeds.

C4 (`val_heldout`): FAIL, not exposure-driven: the permuted exposure was tracked on only 0 of 6 seeds.

`cat_told` (diagnostic, not counted): FAIL, correct 0/6.

`val_novel` (diagnostic, not counted): FAIL, correct 0/6.
