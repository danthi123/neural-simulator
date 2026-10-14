---
type: finding
status: provisional
date: 2026-10-14
lane: integration-first (Stage-1 exit criterion C1)
mechanism: Stage-1 C1 category held-out inheritance on the integrated production brain with the default-off category-inheritance wiring switched on, six seeds, scored by the registered scorer
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage1_c1_category.py
artifacts:
  - research/findings/raw/_stage1_c1_category/gate/aggregate.json
---

# Stage-1 C1 with the category-inheritance wiring on: result (2026-10-14, provisional until Claude inspects)

Prereg: research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md (gate revision 6c2cb1362b34d08d5be1807ce82ad333f221f84c). Scored only by `python -m research.runners.stage1_c1_category --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage1_c1_category/gate/aggregate.json. The production default is unchanged (the wiring stays off by default).

**C1 with the category-inheritance wiring ON: PASS**

| verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|
| PASS | 6/6 | 6/6 | 6/6 | 6/6 | - |

Flip-readiness (not part of the C1 verdict): READY (safety 6/6, specificity 6/6)

Informational arms (defined seeds): off inherits 0/6, changes a C1 probe 6/6; lesion_lateral_learn inherits 0/6, changes a C1 probe 6/6; lesion_lateral_read inherits 6/6, changes a C1 probe 0/6; lesion_draw inherits 6/6, changes a C1 probe 0/6

Per seed:
- seed 42: direct=worm generative=None; lesion direct=None; permuted direct=bug; organ [['worm', 0.1208], ['bug', 0.0]]; defined
- seed 43: direct=worm generative=None; lesion direct=None; permuted direct=bug; organ [['worm', 0.1258], ['bug', 0.0]]; defined
- seed 44: direct=worm generative=None; lesion direct=None; permuted direct=bug; organ [['worm', 0.11], ['bug', 0.0]]; defined
- seed 100: direct=worm generative=None; lesion direct=None; permuted direct=bug; organ [['worm', 0.1083], ['bug', 0.0]]; defined
- seed 101: direct=worm generative=None; lesion direct=None; permuted direct=bug; organ [['worm', 0.12], ['bug', 0.0]]; defined
- seed 102: direct=worm generative=None; lesion direct=None; permuted direct=bug; organ [['worm', 0.1117], ['bug', 0.0]]; defined


## Reading (fixed sentences from the HANDOFF section 6)

C1 with the wiring on: PASS on 6 of 6 seeds (correct, lesion effect reproduced, permuted exposure tracked); flip-readiness READY. Provisional until Claude's adversarial verification.

Informational arms (defined seeds): off inherits 0/6, changes a C1 probe 6/6; lesion_lateral_learn inherits 0/6, changes a C1 probe 6/6; lesion_lateral_read inherits 6/6, changes a C1 probe 0/6; lesion_draw inherits 6/6, changes a C1 probe 0/6
