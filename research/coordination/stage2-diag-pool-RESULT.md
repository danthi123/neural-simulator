# stage2_diag pool batch - [s2-diag-pool] RESULT (Hermes, 2026-10-26)

Per RULING stage2-fixes-1026 (2): results synced, committed with sidecars as they landed
(29e9d40d6, cfcd50b8c, 3d4beaee3); pool drained (0 queued); all exits rc=0, no deaths to
requeue (the only non-zero exit today is dejits-attempt-3, separately handled under
[dejits-diag-1025]); aggregator run over each directory with its seeds. Counts, verbatim
output below. NOT interpreted - Claude scores.

## Counts

- Total artifacts: 31 json + 31 prov sidecars (31/31 paired, 0 orphans) + 2 aggregate.json
  (+2 sidecars, provenance stamped by the aggregator run).
- c3_gainpinned: 22 artifacts = seeds 1,2,3,4,5,6,8,9,10,11,12 x arms
  ask__da_lesion / ask__reversed_dalesion (11 seeds x 2 = 22, complete).
- c2_comprehension_gateoff: 9 artifacts = seeds 1,2,3 x arms ask__intact_a_gateoff /
  infer__intact_a_gateoff / infer__permuted_gateoff (3 x 3 = 9, matches the 9 planned
  claim rows, complete).
- Pool: 0 queued; exits since stocking all rc=0; deaths today: dejits-attempt-3 only.

## Aggregator command (stage2_exit_gate.py:48 pattern)

    .venv/bin/python -m research.runners.stage2_exit_gate --aggregate \
        research/findings/raw/_stage2_diag/c3_gainpinned \
        --seeds 1,2,3,4,5,6,8,9,10,11,12 --out .../c3_gainpinned/aggregate.json
    .venv/bin/python -m research.runners.stage2_exit_gate --aggregate \
        research/findings/raw/_stage2_diag/c2_comprehension_gateoff \
        --seeds 1,2,3 --out .../c2_comprehension_gateoff/aggregate.json

Both rc=0. run_id 1793047247-70410 (c3), 1793047248-70452 (c2).

## VERBATIM output - c3_gainpinned (44 lines)

```
**Stage-2 exit verdict: UNDEFINED (INCOMPLETE): unmet: no primary shard missing (measured=330 expect=0); unmet: null clean (intact_a == intact_b) on every primary (group, seed) (measured=55 expect=0); unmet: six seeds defined for every primary criterion (measured=5 expect=0)** (primary criteria: C1 INCOMPLETE, C2 INCOMPLETE, C3 INCOMPLETE, C4 INCOMPLETE, C5 INCOMPLETE)

| group | criterion | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|
| conv | C1 | INCOMPLETE | 0/0 | lesion_mouth 0/0 | lesion_mouth 0/0 | 0/0 | 0 of 11 seeds defined |
| infer | C2 | INCOMPLETE | 0/0 | lesion_chain_route 0/0 | lesion_chain_route 0/0 | 0/0 | 0 of 11 seeds defined |
| ask | C3 | INCOMPLETE | 0/0 | lesion_curiosity 0/0, lesion_graded 0/0 | lesion_curiosity 0/0, lesion_graded 0/0 | 0/0 | 0 of 11 seeds defined |
| emo | C4 | INCOMPLETE | 0/0 | lesion_affect 0/0 | lesion_affect 0/0 | 0/0 | 0 of 11 seeds defined |
| inner | C5 | INCOMPLETE | 0/0 | lesion_workspace 0/0, lesion_selfinit 0/0 | lesion_workspace 0/0, lesion_selfinit 0/0 | 0/0 | 0 of 11 seeds defined |

Sub-clauses (seeds true / seeds defined; reported, never a pass condition alone):
- ask: reversed_all 0/0, reversed_discriminates 0/0, reversed_lp_learnable 0/0, reversed_lp_unlearnable 0/0, all_taught_moats_acquired 0/0
- inner: private_selected_by_state 0/0

Per-seed curiosity ask-rates (novel-topic gaps vs taught-topic moats; the bar is gap >= 2/3 and gap >= 2x moat; amendment A4: every clause in BOTH orders, a taught moat not acquired is undefined):
- seed 1: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 2: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 3: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 4: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 5: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 6: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 8: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 9: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 10: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 11: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 12: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing

Per-seed discrete-emotion read (category named per situation; the 2-D sign/arousal read beside):
- seed 1: named None; signs None; arousal None; distinct 2-D reads None
- seed 2: named None; signs None; arousal None; distinct 2-D reads None
- seed 3: named None; signs None; arousal None; distinct 2-D reads None
- seed 4: named None; signs None; arousal None; distinct 2-D reads None
- seed 5: named None; signs None; arousal None; distinct 2-D reads None
- seed 6: named None; signs None; arousal None; distinct 2-D reads None
- seed 8: named None; signs None; arousal None; distinct 2-D reads None
- seed 9: named None; signs None; arousal None; distinct 2-D reads None
- seed 10: named None; signs None; arousal None; distinct 2-D reads None
- seed 11: named None; signs None; arousal None; distinct 2-D reads None
- seed 12: named None; signs None; arousal None; distinct 2-D reads None

Missing shards: 330

-> research/findings/raw/_stage2_diag/c3_gainpinned/aggregate.json
[provenance] stamped 1 artifact(s) | run_id 1793047247-70410
```

## VERBATIM output - c2_comprehension_gateoff (28 lines)

```
**Stage-2 exit verdict: UNDEFINED (INCOMPLETE): unmet: no primary shard missing (measured=90 expect=0); unmet: null clean (intact_a == intact_b) on every primary (group, seed) (measured=15 expect=0); unmet: six seeds defined for every primary criterion (measured=5 expect=0)** (primary criteria: C1 INCOMPLETE, C2 INCOMPLETE, C3 INCOMPLETE, C4 INCOMPLETE, C5 INCOMPLETE)

| group | criterion | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|
| conv | C1 | INCOMPLETE | 0/0 | lesion_mouth 0/0 | lesion_mouth 0/0 | 0/0 | 0 of 3 seeds defined |
| infer | C2 | INCOMPLETE | 0/0 | lesion_chain_route 0/0 | lesion_chain_route 0/0 | 0/0 | 0 of 3 seeds defined |
| ask | C3 | INCOMPLETE | 0/0 | lesion_curiosity 0/0, lesion_graded 0/0 | lesion_curiosity 0/0, lesion_graded 0/0 | 0/0 | 0 of 3 seeds defined |
| emo | C4 | INCOMPLETE | 0/0 | lesion_affect 0/0 | lesion_affect 0/0 | 0/0 | 0 of 3 seeds defined |
| inner | C5 | INCOMPLETE | 0/0 | lesion_workspace 0/0, lesion_selfinit 0/0 | lesion_workspace 0/0, lesion_selfinit 0/0 | 0/0 | 0 of 3 seeds defined |

Sub-clauses (seeds true / seeds defined; reported, never a pass condition alone):
- ask: reversed_all 0/0, reversed_discriminates 0/0, reversed_lp_learnable 0/0, reversed_lp_unlearnable 0/0, all_taught_moats_acquired 0/0
- inner: private_selected_by_state 0/0

Per-seed curiosity ask-rates (novel-topic gaps vs taught-topic moats; the bar is gap >= 2/3 and gap >= 2x moat; amendment A4: every clause in BOTH orders, a taught moat not acquired is undefined):
- seed 1: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 2: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing
- seed 3: gap None, moat None, learnable None, unlearnable-stops None, moats undefined none; intact arm missing

Per-seed discrete-emotion read (category named per situation; the 2-D sign/arousal read beside):
- seed 1: named None; signs None; arousal None; distinct 2-D reads None
- seed 2: named None; signs None; arousal None; distinct 2-D reads None
- seed 3: named None; signs None; arousal None; distinct 2-D reads None

Missing shards: 90

-> research/findings/raw/_stage2_diag/c2_comprehension_gateoff/aggregate.json
[provenance] stamped 1 artifact(s) | run_id 1793047248-70452
```