---
type: finding
status: live
date: 2026-10-04
lane: stage-1 (prove who owns the computation)
mechanism: informational cupy replication of Stage-1 C3 lived-fact recall at the gate runner pin (T24), six seeds, registered aggregate
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage1_exit_gate.py
artifacts:
  - research/findings/raw/_stage1_exit/gpu_lived_cupy/aggregate.json
---

# Stage-1 C3 (`lived`) cupy replication: informational (2026-10-04)

**PROVISIONAL**: written by the local model from the registered scorer's output; not yet inspected by Claude.

Every number below is read from `research/findings/raw/_stage1_exit/gpu_lived_cupy/aggregate.json` by
`python -m research.runners.stage1_exit_gate --aggregate` (report block pasted from
`research/findings/raw/_stage1_exit/gpu_lived_cupy/report.md`).

Informational cupy replication of Stage-1 criterion C3 (lived-fact recall), the gate's runner at `c238ff325`, seeds 42 43 44 100 101 102, the production GPU default (episodic write on by default). Not a gate arm.

On cupy the `lived` group reads FAIL (5 of 6 seeds pass; per seed: 42 PASS, 43 PASS, 44 PASS, 100 PASS, 101 PASS, 102 FAIL — correct=False on seed 102; aggregate why: "capability absent on 1 of 6 seeds").

The numpy gate's C3 reads FAIL (T6 finding). The two are different networks at the same seeds (docs/ENGINE_REFERENCE.md, cross-backend seed trap): a difference is not a backend effect on one network.

What this does not show: any gate verdict; C1, C2 or C4; that numpy and cupy agree cell by cell.

From `research/findings/raw/_stage1_exit/gpu_lived_cupy/report.md`, the `lived` block, unedited:

| group | criterion | role | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|---|
| lived | C3 | primary | FAIL | 5/6 | lesion_recall 0/6, lesion_episodic 5/6 | lesion_recall 6/6, lesion_episodic 6/6 | 5/6 | capability absent on 1 of 6 seeds |
