---
type: finding
status: undefined
date: 2026-08-03
mechanism: neural-vocal-action-credit-v10-policy-eligibility
runner: research/runners/_vocal_action_credit_gate_v10_policy.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v10/engagement_seed0_cupy.json
  - research/findings/raw/vocal_action_credit_gate_v10/engagement_seed0_cupy.json.prov.json
---

# V10 policy eligibility is undefined because actions did not remain exclusive

<!--derived-->
**Verdict: UNDEFINED_ACTION_COVERAGE.** The reserved-seed CuPy smoke ran from
committed revision `5e7130fab` under preregistration `696603c5a`. Both neural
actions crossed first across the 12 trials, but the other action crossed later
in every fixed 600-step action window. This left zero clean trials, so the
preregistered action-local selectivity question cannot be scored.

Artifact:
`research/findings/raw/vocal_action_credit_gate_v10/engagement_seed0_cupy.json`
with its `.prov.json` provenance sidecar.

## What the run establishes

- First crossings were balanced enough to observe both actions: action `0`
  crossed first in `7/12` trials and action `1` in `5/12`.
- The intact and coactivity-lesion conditions began with byte-identical
  weights and had identical complete firing hashes, first crossings, decision
  steps, motor counts, and cleanliness results.
- The intact condition produced nonzero policy eligibility. The lesion
  produced exactly zero D1 and D2 policy eligibility, so all observed
  eligibility was attributable to the local pre/post coactivity mechanism.
- Construction, strict CuPy execution on the RTX 3090, phase schedules,
  washout, pre-action trace hygiene, weight immutability, and the independent
  clipping-path control all passed.
- No reward, dopamine teaching signal, or policy weight update ran. This was
  an engagement smoke only.

The lesion-collapse validity field is false only because its denominator is
defined from clean winner-labelled rows and that set was empty. The lesion's
actual maximum D1 and D2 eligibility was exactly zero.

## Diagnostic characterization only

At the first unique crossing, grouping all rows by the action that crossed
first gives mean selected-versus-other route eligibility of `0.0267` versus <!--derived-->
`0.00247` for D1 and `0.0279` versus `0.00234` for D2. <!--derived--> The selected route was
larger in `11/12` rows for each cell class. These post-run summaries suggest
that a neural commitment can initially leave an action-local tag, but they are
not a GO result: all 12 rows violated the locked cleanliness prerequisite.

Continuing the same shared arousal after the first commitment allowed the
other motor channel to cross in every trial. The smoke therefore exposed a
whole-circuit gap that earlier stop-on-commit protocols hid: this selector can
choose an action, but it does not preserve one exclusive action under a long,
unchanged drive period.

## Decision

Do not repeat seed `0`, tune the selector, relax cleanliness after the fact,
open formal seeds, or proceed to policy learning. This run does not show that
coactivity eligibility failed; it shows that the locked protocol could not
isolate one completed action long enough to evaluate delayed action-local
eligibility.

A successor needs a new research gate grounded in biological action
boundaries. It should test a neural commitment, termination, refractory, or
corollary-discharge mechanism that changes circuit state after commitment
while keeping host timing and winner handling symmetric. Merely stopping the
simulator from Python at the first threshold crossing would restore the old
host shortcut and would not close the revealed gap.
