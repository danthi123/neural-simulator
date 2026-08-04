---
type: finding
status: undefined
date: 2026-08-03
mechanism: neural-vocal-action-credit-v9-gabab-output
runner: research/runners/_vocal_action_credit_gate_v9_graded_dendritic.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v9/output_seed0_cupy.json
  - research/findings/raw/vocal_action_credit_gate_v9/output_seed0_cupy.json.prov.json
---

# The first v9 output run is undefined because protocol and action coverage failed

<!--derived-->
**Verdict: UNDEFINED.** The CuPy run omitted the passing engagement protocol's
pre-learning baseline trial, despite the Phase-2 preregistration retaining its
timing and engagement checks. The altered trajectory produced only `10/12`
clean training actions in every condition, below the locked 90% requirement.
Every fixed probe then selected neural action `1`; only action `0` had received
reward during training. The run therefore neither conformed to the filed
protocol nor sampled the learned consequence required to judge reward
suppression or expected omission.

## What remains valid

All conditions trained with the expectation output gate closed. Probe
plasticity was disabled, every probe weight remained frozen, formal execution
stayed sealed, and the output-intact reward probe recorded nonzero pre-outcome
SNc GABA-B/GIRK conductance. These facts establish that the runner executed and
the neural output route was physically active; they do not rescue the invalid
training trajectory or establish an appropriate learned consequence.

## Why there is no mechanism verdict

The implementation skipped `run_engagement_condition`'s baseline no-reward
trial and began the 12 training trials from a different neural state. The
resulting `10/12` clean count is a protocol-implementation failure, not a valid
failure of the locked candidate. The original protocol also used one fixed
probe per condition and assumed that the unchanged neural selector would
choose rewarded action `0`. Instead, all six probes chose action `1`. Applying
the action-`0` reward or omission criteria to those rows would confuse an
untrained-action response with a failed learned expectation. The recorded
false checks are therefore diagnostic telemetry, not evidence against the
GABA-B output mechanism.

The exact run is preserved at
`research/findings/raw/vocal_action_credit_gate_v9/output_seed0_cupy.json` with
its provenance sidecar. It used center `2`, seed `0`, the CuPy backend, and
commit `0dd551bd1`.

## Decision

Do not claim a pass or failure from this artifact. The
[v2 preregistration](2026-08-03-neural-vocal-credit-gateB-v9-output-v2-PREREGISTRATION.md)
restores exactly the omitted baseline trial and replaces each single probe with
a fixed four-probe block. It analyzes only brain-selected action-`0` rows for
the learned consequence, requires and retains action-`1` negative controls,
and declares the replacement undefined if matched action coverage is still
absent. No circuit parameter, threshold, center, seed, or action policy may
change.
