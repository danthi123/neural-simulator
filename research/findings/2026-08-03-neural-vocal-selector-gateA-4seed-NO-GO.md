---
type: finding
status: negative
date: 2026-08-03
mechanism: neural-vocal-action-selection
runner: research/runners/_vocal_action_selector_gate.py
artifacts:
  - research/findings/raw/vocal_action_selector_gate/seed42.json
  - research/findings/raw/vocal_action_selector_gate/seed43.json
  - research/findings/raw/vocal_action_selector_gate/seed44.json
  - research/findings/raw/vocal_action_selector_gate/seed100.json
---

# Neural vocal selection is causal but misses the repeatability gate

<!--derived-->
**Verdict: NO-GO at Gate A.** The isolated spiking selector produced clean,
balanced actions with causal arousal and basal-ganglia dependence in three of
four development seeds. Seed 42 committed on 92% of trials, below the fixed 95%
requirement. Local-credit learning and held-out seeds remain locked.

## Question

Can shared, target-independent practice arousal and premotor variability drive
a two-channel basal-ganglia selector that executes one vocal motor action, uses
both choices, and stops acting when either arousal or direct-path
disinhibition is removed?

The host supplied the same tonic input to both channels, presented shared
practice onset, observed the first motor threshold crossing, and ended the
decision epoch when that neural action executed. It did not inject a channel,
compare channel rates, or choose a fallback action.

## Result

Each main condition and each lesion contained 100 trials on the CuPy production
bridge. The source was commit `e7f009a37`; automatic provenance sidecars beside
the raw artifacts record the command, backend, repository state, and run ID.
Artifacts: `research/findings/raw/vocal_action_selector_gate/seed42.json`,
`research/findings/raw/vocal_action_selector_gate/seed43.json`,
`research/findings/raw/vocal_action_selector_gate/seed44.json`, and
`research/findings/raw/vocal_action_selector_gate/seed100.json`.

| seed | clean commits | channel 0 / 1 wins | smaller share | loser ratio p95 | no arousal | no direct path | GO |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 0.92 | 43 / 49 | 0.467<!--derived--> | 0.00 | 0.00 | 0.00 | no |
| 43 | 0.99 | 47 / 52 | 0.475<!--derived--> | 0.00 | 0.00 | 0.00 | yes |
| 44 | 0.98 | 47 / 51 | 0.480<!--derived--> | 0.00 | 0.00 | 0.00 | yes |
| 100 | 1.00 | 62 / 38 | 0.380 | 0.00 | 0.00 | 0.00 | yes |

Every seed used both actions well above the 25% floor. No losing motor
population spiked before the winner committed, and neither lesion produced a
single commit. These controls establish that the shared practice signal and
the D1-to-GPi-to-thalamus route are load-bearing at this operating point. They
do not rescue the failed four-seed repeatability claim.

## What Failed

Seed 42 had eight omissions. In each, premotor and striatal-interneuron activity
was present, but D1 activity stayed too weak to silence GPi; thalamus, commit,
and motor therefore remained silent. This is an omission rather than a false
or dual action.

Post-failure development diagnostics reduced the cross-channel
striatal-interneuron gain. Small reductions did not cross the gate; removing
that branch allowed seed 42 to cross 95% in a main-effect diagnostic. That
points to bilateral feed-forward suppression as counterproductive, but it is
not a new result: the simplified circuit has not run the four-seed protocol or
its lesions.

## Decision

Do not proceed to Gate B local credit and do not run held-out seeds 101 or 102.
Revise the striatal competition so equal premotor activity cannot suppress both
direct pathways, preregister the changed topology as a new version, and rerun
all four development seeds with the same 100-trial controls. The v1 artifacts
remain immutable evidence of the boundary.
