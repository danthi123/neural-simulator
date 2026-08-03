---
type: finding
status: positive
date: 2026-08-03
mechanism: neural-vocal-action-selection
runner: research/runners/_vocal_action_selector_gate.py
artifacts:
  - research/findings/raw/vocal_action_selector_gate_v2/seed42.json
  - research/findings/raw/vocal_action_selector_gate_v2/seed43.json
  - research/findings/raw/vocal_action_selector_gate_v2/seed44.json
  - research/findings/raw/vocal_action_selector_gate_v2/seed100.json
  - research/findings/raw/vocal_action_selector_gate_v2/performance_seed7.json
  - research/findings/raw/vocal_action_selector_gate_v2/performance_seed11.json
---

# Neural vocal selection passes the four-seed physiology gate

<!--derived-->
**Verdict: GO at Gate A v2.** The isolated spiking selector produced one clean,
balanced action on 98-100% of trials in every development seed. Shared arousal
and the D1-to-GPi direct path were both necessary. Gate B local-credit work is
unlocked; held-out seeds 101 and 102 remain locked until Gate C.

## Question

Can shared, target-independent practice arousal and premotor variability drive
a two-channel basal-ganglia selector that executes one vocal motor action, uses
both choices, and stops acting when either arousal or direct-path
disinhibition is removed?

The host supplied the same tonic input to both channels, presented shared
practice onset, observed the first motor threshold crossing, and ended the
decision epoch when that neural action executed. It did not inject a channel,
compare channel rates, or choose a fallback action.

## Preregistered Change

Gate A v1 missed its four-seed repeatability criterion because seed 42 omitted
eight actions. Diagnostics localized those omissions to simultaneous
recruitment of two cross-channel striatal fast-spiking-interneuron pools, which
could suppress both D1 routes.

V2 removed only those two 16-neuron pools and their eight proposal, cross-MSN,
and reset pathways. All currents, remaining weights, population sizes,
thresholds, durations, seeds, controls, and GO criteria stayed fixed. The
runner preserves v1 and selects this topology only with
`--selector-version v2`.

## Result

Each main condition and each lesion contained 100 trials on the CuPy production
bridge. The scientific runs used source commit `5ff2bd439`; automatic
provenance sidecars record the command, backend, repository state, and run ID.
Artifacts: `research/findings/raw/vocal_action_selector_gate_v2/seed42.json`,
`research/findings/raw/vocal_action_selector_gate_v2/seed43.json`,
`research/findings/raw/vocal_action_selector_gate_v2/seed44.json`, and
`research/findings/raw/vocal_action_selector_gate_v2/seed100.json`.

| seed | clean commits | channel 0 / 1 wins | smaller share | loser ratio p95 | no arousal | no direct path | GO |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 1.00 | 53 / 47 | 0.470 | 0.00 | 0.00 | 0.00 | yes |
| 43 | 0.98 | 66 / 32 | 0.327<!--derived--> | 0.00 | 0.00 | 0.00 | yes |
| 44 | 1.00 | 43 / 57 | 0.430 | 0.00 | 0.00 | 0.00 | yes |
| 100 | 0.98 | 47 / 51 | 0.480<!--derived--> | 0.00 | 0.00 | 0.00 | yes |

Every seed exceeded the fixed 95% clean-commit floor and 25% smaller-channel
share. No losing motor population spiked before the winner committed. Neither
lesion produced a commit in 400 trials, establishing that shared practice
arousal and the gated D1-to-GPi-to-thalamus route are load-bearing at this
operating point.

The control battery did not separately lesion D2, GPe, or STN, so this result
does not establish that every indirect-path component is causally necessary.

## Performance

The v2 probe contains 600 neurons and 36 declared regional pathways, down from
632 and 44 in v1. Two independent 60-trial calibration/control runs executed
concurrently on the RTX 3090 from source commit `91cc00c14`. Their artifacts
are `research/findings/raw/vocal_action_selector_gate_v2/performance_seed7.json`
and
`research/findings/raw/vocal_action_selector_gate_v2/performance_seed11.json`.

| worker seed | elapsed | trials | worker ms/trial | retained CuPy pool |
|---:|---:|---:|---:|---:|
| 7 | 23.61 s | 60 | 393.56 | 1,612,288 bytes |
| 11 | 24.83 s | 60 | 413.87 | 1,605,120 bytes |

<!--derived-->
The concurrent pair completed 120 trials in 24.83 seconds, or 4.83 trials per
second (206.94 aggregate milliseconds per trial). The retained CuPy pool is an
after-run allocation-pool measure, not an exact `nvidia-smi` peak. It shows
that this isolated probe is not close to the 24 GB memory boundary; it does not
yet establish the cost of three selectors inside the larger continuous brain.

## Decision

Gate A v2 passes. Preserve the v1 NO-GO as evidence that extra bilateral
striatal suppression was harmful at this scale. Proceed to Gate B using only
development seeds 42, 43, 44, and 100: add executed-action collateral and test
whether delayed global dopamine changes only the locally eligible
corticostriatal route. Keep held-out seeds 101 and 102 untouched.

This result establishes selector physiology only. It does not establish reward
learning, convention reversal, words, syntax, or conversation. Injected motor
babbling remains an active scaffold in the integrated vocal-learning runner
until Gates B and C replace it.
