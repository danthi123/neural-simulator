---
type: finding
status: negative
date: 2026-08-03
mechanism: source-monitor-coresidency-v3
runner: research/runners/_laneC_source_monitor_coresidency_gate_v3.py
artifacts:
  - research/findings/raw/parallel_gates/source_v3_seed232.json
  - research/findings/raw/parallel_gates/source_v3_seed232.json.prov.json
  - research/findings/raw/parallel_gates/source_v3_seed233.json
  - research/findings/raw/parallel_gates/source_v3_seed233.json.prov.json
---

# Local threshold homeostasis does not solve the source-margin tradeoff

<!--derived-->
**Verdict: NO-GO at v3 calibration.** Seeds 232 and 233 both returned
`CALIBRATION_FAIL`, with no undefined reasons. The source-local intrinsic
threshold homeostasis added after v2 did not reliably improve the weakest
source margin while protecting the others. Development and held-out seeds
remain locked.

## Result

<!--derived-->
Seed 232 reached a minimum source margin of `0.0016667`. Homeostasis changed
source-region thresholds but produced zero margin gain for seen, heard, and
self-generated recall. This seed also failed multiple inherited causal
controls: the source-path lesion did not collapse recall, source-path
attribution missed its floor, the ACC lesion did not preserve source activity
while silencing ACC, ACC attribution missed its floor, and an unseen episode
still produced source recall.

<!--derived-->
Seed 233 reached a minimum source margin of `0.0091667`. Its heard-source
margin worsened by `0.0008333` relative to the matched homeostasis lesion, and
the weakest margin did not improve. The inherited causal controls passed on
this seed, so the homeostasis failure is not explained by an invalid lesion
path.

| seed | minimum margin | weakest-margin improvement | heard-margin change | status |
|---:|---:|:---:|---:|:---:|
| 232 | 0.0016667 | no | 0.0000000 | `CALIBRATION_FAIL` | <!--derived-->
| 233 | 0.0091667 | no | -0.0008333 | `CALIBRATION_FAIL` | <!--derived-->

## Provenance

Both seeds were run on mini-PC CPUs with the NumPy backend from clean immutable
source `d24548b6` and manifest
`f4561464b4e03be3a8ac9b024624c8bd7ab67fa56348f4eddd3764a42caad723`.
Their sidecars report `git_dirty=false` and fresh lane-specific corpus checks.
Exact repeats were identical after the provenance workflow was repaired.

Artifacts: `research/findings/raw/parallel_gates/source_v3_seed232.json` and
`research/findings/raw/parallel_gates/source_v3_seed233.json`; their adjacent
`.prov.json` sidecars contain the corresponding provenance records.

## Decision

Do not tune v3 against these calibration seeds, and do not open its later
development or held-out seeds. The next source-monitoring design must address
the underlying representational competition and causal-path instability rather
than retune this local threshold-homeostasis mechanism.
