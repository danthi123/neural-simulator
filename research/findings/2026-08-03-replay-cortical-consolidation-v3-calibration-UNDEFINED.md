---
type: finding
status: invalid
date: 2026-08-03
mechanism: replay-driven-cortical-consolidation-v3
runner: research/runners/_replay_cortical_consolidation_gate_v3.py
artifacts:
  - research/findings/raw/parallel_gates/replay_v3_calibration.json
  - research/findings/raw/parallel_gates/replay_v3_calibration.json.prov.json
---

# Replay consolidation v3 calibration is undefined

<!--derived-->
**Verdict: UNDEFINED.** Both calibration seeds, `228` and `229`, failed the
preregistered validity precondition requiring intact sleep to contain uncued
replay, learned index-relay activity, and both inhibitory loops. Intact mean
recovery was `0.0` on both seeds. The downstream scientific checks are
therefore non-interpretable: this is an invalid calibration result, not a
mechanism `NO-GO`.

<!--derived-->
The telemetry localizes the failed precondition. Both seeds contained all 24
uncued reactivation events, cortical-index activity (`156` and `240` spikes),
index fast-spiking activity (`21` and `33` spikes), and nonzero slow index
balance conductance. The cortical-target fast-spiking population produced
exactly zero spikes on both seeds while cortical-target activity remained weak
(`19` and `39` spikes). The missing element was therefore the local target
inhibition loop, not hippocampal replay or the index relay.

Evidence: `research/findings/raw/parallel_gates/replay_v3_calibration.json`
and `research/findings/raw/parallel_gates/replay_v3_calibration.json.prov.json`.

## Provenance

The exact two-seed calibration ran with the NumPy backend on a mini-PC from
clean immutable source `d24548b6` and source manifest
`f4561464b4e03be3a8ac9b024624c8bd7ab67fa56348f4eddd3764a42caad723`.
The sidecar records `git_dirty=false`, a fresh lane-specific corpus check, and
no inspection of reserved seeds. After the provenance workflow was repaired,
the exact repeat was scientifically identical.

## Decision

Development seeds `230`, `231`, and `326` and held-out seeds `327`, `328`, and
`329` remain locked. No scientific claim about replay consolidation v3 follows
from this calibration run. Diagnose which required sleep population is absent
using non-scientific smoke seed `216` only. Any mechanism correction must be
preregistered with a fresh calibration partition rather than tuned against
seeds `228` and `229`.
