---
type: finding
status: negative
date: 2026-08-03
mechanism: source-monitor-coresidency
runner: research/runners/_laneC_source_monitor_coresidency_gate.py
artifacts:
  - research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed214.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed214.json.prov.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed215.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed215.json.prov.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed310.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed310.json.prov.json
---

# Source-memory co-residency misses the development repeatability gate

<!--derived-->
**Verdict: NO-GO at preregistered development.** Two of three fresh seeds
passed every criterion, but seed 214 recalled the heard source with a `0.11`
margin against the fixed `0.15` floor. Held-out seeds remain locked.

## Fixed Test

The criteria and seeds were committed before any development run. The three
allowed seeds were 214, 215, and 310; all had to pass every criterion without
tuning or exclusion. In addition to the 12 calibration checks, development
required a `0.15` minimum margin for each source, at least `90%` attribution to
each lesioned pathway, and exactly zero source spikes for an unseen episode.

Artifacts:
`research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed214.json`,
`research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed215.json`,
and
`research/findings/raw/parallel_gates/source_monitor_coresidency_development_seed310.json`.
All three ran on the mini-PC pool from clean commit `f2c310a0c` and the same
revision-addressed source manifest.

## Result

| seed | seen margin | heard margin | self-generated margin | source attribution | ACC attribution | unseen spikes | checks | result |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 214 | 0.2317 | 0.1100 | 0.2333 | 1.00 | 1.00 | 0 | 15/16 | fail | <!--derived-->
| 215 | 0.2808 | 0.2492 | 0.3417 | 1.00 | 1.00 | 0 | 16/16 | pass | <!--derived-->
| 310 | 0.1992 | 0.2225 | 0.2150 | 1.00 | 1.00 | 0 | 16/16 | pass | <!--derived-->

The failure is narrow but real. On seed 214, seen and self-generated recall
were comfortably above the fixed floor, source and ACC lesions removed all of
their target activity, learning-off and unseen episodes produced no source
spikes, source swapping followed the physical afferent, and mixed episodes
reinstated both learned sources. Only the heard-source strength was not robust
enough.

## Interpretation

The pathway learns the right association and rejects unsupported episodes, but
the source populations do not yet regulate their learned response strongly
enough across network initializations. The learned route's total strength varied
substantially across the three seeds, and one source pool landed below the
operating margin despite identical training. Averaging the three seeds would
hide the exact repeatability problem the preregistration was designed to catch.

## Decision

Do not open held-out seeds 311, 312, or 313 and do not lower the `0.15` margin.
Preserve v1 as the causal baseline. The next attempt should add a biologically
local stabilizer, such as source-pool competition or region-scoped homeostatic
excitability, without source-specific host gain or stronger labelled drive.
It must use fresh calibration, development, and held-out seed sets because the
v1 development seeds are now observed.

Even a future pass would leave upstream episode allocation, source-afferent
identity, and the externally opened learning window as explicit scaffolds.
