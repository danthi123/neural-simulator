---
status: infrastructure_failure
lane: gateb-v14-performance
date: 2026-08-04
type: finding
---

# V14 Stage A Immutable Performance Worker Timed Out

The replacement performance matrix used clean, detached source trees at
candidate `c672b1708` and control `6c9034991`. Its first randomized cell,
`candidate-active` repetition 3, reached the fixed 1,800-second worker timeout
without returning a measurement.

This is an infrastructure and performance-harness failure, not a physiology or
promotion verdict. No timing ratio was computed, no scientific seeds were
opened, and the remaining eleven cells were not started. The terminal receipt
is `research/findings/raw/v14_stageA_performance_c672b1708_immutable.json`.

GPU memory use remained small and the worker occupied one CPU core throughout,
consistent with the already identified per-step host/launch-overhead boundary.
The next performance action is therefore a bounded profiling and kernel-fusion
slice before another long matrix. Repeating the unchanged 20,000-step cell is
not authorized.
