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

The worker configuration was 500 warmup steps followed by 2,000 measured
steps. Diagnosis after the receipt showed that CuPy could not locate
`libcudadevrt.a` for the cold fusion compile. Non-strict bridge execution
swallowed that exception and retried the failed compile every step. The timeout
therefore did not measure steady-state launch overhead. The repaired benchmark
restores the explicit CUDA toolkit root and enables strict step errors so this
failure mode cannot be timed as useful work.

This replacement run is distinct from the earlier abandoned 20,000-step
attempt described in the preregistration amendment. Repeating either failed
run unchanged is not authorized.
