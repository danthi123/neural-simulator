---
status: no_go
lane: gateb-v14-performance
date: 2026-08-04
type: finding
---

# V14 Dispatch Cache Passes Active Speed But Fails Default Overhead

The source-sealed matrix at candidate `0ce6cb274` versus historical control
`6c9034991` completed all twelve randomized workers. Both source boundaries
were clean, every structural check passed, and no scientific seed was opened.
The receipt is
`research/findings/raw/v14_stageA_performance_0ce6cb274.json`.

Caching CuPy's exact compiled executors and reusing the effective-current
scratch reduced the direct-output host ratio to `0.7184`, passing the fixed
`0.85` requirement. Active overhead also passed at `0.8748` versus its `1.25`
limit. Default-off performance measured `1.0553` versus the fixed `1.02`
ceiling, so the overall verdict remains **NO-GO**. Physiology was not tested
and promotion remains blocked.

Two preceding attempts to combine SNr and HH into one CuPy graph were rejected
before timing because each changed voltage by `7.629e-6` mV after 64 steps.
The exact float32 materialization boundary remains required. Do not rerun
`0ce6cb274` unchanged or weaken byte equivalence or the performance thresholds.
Profile the default path and introduce a source-level, default-safe performance
hypothesis before another sealed matrix.
