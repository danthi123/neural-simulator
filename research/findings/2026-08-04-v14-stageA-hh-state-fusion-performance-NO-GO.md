---
status: no_go
lane: gateb-v14-performance
date: 2026-08-04
type: finding
---

# V14 Stage A HH State Fusion Remains Below Performance Gate

The source-sealed matrix at candidate `558bb8f55` versus historical control
`6c9034991` completed all twelve randomized workers with every structural and
precondition check passing. The receipt is
`research/findings/raw/v14_stageA_performance_558bb8f55.json`.

The exact in-place HH state/spike fusion improved the direct-output host ratio
to `0.9162`, but the fixed requirement remains `0.85`.
Default-off (`0.9861`) and active overhead (`1.1598`) passed their respective
`1.02` and `1.25` limits. The overall performance verdict remains **NO-GO**,
with no physiology or promotion effect.

The bounded Nsight trace explains why: the HH change removed seven launches
per step, but SNr update and effective-current subtraction remain separate
from the HH graph. The next authorized boundary is a CuPy-generated in-place
SNr-to-HH graph built from the same equation helpers. It must pass exact state,
spike, checkpoint, and continuation controls before another sealed matrix.
Do not rerun `558bb8f55` unchanged or weaken the threshold.
