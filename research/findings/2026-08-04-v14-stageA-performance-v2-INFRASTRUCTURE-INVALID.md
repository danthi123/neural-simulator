---
status: infrastructure_invalid
lane: gateb-v14-performance
date: 2026-08-04
type: finding
---

# V14 Performance V2 Exposes Unstable Benchmark Environment

The prospective v2 matrix at candidate `59ca01e7d` versus historical control
`6c9034991` completed all twelve observations from clean source boundaries.
No scientific seed was opened and every structural check passed. The receipt
is `research/findings/raw/v14_stageA_performance_v2_59ca01e7d.json`.

<!--derived-->
The preregistered infrastructure checks failed. Three observations exceeded
the `0.10` within-observation block-range limit, reaching `0.119`, `0.170`,
and `0.206`. Default and active paired-ratio ranges were about `0.073` and
`0.074`, above the `0.05` limit. The result is therefore
**INFRASTRUCTURE_INVALID**, with no performance, physiology, or promotion
verdict.

<!--derived-->
For diagnosis only, the median paired ratios were `1.0205` default, `0.9249`
active, and `0.7550` direct versus unfused. The direct ratios were relatively
stable (`0.7484` to `0.7667`), but none may promote the mechanism because the
whole matrix failed its environmental validity contract.

The next run requires a documented environment change: isolate host dispatch,
record and stabilize GPU clock/power state where supported, avoid desktop or
foreign GPU interference, and preserve adjacent pairing and the existing
thresholds. Do not reinterpret this receipt or rerun unchanged conditions as
promotion evidence.
