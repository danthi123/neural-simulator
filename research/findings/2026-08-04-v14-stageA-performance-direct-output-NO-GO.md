---
status: no_go
lane: gateb-v14-performance
date: 2026-08-04
type: finding
---

# V14 Stage A Direct-Output Performance Is Insufficient

The complete source-sealed V14 performance matrix ran candidate `7cfc2607e`
against historical control `6c9034991`. All twelve randomized workers
completed, all structural checks passed, and no scientific seeds were opened.
The receipt is
`research/findings/raw/v14_stageA_performance_7cfc2607e.json`.

Median host-time ratios were:

- candidate default / historical default: `0.9824` (limit `1.02`);
- candidate active / candidate default: `1.1706` (limit `1.25`);
- direct-output active / unfused active: `0.9366` (required `0.85`).

CUDA-event ratios produced the same gate outcome. The default-off and active-overhead
requirements passed, but direct outputs saved only about 6.3 percent rather
than the required 15 percent. The performance verdict is therefore **NO-GO**.
This has no physiology or promotion effect and does not invalidate the prior
Stage A correctness evidence.

The scalar transfer defect is fixed and must not be restored. The next
authorized engineering action is a bounded launch-level profile of this exact
source, followed by a larger byte-equivalent fusion boundary only if the trace
identifies removable launches or synchronization. Do not weaken the threshold
or combine this receipt with either earlier partial matrix.
