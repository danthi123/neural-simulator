---
type: finding
status: complete
date: 2026-08-04
mechanism: gateB-v13-stage0-performance-diagnostic-v9
promotion_value: none
seed-waiver: process-only diagnostic; no physiology, learning, or Stage-1 seed was consumed
instrument: RTX 3090 CuPy four-cell comparison, three repetitions per cell, locked V7 control
---

# V13 Stage-0 v9 diagnostic completed without changing the sealed NO-GO

The valid receipt is
`research/findings/raw/v13_stage0_performance_diagnostic_v9-rerun1.json` and its SHA-256 is
`9fa747523aacffa41905be440d272abc9847e83492b67892dfd809220c01aca4`.
It used the project environment (`/home/dant123/Projects/sim/.venv/bin/python`, Python 3.11.14,
CuPy 14.1.1) on the RTX 3090. All 12 repetitions completed and all structural checks passed: the
ordinary path was active, learning was disabled, intrinsic current was absent, external current was
exactly zero, and no Stage-1 configuration was read.

The median CUDA times were:

| Cell | Source | Cache condition | Median seconds |
| --- | --- | --- | ---: |
| A | candidate | cold process | 5.796252 |
| B | locked V7 control | cold process | 5.716331 |
| C | candidate | after v2 warmup | 5.729694 |
| D | locked V7 control | after v2 warmup | 5.768326 |

The candidate/control ratios were `1.013981` for the cold comparison and `0.993303` after the
v2 warmup. The candidate-to-historical-baseline ratios were `1.001282` cold and `0.989784` after
warmup. These measurements help separate the earlier normal-path overhead from process/cache state,
but they do not establish a capability result or authorize a gate promotion.

The receipt explicitly preserves the sealed V8 `PERFORMANCE_NO_GO`, leaves the `1.02` boundary
unchanged, and keeps Stage-1 seed `1031` sealed. Do not rerun v9 unchanged or reinterpret this
process diagnostic as evidence that the vocal-credit mechanism works.

## Environment failure recorded separately

The first attempted receipt,
`research/findings/raw/v13_stage0_performance_diagnostic_v9.json`, is intentionally retained as an
incomplete environment-failure record. It used system `/usr/bin/python` and all 12 workers stopped
at import time with `ModuleNotFoundError: No module named 'h5py'`. No GPU steps ran and it is not a
scientific negative. The rerun used the declared project environment, where `h5py 3.16.0` and CuPy
were available.

## Operational correction

The same handoff review found a stale GPU claim from a completed August 1 job: the command was in
`gpu.queue.done`, its result and log existed, but the single line remained in `gpu.queue.running`.
The dispatcher had fixed the analogous pending-queue `grep` bug but still gated completion cleanup
on `grep` returning zero. Completion cleanup is now unconditional, and the coordinator distinguishes
claim-ledger entries from live processes so this failure cannot make an idle GPU look occupied.
