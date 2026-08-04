---
type: finding
status: infrastructure-failure-performance-pending
date: 2026-08-04
mechanism: neural-vocal-action-credit-v14-explicit-snr-pacemaker-stageA
promotion_value: none
seed-waiver: seed-free engineering comparison; no physiology, learning, calibration, replication, or held-out seed opened
instrument: RTX 3090 CuPy candidate/control/active timing harness
---

# Gate B v14 Stage A performance attempt 1 timed out without a measurement

The first preregistered performance worker ran the prechange default-off control
for 3,600 seconds and then hit the harness timeout. The worker remained live,
with one CPU core near full use and low RTX 3090 utilization, but did not finish
the requested 20,000 measured steps. It emitted no host or CUDA-event timing.

This is an **infrastructure failure**, not a performance NO-GO. It changes no
correctness result, opens no scientific seed, and does not promote or reject the
SNr conductance substrate. The durable receipt is
`research/findings/raw/v14_snr_conductance_stageA/performance-matrix-attempt1-timeout.json`.

## Cause and protocol amendment

Static path inspection found that this 600-neuron workload submits roughly
twenty small kernels or device copies per simulation step. Python launch and
orchestration overhead keeps one CPU core busy while the GPU waits between
small operations. The failed harness also wrote no partial receipt before the
worker returned, so the timeout erased the controller's live state.

Before observing any completed timing value, the Stage A performance protocol
was amended from 20,000 to 2,000 measured steps. The original 500-step warmup,
three repetitions, randomized cell order, candidate/control/active definitions,
and fixed `1.02` and `1.25` thresholds are unchanged. The expected total cost is
now 4,800 seconds. The harness checkpoints after every cell and records a
timeout as a durable infrastructure failure.

## Next action

Seal the amended harness and rerun the same seed-free matrix under the shared
GPU lease. Independently, preregister a behavior-equivalent execution-efficiency
lane for an in-place HH plus SNr kernel and then guarded multi-step batching;
do not edit the live simulator tree while a contemporaneous matrix is running.
