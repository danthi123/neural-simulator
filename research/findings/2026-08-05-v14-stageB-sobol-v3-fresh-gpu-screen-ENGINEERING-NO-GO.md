---
type: finding
status: no-go
date: 2026-08-05
mechanism: v14-snr-stageB-fresh-sobol-v3-gpu-screen
lane: executed-action-credit
promotion_value: none
instrument: RTX 3090 CuPy batched engineering screen
artifacts:
  - research/findings/raw/v14_stageB_v3_successor_campaign.json
  - research/findings/raw/v14_stageB_v3_successor_gpu_triage.json
  - research/findings/raw/v14_stageB_v3_batch_width_benchmark_result.json
  - research/experiment-runtime/v14-stageB-sobol-v3-512-results/intact_autonomous/batch-000/receipt.json
  - research/experiment-runtime/v14-stageB-sobol-v3-512-results/nap_lesion/batch-000/receipt.json
  - research/experiment-runtime/v14-stageB-sobol-v3-512-results/cav2_2_lesion/batch-000/receipt.json
  - research/experiment-runtime/v14-stageB-sobol-v3-512-results/sk_lesion/batch-000/receipt.json
  - research/experiment-runtime/v14-stageB-sobol-v3-512-results/hcn_baseline_lesion/batch-000/receipt.json
---

# Fresh V3 Stage B Sobol GPU screen: engineering NO-GO

## Scope

This is an engineering-screening result only. It does not establish a
physiology verdict, does not claim source equivalence, and has no promotion
value. GPU survivors would still require the authoritative NumPy/CPU
confirmation contract; this screen produced none.

## Executed campaign

The fresh successor partition used the exact seed-free Sobol candidates at
global indices 512-1023. The filed campaign contains 512 candidates, five
arms, and batch size 512. All five GPU arm receipts completed against the same
campaign. <!--derived--> That is 2,560 candidate-arm executions. <!--derived-->

The batch-width benchmark selected width 512 under its preregistered
performance-only rule. This is a throughput choice, not a scientific result.

## Strict triage

The completed triage artifact is
`research/findings/raw/v14_stageB_v3_successor_gpu_triage.json`.
Its internal self SHA-256 is
`3c830c82159113cd0e43c5846a666c63726c7863899b745d0ff0eba5edf92b90`.
The artifact records 512 candidates, `process_status: completed`,
`engineering_screening_only: true`, and `scientific_verdict: null`.

Strict noncompensating triage classified the complete fresh partition as:

| classification | count |
|---|---:|
| `engineering_fail` | 421 |
| `engineering_inconclusive` | 91 |
| `engineering_pass` | 0 |

No candidate is eligible for CPU confirmation from this screen. The GPU
result therefore closes this fresh search partition as an engineering NO-GO;
it does not close the biological capability.

## Boundary and next action

Candidates 284 and 404 remain closed and are not retuned or reopened. The
heterogeneous 12-cell SK cohort remains unavailable. The next action is to
resolve the still-missing biological measurement contracts before proposing
another authority search; do not select or tune candidates from this negative
screen by inspection.
