---
status: qualified
type: finding
date: 2026-08-05
---

# V14 Stage B phased NaP GPU screen - IMPLEMENTED

## Result

The engineering screen can now run the V3 NaP assay on the GPU as one
continuous simulated cell: 2 seconds intact, complete NaP lesion at the filed
sample boundary, then 1 second post-lesion. GPU triage independently recomputes
baseline stability, post-lesion spike count, and median voltage direction from
the authenticated raw trace. A candidate cannot pass by becoming silently
depolarized.

This is engineering triage only. Any selected candidate still requires the
authenticated NumPy confirmation path before scientific interpretation.

## Why this was needed

The first screen initialized the NaP-lesion arm with NaP already absent and
tested only one-second spike silence. Candidates 284 and 404 passed that proxy,
but V3 same-cell confirmation showed that both became silent at a more positive
median voltage. Candidate 404 therefore failed the filed biological direction;
284 was unavailable because its intact baseline drift exceeded 0.5 mV.

Inspection of the equations and authenticated traces rules out a NaP sign bug.
NaP is inward below its sodium reversal potential. Removing it stops the spike
cycle, which also removes spike-maintained calcium/SK outward current. The cell
can then settle on a quiet subthreshold branch above the median of its previous
spike/AHP waveform. The source nevertheless reports that riluzole moved SNr
cells from a baseline near -50 mV to a hyperpolarized state and stopped firing
(Ding, Wei, and Zhou 2011, Figure 11). The filed median estimator remains a
project-operational directional gate; it is not relabelled as source-exact after
seeing these results.

## Implementation

- V1 declarations retain their exact 20,000-step pre-lesioned behavior and V1
  output schemas.
- V3 declarations produce versioned campaign, runner, receipt, and triage
  schemas.
- The V3 NaP arm executes 39,999 intact updates, lesions NaP between samples,
  and executes 20,001 lesioned updates. This places the first lesioned sample at
  2.0 s and preserves the filed 60,000-sample trace endpoint.
- Triage uses the same half-open windows as authoritative confirmation:
  `[1.0, 1.5)`, `[1.5, 2.0)`, and `[2.0, 3.0)` seconds.
- The noncompensating GPU checks now require baseline drift at most 0.5 mV,
  zero post-lesion spikes, and a strictly negative post-minus-baseline median.
- Existing V1 triage remains unchanged when no phased trace is present.

## Verification

- Implementation artifacts:
  `research/runners/v14_stageB_batched_physiology.py`,
  `tools/v14_stageB_campaign.py`, and `tools/v14_stageB_gpu_triage.py`.
- Measurement artifacts:
  `research/experiment-runtime/v14-stageB-v3-companion-r2/results/v14-stageB-sobol-284-b89bd2d48cda/confirmation-receipt.json`,
  `research/experiment-runtime/v14-stageB-v3-companion-r2/results/v14-stageB-sobol-404-1676dc9e28ab/confirmation-receipt.json`,
  and `research/experiment-runtime/v14-stageB-sobol-512-results/triage.json`.
<!--derived-->
- Focused runner, campaign, and triage suite: 19 passed, 2 GPU tests skipped.
- Broad Stage B and SNr kernel suite: 224 passed, 3 skipped.
- RTX 3090 V1 two-candidate smoke: passed in 4.17 s test time.
- RTX 3090 V3 two-candidate same-cell smoke: passed in 10.26 s test time.
- Historical V1 campaign replay was exactly equal to its stored triage artifact;
  both self digests were
  `f78ada09406b967176602e2ac25c4a696da305cfeae5c5a8a13f60bcb6d25136`.

## Boundary and next action

The recurrent simulation still launches one fused step megakernel per 0.05 ms
timestep, so very small batches underuse the RTX 3090. The fresh campaign should
use the widest validated candidate batch that fits the trace and scratch-memory
budget, benchmark throughput before full execution, and retain CPU confirmation.

Preregister a fresh candidate search under the V3 protocol. Do not reopen or
retune candidates 284 or 404. Keep the heterogeneous SK cohort unavailable.
