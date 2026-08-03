---
type: finding
status: negative
date: 2026-08-03
mechanism: visual-identity-spike-latency-selection
runner: research/runners/_laneD_visual_identity_homeostasis_gate.py
artifacts:
  - research/findings/raw/parallel_gates/visual_identity_spike_latency_calibration_seed212.json
  - research/findings/raw/parallel_gates/visual_identity_spike_latency_calibration_seed212.json.prov.json
  - research/findings/raw/parallel_gates/visual_identity_spike_latency_calibration_seed213.json
  - research/findings/raw/parallel_gates/visual_identity_spike_latency_calibration_seed213.json.prov.json
---

# Spike latency removes visual top-k selection but identity remains unstable

<!--derived-->
**Verdict: NO-GO at calibration.** First-spike timing now chooses every pooler
winner during learning and inference, replacing host top-k drive selection.
The selector passed its causal controls, but the learned codes did not identify
objects reliably across new views. Development and held-out seeds remain locked.

## Question

Can the existing Gabor/V1-complex front end, trace learning, and slow usage
homeostasis learn a stable identity for four objects viewed continuously across
position, scale, and lighting changes when neural spike latency, rather than
host ranking, selects the active columns?

Object labels were used only after inference for scoring. They did not enter
pooler learning, current generation, or winner selection.

## Result

Artifacts:
`research/findings/raw/parallel_gates/visual_identity_spike_latency_calibration_seed212.json`
and
`research/findings/raw/parallel_gates/visual_identity_spike_latency_calibration_seed213.json`.
Both cluster runs used clean source commit `ea3f51654`, the NumPy backend, and
the same revision-addressed source manifest. All five artifact-validity
preconditions passed.

| seed | intact identity decode | chance | identity margin | temporal shuffle | trace lesion | homeostasis lesion | pixel scramble | result |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 212 | 0.3125 | 0.2500 | +0.0186 | 0.4062 | 0.3125 | 0.3125 | 0.4062 | no-go |
| 213 | 0.1875 | 0.2500 | -0.0073 | 0.3438 | 0.3438 | 0.2812 | 0.2188 | no-go |

<!--derived-->
The neural-selection controls were clean. First-spike winners overlapped the
graded-drive reference by 1.000 on seed 212 and 0.917 on seed 213. Flattening
the drive reduced overlap with the original winners to 0.083 and 0.000.
Lesioning neural current produced zero winners. Removing the fast-spiking
inhibitory path preserved the earliest winners but increased the number of
columns that eventually fired from 21 to 176 and from 27 to 149.

Slow homeostasis reduced usage imbalance: intact usage CV was 0.905 and 0.881,
versus 2.530 and 2.537 without homeostasis. <!--derived--> That regulation did
not create stable identity. Temporal shuffling, trace lesion, homeostasis
lesion, and pixel scrambling often matched or exceeded intact decode. The
learning mechanism therefore has not been shown to use temporal continuity or
the preserved image structure to form invariant object codes.

## Decision

Keep the spike-latency selector as a scaffold reduction, but do not promote the
visual identity mechanism and do not open seeds 214, 215, 310, 311, 312, or
313. The next attempt should improve the representation or local learning rule
that feeds the selector, with temporal-order and pixel-structure controls still
required. Further tuning of the selector itself is low value because its
causal physiology already works while identity does not.

Remaining scaffolds are explicit: host overlap-to-current normalization, host
first-spike readout and same-step tie handling, and host V1 feature
sparsification. These must be removed in later rungs even if identity learning
eventually passes.
