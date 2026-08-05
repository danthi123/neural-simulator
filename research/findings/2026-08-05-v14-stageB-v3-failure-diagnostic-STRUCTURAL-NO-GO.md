---
type: finding
status: no-go
date: 2026-08-05
mechanism: v14-snr-stageB-single-compartment-intrinsic-packet
lane: executed-action-credit
promotion_value: none
instrument: RTX 3090 CuPy authenticated failure diagnostic
artifacts:
  - research/findings/raw/v14_stageB_v3_failure_diagnostic_r2_receipt.json
  - research/findings/raw/v14_stageB_v3_failure_diagnostic_r2_analysis.json
  - research/specs/v14_snr_stageB_failure_diagnostic_v1.json
---

# Stage B V3 failure diagnostic: structural engineering NO-GO

## Decision

Retire parameter search on the current single-compartment SNr packet. Do not
widen its ranges or reopen screened candidates. The next Stage B build must
first add the missing fast-spike kinetics, compartment coupling, and local
Cav2.2-to-SK calcium pathway, then pass waveform and causal-lesion gates before
another population search.

This is an engineering verdict on the current model structure. It is not a
negative claim about biological SNr pacemaking, delayed action credit, or the
project's whole-brain goal.

## Authenticated experiment

Artifacts: `research/findings/raw/v14_stageB_v3_failure_diagnostic_r2_receipt.json`
and `research/findings/raw/v14_stageB_v3_failure_diagnostic_r2_analysis.json`.

The preregistered diagnostic selected the exact nine V3 candidates that failed
only the NaP voltage-direction criterion. Each candidate ran from a fresh,
identical bridge in four fixed rescue arms (`0`, `-10`, `-20`, and `-30 pA`):
`2 s` intact baseline, `1 s` NaP lesion, `0.5 s` pulse, and `1 s` release.
No result could tune parameters or promote a candidate.

The completed CuPy receipt binds 36 multichannel traces totaling 166,024,410
bytes. <!--derived--> Current-decomposition closure remained below
`7.63e-06 uA/cm2` for the SNr packet and `8.78e-05 uA/cm2` for the membrane
update. The deterministic NumPy analyzer authenticated the receipt and every
trace before calculating summaries.

## Results

- All nine intact baselines fired at `48-66 spikes/s`; all nine stopped after
  NaP removal. The runner's explicit attribution record assigns `100%` of the
  measured cohort-median sustained firing difference to NaP presence in this
  model. <!--derived--> This is model attribution, not a claim that biological
  SNr firing is exclusively NaP-driven.
- Calcium declined in all nine. Six also showed a robust SK-current collapse,
  while three retained `115-123%` of baseline SK current with saturated SK
  activation despite their calcium decline. <!--derived--> A single stable
  calcium-to-SK failure chain therefore does not explain the cohort.
- A `-30 pA` pulse increased fast-sodium availability in all nine models, but
  only candidate `0961` emitted any additional release spike, and it emitted
  one. No candidate regained sustained firing. <!--derived-->
- The rescue arms were bit-identical before pulse for all nine candidates, so
  the pulse comparison itself is controlled.

Neither preregistered complete explanation is cohort-stable. The packet can
remove NaP exactly and can recover its simplified fast-sodium availability,
yet almost never recovers autonomous spiking. At the same time, its calcium-SK
relationship splits into incompatible regimes. More search within the same
equations would optimize around those structural omissions rather than test
the biological mechanism.

## Limitation

Version 1 included a zero-pulse lesion control but no intact-NaP continuation
matched to spike phase. Immediate post-lesion voltage direction is therefore
confounded by where the lesion interrupts the spike cycle and cannot establish
a causal voltage sign. The analyzer records that boundary explicitly. Any
successor making a causal claim about immediate voltage must preregister an
intact continuation or phase-matched sham.

## Source-bound successor

The successor must preserve measured SNr sodium constraints from
[Ding, Wei, and Zhou 2011](https://doi.org/10.1152/jn.00305.2011), including <!--derived-->
fast-Na activation/inactivation and its `0.59/35.1 ms` recovery components.
It must represent the high-rate repolarizing current with a Kv3-like gate,
instead of using the current packet's generic fast-K approximation. It must
also separate soma/proximal-dendrite voltage through axial coupling and keep a
local Cav2.2-SK calcium signal distinct from bulk calcium, consistent with the
causal AHP evidence in
[Atherton and Bevan 2005](https://doi.org/10.1523/JNEUROSCI.1475-05.2005) and <!--derived-->
[Yanovsky et al. 2006](https://doi.org/10.1113/jphysiol.2006.117622). <!--derived-->
Source synthesis and evidence boundaries are recorded in
`research/findings/2026-08-04-gpi-snr-autonomous-pacemaking-biophysical-fallback-RESEARCH.md`.

Before calibration, the new equations must pass current closure, voltage-clamp
activation/inactivation/recovery, action-potential width and AHP waveform,
pause/release, timestep convergence, NumPy/CuPy parity, and consumer-GPU
throughput checks. Only then may a bounded multi-objective search fit intact
and lesion phenotypes. Held-out learning seeds remain sealed.
