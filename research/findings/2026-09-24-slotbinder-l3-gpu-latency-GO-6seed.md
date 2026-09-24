---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
mechanism: SlotBinderComposer (coincidence binding, rung L3) per-query latency on the GPU (SIM_BACKEND=cupy)
seeds: [42, 43, 44, 100, 101, 102]
instrument: control comparison on every seed -- the host FHRR composer as the reference arm (vs_fhrr_mean_ratio 1.08-1.23), the
  identical runner on numpy as the backend control (68.5 s/query), and the moat and mismatch probes as correctness controls
prereg: research/findings/2026-09-24-slotbinder-l3-cupy-latency-derisk-PREREG.md
artifacts:
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s42.json
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s43.json
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s44.json
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s100.json
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s101.json
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s102.json
verdict: L3 GPU latency GO 6/6 (AMENDMENT 1). Mean per-query latency 0.97-1.11 s on every seed (at most 2.0 s required), recall
  1.0, moat and mismatch probes pass on all six. The numpy L3 NO-GO (68.5 s/query) does not hold on the GPU. The SlotBinder wire-in
  into the production composer path still needs its own gate.
---

# SlotBinder L3 on the GPU: about 1 s per query on all six seeds (GO for this measurement)

Per-seed artifacts: `research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s42.json` and the five sibling files
(SIM_BACKEND=cupy, one RTX 3090, fanout 32). Seeds 42/43 ran first (the 2-seed de-risk finding); seeds 44/100/101/102 ran under
AMENDMENT 1 with the identical command and criterion.

<!--derived-->
| seed | mean s/query | max s/query | recall | moat | mismatch | ratio to FHRR |
|---|---|---|---|---|---|---|
| 42 | 1.04 | 1.39 | 1.0 | pass | pass | 1.15 |
| 43 | 1.11 | 1.55 | 1.0 | pass | pass | 1.23 |
| 44 | 1.02 | 1.33 | 1.0 | pass | pass | 1.13 |
| 100 | 0.97 | 1.26 | 1.0 | pass | pass | 1.08 |
| 101 | 1.01 | 1.40 | 1.0 | pass | pass | 1.12 |
| 102 | 1.05 | 1.40 | 1.0 | pass | pass | 1.16 |

## What it means

The spiking coincidence binder answers within about 1.1x of the host FHRR composer on the consumer reference GPU, with the same
correctness probes passing. The host exact-inverse bind/unbind in the recall composer (an open item in tonight's crutch register)
now has a latency-viable spiking replacement. Next rung: wire the SlotBinder into the production composer path at production fact
scale, with a 6-seed gate on recall, moat, mismatch and per-turn latency.

## Honesty

Functional read-outs only.
