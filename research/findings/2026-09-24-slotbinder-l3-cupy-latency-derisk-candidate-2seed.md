---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
mechanism: SlotBinderComposer (coincidence binding, rung L3) per-query latency on the GPU (SIM_BACKEND=cupy)
seeds: [42, 43]
prereg: research/findings/2026-09-24-slotbinder-l3-cupy-latency-derisk-PREREG.md
artifacts:
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/summary_f32.json
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s42.json
  - research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s43.json
verdict: De-risk (2 seeds) -- CANDIDATE for wire-in under the pre-registered criterion. Mean per-query latency 1.04 s and 1.11 s
  (at most 2.0 s required), recall 1.0, moat and mismatch probes pass on both seeds. Not a GO: the wire-in needs its own 6-seed gate.
---

# SlotBinder L3 on the GPU: about 1 s per query, a wire-in candidate (de-risk, 2 seeds)

## Result

Artifacts: `research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s42.json`, `..._s43.json` and
`research/findings/raw/_slotbinder_l3_latency_derisk_cupy/summary_f32.json` (SIM_BACKEND=cupy, one RTX 3090, fanout 32).

<!--derived-->
| seed | mean s/query | max s/query | recall | moat | mismatch | build + store (s) | ratio to FHRR |
|---|---|---|---|---|---|---|---|
| 42 | 1.04 | 1.39 | 1.0 | pass | pass | 50.3 | 1.15 |
| 43 | 1.11 | 1.55 | 1.0 | pass | pass | 44.0 | 1.23 |

The same measurement on numpy CPU read a mean of 68.5 s per query across six seeds
(`research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md`), so the GPU path removes the
per-step cost wall that made L3 a NO-GO. The spiking binder runs within about 1.2x of the host FHRR composer's latency here.

## What it means

The recall composer's exact-inverse bind/unbind is host arithmetic (an open item in tonight's crutch register). This measurement
makes the spiking replacement latency-viable on the consumer reference GPU. Next rung: the SlotBinder wire-in with a 6-seed gate
on recall, moat, mismatch and latency, at the production fact scale.

## Honesty

Functional read-outs only.
