---
type: preregistration
status: preregistered
date: 2026-09-24
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
mechanism: SlotBinderComposer (coincidence binding, rung L3) per-query latency on the GPU (SIM_BACKEND=cupy)
seeds: [42, 43]
verdict: PRE-REGISTRATION only. No cupy result exists yet. This is a 2-seed DE-RISK, never a GO.
---

# SlotBinder L3 per-query latency on the GPU: pre-registration (de-risk, 2 seeds)

The L3 wire-in de-risk read NO-GO on numpy CPU because per-step cost dominated latency: mean 68.5 s per query across six
seeds against the FHRR reference's 0.9 s (`research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md`).
<!--derived--> That run never measured the GPU path. The midnight plan (step G2) asks whether the GPU removes the wall.

## Run (fixed)

```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._slotbinder_l3_latency_derisk --seeds 42 43 \
    --out-dir research/findings/raw/_slotbinder_l3_latency_derisk_cupy
```
Queued through `tools/gpu_queue.sh` (one brain-loading GPU process at a time). The committed numpy artifacts in
`research/findings/raw/_slotbinder_l3_latency_derisk/` are not touched.

## Criterion (written before any cupy result)

- **candidate for wire-in:** `mean_query_latency_s` <= 2.0 on BOTH seeds, with `recall_accuracy_query_patient` 1.0,
  `moat_pass` and `mismatch_pass` true on both.
- **not yet:** latency above 2.0 s on either seed, or any correctness field false.
- The status is always "de-risk (2 seeds)". A wire-in needs its own 6-seed gate.

## AMENDMENT 1 (2026-09-24, after seeds 42/43 read candidate; criterion unchanged)

Seeds 44, 100, 101 and 102 are run with the identical command and criterion into the same directory, so the rung is
measured on all six project seeds. A six-seed pass (every seed at most 2.0 s mean per query with recall 1.0, moat and
mismatch passing) is reported as "L3 GPU latency GO 6/6" for this measurement only. The SlotBinder wire-in into the
production composer path still needs its own gate.

```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._slotbinder_l3_latency_derisk --seeds 44 100 101 102 \
    --out-dir research/findings/raw/_slotbinder_l3_latency_derisk_cupy
```

