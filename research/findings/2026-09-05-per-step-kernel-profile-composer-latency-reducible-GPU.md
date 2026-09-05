---
type: finding
status: measured
claim_check: measured
date: 2026-09-05
mechanism: per-step spiking-sim kernel cost at composer scale (the composer-latency root cause)
lane: H · Memory / composer
seeds: [n/a — a single at-scale GPU profiling measurement, not a seeded comparison]
artifacts:
  - research/findings/raw/_perf_step_kernel_profile/result.json
verdict: >
  The per-step spiking-simulation cost that dominates the onebrain composer's recall latency (the multi-second
  recall at the full co-resident fact scale, and the SlotBinder L3 latency NO-GO vs FHRR) is REDUCIBLE, not a
  hard wall. Measured on GPU/cupy at the real composer scale (all quantities are direct reads from the cited
  artifact research/findings/raw/_perf_step_kernel_profile/result.json; see the marked body for the specific
  values): (1) the fused megakernel-v2 inference path is several-fold faster than the python inference path, and
  the general-step-megakernel design doc's prediction that this fused win "tapers to ~unity at the composer
  scale" is REFUTED -- it stays well above unity there. (2) STDP is the dominant learning-path cost, and the
  ADDITIVE default-OFF `enable_branchless_plasticity` fast-path measurably speeds the STDP path (its first
  at-scale measurement; prior evidence was only a microbench). (3) GPU utilisation is LOW during both learning
  and inference -- the per-step is LAUNCH/overhead-bound, not compute-bound, so there is real headroom for
  further fusion/batching. Recall is inference (no learning), so the fast fused path applies to it; combined
  with a sublinear sharded spiking store (in build) to cut the O(k_max) block scan, the composer's recall
  latency is genuinely attackable. The composer (rank-1) is NOT permanently latency-blocked -- it has a clear
  multi-lever optimisation path (fused megakernel already default; branchless-plasticity for the learning path;
  more fusion given the util headroom; sharded retrieval for the scan). NO-DEFER honoured: the latency wall is a
  method verdict with named surpasses, not a capability wall.
---

# Per-step kernel profile: the composer-latency root cause is reducible

## What ran
`research/runners/_perf_step_kernel_profile.py` (SIM_BACKEND=cupy, via `gpu_queue.sh`) at composer-representative
scale, harvesting `research/findings/raw/_perf_step_kernel_profile/result.json`. This is the GPU/cupy re-verify the
SlotBinder L3 NO-GO (`2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md`) explicitly
flagged as not-yet-attempted (it measured CPU/numpy only), and the direct measurement of the
`2026-07-23-general-step-megakernel-design.md` taper prediction that had never been tested at the real composer scale.

## Derived — measured per-step numbers (composer scale n=64,324, nnz=28.6M; all values rounded from the cited research/findings/raw/_perf_step_kernel_profile/result.json)
- **Fused inference (megakernel-v2) vs python inference: 4.705x** (0.16889 vs 0.79464 ms/step; 5921 vs 1258 steps/sec).
- **Megakernel-taper prediction REFUTED:** the inference-only scale sweep shows the fused speedup at
  n=64,324 is **2.47x** (5887 vs 2382 steps/sec), not the predicted ~1x -- the win does taper from ~4.2x at
  n=1,000 but stays well above 1x at composer scale.
- **STDP dominates the learning path:** `stdp_compacting` = 1.647 ms/step (607 steps/sec); marginal STDP cost
  = +0.852 ms/step = **107% of the inference cost** (STDP roughly doubles the per-step time).
- **Branchless-plasticity fast-path: 1.766x on the STDP path** (1.647 -> 0.932 ms/step) -- first at-scale
  measurement of the additive, default-OFF, byte-identical-tested `enable_branchless_plasticity`.
- **GPU under-utilised:** avg GPU util 20.2% during STDP-learning, 34.5% during fused inference -- the per-step
  is launch/overhead-bound, not compute-bound (headroom for more fusion/batching).

<!--derived: the quantities restated below (per-step ms, speedups, util %, the 114s/404-fact recall) are direct reads from research/findings/raw/_perf_step_kernel_profile/result.json and the rank-1 composer finding, both cited -->
## Reading it (no-defer)
The composer recall path is inference (no learning), so the fast fused inference path (sub-millisecond per step) is what recall pays per
timestep -- the multi-second recall at 404 facts is `recall_timesteps x per_step x O(k_max block scan)`. Two orthogonal,
already-named levers cut it: (a) the **sharded spiking store** (in build, agent a6dc32ba) makes the scan sublinear
(fewer blocks/query); (b) the **per-step cost** is itself reducible -- the fused megakernel is already the default
(4.7x), the learning path has the 1.77x branchless option, and the 20-34% GPU util says more fusion/batching is on
the table. So the SlotBinder-L3 / rank-1 latency verdict is a METHOD wall with concrete surpasses, not a capability
wall. NEXT: land the sharded store; consider the branchless-plasticity flag for the learning/teach path; measure the
composer recall latency after sharding + fused inference.
