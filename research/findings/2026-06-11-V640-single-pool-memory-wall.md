# V=640 single-pool: OOM at the synapse install — the single-pool memory wall is ~V=320–450 (lower than estimated); single-pool scaling demonstration completes at V=160→V=320

**Date:** 2026-06-11. **Runner:** `research/runners/learned_graded_embedding_homeostasis_probe.py --n-clusters 80 --per-cluster 7 --n-pool 24000` (V=640 at the clean 37.5 neurons/concept density). **Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090, 24 GB). **Raw:** `research/findings/raw/_lge_v640_seed42.log`.

> **Finding: clean-density V=640 is INFEASIBLE on the 3090 — it OOMs at the synapse install, before any compute.** `inject_explicit_wiring` → `cp.asarray(pre_np)` → `cudaErrorMemoryAllocation: out of memory` (the host→device transfer of the ~354M-synapse arrays exhausts the pinned-memory pool). The learned recurrent's synapses scale quadratically with pool size (V=160 pool 7000 = 30.7M; V=320 pool 12000 = 88.6M; V=640 pool 24000 = **354M**), and 354M is past what the install can transfer. So the **single-pool memory wall is ~V=320–450, not the V=640–800 the build plan estimated** — V=320 (88.6M synapses, ~13 GB) is near the practical single-pool ceiling.

## Consequence for the scaling story
- **The single-pool scaling demonstration completes at V=160 → V=320**, both near/above the host ceiling and *improving* with scale (Oja: +0.977 → +0.991; host ceilings +0.929 → +0.959). That is strong evidence the recipe scales across the single-pool range.
- A *feasible* V=640 would require dropping the pool below the 37.5 neurons/concept density that the curve used (e.g. n_pool=14000 = 22/concept), which **confounds** the scaling curve (a sag could be density, not scale). So single-pool cannot cleanly extend past ~V=320–450 on this GPU; chasing a degraded V=640 is not worth the ~5–6 hr + the confound.
- **The real path to large vocabulary is multi-bridge sparse-distributed** (the project's existing 320-tier method: 5 bridges × 64). The OOM *confirms* the build plan's premise that single-pool dead-ends near the production tier and multi-bridge is the production scaling route. The dual/CLS recipe is per-bridge (64 concepts ≪ V=160, where it is GO), so it composes; the open work is cross-bridge composition + semantic-cluster sharding (build plan §"Scaling path").

## What this does NOT change
- The recipe is validated single-pool to V=320 (production tier), multi-seed-proven mechanism (3/3, commit f0800378), near-ceiling, store-volume-robust, Oja-default.
- The honest correction is only to the single-pool *reach* (V=320, not V=640) — not to the recipe.

## Next (overnight)
1. **Pre-build multi-seed confirmation at V=320** (the production single-pool scale) — seeds 43,44 at the validated config; low-risk, retires the multi-seed-at-production-scale question before the build.
2. **Design the multi-bridge de-risk** (read-only, the standing opening move for the real large-vocab frontier) — how to test the dual/CLS recipe per-bridge + cross-bridge composition cheaply, reusing `g20_multibridge --sparse`.

**No banking** — the OOM is reported as the honest single-pool memory-wall finding; it sharpens the strategy (multi-bridge is the large-vocab path) rather than blocking it.
