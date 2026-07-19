# Direction 7 PARALLEL multi-process launch: NEGATIVE — zero speedup on Windows CUDA WDDM; binding bottleneck is GPU compute not VRAM

**Date:** 2026-05-27 ~18:10 EDT
**Status:** Empirical NEGATIVE finding (no architectural science at stake; pure performance measurement). Sequential D7 production restarted (PID 30216 at 18:08:40).

## Hypothesis tested

Sequential D7 V=320 production had a measured ETA of ~50-55 hr (longer than initially predicted 27-32 hr; the events-per-word 4x multiplier dominates wall scaling). nvidia-smi showed only 8/24 GB VRAM used by the single sequential process. Hypothesis: launching 5 parallel python processes (one per bridge, all 3 seeds) would let multiple CuPy contexts share the GPU concurrently, exploiting the unused 16 GB VRAM headroom for a 3-5x throughput gain.

## Measurement

Killed the sequential run at 16:08 (after ~31 min; zero production cells completed yet). Launched 5 parallel processes via `direction_7_parallel_production.ps1` at 16:09. Five CuPy contexts initialized cleanly; total VRAM 5.0 GB / 24 GB; GPU util 36% (well below saturation).

At 116 min into the parallel run (~18:06), 4 of 5 bridges had logged their first 10% per-event progress line:
- A_nouns: 1280/12800 events in 116.4 min
- B_verbs: 1280/12800 events in 116.6 min
- D_spatial: 1280/12800 events in 116.0 min
- E_functional: 1280/12800 events in 116.0 min
- C_adj: still in setup (slightly slower init)

Linear extrapolation: ~1165 min per cell at 5-way contention, vs ~210 min per cell sequential (smoke timing-anchored: smoke at V=64 was 28.5 min/cell with 1/8x events; production scales to ~228 min/cell sequential).

| mode | per-cell wall (predicted) | concurrent cells | per-process cells | total wall |
|---|---|---|---|---|
| sequential | ~210 min | 1 | 15 | ~52 hr |
| 5-way parallel | ~1165 min (5x slower per-cell) | 5 | 3 | ~58 hr |

5-way parallel is essentially the same throughput as sequential — actually slightly worse due to CUDA context-switch overhead.

## Diagnosis

On Windows + RTX 3090 with the WDDM driver, multiple CuPy contexts on the same GPU are time-sliced almost perfectly (each process gets ~1/N of compute). The GPU is compute-bound on these sparse-matrix kernels; VRAM headroom is irrelevant because adding more memory-resident state doesn't unlock more compute.

CUDA MPS (Multi-Process Service), which DOES allow genuine concurrent kernel execution across processes, requires the TCC (Tesla Compute Cluster) driver, which is only available on datacenter cards (A100/H100) or via Linux. On Windows + WDDM consumer cards, this is structurally not available.

The "unused VRAM" intuition is misleading on consumer Windows GPU drivers. VRAM headroom indicates room for bigger single-process state (more neurons, more synapses), not for multi-process parallelism.

## Sequential restarted

Killed the 5 parallel processes at 18:08 (~5 hr of GPU time spent but zero production cells completed; ~5 GPU-hr wasted on this measurement). Restarted sequential D7 production (PID 30216) at 18:08:40. ETA ~50-55 hr → expected completion **2026-05-29 ~22:00 EDT**.

## What WOULD speed up future runs (ranked by expected payoff vs implementation cost)

**1. fp16 eligibility traces** (`cfg.fp16_synapse_state = True`). CLAUDE.md documents this is validated at <1mV voltage drift over 1000 steps; the D7 runner doesn't currently set it. Estimated 10-25% speedup from halved memory bandwidth on the STDP hot path. **Cheap** (one line config change) and **safe** (validated).

**2. Reduce stim_steps per training event** (currently 50 sub-steps = 25 ms simulated per event; could try 25). Estimated 1.5-2x speedup if training quality preserved. **Cheap** to test on smoke before committing to production. Risk: STDP signal weakens, leading to PARTIAL instead of PASS at higher V.

**3. Linux + CUDA MPS migration**. The 5-way parallel launcher (kept as `direction_7_parallel_production.ps1`) would actually deliver 3-5x speedup on Linux with MPS enabled. **Medium effort** (migration of one machine; the codebase itself is platform-agnostic via SIM_BACKEND env). On a Linux box with the same RTX 3090, sequential D7 production would be ~50 hr; with MPS 5-way parallel it'd be ~10-15 hr.

**4. Consolidated multi-category bridge**. Instead of 5 separate bridges, build ONE bridge holding all 5 categories' pools and train with category-interleaved schedule. Single GPU process keeps GPU saturated AND eliminates per-cell init overhead. Estimated **3-5x speedup**. **High effort** (requires architectural rework + new anti-cheat mechanism to replace bridge-seed-offsets that pillars n=108/n=109 depend on for cross-bridge distinctness). Needs design + adversarial review before science.

**5. Datacenter GPU upgrade** (H100/A100 with TCC driver) — orthogonal to the rest; would enable Windows MPS but is hardware spend.

## Discipline

Pure performance finding; no architectural science at stake. No bar, no verdict module. No protected/frozen/moat module touched. Sequential restart preserves the pillar n=110 scientific path unchanged.

The empirical caveat is recorded in the launcher script (`direction_7_parallel_production.ps1` lines 25-49) so future runs (D8 V=640, D9 V=1280) don't repeat the same wasted experiment on Windows.

## Files

- Launcher (now with CAVEAT): `research/findings/raw/direction_7_parallel_production.ps1`
- Parallel logs (5 processes, ~116 min each before kill): `research/findings/raw/direction_7_parallel_logs/{A_nouns,B_verbs,C_adj,D_spatial,E_functional}.log`
- Orchestrator log: `research/findings/raw/direction_7_parallel_orchestrator.log`
- Sequential restart log (in flight): `research/findings/raw/direction_7_5bridge_production.log`
