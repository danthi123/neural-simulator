"""Quick GPU parallelism benchmark.

Runs N steps of a representative bridge on the GPU, prints wall clock.
Used to test how multiple instances share the GPU when launched
concurrently in separate Python processes.

Usage (from shell, launching multiple in parallel):
    python -m research.runners.bench_parallel_gpu --steps 3000 --tag p1 &
    python -m research.runners.bench_parallel_gpu --steps 3000 --tag p2 &
    wait

Compare wall clock from each tag. If 2 processes both finish in ~1.3x
single-process time, parallelism is good (40% slower each but 2 done
in 1.3x = 1.5x effective speedup).
"""

import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tag", type=str, default="p1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_visual_cortex=True,
        enable_text_io=True,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = args.seed
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 5.0

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    t_init = time.time() - t0
    print(f"[{args.tag}] init: {t_init:.1f}s", flush=True)

    # Warmup: 100 steps to compile fused kernels
    for _ in range(100):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    cp.cuda.Stream.null.synchronize()

    t1 = time.time()
    for _ in range(args.steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    cp.cuda.Stream.null.synchronize()
    dt = time.time() - t1

    rate = args.steps / dt
    print(f"[{args.tag}] {args.steps} sub-steps in {dt:.1f}s = {rate:.0f} sub-steps/sec",
          flush=True)


if __name__ == "__main__":
    main()
