"""Profile a single sub-step's time breakdown using the built-in
GPUConfig.enable_step_profiler infrastructure.

Runs 1500 steps with the profiler enabled. The bridge logs an avg/step
summary every 500 steps. Output shows which phase of the step takes
the most wall-clock time:

  t_init  — setup at start of step (cp_prev_firing.any(), etc.)
  t_stp   — STP update + effective_synaptic_strength
  t_syn   — synaptic conductance update + currents
  t_dyn   — neuron dynamics (Izh/HH/AdEx)
  t_plast — STDP + Hebbian + eligibility traces
  t_homeo — homeostasis
  t_final — rest (recording, viz, neuromodulator step, etc.)

Each section is wrapped with cp.cuda.Device().synchronize() so the
times include GPU compute, not just kernel launches. This gives the
true "where does the wall clock go" answer.

Usage:
    python -m research.runners.profile_step [--steps 1500]

The output of three avg/step lines (one per 500 steps) is printed
to stdout. Use to validate the "launch overhead is the bottleneck"
hypothesis: if t_syn + t_dyn + t_plast = ~70% of total, GPU compute
dominates. If they sum to <30%, Python orchestration dominates.
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", choices=["v2", "minimal"], default="v2",
                    help="Architecture to profile. v2 = 5234 neurons "
                    "(text I/O, full cascade). minimal = ~356 neurons "
                    "(language_input + motor_X only, cascade-free; "
                    "matches the biology sweep target arch).")
    args = ap.parse_args()

    t0 = time.time()
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    if args.arch == "minimal":
        from research.runners.text_minimal_isolation import build_minimal_brain_regions
        print("Building MINIMAL architecture (cascade-free, ~356 neurons)...")
        regions, pathways = build_minimal_brain_regions()
    else:
        from research.runners.g11_bg_runner import build_bg_brain_regions
        print("Building v2 text-IO architecture (5234 neurons, ~175k synapses)...")
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
    cfg.dt_ms = 0.5 if args.arch == "v2" else 1.0  # match production
    cfg.seed = args.seed
    cfg.enable_nmda = (args.arch == "v2")  # minimal arch doesn't use NMDA
    cfg.nmda_ratio = 0.5
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 5.0

    gpu_cfg = GPUConfig()
    gpu_cfg.enable_step_profiler = True   # <-- the magic flag

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=gpu_cfg,
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    print(f"Init: {time.time() - t0:.1f}s. Warmup 200 steps to compile fused kernels...")

    # Warmup with profiler off so we don't include kernel-compile time
    bridge.gpu_config.enable_step_profiler = False
    for _ in range(200):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    cp.cuda.Stream.null.synchronize()

    # Now profile
    bridge.gpu_config.enable_step_profiler = True
    print(f"Profiling {args.steps} steps. PROFILER lines below show per-phase ms.")
    print("=" * 70)
    t1 = time.time()
    for _ in range(args.steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    cp.cuda.Stream.null.synchronize()
    dt = time.time() - t1

    rate = args.steps / dt
    print("=" * 70)
    print(f"\nTotal: {args.steps} steps in {dt:.1f}s = {rate:.0f} sub-steps/sec")
    print(f"Avg/step (with sync overhead from profiler): {dt*1000/args.steps:.2f}ms")


if __name__ == "__main__":
    main()
