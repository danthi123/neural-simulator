"""Route C throughput measurement probe.

Compare per-step wall-clock and GPU utilization for two G9 reservoir sizes:
  - Small (200 hidden neurons, the historical baseline)
  - Large (5000 hidden neurons, Route C)

The point: at 200 neurons, the GPU is dispatch-bound (~2.4 ms/step, mostly
Python dispatch overhead). At 5000 neurons, we expect actual GPU compute to
dominate, so per-step time grows sublinearly with neuron count and GPU
utilization rises.

Runs each size for 100 sim steps after warmup. Reports ms/step + total
wall time. Doesn't run a full episode (no reward shaping, no learning) —
just synaptic propagation + neuron dynamics.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _measure(n_hidden_exc: int, n_hidden_inh: int,
              hidden_to_hidden_density: float,
              input_to_hidden_density: float,
              n_warmup: int = 30,
              n_steps: int = 100) -> dict:
    import cupy as cp
    import numpy as np

    from sim import (SimulationBridge, VisualizationConfig,
                       RuntimeState, GPUConfig)
    from research.runners.g9_runner import _build_g9_plan

    core_cfg, plan = _build_g9_plan(
        seed=42,
        n_hidden_exc=n_hidden_exc,
        n_hidden_inh=n_hidden_inh,
        hidden_to_hidden_density=hidden_to_hidden_density,
        input_to_hidden_density=input_to_hidden_density,
    )
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    layout = plan["layout"]
    new_traits = np.zeros(core_cfg.num_neurons, dtype=np.int32)
    for i in layout["hidden_inh_idx"]:
        new_traits[i] = 1
    bridge.cp_traits = cp.asarray(new_traits)
    bridge._cached_inhibitory_mask = None
    bridge.inject_explicit_wiring(plan, output_inhibitory_indices=None)

    n_synapses = int(bridge.cp_connections.nnz)
    n_total_neurons = core_cfg.num_neurons

    # Warmup
    for _ in range(n_warmup):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Sync GPU and time
    cp.cuda.Device().synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    cp.cuda.Device().synchronize()
    elapsed = time.time() - t0
    ms_per_step = elapsed * 1000.0 / n_steps

    # Memory used (CuPy mempool)
    mempool = cp.get_default_memory_pool()
    used_bytes = mempool.used_bytes()

    bridge.clear_simulation_state_and_gpu_memory()
    return {
        "n_neurons": n_total_neurons,
        "n_synapses": n_synapses,
        "ms_per_step": ms_per_step,
        "elapsed_total_sec": elapsed,
        "vram_used_gb": used_bytes / 1024**3,
    }


def main():
    print("\n=== Route C throughput probe ===\n")

    # Small (G9 historical baseline)
    print("Small (200 hidden, density 0.1)...")
    small = _measure(
        n_hidden_exc=160, n_hidden_inh=40,
        hidden_to_hidden_density=0.1, input_to_hidden_density=0.5,
    )
    print(f"  neurons: {small['n_neurons']:,}  synapses: {small['n_synapses']:,}")
    print(f"  per-step: {small['ms_per_step']:.3f} ms")
    print(f"  vram used: {small['vram_used_gb']:.2f} GB")

    # Large (Route C, ~5k hidden)
    print("\nLarge (5000 hidden, density 0.05)...")
    large = _measure(
        n_hidden_exc=4000, n_hidden_inh=1000,
        hidden_to_hidden_density=0.05, input_to_hidden_density=0.1,
    )
    print(f"  neurons: {large['n_neurons']:,}  synapses: {large['n_synapses']:,}")
    print(f"  per-step: {large['ms_per_step']:.3f} ms")
    print(f"  vram used: {large['vram_used_gb']:.2f} GB")

    # Comparison
    print(f"\n=== Comparison ===")
    print(f"  Network size scale-up: {large['n_neurons'] / small['n_neurons']:.1f}x neurons, "
          f"{large['n_synapses'] / small['n_synapses']:.1f}x synapses")
    print(f"  Per-step time scale-up: {large['ms_per_step'] / small['ms_per_step']:.2f}x")
    print(f"  Effective compute throughput ratio (synapse-updates/sec):")
    small_throughput = small['n_synapses'] / (small['ms_per_step'] / 1000)
    large_throughput = large['n_synapses'] / (large['ms_per_step'] / 1000)
    print(f"    small: {small_throughput:,.0f} syn-updates/sec")
    print(f"    large: {large_throughput:,.0f} syn-updates/sec")
    print(f"    speedup: {large_throughput / small_throughput:.1f}x more synapses processed per second")
    print(f"  VRAM usage: small={small['vram_used_gb']:.2f} GB, large={large['vram_used_gb']:.2f} GB")


if __name__ == "__main__":
    main()
