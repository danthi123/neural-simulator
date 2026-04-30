"""g11 batched-replica smoke (POC, 2026-04-29).

Proves the existing batched-replica framework (sim/replicas.py, ported in
commit f1497e0) can host the cluster A + F brain stack from g11_bg_runner.

Scope:
- Build single-replica regions+pathways via build_bg_brain_regions(...)
- Replicate to N=2 blocks via replicate_wiring_plan_with_seeds
- Boot a SimulationBridge sized for 2 replicas
- Inject the replicated plan
- Run 10 simulation steps
- Verify no crashes; per-replica region indices accessible.

This is NOT a full-runner retrofit — see
docs/plans/2026-04-29-g11-batched-replica-retrofit.md for the proper
g11_bg_runner.py retrofit (4-6 hours of focused work). This POC just
proves the bridge-side machinery works for the current cluster topology.

Usage:
    python -m research.runners.g11_replicated_smoke
"""
from __future__ import annotations

import sys


def main() -> int:
    import cupy as cp
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState,
        GPUConfig,
    )
    from sim.regions import RegionManager
    from sim.replicas import (
        ReplicaConfig, ReplicaManager, replicate_wiring_plan_with_seeds,
    )
    from research.runners.g11_bg_runner import build_bg_brain_regions

    n_replicas = 2

    # 1) Build single-replica regions+pathways. Cluster A + F.
    regions, pathways = build_bg_brain_regions(
        enable_cluster_a_closed_loop=True,
        enable_cluster_f_cerebellum=True,
    )
    print(f"[smoke] template: {len(regions)} regions, {len(pathways)} pathways")

    # 2) Build template wiring plan
    rmgr_template = RegionManager(regions, pathways)
    rmgr_template.initialize(seed=42)
    template_plan = rmgr_template.build_wiring_plan(seed=42)
    n_per_replica = rmgr_template.total_neurons()
    print(f"[smoke] template plan: {len(template_plan)} groups; n_per_replica={n_per_replica}")

    # 3) Replicate
    replicas = [
        ReplicaConfig(replica_id=i, seed_offset=42 * 1000 + i)
        for i in range(n_replicas)
    ]
    replicated_plan = replicate_wiring_plan_with_seeds(
        template_plan,
        replicas=replicas,
        neurons_per_replica=n_per_replica,
        weight_jitter=0.2,
    )
    rmgr_replicas = ReplicaManager(replicas, neurons_per_replica=n_per_replica)
    rmgr_replicas.initialize()
    total_neurons = rmgr_replicas.total_neurons()
    print(f"[smoke] replicated to {n_replicas} blocks; total_neurons={total_neurons}")

    # 4) Build bridge
    cfg = CoreSimConfig()
    cfg.num_neurons = total_neurons
    cfg.dt_ms = 0.5
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    sb = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)

    # 5) Initialize simulation data
    sb._initialize_simulation_data()
    print(f"[smoke] bridge booted: cp_membrane_potential_v.shape={sb.cp_membrane_potential_v.shape}")

    # 6) Inject the replicated wiring
    sb.inject_explicit_wiring(replicated_plan)
    nnz = int(sb.cp_connections.nnz) if sb.cp_connections is not None else 0
    print(f"[smoke] wiring injected: {nnz} synapses")

    # 7) Verify per-replica indices accessible
    for r in range(n_replicas):
        block = rmgr_replicas.indices(r)
        print(f"[smoke]   replica {r}: neurons {block[0]}..{block[-1]} ({len(block)} total)")

    # 8) Run 10 steps
    sb.cp_external_input_current[:] = cp.float32(0.0)
    for step in range(10):
        sb._run_one_simulation_step()
    print(f"[smoke] ran 10 steps successfully")

    # 9) Verify state
    n_fired = int(cp.sum(sb.cp_firing_states).get()) if sb.cp_firing_states is not None else 0
    v_min = float(cp.min(sb.cp_membrane_potential_v).get())
    v_max = float(cp.max(sb.cp_membrane_potential_v).get())
    print(f"[smoke] post-step: n_fired_this_step={n_fired}, V range [{v_min:.1f}, {v_max:.1f}]")

    print(f"[smoke] DONE: 2-replica cluster A+F brain stack runs without crashes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
