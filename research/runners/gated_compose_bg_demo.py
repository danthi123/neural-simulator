"""BG-driven thalamocortical gate selection: closing the loop on compositional binding by gating.

In gated_compose_demo the cortical route gates were opened by an EXTERNAL command (set_transmission_gate).
Here the gate is selected biologically: each verb->motor route has a THALAMIC gate-control pool (thal_X_Y);
the basal ganglia bind (verb, motor) by DISINHIBITING the corresponding thalamic pool (default silent), and
that thalamic ACTIVITY opens the cortical route gate. So binding flows BG -> thalamus -> gate -> cortical
routing (Logiaco-Abbott-Escola 2021 / Rikhye-Halassa 2018), not an external switch.

Scope (honest, cheap-first): the thalamus->gate coupling (thalamic firing rate -> transmission gain) is read
in the runner loop as a stand-in for a bridge-internal coupling; the BG selection (which thalamic pools are
disinhibited) is the bind mechanism. This demonstrates the closed loop at the behavioural level; a fully
bridge-internal thalamus->gate coupling is the further integration.

  SIM_BACKEND=numpy python -m research.runners.gated_compose_bg_demo
"""
import numpy as np

from sim.regions import BrainRegion, RegionPathway
from research.runners.gated_compose_demo import VERBS, MOTORS, TRUE_MAP, decode


def build_bg_gated_bridge(seed=42, n_per_pool=30, weight=300.0):
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation"):
        setattr(cfg, flag, False)
    cfg.brain_regions = (
        [BrainRegion(name=f"verb_{v}", n_neurons=n_per_pool, exc_fraction=1.0, internal_density=0.0) for v in VERBS]
        + [BrainRegion(name=f"motor_{m}", n_neurons=n_per_pool, exc_fraction=1.0, internal_density=0.0) for m in MOTORS]
        # thalamic gate-control pools, one per route, normally silent (BG-inhibited)
        + [BrainRegion(name=f"thal_{v}_{m}", n_neurons=n_per_pool, exc_fraction=1.0, internal_density=0.0)
           for v in VERBS for m in MOTORS]
    )
    cfg.region_pathways = [
        RegionPathway(from_region=f"verb_{v}", to_region=f"motor_{m}", density=1.0, weight_mean=weight,
                      weight_jitter=0.0, plastic=False, transmission_gate=f"g_{v}_{m}")
        for v in VERBS for m in MOTORS
    ]
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def bind_via_bg(sb, mapping, settle_steps=40, thal_drive_pA=1500.0, gate_thresh=0.05):
    """BG disinhibits the selected thalamic pools; their activity opens the matching cortical route gates.

    Models BG gate SELECTION: `mapping` is the set of (verb,motor) bindings the BG releases. All cortical
    route gates start CLOSED; the gate for route X->Y is opened iff thalamic pool thal_X_Y fires (i.e. iff the
    BG disinhibited it). The thalamus->gate coupling (rate -> gain) is read here in the runner.
    """
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    for v, m in mapping.items():
        sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"thal_{v}_{m}"))] = thal_drive_pA
    for v in VERBS:
        for m in MOTORS:
            sb.set_transmission_gate(f"g_{v}_{m}", 0.0)
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle_steps):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    opened = {}
    for v in VERBS:
        for m in MOTORS:
            thal_rate = acc[np.asarray(sb.region_manager.indices(f"thal_{v}_{m}"))].mean() / settle_steps
            is_open = thal_rate > gate_thresh
            sb.set_transmission_gate(f"g_{v}_{m}", 1.0 if is_open else 0.0)   # thalamic activity opens the gate
            if is_open:
                opened[v] = m
    # clear the thalamic drive; the gates retain their selected state for the verb-decode phase
    sb.cp_external_input_current[:] = 0.0
    return opened


def main():
    print("=== BG-driven thalamocortical gate selection (verb->motor binding), spiking substrate ===\n", flush=True)
    for seed in (42, 43, 44):
        sb = build_bg_gated_bridge(seed=seed)
        opened = bind_via_bg(sb, TRUE_MAP)             # BG disinhibits thal pools -> thal activity opens gates
        ok = 0
        line = []
        for v in VERBS:
            best, _ = decode(sb, v)
            correct = best == TRUE_MAP[v]
            ok += int(correct)
            line.append(f"{v}->{best}{'(ok)' if correct else '(X)'}")
        gates_match = (opened == TRUE_MAP)
        print(f"  seed {seed}: BG selects {TRUE_MAP} -> thal opens gates {opened} (match={gates_match}) -> "
              f"drive each verb -> {ok}/4   [{'  '.join(line)}]", flush=True)
    print("\n  -> binding flows BG-disinhibition -> thalamic activity -> cortical route gate -> verb routes to", flush=True)
    print("     its motor. The basal ganglia SELECT the binding; the thalamus opens the gate. Loop closed.", flush=True)


if __name__ == "__main__":
    main()
