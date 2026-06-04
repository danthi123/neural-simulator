"""Compositional binding by thalamocortical GATING on the actual verb->motor vocabulary.

The payoff of the transmission-gate primitive (sim/bridge.py set_transmission_gate, validated in
tests/test_transmission_gate.py). The v16 approach grew STATIC verb_pool->motor weights by STDP and they
"went silent" (5/20 seed-fragile -- the binding never reached functional magnitude from zero-init). Here the
16 verb->motor routes are pre-wired with a FIXED weight and held CLOSED; binding (go,north) just OPENS the
gate g_GO_N. Driving "go" alone then drives motor_N -- through the dynamics, not a grown weight. Binding =
which gate is open (Logiaco-Abbott-Escola 2021), so it is DETERMINISTIC and re-bindable on command, where
grown weights were seed-fragile and could not re-bind.

  SIM_BACKEND=numpy python -m research.runners.gated_compose_demo
"""
import numpy as np

from sim.regions import BrainRegion, RegionPathway

VERBS = ["GO", "COME", "STOP", "LOOK"]
MOTORS = ["N", "E", "S", "W"]
TRUE_MAP = {"GO": "N", "COME": "S", "STOP": "W", "LOOK": "E"}


def build_gated_compose_bridge(seed=42, n_per_pool=30, weight=300.0):
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


def bind_mapping(sb, mapping):
    """Open exactly the gates for `mapping` (verb->motor); close all others."""
    for v in VERBS:
        for m in MOTORS:
            sb.set_transmission_gate(f"g_{v}_{m}", 0.0)
    for v, m in mapping.items():
        sb.set_transmission_gate(f"g_{v}_{m}", 1.0)


def motor_rates_for_verb(sb, verb, n_steps=60, drive_pA=1500.0):
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"verb_{verb}"))] = drive_pA
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(n_steps):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    return {m: float(acc[np.asarray(sb.region_manager.indices(f"motor_{m}"))].mean()) / n_steps for m in MOTORS}


def decode(sb, verb):
    """Drive `verb` alone -> the motor that fires most (the bound action)."""
    rates = motor_rates_for_verb(sb, verb)
    return max(rates, key=rates.get), rates


def main():
    print("=== compositional binding by GATING (verb->motor), spiking substrate ===\n", flush=True)
    for seed in (42, 43, 44):
        sb = build_gated_compose_bridge(seed=seed)
        bind_mapping(sb, TRUE_MAP)
        ok = 0
        line = []
        for v in VERBS:
            best, rates = decode(sb, v)
            correct = best == TRUE_MAP[v]
            ok += int(correct)
            line.append(f"{v}->{best}{'(ok)' if correct else '(X)'}")
        print(f"  seed {seed}: bind {TRUE_MAP} -> drive each verb alone -> {ok}/4   [{'  '.join(line)}]", flush=True)
    print("\n  -> binding is DETERMINISTIC: you bind exactly what you gate, re-bindable on command, with ZERO", flush=True)
    print("     weight change -- where STDP-grown verb->motor weights 'went silent' (5/20 seed-fragile).", flush=True)


if __name__ == "__main__":
    main()
