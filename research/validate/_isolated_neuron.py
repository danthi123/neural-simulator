"""Isolated single-neuron-type validation utilities.

Sets up a simulation with a single BrainRegion of N HH neurons, all of the
same neuron type, with no connectivity to anything else, no plasticity, no
homeostasis, no OU noise (or controlled noise). Then injects step currents
and measures cellular response metrics (resting Vm, F-I curve, adaptation).

This isolates the cellular biophysics from network effects so we can
validate parameter presets cleanly against published single-neuron data.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@dataclass
class FICurvePoint:
    current_pA: float
    rate_hz_initial: float       # First 100 ms of stimulus (transient)
    rate_hz_steady: float        # Last 500 ms of stimulus (after adaptation)
    spike_times_ms: List[float]
    mean_vm_pre_stim: float      # Vm in last 100 ms before stim (rest test)
    max_vm_during_stim: float    # Peak Vm reached during stim (diagnostic)
    mean_vm_during_stim: float   # Mean Vm during stim (sub-threshold plateau)
    n_neurons_averaged: int


@dataclass
class ValidationResult:
    preset_name: str
    n_neurons: int
    fi_curve: List[FICurvePoint]
    rest_vm: float                       # Mean Vm across all pre-stim baselines
    rest_vm_std: float
    spike_threshold_pA: float            # Smallest current that elicits >=1 spike
    rheobase_rate_hz: float              # Rate just above threshold
    f1_nA_rate_hz: float                  # Rate at 1 nA = 1000 pA (typical literature reference)
    adaptation_ratio_at_1nA: float       # steady / initial at 1 nA (1.0 = no adapt; 0.5 = 50% adapt)
    metadata: Dict


def build_hh_isolated_config(
    neuron_type_name: str,
    n_neurons: int = 10,
    dt_ms: float = 0.05,
    enable_ou_noise: bool = False,
    enable_conductance_noise: bool = False,
    enable_parameter_heterogeneity: bool = False,
    temperature_celsius: float = 37.0,
    q10_factor: float = 3.0,
    seed: int = 42,
):
    """Returns a CoreSimConfig wired up for an isolated population of N
    identical HH neurons of the given NeuronType.

    NOTE: the simulator's bridge initializes HH parameters from
    `DefaultHodgkinHuxleyParams.PARAMS[NeuronType[neuron_type_name]]` —
    NOT from cfg.hh_* fields. So validation must work via the registered
    NeuronType enum names (HH_L5_CORTICAL_PYRAMIDAL_RS, etc.) rather than
    arbitrary param dicts.

    All plasticity, OU noise, conductance noise, and parameter heterogeneity
    are OFF by default — turn on individually for noise sensitivity tests.
    The neurons live in a single BrainRegion with internal_density=0 (no
    recurrent connectivity) and no cross-region pathways.
    """
    from sim import CoreSimConfig
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType, DefaultHodgkinHuxleyParams

    cfg = CoreSimConfig()
    cfg.num_neurons = 0  # Set by region framework
    cfg.dt_ms = dt_ms
    cfg.seed = seed
    cfg.neuron_model_type = NeuronModel.HODGKIN_HUXLEY.name
    cfg.default_neuron_type_hh = neuron_type_name  # KEY: this is what the bridge reads
    cfg.hh_temperature_celsius = float(temperature_celsius)
    cfg.hh_q10_factor = float(q10_factor)
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"

    # Also seed cfg.hh_* fields for any code path that reads them (UI, etc.)
    nt_enum = NeuronType[neuron_type_name]
    hh_params = DefaultHodgkinHuxleyParams.get_params(nt_enum)
    cfg.hh_C_m = float(hh_params["C_m"])
    cfg.hh_g_Na_max = float(hh_params["g_Na_max"])
    cfg.hh_g_K_max = float(hh_params["g_K_max"])
    cfg.hh_g_L = float(hh_params["g_L"])
    cfg.hh_E_Na = float(hh_params["E_Na"])
    cfg.hh_E_K = float(hh_params["E_K"])
    cfg.hh_E_L = float(hh_params["E_L"])
    cfg.hh_v_rest_init = float(hh_params["v_rest_hh"])
    cfg.hh_v_peak = float(hh_params["v_peak_hh"])
    cfg.hh_m_init = float(hh_params["m_init"])
    cfg.hh_h_init = float(hh_params["h_init"])
    cfg.hh_n_init = float(hh_params["n_init"])
    cfg.hh_g_M_max = float(hh_params.get("g_M_max", 0.0))
    cfg.hh_g_CaT_max = float(hh_params.get("g_CaT_max", 0.0))
    cfg.hh_E_CaT = float(hh_params.get("E_CaT", 120.0))
    cfg.hh_g_h_max = float(hh_params.get("g_h_max", 0.0))
    cfg.hh_E_h = float(hh_params.get("E_h", -30.0))
    cfg.hh_g_NaP_max = float(hh_params.get("g_NaP_max", 0.0))

    # No noise sources (cleanest cellular response)
    cfg.enable_ou_process = bool(enable_ou_noise)
    cfg.enable_conductance_noise = bool(enable_conductance_noise)
    cfg.enable_parameter_heterogeneity = bool(enable_parameter_heterogeneity)

    # No network effects
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(
            name="cell",
            n_neurons=n_neurons,
            exc_fraction=1.0,           # All excitatory (no inhibitory cells in this isolated test)
            internal_density=0.0,       # No recurrent connectivity
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
            plastic_internal=False,
        )
    ]
    cfg.region_pathways = []
    cfg.connections_per_neuron = 0
    cfg.enable_watts_strogatz = False
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False

    return cfg


def run_step_current_protocol(
    neuron_type_name: str,
    current_steps_pA: List[float],
    pre_stim_ms: float = 200.0,
    stim_duration_ms: float = 1000.0,
    post_stim_ms: float = 100.0,
    n_neurons: int = 10,
    dt_ms: float = 0.05,
    seed: int = 42,
    initial_settle_ms: float = 200.0,
    temperature_celsius: float = 37.0,
    q10_factor_override: float = None,
) -> List[FICurvePoint]:
    """Run a sequence of step-current injections and measure firing rate at each.

    Each step:
      [pre_stim_ms baseline] → [stim_duration_ms with current] → [post_stim_ms baseline]
    Between steps the bridge is rebuilt fresh, so each step is independent.

    Bypasses the StimulusManager — sets bridge.cp_external_input_current
    directly. Cleaner for single-cell validation; one-step delay vs.
    StimulusManager is irrelevant for steady-state F-I.

    Returns a list of FICurvePoint, one per current level.
    """
    import cupy as cp
    from sim import (
        SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig,
    )

    n_steps_pre = int(pre_stim_ms / dt_ms)
    n_steps_stim = int(stim_duration_ms / dt_ms)
    n_steps_post = int(post_stim_ms / dt_ms)
    n_steps_settle = int(initial_settle_ms / dt_ms)

    points = []
    for I_pA in current_steps_pA:
        kwargs = {}
        if q10_factor_override is not None:
            kwargs["q10_factor"] = float(q10_factor_override)
        cfg = build_hh_isolated_config(
            neuron_type_name=neuron_type_name,
            n_neurons=n_neurons, dt_ms=dt_ms,
            temperature_celsius=temperature_celsius, seed=seed,
            **kwargs,
        )
        bridge = SimulationBridge(
            core_config=cfg, viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(), gpu_config=GPUConfig(),
        )
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)

        # Zero out the hardcoded HH baseline drive (5e6-20e6 pA random
        # uniform) — it's meant for ensuring spontaneous activity in
        # network simulations, but for cellular validation we want clean
        # rest behavior with only our injected step current.
        # See sim/bridge.py:795-815.
        if bridge.cp_external_input_current is not None:
            bridge.cp_external_input_current[:] = 0.0

        # Settle the cell at rest first (no input)
        for _ in range(n_steps_settle):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * dt_ms
            )

        # Run pre-stimulus baseline (no input) to measure rest Vm
        vm_pre_samples = []
        for _ in range(n_steps_pre):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * dt_ms
            )
            # Sample Vm sparsely (every 10 steps = every 0.5ms at dt=0.05)
            if bridge.runtime_state.current_time_step % 10 == 0:
                vm_pre_samples.append(
                    float(cp.mean(bridge.cp_membrane_potential_v).get())
                )
        vm_pre = float(np.mean(vm_pre_samples)) if vm_pre_samples else 0.0

        # Apply step current directly to all neurons via external_input_current.
        # This bypasses the StimulusManager (avoiding any timing-window issues)
        # and applies a constant current injection identical to a current-clamp
        # step in a real recording.
        bridge.cp_external_input_current[:] = cp.float32(I_pA)

        # Track per-neuron spike times (in ms relative to stim onset)
        spike_times_per_neuron: List[List[float]] = [[] for _ in range(n_neurons)]
        prev_firing = np.zeros(n_neurons, dtype=bool)
        t_stim_start_ms = bridge.runtime_state.current_time_ms
        max_vm_seen = -200.0
        vm_samples_during_stim = []
        for s in range(n_steps_stim):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * dt_ms
            )
            firing = bridge.cp_firing_states.get().astype(bool)
            new_spikes = firing & ~prev_firing  # rising-edge detection
            t_in_stim_ms = (bridge.runtime_state.current_time_ms - t_stim_start_ms)
            for n_idx in np.where(new_spikes)[0]:
                spike_times_per_neuron[int(n_idx)].append(float(t_in_stim_ms))
            prev_firing = firing
            if bridge.runtime_state.current_time_step % 10 == 0:
                vm_now = float(cp.mean(bridge.cp_membrane_potential_v).get())
                vm_samples_during_stim.append(vm_now)
                if vm_now > max_vm_seen:
                    max_vm_seen = vm_now

        # Compute rates
        all_spike_times = [t for st in spike_times_per_neuron for t in st]
        # initial 100ms window
        n_spikes_initial = sum(
            1 for t in all_spike_times if t < 100.0
        )
        # last 500ms window (steady state)
        n_spikes_steady = sum(
            1 for t in all_spike_times if t >= (stim_duration_ms - 500.0)
        )
        rate_initial = n_spikes_initial / n_neurons / 0.1   # spikes / neuron / 0.1 sec
        rate_steady = n_spikes_steady / n_neurons / 0.5     # spikes / neuron / 0.5 sec

        mean_vm_during = float(np.mean(vm_samples_during_stim)) if vm_samples_during_stim else 0.0
        points.append(FICurvePoint(
            current_pA=float(I_pA),
            rate_hz_initial=float(rate_initial),
            rate_hz_steady=float(rate_steady),
            spike_times_ms=all_spike_times,
            mean_vm_pre_stim=float(vm_pre),
            max_vm_during_stim=float(max_vm_seen),
            mean_vm_during_stim=mean_vm_during,
            n_neurons_averaged=n_neurons,
        ))

        bridge.clear_simulation_state_and_gpu_memory()

    return points


def summarize_fi_curve(
    fi: List[FICurvePoint],
    target_currents_for_metrics: Dict[str, float] = None,
) -> Dict[str, float]:
    """Extract validation metrics from an F-I curve.

    Returns dict with:
      - rest_vm: mean Vm during pre-stim, averaged across all current levels
      - spike_threshold_pA: smallest current with rate_initial > 0
      - f_at_1nA: rate_steady at 1000 pA (or closest available)
      - adaptation_ratio_at_1nA: rate_steady / rate_initial at 1nA
      - max_rate: highest rate_steady observed
    """
    if not fi:
        return {}
    rest_vms = [p.mean_vm_pre_stim for p in fi]
    rest_vm = float(np.mean(rest_vms))
    rest_vm_std = float(np.std(rest_vms))

    threshold = None
    for p in fi:
        if p.rate_hz_initial > 0.5:  # at least 1 spike across all neurons
            threshold = p.current_pA
            break
    threshold = float(threshold) if threshold is not None else float("nan")

    # Find closest to 1000 pA
    target_I = 1000.0
    closest = min(fi, key=lambda p: abs(p.current_pA - target_I))
    rate_at_1nA = closest.rate_hz_steady
    if closest.rate_hz_initial > 0:
        adapt_ratio_at_1nA = closest.rate_hz_steady / closest.rate_hz_initial
    else:
        adapt_ratio_at_1nA = float("nan")

    max_rate = max(p.rate_hz_steady for p in fi)

    # Per-current threshold rate (rheobase)
    rheobase_rate = next(
        (p.rate_hz_steady for p in fi if p.rate_hz_initial > 0.5), 0.0
    )

    return {
        "rest_vm": rest_vm,
        "rest_vm_std": rest_vm_std,
        "spike_threshold_pA": threshold,
        "rheobase_rate_hz": float(rheobase_rate),
        "rate_at_1nA": float(rate_at_1nA),
        "adaptation_ratio_at_1nA": float(adapt_ratio_at_1nA),
        "max_steady_rate": float(max_rate),
        "current_at_1nA": float(closest.current_pA),
    }
