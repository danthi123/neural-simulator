"""G11: Basal-ganglia-style action selection module.

Phase B follow-up to the silent-motor trap arc (Sessions G/H/I, all NEGATIVE).
The trap was diagnosed (V6) as a *reservoir-state bias problem* — random
hidden->motor weights on a shared reservoir naturally favor whichever motor
the input pattern happens to align with. Argmax + reservoir bias = lock-in.

Phase B fix (architectural): replace the shared-reservoir + argmax-readout
with a real basal-ganglia-style circuit. Each motor has its own dedicated
striatum_D1, striatum_D2, GPi, thalamus, and motor populations. Lateral
inhibition between motor populations provides structural winner-take-all
(no shared spike count to bias).

Architecture:
    cortex ─-> str_D1[N,E,S,W]    str_D2[N,E,S,W]
                  │                     │
              direct path          indirect path
                  v                     v
              GPi[N,E,S,W] <-── STN <-── GPe[N,E,S,W]
                  │
                  v (disinhibition)
              thal[N,E,S,W]
                  │
                  v
              motor[N,E,S,W]   (lateral inhibition between)

DA modulation: VTA/SNc DA neurons project to all striatal pools. DA enhances
direct pathway (D1+ sensitivity) and suppresses indirect pathway (D2-).

Built on validated Phase A presets:
- IZH2007_STRIATAL_MSN_D1 / D2 (rest=-80 mV down-state, fires when driven)
- IZH2007_GPE_PACEMAKER, IZH2007_GPI_OUTPUT (high tonic rates)
- IZH2007_STN_BURST (autonomous + scales with input)
- IZH2007_THALAMIC_RELAY (tonic mode)
- IZH2007_RS_CORTICAL_PYRAMIDAL, IZH2007_FS_CORTICAL_INTERNEURON (cortex)
- IZH2007_DOPAMINE (slow tonic + phasic)

Reference: Frank 2005 J Neurosci; Schroll & Hamker 2013 Front Comp Neurosci.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
N_ACTIONS = 4


def build_bg_brain_regions(
    n_cortex: int = 100,
    n_striatum_per_action: int = 50,
    n_gpe_per_action: int = 10,
    n_gpi_per_action: int = 10,
    n_stn: int = 20,
    n_thal_per_action: int = 10,
    n_motor_per_action: int = 10,
    n_dopamine: int = 10,
):
    """Returns list of BrainRegion + list of RegionPathway for the BG circuit."""
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    regions = []
    pathways = []

    # Cortex (input layer for goal-directed signals).
    # Split into per-action pools so different inputs preferentially activate
    # different actions. This is a phenomenological substitute for what
    # learning would produce: differential cortex→striatum weights.
    n_cortex_per_action = n_cortex // N_ACTIONS
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"cortex_{action}",
            n_neurons=n_cortex_per_action,
            exc_fraction=1.0,  # All excitatory for cortex inputs
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Per-action striatal pools (D1 direct, D2 indirect).
    # internal_density=0 (no lateral inhibition) initially — MSNs need
    # strong cortex drive to escape the down-state and lateral inhibition
    # makes that even harder. Add it back later if action selection needs
    # sharpening.
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"str_D1_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
        ))
        regions.append(BrainRegion(
            name=f"str_D2_{action}",
            n_neurons=n_striatum_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D2.name,
        ))

    # Per-action BG output (GPe / GPi)
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"gpe_{action}",
            n_neurons=n_gpe_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPE_PACEMAKER.name,
        ))
        regions.append(BrainRegion(
            name=f"gpi_{action}",
            n_neurons=n_gpi_per_action,
            exc_fraction=0.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
        ))

    # Single STN (excitatory, projects diffusely to all GPi)
    regions.append(BrainRegion(
        name="stn",
        n_neurons=n_stn,
        exc_fraction=1.0,  # STN is glutamatergic (excitatory)
        internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_STN_BURST.name,
    ))

    # Per-action thalamic relay + motor cortex
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"thal_{action}",
            n_neurons=n_thal_per_action,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_THALAMIC_RELAY.name,
        ))
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # Dopamine neurons (single pool, broadcasts via neuromodulator subsystem)
    regions.append(BrainRegion(
        name="dopamine",
        n_neurons=n_dopamine,
        exc_fraction=1.0,
        internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
    ))

    # ---- Pathways (cross-region projections) ----

    # Cortex -> striatum (LEARNING site).
    # Each cortex_X projects strongly to its corresponding str_D1_X / str_D2_X
    # AND weakly to other actions' striatum (cross-projection allows learning
    # to redistribute action representations on goal change).
    for cortex_action in ACTION_NAMES:
        for str_action in ACTION_NAMES:
            same = (cortex_action == str_action)
            density = 1.0 if same else 0.3
            weight = 25.0 if same else 5.0
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D1_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
            ))
            pathways.append(RegionPathway(
                from_region=f"cortex_{cortex_action}",
                to_region=f"str_D2_{str_action}",
                density=density, weight_mean=weight, weight_jitter=0.2, plastic=True,
            ))

    # Direct pathway: D1 -> GPi (inhibitory). Strong weight needed to overcome
    # GPi tonic firing (~30-75 Hz baseline).
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D1_{action}", to_region=f"gpi_{action}",
            density=1.0, weight_mean=15.0, weight_jitter=0.2, plastic=False,
        ))

    # Indirect pathway: D2 -> GPe -> STN -> GPi
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"str_D2_{action}", to_region=f"gpe_{action}",
            density=0.6, weight_mean=2.5, weight_jitter=0.2, plastic=False,
        ))
        pathways.append(RegionPathway(
            from_region=f"gpe_{action}", to_region="stn",
            density=0.3, weight_mean=1.5, weight_jitter=0.2, plastic=False,
        ))

    # STN -> all GPi (diffuse excitation; this is the "hyperdirect"-like
    # contribution that biases against premature action selection)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="stn", to_region=f"gpi_{action}",
            density=0.4, weight_mean=1.0, weight_jitter=0.2, plastic=False,
        ))

    # GPi -> thalamus (inhibitory). Strong weight + density needed so
    # GPi tonic firing fully suppresses thal, AND so D1-mediated GPi
    # silence cleanly releases the gate.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"gpi_{action}", to_region=f"thal_{action}",
            density=1.0, weight_mean=8.0, weight_jitter=0.2, plastic=False,
        ))

    # Thalamus -> motor cortex (excitatory). Very strong weight needed
    # because thal pool is small (10 cells) and we need ~50 Hz motor output
    # from ~24 Hz thal input.
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region=f"thal_{action}", to_region=f"motor_{action}",
            density=1.0, weight_mean=20.0, weight_jitter=0.2, plastic=False,
        ))

    # NOTE: motor→motor "lateral inhibition" was creating EXCITATORY synapses
    # because motor regions have exc_fraction=1.0 (RegionPathway sign comes
    # from source region's exc_fraction). For real lateral inhibition we'd
    # need motor-pool FS interneuron sub-populations. Removed for now; BG
    # gating already provides selectivity (only the action with silenced
    # GPi gets thalamic drive).

    return regions, pathways


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: build + 50 steps at rest")
    ap.add_argument("--probe-action", type=str, default=None,
                    choices=ACTION_NAMES,
                    help="Drive cortex toward this action and measure motor output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    import cupy as cp

    print(f"\n{'='*72}")
    print(f"  G11 BG Action Selection Module — Smoke Test")
    print(f"{'='*72}\n", flush=True)

    regions, pathways = build_bg_brain_regions()
    n_total = sum(r.n_neurons for r in regions)
    print(f"  Built {len(regions)} regions with {n_total} total neurons")
    print(f"  Built {len(pathways)} pathways")
    print()

    # Verify no name collisions
    names = [r.name for r in regions]
    assert len(set(names)) == len(names), "Region name collision!"

    cfg = CoreSimConfig()
    cfg.num_neurons = 0  # Set by region framework
    cfg.dt_ms = 1.0
    cfg.seed = int(args.seed)
    cfg.num_traits = 1  # Force single neuron type per region
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False  # Smoke test: no plasticity
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False

    print(f"  Initializing bridge...", flush=True)
    t0 = time.time()
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    elapsed = time.time() - t0
    print(f"  Bridge initialized in {elapsed:.1f}s", flush=True)
    print(f"  Total neurons: {cfg.num_neurons}")
    print(f"  Total synapses: {bridge.cp_connections.nnz}")

    if not args.smoke and not args.probe_action:
        return 0

    # Quick 30-step smoke run with no input — should show GPe/GPi tonic firing
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0
    n_steps = 50
    n_motor_total = sum(r.n_neurons for r in regions if r.name.startswith("motor_"))

    spike_counts = np.zeros(cfg.num_neurons, dtype=np.int32)
    print(f"\n  Running {n_steps} steps with no input (rest dynamics)...", flush=True)
    for s in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        firing = bridge.cp_firing_states.get().astype(np.int32)
        spike_counts += firing

    # Per-region firing rate
    print(f"\n  Per-region firing rates (Hz over {n_steps}ms with no input):")
    for r in regions:
        idx = bridge.region_manager.indices(r.name)
        rate_hz = spike_counts[list(idx)].sum() / r.n_neurons / (n_steps * cfg.dt_ms / 1000.0)
        print(f"    {r.name:<24s} ({r.izh_neuron_type or 'default':<32s}): {rate_hz:.1f} Hz")

    print(f"\n  Smoke test PASSED — {len(regions)} regions, "
          f"{bridge.cp_connections.nnz} synapses initialized cleanly.")

    # ---- Phase B.T4 / T5: action selection probe ----
    if args.probe_action:
        print(f"\n{'='*72}")
        print(f"  Action selection probe: drive cortex -> {args.probe_action} pathway")
        print(f"{'='*72}\n", flush=True)

        # Inject strong current into a SUBSET of cortex neurons. The cortex->D1/D2
        # weights are random — so the input pattern preferentially activates
        # whichever D1/D2 happens to have stronger weights from these inputs.
        # For a clean probe, manually override: inject ONLY into cortex neurons
        # whose hash maps to the target action.
        # Apply tonic baseline drive to BG output nuclei (mimics intrinsic
        # depolarizing conductance that makes real GPe/GPi/STN autonomously
        # fire 30-80 Hz). Without this, our Izh presets sit at rest because
        # Izh doesn't model intrinsic Ca pacemaker currents.
        bridge.cp_external_input_current[:] = 0.0
        # Per-region tonic drive levels:
        for region_name in [f"gpe_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(150.0)
        for region_name in [f"gpi_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                # Lower baseline for GPi → easier to silence by D1 inhibition
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(110.0)
        for region_name in ["stn", "dopamine"]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(150.0)
        # Thalamus baseline drive — set such that GPi inhibition (when active)
        # keeps thal silent, AND when GPi drops to 0 (D1 suppression),
        # thal fires actively.
        for region_name in [f"thal_{a}" for a in ACTION_NAMES]:
            idx = list(bridge.region_manager.indices(region_name))
            if idx:
                bridge.cp_external_input_current[cp.asarray(idx, dtype=cp.int64)] = cp.float32(300.0)

        # Drive ONLY the target action's cortex pool
        cortex_idx = list(bridge.region_manager.indices(f"cortex_{args.probe_action}"))
        cortex_cp = cp.asarray(cortex_idx, dtype=cp.int64)

        bridge.runtime_state.current_time_step = 0
        bridge.runtime_state.current_time_ms = 0.0

        drive_pA = 800.0
        n_probe_steps = 500
        target_cortex = cortex_idx
        spike_counts = np.zeros(cfg.num_neurons, dtype=np.int32)
        for s in range(n_probe_steps):
            bridge.cp_external_input_current[cortex_cp] = cp.float32(drive_pA)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            firing = bridge.cp_firing_states.get().astype(np.int32)
            spike_counts += firing

        # Per-region firing rate
        print(f"  Driving {len(target_cortex)}/{len(cortex_idx)} cortex neurons "
              f"with {drive_pA} pA for {n_probe_steps}ms")
        print(f"\n  Per-region firing rates over {n_probe_steps}ms:")
        ordered_groups = [f"cortex_{a}" for a in ACTION_NAMES]
        for a in ACTION_NAMES:
            ordered_groups += [f"str_D1_{a}", f"str_D2_{a}", f"gpe_{a}",
                                f"gpi_{a}", f"thal_{a}", f"motor_{a}"]
        ordered_groups += ["stn", "dopamine"]
        for region_name in ordered_groups:
            r = next((reg for reg in regions if reg.name == region_name), None)
            if r is None:
                continue
            idx = bridge.region_manager.indices(r.name)
            if not idx:
                continue
            rate_hz = spike_counts[list(idx)].sum() / r.n_neurons / (n_probe_steps / 1000.0)
            marker = " <-" if (region_name.endswith(f"_{args.probe_action}") and
                              region_name.startswith(("str_D1_", "thal_", "motor_"))) else ""
            print(f"    {r.name:<15s} {rate_hz:>6.1f} Hz{marker}")

        # Quick check: did the right motor pop fire most?
        motor_rates = {}
        for a in ACTION_NAMES:
            idx = bridge.region_manager.indices(f"motor_{a}")
            n = len(idx)
            r = spike_counts[list(idx)].sum() / max(n, 1) / (n_probe_steps / 1000.0)
            motor_rates[a] = r
        winner = max(motor_rates, key=motor_rates.get)
        print(f"\n  Motor rates: {motor_rates}")
        print(f"  Winner: {winner}  (target: {args.probe_action})")
        if winner == args.probe_action and motor_rates[winner] > 5:
            print(f"  [OK] BG circuit selected the correct motor")
        else:
            print(f"  -> BG circuit did not produce a clean winner (rates may be too low/noisy)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
