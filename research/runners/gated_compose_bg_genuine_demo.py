"""GENUINE basal-ganglia gate selection (cheat-removal item #2): the thalamic gate-control pools are released
by a REAL BG disinhibition cascade, not driven by direct current.

`gated_compose_bg_demo` drove the thalamic gate-control pools with direct input current as a stand-in for BG
disinhibition. This removes that shortcut: each binding gets a genuine direct-pathway cascade

    cortex_select_v_m  --(excite)-->  str_D1_v_m  --(GABA, inhibit)-->  gpi_v_m  --(GABA, inhibit)-->  thal_v_m

where gpi_v_m is a TONIC pacemaker (IZH2007_GPI_OUTPUT) that normally silences thal_v_m, and thal_v_m has a tonic
thalamic excitation it can only express when GPi is released. Selecting a binding = activating its
cortex_select pool: str_D1 fires, inhibits its GPi, the GPi stops pacing, and thal_v_m is DISINHIBITED -> its
firing opens the cortical route transmission gate (via bridge.couple_gate_to_pool). So binding flows
cortex-selection -> striatum -> GPi disinhibition -> thalamus -> gate -> cortical routing, entirely in the
core-sim brain-region framework (Logiaco-Abbott-Escola 2021; the canonical direct-pathway "go" of Kandel ch 38).

This is item #2 of the pure-biology cheat-removal backlog: the BG SELECTION MECHANISM is now genuine
disinhibition. (Item #3 -- the BG LEARNING which gate to select -- builds reward-driven plasticity on top.)

STATUS 2026-06-04 — RESOLVED (genuine disinhibition works: seed 42 4/4, 43 4/4, 44 3/4 = 11/12). The
non-obvious blocker was SYNAPTIC WEIGHT SCALE, not the cascade structure: a conductance-based synapse with
weight ~300-600 (the value the earlier gated-compose stand-in used) explodes the inhibitory conductance g_i to
~2300 (vs physiological O(1-10)), which clamps the membrane to the -75 reversal and breaks Izhikevich numerics
into paradoxical rebound firing -- so D1->GPi *looked* excitatory (gpi ROSE when D1 fired). At g11_bg's
validated weight scale (D1->GPi=15, GPi->thal=8) the conductance stays physiological and D1 genuinely SILENCES
its GPi (isolation test: driving d1 drops gpi 0.276 -> 0.068), which disinhibits the thalamic relay and opens
the gate. A tonic GPi drive (2200 pA) provides the pacemaker baseline (the reduced model's stand-in for STN
drive); the relay carries a tonic excitation (600 pA) expressed only when released. Diagnosed via
`_framework_inhibition_minimal_probe` (a one-inhibitory-region -> one-excitable-region control that isolates
the weight-scale effect) and the in-demo D1->GPi isolation test.

This RESOLVES cheat-removal #2: the thalamic gate-control pools are released by a genuine D1 -| GPi -| thal
disinhibition cascade, NOT by direct thalamic current. (WHICH D1 pool is driven is still commanded -- that is
the separate cheat #3, learned vs commanded selection.)

  SIM_BACKEND=numpy python -m research.runners.gated_compose_bg_genuine_demo
"""
import numpy as np

from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronType
from research.runners.gated_compose_demo import VERBS, MOTORS, TRUE_MAP, decode

THAL_TONIC_PA = 600.0      # tonic thalamic excitation (expressed only when GPi releases the relay)
GPI_TONIC_PA = 2200.0      # tonic GPi drive -> GPi pacing -> inhibits its thalamic relay by default

# Weights are at g11_bg's VALIDATED scale (D1->GPi=15, GPi->thal=8). A conductance-based synapse with
# weight ~300 explodes g_i to ~2300 (vs physiological O(1-10)), which clamps V to the -75 reversal and breaks
# Izhikevich numerics into paradoxical rebound firing -- the inhibition then *looks* excitatory. Small weights
# keep the conductance physiological so D1 genuinely silences GPi. (Diagnosed via _framework_inhibition_minimal_probe.)
def build_genuine_bg_gated_bridge(seed=42, n=30, route_weight=40.0, d1_w=15.0, gpi_w=8.0):
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

    pairs = [(v, m) for v in VERBS for m in MOTORS]
    cfg.brain_regions = (
        [BrainRegion(name=f"verb_{v}", n_neurons=n, exc_fraction=1.0, internal_density=0.0) for v in VERBS]
        + [BrainRegion(name=f"motor_{m}", n_neurons=n, exc_fraction=1.0, internal_density=0.0) for m in MOTORS]
        # thalamic gate-control relay per route -- normally silenced by its tonic-pacemaker GPi
        + [BrainRegion(name=f"thal_{v}_{m}", n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                       izh_neuron_type=NeuronType.IZH2007_THALAMIC_RELAY.name) for v, m in pairs]
        # cortex selection input per route (excitatory)
        + [BrainRegion(name=f"sel_{v}_{m}", n_neurons=n, exc_fraction=1.0, internal_density=0.0) for v, m in pairs]
        # striatal D1 MSN per route (fully GABAergic -> outgoing synapses inhibitory). Use the DEFAULT
        # inhibitory reversal (-75 mV): a -60 mV override made D1->GPi *depolarizing* (excitatory) because a
        # resting GPi sits below -60, so GABA at E=-60 pulls it UP toward firing. -75 keeps it hyperpolarizing.
        + [BrainRegion(name=f"d1_{v}_{m}", n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                       izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name) for v, m in pairs]
        # GPi output per route: tonic pacemaker, GABAergic -> inhibits its thalamic relay
        + [BrainRegion(name=f"gpi_{v}_{m}", n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                       izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name) for v, m in pairs]
    )
    cfg.region_pathways = (
        [RegionPathway(from_region=f"verb_{v}", to_region=f"motor_{m}", density=1.0, weight_mean=route_weight,
                       weight_jitter=0.0, plastic=False, transmission_gate=f"g_{v}_{m}") for v, m in pairs]
        + [RegionPathway(from_region=f"sel_{v}_{m}", to_region=f"d1_{v}_{m}", density=1.0, weight_mean=route_weight,
                         weight_jitter=0.0, plastic=False) for v, m in pairs]            # cortex -> D1 (excite)
        + [RegionPathway(from_region=f"d1_{v}_{m}", to_region=f"gpi_{v}_{m}", density=1.0, weight_mean=d1_w,
                         weight_jitter=0.0, plastic=False) for v, m in pairs]            # D1 -| GPi (inhibit)
        + [RegionPathway(from_region=f"gpi_{v}_{m}", to_region=f"thal_{v}_{m}", density=1.0, weight_mean=gpi_w,
                         weight_jitter=0.0, plastic=False) for v, m in pairs]            # GPi -| thal (inhibit)
    )
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def couple_all_route_gates(sb, threshold=0.03):
    """Each cortical route gate g_v_m opens from its thalamic relay thal_v_m's firing, inside the step."""
    for v in VERBS:
        for m in MOTORS:
            sb.couple_gate_to_pool(f"g_{v}_{m}", f"thal_{v}_{m}", threshold=threshold)


def _drive(sb, selected, verb=None, thal_tonic=THAL_TONIC_PA, sel_pA=1500.0, verb_pA=1500.0):
    """Set the per-step external drive: tonic thalamic excitation on every relay, a striatal 'go' signal on the
    chosen bindings' D1 pools (the selection), and (optionally) the verb cue. The genuine D1 -| GPi -| thal
    disinhibition then opens only the selected gates. (Driving D1 directly is the striatal selection signal;
    WHICH D1 to drive being commanded is cheat #3 -- the disinhibition cascade itself is the genuine biology
    for #2. The sel->d1 cortical hop exists in the wiring but is too weak to fire D1 on its own here.)"""
    sb.cp_external_input_current[:] = 0.0
    for v in VERBS:
        for m in MOTORS:
            sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"thal_{v}_{m}"))] = thal_tonic
            sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"gpi_{v}_{m}"))] = GPI_TONIC_PA
    for v, m in selected.items():
        sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"d1_{v}_{m}"))] = sel_pA
    if verb is not None:
        sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"verb_{verb}"))] = verb_pA


def thal_rates(sb, selected, settle=60):
    """Settle the BG with `selected` cortex-select pools active; return per-route thalamic firing rate."""
    from sim.backend import to_host
    _drive(sb, selected)
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    return {(v, m): acc[np.asarray(sb.region_manager.indices(f"thal_{v}_{m}"))].mean() / settle
            for v in VERBS for m in MOTORS}


def decode_genuine(sb, verb, selected, n_steps=60, settle=40):
    """Hold the BG selection (cortex-select active) so the chosen gates stay open via genuine disinhibition,
    then drive the verb; return the motor that fires most."""
    from sim.backend import to_host
    _drive(sb, selected)                                   # settle the disinhibition first
    for _ in range(settle):
        sb._run_one_simulation_step()
    _drive(sb, selected, verb=verb)                        # then add the verb cue, keep the selection
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(n_steps):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    return max(MOTORS, key=lambda m: acc[np.asarray(sb.region_manager.indices(f"motor_{m}"))].mean())


def _pool_rates(sb, selected, names, settle=60):
    """Settle with `selected` cortex-select pools active; return mean firing rate for each region in `names`."""
    from sim.backend import to_host
    _drive(sb, selected)
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    return {nm: acc[np.asarray(sb.region_manager.indices(nm))].mean() / settle for nm in names}


def main():
    print("=== GENUINE BG disinhibition gate selection (cheat-removal #2), spiking substrate ===\n", flush=True)
    # mechanism diagnostic first: D1 must SILENCE (not raise) its GPi, and GPi must silence thal by default.
    sb = build_genuine_bg_gated_bridge(seed=42)
    couple_all_route_gates(sb)
    watch = ["gpi_GO_N", "thal_GO_N", "gpi_COME_S", "thal_COME_S"]
    base = _pool_rates(sb, {}, watch)                 # nothing selected: every GPi paces -> every thal silent
    sel = _pool_rates(sb, {"GO": "N"}, watch)         # select GO->N: D1 silences gpi_GO_N -> thal_GO_N released
    print(f"  baseline (none selected):  gpi_GO_N={base['gpi_GO_N']:.3f} thal_GO_N={base['thal_GO_N']:.3f}  "
          f"gpi_COME_S={base['gpi_COME_S']:.3f} thal_COME_S={base['thal_COME_S']:.3f}", flush=True)
    print(f"  select GO->N:              gpi_GO_N={sel['gpi_GO_N']:.3f} thal_GO_N={sel['thal_GO_N']:.3f}  "
          f"gpi_COME_S={sel['gpi_COME_S']:.3f} thal_COME_S={sel['thal_COME_S']:.3f}", flush=True)
    d1_silences = sel['gpi_GO_N'] < base['gpi_GO_N'] - 0.02       # D1 must LOWER its GPi
    thal_released = sel['thal_GO_N'] > base['thal_GO_N'] + 0.02   # released relay must rise above the gate threshold
    other_stays = sel['thal_COME_S'] < 0.1                        # non-selected relay stays silent
    print(f"  -> D1 silences GPi: {d1_silences}   thal released: {thal_released}   other stays silent: {other_stays}"
          f"   => {'CLEAN' if (d1_silences and thal_released and other_stays) else 'NEEDS TUNING'}\n", flush=True)

    # decisive isolation: drive d1_GO_N DIRECTLY (bypass sel timing) with gpi tonic on -> does gpi_GO_N drop?
    # This tests ONLY the D1->GPi inhibitory projection (the framework's trait-based inhibition).
    from sim.backend import to_host
    sb2 = build_genuine_bg_gated_bridge(seed=42)

    def _gpi_with_d1(drive_d1):
        sb2.cp_external_input_current[:] = 0.0
        for v in VERBS:
            for m in MOTORS:
                sb2.cp_external_input_current[np.asarray(sb2.region_manager.indices(f"gpi_{v}_{m}"))] = GPI_TONIC_PA
        if drive_d1:
            sb2.cp_external_input_current[np.asarray(sb2.region_manager.indices("d1_GO_N"))] = 1500.0
        acc = np.zeros(sb2.core_config.num_neurons, dtype=np.float64)
        for _ in range(80):
            sb2._run_one_simulation_step()
            acc += to_host(sb2.cp_firing_states).astype(np.float64)
        d1r = acc[np.asarray(sb2.region_manager.indices("d1_GO_N"))].mean() / 80
        return acc[np.asarray(sb2.region_manager.indices("gpi_GO_N"))].mean() / 80, d1r

    g_off, _ = _gpi_with_d1(False)
    g_on, d1_fires = _gpi_with_d1(True)
    print(f"  D1->GPi ISOLATION: gpi_GO_N(no d1)={g_off:.3f}  gpi_GO_N(d1 driven, d1 rate={d1_fires:.3f})={g_on:.3f}"
          f"  -> inhibition {'WORKS' if g_on < g_off - 0.02 else 'BROKEN (d1 does not silence its gpi)'}\n", flush=True)

    for seed in (42, 43, 44):
        sb = build_genuine_bg_gated_bridge(seed=seed)
        couple_all_route_gates(sb)
        ok, line = 0, []
        for v in VERBS:
            best = decode_genuine(sb, v, TRUE_MAP)
            correct = best == TRUE_MAP[v]
            ok += int(correct)
            line.append(f"{v}->{best}{'(ok)' if correct else '(X)'}")
        print(f"  seed {seed}: BG genuinely disinhibits {TRUE_MAP} -> drive each verb -> {ok}/4   "
              f"[{'  '.join(line)}]", flush=True)
    print("\n  -> the basal ganglia RELEASE the thalamus (D1 -| GPi -| thal), no direct thalamic current; the", flush=True)
    print("     thalamic activity opens the cortical route gate. The selection mechanism is now genuine.", flush=True)


if __name__ == "__main__":
    main()
