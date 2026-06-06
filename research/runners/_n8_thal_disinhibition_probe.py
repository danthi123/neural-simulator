"""N8 verification + weight-scale probe for the navigation BG cascade.

STEP 1 of the N8 cheat conversion (thalamic tonic drive -> genuine GPi disinhibition).

Builds the REAL navigation BG regions/pathways (research.runners.g11_bg_runner.build_bg_brain_regions
with the flagship cluster-A/E config) and measures GPi + thalamus firing per action pool under two
drive regimes, with and without a single cortex pool driven (the "selection"):

  - TONIC  (the current cheat): thal_X <- 300 pA, gpi_X <- 110 pA, gpe/stn/snc <- 150 pA.
  - GENUINE (the port):         thal_X <- THAL_TONIC pA, gpi_X <- GPI_TONIC pA, NO direct thal drive
                                beyond THAL_TONIC; thalamus expresses only when its GPi is silenced
                                by the selected action's D1 (which the cortex pool drives via
                                cortex -> D1 -> GPi disinhibition).

The diagnostic answers STEP 1's questions:
  - Is GPi->thal wired? (yes - weight 8.0, build_bg_brain_regions line ~1171)
  - Is D1->GPi wired?   (yes - weight 15.0, line ~1116)
  - Under TONIC, does thalamus fire for ALL actions regardless of selection (the cheat)?
  - Under GENUINE, does driving cortex_<sel> silence gpi_<sel> and RELEASE thal_<sel> while
    leaving the non-selected thal pools silent? (the whole point of disinhibition)

  SIM_BACKEND=numpy python -m research.runners._n8_thal_disinhibition_probe
"""
import numpy as np

from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES

# Genuine-disinhibition weight scales ported from gated_compose_bg_genuine_demo.py (VALIDATED).
GPI_TONIC_PA = 2200.0   # tonic GPi pacemaker drive (silences thal by default)
THAL_TONIC_PA = 600.0   # tonic thalamic excitation (expressed only when GPi releases the relay)
# The cortex->D1->GPi->thal pathway weights ALREADY exist in build_bg_brain_regions at the
# validated scale: cortex->D1 (~125 effective), D1->GPi=15, GPi->thal=8.


def _build_nav_bridge(seed=42):
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    regions, pathways = build_bg_brain_regions(
        n_cortex=100,
        enable_bg_lateral_inhibition=True,   # flagship A+E config (wiring-level flags only;
        enable_striatal_fsis=True,           # d1_d2_asymmetry is a cfg/plasticity flag, not wiring)
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
    )
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False
    cfg.ou_std_current_pA = 0.0
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _idx(sb, name):
    return np.asarray(sb.region_manager.indices(name))


def _rates(sb, regime, selected=None, cortex_drive=800.0, settle=80):
    """Drive the BG with `regime` ('tonic' or 'genuine'), optionally driving cortex_<selected>,
    settle `settle` steps, return mean firing rate per neuron for gpi_X and thal_X pools."""
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    region_names = set(sb.region_manager.region_indices_dict().keys())
    # Shared BG operating-point drives (same in both regimes for the indirect/STN/SNc loop).
    for a in ACTION_NAMES:
        sb.cp_external_input_current[_idx(sb, f"gpe_{a}")] = 150.0
        if f"gpe_arky_{a}" in region_names:
            sb.cp_external_input_current[_idx(sb, f"gpe_arky_{a}")] = 120.0
    sb.cp_external_input_current[_idx(sb, "stn")] = 150.0
    sb.cp_external_input_current[_idx(sb, "snc")] = 150.0
    if regime == "tonic":
        for a in ACTION_NAMES:
            sb.cp_external_input_current[_idx(sb, f"gpi_{a}")] = 110.0
            sb.cp_external_input_current[_idx(sb, f"thal_{a}")] = 300.0
    elif regime == "genuine":
        for a in ACTION_NAMES:
            sb.cp_external_input_current[_idx(sb, f"gpi_{a}")] = GPI_TONIC_PA
            sb.cp_external_input_current[_idx(sb, f"thal_{a}")] = THAL_TONIC_PA
    else:
        raise ValueError(regime)
    if selected is not None:
        sb.cp_external_input_current[_idx(sb, f"cortex_{selected}")] = cortex_drive
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    gpi = {a: acc[_idx(sb, f"gpi_{a}")].mean() / settle for a in ACTION_NAMES}
    thal = {a: acc[_idx(sb, f"thal_{a}")].mean() / settle for a in ACTION_NAMES}
    d1 = {a: acc[_idx(sb, f"str_D1_{a}")].mean() / settle for a in ACTION_NAMES}
    motor = {a: acc[_idx(sb, f"motor_{a}")].mean() / settle for a in ACTION_NAMES}
    return gpi, thal, d1, motor


def _fmt(d):
    return " ".join(f"{a}={d[a]:.3f}" for a in ACTION_NAMES)


def main():
    print("=== N8 nav-BG disinhibition probe (build_bg_brain_regions, flagship A+E config) ===\n",
          flush=True)
    sb = _build_nav_bridge(seed=42)

    print("--- TONIC regime (the current cheat: thal<-300, gpi<-110) ---", flush=True)
    g0, t0, d0, m0 = _rates(sb, "tonic", selected=None)
    print(f"  no selection : gpi[{_fmt(g0)}]", flush=True)
    print(f"                 thal[{_fmt(t0)}]  motor[{_fmt(m0)}]", flush=True)
    gN, tN, dN, mN = _rates(sb, "tonic", selected="N")
    print(f"  cortex_N on  : gpi[{_fmt(gN)}]", flush=True)
    print(f"                 thal[{_fmt(tN)}]  motor[{_fmt(mN)}]", flush=True)
    # In TONIC: thal fires for ALL actions whether or not selected -> the cheat.
    tonic_all_fire = all(t0[a] > 0.05 for a in ACTION_NAMES)
    print(f"  -> thal fires for ALL actions with NO selection: {tonic_all_fire} "
          f"(=> tonic short-circuits the gate; thalamus externally paced)\n", flush=True)

    print("--- GENUINE regime (port: thal<-600 tonic, gpi<-2200 pacemaker, NO direct thal selection) ---",
          flush=True)
    g0g, t0g, d0g, m0g = _rates(sb, "genuine", selected=None)
    print(f"  no selection : gpi[{_fmt(g0g)}]", flush=True)
    print(f"                 thal[{_fmt(t0g)}]  motor[{_fmt(m0g)}]", flush=True)
    for sel in ACTION_NAMES:
        gg, tg, dg, mg = _rates(sb, "genuine", selected=sel)
        released = tg[sel] > t0g[sel] + 0.02
        others_silent = all(tg[a] < 0.1 for a in ACTION_NAMES if a != sel)
        gpi_dropped = gg[sel] < g0g[sel] - 0.02
        motor_wins = (mg[sel] >= max(mg[a] for a in ACTION_NAMES)) and mg[sel] > 0.02
        verdict = "CLEAN" if (released and others_silent and gpi_dropped and motor_wins) else "NEEDS TUNING"
        print(f"  select {sel}: d1_{sel}={dg[sel]:.3f} gpi_{sel} {g0g[sel]:.3f}->{gg[sel]:.3f} "
              f"thal_{sel} {t0g[sel]:.3f}->{tg[sel]:.3f}  thal[{_fmt(tg)}]  motor[{_fmt(mg)}]  "
              f"=> released={released} others_silent={others_silent} gpi_dropped={gpi_dropped} "
              f"motor_wins={motor_wins} [{verdict}]", flush=True)

    # --- Cheap-first weight-scale sweep: vary GPi tonic + cortex drive to find a regime where
    # D1 fully silences its GPi and thal_<sel> + motor_<sel> rise cleanly above the others. ---
    print("\n--- GENUINE weight-scale sweep (select=N), settle=120 ---", flush=True)
    print(f"  {'gpi_tonic':>9} {'thal_tonic':>10} {'cortex':>7} | {'d1_N':>6} {'gpi_N(none->N)':>16} "
          f"{'thal_N(none->N)':>16} {'motor_N':>8} {'motor_max_other':>15}", flush=True)
    global GPI_TONIC_PA, THAL_TONIC_PA
    for gpi_tonic in (2200.0, 1500.0, 1000.0):
        for thal_tonic in (600.0, 400.0):
            for cortex in (800.0, 1200.0, 1600.0):
                GPI_TONIC_PA, THAL_TONIC_PA = gpi_tonic, thal_tonic
                gn, tn, dn, mn = _rates(sb, "genuine", selected=None, settle=120)
                gs, ts, ds, ms = _rates(sb, "genuine", selected="N", cortex_drive=cortex, settle=120)
                other = max(ms[a] for a in ACTION_NAMES if a != "N")
                clean = (ts["N"] > tn["N"] + 0.02 and ms["N"] > 0.02 and ms["N"] > other + 0.01)
                print(f"  {gpi_tonic:>9.0f} {thal_tonic:>10.0f} {cortex:>7.0f} | {ds['N']:>6.3f} "
                      f"{gn['N']:>7.3f}->{gs['N']:<7.3f} {tn['N']:>7.3f}->{ts['N']:<7.3f} "
                      f"{ms['N']:>8.3f} {other:>15.3f} {'<= CLEAN' if clean else ''}", flush=True)
    GPI_TONIC_PA, THAL_TONIC_PA = 2200.0, 600.0

    # --- Production-faithful selection test: replicate the exact trial readout window. ---
    # The smoke gate failed because motor fires on only ~5% of trials -> 95% fall through to the
    # random-action fallback. This sweeps thal_tonic (the released relay's headroom) measuring the
    # PRODUCTION metric: total motor spikes over the 30-100ms readout window (70 steps x 10 neurons),
    # for each of the 4 single-pool selections, with heuristic-style 800 pA cortex drive.
    print("\n--- PRODUCTION-faithful selection (single cortex pool @ 800 pA, 100-step window, "
          "readout 30-100) ---", flush=True)
    print(f"  {'gpi_tonic':>9} {'thal_tonic':>10} | per-selection motor_counts [sel -> (N,E,S,W)]  "
          f"| correct/4  any-fired/4", flush=True)
    for gpi_tonic in (1000.0, 1500.0):
        for thal_tonic in (600.0, 900.0, 1200.0, 1500.0):
            GPI_TONIC_PA, THAL_TONIC_PA = gpi_tonic, thal_tonic
            n_correct, n_anyfired = 0, 0
            details = []
            for sel in ACTION_NAMES:
                mc = _production_motor_counts(sb, sel, cortex_drive=800.0)
                top = max(ACTION_NAMES, key=lambda a: mc[a])
                anyf = sum(mc.values()) > 0
                if anyf:
                    n_anyfired += 1
                    if top == sel:
                        n_correct += 1
                details.append(f"{sel}->({mc['N']},{mc['E']},{mc['S']},{mc['W']})")
            print(f"  {gpi_tonic:>9.0f} {thal_tonic:>10.0f} | {'  '.join(details)}  "
                  f"| {n_correct}/4  {n_anyfired}/4", flush=True)
    GPI_TONIC_PA, THAL_TONIC_PA = 1000.0, 600.0


def _production_motor_counts(sb, selected, cortex_drive=800.0):
    """Replicate g11_bg_runner's exact trial: set genuine drives + cortex_<selected>, run 100 substeps,
    count motor spikes per pool over the 30..100 readout window (matching READOUT_START/END_MS)."""
    from sim.backend import to_host
    sb.cp_external_input_current[:] = 0.0
    region_names = set(sb.region_manager.region_indices_dict().keys())
    for a in ACTION_NAMES:
        sb.cp_external_input_current[_idx(sb, f"gpe_{a}")] = 150.0
        if f"gpe_arky_{a}" in region_names:
            sb.cp_external_input_current[_idx(sb, f"gpe_arky_{a}")] = 120.0
        sb.cp_external_input_current[_idx(sb, f"gpi_{a}")] = GPI_TONIC_PA
        sb.cp_external_input_current[_idx(sb, f"thal_{a}")] = THAL_TONIC_PA
    sb.cp_external_input_current[_idx(sb, "stn")] = 150.0
    sb.cp_external_input_current[_idx(sb, "snc")] = 150.0
    sb.cp_external_input_current[_idx(sb, f"cortex_{selected}")] = cortex_drive
    mc = {a: 0 for a in ACTION_NAMES}
    idx = {a: _idx(sb, f"motor_{a}") for a in ACTION_NAMES}
    for s in range(100):
        sb._run_one_simulation_step()
        if 30 <= s < 100:
            firing = to_host(sb.cp_firing_states).astype(bool)
            for a in ACTION_NAMES:
                mc[a] += int(firing[idx[a]].sum())
    return mc


if __name__ == "__main__":
    main()
