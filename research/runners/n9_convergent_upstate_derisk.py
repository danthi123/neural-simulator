"""N9 convergent-up-state value-critic de-risk (CuPy-only, two-region afferent).

The FAITHFUL mechanism (research/findings/2026-06-09-N9-faithful-value-cell-design.md, Option A),
with the wiring-collision fix (RegionManager keys pathways by (from,to), so two pathways from the
SAME region COLLIDE — sim/regions.py:537). The two arms are therefore TWO DISTINCT afferent
regions, both rendered with the SAME grid-32 Gaussian place code each step:

  A1  vs_place_drive   -> striosome_value : dense (0.8), NON-plastic, many weak synapses summing
                          PAST the MSN-D1 ~339 pA rheobase at the goal -> the convergent-excitation
                          up-state (B.02). Fires the cell from INIT (breaks the LTP bootstrap).
  A2  vs_place_context -> striosome_value : sparse (0.4), PLASTIC init 0.2, DA-delta-gated STDP ->
                          learns the place-specific V(s) ON TOP of the already-firing cell.

GATES (>=3 CuPy seeds, deterministic-nav regime):
  1 FIRE          : critic >= ~5 Hz at the goal (NEAR) on CuPy.
  2 PLACE-GRADED  : critic NEAR >> FAR (ratio >= 3x, far ~0) AFTER training. (NB: A1 alone is a
                    density-blob — gate 2 is the load-bearing test of whether A2's learned boost
                    makes the trained critic NEAR-selective.)
  3 LEARNS (LTP)  : the A2 near-ensemble weight GROWS from 0.2 and exceeds the far weight.
  4 ACTOR-NOT-PERTURBED : actor cortex firing within +-10% vs a critic-absent twin.
  5 GABA_B-lesion : zero the GABA_B mask -> the SNc predicted-vs-unpredicted gap vanishes.

ANTI-CHEATS:
  (a) population code, not coordinate (Jaccard < 0.5).
  (b) PLACE-SHUFFLE control: permute the place labels (the bump position decoupled from the
      learned ensemble); gates 2+3 must FAIL under the shuffle.
  (d) regime fidelity: assert backend==cupy AND OU/conductance-noise/global-homeostasis OFF.

USAGE (MUST be cupy):
  SIM_BACKEND=cupy python -m research.runners.n9_convergent_upstate_derisk --seeds 42,43,44 \
      --a1-weight 24 --out research/findings/raw/_n9_upstate_derisk.json
  SIM_BACKEND=cupy python -m research.runners.n9_convergent_upstate_derisk --seed 42 --lesion
  SIM_BACKEND=cupy python -m research.runners.n9_convergent_upstate_derisk --seed 42 --shuffle
"""
from __future__ import annotations
import argparse, json, os, sys
import statistics as _st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _idx(bridge, name):
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


# ---- grid-32 2-D Gaussian place code (the nav rendering; a distributed population pattern) ----
def _grid_prefs(n_cells, grid_size):
    side = int(round(np.sqrt(n_cells)))
    xs = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    ys = np.linspace(0.0, grid_size - 1.0, side, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    px = gx.ravel(); py = gy.ravel()
    if px.size < n_cells:
        reps = int(np.ceil(n_cells / px.size))
        px = np.tile(px, reps)[:n_cells]; py = np.tile(py, reps)[:n_cells]
    return px[:n_cells].copy(), py[:n_cells].copy()


def _place_code(pos_xy, prefs_xy, max_pA, sigma):
    px, py = prefs_xy
    x, y = float(pos_xy[0]), float(pos_xy[1])
    dsq = (px - x) ** 2 + (py - y) ** 2
    return (max_pA * np.exp(-dsq / (2.0 * sigma ** 2))).astype(np.float32)


def _build(seed, *, a1_weight, a1_density=0.8, a2_weight=0.2, a2_density=0.4,
           n_vs=200, n_strio=80, n_snc=30, grid_size=32, include_actor=True,
           n_sensor_place=64, n_cortex_per_action=50, sensor_place_to_cortex_weight=10.0,
           strio_to_snc_weight=10.0, snc_da_sensitivity=8.0, reward_learning_rate=0.12,
           gabab=True, gabab_tau_decay=150.0, gabab_propagation_strength=0.02,
           a1_to_critic_active=True, nmda_critic=False):
    """Two-region convergent-up-state critic + (optional) actor stub, deterministic-nav regime.

    a1_to_critic_active=False builds the critic-ABSENT twin for gate 4 (same actor wiring, A1
    weight 0 so the up-state arm injects nothing — but the regions/drive injection are identical,
    isolating whether the dense afferent leaks onto the actor through shared global state).
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0       # BRAIN-BASED: the SNc FIRING is the signal, not a host scalar
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 40.0
    # === deterministic-nav regime (g11_bg_runner.py:3340-3344) ===
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False
    if gabab:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = float(gabab_tau_decay)
        cfg.gabab_propagation_strength = float(gabab_propagation_strength)
    if nmda_critic:
        # Option B fidelity upgrade (design §4.5): per-region NMDA on the critic ONLY (the
        # same cp_nmda_neuron_mask --enable-pfc-nmda uses). NMDA sustains the up-state near
        # threshold (Pomata 2008) and is VOLTAGE-DEPENDENT (Mg block) -> it amplifies the cell
        # MORE where it is already depolarized (the A2-driven NEAR) than where only the A1 blob
        # drives it (FAR) -> could deepen the NEAR>>FAR grade. Scoped to the critic slice via
        # BrainRegion.enable_nmda=True below; global flag required to allocate g_nmda.
        cfg.enable_nmda = True

    regions = [
        # A1 up-state drive (dense, NON-plastic, many weak synapses) — the pre-wired corticostriatal
        # convergent excitation that puts the MSN in a location-gated up-state from init.
        BrainRegion(name="vs_place_drive", n_neurons=n_vs, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        # A2 plastic value learner.
        BrainRegion(name="vs_place_context", n_neurons=n_vs, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name),
        BrainRegion(name="striosome_value", n_neurons=n_strio, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0,
                    enable_nmda=bool(nmda_critic)),
        BrainRegion(name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_DOPAMINE.name,
                    syn_reversal_potential_i_override=-55.0),
    ]
    pathways = [
        RegionPathway(from_region="vs_place_drive", to_region="striosome_value",
                      density=float(a1_density),
                      weight_mean=float(a1_weight if a1_to_critic_active else 0.0),
                      weight_jitter=0.5, plastic=False),
        RegionPathway(from_region="vs_place_context", to_region="striosome_value",
                      density=float(a2_density), weight_mean=float(a2_weight),
                      weight_jitter=0.1, plastic=True, plasticity_gate="value_input"),
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False,
                      receptor=("gaba_b" if gabab else "gaba_a")),
    ]
    if include_actor:
        regions.append(BrainRegion(
            name="sensor_place_readout", n_neurons=n_sensor_place, exc_fraction=1.0,
            internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
            plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name))
        for ai, action in enumerate(("N", "E", "S", "W")):
            regions.append(BrainRegion(
                name=f"cortex_{action}", n_neurons=n_cortex_per_action, exc_fraction=1.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                plastic_internal=False, izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
                action_index=ai))
            pathways.append(RegionPathway(
                from_region="sensor_place_readout", to_region=f"cortex_{action}",
                density=1.0, weight_mean=float(sensor_place_to_cortex_weight),
                weight_jitter=0.2, plastic=True, plasticity_gate="place_goal_to_cortex"))

    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [NeuromodulatorConfig(
        name="dopamine", baseline=0.5, decay_tau_ms=200.0, concentration_min=0.0, concentration_max=2.0,
        targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
        production_rules=[ProductionRule(rule_type="from_region_firing_signed",
                                         sensitivity=float(snc_da_sensitivity), threshold=0.30,
                                         window_ms=200.0, source_regions=["snc"])])]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _assert_cupy_regime(cfg, backend_name):
    """Anti-cheat (d): the de-risk MUST run on cupy in the deterministic regime."""
    if backend_name != "cupy":
        raise AssertionError(
            f"REGIME FIDELITY (anti-cheat d): this critic de-risk MUST run on CuPy "
            f"(numpy is DISQUALIFIED — the aliasing class lived in the weak-drive/near-rest MSN "
            f"regime; see 2026-06-09-N9-cupy-membrane-divergence-ROOT.md). Got backend={backend_name!r}.")
    bad = [k for k in ("enable_ou_process", "enable_conductance_noise", "enable_homeostasis",
                       "enable_parameter_heterogeneity") if getattr(cfg, k, False)]
    if bad:
        raise AssertionError(f"REGIME FIDELITY (anti-cheat d): deterministic-regime knobs ON: {bad}")


def _step(bridge, n, xp):
    for _ in range(n):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _drive(bridge, idx_map, drive_regions, region_pa, n_steps, xp, freeze_lr=None, cfg=None):
    """drive_regions: {region_name: per-cell pA vector}. region_pa: {region_name: scalar pA}.
    Returns (snc_rate_hz, strio_rate_hz, mean_da)."""
    bridge.cp_external_input_current[:] = 0.0
    for rname, vec in drive_regions.items():
        if vec is not None:
            bridge.cp_external_input_current[idx_map[rname]] = xp.asarray(vec, dtype=xp.float32)
    for rname, pA in region_pa.items():
        bridge.cp_external_input_current[idx_map[rname]] = xp.float32(pA)
    saved = None
    if freeze_lr is not None and cfg is not None:
        saved = cfg.reward_learning_rate; cfg.reward_learning_rate = float(freeze_lr)
    snc_idx, strio_idx = idx_map["snc"], idx_map["striosome_value"]
    n_snc = len(_host(snc_idx)); n_strio = len(_host(strio_idx))
    snc_spk = strio_spk = 0; da_sum = 0.0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        snc_spk += int(bridge.cp_firing_states[snc_idx].sum())
        strio_spk += int(bridge.cp_firing_states[strio_idx].sum())
        da_sum += float(bridge.neuromodulator_manager.get_concentration("dopamine"))
    if saved is not None:
        cfg.reward_learning_rate = saved
    dur = n_steps * 1e-3
    return (snc_spk / max(n_snc, 1) / dur, strio_spk / max(n_strio, 1) / dur, da_sum / max(n_steps, 1))


def _calibrate_da(bridge, cfg, idx_map, tonic_pa, xp, n_steps=300):
    snc_idx = idx_map["snc"]; n_snc = len(_host(snc_idx))
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_idx] = xp.float32(tonic_pa)
    frac = 0.0; m = 0
    for i in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * cfg.dt_ms)
        if i >= n_steps // 2:
            frac += float(bridge.cp_firing_states[snc_idx].sum()) / max(n_snc, 1); m += 1
    tf = frac / max(m, 1)
    cfg.neuromodulators[0].production_rules[0].threshold = float(tf)
    return tf


def _mean_w(bridge, pre_name, post_name, pre_subset=None):
    pre = set(int(i) for i in _idx(bridge, pre_name)) if pre_subset is None else set(int(i) for i in pre_subset)
    post = set(int(i) for i in _idx(bridge, post_name))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row)); cols = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    m = np.isin(rows, list(pre)) & np.isin(cols, list(post))
    if not m.any():
        m = np.isin(rows, list(post)) & np.isin(cols, list(pre))
    return float(data[m].mean()) if m.any() else 0.0


def _ensemble_global(bridge, region, vec, frac=0.25):
    g = np.asarray(_idx(bridge, region), dtype=np.int64)
    drive = np.asarray(vec, dtype=np.float64)
    k = max(1, int(round(frac * len(drive))))
    top = np.argsort(drive)[-k:]
    return set(int(g[i]) for i in top)


def _critic_rate(bridge, idx_map, drive_regions, xp, n_steps=100, warmup=30, freeze_lr=0.0, cfg=None):
    _, r, _ = _drive(bridge, idx_map, drive_regions, {}, n_steps, xp, freeze_lr=freeze_lr, cfg=cfg)
    # _drive returns rate over ALL n_steps; re-measure just the post-warmup window for cleanliness.
    return r


def _critic_rate_windowed(bridge, idx_map, drive_regions, xp, n_steps=100, warmup=30):
    """Critic firing rate over a post-warmup window, learning frozen."""
    crit_idx = idx_map["striosome_value"]; n_crit = len(_host(crit_idx))
    saved = bridge.core_config.reward_learning_rate; bridge.core_config.reward_learning_rate = 0.0
    bridge.cp_external_input_current[:] = 0.0
    for rname, vec in drive_regions.items():
        if vec is not None:
            bridge.cp_external_input_current[idx_map[rname]] = xp.asarray(vec, dtype=xp.float32)
    spk = 0; m = 0
    for t in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        if t >= warmup:
            spk += int(bridge.cp_firing_states[crit_idx].sum()); m += 1
    bridge.core_config.reward_learning_rate = saved
    return spk / max(n_crit, 1) / max(m * 1e-3, 1e-9)


def _ensemble_overlap(bridge, idx_map, region, vec_a, vec_b, xp, n_steps=40, thresh_hz=1.0):
    pidx = idx_map[region]; n = len(_host(pidx))

    def active(vec):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[pidx] = xp.asarray(vec, dtype=xp.float32)
        c = np.zeros(n, dtype=np.int64)
        for _ in range(n_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
            c += np.asarray(_host(bridge.cp_firing_states[pidx])).astype(np.int64)
        rate = c / (n_steps * 1e-3)
        return set(int(i) for i in np.where(rate > thresh_hz)[0])

    sa, sb = active(vec_a), active(vec_b)
    return len(sa), len(sb), len(sa & sb) / max(len(sa | sb), 1)


def _lesion_gabab(bridge):
    m = getattr(bridge, "cp_gabab_synapse_mask", None)
    if m is None:
        return 0
    n_was = int(_host(m).sum())
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge.cp_gabab_synapse_mask = xp.zeros_like(m)
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    return n_was


def _measure_actor_rate(bridge, idx_map, sensor_vec, xp, *, cortex_tonic_pa=400.0,
                        a1_vec=None, a2_vec=None, n_steps=120, warmup=40):
    """Gate 4: actor cortex output rate under a fixed cortical tonic + the sparse actor place code,
    while ALSO driving the dense critic afferents (A1+A2) concurrently. The critic afferents have NO
    edge to cortex, so a faithful dedicated afferent leaves the cortex rate unchanged vs a twin."""
    sp_idx = idx_map["sensor_place_readout"]
    cortex_idx = xp.asarray(np.concatenate([np.asarray(_host(idx_map[f"cortex_{a}"])) for a in ("N", "E", "S", "W")]))
    n_cx = int(cortex_idx.size)
    saved = bridge.core_config.reward_learning_rate; bridge.core_config.reward_learning_rate = 0.0
    spk = 0; m = 0
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[cortex_idx] = xp.float32(cortex_tonic_pa)
        bridge.cp_external_input_current[sp_idx] = xp.asarray(sensor_vec, dtype=xp.float32)
        if a1_vec is not None and "vs_place_drive" in idx_map:
            bridge.cp_external_input_current[idx_map["vs_place_drive"]] = xp.asarray(a1_vec, dtype=xp.float32)
        if a2_vec is not None and "vs_place_context" in idx_map:
            bridge.cp_external_input_current[idx_map["vs_place_context"]] = xp.asarray(a2_vec, dtype=xp.float32)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = (
            bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)
        if t >= warmup:
            spk += int(bridge.cp_firing_states[cortex_idx].sum()); m += 1
    bridge.core_config.reward_learning_rate = saved
    return spk / max(n_cx, 1) / max(m * 1e-3, 1e-9)


def run_seed(seed, *, a1_weight=24.0, grid_size=32, p_near=(26.571, 26.571), p_far=(4.429, 4.429),
             vs_sigma=4.0, vs_drive_pa=800.0, sensor_sigma=1.5, sensor_drive_pa=1500.0,
             actor_cortex_tonic_pa=400.0, snc_tonic_pa=180.0, snc_reward_gain=300.0,
             hold_steps=40, n_train=40, lead_steps=150, reward_learning_rate=0.12,
             a2_weight=0.2, strio_to_snc_weight=10.0, snc_da_sensitivity=8.0,
             gabab=True, gabab_propagation_strength=0.02, nmda_critic=False,
             lesion=False, shuffle=False, verbose=True):
    from sim.backend import get_backend
    xp, bk = get_backend()

    bridge, cfg = _build(seed, a1_weight=a1_weight, a2_weight=a2_weight, grid_size=grid_size,
                         strio_to_snc_weight=strio_to_snc_weight, snc_da_sensitivity=snc_da_sensitivity,
                         reward_learning_rate=reward_learning_rate, gabab=gabab,
                         gabab_propagation_strength=gabab_propagation_strength, nmda_critic=nmda_critic)
    _assert_cupy_regime(cfg, bk)

    regions = ("vs_place_drive", "vs_place_context", "striosome_value", "snc",
               "sensor_place_readout", "cortex_N", "cortex_E", "cortex_S", "cortex_W")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
    n_vs = len(_host(idx_map["vs_place_drive"]))
    n_sp = len(_host(idx_map["sensor_place_readout"]))

    vs_prefs = _grid_prefs(n_vs, grid_size)
    near_vec = _place_code(p_near, vs_prefs, vs_drive_pa, vs_sigma)
    far_vec = _place_code(p_far, vs_prefs, vs_drive_pa, vs_sigma)

    # --- SHUFFLE control (anti-cheat b): permute the A2 (plastic) afferent's cell->preferred-(x,y)
    #     mapping so the bump position is DECOUPLED from the learned ensemble identity. The same
    #     "strong drive present" still fires the cell (A1 up-state intact), but the A2 synapses that
    #     potentiate are NO LONGER the ones whose preferred location is NEAR -> a learned
    #     value-of-LOCATION must FAIL (gates 2+3). A1 is left UNshuffled (it is the position-blind
    #     up-state by construction); only A2's place->learning correspondence is broken.
    a2_near_vec = near_vec; a2_far_vec = far_vec
    if shuffle:
        rng = np.random.RandomState(seed ^ 0x5A5A)
        perm = rng.permutation(n_vs)
        a2_near_vec = near_vec[perm].copy()
        a2_far_vec = far_vec[perm].copy()

    # Anti-cheat (a): NEAR vs FAR are distinct dense ensembles on the A1 afferent (provenance).
    na, nb, overlap = _ensemble_overlap(bridge, idx_map, "vs_place_drive", near_vec, far_vec, xp)
    distinct = (overlap < 0.5 and na > 0 and nb > 0)
    if verbose:
        print(f"  [anti-cheat a] NEAR active={na}/{n_vs}, FAR active={nb}/{n_vs}, "
              f"Jaccard={overlap:.2f} (distinct={distinct})")

    # === GATE 4 baseline: actor cortex firing on a critic-ABSENT twin (A1 weight 0) ===
    base_bridge, base_cfg = _build(seed, a1_weight=a1_weight, a2_weight=a2_weight, grid_size=grid_size,
                                   strio_to_snc_weight=strio_to_snc_weight, snc_da_sensitivity=snc_da_sensitivity,
                                   reward_learning_rate=reward_learning_rate, gabab=gabab,
                                   gabab_propagation_strength=gabab_propagation_strength,
                                   nmda_critic=nmda_critic, a1_to_critic_active=False)
    _assert_cupy_regime(base_cfg, bk)
    base_idx = {n: xp.asarray(_idx(base_bridge, n)) for n in
                ("sensor_place_readout", "cortex_N", "cortex_E", "cortex_S", "cortex_W",
                 "vs_place_drive", "vs_place_context")}
    sp_prefs = _grid_prefs(n_sp, grid_size)
    sensor_near = _place_code(p_near, sp_prefs, sensor_drive_pa, sensor_sigma)
    actor_no_critic = _measure_actor_rate(base_bridge, base_idx, sensor_near, xp,
                                          cortex_tonic_pa=actor_cortex_tonic_pa, a1_vec=None, a2_vec=None)
    del base_bridge

    tonic_frac = _calibrate_da(bridge, cfg, idx_map, snc_tonic_pa, xp)
    if verbose:
        print(f"  [calib] SNc tonic frac={tonic_frac:.4f}; gate-4 baseline actor(no-critic)={actor_no_critic:.2f} Hz")

    # A2 ensemble synapse sets (disjoint NEAR/FAR bump cores) for location-selective LTP tracking.
    near_set = _ensemble_global(bridge, "vs_place_context", a2_near_vec, frac=0.25)
    far_set = _ensemble_global(bridge, "vs_place_context", a2_far_vec, frac=0.25) - near_set
    w_near_init = _mean_w(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    w_far_init = _mean_w(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)

    # === FIRE gate (gate 1): the A1 up-state fires the critic at NEAR, at init (no learning yet) ===
    critic_fire_init = _critic_rate_windowed(bridge, idx_map,
                                             {"vs_place_drive": near_vec, "vs_place_context": a2_near_vec}, xp)
    if verbose:
        print(f"  [gate-1 FIRE] critic at NEAR (init, A1 up-state) = {critic_fire_init:.2f} Hz")

    # === value-leads-reward acquisition: visit NEAR + reward burst; FAR held out ===
    near_v_curve = []
    for t in range(n_train):
        _drive(bridge, idx_map, {"vs_place_drive": None, "vs_place_context": None}, {"snc": snc_tonic_pa},
               hold_steps, xp)            # ITI floor
        if getattr(bridge, "cp_eligibility_trace", None) is not None:
            bridge.cp_eligibility_trace[:] = 0.0
        # LEARN: drive BOTH A1 (up-state -> post-spike) and A2 (plastic) at NEAR + reward burst.
        _, strio_r, da = _drive(bridge, idx_map,
                                {"vs_place_drive": near_vec, "vs_place_context": a2_near_vec},
                                {"snc": snc_tonic_pa + snc_reward_gain}, hold_steps, xp)
        near_v_curve.append(strio_r)
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            wn = _mean_w(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
            wf = _mean_w(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)
            print(f"    [acq t={t:02d}] V(near)={strio_r:6.2f}Hz w_near={wn:.3f} w_far={wf:.3f} "
                  f"(near/far {wn/max(wf,1e-6):.2f}) DA={da:.3f}")

    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    near_v_early = _st.mean(near_v_curve[early]); near_v_late = _st.mean(near_v_curve[late])
    w_near_final = _mean_w(bridge, "vs_place_context", "striosome_value", pre_subset=near_set)
    w_far_final = _mean_w(bridge, "vs_place_context", "striosome_value", pre_subset=far_set)

    # === GATE 2 PLACE-GRADED: trained critic rate NEAR vs FAR (learning frozen) ===
    crit_near_trained = _critic_rate_windowed(bridge, idx_map,
                                              {"vs_place_drive": near_vec, "vs_place_context": a2_near_vec}, xp)
    crit_far_trained = _critic_rate_windowed(bridge, idx_map,
                                             {"vs_place_drive": far_vec, "vs_place_context": a2_far_vec}, xp)
    place_graded_ratio = crit_near_trained / max(crit_far_trained, 1e-3)
    place_graded = bool(crit_near_trained >= 5.0 and place_graded_ratio >= 3.0)

    # === GATE 4 (critic present): actor cortex firing on the trained bridge, critic afferents active ===
    actor_with_critic = _measure_actor_rate(bridge, idx_map, sensor_near, xp,
                                            cortex_tonic_pa=actor_cortex_tonic_pa,
                                            a1_vec=near_vec, a2_vec=a2_near_vec)
    actor_ratio = actor_with_critic / max(actor_no_critic, 1e-9)
    actor_ok = (0.90 <= actor_ratio <= 1.10) if actor_no_critic > 1e-6 else (actor_with_critic <= 1e-6)

    # ---- the SNc state-specific gap (the value-leads-reward LEAD test) + gate-5 lesion ----
    def _test(a1_vec, a2_vec, snc_pa):
        _drive(bridge, idx_map, {"vs_place_drive": None, "vs_place_context": None}, {"snc": snc_tonic_pa},
               hold_steps + 20, xp, freeze_lr=0.0, cfg=cfg)
        if lead_steps > 0 and a1_vec is not None:
            _drive(bridge, idx_map, {"vs_place_drive": a1_vec, "vs_place_context": a2_vec},
                   {"snc": snc_tonic_pa}, int(lead_steps), xp, freeze_lr=0.0, cfg=cfg)
        return _drive(bridge, idx_map, {"vs_place_drive": a1_vec, "vs_place_context": a2_vec},
                      {"snc": snc_pa}, hold_steps, xp, freeze_lr=0.0, cfg=cfg)

    if lesion:
        n_cut = _lesion_gabab(bridge)
        if verbose:
            print(f"  [gate-5 lesion] zeroed {n_cut} GABA_B synapses")

    pred_r, _, _ = _test(near_vec, a2_near_vec, snc_tonic_pa + snc_reward_gain)   # predicted (NEAR)
    unpred_r, _, _ = _test(far_vec, a2_far_vec, snc_tonic_pa + snc_reward_gain)   # unpredicted (FAR)
    gap_ratio = unpred_r / max(pred_r, 1e-6)
    state_specific = (unpred_r > 1.30 * max(pred_r, 1e-6)) and (unpred_r >= 10.0)

    # ---- gates ----
    fire = bool(critic_fire_init >= 5.0)
    weight_grew = bool(w_near_final > 1.05 * max(w_near_init, 1e-6)
                       and w_near_final > 1.05 * max(w_far_final, 1e-6))
    v_learned = bool(near_v_late > 1.10 * near_v_early)

    if verbose:
        print(f"  [gate-2 PLACE-GRADED] trained critic NEAR={crit_near_trained:.2f}Hz "
              f"FAR={crit_far_trained:.2f}Hz ratio={place_graded_ratio:.2f} (>=3 & near>=5 => {place_graded})")
        print(f"  [gate-3 LEARNS] w_near {w_near_init:.3f}->{w_near_final:.3f} "
              f"w_far {w_far_init:.3f}->{w_far_final:.3f} (LTP near>far => {weight_grew}); "
              f"V(near) {near_v_early:.2f}->{near_v_late:.2f}Hz")
        print(f"  [gate-4 ACTOR] no-critic={actor_no_critic:.2f}Hz with-critic={actor_with_critic:.2f}Hz "
              f"ratio={actor_ratio:.3f} (not-perturbed => {actor_ok})")
        print(f"  [SNc gap] predicted(NEAR)={pred_r:.2f}Hz unpredicted(FAR)={unpred_r:.2f}Hz "
              f"gap={gap_ratio:.2f} state-specific-above-floor={state_specific} (lesion={lesion})")

    return dict(
        seed=seed, backend=bk, lesion=lesion, shuffle=shuffle, a1_weight=a1_weight,
        na=na, nb=nb, overlap=overlap, distinct_ensembles=distinct,
        critic_fire_init_hz=critic_fire_init, fire=fire,
        crit_near_trained_hz=crit_near_trained, crit_far_trained_hz=crit_far_trained,
        place_graded_ratio=place_graded_ratio, place_graded=place_graded,
        w_near_init=w_near_init, w_near_final=w_near_final,
        w_far_init=w_far_init, w_far_final=w_far_final, weight_grew=weight_grew,
        near_v_early_hz=near_v_early, near_v_late_hz=near_v_late, v_learned=v_learned,
        actor_no_critic_hz=actor_no_critic, actor_with_critic_hz=actor_with_critic,
        actor_ratio=actor_ratio, actor_not_perturbed=actor_ok,
        test_predicted_near_hz=pred_r, test_unpredicted_far_hz=unpred_r,
        gap_ratio=gap_ratio, state_specific_above_floor=state_specific,
        ou_off=(not cfg.enable_ou_process), cond_noise_off=(not cfg.enable_conductance_noise),
        global_homeo_off=(not cfg.enable_homeostasis),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--a1-weight", type=float, default=24.0)
    ap.add_argument("--near-x", type=float, default=26.571)
    ap.add_argument("--near-y", type=float, default=26.571)
    ap.add_argument("--far-x", type=float, default=4.429)
    ap.add_argument("--far-y", type=float, default=4.429)
    ap.add_argument("--a2-weight", type=float, default=0.2)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--lead-ms", type=float, default=150.0)
    ap.add_argument("--snc-tonic-pa", type=float, default=180.0)
    ap.add_argument("--snc-reward-gain", type=float, default=300.0)
    ap.add_argument("--gabab-propagation-strength", type=float, default=0.02)
    ap.add_argument("--lesion", action="store_true", help="gate-5: zero GABA_B mask -> gap must vanish")
    ap.add_argument("--shuffle", action="store_true", help="anti-cheat b: permute A2 place labels -> gates 2+3 must FAIL")
    ap.add_argument("--no-gabab", action="store_true", help="GABA_A A/B control")
    ap.add_argument("--nmda-critic", action="store_true",
                    help="Option B: per-region NMDA on the critic (voltage-dependent up-state sustain)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    kw = dict(a1_weight=args.a1_weight, a2_weight=args.a2_weight, n_train=args.n_train,
              p_near=(args.near_x, args.near_y), p_far=(args.far_x, args.far_y),
              lead_steps=int(round(args.lead_ms / 1.0)), snc_tonic_pa=args.snc_tonic_pa,
              snc_reward_gain=args.snc_reward_gain, gabab=(not args.no_gabab),
              gabab_propagation_strength=args.gabab_propagation_strength,
              nmda_critic=args.nmda_critic, lesion=args.lesion, shuffle=args.shuffle)

    results = []
    for s in seeds:
        tag = ("LESION" if args.lesion else "SHUFFLE-CONTROL" if args.shuffle else
               "GABA_A A/B" if args.no_gabab else "convergent-up-state critic (GABA_B)")
        print(f"\n[n9-upstate seed={s}] {tag} (a1_w={args.a1_weight}):")
        r = run_seed(s, **kw)
        results.append(r)
        if not args.lesion and not args.shuffle:
            primary = r["fire"] and r["place_graded"] and r["weight_grew"] and r["actor_not_perturbed"]
            print(f"  => seed {s} PRIMARY {'PASS' if primary else 'FAIL'} "
                  f"[fire {r['fire']}, place-graded {r['place_graded']}, LTP {r['weight_grew']}, "
                  f"actor-ok {r['actor_not_perturbed']}]")

    if len(results) >= 1:
        n_fire = sum(1 for r in results if r["fire"])
        n_graded = sum(1 for r in results if r["place_graded"])
        n_ltp = sum(1 for r in results if r["weight_grew"])
        n_actor = sum(1 for r in results if r["actor_not_perturbed"])
        n_gap = sum(1 for r in results if r["state_specific_above_floor"])
        N = len(results)
        print(f"\n=== MULTI-SEED ({N} seeds, {tag}) ===")
        print(f"  FIRE(>=5Hz)               : {n_fire}/{N}")
        print(f"  PLACE-GRADED(near>=3xfar) : {n_graded}/{N}")
        print(f"  LEARNS(LTP near>far)      : {n_ltp}/{N}")
        print(f"  ACTOR-NOT-PERTURBED       : {n_actor}/{N}")
        print(f"  SNc state-specific gap    : {n_gap}/{N}  (LESION expects 0/N; GABA_B expects >=3/N)")
        if not args.lesion and not args.shuffle:
            n_primary = sum(1 for r in results if r["fire"] and r["place_graded"]
                            and r["weight_grew"] and r["actor_not_perturbed"])
            print(f"  PRIMARY (fire+graded+LTP+actor): {n_primary}/{N}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": ("lesion" if args.lesion else "shuffle" if args.shuffle
                                else "gaba_a" if args.no_gabab else "gabab"),
                       "deterministic_regime": True, "results": results}, f, indent=2, default=float)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
