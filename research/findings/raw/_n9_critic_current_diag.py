"""N9 forensic: WHY is the afferent->critic excitatory current ~20x weaker in the
DEPLOYED nav bridge than in the ISOLATION probe, on identical warm-up drive?

Builds BOTH bridges (deployed nav-critic via build_bg_brain_regions + Gabor growth;
isolation via snc_stageb_critic_probe_navfaithful._build_navfaithful_bridge), injects
the IDENTICAL Gaussian place-code drive into vs_place_context, and instruments the
striosome_value pool: per-step g_e, g_i, membrane V; afferent firing rate; and the
vs_place_context->striosome_value CSR slice (n synapses, mean weight, row/col targets).

READ-ONLY on sim/. SIM_BACKEND=numpy. Short run only (no full nav).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
from sim.backend import get_backend
xp, _bk = get_backend()
print("backend:", _bk)


def _idx(bridge, name):
    return list(bridge.region_manager.indices(name))


def _csr_slice(bridge, pre_idx, post_idx):
    """Return (n_syn, mean_w, frac_pre_ok, frac_post_ok) for edges whose row in
    pre_idx and col in post_idx, pulled from cp_connections."""
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(coo.row); cols = np.asarray(coo.col); data = np.asarray(coo.data)
    pre_set = set(int(i) for i in pre_idx); post_set = set(int(i) for i in post_idx)
    mask = np.array([(int(r) in pre_set and int(c) in post_set) for r, c in zip(rows, cols)])
    if mask.sum() == 0:
        return 0, float("nan"), 0.0, 0.0
    w = data[mask]
    # how many distinct post-cells are actually targeted
    tgt_posts = set(int(c) for c in cols[mask])
    return int(mask.sum()), float(w.mean()), float(w.min()), float(w.max()), len(tgt_posts)


def _gaussian_place_drive(prefs_x, prefs_y, gx, gy, max_pA, sigma):
    dsq = (np.asarray(prefs_x) - float(gx)) ** 2 + (np.asarray(prefs_y) - float(gy)) ** 2
    return (max_pA * np.exp(-dsq / (2.0 * sigma ** 2))).astype(np.float32)


def instrument(bridge, aff_global, crit_global, snc_global, aff_drive, snc_tonic_pa,
               snc_reward_gain, n_steps=120, label=""):
    """Drive the afferent with aff_drive + snc tonic+reward; record per-step
    g_e/g_i/V over the critic pool and afferent firing rate."""
    aff_cp = xp.asarray(aff_global, dtype=xp.int64)
    crit_cp = xp.asarray(crit_global, dtype=xp.int64)
    snc_cp = xp.asarray(snc_global, dtype=xp.int64)
    aff_drive_cp = xp.asarray(aff_drive, dtype=xp.float32)
    n_aff = len(aff_global); n_crit = len(crit_global)
    bridge.core_config.current_reward_signal = 1.0
    ge_log, gi_log, v_log, aff_rate_log, crit_spk_log = [], [], [], [], []
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[aff_cp] = aff_drive_cp
        bridge.cp_external_input_current[snc_cp] = xp.float32(snc_tonic_pa + snc_reward_gain)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * bridge.core_config.dt_ms
        ge = np.asarray(bridge.cp_conductance_g_e)[np.asarray(crit_global)]
        gi = np.asarray(bridge.cp_conductance_g_i)[np.asarray(crit_global)]
        v = np.asarray(bridge.cp_membrane_potential_v)[np.asarray(crit_global)]
        ge_log.append(float(ge.mean())); gi_log.append(float(gi.mean())); v_log.append(float(v.mean()))
        aff_fired = np.asarray(bridge.cp_firing_states)[np.asarray(aff_global)]
        aff_rate_log.append(float(aff_fired.sum()) / max(n_aff, 1))
        crit_fired = np.asarray(bridge.cp_firing_states)[np.asarray(crit_global)]
        crit_spk_log.append(int(crit_fired.sum()))
    # last-half means (after transient)
    h = n_steps // 2
    print(f"\n=== {label} ===")
    print(f"  critic pool n={n_crit}, afferent n={n_aff}")
    print(f"  g_e (critic)  last-half mean = {np.mean(ge_log[h:]):.4f}  (max over run {max(ge_log):.4f})")
    print(f"  g_i (critic)  last-half mean = {np.mean(gi_log[h:]):.4f}  (max over run {max(gi_log):.4f})")
    print(f"  V   (critic)  last-half mean = {np.mean(v_log[h:]):.3f} mV  (max V reached {max(v_log):.3f})")
    print(f"  afferent firing frac last-half = {np.mean(aff_rate_log[h:]):.4f}  (~{np.mean(aff_rate_log[h:])*1000:.1f} Hz if 1ms dt)")
    print(f"  critic spikes total over {n_steps} steps = {sum(crit_spk_log)}")
    return dict(ge=np.mean(ge_log[h:]), gi=np.mean(gi_log[h:]), v_mean=np.mean(v_log[h:]),
                v_max=max(v_log), aff_rate=np.mean(aff_rate_log[h:]), crit_spk=sum(crit_spk_log))


# ===========================================================================
# 1. ISOLATION bridge (the de-risk probe that PASSES)
# ===========================================================================
def build_isolation(seed=42):
    from research.runners.snc_stageb_critic_probe_navfaithful import (
        _build_navfaithful_bridge, _grid_prefs, grid_place_code_drive,
    )
    bridge, cfg = _build_navfaithful_bridge(
        seed, grid_size=32,
        vs_place_to_strio_weight=0.2, strio_to_snc_weight=10.0,
        snc_da_sensitivity=8.0, reward_learning_rate=0.12,
        gabab=True, gabab_tau_decay=150.0, gabab_propagation_strength=0.02,
        critic_homeostasis=True, afferent_homeostasis=True)
    n_vs = len(_idx(bridge, "vs_place_context"))
    prefs = _grid_prefs(n_vs, 32)
    return bridge, cfg, prefs


# ===========================================================================
# 2. DEPLOYED nav-critic bridge (build_bg_brain_regions + Gabor growth)
# ===========================================================================
def build_deployed(seed=42, with_gabor=True, full_flags=False):
    import research.runners.g11_bg_runner as R
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.enums import NeuronModel

    # Visual-cortex Gabor layout params (nav flagship defaults: 8 orient / 2 freq / 8 pos).
    VN_ORIENT, VN_FREQ, VN_POS, VIMG = 8, 2, 8, 32
    extra = dict(enable_bg_lateral_inhibition=True, enable_striatal_fsis=True) if full_flags else {}
    regions, pathways = R.build_bg_brain_regions(
        n_cortex=100,
        enable_neural_critic=True,
        enable_critic_homeostasis=True,
        n_vs_place_context=200,
        vs_place_to_value_weight=0.2,
        vs_place_to_value_density=0.5,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_visual_cortex=with_gabor,
        visual_n_orientations=VN_ORIENT,
        visual_n_frequencies=VN_FREQ,
        visual_n_positions_per_dim=VN_POS,
        visual_image_size=VIMG,
        **extra,
    )

    cfg = CoreSimConfig()
    cfg.seed = seed; cfg.heterogeneity_seed = seed; cfg.ou_seed = seed
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.12
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    # deterministic regime (g11_bg_runner.py:3446-3451)
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_structural_plasticity = False
    _ctx_msn_density = 0.20
    _ctx_msn_weight = (25.0 / _ctx_msn_density)
    cfg.stdp_w_max = max(30.0, _ctx_msn_weight * 1.2)
    # GABA_B (neural critic)
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.02
    if full_flags:
        cfg.enable_d1_d2_asymmetry = True   # the real flagship sets this (builds cp_d1_d2_sign)

    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    # dopamine neuromodulator (matching nav: plasticity_rate / all, from_region_firing_signed snc)
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0,
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(
                rule_type="from_region_firing_signed", sensitivity=8.0,
                threshold=0.30, window_ms=200.0, source_regions=["snc"])],
        )
    ]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    nnz_before = int(bridge.cp_connections.nnz)
    # afferent->critic CSR slice BEFORE gabor growth
    aff_g = _idx(bridge, "vs_place_context"); crit_g = _idx(bridge, "striosome_value")
    slice_before = _csr_slice(bridge, aff_g, crit_g)

    if with_gabor:
        from sim.visual_cortex import apply_v1_gabor_weights
        n_gabor = apply_v1_gabor_weights(
            bridge, n_orientations=VN_ORIENT, n_frequencies=VN_FREQ,
            n_positions_per_dim=VN_POS, retina_size=VIMG, weight_scale=1.0)
        print(f"  [deployed] Gabor grew/updated {n_gabor} synapses; nnz {nnz_before} -> {int(bridge.cp_connections.nnz)}")

    slice_after = _csr_slice(bridge, aff_g, crit_g)
    print(f"  [deployed] afferent->critic CSR slice BEFORE gabor: n={slice_before[0]} mean_w={slice_before[1]:.4f} "
          f"min={slice_before[2]:.4f} max={slice_before[3]:.4f} n_target_posts={slice_before[4]}")
    print(f"  [deployed] afferent->critic CSR slice AFTER  gabor: n={slice_after[0]} mean_w={slice_after[1]:.4f} "
          f"min={slice_after[2]:.4f} max={slice_after[3]:.4f} n_target_posts={slice_after[4]}")

    # preferred (x,y) for vs_place_context (mirror g11_bg_runner.py:3370-3380)
    n_vs = len(aff_g)
    _vs_side = int(round(n_vs ** 0.5))
    _vs_xs = np.linspace(0.0, 31.0, _vs_side, dtype=np.float32)
    _vs_ys = np.linspace(0.0, 31.0, _vs_side, dtype=np.float32)
    _gx, _gy = np.meshgrid(_vs_xs, _vs_ys)
    _px = _gx.ravel(); _py = _gy.ravel()
    if _px.size < n_vs:
        reps = int(np.ceil(n_vs / max(_px.size, 1)))
        _px = np.tile(_px, reps)[:n_vs]; _py = np.tile(_py, reps)[:n_vs]
    prefs = (_px[:n_vs].copy(), _py[:n_vs].copy())
    return bridge, cfg, prefs


def calibrate_da(bridge, cfg, snc_global, snc_tonic_pa, n_steps=300):
    """Mirror _calibrate_da_threshold: set dopamine rule threshold to SNc tonic frac."""
    snc_cp = xp.asarray(snc_global, dtype=xp.int64); n_snc = len(snc_global)
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    bridge.cp_external_input_current[snc_cp] = xp.float32(snc_tonic_pa)
    frac_sum = 0.0; m = 0
    for i in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        if i >= n_steps // 2:
            frac_sum += float(np.asarray(bridge.cp_firing_states)[np.asarray(snc_global)].sum()) / max(n_snc, 1); m += 1
    frac = frac_sum / max(m, 1)
    cfg.neuromodulators[0].production_rules[0].threshold = float(frac)
    return frac


def afferent_rate(bridge, aff_global, aff_drive, n_steps=40, warmup=10):
    """Measure afferent firing rate (Hz) under aff_drive, learning frozen."""
    aff_cp = xp.asarray(aff_global, dtype=xp.int64); n_aff = len(aff_global)
    saved = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    bridge.cp_external_input_current[:] = xp.float32(0.0)
    bridge.cp_external_input_current[aff_cp] = xp.asarray(aff_drive, dtype=xp.float32)
    spk = 0; m = 0
    for t in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * bridge.core_config.dt_ms
        if t >= warmup:
            spk += int(np.asarray(bridge.cp_firing_states)[np.asarray(aff_global)].sum()); m += 1
    bridge.core_config.reward_learning_rate = saved
    return spk / max(n_aff, 1) / max(m * 1e-3, 1e-9)


def train_value_leads_reward(bridge, cfg, aff_global, crit_global, snc_global, near_drive,
                             snc_tonic_pa, snc_reward_gain, n_train=40, hold=40, label=""):
    """Mirror run_navfaithful's training loop on ANY bridge: ITI floor, clear eligibility,
    LEARN (place + reward burst). Track per-trial critic rate, afferent rate, weight."""
    aff_cp = xp.asarray(aff_global, dtype=xp.int64)
    crit_cp = xp.asarray(crit_global, dtype=xp.int64)
    snc_cp = xp.asarray(snc_global, dtype=xp.int64)
    near_cp = xp.asarray(near_drive, dtype=xp.float32)
    n_crit = len(crit_global)

    def mean_w():
        return _csr_slice(bridge, aff_global, crit_global)[1]

    print(f"\n=== TRAINING {label} (value-leads-reward, {n_train} trials) ===")
    print(f"  w_init = {mean_w():.4f}")
    cfg_da = calibrate_da(bridge, cfg, snc_global, snc_tonic_pa)
    print(f"  DA threshold calibrated to SNc tonic frac = {cfg_da:.4f}")
    for t in range(n_train):
        # ITI floor
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[snc_cp] = xp.float32(snc_tonic_pa)
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(hold):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
        # clear eligibility
        if bridge.cp_eligibility_trace is not None:
            bridge.cp_eligibility_trace[:] = xp.float32(0.0)
        # LEARN: place + reward burst
        bridge.cp_external_input_current[:] = xp.float32(0.0)
        bridge.cp_external_input_current[aff_cp] = near_cp
        bridge.cp_external_input_current[snc_cp] = xp.float32(snc_tonic_pa + snc_reward_gain)
        bridge.core_config.current_reward_signal = 1.0
        crit_spk = 0
        for _ in range(hold):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cfg.dt_ms
            crit_spk += int(np.asarray(bridge.cp_firing_states)[np.asarray(crit_global)].sum())
        if t < 3 or t % 5 == 0 or t == n_train - 1:
            v_rate = crit_spk / max(n_crit, 1) / (hold * 1e-3)
            af = afferent_rate(bridge, aff_global, near_drive, n_steps=30)
            da = float(bridge.neuromodulator_manager.get_concentration("dopamine"))
            print(f"  [t={t:02d}] crit_rate={v_rate:6.2f}Hz  aff_rate={af:6.2f}Hz  w={mean_w():.4f}  DA={da:.3f}")
    print(f"  w_final = {mean_w():.4f}")
    return mean_w()


if __name__ == "__main__":
    SEED = 42
    GX, GY = 26.571, 26.571   # the de-risk's p_near
    SIGMA = 4.0
    DRIVE = 800.0
    SNC_TONIC = 220.0; SNC_REWARD = 400.0   # nav defaults

    print("\n########## BUILDING ISOLATION BRIDGE ##########")
    iso, iso_cfg, iso_prefs = build_isolation(SEED)
    iso_aff = _idx(iso, "vs_place_context"); iso_crit = _idx(iso, "striosome_value"); iso_snc = _idx(iso, "snc")
    iso_drive = _gaussian_place_drive(iso_prefs[0], iso_prefs[1], GX, GY, DRIVE, SIGMA)
    iso_slice = _csr_slice(iso, iso_aff, iso_crit)
    print(f"  [iso] afferent->critic CSR slice: n={iso_slice[0]} mean_w={iso_slice[1]:.4f} "
          f"min={iso_slice[2]:.4f} max={iso_slice[3]:.4f} n_target_posts={iso_slice[4]}")
    print(f"  [iso] drive: {int((iso_drive>1.0).sum())} cells > 1 pA, max {iso_drive.max():.1f} pA")

    print("\n########## BUILDING DEPLOYED BRIDGE (with Gabor) ##########")
    dep, dep_cfg, dep_prefs = build_deployed(SEED, with_gabor=True)
    dep_aff = _idx(dep, "vs_place_context"); dep_crit = _idx(dep, "striosome_value"); dep_snc = _idx(dep, "snc")
    dep_drive = _gaussian_place_drive(dep_prefs[0], dep_prefs[1], GX, GY, DRIVE, SIGMA)
    print(f"  [dep] drive: {int((dep_drive>1.0).sum())} cells > 1 pA, max {dep_drive.max():.1f} pA")
    print(f"  [dep] total neurons={dep_cfg.num_neurons}, synapses={int(dep.cp_connections.nnz)}")

    # sanity: are the two drives identical? (same prefs construction)
    print(f"\n  drive vectors equal (iso vs dep): {np.allclose(iso_drive, dep_drive)}; "
          f"max|diff|={np.max(np.abs(iso_drive - dep_drive)):.3e}")

    # ---- PHASE 1: cold (t=0) instrument under warm-up drive ----
    r_iso = instrument(iso, iso_aff, iso_crit, iso_snc, iso_drive, SNC_TONIC, SNC_REWARD,
                       n_steps=120, label="ISOLATION critic COLD (no adaptation)")
    r_dep = instrument(dep, dep_aff, dep_crit, dep_snc, dep_drive, SNC_TONIC, SNC_REWARD,
                       n_steps=120, label="DEPLOYED critic COLD (no adaptation)")

    print("\n\n########## COLD DIVERGENCE SUMMARY ##########")
    print(f"  critic g_e:   iso {r_iso['ge']:.4f}  vs  dep {r_dep['ge']:.4f}   "
          f"(ratio iso/dep = {r_iso['ge']/max(r_dep['ge'],1e-9):.2f})")
    print(f"  critic V mean:iso {r_iso['v_mean']:.3f} vs dep {r_dep['v_mean']:.3f} mV")
    print(f"  afferent rate:iso {r_iso['aff_rate']:.4f} vs dep {r_dep['aff_rate']:.4f} "
          f"(ratio iso/dep = {r_iso['aff_rate']/max(r_dep['aff_rate'],1e-9):.2f})")
    print(f"  critic spikes:iso {r_iso['crit_spk']} vs dep {r_dep['crit_spk']}")

    # ---- PHASE 2: DECISIVE — run BOTH through the isolation probe's 40-trial training ----
    # Rebuild fresh bridges so the cold instrument didn't pre-adapt them.
    print("\n\n########## DECISIVE TEST: 40-trial value-leads-reward training on BOTH ##########")
    iso2, iso2_cfg, iso2_prefs = build_isolation(SEED)
    iso2_aff = _idx(iso2, "vs_place_context"); iso2_crit = _idx(iso2, "striosome_value"); iso2_snc = _idx(iso2, "snc")
    iso2_drive = _gaussian_place_drive(iso2_prefs[0], iso2_prefs[1], GX, GY, DRIVE, SIGMA)
    w_iso_final = train_value_leads_reward(
        iso2, iso2_cfg, iso2_aff, iso2_crit, iso2_snc, iso2_drive,
        # isolation probe uses snc_tonic=180, reward_gain=300
        180.0, 300.0, n_train=40, hold=40, label="ISOLATION (probe knobs)")

    dep2, dep2_cfg, dep2_prefs = build_deployed(SEED, with_gabor=True)
    dep2_aff = _idx(dep2, "vs_place_context"); dep2_crit = _idx(dep2, "striosome_value"); dep2_snc = _idx(dep2, "snc")
    dep2_drive = _gaussian_place_drive(dep2_prefs[0], dep2_prefs[1], GX, GY, DRIVE, SIGMA)
    # Test BOTH with the isolation probe's exact knobs (so the ONLY difference is the bridge).
    w_dep_final = train_value_leads_reward(
        dep2, dep2_cfg, dep2_aff, dep2_crit, dep2_snc, dep2_drive,
        180.0, 300.0, n_train=40, hold=40, label="DEPLOYED (same probe knobs)")

    print("\n\n########## DECISIVE VERDICT ##########")
    print(f"  ISOLATION critic weight 0.20 -> {w_iso_final:.4f}  (bootstrapped: {w_iso_final > 0.25})")
    print(f"  DEPLOYED  critic weight 0.20 -> {w_dep_final:.4f}  (bootstrapped: {w_dep_final > 0.25})")
    if w_dep_final > 0.25:
        print("  => DEPLOYED critic DOES bootstrap with the probe's 40-trial protocol+knobs.")
        print("     The 1800-step nav negative is a WARM-UP-INSUFFICIENCY (trial count / knobs), NOT a deployment block.")
    else:
        print("  => DEPLOYED critic does NOT bootstrap even with the probe's protocol -> deployment-specific block (honest negative).")
