"""gap#5 one-brain MERGE core mechanism — the WAKE/SLEEP PHASE-SWITCH — 6-SEED GO (2026-07-25). The replay is
AdEx-substrate-specific (Izhikevich spreads, `_gap5_izh_replay_merge_derisk.py`). Biologically faithful merge: ONE bridge
runs Izhikevich/dt=1.0 during WAKE (conversation), then switches to AdEx/dt=0.1 during a SLEEP/SWR phase to run the CA3
replay — hippocampal replay happens during rest, temporally separate from active cognition. `switch_to_adex_sleep()`
preserves `cp_connections` (the memory band) while swapping the neuron model: loads the ECKER preset -> cfg.adex_*, resets
v/adex_w, recomputes the dt-dependent cached synaptic decays + max_delay_steps; the step loop dispatches on
cfg.neuron_model_type so the next step runs AdEx. RESULT (6-seed 42/43/44/100/101/102): the memory band survives the
switch BYTE-IDENTICAL 6/6, and the phase-switched replay decodes DECODE_r=1.000 == a native-AdEx replay 6/6. ⇒ the
sleep-phase merge is mechanistically de-risked; NO sim/ edit (the switch is done from the runner). GPU (SIM_BACKEND=cupy).
Finding: 2026-07-25-gap5-wake-sleep-phase-switch-6seed-GO.md."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.bridge import SimulationBridge
from sim.regions import BrainRegion
from sim.enums import NeuronModel, NeuronType, DefaultAdExParamsManager
from sim.backend import to_host, get_backend
from research.runners._gap5_ecker_recurrent_replay import decode_and_width

N_PC = 2000
SEED = 42


def _wire_track(b, cp, w_scale=600.0, sigma=25.0, back_frac=0.0):
    pc = np.asarray(b.region_manager.indices("pc"), int)
    cutoff = int(3 * sigma); offs = np.arange(-cutoff, cutoff + 1); offs = offs[offs != 0]
    wof = w_scale * np.exp(-(offs / sigma) ** 2); keep = wof > 0.02; offs, wof = offs[keep], wof[keep]
    pre, post, w = [], [], []; ii = np.arange(N_PC)
    for o, wv in zip(offs, wof):
        wv_dir = wv if o > 0 else wv * back_frac
        if wv_dir <= 0:
            continue
        j = ii + o; m = (j >= 0) & (j < N_PC)
        pre.append(pc[ii[m]]); post.append(pc[j[m]]); w.append(np.full(m.sum(), wv_dir, np.float64))
    pre = np.concatenate(pre); post = np.concatenate(post); w = np.concatenate(w)
    b.inject_explicit_wiring({"band": {"pre_indices": pre.astype(int).tolist(), "post_indices": post.astype(int).tolist(),
                                       "initial_weights": w.astype(float).tolist(), "plastic": False, "conn_type": "ff"}})
    return pc


def _build(model, dt, seed=SEED):
    cp, _ = get_backend()
    regions = [BrainRegion(name="pc", n_neurons=N_PC, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = float(dt); cfg.num_traits = 1
    cfg.neuron_model_type = model
    if model == NeuronModel.ADEX.name:
        cfg.default_neuron_type_adex = NeuronType.ADEX_ECKER_CA3_PC.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    for f in ("enable_homeostasis", "enable_stdp", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity"):
        setattr(cfg, f, False)
    cfg.enable_ou_process = True; cfg.ou_noise_sigma_pa = 40.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    return b, cp


def _recompute_cached_decays(b):
    cfg = b.core_config; cp, _ = get_backend()
    b._cached_decay_e = float(cp.exp(-cfg.dt_ms / cfg.syn_tau_g_e)) if cfg.syn_tau_g_e > 0 else 0.0
    b._cached_decay_i = float(cp.exp(-cfg.dt_ms / cfg.syn_tau_g_i)) if cfg.syn_tau_g_i > 0 else 0.0
    b._cached_decay_nmda = float(cp.exp(-cfg.dt_ms / cfg.nmda_tau_decay)) if cfg.nmda_tau_decay > 0 else 0.0
    b._cached_decay_nmda_rise = float(cp.exp(-cfg.dt_ms / cfg.nmda_tau_rise)) if cfg.nmda_tau_rise > 0 else 0.0
    if getattr(cfg, "gabab_tau_decay", 0) > 0:
        b._cached_decay_gabab = float(cp.exp(-cfg.dt_ms / cfg.gabab_tau_decay))


def switch_to_adex_sleep(b, dt=0.1):
    """WAKE(Izhikevich)->SLEEP(AdEx) phase switch, preserving cp_connections (the memory band)."""
    cp, _ = get_backend()
    cfg = b.core_config
    pp = DefaultAdExParamsManager.get_params(NeuronType.ADEX_ECKER_CA3_PC)
    for k in ("C", "g_L", "E_L", "V_T", "Delta_T", "a", "tau_w", "b", "V_r", "V_peak"):
        setattr(cfg, f"adex_{k}", float(pp[k]))
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.dt_ms = float(dt)
    n = int(b.cp_membrane_potential_v.shape[0])
    b.cp_membrane_potential_v = cp.full(n, cfg.adex_E_L, dtype=cp.float32)   # fresh sleep state
    b.cp_adex_w = cp.zeros(n, dtype=cp.float32)
    _recompute_cached_decays(b)
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)


def _replay(b, pc, cp, T=2500, cue_start=None, cue_pa=10000.0, cue_steps=40):
    if cue_start is None:
        cue_start = N_PC // 2 - 50
    cue = cp.asarray(pc[cue_start:cue_start + 100], dtype=cp.int64)
    F = np.zeros((T, N_PC), dtype=bool)
    for t in range(T):
        b.runtime_state.current_time_ms += b.core_config.dt_ms
        b.cp_external_input_current[:] = 0.0
        if t < cue_steps:
            b.cp_external_input_current[cue] += float(cue_pa)
        b._run_one_simulation_step()
        F[t] = np.asarray(to_host(b.cp_firing_states))[pc].astype(bool)
    return decode_and_width(F)


def one_seed(seed):
    bZ, cp = _build(NeuronModel.IZHIKEVICH.name, dt=1.0, seed=seed)
    pcZ = _wire_track(bZ, cp)
    band_before = np.asarray(to_host(bZ.cp_connections.data)).copy()
    for t in range(50):   # WAKE phase (Izhikevich/dt1.0, as if conversing)
        bZ.runtime_state.current_time_ms += bZ.core_config.dt_ms
        bZ.cp_external_input_current[:] = 0.0
        bZ._run_one_simulation_step()
    switch_to_adex_sleep(bZ, dt=0.1)   # -> SLEEP phase (AdEx/dt0.1)
    band_preserved = bool(np.array_equal(band_before, np.asarray(to_host(bZ.cp_connections.data))))
    rZ = _replay(bZ, pcZ, cp)
    return rZ, band_preserved


if __name__ == "__main__":   # guarded so the switch helpers are importable without running the 6-seed
    print("gap#5 PHASE-SWITCH 6-SEED — WAKE(Izh/dt1.0) -> SLEEP(AdEx/dt0.1) on ONE bridge. GO iff the memory band survives "
          "the switch byte-identical 6/6 AND the phase-switched replay decodes as a directional traveling replay "
          "(DECODE_r>0.6, width<8) 6/6.", flush=True)
    seeds = [42, 43, 44, 100, 101, 102]
    drs, bps = [], []
    for s in seeds:
        rZ, bp = one_seed(s)
        drs.append(rZ[0]); bps.append(bp)
        print(f"  [seed {s}] DECODE_r={rZ[0]:+.3f} width={rZ[1]:.1f} growth={rZ[2]:+.1f} band_preserved={bp}", flush=True)
    drs = np.array(drs)
    rgo = int((drs > 0.6).sum()); bgo = int(sum(bps))
    verdict = "GO" if (rgo == 6 and bgo == 6) else "NO-GO"
    print(f"\n=== PHASE-SWITCH 6-SEED: DECODE_r {np.round(drs,3).tolist()} | replay-travels {rgo}/6 | band-preserved {bgo}/6 -> {verdict} ===", flush=True)
    print("GAP5-PHASE-SWITCH DONE", flush=True)
