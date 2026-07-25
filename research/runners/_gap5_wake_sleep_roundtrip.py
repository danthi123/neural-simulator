"""gap#5 one-brain MERGE — the full WAKE->SLEEP->WAKE round-trip on a CO-RESIDENT bridge — 6-SEED GO (2026-07-25). ONE
bridge co-hosts a CONVERSATIONAL slice (Izhikevich, plastic recurrent -> learns weights during WAKE) + a REPLAY slice (the
CA3 place-field track). WAKE1: STDP on, drive a pattern into conv -> it learns. SLEEP: switch the whole bridge to AdEx/dt0.1
+ FREEZE STDP, run the CA3 replay on the track. WAKE2: switch BACK to Izhikevich/dt1.0 + thaw STDP. RESULT (6-seed): the
conversational memory (conv-slice weights) survives the full round-trip BYTE-IDENTICAL 6/6 AND the replay travels in sleep
DECODE_r=1.000 6/6. Reverse switch `switch_to_izhikevich_wake` (the cp_izh_* param arrays persist across the AdEx phase, so
it just restores v/u + model/dt + cached decays). TWO precisely-isolated integration requirements the diagnostic block
documents: (1) reset transient synaptic conductances/STP on each phase onset; (2) **FREEZE the wake STDP during the
sleep/replay phase** — with STDP left on, prior-wake conv FIRING (not idle) suppresses the sleep replay (the G-diagnostic:
freezing STDP restores DECODE_r 0.000->1.000; the band stays intact at 599, so it is not a weight collapse — the wake
plasticity rule must not run unmodified during replay). NO sim/ edit. GPU. Finding:
2026-07-25-gap5-wake-sleep-roundtrip-coresident-merge-6seed-GO.md."""
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
from research.runners._gap5_wake_sleep_phase_switch import switch_to_adex_sleep, _recompute_cached_decays

N_CONV = 300
N_PC = 2000
SEED = 42


def reset_transient_synaptic_state(b):
    """Sleep/wake ONSET reset — clear transient synaptic conductances + STP (the wake phase leaves these elevated; the
    phase-switch resets v/adex_w but not these, so residual conductance from the prior phase corrupts the next phase).
    Biologically, a state transition (wake<->sleep) clears the fast transient activity; the MEMORY (weights) persists."""
    cp, _ = get_backend()
    for arr in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
                "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise"):
        a = getattr(b, arr, None)
        if a is not None:
            a[:] = 0.0
    if getattr(b, "cp_stp_x", None) is not None:
        b.cp_stp_x[:] = 1.0
    if getattr(b, "cp_stp_u", None) is not None:
        b.cp_stp_u[:] = float(b.core_config.stp_U)
    if getattr(b, "cp_external_input_current", None) is not None:
        b.cp_external_input_current[:] = 0.0


def switch_to_izhikevich_wake(b, dt=1.0):
    """SLEEP(AdEx)->WAKE(Izhikevich) reverse switch. cp_izh_* param arrays persist from the original init; restore v/u to
    Izhikevich rest + model/dt + cached decays. cp_connections (all learned memory) is never touched."""
    cp, _ = get_backend()
    cfg = b.core_config
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.dt_ms = float(dt)
    b.cp_membrane_potential_v = b.cp_izh_vr.copy()      # reset membranes to Izhikevich rest
    b.cp_recovery_variable_u = cp.zeros(int(b.cp_membrane_potential_v.shape[0]), dtype=cp.float32)
    _recompute_cached_decays(b)
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)


def build(seed=SEED):
    cp, _ = get_backend()
    regions = [
        BrainRegion(name="conv", n_neurons=N_CONV, exc_fraction=1.0, internal_density=0.05, exc_weight_mean=1.0,
                    inh_weight_mean=0.0, weight_jitter=0.2, plastic_internal=True),   # plastic -> learns during WAKE
        BrainRegion(name="pc", n_neurons=N_PC, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    cfg.enable_stdp = True; cfg.stdp_w_max = 8.0                    # conv slice learns via STDP during WAKE
    for f in ("enable_homeostasis", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity"):
        setattr(cfg, f, False)
    cfg.enable_ou_process = True; cfg.ou_noise_sigma_pa = 40.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    conv = np.asarray(b.region_manager.indices("conv"), int)
    pc = np.asarray(b.region_manager.indices("pc"), int)
    # wire the replay track (forward-biased Gaussian band) on the pc slice, fixed
    sigma = 25.0; cutoff = int(3 * sigma); offs = np.arange(-cutoff, cutoff + 1); offs = offs[offs != 0]
    wof = 600.0 * np.exp(-(offs / sigma) ** 2); keep = wof > 0.02; offs, wof = offs[keep], wof[keep]
    pre, post, w = [], [], []; ii = np.arange(N_PC)
    for o, wv in zip(offs, wof):
        if o <= 0:
            continue                                   # forward-only (directional)
        j = ii + o; m = (j >= 0) & (j < N_PC)
        pre.append(pc[ii[m]]); post.append(pc[j[m]]); w.append(np.full(m.sum(), wv, np.float64))
    pre = np.concatenate(pre); post = np.concatenate(post); w = np.concatenate(w)
    b.inject_explicit_wiring({"band": {"pre_indices": pre.astype(int).tolist(), "post_indices": post.astype(int).tolist(),
                                       "initial_weights": w.astype(float).tolist(), "plastic": False, "conn_type": "ff"}})
    return b, conv, pc, cp


def conv_weights(b, conv):
    # extract the conv-slice recurrent weights (the conversational MEMORY) -> a hashable snapshot
    coo = b.cp_connections.tocoo()
    rows = np.asarray(to_host(coo.row)); cols = np.asarray(to_host(coo.col)); vals = np.asarray(to_host(coo.data))
    cset = set(conv.tolist())
    mask = np.array([r in cset and c in cset for r, c in zip(rows, cols)])
    return vals[mask].copy()


def drive_conv(b, conv, cp, steps, drive=6000.0):
    # WAKE activity: drive a fixed sub-pattern of conv so its recurrent STDP learns a memory
    patt = cp.asarray(conv[:120], dtype=cp.int64); spk = 0
    for t in range(steps):
        b.runtime_state.current_time_ms += b.core_config.dt_ms
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[patt] += float(drive)
        b._run_one_simulation_step()
        spk += int(to_host(b.cp_firing_states)[conv].sum())
    return spk


def replay_sleep(b, pc, cp, T=2500):
    cue = cp.asarray(pc[N_PC // 2 - 50:N_PC // 2 + 50], dtype=cp.int64)
    F = np.zeros((T, N_PC), dtype=bool)
    for t in range(T):
        b.runtime_state.current_time_ms += b.core_config.dt_ms
        b.cp_external_input_current[:] = 0.0
        if t < 40:
            b.cp_external_input_current[cue] += 10000.0
        b._run_one_simulation_step()
        F[t] = np.asarray(to_host(b.cp_firing_states))[pc].astype(bool)
    dec = decode_and_width(F)
    return dec[0], dec[1], float(F.mean())


def one_seed(seed):
    b, conv, pc, cp = build(seed)
    drive_conv(b, conv, cp, steps=400)          # WAKE 1: learn a conversational memory
    W1 = conv_weights(b, conv)
    switch_to_adex_sleep(b, dt=0.1); reset_transient_synaptic_state(b)   # -> SLEEP (AdEx/dt0.1) + sleep-onset transient reset
    b.core_config.enable_stdp = False           # freeze WAKE plasticity for the sleep/replay phase (the fix — G-diagnostic)
    dr, wdt, fa = replay_sleep(b, pc, cp)       # run the CA3 replay during sleep
    switch_to_izhikevich_wake(b, dt=1.0); reset_transient_synaptic_state(b)   # -> WAKE 2 (Izhikevich/dt1.0) + reset
    b.core_config.enable_stdp = True            # thaw for wake
    W2 = conv_weights(b, conv)
    b.runtime_state.current_time_ms += b.core_config.dt_ms   # a wake step to confirm Izhikevich dynamics resume
    b.cp_external_input_current[:] = 0.0; b._run_one_simulation_step()
    preserved = bool(W1.shape == W2.shape and np.array_equal(W1, W2))
    return dr, wdt, preserved


# DIAGNOSTIC (seed 42): (A) works, full round-trip fails -> the wake-drive breaks it. Track the pc BAND max weight across
# the wake phase: if it collapses (600 -> stdp_w_max=8) the STDP clip is hitting the fixed band (the soft-bound gotcha).
def band_max(b, pc):
    coo = b.cp_connections.tocoo()
    rows = np.asarray(to_host(coo.row)); cols = np.asarray(to_host(coo.col)); vals = np.asarray(to_host(coo.data))
    pcset = set(pc.tolist()); m = np.array([r in pcset and c in pcset for r, c in zip(rows, cols)])
    return float(vals[m].max()) if m.any() else 0.0

print("-- DIAGNOSTIC seed 42 --", flush=True)
bd, convd, pcd, cpd = build(42)
print(f"  band_max after build={band_max(bd, pcd):.1f}", flush=True)
drive_conv(bd, convd, cpd, steps=400)
print(f"  band_max after WAKE-drive (STDP on)={band_max(bd, pcd):.1f}  <- collapsed to ~stdp_w_max(8)? = the clip gotcha", flush=True)
switch_to_adex_sleep(bd, dt=0.1)
_r, _w, _fa = replay_sleep(bd, pcd, cpd)
print(f"  (B) full: after wake-drive, sleep replay: DECODE_r={_r:+.3f} F_active={_fa:.4f} width={_w:.1f} | "
      f"band_max AFTER sleep replay={band_max(bd, pcd):.1f}  <- collapsed? = STDP hit the fixed band during the replay", flush=True)
# (A) clean control
bA, cA, pA, cpA = build(42)
switch_to_adex_sleep(bA, dt=0.1); reset_transient_synaptic_state(bA)
_rA, _wA, _faA = replay_sleep(bA, pA, cpA)
print(f"  (A) NO wake-drive control: DECODE_r={_rA:+.3f} F_active={_faA:.4f}", flush=True)
# (C) IDLE-wake (400 steps, NO conv drive) — isolates time/RNG/step carryover vs conv firing activity
bC, cC, pC, cpC = build(42)
for t in range(400):
    bC.runtime_state.current_time_ms += bC.core_config.dt_ms
    bC.cp_external_input_current[:] = 0.0; bC._run_one_simulation_step()
switch_to_adex_sleep(bC, dt=0.1); reset_transient_synaptic_state(bC)
_rC, _wC, _faC = replay_sleep(bC, pC, cpC)
print(f"  (C) IDLE-wake 400 steps (no drive) then sleep: DECODE_r={_rC:+.3f} F_active={_faC:.4f} "
      f"-> {'time/step carryover' if _rC < 0.6 else 'conv-firing-specific (idle is fine)'}", flush=True)
# (G) wake-DRIVE conv, then FREEZE STDP before sleep (STDP is the only wake-history-carrying process active in sleep)
bG, cG, pG, cpG = build(42)
drive_conv(bG, cG, cpG, steps=400)
bG.core_config.enable_stdp = False            # freeze plasticity for the sleep/replay phase
switch_to_adex_sleep(bG, dt=0.1); reset_transient_synaptic_state(bG)
_rG, _wG, _faG = replay_sleep(bG, pG, cpG)
print(f"  (G) wake-drive + FREEZE STDP before sleep: DECODE_r={_rG:+.3f} F_active={_faG:.4f} "
      f"-> {'STDP-in-sleep was the culprit!' if _rG > 0.6 else 'not STDP'}", flush=True)
# (diagnostic done; run the full 6-seed below)

print("gap#5 ROUND-TRIP MERGE — WAKE(Izh)->SLEEP(AdEx replay)->WAKE(Izh) on a CO-RESIDENT conv+replay bridge, 6-seed. GO "
      "iff the conversational-slice memory survives the full round-trip BYTE-IDENTICAL 6/6 AND the replay travels in sleep "
      "(DECODE_r>0.6) 6/6.", flush=True)
seeds = [42, 43, 44, 100, 101, 102]
drs, pres = [], []
for s in seeds:
    dr, wdt, pr = one_seed(s)
    drs.append(dr); pres.append(pr)
    print(f"  [seed {s}] sleep-replay DECODE_r={dr:+.3f} width={wdt:.1f} | conv-memory-preserved={pr}", flush=True)
drs = np.array(drs)
rgo = int((drs > 0.6).sum()); pgo = int(sum(pres))
verdict = "GO" if (rgo == 6 and pgo == 6) else "NO-GO"
print(f"\n=== ROUND-TRIP 6-SEED: replay {np.round(drs,3).tolist()} travels {rgo}/6 | conv-memory-preserved {pgo}/6 -> {verdict} ===", flush=True)
print("GAP5-ROUNDTRIP DONE", flush=True)
