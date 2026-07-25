"""gap#5 MERGE de-risk — can the traveling replay reproduce on IZHIKEVICH at dt=1.0 (the conversational/nav substrate),
not the stiff ECKER-AdEx (dt=0.1)? verify-go proved the travel is driven by BAND + spike-reset REFRACTORINESS, NOT the
neg-a adaptation (inert). Izhikevich has refractoriness (the c-reset + recovery u), so the SAME mechanism should give a
traveling bump on Izhikevich at dt=1.0. If YES -> the merge onto the one-brain is TRIVIAL (same neuron model + same dt as
conversation -> no per-region-neuron-model wall, no dt-stiffness wall). Forward-biased Gaussian band + interior cue +
the committed decoder. GO iff a localized directional traveling replay decodes (DECODE_r>0.6, width small, no growth) AND
no-band collapses."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.bridge import SimulationBridge
from sim.regions import BrainRegion
from sim.enums import NeuronModel, NeuronType
from sim.backend import to_host, get_backend
from research.runners._gap5_ecker_recurrent_replay import decode_and_width

N_PC = 2000
SEED = 42


def build_izh(w_scale, sigma=25.0, back_frac=0.0, seed=SEED, dt=1.0, izh_type=None):
    cp, _ = get_backend()
    regions = [BrainRegion(name="pc", n_neurons=N_PC, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = float(dt)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name             # the conversational/nav substrate model
    if izh_type is not None:
        cfg.num_traits = 2                     # default_neuron_type_izh is only honored when num_traits>1 (CLAUDE.md)
        cfg.default_neuron_type_izh = izh_type
    else:
        cfg.num_traits = 1
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
    if pre:
        pre = np.concatenate(pre); post = np.concatenate(post); w = np.concatenate(w)
    else:
        pre = np.zeros(0, int); post = np.zeros(0, int); w = np.zeros(0, float)
    b.inject_explicit_wiring({"band": {"pre_indices": pre.astype(int).tolist(), "post_indices": post.astype(int).tolist(),
                                       "initial_weights": w.astype(float).tolist(), "plastic": False, "conn_type": "ff"}})
    return b, pc, cp


def run(tag, w_scale, dt=1.0, back_frac=0.0, cue_start=None, cue_pa=12000.0, cue_steps=40, T=250, seed=SEED, izh_type=None):
    t0 = time.time()
    b, pc, cp = build_izh(w_scale, back_frac=back_frac, seed=seed, dt=dt, izh_type=izh_type)
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
    dec_r, w, wg, drange = decode_and_width(F, tau_bin=int(round(3 / dt)) or 1)
    print(f"  [{tag}] dt={dt} w={w_scale} back_frac={back_frac} izh={izh_type}: F_active={F.mean():.4f} "
          f"DECODE_r={dec_r:+.3f} width={w:.1f} growth={wg:+.1f} dec_range={drange:.0f}/100 ({time.time()-t0:.0f}s)", flush=True)
    return dec_r, w, wg


print("gap#5 MERGE de-risk 2 — Izhikevich at dt=1.0 SPREADS (width~23 growing). Is the blocker the coarse dt or the Izh "
      "model? Sweep dt on Izhikevich (finer dt = more like AdEx's fine-timescale traveling wave). T scaled to hold real "
      "time (250 ms) constant.", flush=True)
for dt in (1.0, 0.5, 0.25, 0.1):
    T = int(round(250 / dt))
    for ws in (80.0, 160.0):
        run(f"IZH dt={dt} w={ws}", w_scale=ws, dt=dt, back_frac=0.0, T=T)
print("-- NO-BAND control at dt=0.1 (must collapse) --", flush=True)
run("IZH dt=0.1 NO-BAND", w_scale=0.0, dt=0.1, back_frac=0.0, T=2500)
print("  READ: if a FINER dt localizes Izhikevich (DECODE_r>0.6, width small, no growth) -> the merge needs only a "
      "finer-dt sleep-phase (SAME Izhikevich model as conversation -> no per-region-model wall). If Izhikevich SPREADS at "
      "every dt -> the AdEx's high-threshold sparse-single-fire dynamics are essential -> merge = AdEx replay slice in a "
      "sleep-phase (per-region model). Either way the merge is a TEMPORAL SWR/rest phase, not concurrent co-hosting.", flush=True)
print("GAP5-IZH-REPLAY DONE", flush=True)
