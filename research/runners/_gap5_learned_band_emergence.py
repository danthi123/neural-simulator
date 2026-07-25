"""gap#5 LEARNED BAND (emergence-bar version of the traveling-replay band) — 6-SEED GO, structural + functional
(2026-07-25). The gap#5 replay model (`_gap5_ecker_recurrent_replay.py`) uses a HAND-WIRED forward-biased near-diagonal
band. This shows that band EMERGES from experience via STDP instead of being designed. Biology: Mehta-Blum-Abbott
experience-dependent asymmetric place-field expansion — a rat running a track one way fires place cells in sequence; the
causal STDP window potentiates i->i+1 (pre-before-post) and depresses i+1->i, so the recurrent connectivity becomes
forward-biased. Method: start a WEAK SYMMETRIC plastic near-diagonal band + STDP-on, sweep a drive bump along the track
N laps, measure the developed forward/backward weight ratio (STRUCTURAL), then freeze + scale to operating strength +
cue the interior + Bayesian-decode (FUNCTIONAL). Modes: default = 3-arm structural; `sixseed` = 6-seed STRUCTURAL
(FWD-traversal grows fwd-bias 6/6 ratio~1.66; REVERSE grows bwd-bias 6/6 ratio~0.60; NO-STDP stays symmetric 6/6); `func6`
= 6-seed FUNCTIONAL (FWD-learned band replays FORWARD DECODE_r>0.6, REV-learned replays REVERSE DECODE_r<-0.6 — direction
follows training). GOTCHA (reusable): a raw `_run_one_simulation_step()` loop does NOT advance `runtime_state.current_time_ms`
(only `step_simulation()` does), so STDP timestamps every spike identically -> delta_t=0 -> STDP silently no-ops; the
traversal loop advances time manually. dt=0.1. NO sim/ edit. GPU (SIM_BACKEND=cupy). Finding:
2026-07-25-gap5-learned-band-emergence-STDP-directed-traversal-6seed-GO.md."""
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

N_PC = 400
SEED = 42


def build_plastic(w0=1.0, sigma=8.0, seed=SEED, stdp_wmax=60.0, stdp=True):
    cp, _ = get_backend()
    regions = [BrainRegion(name="pc", n_neurons=N_PC, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 0.1; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.default_neuron_type_adex = NeuronType.ADEX_ECKER_CA3_PC.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    cfg.enable_stdp = bool(stdp)                 # STDP ON = the learning rule (default asymmetric Hebbian window)
    cfg.stdp_w_max = float(stdp_wmax); cfg.stdp_w_min = 0.0
    for f in ("enable_homeostasis", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity"):
        setattr(cfg, f, False)
    cfg.enable_ou_process = True; cfg.ou_noise_sigma_pa = 40.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    pc = np.asarray(b.region_manager.indices("pc"), int)
    # WEAK SYMMETRIC plastic near-diagonal band (local connectivity exists; NO directional bias yet)
    cutoff = int(3 * sigma); offs = np.arange(-cutoff, cutoff + 1); offs = offs[offs != 0]
    wof = w0 * np.exp(-(offs / sigma) ** 2); keep = wof > 0.02; offs, wof = offs[keep], wof[keep]
    pre, post, w = [], [], []; ii = np.arange(N_PC)
    for o, wv in zip(offs, wof):
        j = ii + o; m = (j >= 0) & (j < N_PC)
        pre.append(pc[ii[m]]); post.append(pc[j[m]]); w.append(np.full(m.sum(), wv, np.float64))
    pre = np.concatenate(pre); post = np.concatenate(post); w = np.concatenate(w)
    b.inject_explicit_wiring({"band": {"pre_indices": pre.astype(int).tolist(), "post_indices": post.astype(int).tolist(),
                                       "initial_weights": w.astype(float).tolist(), "plastic": True, "conn_type": "ff"}})
    return b, pc, cp, pre, post


def traverse(b, pc, cp, n_laps=15, dwell=8, drive=9000.0, bump_w=25, direction=+1):
    # sweep a Gaussian drive bump along the track: place cells fire IN SEQUENCE -> STDP develops the asymmetry
    order = np.arange(0, N_PC, 3) if direction > 0 else np.arange(N_PC - 1, -1, -3)
    spk = 0
    for lap in range(n_laps):
        for c in order:
            lo, hi = max(0, c - bump_w), min(N_PC, c + bump_w)
            cue = cp.asarray(pc[lo:hi], dtype=cp.int64)
            for _ in range(dwell):
                b.runtime_state.current_time_ms += b.core_config.dt_ms   # STDP timestamps spikes from this — MUST advance (step_simulation does; a raw _run_one_simulation_step does NOT)
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[cue] += float(drive)
                b._run_one_simulation_step()
                spk += int(to_host(b.cp_firing_states)[pc].sum())
    return spk


def measure(b, pre, post):
    # read learned band weights -> forward (post ahead of pre) vs backward mean weight
    coo = b.cp_connections.tocoo()
    rows = np.asarray(to_host(coo.row)); cols = np.asarray(to_host(coo.col)); vals = np.asarray(to_host(coo.data))
    off = cols - rows   # orientation ambiguous (row=pre or post) — report both splits; traversal direction disambiguates
    fwd = vals[off > 0]; bwd = vals[off < 0]
    return float(fwd.mean()) if len(fwd) else 0.0, float(bwd.mean()) if len(bwd) else 0.0, len(vals)


def run(tag, stdp=True, direction=+1, n_laps=15, seed=SEED):
    t0 = time.time()
    b, pc, cp, pre, post = build_plastic(seed=seed, stdp=stdp)
    f0, b0, n = measure(b, pre, post)
    spk = traverse(b, pc, cp, n_laps=n_laps, direction=direction)
    f1, b1, _ = measure(b, pre, post)
    ratio0 = f0 / b0 if b0 > 0 else 0.0; ratio1 = f1 / b1 if b1 > 0 else 0.0
    print(f"  [{tag}] spikes={spk} pre: fwd={f0:.3f} bwd={b0:.3f} ratio={ratio0:.3f} (n={n}) -> post: fwd={f1:.3f} bwd={b1:.3f} "
          f"ratio={ratio1:.3f} | Δratio={ratio1-ratio0:+.3f} ({time.time()-t0:.0f}s)", flush=True)
    return ratio1


def functional(direction=+1, seed=SEED, gain=18.0, n_laps=15, stdp_wmax=60.0, sigma=25.0, tag=""):
    # Does the LEARNED band produce DIRECTIONAL replay whose direction follows the training? Learn -> freeze STDP ->
    # scale to self-sustaining (uniform gain preserves the learned asymmetry ratio) -> cue interior -> decode.
    # sigma=25 MATCHES the replay regime (the earlier narrow sigma=8 learned but couldn't self-sustain replay).
    from research.runners._gap5_ecker_recurrent_replay import decode_and_width
    t0 = time.time()
    b, pc, cp, pre, post = build_plastic(seed=seed, stdp=True, stdp_wmax=stdp_wmax, sigma=sigma)
    traverse(b, pc, cp, n_laps=n_laps, direction=direction)
    f1, b1, _ = measure(b, pre, post); ratio = f1 / b1 if b1 > 0 else 0.0
    b.core_config.enable_stdp = False                       # freeze the learned band
    b.core_config.adex_b = 120.0                            # soften adaptation for the replay phase (as the replay model)
    b.cp_membrane_potential_v[:] = b.core_config.adex_E_L   # reset neuron state (learning left adaptation huge -> hyperpolarized)
    b.cp_adex_w[:] = 0.0                                    # AdEx adaptation var (NOT cp_recovery_variable_u, which is None for AdEx)
    b.cp_connections.data[:] = b.cp_connections.data * float(gain)   # maturation gain to operating strength (uniform -> ratio preserved)
    mid = N_PC // 2 - 25
    cue = cp.asarray(pc[mid:mid + 50], dtype=cp.int64)
    T = 2500; F = np.zeros((T, N_PC), dtype=bool)
    for t in range(T):
        b.runtime_state.current_time_ms += b.core_config.dt_ms
        b.cp_external_input_current[:] = 0.0
        if t < 40:
            b.cp_external_input_current[cue] += 10000.0
        b._run_one_simulation_step()
        F[t] = np.asarray(to_host(b.cp_firing_states))[pc].astype(bool)
    dec_r, w, wg, drange = decode_and_width(F)
    print(f"  [{tag}] learned_ratio={ratio:.3f} gain={gain} F_active={F.mean():.4f}: DECODE_r={dec_r:+.3f} "
          f"width={w:.1f} growth={wg:+.1f} dec_range={drange:.0f}/100 ({time.time()-t0:.0f}s)", flush=True)
    return dec_r, ratio


def functional_6seed():
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"gap#5 LEARNED-BAND FUNCTIONAL 6-SEED (dt=0.1), seeds {seeds}, sigma=25 gain=25. GO iff FWD-learned replays "
          f"FORWARD (DECODE_r>0.6) 6/6 AND REV-learned replays REVERSE (DECODE_r<-0.6) 6/6 — same protocol, opposite "
          f"training -> opposite replay direction PROVES the learned asymmetry (not the band per se) sets the direction.", flush=True)
    fwd, rev = [], []
    for s in seeds:
        fwd.append(functional(direction=+1, gain=25.0, sigma=25.0, seed=s, tag=f"FWD s{s}")[0])
        rev.append(functional(direction=-1, gain=25.0, sigma=25.0, seed=s, tag=f"REV s{s}")[0])
    fwd = np.array(fwd); rev = np.array(rev)
    fgo = int((fwd > 0.6).sum()); rgo = int((rev < -0.6).sum())
    print(f"\n=== FUNCTIONAL 6-SEED SUMMARY ===", flush=True)
    print(f"FWD-learned DECODE_r: {np.round(fwd,3).tolist()} (forward replay {fgo}/6)", flush=True)
    print(f"REV-learned DECODE_r: {np.round(rev,3).tolist()} (reverse replay {rgo}/6)", flush=True)
    verdict = "GO" if (fgo == 6 and rgo == 6) else "NO-GO"
    print(f"-> {verdict} (LEARNED band replays in the TRAINED direction; fwd/rev contrast isolates the asymmetry)", flush=True)
    print("GAP5-LEARNED-BAND-FUNC6 DONE", flush=True)


def functional_test():
    print("gap#5 LEARNED-BAND FUNCTIONAL — does the learned band REPLAY directionally? (dt=0.1). GO iff FWD-learned band "
          "-> forward replay (DECODE_r>0.6), REV-learned -> reverse replay (DECODE_r<-0.6 OR opposite), NO-STDP symmetric "
          "-> spreads (|DECODE_r|<0.5). Direction of replay follows the training direction = emergence closed.", flush=True)
    for g in (12.0, 18.0, 25.0):
        functional(direction=+1, gain=g, sigma=25.0, tag=f"FWD-learned sig25 gain={g}")
    functional(direction=-1, gain=18.0, sigma=25.0, tag="REV-learned sig25 gain=18 (expect reverse DECODE_r<0)")
    print("GAP5-LEARNED-BAND-FUNCTIONAL DONE", flush=True)


def sixseed():
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"gap#5 LEARNED-BAND 6-SEED structural emergence (dt=0.1), seeds {seeds}. GO iff FORWARD+STDP develops "
          f"fwd-bias (ratio>1.2) 6/6 AND REVERSE+STDP develops bwd-bias (ratio<0.83) 6/6 AND NO-STDP stays symmetric "
          f"(|ratio-1|<0.1) 6/6 — the asymmetry EMERGES from experience + tracks the traversal direction.", flush=True)
    fwd, rev, ctl = [], [], []
    for s in seeds:
        fwd.append(run(f"FWD+STDP s{s}", stdp=True, direction=+1, seed=s))
        rev.append(run(f"REV+STDP s{s}", stdp=True, direction=-1, seed=s))
        ctl.append(run(f"NO-STDP s{s}", stdp=False, direction=+1, seed=s))
    fwd = np.array(fwd); rev = np.array(rev); ctl = np.array(ctl)
    fgo = int((fwd > 1.2).sum()); rgo = int((rev < 0.83).sum()); cgo = int((np.abs(ctl - 1) < 0.1).sum())
    print(f"\n=== 6-SEED SUMMARY ===", flush=True)
    print(f"FWD+STDP ratio: {np.round(fwd,3).tolist()} (fwd-bias {fgo}/6)", flush=True)
    print(f"REV+STDP ratio: {np.round(rev,3).tolist()} (bwd-bias {rgo}/6)", flush=True)
    print(f"NO-STDP  ratio: {np.round(ctl,3).tolist()} (symmetric {cgo}/6)", flush=True)
    verdict = "GO" if (fgo == 6 and rgo == 6 and cgo == 6) else "NO-GO"
    print(f"-> {verdict} (asymmetric band EMERGES from directed traversal via STDP)", flush=True)
    print("GAP5-LEARNED-BAND-6SEED DONE", flush=True)


if __name__ == "__main__":
    _m = sys.argv[1] if len(sys.argv) > 1 else ""
    if _m == "sixseed":
        sixseed()
    elif _m == "functional":
        functional_test()
    elif _m == "func6":
        functional_6seed()
    else:
        print("gap#5 LEARNED-BAND emergence (dt=0.1) — does STDP + directed traversal grow a FORWARD-asymmetric band?", flush=True)
        run("FORWARD traversal + STDP", stdp=True, direction=+1)
        run("REVERSE traversal + STDP", stdp=True, direction=-1)
        run("FORWARD traversal, NO-STDP (control)", stdp=False, direction=+1)
        print("GAP5-LEARNED-BAND DONE", flush=True)
