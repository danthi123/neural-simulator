"""gap#5 one-brain MERGE — the END-TO-END CAPSTONE. The conversing brain SLEEPS, RUNS A REAL DECODABLE CA3 REPLAY, WAKES,
and STILL CONVERSES — all on ONE bridge. Assembles the two proven halves: the co-resident replay round-trip (1bdcc5a4)
+ the production composer surviving the switch (e2b86dce). Monkeypatch build_coresident_bridge to append a CA3 place-field
track slice (extra neurons + the forward-biased band) BEYOND the composer's n_total layout (invisible to the composer's
[0:n_total] parser/RF). Then: store facts + query (WAKE); switch to AdEx/dt0.1 + freeze plasticity, run the REAL CA3
replay on the track slice + Bayesian-decode it (SLEEP); switch back to Izhikevich/dt1.0 (WAKE); re-query. GO iff the
conversational recall + no-confab moat are preserved AND the sleep replay travelled + decoded (DECODE_r>0.6)."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.bridge import SimulationBridge
from sim.enums import NeuronModel
from sim.backend import to_host, get_backend, is_gpu_backend
from research.runners._gap5_ecker_recurrent_replay import decode_and_width
from research.runners._gap5_wake_sleep_phase_switch import switch_to_adex_sleep
from research.runners._gap5_wake_sleep_roundtrip import switch_to_izhikevich_wake, reset_transient_synaptic_state
import research.runners.one_brain_composer as obc

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]
FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
QUERIES = [("dog", "go"), ("cat", "come"), ("bird", "look")]
MOAT = [("apple", "swim"), ("river", "stop")]
N_CA3 = 2000

_orig_build = obc.build_coresident_bridge


def _patched_build(seed, n_total, **kw):
    """Enlarge the composer's bridge by N_CA3 neurons (the CA3 track slice [n_total : n_total+N_CA3]) AND enable
    short-term plasticity (STP) -- the CA3-replay sharpener that `build_coresident_bridge` disables. Replicates
    build_coresident_bridge's config verbatim EXCEPT enable_short_term_plasticity=True. The composer's own layout
    (parser + RF, indices < n_total) is unchanged; the band is injected AFTER the composer wires its parser/RF (in run())."""
    N = n_total + N_CA3
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_max_weight = 400.0; cfg.hebbian_learning_rate = 0.005
    for f in ("enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.enable_short_term_plasticity = True   # <-- THE FIX (build_coresident_bridge sets this False; STP sharpens the replay)
    cfg.enable_rf_cudagraph = bool(kw.get("enable_rf_cudagraph", False))
    cfg.ou_std_current_pA = 20.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    b._ca3_pc = np.arange(n_total, n_total + N_CA3)
    return b


def inject_ca3_band(b):
    pc = b._ca3_pc
    sigma = 25.0; cutoff = int(3 * sigma); offs = np.arange(-cutoff, cutoff + 1); offs = offs[offs != 0]
    wof = 600.0 * np.exp(-(offs / sigma) ** 2); keep = wof > 0.02; offs, wof = offs[keep], wof[keep]
    pre, post, w = [], [], []; ii = np.arange(N_CA3)
    for o, wv in zip(offs, wof):
        if o <= 0:
            continue
        j = ii + o; m = (j >= 0) & (j < N_CA3)
        pre.append(pc[ii[m]]); post.append(pc[j[m]]); w.append(np.full(m.sum(), wv, np.float64))
    pre = np.concatenate(pre); post = np.concatenate(post); w = np.concatenate(w)
    b.inject_explicit_wiring({"ca3band": {"pre_indices": pre.astype(int).tolist(), "post_indices": post.astype(int).tolist(),
                                          "initial_weights": w.astype(float).tolist(), "plastic": False, "conn_type": "ff"}})


def replay_on_track(b, pc, T=2500, silence_below=True):
    cp, _ = get_backend()
    cue = cp.asarray(pc[N_CA3 // 2 - 50:N_CA3 // 2 + 50], dtype=cp.int64)
    off = int(pc[0])                                  # the CA3 slice starts at n_total; [0:off] = parser/RF
    F = np.zeros((T, N_CA3), dtype=bool)
    for t in range(T):
        b.runtime_state.current_time_ms += b.core_config.dt_ms
        b.cp_external_input_current[:] = 0.0
        if silence_below and off > 0:
            b.cp_external_input_current[:off] = -1e5   # hold the parser/RF slice silent during the SWR/sleep replay
        if t < 40:
            b.cp_external_input_current[cue] += 10000.0
        b._run_one_simulation_step()
        F[t] = np.asarray(to_host(b.cp_firing_states))[pc].astype(bool)
    return decode_and_width(F)


def run(seed):
    from research.runners.one_brain_composer import OneBrainComposer
    c = OneBrainComposer(seed=seed, D=64, vocab=VOCAB)
    # the composer's layout indices are all < original n_total (computed in __init__); widen n_total to the full bridge
    # so RF kick vectors are sized to num_neurons, and pad rf_mask (the only n_total-sized mask) with False over the CA3
    # slice -> the composer's RF ops never touch the CA3 track (its memory + ops stay disjoint)
    num = int(c.b.core_config.num_neurons)
    c.rf_mask = np.concatenate([c.rf_mask, np.zeros(num - c.n_total, dtype=bool)])
    c.n_total = num
    inject_ca3_band(c.b)   # inject the CA3 band NOW (after parser/RF wiring, so it isn't wiped)
    cp, _ = get_backend()
    if getattr(c.b, "cp_traits", None) is not None:   # force the CA3 slice ALL-EXCITATORY (a flat bridge may assign some inhibitory; the band is excitatory-designed)
        ca3 = c.b._ca3_pc
        n_inh = int((np.asarray(to_host(c.b.cp_traits))[ca3] == c.b.core_config.inhibitory_trait_index).sum())
        if seed == 42:
            print(f"  [diag] CA3 slice inhibitory neurons={n_inh}/{len(ca3)}", flush=True)
        c.b.cp_traits[cp.asarray(ca3)] = 0
        c.b._cached_inhibitory_mask = None
    for a, v, p in FACTS:
        c.store(a, v, p)
    ans1 = [c.query_patient(a, v) for a, v in QUERIES]; moat1 = [c.query_patient(a, v) for a, v in MOAT]
    # --- SLEEP: switch the composer's OWN bridge to AdEx + run the REAL CA3 replay on the track slice ---
    b = c.b
    heb0 = b.core_config.enable_hebbian_learning
    b.core_config.enable_stdp = False; b.core_config.enable_hebbian_learning = False   # freeze plasticity for sleep
    for _f, _v in (("ou_std_current_pA", 0.0), ("ou_noise_sigma_pa", 40.0), ("enable_ou_process", True)):
        if hasattr(b.core_config, _f):
            setattr(b.core_config, _f, _v)             # kill the composer's own OU noise; use the standalone replay's
    switch_to_adex_sleep(b, dt=0.1); reset_transient_synaptic_state(b)
    if seed == 42:   # diagnostic: is the CA3 band intact after the composer's wake (Hebbian) ops?
        coo = b.cp_connections.tocoo(); rr = np.asarray(to_host(coo.row)); cc = np.asarray(to_host(coo.col)); vv = np.asarray(to_host(coo.data))
        ca3set = set(b._ca3_pc.tolist()); mm = np.array([r in ca3set and c in ca3set for r, c in zip(rr, cc)])
        print(f"  [diag] CA3 band synapses={int(mm.sum())} max_w={float(vv[mm].max()) if mm.any() else 0:.1f} (want ~600; hebbian_max=400)", flush=True)
    dec_r, dec_w, dec_g, dec_range = replay_on_track(b, b._ca3_pc)
    switch_to_izhikevich_wake(b, dt=1.0); reset_transient_synaptic_state(b)
    b.core_config.enable_hebbian_learning = heb0                                       # thaw for wake
    # --- WAKE again: re-query ---
    ans2 = [c.query_patient(a, v) for a, v in QUERIES]; moat2 = [c.query_patient(a, v) for a, v in MOAT]
    recall_ok = (ans1 == ans2) and all(a == p for (a, (_, _, p)) in zip(ans2, FACTS))
    moat_ok = all(m is None for m in moat1) and all(m is None for m in moat2)
    replay_ok = dec_r > 0.6 and dec_w < 8
    print(f"  [seed {seed}] SLEEP replay DECODE_r={dec_r:+.3f} width={dec_w:.1f} range={dec_range:.0f}/100 | "
          f"WAKE recall {ans1}=={ans2}:{recall_ok} moat:{moat_ok} -> {'GO' if (recall_ok and moat_ok and replay_ok) else 'NO-GO'}", flush=True)
    return recall_ok and moat_ok and replay_ok


if not is_gpu_backend():
    print("SKIP: needs GPU", flush=True); sys.exit(0)
obc.build_coresident_bridge = _patched_build      # install the CA3-track-appending bridge builder
print("gap#5 END-TO-END CAPSTONE — the conversing brain SLEEPS, runs a REAL decodable CA3 replay on its own bridge, WAKES, "
      "and STILL CONVERSES. GO iff recall+moat preserved AND the sleep replay travels (DECODE_r>0.6), all seeds.", flush=True)
seeds = [42, 43, 44, 100, 101, 102]
oks = []
for s in seeds:
    try:
        oks.append(run(s))
    except Exception as e:
        import traceback; print(f"  [seed {s}] ERROR: {type(e).__name__}: {e}", flush=True); traceback.print_exc(); oks.append(False)
print(f"\n=== CAPSTONE: {sum(oks)}/{len(seeds)} -> {'GO' if all(oks) and len(oks)==len(seeds) else 'NO-GO'} "
      f"(converse -> sleep+replay -> converse, one brain) ===", flush=True)
print("GAP5-CAPSTONE DONE", flush=True)
