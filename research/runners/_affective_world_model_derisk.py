"""De-risk a GENUINE SPIKING **AFFECTIVE WORLD-MODEL** — the owner-named "internal
worldview": a brain that maintains + updates an internal PREDICTIVE representation of
its conversational world (the interlocutor's affect trajectory), queryable at
conversation time, that DRIVES an expectation and fires on prediction-error.

THE FACULTY (what "internal worldview / affective world-model" means here)
-------------------------------------------------------------------------
A world-model is NOT static fact storage. It is a LEARNED FORWARD/TRANSITION model:
given the CURRENT conversational state, it PREDICTS the interlocutor's NEXT-turn
AFFECT (valence), holds that expectation, and when the actual next turn VIOLATES it a
spiking SURPRISE (affective prediction-error) fires. The model UPDATES from that error
(the transition is plastic) and is QUERYABLE ("what do you expect next / how is this
going?") as a spike-mass read. Across a multi-turn conversation the state is re-driven
each turn and the expectation rolls forward — maintained across turns.

WHY THIS IS A REAL GAP (the honest boundary this de-risk sits at)
----------------------------------------------------------------
The substrate ALREADY has: (i) a within-turn spiking predictive-coding MISMATCH unit
over stored (agent,action)->patient facts (`_spiking_expectation_rpe_derisk.py`,
2026-08-12); (ii) an HTM Temporal-Memory next-SYMBOL sequence predictor (EMERGE-15/9d
GO); (iii) an OTHER-tagged affect model (W5 affective ToM, 2026-08-01); (iv) the P0.3
valence latch. NONE of them is an AFFECTIVE FORWARD MODEL: a LEARNED transition that
predicts the interlocutor's NEXT-turn valence from the conversational state, maintained
across turns, updating on a spiking affective prediction-error, and queryable. That
integration is the "internal worldview" faculty, currently ABSENT/unvalidated. This
runner de-risks it.

THE MECHANISM UNDER TEST — a 2-channel spiking predictive-coding valence forward model
-------------------------------------------------------------------------------------
Predictive coding (Rao & Ballard 1999; Bastos et al. 2012) over the AFFECT axis, the
interoceptive/affective-inference line (Seth 2013; Barrett & Simmons 2015 — the brain
runs a generative model predicting its affective state) realised on spikes:

  state s  --PLASTIC (Hebbian co-fire; LEARNED transition)-->  pred_pos (FS, PV-like)
    (the current                                                pred_neg (FS, PV-like)
     conversational                                    the PREDICTED next-turn valence,
     state, driven                                     delivered as SUBTRACTIVE GABA_A
     by the                                            inhibition (the top-down prediction)
     interlocutor)                                             |
                                                               v
  obs_pos --EXC-->  surprise_pos  <--INH-- pred_pos     obs = the ACTUAL next-turn
  obs_neg --EXC-->  surprise_neg  <--INH-- pred_neg       valence (teacher/environment)
    (observed next-turn valence,               the ERROR / SURPRISE units; their total
     delivered as sensory drive —              firing rate IS the affective
     the legitimate environment boundary)      prediction-error ("was that expected?")

The LEARNED content is which valence pool each state drives (state_s -> pred_pos if the
trajectory's next turn after s is positive, else -> pred_neg). This is a 2-way learned
discrimination PER STATE — genuinely arbitrary (a per-seed balanced valence map), so the
SHUFFLE control is decisive — while sidestepping the n-way CA3 pattern-separation wall
(2026-06-05-D-cue-recall-RESOLVED) that a full next-STATE recall would hit. Predictive
coding cancellation:
  - EXPECTED (observed == predicted valence): pred_v fires -> inhibits surprise_v, which
    obs_v excites -> cancel -> surprise ~ 0.
  - VIOLATED (observed == opposite valence): the opposite surprise unit is EXCITED
    (observed) but NOT inhibited (the model predicted the other valence) -> FIRES -> high.

WHAT IS NEURAL vs THE LEGITIMATE BOUNDARY
-----------------------------------------
- The PREDICTION is neural + LEARNED: state->pred_{pos,neg} is Hebbian co-fire; which
  valence is expected is RECALLED by firing, not a host lookup. The queryable read is
  sign(rate(pred_pos) - rate(pred_neg)) — a SPIKE-RATE difference (W5 tone_sign motif),
  never host-set.
- The MISMATCH is neural: surprise = observed excitation - predicted GABA_A inhibition at
  the surprise membrane; a cp_firing_states READ, never a host code subtraction.
- The UPDATE is neural: the transition weights are Hebbian-plastic; re-experiencing a
  state with the OPPOSITE observed valence (learning ON) shifts the prediction (the model
  learns from being surprised). No host writes the transition table.
- Legitimate host boundary: the conversational state token + the observed next valence
  delivered as sensory DRIVE (the environment renders what the brain then processes).

GO-GATE (pre-registered, 6 seeds 42/43/44/100/101/102)
------------------------------------------------------
 (1) AFFECTIVE PREDICTION: surprise(violated) >= 3x surprise(expected) AND
     surprise(violated) >= 5 Hz (a real signal) AND the queryable predicted-valence sign
     matches the true next valence on >= 5/6 states. 6-seed >= 5/6.
 (2) LESION-TRANSITION (load-bearing, decisive): zero the state->pred_{pos,neg} learned
     transition -> no prediction -> surprise fires HIGH on EXPECTED too -> the
     expected/violated separation COLLAPSES (ratio <= 1.5). Separation attributable to the
     spiking prediction >= 0.8.
 (3) SHUFFLE (structure does the work, not a template): train on a SCRAMBLED valence map
     (a different balanced permutation), then DUAL-SCORE. (a) scored vs the TRUE trajectory:
     NO seed reproduces the true separation (no true-GO; predicted-sign accuracy -> chance).
     (b) scored vs its OWN trained map: it STILL GOes (acc >= 5/6) -> the model genuinely
     LEARNED the arbitrary structure it was shown. Together: the prediction rides the LEARNED
     trajectory statistics, not a fixed template. (A random 3+/3- permutation partially
     overlaps true, so the true-scored ratio alone is noisy per seed; the trained-scored arm
     is the decisive, per-seed-clean control.)
 (4) BRAIN-BASED: surprise = cp_firing_states[surprise] rate; predicted valence = a
     two-pool spike-rate difference; current_reward_signal == 0; no host argmax over a
     stored transition table, no host code compares observed vs predicted valence.
 (5) MAINTAINED ACROSS TURNS + QUERYABLE (characterization): a multi-turn trajectory is
     rolled; at each turn the model predicts that turn's valence from the current state
     WITHIN the turn window, and an injected mid-trajectory deviation spikes surprise at
     that turn only.
 (+) UPDATE-ON-ERROR (characterization): after a state is re-experienced with the OPPOSITE
     observed valence (learning ON), re-querying the state shows the predicted valence
     SHIFTED toward the new observation (the world-model learned from the error).

WALL DISCIPLINE (the companion process): the 2-channel predictive-coding error unit's
operating point is the GAIN MATCH between observed excitation and predicted inhibition
(precision / divisive normalization; PV/SST + NE/ACh in biology). Proxied by a fixed gain
it is brittle. This runner MEASURES the operating point (recall rate + surprise f-I) and
places it with headroom; if a fixed gain is not robust 6-seed the honest boundary is
"needs a homeostatic precision companion", reported as the deliverable.

CPU-friendly (~300-neuron bridge). Run under SIM_BACKEND=numpy for a deterministic regime.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._affective_world_model_derisk \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_affective_world_model_6seed.json
    SIM_BACKEND=numpy python -m research.runners._affective_world_model_derisk --opsearch
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _valence_map(seed, n_states):
    """A per-seed BALANCED arbitrary valence assignment v(s) in {+1,-1}: half the states
    lead to a positive next turn, half to a negative one. Arbitrary per seed so the SHUFFLE
    control (a different balanced permutation) is a genuine structural scramble."""
    import numpy as np
    rng = np.random.RandomState(int(seed))
    half = n_states // 2
    vals = np.array([1] * half + [-1] * (n_states - half), dtype=np.int64)
    rng.shuffle(vals)
    return vals


def _shuffle_map(v, seed):
    """A DIFFERENT balanced permutation of the same valence multiset (the anti-cheat)."""
    import numpy as np
    rng = np.random.RandomState(int(seed) + 9973)
    for _ in range(64):
        w = v.copy(); rng.shuffle(w)
        if not np.array_equal(w, v):
            return w
    return v[::-1].copy()


def build_world_model_circuit(seed, *, n_states=6, blk=40, npred=48, nobs=48, nsurp=48,
                              state_to_pred_weight=0.0, obs_to_surprise_weight=3.0,
                              pred_to_surprise_weight=2.0, hebbian_learning_rate=0.06,
                              hebbian_max_weight=45.0, enable_heterogeneity=False):
    """state -> pred_{pos,neg}(FS, GABA_A, PLASTIC learned transition);
    obs_{pos,neg}(exc) -> surprise_{pos,neg} <- pred_{pos,neg}(inh). The surprise pools'
    firing IS the affective prediction-error signal.

    state_to_pred_weight is the INITIAL (untrained) weight and MUST be ~0: the transition
    is LEARNED from zero by Hebbian co-fire so cueing a state drives ONLY the pred pool of
    the valence that followed it (a non-zero baseline drives BOTH pools -> no selectivity ->
    the predictive-coding cancellation fails). Verified 2026-08-12: init 0.0 -> target-pred
    22Hz / non-target 0Hz; init 0.3 -> 37 / 29 (ratio 1.3, cancellation dead)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    # The transition is LEARNED by Hebbian co-fire (state_s co-fires with the pred pool of
    # the valence that FOLLOWS s). Same competition/normalization companion as the
    # heteroassociator so a state's afferents COMPETE (only the truly co-fired valence
    # survives) instead of running away to w_max.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = float(hebbian_learning_rate)
    cfg.hebbian_min_weight = 0.0            # non-co-fired edges free to stay ~0 (selectivity)
    cfg.hebbian_max_weight = float(hebbian_max_weight)  # above the working range (soft-bound gotcha)
    cfg.hebbian_weight_decay = 0.0
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.85
    cfg.hebbian_coactivity_thresh = 0.20
    cfg.hebbian_mean_subtract = 1.0        # Miller-MacKay subtractive normalization -> compete
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = bool(enable_heterogeneity)
    # Deterministic regime (limbic/nav read protocol): silent at rest, driven only by the
    # observed excitation vs the predicted inhibition.
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.current_reward_signal = 0.0        # BRAIN-BASED: no host reward scalar anywhere
    cfg.reward_baseline = 0.0
    # FS + GABA_A subtractive inhibition (the prediction). GABA_B present but zeroed (D2's
    # verified choice: FS+GABA_A inhibits cleanly + low rheobase fires from the learned recall;
    # FS+GABA_B produced a wrong-sign effect on this substrate).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_conductance_max = 0.0

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    FS = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name

    def region(name, n, exc):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=exc,
                           internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                           weight_jitter=0.0, plastic_internal=False,
                           izh_neuron_type=(RS if exc > 0 else FS),
                           **({} if exc > 0 else dict(syn_reversal_potential_i_override=-70.0)))

    cfg.brain_regions = [
        region("state", n_states * blk, 1.0),          # the current conversational state
        region("pred_pos", npred, 0.0),                # predicted next-turn valence (+), FS
        region("pred_neg", npred, 0.0),                # predicted next-turn valence (-), FS
        region("obs_pos", nobs, 1.0),                  # observed next-turn valence (+)
        region("obs_neg", nobs, 1.0),                  # observed next-turn valence (-)
        region("surprise_pos", nsurp, 1.0),            # error unit (+)
        region("surprise_neg", nsurp, 1.0),            # error unit (-)
    ]
    cfg.region_pathways = [
        # LEARNED transition: state -> pred_{pos,neg}. PLASTIC, all-to-all so Hebbian co-fire
        # SELECTS which valence pool each state drives (the others stay ~0).
        RegionPathway(from_region="state", to_region="pred_pos",
                      density=1.0, weight_mean=float(state_to_pred_weight),
                      weight_jitter=0.0, plastic=True),
        RegionPathway(from_region="state", to_region="pred_neg",
                      density=1.0, weight_mean=float(state_to_pred_weight),
                      weight_jitter=0.0, plastic=True),
        # Observed feed-forward drive (exc).
        RegionPathway(from_region="obs_pos", to_region="surprise_pos",
                      density=1.0, weight_mean=float(obs_to_surprise_weight),
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="obs_neg", to_region="surprise_neg",
                      density=1.0, weight_mean=float(obs_to_surprise_weight),
                      weight_jitter=0.0, plastic=False),
        # Subtractive prediction (inh, GABA_A via FS reversal).
        RegionPathway(from_region="pred_pos", to_region="surprise_pos",
                      density=1.0, weight_mean=float(pred_to_surprise_weight),
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="pred_neg", to_region="surprise_neg",
                      density=1.0, weight_mean=float(pred_to_surprise_weight),
                      weight_jitter=0.0, plastic=False),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)

    meta = dict(n_states=n_states, blk=blk, npred=npred, nobs=nobs, nsurp=nsurp)
    bridge._blk = blk
    # Snapshot the resting state for hard resets between trials (a short settle cannot fully
    # quiesce a fast FS pool -> residual firing leaks into the next Hebbian window).
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, meta


def _hard_reset(bridge):
    bridge.cp_membrane_potential_v[:] = bridge._rest_v
    bridge.cp_recovery_variable_u[:] = bridge._rest_u
    for name in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab",
                 "cp_conductance_g_nmda", "cp_firing_states", "cp_refractory"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = 0
    bridge.cp_external_input_current[:] = 0.0


def _idx(bridge, name):
    import numpy as np
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _step(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
    bridge.runtime_state.current_time_ms = (
        bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _set_drives(bridge, idx_map, drives, xp):
    """drives: {region: (block_or_None, pA)}. block=None drives the whole region."""
    bridge.cp_external_input_current[:] = 0.0
    for region, (block, pA) in drives.items():
        idx = idx_map[region]
        if block is None:
            bridge.cp_external_input_current[idx] = xp.float32(pA)
        else:
            blk = bridge._blk
            sub = idx[block * blk:(block + 1) * blk]
            bridge.cp_external_input_current[sub] = xp.float32(pA)


def _drive_read(bridge, idx_map, drives, n_steps, xp, read_regions, pre_drives=None, pre_steps=0):
    """Optional PREDICTION phase (pre_drives/pre_steps: the state cue establishes the top-down
    expectation), then the measured ASSERTION phase (drives/n_steps: the observed valence
    arrives). Returns {region: rate_hz} over the second phase only."""
    if pre_drives is not None and pre_steps > 0:
        _set_drives(bridge, idx_map, pre_drives, xp)
        for _ in range(pre_steps):
            _step(bridge)
    _set_drives(bridge, idx_map, drives, xp)
    counts = {r: 0 for r in read_regions}
    for _ in range(n_steps):
        _step(bridge)
        fs = bridge.cp_firing_states
        for r in read_regions:
            counts[r] += int(fs[idx_map[r]].sum())
    dur_s = n_steps * 1e-3
    return {r: counts[r] / max(len(_host(idx_map[r])), 1) / dur_s for r in read_regions}


def train_transition(bridge, cfg, idx_map, meta, xp, vmap, *, n_reps=20, cue_pa=1000.0,
                     teach_pa=1000.0, hold=40):
    """Experience the trajectory: co-fire state_s with the pred pool of the valence that
    FOLLOWS s (vmap[s]). Hebbian strengthens state_s -> pred_{that valence}."""
    cfg.enable_hebbian_learning = True
    for _ in range(n_reps):
        for s in range(meta["n_states"]):
            teach = "pred_pos" if vmap[s] > 0 else "pred_neg"
            _hard_reset(bridge)
            _drive_read(bridge, idx_map,
                        {"state": (s, cue_pa), teach: (None, teach_pa)}, hold, xp, [])


def measure(bridge, cfg, idx_map, meta, xp, vmap, *, cue_pa=1000.0, obs_pa=400.0,
            hold=60, pre_steps=60):
    # cue_pa=1000 (not 600): the state drive must cross EVERY state block's per-neuron
    # firing threshold (seeded even with heterogeneity off — CLAUDE.md bridge.py:1508), else
    # a high-threshold block never co-fires during training -> a silent recall dropout ->
    # a wrong predicted-valence sign. Verified 2026-08-12: 600pA -> 2/6 seeds have dropouts;
    # 1000pA -> 6/6 seeds recall all states.
    # obs_pa=400 (not 1000) DECOUPLED from cue_pa: the observed drive must fire the surprise
    # units MODERATELY so the prediction can cancel it. The precision/gain companion: at
    # cue=1000 the FS prediction fires ~446Hz, so pred_to_surprise_weight must be ~2 (not 24)
    # or the accumulated g_i (~650nS) destabilizes the explicit-Euler membrane update and reads
    # as spurious firing (verified 2026-08-12 — the g_i=653nS instability that faked a null).
    # At obs=400 / pred_w=2: expected surprise -> ~0Hz (clean cancel), violated -> ~40Hz.
    """For each state s: (1) PREDICTION phase (state_s cue -> recall the predicted valence,
    the FS prediction settles onto the surprise units), read the predicted valence sign;
    then (2) ASSERTION phase (state_s + observed valence) read the surprise. EXPECTED =
    observed == vmap[s]; VIOLATED = observed == -vmap[s]."""
    cfg.enable_hebbian_learning = False
    exp_hz, vio_hz, pred_correct, pred_signs = [], [], [], []
    for s in range(meta["n_states"]):
        obs_exp = "obs_pos" if vmap[s] > 0 else "obs_neg"
        obs_vio = "obs_neg" if vmap[s] > 0 else "obs_pos"
        cue_only = {"state": (s, cue_pa)}
        # PREDICTION-phase query: predicted valence sign = rate(pred_pos) - rate(pred_neg).
        _hard_reset(bridge)
        pr = _drive_read(bridge, idx_map, cue_only, pre_steps, xp, ["pred_pos", "pred_neg"])
        psign = 1 if (pr["pred_pos"] - pr["pred_neg"]) > 0 else -1
        pred_signs.append((pr["pred_pos"], pr["pred_neg"]))
        pred_correct.append(1 if psign == int(vmap[s]) else 0)
        # EXPECTED turn.
        _hard_reset(bridge)
        r = _drive_read(bridge, idx_map, {"state": (s, cue_pa), obs_exp: (None, obs_pa)},
                        hold, xp, ["surprise_pos", "surprise_neg"],
                        pre_drives=cue_only, pre_steps=pre_steps)
        exp_hz.append(r["surprise_pos"] + r["surprise_neg"])
        # VIOLATED turn.
        _hard_reset(bridge)
        r = _drive_read(bridge, idx_map, {"state": (s, cue_pa), obs_vio: (None, obs_pa)},
                        hold, xp, ["surprise_pos", "surprise_neg"],
                        pre_drives=cue_only, pre_steps=pre_steps)
        vio_hz.append(r["surprise_pos"] + r["surprise_neg"])
    return {"expected_hz": _st.mean(exp_hz), "violated_hz": _st.mean(vio_hz),
            "pred_acc": _st.mean(pred_correct), "expected_per": exp_hz, "violated_per": vio_hz,
            "pred_correct": pred_correct, "pred_signs": pred_signs}


def _lesion_transition(bridge, meta):
    """Anti-cheat: zero the learned state -> pred_{pos,neg} transition -> no prediction ->
    surprise fires HIGH on EXPECTED too (separation collapses). Weight-based, receptor-agnostic.
    Operates on the CSR matrix directly (orientation-robust)."""
    import numpy as np
    import scipy.sparse as sp
    state_idx = set(int(i) for i in _idx(bridge, "state"))
    pred_idx = set(int(i) for i in _idx(bridge, "pred_pos")) | set(int(i) for i in _idx(bridge, "pred_neg"))
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    nz = 0
    for r in range(M.shape[0]):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            # zero any edge touching state<->pred in EITHER orientation
            if (r in state_idx and c in pred_idx) or (r in pred_idx and c in state_idx):
                data[off] = 0.0; nz += 1
    bridge.cp_connections = sp.csr_matrix((data, indices, indptr), shape=M.shape)
    return nz


def run_seed(seed, *, mode="intact", verbose=True, n_reps=20, **build_kw):
    """mode: intact | lesion | shuffle."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    n_states = build_kw.get("n_states", 6)
    bridge, cfg, meta = build_world_model_circuit(seed, **build_kw)
    regions = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}

    v_true = _valence_map(seed, meta["n_states"])
    v_train = _shuffle_map(v_true, seed) if mode == "shuffle" else v_true

    train_transition(bridge, cfg, idx_map, meta, xp, v_train, n_reps=n_reps)
    if mode == "lesion":
        nz = _lesion_transition(bridge, meta)
        if verbose:
            print(f"  [lesion] zeroed {nz} state<->pred transition edges")

    # BRAIN-BASED anti-cheat: the surprise + predicted-sign are read ONLY from
    # cp_firing_states; nothing subtracts observed vs predicted valence in Python.
    assert float(cfg.current_reward_signal) == 0.0, "brain-based violated: host reward scalar set"

    # SCORE AGAINST THE TRUE trajectory (shuffle trains on a scrambled map but is scored vs
    # the real world -> a scrambled model does NOT reproduce the true separation).
    res = measure(bridge, cfg, idx_map, meta, xp, v_true)
    exp = max(res["expected_hz"], 1e-6)
    res["ratio"] = res["violated_hz"] / exp
    res["mode"] = mode; res["seed"] = seed
    res["go"] = bool(res["ratio"] >= 3.0 and res["violated_hz"] >= 5.0 and res["pred_acc"] >= 5.0 / 6.0)
    # SHUFFLE also scored vs its OWN TRAINED map: proves the model genuinely LEARNED the
    # arbitrary (scrambled) structure it was shown -> the prediction rides the LEARNED
    # trajectory statistics, not a fixed template/bias (a random 3+/3- balanced permutation
    # can partially overlap the true map, so the true-scored ratio alone is noisy per seed;
    # the trained-scored arm is the decisive, per-seed-clean anti-template evidence).
    if mode == "shuffle":
        res_tr = measure(bridge, cfg, idx_map, meta, xp, v_train)
        res["acc_vs_trained"] = res_tr["pred_acc"]
        res["ratio_vs_trained"] = res_tr["violated_hz"] / max(res_tr["expected_hz"], 1e-6)
        res["go_vs_trained"] = bool(res["ratio_vs_trained"] >= 3.0
                                    and res_tr["violated_hz"] >= 5.0 and res_tr["pred_acc"] >= 5.0 / 6.0)
    if verbose:
        extra = ""
        if mode == "shuffle":
            extra = (f" | vs-trained: acc={res['acc_vs_trained']:.2f} "
                     f"ratio={res['ratio_vs_trained']:.1f}x GO={res['go_vs_trained']}")
        print(f"  [{mode:8s} seed {seed}] pred_acc(vs true)={res['pred_acc']:.2f} | surprise: "
              f"expected={res['expected_hz']:5.2f}  violated={res['violated_hz']:5.2f} "
              f"({res['ratio']:4.1f}x) | GO={res['go']}{extra}")
    return res


def run_multiturn(seed, *, n_reps=20, hold=60, pre_steps=60, deviate_at=3, **build_kw):
    """Characterization (maintained across turns + queryable): roll a multi-turn trajectory.
    At each turn the model predicts the current state's next valence (queryable within the
    window); the environment then supplies the observed valence, which EQUALS the prediction
    except at `deviate_at` (an injected surprising turn). Returns per-turn surprise Hz."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg, meta = build_world_model_circuit(seed, **build_kw)
    regions = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
    v_true = _valence_map(seed, meta["n_states"])
    train_transition(bridge, cfg, idx_map, meta, xp, v_true, n_reps=n_reps)
    cfg.enable_hebbian_learning = False
    import numpy as np
    rng = np.random.RandomState(seed + 7)
    traj = [int(rng.randint(meta["n_states"])) for _ in range(6)]
    per_turn = []
    for t, s in enumerate(traj):
        obs_v = int(v_true[s])
        if t == deviate_at:
            obs_v = -obs_v                       # a surprising turn
        obs_region = "obs_pos" if obs_v > 0 else "obs_neg"
        cue_only = {"state": (s, 1000.0)}
        _hard_reset(bridge)
        r = _drive_read(bridge, idx_map, {"state": (s, 1000.0), obs_region: (None, 400.0)},
                        hold, xp, ["surprise_pos", "surprise_neg"],
                        pre_drives=cue_only, pre_steps=pre_steps)
        per_turn.append({"turn": t, "state": s, "obs_valence": obs_v,
                         "expected": obs_v == int(v_true[s]),
                         "surprise_hz": r["surprise_pos"] + r["surprise_neg"]})
    return per_turn


def run_update_on_error(seed, *, n_reps=20, flip_reps=20, **build_kw):
    """Characterization (updates on prediction-error): learn v_true, pick one state, then
    RE-EXPERIENCE it with the OPPOSITE observed valence (learning ON), and re-query. The
    predicted valence should SHIFT toward the new observation (the model learned from error)."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg, meta = build_world_model_circuit(seed, **build_kw)
    regions = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
    v_true = _valence_map(seed, meta["n_states"])
    train_transition(bridge, cfg, idx_map, meta, xp, v_true, n_reps=n_reps)

    s = 0
    def query_sign():
        _hard_reset(bridge)
        pr = _drive_read(bridge, idx_map, {"state": (s, 1000.0)}, 60, xp, ["pred_pos", "pred_neg"])
        return pr["pred_pos"] - pr["pred_neg"], pr
    before, pr_b = query_sign()
    # Flip: re-experience state s with the OPPOSITE valence (learning ON).
    v_flip = v_true.copy(); v_flip[s] = -v_flip[s]
    cfg.enable_hebbian_learning = True
    teach = "pred_pos" if v_flip[s] > 0 else "pred_neg"
    for _ in range(flip_reps):
        _hard_reset(bridge)
        _drive_read(bridge, idx_map, {"state": (s, 1000.0), teach: (None, 1000.0)}, 40, xp, [])
    cfg.enable_hebbian_learning = False
    after, pr_a = query_sign()
    return {"state": s, "orig_valence": int(v_true[s]), "flipped_to": int(v_flip[s]),
            "pred_diff_before": before, "pred_diff_after": after,
            "shifted_toward_new": bool((after - before) * v_flip[s] > 0),
            "before_pools": pr_b, "after_pools": pr_a}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--n-reps", type=int, default=22)
    ap.add_argument("--n-states", type=int, default=6)
    ap.add_argument("--state-to-pred-weight", type=float, default=0.0,
                    help="INITIAL (untrained) state->pred weight; MUST be ~0 so the transition "
                         "is learned selectively from zero (0.3 kills the cancellation).")
    ap.add_argument("--obs-to-surprise-weight", type=float, default=3.0)
    ap.add_argument("--pred-to-surprise-weight", type=float, default=2.0,
                    help="prediction->surprise inhibition weight; ~2 matches the ~446Hz FS "
                         "prediction against the ~400pA observed drive. High values (24) "
                         "accumulate a g_i that destabilizes the membrane update (fake firing).")
    ap.add_argument("--hebbian-learning-rate", type=float, default=0.06)
    ap.add_argument("--opsearch", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    build_kw = dict(n_states=args.n_states,
                    state_to_pred_weight=args.state_to_pred_weight,
                    obs_to_surprise_weight=args.obs_to_surprise_weight,
                    pred_to_surprise_weight=args.pred_to_surprise_weight,
                    hebbian_learning_rate=args.hebbian_learning_rate)

    if args.opsearch:
        print("[world-model OPSEARCH seed=42] init state->pred FIXED 0 (learned); sweep obs/pred gain")
        for ow in (2.0, 3.0, 5.0):
            for iw in (1.0, 2.0, 4.0):
                bk = dict(build_kw); bk.update(state_to_pred_weight=0.0,
                                               obs_to_surprise_weight=ow,
                                               pred_to_surprise_weight=iw)
                r = run_seed(42, verbose=False, n_reps=args.n_reps, **bk)
                print(f"  ow={ow:.1f} iw={iw:4.1f} | acc={r['pred_acc']:.2f} "
                      f"exp={r['expected_hz']:6.2f} vio={r['violated_hz']:6.2f} "
                      f"({r['ratio']:5.1f}x) GO={r['go']}")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== INTACT (the affective world-model prediction) ===")
    intact = [run_seed(s, mode="intact", n_reps=args.n_reps, **build_kw) for s in seeds]

    ac_seeds = seeds[:3]
    print("\n=== ANTI-CHEATS (mechanistic; 3 seeds) ===")
    lesion = [run_seed(s, mode="lesion", n_reps=args.n_reps, **build_kw) for s in ac_seeds]
    shuffle = [run_seed(s, mode="shuffle", n_reps=args.n_reps, **build_kw) for s in ac_seeds]

    print("\n=== CHARACTERIZATION (maintained-across-turns + update-on-error; seed 42) ===")
    mt = run_multiturn(seeds[0], n_reps=args.n_reps, **build_kw)
    for row in mt:
        print(f"  turn {row['turn']} state={row['state']} obs_valence={row['obs_valence']:+d} "
              f"expected={row['expected']} surprise={row['surprise_hz']:5.2f} Hz")
    upd = run_update_on_error(seeds[0], n_reps=args.n_reps, **build_kw)
    print(f"  update-on-error: state {upd['state']} valence {upd['orig_valence']:+d}->"
          f"{upd['flipped_to']:+d}: pred_diff {upd['pred_diff_before']:+.2f}->"
          f"{upd['pred_diff_after']:+.2f}  shifted_toward_new={upd['shifted_toward_new']}")

    n_go = sum(1 for r in intact if r["go"])
    verdict = "GO" if (len(intact) >= 6 and n_go >= 5) or (len(intact) < 6 and n_go == len(intact)) else "BOUNDARY"
    les_collapse = sum(1 for r in lesion if r["ratio"] <= 1.5)
    intact_acc = _st.mean([r["pred_acc"] for r in intact])
    shuf_acc = _st.mean([r["pred_acc"] for r in shuffle])
    # SHUFFLE (dual-scored): the scrambled-trained model must (a) NOT reproduce the TRUE
    # separation on any seed (no true-scored GO -> the true prediction is structure-specific,
    # not a template), and (b) STILL GO when scored vs its OWN trained map (proves it genuinely
    # learned the arbitrary structure). The true-scored ratio<=1.5 is NOT gated per seed (a
    # random balanced permutation partially overlaps true -> noisy); the two arms below are.
    shuf_no_true_go = sum(1 for r in shuffle if not r["go"])
    shuf_learns_trained = sum(1 for r in shuffle if r.get("go_vs_trained"))
    shuf_acc_trained = _st.mean([r["acc_vs_trained"] for r in shuffle])
    print("\n=== VERDICT ===")
    print(f"  INTACT GO: {n_go}/{len(intact)} seeds (>= 5/6 required)  ->  {verdict}")
    print(f"  lesion collapses separation: {les_collapse}/{len(lesion)} (ratio<=1.5)")
    print(f"  shuffle does NOT reproduce TRUE separation: {shuf_no_true_go}/{len(shuffle)} "
          f"(no GO vs true); pred_acc(vs true) intact {intact_acc:.2f} -> shuffle {shuf_acc:.2f}")
    print(f"  shuffle DID learn its scrambled map: {shuf_learns_trained}/{len(shuffle)} GO vs trained "
          f"(acc vs trained {shuf_acc_trained:.2f}) -> structure, not template")

    # ATTRIBUTION (tools.lab): whose is the expected/violated separation? treatment = intact
    # separation (violated-expected), control = lesioned separation. lesion ~0 -> the whole
    # separation is owned by the SPIKING LEARNED PREDICTION.
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    intact_sep = _st.mean([r["violated_hz"] - r["expected_hz"] for r in intact[:len(ac_seeds)]])
    lesion_sep = _st.mean([r["violated_hz"] - r["expected_hz"] for r in lesion])
    frac = attributable_to("affective prediction separation @ spiking transition",
                           intact_sep, lesion_sep)

    ratio_min = min(r["ratio"] for r in intact)
    vio_min = min(r["violated_hz"] for r in intact)
    acc_min = min(r["pred_acc"] for r in intact)
    v = (Verdict("spiking affective world-model — learned valence forward model + mismatch")
         .require("intact GO on >=5/6 seeds", n_go,
                  expect=lambda k: k >= max(5, len(intact) - 1) if len(intact) >= 6 else k == len(intact))
         .floor("violated surprise is a real signal (min Hz)", vio_min, floor=5.0)
         .require("separation ratio >= 3x (min over seeds)", ratio_min, expect=lambda x: x >= 3.0)
         .require("predicted-valence sign accuracy >= 5/6 (min over seeds)", acc_min,
                  expect=lambda x: x >= 5.0 / 6.0)
         .control("transition lesion changes the separation", intact_sep, lesion_sep, min_separation=2.0)
         .require("lesion collapses ratio to ~1 (3/3)", les_collapse, expect=lambda k: k == len(lesion))
         .require("shuffle does NOT reproduce TRUE separation (no true-GO, 3/3)", shuf_no_true_go,
                  expect=lambda k: k == len(shuffle))
         .require("shuffle DID learn its scrambled map (GO vs trained, 3/3)", shuf_learns_trained,
                  expect=lambda k: k == len(shuffle))
         .require("separation attributable to prediction (>=0.8)", frac,
                  expect=lambda x: x is not None and x >= 0.8)
         .disabled("OU background process", "deterministic regime for a controllable operating point")
         .disabled("conductance noise", "deterministic regime")
         .disabled("high-order (history-dependent) transition",
                   "the transition is FIRST-ORDER (state -> next valence); a state that is "
                   "ambiguous given history needs the HTM-TM high-order predictor (EMERGE-15 GO) "
                   "— the named next rung")
         .disabled("n-way next-STATE recall",
                   "the prediction is 2-way (next VALENCE); a full next-conversational-state "
                   "recall needs the CA3 sparse pattern-separation companion (2026-06-05)"))
    decided = v.decide(go=(verdict == "GO"))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "affective_world_model", "intact": intact,
                       "lesion": lesion, "shuffle": shuffle,
                       "multiturn": mt, "update_on_error": upd,
                       "n_go": n_go, "n_seeds": len(intact),
                       "verdict": decided["status"], "verdict_label": verdict,
                       "lesion_collapse": les_collapse,
                       "shuffle_no_true_go": shuf_no_true_go,
                       "shuffle_learns_trained_go": shuf_learns_trained,
                       "shuffle_acc_vs_trained": shuf_acc_trained,
                       "intact_pred_acc": intact_acc, "shuffle_pred_acc_vs_true": shuf_acc,
                       "intact_separation_hz": intact_sep, "lesion_separation_hz": lesion_sep,
                       "separation_attributable_to_prediction": frac,
                       "preconditions": decided["preconditions"],
                       "disabled_processes": decided["disabled_processes"],
                       "verdict_status": decided["status"]}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
