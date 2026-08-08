"""BIOLOGIZE the forward-model world-model READ-OUT: host ridge/argmax -> a SPIKING SYNAPTIC read-out.

THE SHORTCUT BEING BURNED DOWN. The wired world-model content decode (`fm_decode` in
`_stageA_full_integration_derisk.py`; the same shape as `_forward_model_reservoir_derisk`'s host read-out) is a
DECLARED host shortcut: `argmax(spikecounts @ Ws)` -- a ridge least-squares map + a host argmax over the frozen
reservoir spike-counts. The reservoir SPIKES are brain-based; the linear decode + the argmax winner are host numpy.

THE BIOLOGIZATION (this runner). Keep the SPIKING reservoir (`OnBridgeLSM` statistics, on a real `SimulationBridge`)
and the LOCAL delta rule that TRAINS the read-out weights (a post-synaptic-error x pre-synaptic-activity three-factor
rule -- biologically legitimate). REPLACE the host `feat @ Ws` + `argmax` with a GENUINE SPIKING SYNAPTIC read-out,
co-resident on the SAME bridge as the reservoir:
  * the trained read-out weights become EXCITATORY synapses reservoir_slice -> output ensembles (`Ws_shifted =
    Ws - Ws.min()` per block: Dale-legal, and a uniform per-block offset PRESERVES the argmax -- the rungB1c insight);
  * the predicted next state s' = (x', y') is read by a NEURAL WINNER-TAKE-ALL: G output ensembles per coordinate
    block compete through shared mutual inhibition; the surviving ensemble's LABEL is the predicted coordinate. The
    winner is a neural read of the co-resident ensembles' firing (`cp_firing_states`), NOT a host `feat @ Ws` argmax.

So the whole predict-s' path runs on NEURONS + SYNAPSES: reservoir spikes --Ws_shifted synapses--> output ensembles
--> mutual-inhibition WTA --> the winning ensemble's coordinate. The host ridge/argmax is GONE from the content path.

REUSES (by import; NO `sim/` edit -- all wiring is runner-side via `inject_explicit_wiring` / `set_pathway_weights`):
  * `_forward_model_reservoir_derisk`: the 5x5 toroidal world, the (state,action) encoders, the LOCAL delta rule
    (`_train_delta`), the target/decode helpers, the constituent-coverage held-out split.
  * `_emerge82_onbridge_lsm_derisk`: the reservoir statistics (EXC/INH weights, IN_SCALE, BIAS, T_STEP, density) +
    the wash-out snapshot/restore.
  * `_rungB1c_spiking_reservoir_synaptic_readout_derisk`: the Ws_shifted-synapse + mutual-inhibition-WTA + neural-
    winner recipe (ported here onto the region-framework SimulationBridge that the forward model already uses).

SUCCESS (single-seed SMOKE first; 6-seed for the parent). The spiking synaptic read-out MATCHES the host ridge
read-out's HELD-OUT accuracy on the toy world (ridge ~0.72+; GO if synaptic held-out >= ridge - 0.10 AND >> chance
AND >> retrieval) with the winner selected NEURALLY. Anti-cheats: (i) NEURAL SOURCE -- the winner is off
`cp_firing_states`, reservoir genuinely active; (ii) LESION -- zeroing the read-out synapses OR silencing the
reservoir collapses held-out to ~chance; (iii) MATCHED SHAM -- a count-matched lesion of an OFF-DECODE decoy pathway
leaves held-out UNCHANGED (the collapse is specific to the read-out synapses, not to perturbation magnitude);
(iv) the host ridge/argmax is GONE from the content path (grep-verified: `_neural_predict` contains no `@ Ws` /
`argmax(... @ ...)`). HONEST NEGATIVE acceptable: if the spiking read-out UNDER-performs the ridge (the sub-margin
under-resolves on spikes), that maps exactly what the substrate needs (a signed on/off read-out / a wider reservoir /
a better-conditioned draw -- the rungB1c residual), and is a first-class deliverable.

GO bar: reservoir active AND synaptic_heldout >= ridge_heldout - 0.10 AND synaptic_heldout - max(chance,retrieval)
>= 0.30 AND readout-lesion collapses by >= 0.30 AND reservoir-silence collapses by >= 0.30 AND matched-sham
UNCHANGED (|delta| <= 0.08) AND seeded (byte-identical substrate) AND the neural winner is off cp_firing_states.
BOUNDARY otherwise -- name the residual, do NOT force GO.

6-SEED:  SIM_BACKEND=numpy python -m research.runners._fm_spiking_synaptic_readout_derisk --seeds 42 43 44 100 101 102
SMOKE:   SIM_BACKEND=numpy python -m research.runners._fm_spiking_synaptic_readout_derisk --seeds 42 --smoke
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import dataclasses as _dc
import hashlib
import json
import random as _random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---- reuse the world + local read-out rule + split (by import; NO copy) ----
from research.runners._forward_model_reservoir_derisk import (   # noqa: E402
    _ACTIONS, _all_pairs, _step, _encode_seq, _target, _train_delta, _decode,
)
# ---- reuse the wash-out snapshot/restore ----
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (   # noqa: E402
    _snapshot_state, _restore_state,
)
from sim.backend import get_backend, to_host   # noqa: E402
from tools.lab import lever   # noqa: E402
from tools.verdict import Verdict   # noqa: E402

# ── reservoir statistics (EMERGE-82, verbatim) ───────────────────────────────────────────────────────────────────
RES_EXC_W = 6.0
RES_INH_W = 8.0
RES_INTERNAL_DENSITY = 0.1
RES_WEIGHT_JITTER = 0.3
RES_EXC_FRACTION = 0.8
RES_IN_SCALE = 320.0
RES_BIAS = 45.0
RES_T_STEP = 12            # steps/token for the pure reservoir FEATURE read (EMERGE-82 statistics)

# ── the spiking synaptic read-out (output ensembles + mutual-inhibition WTA) ──────────────────────────────────────
ENS_P = 16                 # neurons per output ensemble (one ensemble per coordinate class)
WTA_INH = 12               # shared inhibitory pool per coordinate block (the mutual inhibition)
READ_T_STEP = 24           # steps/token for the SYNAPTIC read-out window (more spike samples -> resolve the margin)
WS_ENS_FLOOR = 40.0        # base tonic (pA) per output ensemble (all fire; the Ws_shifted synapses carry
#                            the winner's drive ADVANTAGE -- the genuine SYNAPTIC selection, not an inhibition crutch)
WTA_W_EI = 8.0             # ens -> inh (excite the shared inhibition)
WTA_W_EE = 3.0             # ens -> ens within-ensemble (positive feedback / ensemble self-sustain)
WTA_W_IE = 14.0            # inh -> ens (the biased-competition mutual inhibition)
SYN_SCALE_GRID = (1.6, 3.6, 8.0)                   # Ws_shifted->ens synapse scale (auto-swept, selected on TRAIN)

# ── the COMMON-MODE CANCELLER (feedforward divisive/subtractive normalization) ────────────────────────────────────
# Realizing a SIGNED read-out weight vector as NON-NEGATIVE (Dale-legal) synapses forces a uniform per-block offset
# `|Ws.min()|` onto every weight; that offset injects a COMMON-MODE drive `|Ws.min()| * (total reservoir spikes)`
# identical across the block's ensembles -- ~140x the discriminative top1-top2 margin here, so the winner is
# swamped (the rungB1c read-out-resolution wall). The biological surpass rungB1c NAMED (a signed read-out) is built
# here as a per-block feedforward INHIBITORY pool: it receives UNIFORM excitation from the reservoir and delivers
# UNIFORM inhibition to every ensemble in the block, SUBTRACTING the common-mode so the ensembles compete on the
# discriminative margin ALONE (the ubiquitous feedforward-interneuron normalization; Dale-legal -- the reservoir
# excites the interneuron, the interneuron inhibits the ensembles). One scalar `cm_gain` is auto-swept with the
# read-out scale on TRAIN agreement (no held-out peek).
CM_N = 8                   # inhibitory canceller neurons per block
CM_IN_BASE = 0.9           # reservoir(exc) -> canceller uniform excitatory weight per unit shift-magnitude*scale
CM_OUT_BASE = 4.0          # canceller -> ensemble uniform inhibitory weight (x cm_gain)
CM_GAIN_GRID = (0.0, 0.6, 1.0, 1.6, 2.4)           # 0.0 == canceller OFF (the naive Dale-shift baseline)


# =====================================================================================================================
# ONE bridge: reservoir + x/y output blocks (G ensembles each + shared inhibition) + a count-matched OFF-DECODE decoy.
# All wiring runner-side; the reservoir recurrence is injected with an INDEPENDENT rng so it is byte-identical
# regardless of how many other regions exist (the stageA SEAM-A pattern).
# =====================================================================================================================
def _build_bridge(seed, G, n_pool, ens_p=ENS_P, inh=WTA_INH):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    from sim.config import NeuronModel

    n_dec = 2 * G  # decoy ensembles, count-matched to x(G)+y(G) output blocks
    regions = [
        BrainRegion(name="reservoir", n_neurons=int(n_pool), exc_fraction=RES_EXC_FRACTION,
                    internal_density=0.0, exc_weight_mean=RES_EXC_W, inh_weight_mean=RES_INH_W,
                    weight_jitter=RES_WEIGHT_JITTER, plastic_internal=False),
        BrainRegion(name="x_ens", n_neurons=int(G * ens_p), exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="x_inh", n_neurons=int(inh), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_ens", n_neurons=int(G * ens_p), exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_inh", n_neurons=int(inh), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="x_cm", n_neurons=int(CM_N), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_cm", n_neurons=int(CM_N), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="dec_ens", n_neurons=int(n_dec * ens_p), exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.dt = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                              # ⛔ seed the SUBSTRATE (not actual_seed_used)
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.enable_ou_process = False
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_structural_pruning"):
        setattr(cfg, f, False)
    cfg.enable_nmda = False
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0

    rt = RuntimeState()
    rt.actual_seed_used = int(seed)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)

    rm = b.region_manager
    res_idx = np.asarray(rm.indices("reservoir"), dtype=np.int64)
    # Dale: the read-out must sample only the EXCITATORY reservoir population -- an inhibitory neuron's read-out
    # synapse would deliver inhibition (g_i), inverting its feature contribution and breaking the Ws_shifted
    # argmax-preservation. The inhibitory subset still shapes the reservoir E/I DYNAMICS; it just does not project
    # to the read-out (biologically standard: read-outs sample the excitatory output population).
    res_inh = np.asarray(rm.inhibitory_indices("reservoir"), dtype=np.int64)
    res_exc = np.asarray(sorted(set(res_idx.tolist()) - set(res_inh.tolist())), dtype=np.int64)
    x_ens_idx = np.asarray(rm.indices("x_ens"), dtype=np.int64)
    x_inh_idx = np.asarray(rm.indices("x_inh"), dtype=np.int64)
    y_ens_idx = np.asarray(rm.indices("y_ens"), dtype=np.int64)
    y_inh_idx = np.asarray(rm.indices("y_inh"), dtype=np.int64)
    x_cm_idx = np.asarray(rm.indices("x_cm"), dtype=np.int64)
    y_cm_idx = np.asarray(rm.indices("y_cm"), dtype=np.int64)
    dec_idx = np.asarray(rm.indices("dec_ens"), dtype=np.int64)

    # per-ensemble slices
    x_ens = [x_ens_idx[k * ens_p:(k + 1) * ens_p] for k in range(G)]
    y_ens = [y_ens_idx[k * ens_p:(k + 1) * ens_p] for k in range(G)]
    dec_ens = [dec_idx[k * ens_p:(k + 1) * ens_p] for k in range(n_dec)]

    idx = dict(res=res_idx, res_exc=res_exc, x_ens=x_ens, x_inh=x_inh_idx, y_ens=y_ens, y_inh=y_inh_idx,
               x_cm=x_cm_idx, y_cm=y_cm_idx,
               dec_ens=dec_ens, x_ens_all=x_ens_idx, y_ens_all=y_ens_idx, dec_all=dec_idx, ens_p=int(ens_p))

    # fixed-random reservoir input projection W_in (EMERGE-82 statistics; seed-derived, independent of region count)
    rng = np.random.default_rng(int(seed) * 7919 + 3)
    in_dim = 2 * G + len(_ACTIONS)
    W_in = (rng.random((len(res_idx), in_dim)) * 2 - 1) * RES_IN_SCALE
    idx["W_in"] = W_in
    idx["in_dim"] = int(in_dim)
    return b, rm, idx, cfg


def _reservoir_internal(rm, seed):
    """Inject the reservoir's fixed-random Erdos-Renyi recurrence with an INDEPENDENT rng (byte-identical regardless
    of how many other regions are appended -- the stageA SEAM-A decoupling)."""
    res_region = next(r for r in rm.regions() if r.name == "reservoir")
    shadow = _dc.replace(res_region, internal_density=RES_INTERNAL_DENSITY)
    internal = rm._build_region_internal(shadow, _random.Random(int(seed) * 100003 + 7))
    return internal


def _dense(pre_idx, post_idx, weight):
    pre = np.repeat(np.asarray(pre_idx, dtype=np.int64), len(post_idx))
    post = np.tile(np.asarray(post_idx, dtype=np.int64), len(pre_idx))
    ww = np.full(pre.shape[0], float(weight), dtype=np.float32)
    return {"pre_indices": pre, "post_indices": post, "initial_weights": ww, "plastic": False, "conn_type": "E_TO_E"}


def _readout_edges(res_idx, ens_list, Ws_block, syn_scale):
    """Ws_shifted read-out synapses: reservoir neuron i -> every neuron of ensemble r at weight
    syn_scale * (Ws_block[r] - Ws_block.min())[i]  (Dale-legal; the per-block uniform offset preserves the argmax)."""
    shift = float(np.min(Ws_block))
    pre, post, w = [], [], []
    for r, ens in enumerate(ens_list):
        wr = (Ws_block[r] - shift) * float(syn_scale)      # (n_res,) >= 0
        for j, i in enumerate(res_idx):
            wv = wr[j]
            for e in ens:
                pre.append(int(i)); post.append(int(e)); w.append(float(wv))
    return (np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64), np.asarray(w, dtype=np.float32))


def _canceller_edges(res_idx, cm_idx, ens_all, shift_mag, syn_scale, cm_gain):
    """The feedforward common-mode canceller for one block. reservoir(exc) -> cm at a UNIFORM excitatory weight
    proportional to the block's shift-magnitude*scale (so the interneuron tracks the common-mode `|shift|*S`);
    cm -> every ensemble neuron at a UNIFORM inhibitory weight (x cm_gain). Returns two edge dicts."""
    w_in = float(CM_IN_BASE) * float(shift_mag) * float(syn_scale)
    w_out = float(CM_OUT_BASE) * float(cm_gain)
    # reservoir(exc) -> cm  (uniform excitatory)
    pin = np.repeat(np.asarray(res_idx, np.int64), len(cm_idx))
    qin = np.tile(np.asarray(cm_idx, np.int64), len(res_idx))
    din = {"pre_indices": pin, "post_indices": qin,
           "initial_weights": np.full(len(pin), w_in, np.float32), "plastic": False, "conn_type": "E_TO_E"}
    # cm -> ensembles (uniform inhibitory; cm neurons carry the inhibitory trait)
    pout = np.repeat(np.asarray(cm_idx, np.int64), len(ens_all))
    qout = np.tile(np.asarray(ens_all, np.int64), len(cm_idx))
    dout = {"pre_indices": pout, "post_indices": qout,
            "initial_weights": np.full(len(pout), w_out, np.float32), "plastic": False, "conn_type": "I_TO_E"}
    return din, dout


def _wire(b, rm, idx, seed, Ws_x, Ws_y, syn_scale, cm_gain=1.0):
    """Inject EVERYTHING runner-side (reservoir recurrence + Ws_shifted read-out synapses + the per-block common-mode
    CANCELLER + the two WTA blocks + the count-matched decoy read-out). Output-inhibitory neurons (x_inh, y_inh,
    x_cm, y_cm, reservoir inh subset) get the inhibitory trait so their synapses are inhibitory. Returns the
    readout/decoy edge tuples (for lesioning)."""
    res = idx["res_exc"]; ens_p = idx["ens_p"]   # read-out samples the EXCITATORY reservoir population only
    union = {}
    ri = _reservoir_internal(rm, seed)
    if ri is not None:
        union["reservoir_internal"] = ri

    # ---- read-out synapses reservoir(exc) -> x/y output ensembles (Ws_shifted) ----
    px, qx, wx = _readout_edges(res, idx["x_ens"], Ws_x, syn_scale)
    py, qy, wy = _readout_edges(res, idx["y_ens"], Ws_y, syn_scale)
    union["readout_x"] = {"pre_indices": px, "post_indices": qx, "initial_weights": wx,
                          "plastic": False, "conn_type": "E_TO_E"}
    union["readout_y"] = {"pre_indices": py, "post_indices": qy, "initial_weights": wy,
                          "plastic": False, "conn_type": "E_TO_E"}
    # ---- count-matched OFF-DECODE decoy read-out (reservoir -> dec_ens); never read by the decoder ----
    # concat the x and y Ws so the decoy carries the SAME per-edge weights as the real read-out (matched sham).
    Ws_dec = np.concatenate([Ws_x, Ws_y], axis=0)           # (2G, n_res)
    pd, qd, wd = _readout_edges(res, idx["dec_ens"], Ws_dec, syn_scale)
    union["readout_dec"] = {"pre_indices": pd, "post_indices": qd, "initial_weights": wd,
                            "plastic": False, "conn_type": "E_TO_E"}

    # ---- the per-block COMMON-MODE CANCELLER (feedforward normalization; the signed-read-out surpass) ----
    if cm_gain > 0.0:
        din_x, dout_x = _canceller_edges(res, idx["x_cm"], idx["x_ens_all"], abs(float(np.min(Ws_x))),
                                         syn_scale, cm_gain)
        din_y, dout_y = _canceller_edges(res, idx["y_cm"], idx["y_ens_all"], abs(float(np.min(Ws_y))),
                                         syn_scale, cm_gain)
        union["cm_in_x"] = din_x; union["cm_out_x"] = dout_x
        union["cm_in_y"] = din_y; union["cm_out_y"] = dout_y

    # ---- the two mutual-inhibition WTA blocks (x, y) ----
    def _wta(ens_list, inh_idx, tag):
        pre_ei, post_ei = [], []
        for ens in ens_list:
            for a in ens:
                for bb in inh_idx:
                    pre_ei.append(int(a)); post_ei.append(int(bb))
        union[f"wta_e2i_{tag}"] = {"pre_indices": np.asarray(pre_ei, np.int64),
                                   "post_indices": np.asarray(post_ei, np.int64),
                                   "initial_weights": np.full(len(pre_ei), WTA_W_EI, np.float32),
                                   "plastic": False, "conn_type": "E_TO_E"}
        pre_ee, post_ee = [], []
        for ens in ens_list:
            for a in ens:
                for bb in ens:
                    if a != bb:
                        pre_ee.append(int(a)); post_ee.append(int(bb))
        union[f"wta_e2e_{tag}"] = {"pre_indices": np.asarray(pre_ee, np.int64),
                                   "post_indices": np.asarray(post_ee, np.int64),
                                   "initial_weights": np.full(len(pre_ee), WTA_W_EE, np.float32),
                                   "plastic": False, "conn_type": "E_TO_E"}
        all_ens = np.concatenate(ens_list)
        pre_ie, post_ie = [], []
        for a in inh_idx:
            for bb in all_ens:
                pre_ie.append(int(a)); post_ie.append(int(bb))
        union[f"wta_i2e_{tag}"] = {"pre_indices": np.asarray(pre_ie, np.int64),
                                   "post_indices": np.asarray(post_ie, np.int64),
                                   "initial_weights": np.full(len(pre_ie), WTA_W_IE, np.float32),
                                   "plastic": False, "conn_type": "I_TO_E"}
    _wta(idx["x_ens"], idx["x_inh"], "x")
    _wta(idx["y_ens"], idx["y_inh"], "y")

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    return {"readout_x": (px, qx), "readout_y": (py, qy), "readout_dec": (pd, qd)}


# =====================================================================================================================
# DRIVE + READ
# =====================================================================================================================
def _reservoir_feature(b, idx, snap, U, silence=False, t_step=RES_T_STEP):
    """Pure reservoir FEATURE read (the training/host-reference feature). Wash, drive reservoir per token, accumulate
    the reservoir slice's spike-count. No ensemble injection (ens do not feed back to the reservoir, so the feature
    is clean). Returns per-neuron mean spike-count over the sequence."""
    xp, _ = get_backend()
    res = idx["res"]; res_exc = idx["res_exc"]; W_in = idx["W_in"]
    res_dev = xp.asarray(res); exc_dev = xp.asarray(res_exc)
    _restore_state(b, snap)
    b.cp_external_input_current[:] = 0.0
    counts = np.zeros(len(res_exc), np.float64)   # feature over the EXCITATORY reservoir population
    steps = 0
    for t in range(len(U)):
        drive = np.zeros(len(res)) if silence else (W_in @ np.asarray(U[t]) + RES_BIAS)
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[res_dev] = xp.asarray(drive.astype(np.float32))
        for _ in range(int(t_step)):
            b._run_one_simulation_step()
            counts += np.asarray(to_host(b.cp_firing_states[exc_dev]), dtype=np.float64)
            steps += 1
    _restore_state(b, snap)
    b.cp_external_input_current[:] = 0.0
    return counts / max(1, steps)


def _neural_predict(b, idx, snap, U, G, silence=False, floors_x=None, floors_y=None, t_step=READ_T_STEP,
                    replay=1):
    """THE BIOLOGIZED READ-OUT. Drive the reservoir per (s,a) token; the reservoir's SPIKES drive the x/y output
    ensembles THROUGH the Ws_shifted synapses; a per-ensemble homeostatic floor (the intrinsic-excitability set-point,
    calibrated on TRAIN so the ensembles start EQUALLY excitable) makes all ensembles fire, so the synaptic drive
    ADVANTAGE (not an inhibition crutch) plus the mutual-inhibition WTA select a winner per coordinate block. The
    predicted coordinate = the LABEL of the ensemble that fired MOST (a raw neural read of cp_firing_states). NO host
    `feat @ Ws`, NO argmax over host logits -- the selection is synaptic + the winner is neural."""
    xp, _ = get_backend()
    res = idx["res"]; res_exc = idx["res_exc"]; W_in = idx["W_in"]
    res_dev = xp.asarray(res)
    x_ens = idx["x_ens"]; y_ens = idx["y_ens"]
    x_dev = [xp.asarray(e) for e in x_ens]
    y_dev = [xp.asarray(e) for e in y_ens]
    fx = np.full(G, WS_ENS_FLOOR) if floors_x is None else np.asarray(floors_x, np.float64)
    fy = np.full(G, WS_ENS_FLOOR) if floors_y is None else np.asarray(floors_y, np.float64)
    _restore_state(b, snap)
    b.cp_external_input_current[:] = 0.0
    x_spk = np.zeros(G, np.float64)
    y_spk = np.zeros(G, np.float64)
    res_spk = 0.0
    steps = 0
    for _rep in range(int(replay)):                          # replay the sequence -> more spike samples resolve the margin
        for t in range(len(U)):
            drive = np.zeros(len(res)) if silence else (W_in @ np.asarray(U[t]) + RES_BIAS)
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[res_dev] = xp.asarray(drive.astype(np.float32))
            for r in range(G):
                b.cp_external_input_current[x_dev[r]] = np.float32(fx[r])
                b.cp_external_input_current[y_dev[r]] = np.float32(fy[r])
            for _ in range(int(t_step)):
                b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states), dtype=np.float64)
                res_spk += float(fs[res_exc].sum())
                for r in range(G):
                    x_spk[r] += float(fs[x_ens[r]].sum())
                    y_spk[r] += float(fs[y_ens[r]].sum())
                steps += 1
    _restore_state(b, snap)
    b.cp_external_input_current[:] = 0.0
    # NEURAL WINNER: the ensemble whose population fired most is the surviving WTA winner (its label = the coordinate).
    x_win = int(np.argmax(x_spk))
    y_win = int(np.argmax(y_spk))
    return (x_win, y_win), dict(x_spk=x_spk.tolist(), y_spk=y_spk.tolist(),
                                res_mean_spk=res_spk / max(1, steps * len(res_exc)))


# ── homeostatic per-ensemble excitability equalization (the companion process we had proxied with ONE constant) ─────
CAL_ITERS = 3
CAL_GAIN = 0.9


def _calibrate_floors(b, idx, snap, G, cal_pairs):
    """Intrinsic-excitability homeostasis on the READ-OUT neurons. Heterogeneity gives each output ensemble a fixed
    baseline-rate BIAS (few neurons/ensemble do not average it out) that swamps the sub-margin. Real neurons regulate
    baseline rate to a set-point (intrinsic plasticity); we had replaced that companion process with ONE uniform
    floor constant. Measure each ensemble's baseline firing on TRAIN and adjust its tonic floor toward the block mean
    -- an upstream excitability set-point, so the raw neural WTA then competes on INPUT drive, not intrinsic bias.
    (TRAIN-only; no held-out peek.) Returns (floors_x, floors_y)."""
    fx = np.full(G, WS_ENS_FLOOR, np.float64)
    fy = np.full(G, WS_ENS_FLOOR, np.float64)
    for _it in range(CAL_ITERS):
        xs = np.zeros(G); ys = np.zeros(G)
        for (s, a) in cal_pairs:
            _pred, dbg = _neural_predict(b, idx, snap, _encode_seq(s, a, G), G, floors_x=fx, floors_y=fy)
            xs += np.asarray(dbg["x_spk"]); ys += np.asarray(dbg["y_spk"])
        xs /= max(1, len(cal_pairs)); ys /= max(1, len(cal_pairs))
        fx = np.clip(fx - CAL_GAIN * (xs - xs.mean()), 0.0, None)
        fy = np.clip(fy - CAL_GAIN * (ys - ys.mean()), 0.0, None)
    return fx, fy


# =====================================================================================================================
# HELPERS: split, standardization-folded read-out weights, host-ridge reference
# =====================================================================================================================
def _covered_split(pairs, G, seed, heldout_frac=0.25):
    rng = np.random.default_rng(seed * 101 + 5)
    idx = np.arange(len(pairs)); rng.shuffle(idx)
    n_hold = int(round(heldout_frac * len(pairs)))
    train_set = set(idx.tolist()); hold = set()
    for i in idx.tolist():
        if len(hold) >= n_hold:
            break
        s_i, a_i = pairs[i]
        tent = train_set - {i}
        if any(pairs[j][0] == s_i for j in tent) and any(pairs[j][1] == a_i for j in tent):
            hold.add(i); train_set = tent
    train = [pairs[i] for i in range(len(pairs)) if i not in hold]
    held = [pairs[i] for i in range(len(pairs)) if i in hold]
    return train, held


def _fold_standardization(W, b, mu, sd):
    """pred_r = W[r] @ ((f-mu)/sd) + b[r] = (W[r]/sd) @ f + (b[r] - sum_i W[r,i]*mu_i/sd_i). Fold the TRAIN-only
    standardization into the raw-feature read-out so the synaptic drive is a pure map over the raw reservoir
    spike-count (the neural feature). Returns W_eff (out,n_res), b_eff (out,)."""
    W_eff = W / sd[None, :]
    b_eff = b - (W_eff @ mu)
    return W_eff, b_eff


def _host_decode(W_eff, b_eff, f, G):
    """HOST RIDGE/ARGMAX reference (the SHORTCUT being replaced) -- for the comparator only, NOT the content path."""
    pred = W_eff @ f + b_eff
    return _decode(pred, G)


# =====================================================================================================================
# THE DE-RISK
# =====================================================================================================================
def _derisk_one(seed, G=5, n_pool=250, heldout_frac=0.25, smoke=False):
    t0 = time.time()
    pairs = _all_pairs(G)
    out_dim = 2 * G

    b, rm, idx, cfg = _build_bridge(seed, G, n_pool)
    # SEED / BYTE-IDENTITY: build a second bridge at the same seed, hash the substrate thresholds.
    b2, _, _, _ = _build_bridge(seed, G, n_pool)
    def _thash(bb):
        arr = getattr(bb, "cp_neuron_firing_thresholds", None)
        if arr is None:
            return None
        return hashlib.sha1(np.asarray(to_host(arr)).astype(np.float64).tobytes()).hexdigest()
    seeded = bool(_thash(b) is not None and _thash(b) == _thash(b2))
    del b2

    # snapshot BEFORE wiring the read-out synapses (we wash to a fixed baseline every drive). The reservoir
    # recurrence must exist for the feature; wire it (+ placeholder read-out at scale 0) first, then snapshot.
    # We do the full wire with the FINAL Ws only after training, so here we wire reservoir-recurrence-only.
    union0 = {}
    ri = _reservoir_internal(rm, seed)
    if ri is not None:
        union0["reservoir_internal"] = ri
    inh0 = []
    for region in rm.regions():
        inh0.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union0, output_inhibitory_indices=inh0 or None)
    snap = _snapshot_state(b)

    # ---- extract the reservoir FEATURE for every (s,a) once (fixed reservoir -> deterministic feature) ----
    feats = {}
    spike_acc = []
    for (s, a) in pairs:
        U = _encode_seq(s, a, G)
        f = _reservoir_feature(b, idx, snap, U)
        feats[(s, a)] = f
        spike_acc.append(float(f.mean()))
    mean_spikes = float(np.mean(spike_acc))

    train, held = _covered_split(pairs, G, seed, heldout_frac)
    Xtr_raw = np.stack([feats[p] for p in train])
    mu = Xtr_raw.mean(0); sd = Xtr_raw.std(0) + 1e-6
    Xtr = np.stack([(feats[p] - mu) / sd for p in train])
    Ttr = np.stack([_target(_step(s, a, G), G) for (s, a) in train])
    tr_sp = [_step(s, a, G) for (s, a) in train]
    ho_sp = [_step(s, a, G) for (s, a) in held]

    # ---- LOCAL delta rule trains the read-out weights (three-factor: post-error x pre-activity) ----
    W, bvec = _train_delta(Xtr, Ttr, out_dim, seed)
    W_eff, b_eff = _fold_standardization(W, bvec, mu, sd)
    # host ridge/argmax reference (the shortcut) -- comparator only
    ridge_train = float(np.mean([_host_decode(W_eff, b_eff, feats[p], G) == sp for p, sp in zip(train, tr_sp)]))
    ridge_held = float(np.mean([_host_decode(W_eff, b_eff, feats[p], G) == sp for p, sp in zip(held, ho_sp)]))

    Ws_x = W_eff[:G, :]        # x-block read-out rows (one per x-coordinate class)
    Ws_y = W_eff[G:2 * G, :]   # y-block read-out rows

    # ---- auto-sweep the read-out SCALE x the canceller GAIN; select on the CALIBRATED raw-neural TRAIN joint
    # accuracy (the genuine on-substrate read; no held-out peek). Each config is wired, the per-ensemble homeostatic
    # floors are calibrated on TRAIN, then the raw neural WTA winner is scored on a TRAIN probe. ----
    scale_grid = SYN_SCALE_GRID if not smoke else (3.6, 8.0)
    gain_grid = (1.0, 1.6, 2.4)
    train_probe = train if not smoke else train[:24]
    tr_probe_sp = [_step(s, a, G) for (s, a) in train_probe]
    cal_pairs = train if not smoke else train[:24]
    best = None
    for sc in scale_grid:
        for cg in gain_grid:
            _wire(b, rm, idx, seed, Ws_x, Ws_y, sc, cm_gain=cg)
            snap_w = _snapshot_state(b)
            fx, fy = _calibrate_floors(b, idx, snap_w, G, cal_pairs)
            hit = 0
            for (s, a), sp in zip(train_probe, tr_probe_sp):
                pred, _dbg = _neural_predict(b, idx, snap_w, _encode_seq(s, a, G), G, floors_x=fx, floors_y=fy)
                hit += int(pred == sp)
            acc = hit / max(1, len(train_probe))
            if best is None or acc > best[0]:
                best = (acc, sc, cg)
    train_agree, syn_scale, cm_gain_sel = best

    # ---- re-wire at the selected (scale, gain); CALIBRATE the per-ensemble homeostatic floors on TRAIN ----
    edges = _wire(b, rm, idx, seed, Ws_x, Ws_y, syn_scale, cm_gain=cm_gain_sel)
    snap_w = _snapshot_state(b)
    floors_x, floors_y = _calibrate_floors(b, idx, snap_w, G, cal_pairs)

    # ---- SYNAPTIC read-out accuracy (TRAIN + HELD) with the RAW neural WTA winner (calibrated floors) ----
    def _syn_acc(pairset, sps, silence=False):
        hit = hx = hy = 0; res_spk = []
        for (s, a), sp in zip(pairset, sps):
            U = _encode_seq(s, a, G)
            pred, dbg = _neural_predict(b, idx, snap_w, U, G, silence=silence,
                                        floors_x=floors_x, floors_y=floors_y)
            hit += int(pred == sp); hx += int(pred[0] == sp[0]); hy += int(pred[1] == sp[1])
            res_spk.append(dbg["res_mean_spk"])
        n = max(1, len(pairset))
        return float(hit / n), float(np.mean(res_spk) if res_spk else 0.0), float(hx / n), float(hy / n)

    syn_train, _, syn_train_x, syn_train_y = _syn_acc(train, tr_sp)
    syn_held, res_mean_read, syn_held_x, syn_held_y = _syn_acc(held, ho_sp)

    # ---- (b) LESION 1: zero the read-out synapses (reservoir -> x/y ens) -> ens see only the floor -> collapse ----
    px, qx = edges["readout_x"]; py, qy = edges["readout_y"]
    b.set_pathway_weights("lesion_readout_x", px, qx, np.zeros(len(px), np.float32), add_missing=False)
    b.set_pathway_weights("lesion_readout_y", py, qy, np.zeros(len(py), np.float32), add_missing=False)
    snap_les = _snapshot_state(b)
    lesion_readout_held, _ = _syn_acc_snapshot(b, idx, snap_les, held, ho_sp, G,
                                               floors_x=floors_x, floors_y=floors_y)
    # restore
    b.set_pathway_weights("restore_readout_x", px, qx,
                          _readout_edges(idx["res_exc"], idx["x_ens"], Ws_x, syn_scale)[2], add_missing=False)
    b.set_pathway_weights("restore_readout_y", py, qy,
                          _readout_edges(idx["res_exc"], idx["y_ens"], Ws_y, syn_scale)[2], add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- (b) LESION 2: silence the reservoir input on the SYNAPTIC read-out -> no reservoir spikes -> collapse ----
    silence_held = _syn_acc(held, ho_sp, silence=True)[0]

    # ---- (matched SHAM): count-matched lesion of the OFF-DECODE decoy read-out -> decode UNCHANGED ----
    pd, qd = edges["readout_dec"]
    b.set_pathway_weights("sham_readout_dec", pd, qd, np.zeros(len(pd), np.float32), add_missing=False)
    snap_sham = _snapshot_state(b)
    sham_held, _ = _syn_acc_snapshot(b, idx, snap_sham, held, ho_sp, G,
                                     floors_x=floors_x, floors_y=floors_y)

    # ---- ATTRIBUTION (tools.lab): whose is the held-out? the read-out synapses (real lesion moves it) vs the
    # OFF-DECODE decoy (matched sham must NOT move it). required=False -- at a chance-level intact read the moves
    # are small BY CONSTRUCTION (that is the negative), but the attribution must still be recorded, not assumed. ----
    lever("readout_synapse_lesion", before=round(syn_held, 4), after=round(lesion_readout_held, 4), required=False)
    lever("reservoir_silence_lesion", before=round(syn_held, 4), after=round(silence_held, 4), required=False)
    lever("matched_sham_decoy_lesion", before=round(syn_held, 4), after=round(sham_held, 4), required=False)

    # ---- retrieval / chance baselines on held-out ----
    from collections import Counter
    tr_counter = Counter(tr_sp)
    prior_sp = tr_counter.most_common(1)[0][0] if tr_counter else (0, 0)
    prior_held = float(np.mean([prior_sp == sp for sp in ho_sp]))
    chance = 1.0 / (G * G)

    elapsed = time.time() - t0
    return dict(
        seed=int(seed), G=int(G), n_pool=int(n_pool), ens_p=int(ENS_P), heldout_n=len(held), train_n=len(train),
        chance=float(chance), chance_per_block=float(1.0 / G),
        mean_reservoir_spikes_feature=mean_spikes, res_mean_spk_read=float(res_mean_read),
        syn_scale_selected=float(syn_scale), cm_gain_selected=float(cm_gain_sel),
        train_agree_neural_vs_host=float(train_agree),
        ridge_train_acc=ridge_train, ridge_heldout_acc=ridge_held,
        syn_train_acc=syn_train, syn_heldout_acc=syn_held,
        syn_train_x=syn_train_x, syn_train_y=syn_train_y, syn_heldout_x=syn_held_x, syn_heldout_y=syn_held_y,
        lesion_readout_heldout=float(lesion_readout_held), lesion_silence_heldout=float(silence_held),
        matched_sham_heldout=float(sham_held), prior_lookup_heldout=prior_held,
        content_path_clean=_content_path_clean(), seeded=seeded, elapsed_s=round(elapsed, 1),
    )


def _content_path_clean():
    """ANTI-CHEAT: the biologized content path (`_neural_predict`) must contain NO host ridge/argmax over logits --
    the winner is a neural read of ensemble spike-counts, the drive is synaptic. Grep this source's `_neural_predict`
    body for a forbidden `@ Ws` matmul or an argmax over a `@`-product. Returns True iff clean."""
    src = Path(__file__).read_text()
    lo = src.find("def _neural_predict("); hi = src.find("\ndef ", lo + 1)
    body = src[lo:hi] if lo >= 0 else src
    # strip the docstring (it deliberately NAMES the forbidden patterns to say they are absent) -> search CODE only.
    q = body.find('"""'); q2 = body.find('"""', q + 3)
    code = (body[:q] + body[q2 + 3:]) if (q >= 0 and q2 > q) else body
    forbidden = ("@ Ws", "@ W_eff", "feat @", "Weff @", "W_eff @", "argmax(pred")
    has_neural_read = ("np.argmax(x_spk)" in code) and ("np.argmax(y_spk)" in code)
    return bool(lo >= 0) and has_neural_read and not any(f in code for f in forbidden)


def _syn_acc_snapshot(b, idx, snap, pairset, sps, G, silence=False, floors_x=None, floors_y=None):
    hit = 0; res_spk = []
    for (s, a), sp in zip(pairset, sps):
        U = _encode_seq(s, a, G)
        pred, dbg = _neural_predict(b, idx, snap, U, G, silence=silence, floors_x=floors_x, floors_y=floors_y)
        hit += int(pred == sp); res_spk.append(dbg["res_mean_spk"])
    return float(hit / max(1, len(pairset))), float(np.mean(res_spk) if res_spk else 0.0)


def _verdict(d):
    """VALIDITY preconditions (the run is well-formed + the read-out is genuinely neural) must hold for the verdict
    to be DEFINED; then GO is earned only if the spiking read-out MATCHES the ridge held-out with lesion teeth. A
    valid run that under-resolves the ridge is an honest NO-GO (a characterized negative), NOT undefined."""
    v = Verdict("fm spiking synaptic read-out matches ridge (neural WTA winner)", chance=d["chance"])
    v.disabled("STDP/Hebbian/STP/structural", "fixed reservoir + fixed read-out synapses; the delta rule set the "
               "weights at train time (off-substrate); the on-substrate homeostasis is the read-out floor calibration")
    # ---- VALIDITY: the run is well-formed and the read-out is genuinely neural (these gate the verdict) ----
    v.require("reservoir active (feature)", d["mean_reservoir_spikes_feature"], expect=lambda x: x > 0.0)
    v.require("reservoir active (read)", d["res_mean_spk_read"], expect=lambda x: x > 0.0)
    v.require("seeded (byte-identical substrate)", d["seeded"], expect=True)
    v.require("content path clean (no host ridge/argmax)", d["content_path_clean"], expect=True)
    v.require("matched sham UNCHANGED (|delta|<=0.08) — specificity control valid",
              abs(d["syn_heldout_acc"] - d["matched_sham_heldout"]), expect=lambda x: x <= 0.08)
    # ---- GO criteria (NOT verdict-gating preconditions): matches-ridge + lesion teeth ----
    go = (d["syn_heldout_acc"] >= d["ridge_heldout_acc"] - 0.10
          and d["syn_heldout_acc"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.30
          and (d["syn_heldout_acc"] - d["lesion_readout_heldout"]) >= 0.30
          and (d["syn_heldout_acc"] - d["lesion_silence_heldout"]) >= 0.30
          and abs(d["syn_heldout_acc"] - d["matched_sham_heldout"]) <= 0.08)
    dec = v.decide(go=go)
    dec["go_criteria"] = {
        "matches_ridge(>=ridge-0.10)": bool(d["syn_heldout_acc"] >= d["ridge_heldout_acc"] - 0.10),
        "beats_chance/retrieval(>=+0.30)": bool(
            d["syn_heldout_acc"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.30),
        "readout_lesion_collapses(>=0.30)": bool((d["syn_heldout_acc"] - d["lesion_readout_heldout"]) >= 0.30),
        "silence_lesion_collapses(>=0.30)": bool((d["syn_heldout_acc"] - d["lesion_silence_heldout"]) >= 0.30),
    }
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--n-pool", type=int, default=250)
    ap.add_argument("--smoke", action="store_true", help="reduced scale sweep / train probe (faster single-seed)")
    ap.add_argument("--out", type=str, default="research/findings/raw/_fm_spiking_synaptic_readout_smoke.json")
    args = ap.parse_args()

    results = []
    for seed in args.seeds:
        try:
            d = _derisk_one(seed, G=args.G, n_pool=args.n_pool, smoke=args.smoke)
            dec = _verdict(d)
            d["verdict"] = dec
            results.append(d)
            print(f"\n=== seed {seed} ===")
            for k in ("mean_reservoir_spikes_feature", "res_mean_spk_read", "syn_scale_selected",
                      "cm_gain_selected", "train_agree_neural_vs_host", "ridge_heldout_acc", "syn_heldout_acc",
                      "syn_train_x", "syn_train_y", "syn_heldout_x", "syn_heldout_y", "chance_per_block",
                      "lesion_readout_heldout", "lesion_silence_heldout", "matched_sham_heldout",
                      "prior_lookup_heldout", "chance", "content_path_clean", "seeded", "elapsed_s"):
                print(f"  {k:38s} {d[k]}")
            print(f"  VERDICT: {dec['status']}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results.append({"seed": int(seed), "error": repr(e)})

    payload = {"runner": "_fm_spiking_synaptic_readout_derisk", "results": results,
               "preconditions": (results[0].get("verdict", {}).get("preconditions") if results else None)}
    outp = _REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
