"""POPULATION-SCALED DIVISIVE NORMALIZATION for the world-model neural read-out WINNER -- the last host op (argmax)
on the content path, with a mechanism that SCALES with reservoir capacity.

THE BOTTLENECK (banked, adversarially verified; `_fm_neural_wta_readout_derisk` + the capacity sweep
`fm_readout_capacity_np{250,500,1000}.json`). The forward-model reservoir (OnBridge LSM) encodes (s,a); a delta-trained
TWO-PATHWAY (W+ excitatory / W- feedforward-inhibitory) read-out decodes s'. The host-decodable EVIDENCE CEILING
(ridge / two-pathway heldout) RISES with reservoir capacity -- n_pool 250->500->1000 gives ridge 0.84 -> 0.68/... ->
0.88-1.0 (per-seed; the mean margin improves with capacity, same lever as the breadth crux). BUT the NEURAL spiking
read-out does the OPPOSITE: the neural-WTA heldout is 0.60/0.56/0.04 (mean 0.40) at n_pool=250 and collapses to
0.04/0.32/0.16 at n_pool=1000. The FIXED lateral-inhibition WTA (`WTA_W_IE` / the `wta_ie` grid, TRAIN-swept at each
n_pool) does NOT track the capacity-improved evidence: at n_pool=1000 the ensembles' discriminative firing DROPS
(`ens_mean_spk_read` 0.0102 -> 0.0069) and the winner reads as chance, EVEN AS the underlying ridge improves.

THE HYPOTHESIS (this runner). Replace the FIXED lateral inhibition with a POPULATION-SCALED DIVISIVE NORMALIZATION
(Carandini & Heeger 2012, canonical normalization: a neuron's gain is divided by the SUMMED activity of a
normalization POOL, so the operating point is population-SIZE-invariant by construction). The normalization is NEURAL:
a real inhibitory POOL per coordinate block whose EXCITATORY DRIVE is delivered by synapses FROM the reservoir
excitatory population (so its firing scales with the reservoir's summed spiking -- the summation is done by the pool's
own membrane, exactly as a biological normalization interneuron pools its afferents) PLUS feedback from the ensembles
(the competition), and whose OUTPUT is divisive inhibition onto every ensemble in the block. As n_pool grows, the
number of reservoir->pool synapses grows with it, so the pool's drive -- and hence the divisive inhibition -- scales
with the population automatically. Then the neural read-out should TRACK the capacity-improved ceiling at n_pool=1000
instead of collapsing. This closes the last host op (argmax) on the content path with a mechanism that SCALES.

WHY NEURAL, NOT A HOST DIVIDE (the exact anti-cheat). The normalization is delivered by an inhibitory neural pool
through real I->E synapses onto the ensembles (`norm_out_{x,y}`); its excitatory drive is real E->E synapses from the
reservoir + ensembles (`norm_ff_{x,y}` / `norm_fb_{x,y}`). There is NO `np.divide` on the read-out LOGITS -- the winner
is still the ensemble that fires MOST, read from `cp_firing_states` (the accepted neural-WTA read). The content path is
the VERBATIM, imported `_neural_predict` from `_fm_neural_wta_readout_derisk` (grep-clean of map-matmul / logit-argmax,
verified via `inspect.getsource`). The pool needs NO per-step host injection -- it is driven purely synaptically during
`_run_one_simulation_step`, which is why the vetted read-out is reused UNCHANGED. We ADDITIONALLY read `cp_firing_states`
of the pool to DEMONSTRATE its summed spiking scales with n_pool (the population-scaling evidence).

TEETH (all in ONE process, on the SAME substrate/feature at each n_pool; nothing imported as a number).
  (i)   the EVIDENCE CEILING (ridge / two-pathway heldout) is MEASURED IN-RUN at each n_pool (not imported).
  (ii)  the FIXED-WTA baseline (the prior mechanism: fixed `wta_ie`, TRAIN-swept) is MEASURED IN-RUN at each n_pool --
        the apples-to-apples contrast on the identical substrate/feature.
  (iii) LOAD-BEARING population-scaling: remove ONLY the reservoir->pool feedforward drive (`norm_ff` lesion) -> the
        pool loses its population-scaled drive -> the read-out reverts toward the fixed-WTA level and (the KEY test)
        collapses again at large n_pool. Reported both as the norm_ff single-lesion AND as scaled-norm vs fixed-WTA.
  (iv)  content path VERBATIM-imported + grep-clean (winner from cp_firing_states; no map-matmul / logit-argmax).
  (v)   REAL lesions -- zero W+ read-out synapses OR silence the reservoir -> collapse.
  (vi)  MATCHED SHAM -- count-matched lesion of an OFF-DECODE decoy read-out -> read-out UNCHANGED.
  (vii) seeded BYTE-IDENTICAL substrate (cfg.seed); backend RECORDED (assert_backend).

GO bar (per seed, at n_pool=1000 -- the KEY test): preconditions [reservoir+pool active, seeded, content clean, sham
UNCHANGED |d|<=0.08, backend=numpy] AND scaled_norm_heldout - fixed_wta_heldout >= 0.20 (population-scaling
load-bearing) AND scaled_norm_heldout - max(chance,prior) >= 0.20 AND scaled_norm_heldout >= twopath_rate_heldout - 0.20
(tracks the ceiling) AND norm_ff-lesion collapses >= 0.20 AND wp-lesion + silence collapse >= 0.20 AND untrained <=
chance+0.08.  HONEST NEGATIVE WITH TEETH otherwise -- if scaled-norm does NOT let the neural read-out exploit the
improved ceiling, report the measured reason (e.g. the n_pool failure is UNDER-drive of the discriminative signal, not
saturation, so activity-proportional inhibition cannot recover it) -- a first-class deliverable that names the true
companion process. Do NOT force GO.

SMOKE (single seed, numpy, reduced grids):
  SIM_BACKEND=numpy python -u -m research.runners._fm_scaled_norm_readout_derisk --seeds 42 --n-pool 250 --smoke \
      --out research/findings/raw/_fm_scaled_norm_readout_smoke.json
KEY TEST (per-seed parallel, each n_pool):
  for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy python -u -m research.runners._fm_scaled_norm_readout_derisk \
      --seeds $s --n-pool 1000 --out research/findings/raw/_fm_scaled_norm_np1000_s$s.json & done; wait
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── reuse-by-import: the world + local read-out rule (forward_model_reservoir) ───────────────────────────────────
from research.runners._forward_model_reservoir_derisk import (   # noqa: E402
    _ACTIONS, _all_pairs, _step, _encode_seq, _target, _train_delta,
)
# ── reuse-by-import: the wash-out snapshot/restore ──────────────────────────────────────────────────────────────
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (   # noqa: E402
    _snapshot_state, _restore_state,
)
# ── reuse-by-import: the VETTED neural-WTA read-out (content path + wiring helpers + constants) ──────────────────
from research.runners._fm_neural_wta_readout_derisk import (   # noqa: E402
    _neural_predict, _calibrate_floors, _accs, _rate_twopath_acc,
    _wp_edges, _ffi_in_edges, _reservoir_internal,
    ENS_P, FFI_P, WTA_INH, WTA_W_EI, WTA_W_EE, WTA_IE_GRID, FLOOR_BASE,
    SYN_SCALE_GRID, FFI_IN_GRID, FFI_OUT_GRID,
)
# ── reuse-by-import: standardization fold + host-ridge reference + covered split + reservoir constants ───────────
from research.runners._fm_spiking_synaptic_readout_derisk import (   # noqa: E402
    _fold_standardization, _host_decode, _covered_split,
    RES_EXC_W, RES_INH_W, RES_INTERNAL_DENSITY, RES_WEIGHT_JITTER, RES_EXC_FRACTION,
    RES_IN_SCALE, RES_BIAS,
)
from sim.backend import get_backend, to_host   # noqa: E402
from tools.lab import lever, assert_backend     # noqa: E402
from tools.verdict import Verdict               # noqa: E402

# ── the POPULATION-SCALED DIVISIVE-NORMALIZATION POOL (replaces the fixed lateral inhibition) ────────────────────
NORM_N = 12                # inhibitory normalization-pool neurons per coordinate block
NORM_FF_W = 6.0            # reservoir(exc) -> norm pool, per-synapse (FIXED; population-scaling is STRUCTURAL: the
#                            NUMBER of these synapses grows with n_pool, so the pool's summed drive scales with it)
NORM_FB_W = 6.0            # ensembles -> norm pool, per-synapse (feedback: the competition drive)
NORM_OUT_BASE = 4.0        # norm pool -> ensembles (inhibitory) base weight; x norm_gain (swept on TRAIN)
NORM_GAIN_GRID = (0.0, 0.4, 0.8, 1.4)  # the divisive-inhibition strength (dimensionless), TRAIN-swept; 0.0 == the
#                            population-scaled pool OFF -> pure fixed-WTA competition (so scaled-norm can only help).


def _build_bridge(seed, G, n_pool, ens_p=ENS_P, ffi_p=FFI_P, inh=WTA_INH, norm_n=NORM_N):
    """Fork of `_fm_neural_wta_readout_derisk._build_bridge` that ADDS the divisive-normalization inhibitory pools
    (`x_norm`, `y_norm`) alongside the fixed-WTA pools (`x_wta`, `y_wta`, kept for the in-run baseline). No sim/ edit."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig, NeuronModel
    from sim.regions import BrainRegion

    n_dec = 2 * G
    regions = [
        BrainRegion(name="reservoir", n_neurons=int(n_pool), exc_fraction=RES_EXC_FRACTION,
                    internal_density=0.0, exc_weight_mean=RES_EXC_W, inh_weight_mean=RES_INH_W,
                    weight_jitter=RES_WEIGHT_JITTER, plastic_internal=False),
        BrainRegion(name="x_ens", n_neurons=int(G * ens_p), exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="x_ffi", n_neurons=int(G * ffi_p), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="x_wta", n_neurons=int(inh), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="x_norm", n_neurons=int(norm_n), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_ens", n_neurons=int(G * ens_p), exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_ffi", n_neurons=int(G * ffi_p), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_wta", n_neurons=int(inh), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_norm", n_neurons=int(norm_n), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
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
    res_inh = np.asarray(rm.inhibitory_indices("reservoir"), dtype=np.int64)
    res_exc = np.asarray(sorted(set(res_idx.tolist()) - set(res_inh.tolist())), dtype=np.int64)

    def _slices(name, per):
        allidx = np.asarray(rm.indices(name), dtype=np.int64)
        return allidx, [allidx[k * per:(k + 1) * per] for k in range(G)]

    x_ens_all, x_ens = _slices("x_ens", ens_p)
    x_ffi_all, x_ffi = _slices("x_ffi", ffi_p)
    y_ens_all, y_ens = _slices("y_ens", ens_p)
    y_ffi_all, y_ffi = _slices("y_ffi", ffi_p)
    x_wta = np.asarray(rm.indices("x_wta"), dtype=np.int64)
    y_wta = np.asarray(rm.indices("y_wta"), dtype=np.int64)
    x_norm = np.asarray(rm.indices("x_norm"), dtype=np.int64)
    y_norm = np.asarray(rm.indices("y_norm"), dtype=np.int64)
    dec_all = np.asarray(rm.indices("dec_ens"), dtype=np.int64)
    dec_ens = [dec_all[k * ens_p:(k + 1) * ens_p] for k in range(n_dec)]

    idx = dict(res=res_idx, res_exc=res_exc,
               x_ens=x_ens, x_ffi=x_ffi, x_wta=x_wta, x_norm=x_norm, x_ens_all=x_ens_all,
               y_ens=y_ens, y_ffi=y_ffi, y_wta=y_wta, y_norm=y_norm, y_ens_all=y_ens_all,
               dec_ens=dec_ens, dec_all=dec_all, ens_p=int(ens_p), ffi_p=int(ffi_p))

    rng = np.random.default_rng(int(seed) * 7919 + 3)
    in_dim = 2 * G + len(_ACTIONS)
    W_in = (rng.random((len(res_idx), in_dim)) * 2 - 1) * RES_IN_SCALE
    idx["W_in"] = W_in
    idx["in_dim"] = int(in_dim)
    return b, rm, idx, cfg


def _dense_edges(pre_idx, post_idx, weight, conn_type="E_TO_E"):
    pre = np.repeat(np.asarray(pre_idx, np.int64), len(post_idx))
    post = np.tile(np.asarray(post_idx, np.int64), len(pre_idx))
    w = np.full(pre.shape[0], float(weight), np.float32)
    return {"pre_indices": pre, "post_indices": post, "initial_weights": w, "plastic": False, "conn_type": conn_type}


def _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, syn_scale, ffi_in, ffi_out,
          wta_ie=0.0, norm_gain=0.0, norm_ff=NORM_FF_W, norm_fb=NORM_FB_W):
    """Inject reservoir recurrence + the TWO-PATHWAY read-out (W+/W- via ffi) + the ensemble self-excitation + BOTH
    competition mechanisms (whichever is enabled by weight):
      * FIXED lateral-inhibition WTA (`wta_ie` > 0): the prior mechanism (ens -> x_wta -> ALL ens, fixed weight).
      * POPULATION-SCALED divisive-normalization pool (`norm_gain` > 0): reservoir(exc) -> x_norm (feedforward,
        population-scaled) + ens -> x_norm (feedback) -> divisive inhibition x_norm -> ALL ens (weight NORM_OUT_BASE
        * norm_gain).  norm_ff=0 disables ONLY the population-scaled feedforward drive (the load-bearing lesion).
    Returns edge tuples for lesioning (wp_x/wp_y, wta_i2e_x/y, norm_ff_x/y, norm_out_x/y, wp_dec)."""
    res = idx["res_exc"]
    union = {}
    ri = _reservoir_internal(rm, seed)
    if ri is not None:
        union["reservoir_internal"] = ri

    # ---- the two-pathway read-out: W+ (excitatory) reservoir -> ensembles ; W- via feedforward-inhib interneurons --
    pxe, qxe, wxe = _wp_edges(res, idx["x_ens"], Wp_x, syn_scale)
    pye, qye, wye = _wp_edges(res, idx["y_ens"], Wp_y, syn_scale)
    union["wp_x"] = {"pre_indices": pxe, "post_indices": qxe, "initial_weights": wxe, "plastic": False,
                     "conn_type": "E_TO_E"}
    union["wp_y"] = {"pre_indices": pye, "post_indices": qye, "initial_weights": wye, "plastic": False,
                     "conn_type": "E_TO_E"}
    pxi, qxi, wxi = _ffi_in_edges(res, idx["x_ffi"], Wm_x, syn_scale, ffi_in)
    pyi, qyi, wyi = _ffi_in_edges(res, idx["y_ffi"], Wm_y, syn_scale, ffi_in)
    union["ffi_in_x"] = {"pre_indices": pxi, "post_indices": qxi, "initial_weights": wxi, "plastic": False,
                         "conn_type": "E_TO_E"}
    union["ffi_in_y"] = {"pre_indices": pyi, "post_indices": qyi, "initial_weights": wyi, "plastic": False,
                         "conn_type": "E_TO_E"}
    for tag, ffi_list, ens_list in (("x", idx["x_ffi"], idx["x_ens"]), ("y", idx["y_ffi"], idx["y_ens"])):
        pre, post = [], []
        for ffi, ens in zip(ffi_list, ens_list):
            for a in ffi:
                for e in ens:
                    pre.append(int(a)); post.append(int(e))
        union[f"ffi_out_{tag}"] = {"pre_indices": np.asarray(pre, np.int64), "post_indices": np.asarray(post, np.int64),
                                   "initial_weights": np.full(len(pre), float(ffi_out), np.float32),
                                   "plastic": False, "conn_type": "I_TO_E"}

    # ---- count-matched OFF-DECODE decoy read-out (matched sham) ----
    Wp_dec = np.concatenate([Wp_x, Wp_y], axis=0)
    pd, qd, wd = _wp_edges(res, idx["dec_ens"], Wp_dec, syn_scale)
    union["wp_dec"] = {"pre_indices": pd, "post_indices": qd, "initial_weights": wd, "plastic": False,
                       "conn_type": "E_TO_E"}

    # ---- within-ensemble self-excitation (shared by both competition mechanisms) ----
    for ens_list, tag in ((idx["x_ens"], "x"), (idx["y_ens"], "y")):
        pre_ee, post_ee = [], []
        for ens in ens_list:
            for a in ens:
                for bb in ens:
                    if a != bb:
                        pre_ee.append(int(a)); post_ee.append(int(bb))
        union[f"ens_e2e_{tag}"] = {"pre_indices": np.asarray(pre_ee, np.int64),
                                   "post_indices": np.asarray(post_ee, np.int64),
                                   "initial_weights": np.full(len(pre_ee), WTA_W_EE, np.float32),
                                   "plastic": False, "conn_type": "E_TO_E"}

    ie_edges = {}
    ff_edges = {}
    out_edges = {}
    for ens_list, wta_idx, norm_idx, tag in ((idx["x_ens"], idx["x_wta"], idx["x_norm"], "x"),
                                             (idx["y_ens"], idx["y_wta"], idx["y_norm"], "y")):
        all_ens = np.concatenate(ens_list)
        # ---- FIXED lateral-inhibition WTA (baseline mechanism): ens -> wta pool -> ALL ens (fixed wta_ie) ----
        pre_ei, post_ei = [], []
        for ens in ens_list:
            for a in ens:
                for bb in wta_idx:
                    pre_ei.append(int(a)); post_ei.append(int(bb))
        union[f"wta_e2i_{tag}"] = {"pre_indices": np.asarray(pre_ei, np.int64),
                                   "post_indices": np.asarray(post_ei, np.int64),
                                   "initial_weights": np.full(len(pre_ei), WTA_W_EI, np.float32),
                                   "plastic": False, "conn_type": "E_TO_E"}
        pre_ie = np.repeat(wta_idx, len(all_ens)); post_ie = np.tile(all_ens, len(wta_idx))
        wie = np.full(len(pre_ie), float(wta_ie), np.float32)
        union[f"wta_i2e_{tag}"] = {"pre_indices": pre_ie, "post_indices": post_ie, "initial_weights": wie,
                                   "plastic": False, "conn_type": "I_TO_E"}
        ie_edges[f"wta_i2e_{tag}"] = (pre_ie, post_ie, wie)

        # ---- POPULATION-SCALED divisive normalization pool ----
        #   feedforward (population-scaled): reservoir(exc) -> norm pool. The #synapses = |res_exc| * norm_n grows
        #   with n_pool, so the pool's summed drive scales with the reservoir population (Carandini-Heeger).
        p_ff = np.repeat(res, len(norm_idx)); q_ff = np.tile(norm_idx, len(res))
        w_ff = np.full(len(p_ff), float(norm_ff), np.float32)
        union[f"norm_ff_{tag}"] = {"pre_indices": p_ff, "post_indices": q_ff, "initial_weights": w_ff,
                                   "plastic": False, "conn_type": "E_TO_E"}
        ff_edges[f"norm_ff_{tag}"] = (p_ff, q_ff, w_ff)
        #   feedback (competition): ensembles -> norm pool
        p_fb = np.repeat(all_ens, len(norm_idx)); q_fb = np.tile(norm_idx, len(all_ens))
        union[f"norm_fb_{tag}"] = {"pre_indices": p_fb, "post_indices": q_fb,
                                   "initial_weights": np.full(len(p_fb), float(norm_fb), np.float32),
                                   "plastic": False, "conn_type": "E_TO_E"}
        #   output (divisive inhibition): norm pool -> ALL ens
        p_out = np.repeat(norm_idx, len(all_ens)); q_out = np.tile(all_ens, len(norm_idx))
        w_out = np.full(len(p_out), float(NORM_OUT_BASE) * float(norm_gain), np.float32)
        union[f"norm_out_{tag}"] = {"pre_indices": p_out, "post_indices": q_out, "initial_weights": w_out,
                                    "plastic": False, "conn_type": "I_TO_E"}
        out_edges[f"norm_out_{tag}"] = (p_out, q_out, w_out)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    return {"wp_x": (pxe, qxe, wxe), "wp_y": (pye, qye, wye), "wp_dec": (pd, qd, wd),
            "wta_i2e_x": ie_edges["wta_i2e_x"], "wta_i2e_y": ie_edges["wta_i2e_y"],
            "norm_ff_x": ff_edges["norm_ff_x"], "norm_ff_y": ff_edges["norm_ff_y"],
            "norm_out_x": out_edges["norm_out_x"], "norm_out_y": out_edges["norm_out_y"]}


def _norm_pool_activity(b, idx, snap, pairs, G, floors_x=None, floors_y=None,
                        t_step=None, replay=None):
    """ANTI-CHEAT EVIDENCE (not the content path): drive the reservoir like `_neural_predict` and read the
    normalization POOL's summed spiking from `cp_firing_states`, to DEMONSTRATE the pool's drive/activity scales with
    the reservoir population. Returns mean pool spikes/neuron/step over the given pairs."""
    from research.runners._fm_neural_wta_readout_derisk import READ_T_STEP, READ_REPLAY
    t_step = READ_T_STEP if t_step is None else t_step
    replay = READ_REPLAY if replay is None else replay
    xp, _ = get_backend()
    res = idx["res"]; W_in = idx["W_in"]; res_dev = xp.asarray(res)
    x_dev = [xp.asarray(e) for e in idx["x_ens"]]; y_dev = [xp.asarray(e) for e in idx["y_ens"]]
    norm_all = np.concatenate([idx["x_norm"], idx["y_norm"]])
    fx = np.full(G, FLOOR_BASE) if floors_x is None else np.asarray(floors_x, np.float64)
    fy = np.full(G, FLOOR_BASE) if floors_y is None else np.asarray(floors_y, np.float64)
    tot = 0.0; steps = 0
    for (s, a) in pairs:
        U = _encode_seq(s, a, G)
        _restore_state(b, snap)
        b.cp_external_input_current[:] = 0.0
        for _rep in range(int(replay)):
            for t in range(len(U)):
                drive = W_in @ np.asarray(U[t]) + RES_BIAS
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[res_dev] = xp.asarray(drive.astype(np.float32))
                for r in range(G):
                    b.cp_external_input_current[x_dev[r]] = np.float32(fx[r])
                    b.cp_external_input_current[y_dev[r]] = np.float32(fy[r])
                for _ in range(int(t_step)):
                    b._run_one_simulation_step()
                    fs = np.asarray(to_host(b.cp_firing_states), dtype=np.float64)
                    tot += float(fs[norm_all].sum()); steps += 1
    _restore_state(b, snap)
    b.cp_external_input_current[:] = 0.0
    return float(tot / max(1, steps * len(norm_all)))


def _content_path_clean():
    """The winner is read from the VETTED, IMPORTED `_neural_predict` (from `_fm_neural_wta_readout_derisk`). Verify
    via inspect.getsource that its code (docstring stripped) reads the winner from `cp_firing_states` (argmax over
    ensemble SPIKE-COUNTS) with NO host map-matmul / logit-argmax -- and that THIS runner's content path IS that
    imported function (not a fork)."""
    code = inspect.getsource(_neural_predict)
    q = code.find('"""'); q2 = code.find('"""', q + 3)
    body = (code[:q] + code[q2 + 3:]) if (q >= 0 and q2 > q) else code
    forbidden = ("@ Ws", "@ W_eff", "feat @", "W_eff @", "Wp_x @", "@ f", "argmax(dx", "argmax(pred")
    reads_spikes = ("np.argmax(x_spk)" in body) and ("np.argmax(y_spk)" in body) and ("cp_firing_states" in body)
    imported_ok = (_neural_predict.__module__ == "research.runners._fm_neural_wta_readout_derisk")
    return bool(imported_ok and reads_spikes and not any(f in body for f in forbidden))


CAL_ITERS_SMOKE = 2


def _sweep_and_score(b, rm, idx, seed, snap_feat, Wp_x, Wm_x, Wp_y, Wm_y, G,
                     train, held, tr_sp, ho_sp, syn_scale, ffi_in, ffi_out,
                     mode, grid, cal_pairs, train_probe, tr_probe_sp, base_wta=0.0):
    """Wire `mode` ('fixed'|'norm') across `grid`, calibrate floors on TRAIN, score the neural WTA on a TRAIN probe;
    select the best, re-wire, calibrate on TRAIN, and score TRAIN+HELD. Returns (best_param, edges, snap_w, floors,
    train_acc, held_acc, ens_mean_read).

    mode='fixed': sweep the fixed lateral-inhibition strength `wta_ie` (the PRIOR mechanism), norm pool OFF.
    mode='norm' : KEEP the fixed lateral-inhibition competition at `base_wta` (the working winner-selector) and ADD the
                  POPULATION-SCALED normalization pool, sweeping its `norm_gain` (0.0 == pool OFF == pure fixed WTA, so
                  the ADDED normalization can only help on TRAIN). The population-scaling is the causal lever tested."""
    def _do_wire(g):
        if mode == "fixed":
            return _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, syn_scale, ffi_in, ffi_out,
                         wta_ie=g, norm_gain=0.0)
        return _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, syn_scale, ffi_in, ffi_out,
                     wta_ie=base_wta, norm_gain=g)
    best = None
    for g in grid:
        _do_wire(g)
        snap_w = _snapshot_state(b)
        fx, fy = _calibrate_floors(b, idx, snap_w, G, cal_pairs)
        hit = 0
        for (s, a), sp in zip(train_probe, tr_probe_sp):
            pred, _dbg = _neural_predict(b, idx, snap_w, _encode_seq(s, a, G), G, floors_x=fx, floors_y=fy)
            hit += int(pred == sp)
        acc = hit / max(1, len(train_probe))
        if best is None or acc > best[0]:
            best = (acc, g)
    best_param = best[1]
    edges = _do_wire(best_param)
    snap_w = _snapshot_state(b)
    floors_x, floors_y = _calibrate_floors(b, idx, snap_w, G, cal_pairs)
    tr_acc, _, _, _, _ = _accs(b, idx, snap_w, train, tr_sp, G, floors_x=floors_x, floors_y=floors_y)
    ho_acc, _, _, _, ens_mean = _accs(b, idx, snap_w, held, ho_sp, G, floors_x=floors_x, floors_y=floors_y)
    return best_param, edges, snap_w, (floors_x, floors_y), tr_acc, ho_acc, ens_mean


def _derisk_one(seed, G=5, n_pool=250, heldout_frac=0.25, smoke=False):
    t0 = time.time()
    backend = assert_backend("numpy", note=f"scaled-norm read-out seed={seed} n_pool={n_pool}")
    pairs = _all_pairs(G)
    out_dim = 2 * G

    b, rm, idx, cfg = _build_bridge(seed, G, n_pool)
    b2, _, _, _ = _build_bridge(seed, G, n_pool)

    def _thash(bb):
        arr = getattr(bb, "cp_neuron_firing_thresholds", None)
        return None if arr is None else hashlib.sha1(np.asarray(to_host(arr)).astype(np.float64).tobytes()).hexdigest()
    seeded = bool(_thash(b) is not None and _thash(b) == _thash(b2))
    del b2

    # reservoir-recurrence-only wiring for the FEATURE extraction (the read-out synapses come after training)
    union0 = {}
    ri = _reservoir_internal(rm, seed)
    if ri is not None:
        union0["reservoir_internal"] = ri
    inh0 = []
    for region in rm.regions():
        inh0.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union0, output_inhibitory_indices=inh0 or None)
    snap = _snapshot_state(b)

    # ---- reservoir FEATURE per (s,a) (reuse the neural-WTA feature: the read-out samples the excitatory pop) ----
    from research.runners._fm_spiking_synaptic_readout_derisk import _reservoir_feature
    feats = {}
    spike_acc = []
    for (s, a) in pairs:
        f = _reservoir_feature(b, idx, snap, _encode_seq(s, a, G))
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

    W, bvec = _train_delta(Xtr, Ttr, out_dim, seed)
    W_eff, b_eff = _fold_standardization(W, bvec, mu, sd)
    ridge_train = float(np.mean([_host_decode(W_eff, b_eff, feats[p], G) == sp for p, sp in zip(train, tr_sp)]))
    ridge_held = float(np.mean([_host_decode(W_eff, b_eff, feats[p], G) == sp for p, sp in zip(held, ho_sp)]))

    Wp = np.clip(W_eff, 0.0, None); Wm = np.clip(-W_eff, 0.0, None)
    Wp_x, Wm_x = Wp[:G, :], Wm[:G, :]
    Wp_y, Wm_y = Wp[G:2 * G, :], Wm[G:2 * G, :]

    twopath_rate_train = _rate_twopath_acc(Wp_x, Wm_x, Wp_y, Wm_y, b_eff, feats, train, tr_sp, G)
    twopath_rate_held = _rate_twopath_acc(Wp_x, Wm_x, Wp_y, Wm_y, b_eff, feats, held, ho_sp, G)

    # ---- the two-pathway gains: fix at the baseline-selected values (not what we test); a small ffi_in grid ----
    syn_scale = SYN_SCALE_GRID[-1]        # 6.0
    ffi_out = FFI_OUT_GRID[0]             # 4.0
    ffi_in = FFI_IN_GRID[0] if smoke else (FFI_IN_GRID[-1] if n_pool >= 800 else FFI_IN_GRID[0])

    n_sweep = 16 if not smoke else 12
    train_probe = train[:n_sweep]
    tr_probe_sp = [_step(s, a, G) for (s, a) in train_probe]
    cal_pairs = train[:n_sweep]

    fixed_grid = WTA_IE_GRID if not smoke else (40.0,)
    norm_grid = NORM_GAIN_GRID if not smoke else (1.0,)

    # ============ BASELINE: the FIXED lateral-inhibition WTA (prior mechanism), MEASURED IN-RUN ============
    (wta_ie_sel, edges_fx, snap_fx, floors_fx, fixed_train, fixed_held,
     fixed_ens_mean) = _sweep_and_score(b, rm, idx, seed, snap, Wp_x, Wm_x, Wp_y, Wm_y, G,
                                        train, held, tr_sp, ho_sp, syn_scale, ffi_in, ffi_out,
                                        "fixed", fixed_grid, cal_pairs, train_probe, tr_probe_sp)

    # ============ THE MECHANISM: FIXED WTA competition + POPULATION-SCALED DIVISIVE NORMALIZATION, IN-RUN ============
    (norm_gain_sel, edges, snap_w, (floors_x, floors_y), scaled_train, scaled_held,
     scaled_ens_mean) = _sweep_and_score(b, rm, idx, seed, snap, Wp_x, Wm_x, Wp_y, Wm_y, G,
                                         train, held, tr_sp, ho_sp, syn_scale, ffi_in, ffi_out,
                                         "norm", norm_grid, cal_pairs, train_probe, tr_probe_sp,
                                         base_wta=wta_ie_sel)

    # normalization-pool activity (ANTI-CHEAT: the pool is neural + its drive scales with the reservoir population)
    norm_pool_spk = _norm_pool_activity(b, idx, snap_w, held, G, floors_x=floors_x, floors_y=floors_y)

    # ---- LOAD-BEARING population-scaling: remove ONLY the reservoir->pool feedforward drive (norm_ff -> 0) ----
    pffx, qffx, wffx = edges["norm_ff_x"]; pffy, qffy, wffy = edges["norm_ff_y"]
    b.set_pathway_weights("les_ff_x", pffx, qffx, np.zeros(len(pffx), np.float32), add_missing=False)
    b.set_pathway_weights("les_ff_y", pffy, qffy, np.zeros(len(pffy), np.float32), add_missing=False)
    snap_noff = _snapshot_state(b)
    fx0, fy0 = _calibrate_floors(b, idx, snap_noff, G, cal_pairs)
    noff_held, _, _, _, _ = _accs(b, idx, snap_noff, held, ho_sp, G, floors_x=fx0, floors_y=fy0)
    b.set_pathway_weights("res_ff_x", pffx, qffx, wffx, add_missing=False)
    b.set_pathway_weights("res_ff_y", pffy, qffy, wffy, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- LESION 1: zero the W+ read-out synapses -> collapse ----
    pxe, qxe, wxe = edges["wp_x"]; pye, qye, wye = edges["wp_y"]
    b.set_pathway_weights("les_wp_x", pxe, qxe, np.zeros(len(pxe), np.float32), add_missing=False)
    b.set_pathway_weights("les_wp_y", pye, qye, np.zeros(len(pye), np.float32), add_missing=False)
    snap_lw = _snapshot_state(b)
    lesion_wp_held, _, _, _, _ = _accs(b, idx, snap_lw, held, ho_sp, G, floors_x=floors_x, floors_y=floors_y)
    b.set_pathway_weights("res_wp_x", pxe, qxe, wxe, add_missing=False)
    b.set_pathway_weights("res_wp_y", pye, qye, wye, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- LESION 2: silence the reservoir input -> collapse ----
    silence_held, _, _, _, _ = _accs(b, idx, snap_w, held, ho_sp, G, silence=True,
                                     floors_x=floors_x, floors_y=floors_y)

    # ---- MATCHED SHAM: count-matched lesion of the OFF-DECODE decoy read-out -> UNCHANGED ----
    pd, qd, wd = edges["wp_dec"]
    b.set_pathway_weights("sham_dec", pd, qd, np.zeros(len(pd), np.float32), add_missing=False)
    snap_sham = _snapshot_state(b)
    sham_held, _, _, _, _ = _accs(b, idx, snap_sham, held, ho_sp, G, floors_x=floors_x, floors_y=floors_y)
    b.set_pathway_weights("res_dec", pd, qd, wd, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- UNTRAINED control: random non-negative weights of matched magnitude -> chance ----
    rng = np.random.default_rng(seed * 4242 + 1)
    Wp_x_r = rng.random(Wp_x.shape) * float(Wp.mean()); Wm_x_r = rng.random(Wm_x.shape) * float(Wm.mean())
    Wp_y_r = rng.random(Wp_y.shape) * float(Wp.mean()); Wm_y_r = rng.random(Wm_y.shape) * float(Wm.mean())
    _wire(b, rm, idx, seed, Wp_x_r, Wm_x_r, Wp_y_r, Wm_y_r, syn_scale, ffi_in, ffi_out,
          wta_ie=wta_ie_sel, norm_gain=norm_gain_sel)
    snap_ut = _snapshot_state(b)
    fxr, fyr = _calibrate_floors(b, idx, snap_ut, G, cal_pairs)
    untrained_held, _, _, _, _ = _accs(b, idx, snap_ut, held, ho_sp, G, floors_x=fxr, floors_y=fyr)

    lever("scaled_norm_vs_fixed_wta", before=round(fixed_held, 4), after=round(scaled_held, 4), required=False)
    lever("scaled_norm_vs_noff(population_scaling)", before=round(noff_held, 4), after=round(scaled_held, 4),
          required=False)
    lever("wp_readout_lesion", before=round(scaled_held, 4), after=round(lesion_wp_held, 4), required=False)
    lever("reservoir_silence_lesion", before=round(scaled_held, 4), after=round(silence_held, 4), required=False)
    lever("matched_sham_decoy", before=round(scaled_held, 4), after=round(sham_held, 4), required=False)

    from collections import Counter
    tr_counter = Counter(tr_sp)
    prior_sp = tr_counter.most_common(1)[0][0] if tr_counter else (0, 0)
    prior_held = float(np.mean([prior_sp == sp for sp in ho_sp]))
    chance = 1.0 / (G * G)

    elapsed = time.time() - t0
    return dict(
        seed=int(seed), G=int(G), n_pool=int(n_pool), ens_p=int(ENS_P), ffi_p=int(FFI_P), norm_n=int(NORM_N),
        backend=backend, heldout_n=len(held), train_n=len(train), chance=float(chance), chance_per_block=float(1.0 / G),
        mean_reservoir_spikes_feature=mean_spikes,
        syn_scale=float(syn_scale), ffi_in=float(ffi_in), ffi_out=float(ffi_out),
        wta_ie_selected=float(wta_ie_sel), norm_gain_selected=float(norm_gain_sel),
        norm_ff_w=float(NORM_FF_W), norm_fb_w=float(NORM_FB_W), norm_out_base=float(NORM_OUT_BASE),
        norm_pool_mean_spk=float(norm_pool_spk),
        ridge_train_acc=ridge_train, ridge_heldout_acc=ridge_held,
        twopath_rate_train=twopath_rate_train, twopath_rate_heldout=twopath_rate_held,
        # the two mechanisms, MEASURED IN-RUN on the SAME substrate/feature
        fixed_wta_train=float(fixed_train), fixed_wta_heldout=float(fixed_held), fixed_wta_ens_mean=float(fixed_ens_mean),
        scaled_norm_train=float(scaled_train), scaled_norm_heldout=float(scaled_held),
        scaled_norm_ens_mean=float(scaled_ens_mean),
        # LOAD-BEARING: population-scaling removed (norm_ff lesion)
        scaled_norm_noff_heldout=float(noff_held),
        lesion_wp_heldout=float(lesion_wp_held), lesion_silence_heldout=float(silence_held),
        matched_sham_heldout=float(sham_held), untrained_control_heldout=float(untrained_held),
        prior_lookup_heldout=prior_held,
        content_path_clean=_content_path_clean(), seeded=seeded, elapsed_s=round(elapsed, 1),
    )


def _verdict(d):
    v = Verdict("fm population-scaled divisive-normalization world-model read-out (Carandini-Heeger)",
                chance=d["chance"])
    v.disabled("STDP/Hebbian/STP/structural", "fixed reservoir + delta-trained two-pathway synapses + a NEURAL "
               "population-scaled divisive-normalization inhibitory pool (reservoir->pool synaptic drive); "
               "on-substrate homeostasis = the read-out floor calibration")
    v.require("backend == numpy", d["backend"] == "numpy", expect=True)
    v.require("two-pathway rate == ridge (decomposition exact)",
              abs(d["twopath_rate_heldout"] - d["ridge_heldout_acc"]), expect=lambda x: x <= 1e-6)
    v.require("reservoir active (feature)", d["mean_reservoir_spikes_feature"], expect=lambda x: x > 0.0)
    v.require("normalization pool active (spikes>0)", d["norm_pool_mean_spk"], expect=lambda x: x > 0.0)
    v.require("seeded (byte-identical substrate)", d["seeded"], expect=True)
    v.require("content path clean (imported neural read; no host matmul/logit-argmax)",
              d["content_path_clean"], expect=True)
    v.require("matched sham UNCHANGED (|d|<=0.08)", abs(d["scaled_norm_heldout"] - d["matched_sham_heldout"]),
              expect=lambda x: x <= 0.08)
    go = (d["scaled_norm_heldout"] - d["fixed_wta_heldout"] >= 0.20               # scaled beats fixed (the claim)
          and d["scaled_norm_heldout"] - d["scaled_norm_noff_heldout"] >= 0.20    # population-scaling load-bearing
          and d["scaled_norm_heldout"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.20
          and d["scaled_norm_heldout"] >= d["twopath_rate_heldout"] - 0.20        # tracks the ceiling
          and (d["scaled_norm_heldout"] - d["lesion_wp_heldout"]) >= 0.20
          and (d["scaled_norm_heldout"] - d["lesion_silence_heldout"]) >= 0.20
          and d["untrained_control_heldout"] <= d["chance"] + 0.08)
    dec = v.decide(go=go)
    dec["go_criteria"] = {
        "scaled_beats_fixed(>=+0.20)": bool(d["scaled_norm_heldout"] - d["fixed_wta_heldout"] >= 0.20),
        "population_scaling_load_bearing(scaled-noff>=0.20)":
            bool(d["scaled_norm_heldout"] - d["scaled_norm_noff_heldout"] >= 0.20),
        "beats_chance/prior(>=+0.20)":
            bool(d["scaled_norm_heldout"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.20),
        "tracks_ceiling(>=twopath-0.20)": bool(d["scaled_norm_heldout"] >= d["twopath_rate_heldout"] - 0.20),
        "wp_lesion_collapses(>=0.20)": bool((d["scaled_norm_heldout"] - d["lesion_wp_heldout"]) >= 0.20),
        "silence_collapses(>=0.20)": bool((d["scaled_norm_heldout"] - d["lesion_silence_heldout"]) >= 0.20),
        "untrained<=chance+0.08": bool(d["untrained_control_heldout"] <= d["chance"] + 0.08),
    }
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--n-pool", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default="research/findings/raw/_fm_scaled_norm_readout_smoke.json")
    args = ap.parse_args()

    results = []
    for seed in args.seeds:
        try:
            d = _derisk_one(seed, G=args.G, n_pool=args.n_pool, smoke=args.smoke)
            dec = _verdict(d)
            d["verdict"] = dec
            results.append(d)
            print(f"\n=== seed {seed} (n_pool={args.n_pool}) ===")
            for k in ("mean_reservoir_spikes_feature", "norm_pool_mean_spk", "wta_ie_selected", "norm_gain_selected",
                      "ridge_heldout_acc", "twopath_rate_heldout", "fixed_wta_heldout", "scaled_norm_heldout",
                      "scaled_norm_noff_heldout", "fixed_wta_ens_mean", "scaled_norm_ens_mean",
                      "lesion_wp_heldout", "lesion_silence_heldout", "matched_sham_heldout",
                      "untrained_control_heldout", "prior_lookup_heldout", "chance",
                      "backend", "content_path_clean", "seeded", "elapsed_s"):
                print(f"  {k:34s} {d[k]}")
            print(f"  VERDICT: {dec['status']}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results.append({"seed": int(seed), "error": repr(e)})

    payload = {"runner": "_fm_scaled_norm_readout_derisk", "argv": sys.argv, "seeds": list(args.seeds),
               "n_pool": args.n_pool, "results": results,
               "preconditions": (results[0].get("verdict", {}).get("preconditions") if results else None)}
    outp = _REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
