"""BURN DOWN the LAST host op in the world-model read-out: the WINNER. The held two-pathway spiking read-out
(branch worktree-wf_122d7930-bc4-2, `_fm_learned_twopath_readout_derisk`) reaches held-out ~0.447 (11x chance) with a
delta-trained TWO-PATHWAY (W+ excitatory / W- feedforward-inhibitory) Dale realization -- but the WINNER was still a
host `np.argmax(x_spk)` over near-tied ensemble spike-counts, and the argmax was DOING the resolving because the
ensemble lateral inhibition was too WEAK (`WTA_W_IE=10`). This runner makes the WINNER genuinely NEURAL: it is the
LATERAL-INHIBITION BIASED-COMPETITION WTA itself. Strengthening the shared inhibitory pool (the divisive-normalization
/ gamma competition, Wang 2002; Carandini-Heeger) AMPLIFIES the small (~8%) feed-forward margin -- the correct output
ensemble suppresses its rivals -- so the ensemble spike-counts come to reflect the answer well enough to BEAT CHANCE
(held ~0.45-0.48, 11-12x chance). Remove the lateral inhibition and the read-out collapses to CHANCE: the argmax alone
resolves nothing (the WINNER is neural + load-bearing). HONEST CEILING NO-GO: the neural WTA does NOT reach the rate
ceiling (~0.73-0.84); it plateaus at the 0.447 baseline level -- the residual is the spike-count EVIDENCE, not the
winner op (see below). This is the accepted project neural-WTA pattern (`2026-07-04-rungB1b-neural-role-wta-GO`,
`2026-07-13-PAST-RESERVOIR-ONBRIDGE...-6seed-GO`): the winner is read from `cp_firing_states` (which competed ensemble
fires most -- a motor-style read), and the RESOLVING is neural, proven load-bearing by the I->E lesion.

WHY THE WINNER, NOT THE EVIDENCE, AND WHY LATERAL INHIBITION (the wall reframe: "what runs alongside this constant?").
More integration does NOT lift the count read (measured: held-out flat ~0.44-0.56 from 2 to 16 replays; the top1-top2
ensemble margin is a STRUCTURAL ~8%, not Poisson-limited). A downstream stochastic decision WTA fed by the WEAK
ensembles only injects its own firing noise and DEGRADES the winner (measured 0.12-0.31 < host argmax). The companion
process the baseline had proxied with a weak constant is the STRENGTH of the ensemble's own recurrent lateral
inhibition -- the biased-competition gain. Restoring it (sweeping the lateral-inhibition strength on TRAIN) is what
makes the winner NEURAL and load-bearing: ie=0 -> chance (~0.16), ie~40-70 -> held ~0.44 (11x chance). HONEST CEILING
RESULT: on this substrate the biased competition amplifies the ~8% margin enough to BEAT CHANCE but NOT to reach the
rate ceiling (~0.73-0.84) -- the residual is the structural rate->spike-count representational margin, which neither
more integration (measured flat 2->16 replays) nor stronger inhibition (>=110 over-suppresses) closes. So the WINNER
is genuinely neural and load-bearing, but it plateaus at the 0.447 baseline level rather than rising to the ceiling;
the bottleneck is the spike-count EVIDENCE representation, not the winner op.

REUSES (by import; NO `sim/` edit -- all wiring runner-side via `inject_explicit_wiring`/`set_pathway_weights`):
`_forward_model_reservoir_derisk` (world/encode/delta/target), `_fm_spiking_synaptic_readout_derisk`
(`_reservoir_feature`/`_fold_standardization`/`_host_decode`/`_covered_split` + reservoir constants), `_emerge61...`
(snapshot/restore). The two-pathway ensemble read is forked here (the baseline is held off-main) so the SAME-process
host-argmax fallback (lateral inhibition removed) is an apples-to-apples contrast on the SAME feature/map/substrate.

TEETH.
  (i)   NEURAL winner off `cp_firing_states` (which competed ensemble fires most); content path grep-clean of the map
        matmul / argmax-over-logits.
  (ii)  LOAD-BEARING WINNER: removing the lateral inhibition (zero the `x_wta`->`x_ens` / `y_wta`->`y_ens` I->E
        synapses) -> the host-argmax fallback over the un-competed counts collapses to ~chance. The neural competition,
        not the argmax, resolves the winner. (The task's exact teeth.)
  (iii) held-out RISES vs the 0.447 weak-WTA baseline toward the rate ceiling 0.68-0.73 (strong biased competition).
  (iv)  REAL LESION -- zero W+ read-out synapses OR silence reservoir -> collapse.
  (v)   MATCHED SHAM -- count-matched lesion of an OFF-DECODE decoy read-out -> UNCHANGED.
  (vi)  UNTRAINED control -- random non-negative weights of matched magnitude -> chance (the MAP carries it).
  (vii) seeded byte-identical substrate (cfg.seed).
HONEST RESIDUAL: a fully argmax-FREE one-hot spiking latch (sole-surviving ensemble) does NOT resolve at the ~0.01
mean firing rate (measured: sole-survivor rate ~0 even at strong inhibition), so the final index-identification remains
a motor-style argmax over the competed population -- named, and consistent with the two GO precedents above.

GO bar (per seed): reservoir+ensembles active AND content path clean AND seeded AND matched-sham UNCHANGED (|d|<=0.08)
[preconditions] ; GO iff neural_wta_heldout - wta_lesion_heldout >= 0.20 (competition load-bearing) AND
neural_wta_heldout - max(chance,prior) >= 0.20 AND neural_wta_heldout >= twopath_rate_heldout - 0.15 (approaches the
ceiling) AND W+-lesion collapses >= 0.20 AND reservoir-silence collapses >= 0.20 AND untrained <= chance+0.08.
BOUNDARY otherwise -- name the residual; do NOT force GO.

SMOKE (single seed, numpy):
  SIM_BACKEND=numpy python -u -m research.runners._fm_neural_wta_readout_derisk --seeds 42 --smoke \
      --out research/findings/raw/_fm_neural_wta_readout_smoke.json
6-SEED (per-seed parallel then aggregate):
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -u -m research.runners._fm_neural_wta_readout_derisk \
      --seeds $s --out research/findings/raw/_fm_neural_wta_readout_s$s.json & done; wait
  SIM_BACKEND=numpy python -u -m research.runners._aggregate_fm_neural_wta_seeds
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

from research.runners._forward_model_reservoir_derisk import (   # noqa: E402
    _ACTIONS, _all_pairs, _step, _encode_seq, _target, _train_delta,
)
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (   # noqa: E402
    _snapshot_state, _restore_state,
)
from research.runners._fm_spiking_synaptic_readout_derisk import (   # noqa: E402
    _reservoir_feature, _fold_standardization, _host_decode, _covered_split,
    RES_EXC_W, RES_INH_W, RES_INTERNAL_DENSITY, RES_WEIGHT_JITTER, RES_EXC_FRACTION,
    RES_IN_SCALE, RES_BIAS,
)
from sim.backend import get_backend, to_host   # noqa: E402
from tools.lab import lever   # noqa: E402
from tools.verdict import Verdict   # noqa: E402

# ── the two-pathway spiking read-out (forked from the held baseline; ensembles + lateral-WTA winner) ─────────────
ENS_P = 40                 # neurons per output ensemble (WIDENED 32->40)
FFI_P = 8                  # feedforward-inhibitory interneurons per ensemble (carry the LEARNED W- weight vector)
WTA_INH = 12               # shared lateral-competition inhibitory pool per coordinate block (the WINNER mechanism)
READ_T_STEP = 48           # sim steps per token for the read-out window
READ_REPLAY = 2            # re-present the sequence N times, accumulating spikes
FLOOR_BASE = 40.0          # base tonic (pA) per ensemble
WTA_W_EI = 6.0             # ens -> shared WTA inh
WTA_W_EE = 2.0             # within-ensemble recurrent excitation (self-sustain)
# swept on TRAIN (no held peek): the two-pathway read gains + the LATERAL-INHIBITION STRENGTH (the neural winner gain).
SYN_SCALE_GRID = (4.0, 6.0)
FFI_IN_GRID = (0.8, 1.4)
FFI_OUT_GRID = (4.0,)                 # ffi_out=8 over-inhibits the ensembles into silence (measured); keep 4
WTA_IE_GRID = (40.0, 70.0, 110.0)    # shared-WTA-inh -> ens: the biased-competition gain (sweet spot ~40-70;
#                                      too strong (>=110) over-suppresses the ensembles into a noisy sparse winner)
DOM_THRESH = 0.60          # a competed winner "dominates" if its share of the block spike-count exceeds this


def _build_bridge(seed, G, n_pool, ens_p=ENS_P, ffi_p=FFI_P, inh=WTA_INH):
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
        BrainRegion(name="y_ens", n_neurons=int(G * ens_p), exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_ffi", n_neurons=int(G * ffi_p), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="y_wta", n_neurons=int(inh), exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
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
    dec_all = np.asarray(rm.indices("dec_ens"), dtype=np.int64)
    dec_ens = [dec_all[k * ens_p:(k + 1) * ens_p] for k in range(n_dec)]

    idx = dict(res=res_idx, res_exc=res_exc,
               x_ens=x_ens, x_ffi=x_ffi, x_wta=x_wta, x_ens_all=x_ens_all,
               y_ens=y_ens, y_ffi=y_ffi, y_wta=y_wta, y_ens_all=y_ens_all,
               dec_ens=dec_ens, dec_all=dec_all, ens_p=int(ens_p), ffi_p=int(ffi_p))

    rng = np.random.default_rng(int(seed) * 7919 + 3)
    in_dim = 2 * G + len(_ACTIONS)
    W_in = (rng.random((len(res_idx), in_dim)) * 2 - 1) * RES_IN_SCALE
    idx["W_in"] = W_in
    idx["in_dim"] = int(in_dim)
    return b, rm, idx, cfg


def _reservoir_internal(rm, seed):
    res_region = next(r for r in rm.regions() if r.name == "reservoir")
    shadow = _dc.replace(res_region, internal_density=RES_INTERNAL_DENSITY)
    return rm._build_region_internal(shadow, _random.Random(int(seed) * 100003 + 7))


def _wp_edges(res_idx, ens_list, Wp_block, syn_scale):
    pre, post, w = [], [], []
    for r, ens in enumerate(ens_list):
        wr = Wp_block[r] * float(syn_scale)
        for j, i in enumerate(res_idx):
            wv = float(wr[j])
            for e in ens:
                pre.append(int(i)); post.append(int(e)); w.append(wv)
    return (np.asarray(pre, np.int64), np.asarray(post, np.int64), np.asarray(w, np.float32))


def _ffi_in_edges(res_idx, ffi_list, Wm_block, syn_scale, ffi_in):
    pre, post, w = [], [], []
    for r, ffi in enumerate(ffi_list):
        wr = Wm_block[r] * float(syn_scale) * float(ffi_in)
        for j, i in enumerate(res_idx):
            wv = float(wr[j])
            for e in ffi:
                pre.append(int(i)); post.append(int(e)); w.append(wv)
    return (np.asarray(pre, np.int64), np.asarray(post, np.int64), np.asarray(w, np.float32))


def _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, syn_scale, ffi_in, ffi_out, wta_ie=WTA_IE_GRID[1]):
    """Inject reservoir recurrence + the TWO-PATHWAY read-out + the ensemble LATERAL-INHIBITION WTA (strength `wta_ie`
    -- the neural winner mechanism) + a count-matched OFF-DECODE decoy. Returns edge tuples for lesioning, INCLUDING
    the lateral-inhibition I->E synapses (`wta_i2e_x`/`wta_i2e_y`) whose lesion is the load-bearing test."""
    res = idx["res_exc"]
    union = {}
    ri = _reservoir_internal(rm, seed)
    if ri is not None:
        union["reservoir_internal"] = ri

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

    Wp_dec = np.concatenate([Wp_x, Wp_y], axis=0)
    pd, qd, wd = _wp_edges(res, idx["dec_ens"], Wp_dec, syn_scale)
    union["wp_dec"] = {"pre_indices": pd, "post_indices": qd, "initial_weights": wd, "plastic": False,
                       "conn_type": "E_TO_E"}

    # ---- the ensemble LATERAL-INHIBITION WTA (the WINNER): ens -> shared inh pool -> ALL ens (biased competition) ----
    ie_edges = {}
    for ens_list, wta_idx, tag in ((idx["x_ens"], idx["x_wta"], "x"), (idx["y_ens"], idx["y_wta"], "y")):
        pre_ei, post_ei = [], []
        for ens in ens_list:
            for a in ens:
                for bb in wta_idx:
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
        for a in wta_idx:
            for bb in all_ens:
                pre_ie.append(int(a)); post_ie.append(int(bb))
        pre_ie = np.asarray(pre_ie, np.int64); post_ie = np.asarray(post_ie, np.int64)
        wie = np.full(len(pre_ie), float(wta_ie), np.float32)
        union[f"wta_i2e_{tag}"] = {"pre_indices": pre_ie, "post_indices": post_ie, "initial_weights": wie,
                                   "plastic": False, "conn_type": "I_TO_E"}
        ie_edges[f"wta_i2e_{tag}"] = (pre_ie, post_ie, wie)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    return {"wp_x": (pxe, qxe, wxe), "wp_y": (pye, qye, wye), "wp_dec": (pd, qd, wd),
            "wta_i2e_x": ie_edges["wta_i2e_x"], "wta_i2e_y": ie_edges["wta_i2e_y"]}


# =====================================================================================================================
# NEURAL READ-OUT (content path -- fully spiking; the winner is the ensemble the LATERAL WTA leaves firing most)
# =====================================================================================================================
def _neural_predict(b, idx, snap, U, G, silence=False, floors_x=None, floors_y=None, t_step=READ_T_STEP):
    """Drive the reservoir per (s,a); reservoir SPIKES drive the ensembles (W+ excitatory, W- via feedforward-inhib
    interneurons); a per-ensemble tonic floor sets baseline; the shared inhibitory pool runs the biased-competition
    LATERAL-INHIBITION WTA that AMPLIFIES the small feed-forward margin. The predicted coordinate = the ensemble the
    competition leaves firing MOST (a motor-style read of `cp_firing_states` counts -- the RESOLVING is the neural
    competition, load-bearing under the I->E lesion). Returns the winner + the winner's dominance share + activity.
    NO host map matmul, NO argmax over host logits (grep-checked)."""
    xp, _ = get_backend()
    res = idx["res"]; res_exc = idx["res_exc"]; W_in = idx["W_in"]
    res_dev = xp.asarray(res)
    x_ens = idx["x_ens"]; y_ens = idx["y_ens"]
    x_dev = [xp.asarray(e) for e in x_ens]
    y_dev = [xp.asarray(e) for e in y_ens]
    fx = np.full(G, FLOOR_BASE) if floors_x is None else np.asarray(floors_x, np.float64)
    fy = np.full(G, FLOOR_BASE) if floors_y is None else np.asarray(floors_y, np.float64)
    _restore_state(b, snap)
    b.cp_external_input_current[:] = 0.0
    x_spk = np.zeros(G, np.float64); y_spk = np.zeros(G, np.float64); res_spk = 0.0; ens_spk = 0.0; steps = 0
    for _rep in range(int(READ_REPLAY)):
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
                    xs = float(fs[x_ens[r]].sum()); ys = float(fs[y_ens[r]].sum())
                    x_spk[r] += xs; y_spk[r] += ys; ens_spk += xs + ys
                steps += 1
    _restore_state(b, snap)
    b.cp_external_input_current[:] = 0.0
    # NEURAL WINNER: which competed ensemble fired most (per block). The lateral WTA did the resolving.
    xw = int(np.argmax(x_spk)); yw = int(np.argmax(y_spk))
    x_dom = float(x_spk[xw] / (x_spk.sum() + 1e-9)); y_dom = float(y_spk[yw] / (y_spk.sum() + 1e-9))
    dbg = dict(x_spk=x_spk.tolist(), y_spk=y_spk.tolist(),
               dominance=float(min(x_dom, y_dom)), dominant=bool(x_dom >= DOM_THRESH and y_dom >= DOM_THRESH),
               res_mean_spk=res_spk / max(1, steps * len(res_exc)),
               ens_mean_spk=ens_spk / max(1, steps * 2 * G * idx["ens_p"]))
    return (xw, yw), dbg


def _content_path_clean():
    """The winner comes from the LATERAL-INHIBITION-competed ensemble firing (`cp_firing_states` counts): grep-check
    NO host map matmul (`W_eff @`, `@ f`, `Wp_x @` ...), NO argmax over host logits; the winner read is argmax over the
    competed ensemble SPIKE-COUNTS (the accepted neural-WTA read, `rungB1b`/`PAST-RESERVOIR` GO), and the RESOLVING is
    the neural competition (load-bearing, proven by the I->E lesion)."""
    src = Path(__file__).read_text()
    lo = src.find("def _neural_predict("); hi = src.find("\ndef ", lo + 1)
    body = src[lo:hi] if lo >= 0 else ""
    q = body.find('"""'); q2 = body.find('"""', q + 3)
    code = (body[:q] + body[q2 + 3:]) if (q >= 0 and q2 > q) else body
    forbidden = ("@ Ws", "@ W_eff", "feat @", "W_eff @", "Wp_x @", "@ f", "argmax(dx", "argmax(pred")
    reads_spikes = ("np.argmax(x_spk)" in code) and ("np.argmax(y_spk)" in code) and ("cp_firing_states" in code)
    return bool(lo >= 0) and reads_spikes and not any(f in code for f in forbidden)


CAL_ITERS = 3
CAL_GAIN = 0.9


def _calibrate_floors(b, idx, snap, G, cal_pairs):
    """Intrinsic-excitability homeostasis on the read-out ensembles (TRAIN-only): equalize baseline firing to a
    set-point so the biased-competition WTA competes on INPUT drive, not intrinsic heterogeneity bias."""
    fx = np.full(G, FLOOR_BASE, np.float64); fy = np.full(G, FLOOR_BASE, np.float64)
    for _it in range(CAL_ITERS):
        xs = np.zeros(G); ys = np.zeros(G)
        for (s, a) in cal_pairs:
            _p, dbg = _neural_predict(b, idx, snap, _encode_seq(s, a, G), G, floors_x=fx, floors_y=fy)
            xs += np.asarray(dbg["x_spk"]); ys += np.asarray(dbg["y_spk"])
        xs /= max(1, len(cal_pairs)); ys /= max(1, len(cal_pairs))
        fx = np.clip(fx - CAL_GAIN * (xs - xs.mean()), 0.0, None)
        fy = np.clip(fy - CAL_GAIN * (ys - ys.mean()), 0.0, None)
    return fx, fy


def _rate_twopath_acc(Wp_x, Wm_x, Wp_y, Wm_y, b_eff, feats, pairs, sps, G):
    hit = 0
    for p, sp in zip(pairs, sps):
        f = feats[p]
        dx = Wp_x @ f - Wm_x @ f + b_eff[:G]
        dy = Wp_y @ f - Wm_y @ f + b_eff[G:2 * G]
        if (int(np.argmax(dx)), int(np.argmax(dy))) == sp:
            hit += 1
    return float(hit / max(1, len(pairs)))


def _accs(b, idx, snap, pairset, sps, G, silence=False, floors_x=None, floors_y=None):
    """Return (neural_wta acc, dominant-resolution rate, mean winner dominance, res_mean, ens_mean)."""
    hit = 0; dom_res = 0; dom = []
    res_spk = []; ens_spk = []
    for (s, a), sp in zip(pairset, sps):
        pred, dbg = _neural_predict(b, idx, snap, _encode_seq(s, a, G), G, silence=silence,
                                    floors_x=floors_x, floors_y=floors_y)
        hit += int(pred == sp); dom_res += int(dbg["dominant"]); dom.append(dbg["dominance"])
        res_spk.append(dbg["res_mean_spk"]); ens_spk.append(dbg["ens_mean_spk"])
    n = max(1, len(pairset))
    return (float(hit / n), float(dom_res / n), float(np.mean(dom)),
            float(np.mean(res_spk)), float(np.mean(ens_spk)))


# =====================================================================================================================
def _derisk_one(seed, G=5, n_pool=250, heldout_frac=0.25, smoke=False):
    t0 = time.time()
    pairs = _all_pairs(G)
    out_dim = 2 * G

    b, rm, idx, cfg = _build_bridge(seed, G, n_pool)
    b2, _, _, _ = _build_bridge(seed, G, n_pool)

    def _thash(bb):
        arr = getattr(bb, "cp_neuron_firing_thresholds", None)
        return None if arr is None else hashlib.sha1(np.asarray(to_host(arr)).astype(np.float64).tobytes()).hexdigest()
    seeded = bool(_thash(b) is not None and _thash(b) == _thash(b2))
    del b2

    union0 = {}
    ri = _reservoir_internal(rm, seed)
    if ri is not None:
        union0["reservoir_internal"] = ri
    inh0 = []
    for region in rm.regions():
        inh0.extend(rm.inhibitory_indices(region.name))
    b.inject_explicit_wiring(union0, output_inhibitory_indices=inh0 or None)
    snap = _snapshot_state(b)

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

    scale_grid = SYN_SCALE_GRID if not smoke else (6.0,)
    ffi_in_grid = FFI_IN_GRID if not smoke else (0.8,)
    ffi_out_grid = FFI_OUT_GRID if not smoke else (4.0,)
    ie_grid = WTA_IE_GRID if not smoke else (40.0,)

    n_sweep = 24
    train_probe = train[:n_sweep]
    tr_probe_sp = [_step(s, a, G) for (s, a) in train_probe]
    cal_pairs = train[:n_sweep]
    cal_pairs_full = train if not smoke else train[:n_sweep]

    # ---- ONE sweep: two-pathway gains + the LATERAL-INHIBITION STRENGTH, on the NEURAL-WTA TRAIN accuracy ----
    best = None
    for sc in scale_grid:
        for fin in ffi_in_grid:
            for fout in ffi_out_grid:
                for ie in ie_grid:
                    _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, sc, fin, fout, wta_ie=ie)
                    snap_w = _snapshot_state(b)
                    fx, fy = _calibrate_floors(b, idx, snap_w, G, cal_pairs)
                    hit = 0
                    for (s, a), sp in zip(train_probe, tr_probe_sp):
                        pred, _dbg = _neural_predict(b, idx, snap_w, _encode_seq(s, a, G), G, floors_x=fx, floors_y=fy)
                        hit += int(pred == sp)
                    acc = hit / max(1, len(train_probe))
                    if best is None or acc > best[0]:
                        best = (acc, sc, fin, fout, ie)
    train_agree, syn_scale, ffi_in_sel, ffi_out_sel, wta_ie_sel = best

    edges = _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, syn_scale, ffi_in_sel, ffi_out_sel, wta_ie=wta_ie_sel)
    snap_w = _snapshot_state(b)
    floors_x, floors_y = _calibrate_floors(b, idx, snap_w, G, cal_pairs_full)

    syn_train, dom_res_train, dom_train, _, ens_mean = _accs(b, idx, snap_w, train, tr_sp, G,
                                                             floors_x=floors_x, floors_y=floors_y)
    syn_held, dom_res_held, dom_held, res_mean_read, ens_mean_read = _accs(b, idx, snap_w, held, ho_sp, G,
                                                                           floors_x=floors_x, floors_y=floors_y)

    # ---- LOAD-BEARING WINNER: zero the lateral-inhibition I->E synapses -> no biased competition -> the argmax over
    # the un-competed counts collapses (the host-argmax fallback). This is the "winner is neural" proof. ----
    pie_x, qie_x, wie_x = edges["wta_i2e_x"]; pie_y, qie_y, wie_y = edges["wta_i2e_y"]
    b.set_pathway_weights("lesion_wtaie_x", pie_x, qie_x, np.zeros(len(pie_x), np.float32), add_missing=False)
    b.set_pathway_weights("lesion_wtaie_y", pie_y, qie_y, np.zeros(len(pie_y), np.float32), add_missing=False)
    snap_les = _snapshot_state(b)
    # recalibrate floors WITHOUT competition (fair fallback: the read-out still gets its homeostatic baseline)
    fx0, fy0 = _calibrate_floors(b, idx, snap_les, G, cal_pairs)
    wta_lesion_held, dom_res_noWTA, dom_noWTA, _, _ = _accs(b, idx, snap_les, held, ho_sp, G,
                                                            floors_x=fx0, floors_y=fy0)
    b.set_pathway_weights("restore_wtaie_x", pie_x, qie_x, wie_x, add_missing=False)
    b.set_pathway_weights("restore_wtaie_y", pie_y, qie_y, wie_y, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- LESION 1: zero the W+ read-out synapses -> collapse ----
    pxe, qxe, wxe = edges["wp_x"]; pye, qye, wye = edges["wp_y"]
    b.set_pathway_weights("lesion_wp_x", pxe, qxe, np.zeros(len(pxe), np.float32), add_missing=False)
    b.set_pathway_weights("lesion_wp_y", pye, qye, np.zeros(len(pye), np.float32), add_missing=False)
    snap_lw = _snapshot_state(b)
    lesion_wp_held, _, _, _, _ = _accs(b, idx, snap_lw, held, ho_sp, G, floors_x=floors_x, floors_y=floors_y)
    b.set_pathway_weights("restore_wp_x", pxe, qxe, wxe, add_missing=False)
    b.set_pathway_weights("restore_wp_y", pye, qye, wye, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- LESION 2: silence the reservoir input -> collapse ----
    silence_held, _, _, _, _ = _accs(b, idx, snap_w, held, ho_sp, G, silence=True, floors_x=floors_x, floors_y=floors_y)

    # ---- matched SHAM: count-matched lesion of the OFF-DECODE decoy read-out -> UNCHANGED ----
    pd, qd, wd = edges["wp_dec"]
    b.set_pathway_weights("sham_dec", pd, qd, np.zeros(len(pd), np.float32), add_missing=False)
    snap_sham = _snapshot_state(b)
    sham_held, _, _, _, _ = _accs(b, idx, snap_sham, held, ho_sp, G, floors_x=floors_x, floors_y=floors_y)
    b.set_pathway_weights("restore_dec", pd, qd, wd, add_missing=False)
    snap_w = _snapshot_state(b)

    # ---- UNTRAINED control: random non-negative weights of matched magnitude -> chance ----
    rng = np.random.default_rng(seed * 4242 + 1)
    Wp_x_r = rng.random(Wp_x.shape) * float(Wp.mean()); Wm_x_r = rng.random(Wm_x.shape) * float(Wm.mean())
    Wp_y_r = rng.random(Wp_y.shape) * float(Wp.mean()); Wm_y_r = rng.random(Wm_y.shape) * float(Wm.mean())
    _wire(b, rm, idx, seed, Wp_x_r, Wm_x_r, Wp_y_r, Wm_y_r, syn_scale, ffi_in_sel, ffi_out_sel, wta_ie=wta_ie_sel)
    snap_ut = _snapshot_state(b)
    fxr, fyr = _calibrate_floors(b, idx, snap_ut, G, cal_pairs)
    untrained_held, _, _, _, _ = _accs(b, idx, snap_ut, held, ho_sp, G, floors_x=fxr, floors_y=fyr)

    edges = _wire(b, rm, idx, seed, Wp_x, Wm_x, Wp_y, Wm_y, syn_scale, ffi_in_sel, ffi_out_sel, wta_ie=wta_ie_sel)
    snap_w = _snapshot_state(b)

    lever("neural_wta_vs_wtalesion(argmax_fallback)", before=round(wta_lesion_held, 4), after=round(syn_held, 4),
          required=False)
    lever("wp_readout_lesion", before=round(syn_held, 4), after=round(lesion_wp_held, 4), required=False)
    lever("reservoir_silence_lesion", before=round(syn_held, 4), after=round(silence_held, 4), required=False)
    lever("matched_sham_decoy", before=round(syn_held, 4), after=round(sham_held, 4), required=False)
    lever("untrained_control", before=round(syn_held, 4), after=round(untrained_held, 4), required=False)

    from collections import Counter
    tr_counter = Counter(tr_sp)
    prior_sp = tr_counter.most_common(1)[0][0] if tr_counter else (0, 0)
    prior_held = float(np.mean([prior_sp == sp for sp in ho_sp]))
    chance = 1.0 / (G * G)

    elapsed = time.time() - t0
    return dict(
        seed=int(seed), G=int(G), n_pool=int(n_pool), ens_p=int(ENS_P), ffi_p=int(FFI_P),
        heldout_n=len(held), train_n=len(train), chance=float(chance), chance_per_block=float(1.0 / G),
        mean_reservoir_spikes_feature=mean_spikes, res_mean_spk_read=float(res_mean_read),
        ens_mean_spk_read=float(ens_mean_read),
        syn_scale_selected=float(syn_scale), ffi_in_selected=float(ffi_in_sel), ffi_out_selected=float(ffi_out_sel),
        wta_ie_selected=float(wta_ie_sel), train_agree_neural_probe=float(train_agree),
        ridge_train_acc=ridge_train, ridge_heldout_acc=ridge_held,
        twopath_rate_train=twopath_rate_train, twopath_rate_heldout=twopath_rate_held,
        # the deliverable + the load-bearing fallback (lateral inhibition removed)
        neural_wta_heldout=float(syn_held), neural_wta_train=float(syn_train),
        wta_lesion_heldout=float(wta_lesion_held),          # host-argmax fallback (no biased competition) -> ~chance
        prior_baseline_host_argmax=0.447,                   # held baseline branch worktree-wf_122d7930-bc4-2, 6-seed
        winner_dominance_heldout=float(dom_held), dominant_rate_heldout=float(dom_res_held),
        wta_lesion_dominance=float(dom_noWTA), wta_lesion_dominant_rate=float(dom_res_noWTA),
        lesion_wp_heldout=float(lesion_wp_held), lesion_silence_heldout=float(silence_held),
        matched_sham_heldout=float(sham_held), untrained_control_heldout=float(untrained_held),
        prior_lookup_heldout=prior_held,
        content_path_clean=_content_path_clean(), seeded=seeded, elapsed_s=round(elapsed, 1),
    )


def _verdict(d):
    v = Verdict("fm neural-WTA world-model read-out (lateral-inhibition winner, load-bearing)", chance=d["chance"])
    v.disabled("STDP/Hebbian/STP/structural", "fixed reservoir + delta-trained two-pathway synapses + a fixed "
               "lateral-inhibition WTA; on-substrate homeostasis = the read-out floor calibration")
    v.require("two-pathway rate == ridge (decomposition exact)",
              abs(d["twopath_rate_heldout"] - d["ridge_heldout_acc"]), expect=lambda x: x <= 1e-6)
    v.require("reservoir active (feature)", d["mean_reservoir_spikes_feature"], expect=lambda x: x > 0.0)
    v.require("reservoir active (read)", d["res_mean_spk_read"], expect=lambda x: x > 0.0)
    v.require("ensembles active (read)", d["ens_mean_spk_read"], expect=lambda x: x > 0.0)
    v.require("seeded (byte-identical substrate)", d["seeded"], expect=True)
    v.require("content path clean (no host matmul/logit-argmax)", d["content_path_clean"], expect=True)
    v.require("matched sham UNCHANGED (|d|<=0.08)", abs(d["neural_wta_heldout"] - d["matched_sham_heldout"]),
              expect=lambda x: x <= 0.08)
    go = (d["neural_wta_heldout"] - d["wta_lesion_heldout"] >= 0.20
          and d["neural_wta_heldout"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.20
          and d["neural_wta_heldout"] >= d["twopath_rate_heldout"] - 0.15
          and (d["neural_wta_heldout"] - d["lesion_wp_heldout"]) >= 0.20
          and (d["neural_wta_heldout"] - d["lesion_silence_heldout"]) >= 0.20
          and d["untrained_control_heldout"] <= d["chance"] + 0.08)
    dec = v.decide(go=go)
    dec["go_criteria"] = {
        "wta_load_bearing(neural-fallback>=0.20)": bool(d["neural_wta_heldout"] - d["wta_lesion_heldout"] >= 0.20),
        "beats_chance/prior(>=+0.20)": bool(d["neural_wta_heldout"] - max(d["chance"], d["prior_lookup_heldout"]) >= 0.20),
        "approaches_rate_ceiling(>=ceil-0.15)": bool(d["neural_wta_heldout"] >= d["twopath_rate_heldout"] - 0.15),
        "wp_lesion_collapses(>=0.20)": bool((d["neural_wta_heldout"] - d["lesion_wp_heldout"]) >= 0.20),
        "silence_collapses(>=0.20)": bool((d["neural_wta_heldout"] - d["lesion_silence_heldout"]) >= 0.20),
        "untrained<=chance+0.08": bool(d["untrained_control_heldout"] <= d["chance"] + 0.08),
    }
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--n-pool", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default="research/findings/raw/_fm_neural_wta_readout_smoke.json")
    args = ap.parse_args()

    results = []
    for seed in args.seeds:
        try:
            d = _derisk_one(seed, G=args.G, n_pool=args.n_pool, smoke=args.smoke)
            dec = _verdict(d)
            d["verdict"] = dec
            results.append(d)
            print(f"\n=== seed {seed} ===")
            for k in ("mean_reservoir_spikes_feature", "res_mean_spk_read", "ens_mean_spk_read",
                      "syn_scale_selected", "ffi_in_selected", "ffi_out_selected", "wta_ie_selected",
                      "train_agree_neural_probe", "ridge_heldout_acc", "twopath_rate_heldout",
                      "prior_baseline_host_argmax", "wta_lesion_heldout", "neural_wta_heldout",
                      "winner_dominance_heldout", "dominant_rate_heldout", "wta_lesion_dominance",
                      "lesion_wp_heldout", "lesion_silence_heldout", "matched_sham_heldout",
                      "untrained_control_heldout", "prior_lookup_heldout", "chance",
                      "content_path_clean", "seeded", "elapsed_s"):
                print(f"  {k:36s} {d[k]}")
            print(f"  VERDICT: {dec['status']}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results.append({"seed": int(seed), "error": repr(e)})

    payload = {"runner": "_fm_neural_wta_readout_derisk", "argv": sys.argv, "seeds": list(args.seeds),
               "results": results,
               "preconditions": (results[0].get("verdict", {}).get("preconditions") if results else None)}
    outp = _REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
