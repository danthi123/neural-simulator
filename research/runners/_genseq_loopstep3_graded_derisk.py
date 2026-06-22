"""LOOP-STEP 3 de-risk #2 — GRADED / analog transmission RESOLUTION of the de-risk #1
rate-saturation NEGATIVE.

Scoping: research/findings/2026-06-22-genseq-consolidation-past-saturation-scoping.md
  resolution #1 (the cheapest, NO-`sim/`-edit surpass): stop reading the SATURATING
  on-bridge SPIKE-RATE (`cp_firing_states`, hard-capped at 0.5 by the 1-step refractory at
  dt=1ms) and instead drive + read the GRADED / analog membrane signal the bridge ALREADY
  transmits (`RegionPathway.graded=True`, `sim/regions.py:355-372`; step block
  `sim/bridge.py:6128-6175`): a_cont = clip((v - rest)/scale, 0, 1).

Clones the de-risk #1 runner (_genseq_loopstep3_multilayer_signed_derisk.py) and changes
ONLY the RUNNER (NO sim/ edit -- the graded path + the `"graded": True` wiring flag in
inject_explicit_wiring already exist, bridge.py:2450/2674/6141):

  1. tag EVERY block's wiring (signed E/I copies) `"graded": True` -> the whole forward chain
     transmits from the SOURCE membrane a_cont, bypassing the spike refractory ceiling, E/I
     routing preserved (an inhibitory graded source feeds g_i).
  2. READ each block's output as the analog a_cont = clip((v - rest)/scale, 0, 1) of its
     output neurons (NOT cp_firing_states).
  3. COMPARE analog<->analog vs an OFF-BRIDGE GRADED analog forward of the SAME blocks
     (a_{L+1} = clip(a_L @ W_L, 0, 1) -- the matched graded/analog ground truth a graded
     chain must reproduce). Also report vs the off-bridge SPIKING forward_unroll's per-layer
     membrane (the secondary reference, for honesty). Spearman/Pearson.
  4. SWEEP graded_source_scale_mV (the operating-point band) + a GREEDY PER-BLOCK graded-gain
     calibration (target each block's output a_cont mean in a live band ~0.3) -- the graded
     analogue of the scoping's per-layer threshold-balance, which AUTO-compensates each dense
     block's fan-in (without it the dense conductance blows up -> the membrane pins/diverges).
  5. Keep the signed E/I split + the matched/mismatched cross-input specificity anti-cheat.

THREE load-bearing findings the probes pinned BEFORE this run (all NO sim/ edit, all in the
runner): (i) the TRAIT FIX -- inject_explicit_wiring's output_inhibitory_indices only SETS
the listed I-copies to inhibitory; it does NOT zero the ~20% of E-copies the bridge init
randomly marked inhibitory, so the excitatory graded channel delivered ~0 (signed split
collapsed). Forcing the EXACT trait assignment lifts block-0 graded Spearman from -0.01 to
0.6-0.8 (vs spike-readout 0.32). (ii) NON-SPIKING regime -- under graded drive the targets
spike+reset, diluting the time-averaged a_cont; pushing V_T/V_peak high so the membrane is a
clean integrator gives the pure settled analog readout. (iii) FAN-IN GAIN -- a dense block's
graded conductance scales with its ~600 active sources; a single global gain that suits the
one-hot block-0 blows up the dense blocks (g_e ~9000 -> membrane pins at rest / diverges).
The greedy per-block gain calibration fixes this (block-1 -0.01 -> 0.57).

THE ONE LOAD-BEARING [VERIFY] (scoping §5.1): whether a_cont's UPPER clip RE-SATURATES under a
DENSE transformer-MLP full-fan-in activation. MEASURED here (frac_features_pinned_hi). If it
saturates, escalate IN THIS RUN to (m2) POPULATION coding (N neurons/feature, mean a_cont) +
threshold heterogeneity, the documented 47%->100%-of-host rate-code-wall lift.

Verdict (scoping §3):
  GO = cumulative analog-Spearman >= ~0.8 across the 2-3 stacked dense layers WITH signed
       graded wiring AND the specificity margin re-opens (matched >> mismatched). vs
       de-risk #1's 0.009 / 0.000.
  PARTIAL = above chance but < 0.8 -> the (m1)/(m2) mitigations are exercised; report best.
  NEGATIVE = even graded-analog + calibration + population stays at chance.

NO sim/ edit. Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_loopstep3_graded_derisk
"""
from __future__ import annotations

import json
import os
import sys
import math
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.bptt_snn import LIFLayer, forward_unroll

NPZ_PATH = _REPO / "research/findings/raw/phase_2_2b/cortex_10M_seed42.npz"
META_PATH = _REPO / "research/findings/raw/phase_2_2b/cortex_10M_seed42.metadata.json"
OUT_PATH = _REPO / "research/findings/raw/_genseq_loopstep3_graded.json"

N_BLOCKS = 3
GRADED_REST_MV = -65.0   # a_cont = clip((v - rest)/scale, 0, 1); maps the AdEx [-65,-50] band.
# greedy per-block gain calibration target (output a_cont mean). The probe scan showed dense
# blocks track best at a LOW occupancy (~0.10-0.20: the membrane stays in the linear-response
# band; higher occupancy enters the g.(V-E) saturation that destroys the matmul rank). 0.18.
A_CONT_TARGET = 0.18


# ---------------------------------------------------------------------------
# Off-bridge references
# ---------------------------------------------------------------------------
def load_artifact():
    # allow_pickle=True: the npz is OUR OWN trusted, local, project-generated training output
    # (research/runners/cortex_pretraining.py save_checkpoint); the n_layers scalar is a 0-d
    # object array. NOT untrusted input -- same as the de-risk #1 runner + step-0 probe.
    d = np.load(NPZ_PATH, allow_pickle=True)
    n_layers = int(d["n_layers"])
    Ws = [d[f"W_layer_{i}"].astype(np.float32) for i in range(n_layers)]
    thresholds = [float(t) for t in d["thresholds"]]
    leaks = [float(l) for l in d["leaks"]]
    layer_sizes = [int(x) for x in d["layer_sizes"]]
    vocab = None
    if META_PATH.exists():
        vocab = json.loads(META_PATH.read_text())["vocab"]
    return Ws, thresholds, leaks, layer_sizes, vocab


def offbridge_graded_forward(Ws, input_oh, n_blocks):
    """The MATCHED analog<->analog ground truth: an off-bridge GRADED analog forward of the
    SAME first n_blocks blocks. a_{L+1} = clip(a_L @ W_L, 0, 1) -- the graded clip
    nonlinearity (ReLU + saturation), the off-bridge analogue of the on-bridge a_cont chain.
    Returns per-block output analog vectors."""
    a = input_oh.astype(np.float64)
    outs = []
    for L in range(n_blocks):
        a = np.clip(a @ Ws[L], 0.0, 1.0)
        outs.append(a.copy())
    return outs


def offbridge_spiking_membrane(Ws, thresholds, leaks, input_oh, T, n_blocks):
    """The SECONDARY reference (the trained net's actual SPIKING forward): forward_unroll
    over the first n_blocks layers, returning per-layer time-averaged MEMBRANE v (analog) AND
    per-layer mean spike rate (for the off active-count, kept for diagnostics)."""
    layers = [LIFLayer(W_in=Ws[i], n_post=Ws[i].shape[1],
                       threshold=thresholds[i], leak=leaks[i])
              for i in range(n_blocks)]
    V_in = Ws[0].shape[0]
    inp = np.tile(input_oh.reshape(1, 1, V_in), (T, 1, 1)).astype(np.float32)
    out = forward_unroll(inp, layers)
    v_analog = [out["v"][li][:, 0, :].mean(axis=0) for li in range(n_blocks)]
    rates = [out["spikes"][li][:, 0, :].mean(axis=0) for li in range(n_blocks)]
    return v_analog, rates


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------
def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)


def topk_overlap(a, b, k):
    a = np.asarray(a); b = np.asarray(b)
    ta = set(np.argsort(-a)[:k].tolist()); tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / max(1, k)


# ---------------------------------------------------------------------------
# Layout (signed E/I split, optional population coding n_per)
# ---------------------------------------------------------------------------
class Layout:
    def __init__(self, feature_sizes, n_blocks, n_per=1):
        self.feature_sizes = feature_sizes
        self.n_blocks = n_blocks
        self.n_per = int(n_per)
        self.e_base = []
        self.i_base = []
        cur = 0
        for li in range(n_blocks):
            nf = feature_sizes[li] * self.n_per
            self.e_base.append(cur); cur += nf
            self.i_base.append(cur); cur += nf
        nf_top = feature_sizes[n_blocks] * self.n_per
        self.readout_base = cur; cur += nf_top
        self.n_total = cur
        self.readout_size = feature_sizes[n_blocks]

    def e_neurons(self, li, f):
        b = self.e_base[li] + f * self.n_per
        return np.arange(b, b + self.n_per, dtype=np.int64)

    def i_neurons(self, li, f):
        b = self.i_base[li] + f * self.n_per
        return np.arange(b, b + self.n_per, dtype=np.int64)


def build_graded_signed_bridge(Ws, layout, seed=42, e_gain=1.0, i_gain=None,
                               per_layer_e_gain=None, per_layer_i_gain=None,
                               graded_scale_mV=40.0, non_spiking=True,
                               threshold_jitter_mV=0.0):
    """ONE AdEx bridge, multi-layer signed E/I split-channel, EVERY block graded=True.

    e_gain / i_gain         : global excitatory / inhibitory graded gains.
    per_layer_e_gain/_i_gain: per-block weight scale (the greedy fan-in calibration).
    graded_scale_mV         : a_cont band width (cfg.graded_source_scale_mV, the operating pt).
    non_spiking             : push V_T/V_peak high so the membrane is a clean integrator (no
                              spike-reset contamination of the time-averaged a_cont readout).
    threshold_jitter_mV     : population-coding threshold heterogeneity (only meaningful for
                              n_per>1 + a spiking read).
    """
    n_blocks = layout.n_blocks
    n_total = layout.n_total
    n_per = layout.n_per
    cfg = CoreSimConfig()
    cfg.num_neurons = n_total
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.default_neuron_type_adex = None

    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.enable_watts_strogatz = False
    cfg.enable_nmda = False

    cfg.enable_inhibitory_neurons = True
    cfg.inhibitory_trait_index = 1
    cfg.inhibitory_trait_indices = [1]
    cfg.num_traits = 2

    tau_m = -cfg.dt_ms / math.log(0.95)
    cfg.adex_g_L = 30.0
    cfg.adex_C = cfg.adex_g_L * tau_m
    cfg.adex_E_L = -70.0
    cfg.adex_V_T = -50.0
    cfg.adex_Delta_T = 0.5
    cfg.adex_a = 0.0
    cfg.adex_tau_w = 144.0
    cfg.adex_b = 0.0
    cfg.adex_V_r = -70.0
    cfg.adex_V_peak = -40.0
    cfg.refractory_period_steps = 1
    cfg.ou_std_current_pA = 0.0

    cfg.graded_source_rest_mV = float(GRADED_REST_MV)
    cfg.graded_source_scale_mV = float(graded_scale_mV)

    # i_gain default: equalize per-unit-conductance magnitude only modestly (NOT the 14x
    # driving-force compensation, which catastrophically over-inhibits in the graded regime
    # where the membrane swings far from rest). 1/14 keeps g_i ~ g_e per unit a_cont.
    if i_gain is None:
        i_gain = e_gain / 14.0
    cfg.propagation_strength = float(e_gain)
    cfg.inhibitory_propagation_strength = float(i_gain)
    cfg.syn_tau_g_e = 5.0
    cfg.syn_tau_g_i = 5.0

    viz_cfg = VisualizationConfig()
    runtime_state = RuntimeState()
    gpu_cfg = GPUConfig()
    bridge = SimulationBridge(core_config=cfg, viz_config=viz_cfg,
                              runtime_state=runtime_state, gpu_config=gpu_cfg)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized, "Bridge init failed"

    # --- signed E/I split-channel wiring, EVERY block graded=True --------------
    wiring = {}
    inhibitory_indices = []
    for li in range(n_blocks):
        nf = layout.feature_sizes[li]
        inhibitory_indices.extend(range(layout.i_base[li],
                                        layout.i_base[li] + nf * n_per))
    w_scale_per = 1.0 / float(n_per)

    for L in range(n_blocks):
        W = Ws[L]
        nz_src, nz_tgt = np.nonzero(np.abs(W) > 0.0)
        wv = W[nz_src, nz_tgt]
        pos = wv > 0.0
        neg = ~pos
        is_top_next = (L + 1) == n_blocks
        if is_top_next:
            tgt_copies = [("ro", layout.readout_base)]
        else:
            tgt_copies = [("e", layout.e_base[L + 1]), ("i", layout.i_base[L + 1])]
        for copy_tag, tgt_base in tgt_copies:
            for sign_tag, sel, src_base in (("pos", pos, layout.e_base[L]),
                                            ("neg", neg, layout.i_base[L])):
                s_feat = nz_src[sel]; t_feat = nz_tgt[sel]
                w_abs = np.abs(wv[sel]).astype(np.float32)
                if per_layer_e_gain is not None and sign_tag == "pos":
                    w_abs = w_abs * float(per_layer_e_gain[L])
                if per_layer_i_gain is not None and sign_tag == "neg":
                    w_abs = w_abs * float(per_layer_i_gain[L])
                if n_per == 1:
                    pre = (src_base + s_feat).astype(np.int64)
                    post = (tgt_base + t_feat).astype(np.int64)
                    wval = w_abs
                else:
                    src0 = src_base + s_feat * n_per
                    tgt0 = tgt_base + t_feat * n_per
                    so = np.arange(n_per)
                    pre = (src0[:, None, None] + so[None, :, None]).repeat(n_per, axis=2).reshape(-1).astype(np.int64)
                    post = (tgt0[:, None, None] + so[None, None, :]).repeat(n_per, axis=1).reshape(-1).astype(np.int64)
                    wval = np.repeat(w_abs * w_scale_per, n_per * n_per).astype(np.float32)
                conn_type = "E_TO_E" if sign_tag == "pos" else "I_TO_E"
                wiring[f"block{L}_{sign_tag}_{copy_tag}"] = {
                    "pre_indices": pre.tolist(), "post_indices": post.tolist(),
                    "initial_weights": wval, "plastic": False,
                    "conn_type": conn_type, "count": int(pre.size), "graded": True,
                }

    bridge.inject_explicit_wiring(wiring, output_inhibitory_indices=inhibitory_indices)

    # TRAIT FIX (load-bearing, NO sim/ edit -- per-neuron state assignment): the bridge init
    # randomly marks ~20% of ALL neurons inhibitory; output_inhibitory_indices only SETS the
    # listed I-copies, NOT zero the rest -> stray-inhibitory E-copies route positive weights
    # through g_i, collapsing the excitatory channel. Force the EXACT assignment.
    import cupy as cp
    bridge.cp_traits[:] = 0
    if inhibitory_indices:
        bridge.cp_traits[cp.asarray(np.asarray(inhibitory_indices, dtype=np.int64))] = 1
    bridge._cached_inhibitory_mask = None

    # NON-SPIKING regime: push the spike threshold + peak far above any reachable membrane so
    # the AdEx never fires -> the membrane settles to the conductance-determined steady state
    # (a clean analog readout, no reset-dilution of the time-averaged a_cont).
    if non_spiking:
        for cand in ("cp_adex_V_T", "cp_v_threshold", "cp_firing_threshold_v",
                     "cp_adex_V_peak", "cp_v_peak"):
            arr = getattr(bridge, cand, None)
            if arr is not None and getattr(arr, "shape", (0,))[0] == n_total:
                arr[:] = 1.0e4
    elif threshold_jitter_mV > 0.0:
        for cand in ("cp_adex_V_T", "cp_v_threshold", "cp_firing_threshold_v"):
            arr = getattr(bridge, cand, None)
            if arr is not None and getattr(arr, "shape", (0,))[0] == n_total:
                rng = np.random.default_rng(seed + 777)
                jit = rng.uniform(-threshold_jitter_mV, threshold_jitter_mV, size=n_total).astype(np.float32)
                arr[:] = arr + cp.asarray(jit)
                break

    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0
    return bridge, cfg


def onbridge_block_analog(bridge, cfg, layout, active_input_dims, drive_pA,
                          n_steps, warmup, graded_scale_mV):
    """Drive the INPUT layer; let the frozen GRADED signed wiring carry the multi-stage analog
    wave forward; READ each BLOCK OUTPUT as the analog a_cont = clip((v-rest)/scale,0,1) of its
    output neurons, time-averaged over the window. For n_per>1 the per-feature value is the
    MEAN a_cont over the feature's n_per neurons (population read).

    Returns (analog_blocks, rate_blocks, sat_blocks)."""
    import cupy as cp
    n_total = layout.n_total
    n_blocks = layout.n_blocks
    n_per = layout.n_per
    rest = cp.float32(GRADED_REST_MV)
    inv_scale = cp.float32(1.0 / max(1e-3, graded_scale_mV))

    drive = cp.zeros(n_total, dtype=cp.float32)
    for d in active_input_dims:
        for nidx in layout.e_neurons(0, int(d)):
            drive[int(nidx)] = cp.float32(drive_pA)
        for nidx in layout.i_neurons(0, int(d)):
            drive[int(nidx)] = cp.float32(drive_pA)
    bridge.cp_external_input_current[:] = drive

    a_sum, a_max, s_sum = [], [], []
    for li in range(n_blocks):
        span = layout.feature_sizes[li + 1] * n_per
        a_sum.append(cp.zeros(span, dtype=cp.float64))
        a_max.append(cp.zeros(span, dtype=cp.float64))
        s_sum.append(cp.zeros(span, dtype=cp.float64))

    counted = 0
    for step in range(warmup + n_steps):
        bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = drive
        if step >= warmup:
            v = bridge.cp_membrane_potential_v
            a_cont_all = cp.clip((v - rest) * inv_scale, 0.0, 1.0)
            fired = bridge.cp_firing_states
            for li in range(n_blocks):
                span = layout.feature_sizes[li + 1] * n_per
                base = layout.e_base[li + 1] if li < n_blocks - 1 else layout.readout_base
                ac = a_cont_all[base:base + span].astype(cp.float64)
                a_sum[li] += ac
                a_max[li] = cp.maximum(a_max[li], ac)
                s_sum[li] += fired[base:base + span].astype(cp.float64)
            counted += 1

    analog_blocks, rate_blocks, sat_blocks = [], [], []
    for li in range(n_blocks):
        out_feat = layout.feature_sizes[li + 1]
        a_mean = a_sum[li] / max(1, counted)
        s_mean = s_sum[li] / max(1, counted)
        a_peak = a_max[li]
        if n_per == 1:
            a_feat, s_feat, a_peak_feat = a_mean, s_mean, a_peak
        else:
            a_feat = a_mean.reshape(out_feat, n_per).mean(axis=1)
            s_feat = s_mean.reshape(out_feat, n_per).mean(axis=1)
            a_peak_feat = a_peak.reshape(out_feat, n_per).max(axis=1)
        analog_blocks.append(a_feat.get())
        rate_blocks.append(s_feat.get())
        sat_blocks.append({
            "a_mean": float(a_feat.mean()), "a_max": float(a_peak_feat.max()),
            "frac_pinned_hi": float((a_feat > 0.99).mean()),
        })
    return analog_blocks, rate_blocks, sat_blocks


def greedy_block_gain_calibration(Ws, layout, cal_dim, *, e_gain, graded_scale_mV,
                                  non_spiking, threshold_jitter_mV, drive_pA, n_steps, warmup,
                                  n_iters=6, target=A_CONT_TARGET):
    """Greedy per-block graded-gain calibration (the graded analogue of de-risk #1's per-layer
    threshold-balance): for each block L in order, holding earlier blocks fixed, geometric-
    bisect block-L's incoming weight scale so block-L's OUTPUT a_cont mean lands near `target`
    (a live, un-pinned band). This AUTO-compensates each dense block's fan-in (a single global
    gain blows up the dense conductance -> the membrane pins/diverges). Calibrated on one char.
    Returns (per_layer_scale, log)."""
    n_blocks = layout.n_blocks
    per_layer_scale = [1.0] * n_blocks
    log = []

    def block_a_cont_mean(scales, L):
        b, c = build_graded_signed_bridge(
            Ws, layout, seed=42, e_gain=e_gain,
            per_layer_e_gain=scales, per_layer_i_gain=scales,
            graded_scale_mV=graded_scale_mV, non_spiking=non_spiking,
            threshold_jitter_mV=threshold_jitter_mV)
        ab, _rb, _sat = onbridge_block_analog(
            b, c, layout, active_input_dims=[cal_dim], drive_pA=drive_pA,
            n_steps=n_steps, warmup=warmup, graded_scale_mV=graded_scale_mV)
        return float(ab[L].mean())

    for L in range(n_blocks):
        lo, hi = 1e-4, 5.0
        chosen = per_layer_scale[L]; best_err = None
        for _it in range(n_iters):
            mid = math.sqrt(lo * hi)
            trial = list(per_layer_scale); trial[L] = mid
            am = block_a_cont_mean(trial, L)
            err = am - target
            log.append({"block": L, "scale": round(mid, 6), "a_cont_mean": round(am, 4)})
            if best_err is None or abs(err) < abs(best_err):
                best_err, chosen = err, mid
            if am > target:
                hi = mid
            else:
                lo = mid
            if abs(err) <= 0.03:
                break
        per_layer_scale[L] = chosen
        print(f"[graded:cal] block {L} gain={chosen:.6f} -> a_cont_mean~{target + (best_err or 0):.3f}",
              flush=True)
    return per_layer_scale, log


# ---------------------------------------------------------------------------
# One full evaluation pass.
# ---------------------------------------------------------------------------
def evaluate_config(Ws, thresholds, leaks, layer_sizes, vocab, probe_dims, *,
                    n_blocks, n_per, graded_scale_mV, e_gain, drive_pA, T, n_steps, warmup,
                    non_spiking, threshold_jitter_mV, calibrate, label):
    feature_sizes = layer_sizes[:n_blocks + 1]
    layout = Layout(feature_sizes, n_blocks, n_per=n_per)
    print(f"[graded:{label}] n_per={n_per} scale={graded_scale_mV} e_gain={e_gain} "
          f"non_spiking={non_spiking} -> n_total={layout.n_total}", flush=True)
    cal_dim = probe_dims[0]

    per_layer_scale = [1.0] * n_blocks
    cal_log = []
    if calibrate:
        per_layer_scale, cal_log = greedy_block_gain_calibration(
            Ws, layout, cal_dim, e_gain=e_gain, graded_scale_mV=graded_scale_mV,
            non_spiking=non_spiking, threshold_jitter_mV=threshold_jitter_mV,
            drive_pA=drive_pA, n_steps=n_steps, warmup=warmup)
    refine = any(abs(s - 1.0) > 1e-9 for s in per_layer_scale)
    ple = per_layer_scale if refine else None

    def make_bridge():
        return build_graded_signed_bridge(
            Ws, layout, seed=42, e_gain=e_gain,
            per_layer_e_gain=ple, per_layer_i_gain=ple,
            graded_scale_mV=graded_scale_mV, non_spiking=non_spiking,
            threshold_jitter_mV=threshold_jitter_mV)

    per_input = []
    on_by_char = {}
    off_graded_by_char = {}
    off_spk_by_char = {}
    sat_agg = [{"a_mean": [], "a_max": [], "frac_pinned_hi": [], "on_spk_max": []}
               for _ in range(n_blocks)]
    V_in = Ws[0].shape[0]
    for dim in probe_dims:
        oh = np.zeros(V_in, dtype=np.float32); oh[dim] = 1.0
        off_graded = offbridge_graded_forward(Ws, oh, n_blocks)
        off_spk_v, _r = offbridge_spiking_membrane(Ws, thresholds, leaks, oh, T, n_blocks)
        bridge, cfg = make_bridge()
        on_analog, on_rates, sat = onbridge_block_analog(
            bridge, cfg, layout, active_input_dims=[dim], drive_pA=drive_pA,
            n_steps=n_steps, warmup=warmup, graded_scale_mV=graded_scale_mV)
        off_graded_by_char[dim] = off_graded
        off_spk_by_char[dim] = off_spk_v
        on_by_char[dim] = on_analog
        for li in range(n_blocks):
            sat_agg[li]["a_mean"].append(sat[li]["a_mean"])
            sat_agg[li]["a_max"].append(sat[li]["a_max"])
            sat_agg[li]["frac_pinned_hi"].append(sat[li]["frac_pinned_hi"])
            sat_agg[li]["on_spk_max"].append(float(on_rates[li].max()))

        layer_metrics = []
        for li in range(n_blocks):
            on_r = on_analog[li]
            sp_gr = spearman(off_graded[li], on_r)
            sp_spk = spearman(off_spk_v[li], on_r)
            pe_gr = pearson(off_graded[li], on_r)
            k = max(10, int(0.25 * on_r.size))
            ov = topk_overlap(off_graded[li], on_r, k)
            layer_metrics.append({
                "layer": li, "spearman_vs_graded": sp_gr, "spearman_vs_spiking": sp_spk,
                "pearson_vs_graded": pe_gr, "topk_overlap": ov, "k": int(k),
                "on_a_cont_mean": sat[li]["a_mean"], "on_a_cont_max": sat[li]["a_max"],
            })
        cumulative_sp = layer_metrics[-1]["spearman_vs_graded"]
        per_input.append({
            "char": (vocab[dim] if vocab else None), "dim": int(dim),
            "layers": layer_metrics, "cumulative_spearman": cumulative_sp,
        })
        ls = " ".join("L%d:%s" % (m["layer"], ("nan" if math.isnan(m["spearman_vs_graded"])
                      else "%.2f" % m["spearman_vs_graded"])) for m in layer_metrics)
        print(f"[graded:{label}] char={per_input[-1]['char']!r} dim={dim:2d} | {ls} | "
              f"CUMUL sp={'nan' if math.isnan(cumulative_sp) else f'{cumulative_sp:.3f}'}", flush=True)

    # ANTI-CHEAT specificity on the FINAL graded stage (vs the matched off-bridge graded ref).
    dims = list(on_by_char.keys())
    matched, mismatched = [], []
    for d_on in dims:
        on_final = on_by_char[d_on][n_blocks - 1]
        for d_off in dims:
            off_final = off_graded_by_char[d_off][n_blocks - 1]
            s = spearman(off_final, on_final)
            if math.isnan(s):
                continue
            (matched if d_on == d_off else mismatched).append(s)
    spec = {
        "matched_mean_spearman": float(np.mean(matched)) if matched else float("nan"),
        "mismatched_mean_spearman": float(np.mean(mismatched)) if mismatched else float("nan"),
    }
    spec["specificity_margin"] = spec["matched_mean_spearman"] - spec["mismatched_mean_spearman"]

    per_layer_agg = []
    for li in range(n_blocks):
        sg = [r["layers"][li]["spearman_vs_graded"] for r in per_input
              if not math.isnan(r["layers"][li]["spearman_vs_graded"])]
        ss = [r["layers"][li]["spearman_vs_spiking"] for r in per_input
              if not math.isnan(r["layers"][li]["spearman_vs_spiking"])]
        ovs = [r["layers"][li]["topk_overlap"] for r in per_input]
        per_layer_agg.append({
            "layer": li,
            "mean_spearman_vs_graded": float(np.mean(sg)) if sg else float("nan"),
            "mean_spearman_vs_spiking": float(np.mean(ss)) if ss else float("nan"),
            "mean_topk_overlap": float(np.mean(ovs)) if ovs else float("nan"),
            "a_cont_mean": float(np.mean(sat_agg[li]["a_mean"])),
            "a_cont_max": float(np.mean(sat_agg[li]["a_max"])),
            "frac_features_pinned_hi": float(np.mean(sat_agg[li]["frac_pinned_hi"])),
            "on_spike_max_rate": float(np.mean(sat_agg[li]["on_spk_max"])),
        })
    cumul_sps = [r["cumulative_spearman"] for r in per_input
                 if not math.isnan(r["cumulative_spearman"])]
    cumulative_mean_spearman = float(np.mean(cumul_sps)) if cumul_sps else float("nan")

    a_cont_saturated = any(
        (per_layer_agg[li]["a_cont_mean"] >= 0.95 and per_layer_agg[li]["frac_features_pinned_hi"] >= 0.9)
        for li in range(1, n_blocks))

    return {
        "label": label, "n_per": n_per, "graded_scale_mV": graded_scale_mV, "e_gain": e_gain,
        "non_spiking": non_spiking, "threshold_jitter_mV": threshold_jitter_mV,
        "n_total_neurons": layout.n_total,
        "per_layer_gain_calibration": per_layer_scale,
        "per_layer_gain_calibration_log": cal_log,
        "per_layer_fidelity": per_layer_agg,
        "cumulative_mean_spearman": cumulative_mean_spearman,
        "anti_cheat_specificity": spec,
        "a_cont_saturated": a_cont_saturated,
        "per_input": per_input,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[graded] SIM_BACKEND={backend}", flush=True)
    Ws, thresholds, leaks, layer_sizes, vocab = load_artifact()
    n_blocks = N_BLOCKS
    feature_sizes = layer_sizes[:n_blocks + 1]
    print(f"[graded] stacking {n_blocks} GRADED signed blocks, feature_sizes={feature_sizes}", flush=True)

    T = 24
    n_steps = 36
    warmup = 18
    drive_pA = 4000.0
    e_gain = 1.0

    if vocab is not None:
        probe_chars = [" ", "e", "t", "a", "o", "h"]
        probe_dims = [vocab.index(c) for c in probe_chars if c in vocab]
    else:
        probe_dims = [2, 44, 59, 40, 54, 47]
    probe_dims = probe_dims[:6]

    GO_BAR = 0.8

    # =====================================================================
    # PHASE A — graded + a_cont readout + greedy per-block gain calibration, sweeping the
    # graded_source_scale_mV operating band. Non-spiking analog regime. n_per=1.
    # =====================================================================
    print("\n[graded] ===== PHASE A: graded + greedy per-block gain calib, scale sweep =====", flush=True)
    scale_sweep = [40.0, 80.0, 20.0]
    phaseA, bestA = [], None
    for sc in scale_sweep:
        res = evaluate_config(
            Ws, thresholds, leaks, layer_sizes, vocab, probe_dims,
            n_blocks=n_blocks, n_per=1, graded_scale_mV=sc, e_gain=e_gain,
            drive_pA=drive_pA, T=T, n_steps=n_steps, warmup=warmup,
            non_spiking=True, threshold_jitter_mV=0.0, calibrate=True,
            label=f"A_scale{int(sc)}")
        phaseA.append(res)
        cm = res["cumulative_mean_spearman"]; mg = res["anti_cheat_specificity"]["specificity_margin"]
        print(f"[graded] PHASE A scale={sc:5.1f} -> cumulative_sp="
              f"{'nan' if math.isnan(cm) else f'{cm:.3f}'} margin={mg:.3f} a_cont_sat={res['a_cont_saturated']} "
              f"per_block_sp={[round(a['mean_spearman_vs_graded'],3) for a in res['per_layer_fidelity']]}",
              flush=True)
        sc_score = cm if not math.isnan(cm) else -2.0
        best_score = (bestA["cumulative_mean_spearman"] if bestA and not math.isnan(bestA["cumulative_mean_spearman"]) else -2.0)
        if bestA is None or sc_score > best_score:
            bestA = res

    best = bestA
    phaseB = []
    used_mitigation = False

    # =====================================================================
    # PHASE B — mitigation (only if Phase A < GO): POPULATION coding (n_per>1, mean a_cont)
    # at the best Phase-A scale, with the greedy calibration. The (m2) lift.
    # =====================================================================
    bestA_cm = bestA["cumulative_mean_spearman"]
    if math.isnan(bestA_cm) or bestA_cm < GO_BAR:
        used_mitigation = True
        best_scale = bestA["graded_scale_mV"]
        print(f"\n[graded] ===== PHASE B: Phase A cumulative={bestA_cm:.3f} < {GO_BAR}; "
              f"escalating to POPULATION coding at scale={best_scale} =====", flush=True)
        for n_per in (4, 8):
            res = evaluate_config(
                Ws, thresholds, leaks, layer_sizes, vocab, probe_dims,
                n_blocks=n_blocks, n_per=n_per, graded_scale_mV=best_scale, e_gain=e_gain,
                drive_pA=drive_pA, T=T, n_steps=n_steps, warmup=warmup,
                non_spiking=True, threshold_jitter_mV=0.0, calibrate=True,
                label=f"B_nper{n_per}")
            phaseB.append(res)
            cm = res["cumulative_mean_spearman"]; mg = res["anti_cheat_specificity"]["specificity_margin"]
            print(f"[graded] PHASE B n_per={n_per} -> cumulative_sp="
                  f"{'nan' if math.isnan(cm) else f'{cm:.3f}'} margin={mg:.3f} a_cont_sat={res['a_cont_saturated']} "
                  f"per_block_sp={[round(a['mean_spearman_vs_graded'],3) for a in res['per_layer_fidelity']]}",
                  flush=True)
            if (not math.isnan(cm)) and (math.isnan(best["cumulative_mean_spearman"]) or cm > best["cumulative_mean_spearman"]):
                best = res

    # =====================================================================
    # Verdict on the BEST config.
    # =====================================================================
    cumulative_mean_spearman = best["cumulative_mean_spearman"]
    spec = best["anti_cheat_specificity"]
    per_layer_agg = best["per_layer_fidelity"]
    margin_ok = (not math.isnan(spec["specificity_margin"]) and spec["specificity_margin"] > 0.1)
    a_cont_saturated = best["a_cont_saturated"]

    if (not math.isnan(cumulative_mean_spearman)) and cumulative_mean_spearman >= GO_BAR and margin_ok:
        verdict = "GO"
    elif (not math.isnan(cumulative_mean_spearman)) and cumulative_mean_spearman >= 0.4 and margin_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    per_layer_sp = [None if math.isnan(a["mean_spearman_vs_graded"]) else round(a["mean_spearman_vs_graded"], 3)
                    for a in per_layer_agg]
    verdict_line = (
        "graded: blocks=%d GRADED_analog n_per=%d scale=%.0f non_spiking cumulative_analog_spearman=%.3f "
        "per_layer_spearman(vs_graded)=%s specificity_margin=%.3f a_cont_saturated=%s -> %s "
        "(vs de-risk#1 spike-rate: cumulative=0.009 margin=0.000)" % (
            n_blocks, best["n_per"], best["graded_scale_mV"], cumulative_mean_spearman,
            per_layer_sp, spec["specificity_margin"], a_cont_saturated, verdict))

    result = {
        "probe": "genseq_loopstep3_graded_analog",
        "resolves": "de-risk #1 rate-saturation NEGATIVE via graded(analog) transmission + a_cont readout",
        "artifact": str(NPZ_PATH.name),
        "n_blocks": n_blocks, "feature_sizes": feature_sizes,
        "neuron_model": "ADEX_as_LIF (signed E/I split-channel, GRADED analog transmission, non-spiking)",
        "method": "EVERY block RegionPathway graded=True (a_cont=clip((v-rest)/scale,0,1)); read each "
                  "block output as analog a_cont (NOT cp_firing_states); compare analog<->analog vs an "
                  "off-bridge GRADED analog forward (a=clip(a@W,0,1)); greedy per-block gain calibration "
                  "(target a_cont mean ~0.3) auto-compensates dense fan-in. NO sim/ edit.",
        "key_fixes_in_runner": [
            "TRAIT FIX: force cp_traits to the exact E/I assignment after inject_explicit_wiring "
            "(output_inhibitory_indices does not zero the init's random ~20% inhibitory marking)",
            "NON-SPIKING: push V_T/V_peak high so the membrane is a clean integrator (no spike-reset "
            "dilution of the a_cont readout)",
            "FAN-IN GAIN: greedy per-block gain calibration (a single global gain blows up the dense "
            "block conductance and pins/diverges the membrane)",
        ],
        "drive_pA": drive_pA, "T_off": T, "n_steps_on": n_steps, "warmup": warmup,
        "graded_rest_mV": GRADED_REST_MV, "a_cont_target": A_CONT_TARGET,
        "graded_scale_sweep": scale_sweep,
        "phaseA_scale_sweep": phaseA,
        "used_population_mitigation": used_mitigation,
        "phaseB_population_mitigation": phaseB,
        "best_config": {
            "label": best["label"], "n_per": best["n_per"], "graded_scale_mV": best["graded_scale_mV"],
            "e_gain": best["e_gain"], "non_spiking": best["non_spiking"],
            "n_total_neurons": best["n_total_neurons"],
            "per_layer_gain_calibration": best["per_layer_gain_calibration"],
        },
        "per_layer_fidelity": per_layer_agg,
        "cumulative_mean_spearman": cumulative_mean_spearman,
        "anti_cheat_specificity": spec,
        "a_cont_saturated": a_cont_saturated,
        "go_bar": GO_BAR,
        "baseline_derisk1": {"cumulative_mean_spearman": 0.0086, "specificity_margin": 0.0,
                             "note": "spike-rate readout pinned at 0.5 refractory ceiling"},
        "verdict_line": verdict_line, "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))
    print("\n[graded] SATURATION diagnostic (a_cont analog readout per block, BEST config):", flush=True)
    for a in per_layer_agg:
        pinned = (a["a_cont_mean"] >= 0.95 and a["frac_features_pinned_hi"] >= 0.9)
        print("[graded]   block %d: sp_vs_graded=%.3f sp_vs_spiking=%.3f | a_cont_mean=%.3f a_cont_max=%.3f "
              "frac_pinned=%.3f%s" % (a["layer"], a["mean_spearman_vs_graded"], a["mean_spearman_vs_spiking"],
              a["a_cont_mean"], a["a_cont_max"], a["frac_features_pinned_hi"],
              "  <- a_cont RE-SATURATED" if pinned else ""), flush=True)
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[graded] wrote {OUT_PATH}", flush=True)
    return result


if __name__ == "__main__":
    main()
