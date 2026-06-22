"""LOOP-STEP 3 de-risk #1 — MULTI-LAYER + SIGNED weights on an MLP-only bridge slice.

Scoping: research/findings/2026-06-22-genseq-loopstep3-consolidation-scoping.md
  ladder step #1 + Q2 (signed E/I) + Q3 (multi-layer fidelity, the largest uncertainty).

Extends step-0 (research/runners/_genseq_step0_bridge_load_probe.py, GO at 0.92 for
ONE positive-weight one-hot layer) along the two residuals step-0 explicitly deferred:

  (a) MULTI-LAYER — stack ≥2-3 SUCCESSIVE feedforward weight blocks of the trained
      cortex_10M_seed42.npz (66 -> 2048 -> 2048 -> 2048) as DISJOINT co-resident bridge
      slices, each feeding the next, so layer L's spiking-rate approximation error feeds
      layer L+1 (the error-accumulation test). Layers 1+ see REAL DENSE activations (the
      previous layer's actual spike output), NOT a one-hot — the harder rate-code case.

  (b) SIGNED weights via the bridge's existing E/I split (bridge.py:6084-6126). The
      bridge's E/I is PER-SOURCE-NEURON (a source is either E or I for ALL its outgoing
      synapses), but a trained weight ROW has both signs. The standard ANN->SNN
      split-channel convention: DUPLICATE each source feature into an excitatory copy and
      an inhibitory copy (both driven IDENTICALLY by the upstream layer), then wire
      W_ij>0 from the E-copy and W_ij<0 (with |W_ij|) from the I-copy. The E-copy has
      trait=excitatory (feeds g_e, depolarizing via E_e=0mV); the I-copy has trait=
      inhibitory (feeds g_i, hyperpolarizing via E_inh=-75mV) -> a signed synaptic sum.

NO sim/ edit (reuse-only: inject_explicit_wiring + output_inhibitory_indices + the E/I
split + the rate-window read). Per-layer AND cumulative fidelity vs the off-bridge
bptt_snn forward_unroll of the SAME blocks (Spearman/Pearson/top-k). Cross-input
specificity anti-cheat (matched vs mismatched), like step-0. Per-layer threshold-balance
sweep (the cheap fix the scoping names) if a single global gain degrades across layers.

HONEST: if cumulative fidelity COLLAPSES across layers (error accumulation on the
Izhikevich/AdEx dynamics) that IS the key Q3 finding -> reported plainly (points to
per-layer threshold-balance, or the surrogate-grad-on-bridge finetune fallback).

Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_loopstep3_multilayer_signed_derisk
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
OUT_PATH = _REPO / "research/findings/raw/_genseq_loopstep3_multilayer.json"

# How many SUCCESSIVE trained blocks to stack. 3 = layers 0,1,2
# (66->2048->2048->2048): layer 0 one-hot driven (signed), layers 1,2 see real dense
# activations (the harder case) with compounding error. (The 4th block is 2048->66, the
# readout; 3 hidden blocks already exercises 2 stages of real-dense error accumulation.)
N_BLOCKS = 3


# ---------------------------------------------------------------------------
# Off-bridge reference (the ground truth the bridge must track)
# ---------------------------------------------------------------------------
def load_artifact():
    # allow_pickle=True: the npz is OUR OWN trusted training output
    # (research/runners/cortex_pretraining.py save_checkpoint); the n_layers scalar is a
    # 0-d object array. Local, project-generated, safe (same as the step-0 probe).
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


def offbridge_layer_rates(Ws, thresholds, leaks, input_oh, T, n_blocks):
    """Off-bridge ground truth: run the trained LIF net's forward_unroll over the FIRST
    n_blocks layers on a (V_in,) one-hot held for T steps. Returns per-layer mean
    spike-rate vectors (the per-stage ground truth for the on-bridge comparison)."""
    layers = [LIFLayer(W_in=Ws[i], n_post=Ws[i].shape[1],
                       threshold=thresholds[i], leak=leaks[i])
              for i in range(n_blocks)]
    V_in = Ws[0].shape[0]
    inp = np.tile(input_oh.reshape(1, 1, V_in), (T, 1, 1)).astype(np.float32)
    out = forward_unroll(inp, layers)
    rates = [out["spikes"][li][:, 0, :].mean(axis=0) for li in range(n_blocks)]
    return rates


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
# Layout: disjoint slices on ONE bridge, signed E/I split-channel.
#
# Slice plan (n_blocks=3 example, sizes [66, 2048, 2048, 2048]):
#   input layer  : 66 features  -> E-copy + I-copy  = 132 neurons
#   hidden 1     : 2048 features -> E-copy + I-copy = 4096 neurons
#   hidden 2     : 2048 features -> E-copy + I-copy = 4096 neurons
#   readout(top) : 2048 features -> single copy     = 2048 neurons (we READ its rate)
#
# Each NON-top layer's feature f has an E-neuron AND an I-neuron, BOTH driven identically
# by the previous layer (so they spike identically with OU off + deterministic AdEx).
# Weight W[f,g] of block L wires:  W>0 -> E-copy(f) -> feature(g) ;  W<0 -> I-copy(f) -> g.
# The READOUT (top) layer needs no split (nothing reads from it downstream); we compare
# its per-feature spike rate to the off-bridge layer (n_blocks-1).
# ---------------------------------------------------------------------------
class Layout:
    def __init__(self, feature_sizes, n_blocks):
        # feature_sizes = layer_sizes[:n_blocks+1] (input + n_blocks hidden feature counts)
        self.feature_sizes = feature_sizes
        self.n_blocks = n_blocks
        # layers 0..n_blocks-1 are SPLIT (E+I copies); layer n_blocks is the readout (single)
        self.e_base = []  # per layer: start index of E-copies
        self.i_base = []  # per layer: start index of I-copies (None for readout)
        self.single_base = []  # per layer: start index for the readout's single copy
        cur = 0
        self.n_split_layers = n_blocks  # input + (n_blocks-1) hidden are split sources
        for li in range(n_blocks):
            nf = feature_sizes[li]
            self.e_base.append(cur); cur += nf
            self.i_base.append(cur); cur += nf
            self.single_base.append(None)
        # readout layer (the output of block n_blocks-1)
        nf_top = feature_sizes[n_blocks]
        self.readout_base = cur; cur += nf_top
        self.n_total = cur
        self.readout_size = nf_top

    def e_idx(self, li, f):
        return self.e_base[li] + f

    def i_idx(self, li, f):
        return self.i_base[li] + f


def build_multilayer_signed_bridge(Ws, layout, seed=42,
                                   e_gain=1.0, i_gain=None,
                                   per_layer_e_gain=None, per_layer_i_gain=None):
    """Build ONE AdEx-as-LIF bridge with the multi-layer signed E/I split-channel wiring.

    e_gain / i_gain : GLOBAL excitatory / inhibitory synaptic gains (propagation_strength /
                      inhibitory_propagation_strength). If per_layer_* are given they
                      OVERRIDE the global gain per block via a per-source weight scale
                      (a per-layer threshold-balance, the scoping's named cheap fix).
    """
    n_blocks = layout.n_blocks
    n_total = layout.n_total
    cfg = CoreSimConfig()
    cfg.num_neurons = n_total
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.default_neuron_type_adex = None  # disable preset overlay (step-0 gotcha)

    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.enable_watts_strogatz = False
    cfg.enable_nmda = False

    # E/I split ON (the whole point of this de-risk): inhibitory sources route to g_i.
    cfg.enable_inhibitory_neurons = True
    cfg.inhibitory_trait_index = 1
    cfg.inhibitory_trait_indices = [1]
    cfg.num_traits = 2  # E (0) + I (1)

    # AdEx-as-LIF tuning (verbatim step-0): zero adaptation, tau_m ~19.5 ms (leak 0.95),
    # sharp Delta_T -> near-hard threshold, reset to rest.
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

    # Global E and I synaptic gains. E_inh=-75, E_e=0: at rest (~-70 mV) the driving
    # forces are 70 (exc) vs -5 (inh), so the inhibitory channel needs a LARGER gain to
    # deliver a comparable-magnitude subtractive current per unit |weight|. Default the
    # I-gain so the per-unit-weight current magnitudes match at rest, then sweep.
    if i_gain is None:
        i_gain = e_gain * (abs(cfg.syn_reversal_potential_e - cfg.adex_E_L) /
                           max(1e-3, abs(cfg.syn_reversal_potential_i - cfg.adex_E_L)))
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

    # --- Build wiring ---------------------------------------------------------
    # For each block L (W: feat_L -> feat_{L+1}):
    #   target features live in layer L+1's E-copy AND I-copy (if L+1 is split) OR the
    #   readout single copy (if L+1 is the top). The previous layer's E/I copies of
    #   feature f BOTH project the SAME |W| to the SAME targets (so the next E-copy and
    #   I-copy of a target receive identical input -> spike identically). Source SIGN:
    #   W>0 read from source-feature f's E-copy; W<0 from f's I-copy.
    wiring = {}
    inhibitory_indices = []  # all I-copy neuron indices (flipped to trait=1)
    # Register every I-copy as inhibitory.
    for li in range(n_blocks):
        nf = layout.feature_sizes[li]
        inhibitory_indices.extend(range(layout.i_base[li], layout.i_base[li] + nf))

    for L in range(n_blocks):
        W = Ws[L]  # (feat_L, feat_{L+1})
        n_src, n_tgt = W.shape
        nz_src, nz_tgt = np.nonzero(np.abs(W) > 0.0)
        wv = W[nz_src, nz_tgt]
        pos = wv > 0.0
        neg = ~pos

        # target index resolver: each target feature g must be driven into BOTH the E-copy
        # and the I-copy of layer L+1 (so they remain identical), unless L+1 is the readout
        # (single copy).
        is_top_next = (L + 1) == n_blocks
        if is_top_next:
            tgt_copies = [("ro", layout.readout_base)]
        else:
            tgt_copies = [("e", layout.e_base[L + 1]), ("i", layout.i_base[L + 1])]

        for copy_tag, tgt_base in tgt_copies:
            # POSITIVE weights: source E-copy -> target copy
            pre_e = (layout.e_base[L] + nz_src[pos]).astype(np.int64)
            post_e = (tgt_base + nz_tgt[pos]).astype(np.int64)
            w_e = np.abs(wv[pos]).astype(np.float32)
            # NEGATIVE weights: source I-copy -> target copy (|W|, routed via g_i)
            pre_i = (layout.i_base[L] + nz_src[neg]).astype(np.int64)
            post_i = (tgt_base + nz_tgt[neg]).astype(np.int64)
            w_i = np.abs(wv[neg]).astype(np.float32)

            # per-layer threshold-balance (cheap fix): scale this block's |weights|.
            if per_layer_e_gain is not None:
                w_e = w_e * float(per_layer_e_gain[L])
            if per_layer_i_gain is not None:
                w_i = w_i * float(per_layer_i_gain[L])

            wiring[f"block{L}_pos_{copy_tag}"] = {
                "pre_indices": pre_e.tolist(),
                "post_indices": post_e.tolist(),
                "initial_weights": w_e,
                "plastic": False, "conn_type": "E_TO_E", "count": int(pre_e.size),
            }
            wiring[f"block{L}_neg_{copy_tag}"] = {
                "pre_indices": pre_i.tolist(),
                "post_indices": post_i.tolist(),
                "initial_weights": w_i,
                "plastic": False, "conn_type": "I_TO_E", "count": int(pre_i.size),
            }

    bridge.inject_explicit_wiring(wiring, output_inhibitory_indices=inhibitory_indices)
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0
    return bridge, cfg


def onbridge_block_output_rates(bridge, cfg, layout, active_input_dims, drive_pA,
                                n_steps, warmup):
    """Drive the INPUT layer's E-copy AND I-copy at the active one-hot dims with a strong
    constant current; let the frozen signed wiring carry the multi-stage wave forward; read
    the per-feature spike rate of every BLOCK OUTPUT.

    Returns a list of n_blocks rate vectors ALIGNED with the off-bridge `forward_unroll`
    block outputs (off block `li` <-> on-bridge layer slice `li+1`):
      block li (li < n_blocks-1) output = the E-copy rate of hidden layer (li+1)
                                          (the feature's spike rate; its I-copy is identical);
      block n_blocks-1 output           = the READOUT single-copy rate (the final stage).
    The input layer's own slice is NOT returned (it is just the driven one-hot, with no
    off-bridge counterpart)."""
    import cupy as cp
    n_total = layout.n_total
    n_blocks = layout.n_blocks
    drive = cp.zeros(n_total, dtype=cp.float32)
    for d in active_input_dims:
        # drive BOTH input copies identically so the signed block-0 sum is faithful.
        drive[layout.e_base[0] + int(d)] = cp.float32(drive_pA)
        drive[layout.i_base[0] + int(d)] = cp.float32(drive_pA)
    bridge.cp_external_input_current[:] = drive

    # one count vector per BLOCK OUTPUT (n_blocks total).
    counts = []
    for li in range(n_blocks):
        out_size = layout.feature_sizes[li + 1]  # block li output dim
        counts.append(cp.zeros(out_size, dtype=cp.float64))

    counted = 0
    total = warmup + n_steps
    for step in range(total):
        bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = drive  # re-assert constant DC drive
        if step >= warmup:
            fired = bridge.cp_firing_states
            for li in range(n_blocks):
                if li < n_blocks - 1:
                    # block li output = E-copy of hidden layer (li+1)
                    eb = layout.e_base[li + 1]; nf = layout.feature_sizes[li + 1]
                    counts[li] += fired[eb:eb + nf].astype(cp.float64)
                else:
                    # final block output = readout single copy
                    rb = layout.readout_base
                    counts[li] += fired[rb:rb + layout.readout_size].astype(cp.float64)
            counted += 1
    rates = [(c / max(1, counted)).get() for c in counts]
    return rates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[ml-signed] SIM_BACKEND={backend}")
    print(f"[ml-signed] loading artifact: {NPZ_PATH.name}")
    Ws, thresholds, leaks, layer_sizes, vocab = load_artifact()
    n_blocks = N_BLOCKS
    feature_sizes = layer_sizes[:n_blocks + 1]
    print(f"[ml-signed] stacking {n_blocks} signed blocks, feature_sizes={feature_sizes}")
    layout = Layout(feature_sizes, n_blocks)
    print(f"[ml-signed] n_total neurons={layout.n_total} (E+I split: layers 0..{n_blocks-1}, "
          f"readout single={layout.readout_size})")

    V_in = Ws[0].shape[0]
    T = 32
    n_steps = 64
    warmup = 16
    drive_pA = 4000.0

    if vocab is not None:
        probe_chars = [" ", "e", "t", "a", "o", "h"]
        probe_dims = [vocab.index(c) for c in probe_chars if c in vocab]
    else:
        probe_dims = [2, 44, 59, 40, 54, 47]
    probe_dims = probe_dims[:6]

    # =====================================================================
    # PHASE 1 — global E/I gain sweep, calibrated on one char ('e'), locked.
    # The fairest test: find the global synaptic gain at which the FINAL readout layer
    # fires in a comparable regime to the trained LIF (the cumulative target), then report
    # fidelity THERE. We track per-block active-feature counts across the sweep so the
    # error-accumulation behavior is visible.
    # =====================================================================
    cal_dim = probe_dims[0]
    cal_oh = np.zeros(V_in, dtype=np.float32); cal_oh[cal_dim] = 1.0
    cal_off = offbridge_layer_rates(Ws, thresholds, leaks, cal_oh, T, n_blocks)
    off_active = [int((r > 0).sum()) for r in cal_off]
    print(f"[ml-signed] calibration char dim={cal_dim}: off-bridge active/layer={off_active}")

    sweep = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0]
    best_gain, best_score, sweep_log = None, -2.0, []
    for g in sweep:
        bridge, cfg = build_multilayer_signed_bridge(Ws, layout, seed=42, e_gain=g)
        on_block = onbridge_block_output_rates(
            bridge, cfg, layout, active_input_dims=[cal_dim],
            drive_pA=drive_pA, n_steps=n_steps, warmup=warmup)
        on_active = [int((r > 0).sum()) for r in on_block]
        # per-block spearman (on_block[li] aligned 1:1 with off block li); final = last.
        sp_blocks = [spearman(cal_off[li], on_block[li]) for li in range(n_blocks)]
        sp_final = sp_blocks[n_blocks - 1]
        sweep_log.append({
            "gain": g, "on_active": on_active,
            "spearman_final": (None if math.isnan(sp_final) else round(sp_final, 4)),
            "spearman_per_block": [None if math.isnan(s) else round(s, 4) for s in sp_blocks],
        })
        print(f"[ml-signed]   gain={g:7.2f} -> on_active/block={on_active} "
              f"sp_final={'nan' if math.isnan(sp_final) else f'{sp_final:.3f}'} "
              f"sp_blocks={[None if math.isnan(s) else round(s,2) for s in sp_blocks]}")
        # 'best' = the final block active in a live regime (>=10% of off target) with the
        # best final spearman.
        target_final = max(1, off_active[n_blocks - 1])
        live = on_active[n_blocks - 1] >= max(1, int(target_final * 0.1))
        sc = sp_final if not math.isnan(sp_final) else -2.0
        if (on_active[n_blocks - 1] > 0) and ((best_gain is None) or (live and sc > best_score)):
            best_gain, best_score = g, sc
    if best_gain is None:
        best_gain = sweep[-1]
    print(f"[ml-signed] locked GLOBAL gain={best_gain} (cal final spearman={best_score:.3f})")

    # =====================================================================
    # PHASE 2 — per-layer threshold-balance refinement (the scoping's named cheap fix),
    # done PROPERLY as a GREEDY per-block bisection on the incoming weight scale: for each
    # block L (in order), holding earlier blocks fixed at their calibrated scale, bisect
    # per_layer_scale[L] so block-L's on-bridge active-count matches the off-bridge target
    # (within tolerance). This is the principled "per-layer threshold-balance" the scoping
    # names — it directly tests whether re-balancing each stage recovers multi-layer
    # fidelity, or whether the collapse is fundamental. Calibrated on the 'e' char; the
    # scale applies to BOTH the E and I incoming wiring of block L (preserving the signed
    # E/I balance). NO sim/ edit (pure weight scaling).
    # =====================================================================
    per_layer_scale = [1.0] * n_blocks
    cal_log = []

    def on_block_with_scales(scales):
        b, c = build_multilayer_signed_bridge(
            Ws, layout, seed=42, e_gain=best_gain,
            per_layer_e_gain=scales, per_layer_i_gain=scales)
        return onbridge_block_output_rates(
            b, c, layout, active_input_dims=[cal_dim],
            drive_pA=drive_pA, n_steps=n_steps, warmup=warmup)

    for L in range(n_blocks):
        off_a = max(1, off_active[L])
        lo, hi = 0.02, 2.0   # search the incoming-weight scale for block L
        chosen = per_layer_scale[L]
        best_err = None
        # 6 bisection probes per block (cheap; each is one bridge build + 80 steps).
        for _it in range(6):
            mid = math.sqrt(lo * hi)  # geometric bisection (scale is multiplicative)
            trial = list(per_layer_scale); trial[L] = mid
            on_b = on_block_with_scales(trial)
            on_a = int((on_b[L] > 0).sum())
            err = on_a - off_a
            cal_log.append({"block": L, "scale": round(mid, 4), "on_active": on_a,
                            "off_active": off_a})
            if best_err is None or abs(err) < abs(best_err):
                best_err, chosen = err, mid
            if on_a > off_a:     # too many active -> reduce drive
                hi = mid
            else:                # too few active -> raise drive
                lo = mid
            if abs(err) <= max(5, int(0.05 * off_a)):
                break
        per_layer_scale[L] = chosen
        print(f"[ml-signed] block {L} threshold-balance: scale={chosen:.4f} "
              f"-> on_active~{off_a + (best_err or 0)} (target off={off_a})")

    refine = any(abs(s - 1.0) > 1e-6 for s in per_layer_scale)
    per_layer_e = per_layer_scale if refine else None
    per_layer_i = per_layer_scale if refine else None
    print(f"[ml-signed] locked per-layer threshold-balance scales={[round(s,4) for s in per_layer_scale]}")

    def make_bridge():
        return build_multilayer_signed_bridge(
            Ws, layout, seed=42, e_gain=best_gain,
            per_layer_e_gain=per_layer_e, per_layer_i_gain=per_layer_i)

    # =====================================================================
    # PHASE 3 — per-input per-layer + cumulative fidelity at the locked config.
    # =====================================================================
    per_input = []
    on_block_by_char = {}  # dim -> list of n_blocks on-bridge block-output rates (anti-cheat)
    off_rates_by_char = {}
    # SATURATION diagnostic: per-block mean/max firing RATE, on-bridge vs off-bridge. The
    # load-bearing Q3 evidence -- if a hidden block's on-bridge mean rate pins at the
    # refractory ceiling (0.5 at refrac=1/dt=1ms) while off-bridge is graded (~0.4, max 1.0),
    # the rank info is destroyed by dense-fan-in saturation (the rate-code wall), which the
    # cheap threshold-balance + I/E-ratio calibration cannot fix.
    sat_rate = [{"on_mean": [], "on_max": [], "off_mean": [], "off_max": []}
                for _ in range(n_blocks)]
    for di, dim in enumerate(probe_dims):
        oh = np.zeros(V_in, dtype=np.float32); oh[dim] = 1.0
        off_rates = offbridge_layer_rates(Ws, thresholds, leaks, oh, T, n_blocks)
        bridge, cfg = make_bridge()
        on_block = onbridge_block_output_rates(
            bridge, cfg, layout, active_input_dims=[dim],
            drive_pA=drive_pA, n_steps=n_steps, warmup=warmup)
        off_rates_by_char[dim] = off_rates
        on_block_by_char[dim] = on_block
        for li in range(n_blocks):
            sat_rate[li]["on_mean"].append(float(on_block[li].mean()))
            sat_rate[li]["on_max"].append(float(on_block[li].max()))
            sat_rate[li]["off_mean"].append(float(off_rates[li].mean()))
            sat_rate[li]["off_max"].append(float(off_rates[li].max()))

        # per-block metrics: on_block[li] aligned 1:1 with off block li (forward_unroll);
        # the final block (n_blocks-1) is the cumulative endpoint -> cumulative fidelity.
        layer_metrics = []
        for li in range(n_blocks):
            off_r = off_rates[li]
            on_r = on_block[li]
            n_off = int((off_r > 0).sum()); n_on = int((on_r > 0).sum())
            pe = pearson(off_r, on_r); sp = spearman(off_r, on_r)
            k = max(10, n_off)
            ov = topk_overlap(off_r, on_r, k)
            layer_metrics.append({
                "layer": li, "off_active": n_off, "on_active": n_on,
                "pearson": pe, "spearman": sp, "topk_overlap": ov, "k": int(k),
            })
        cumulative_sp = layer_metrics[-1]["spearman"]  # final-stage = cumulative fidelity
        per_input.append({
            "char": (vocab[dim] if vocab else None), "dim": int(dim),
            "layers": layer_metrics, "cumulative_spearman": cumulative_sp,
        })
        ls = " ".join(
            "L%d:sp=%s,on=%d" % (
                m["layer"],
                ("nan" if math.isnan(m["spearman"]) else "%.2f" % m["spearman"]),
                m["on_active"],
            ) for m in layer_metrics
        )
        print(f"[ml-signed] char={per_input[-1]['char']!r} dim={dim:2d} | {ls} | "
              f"CUMUL sp={'nan' if math.isnan(cumulative_sp) else f'{cumulative_sp:.3f}'}")

    # =====================================================================
    # ANTI-CHEAT — cross-input specificity on the FINAL (cumulative) stage.
    # The on-bridge final-stage rate for char X must track off char X SPECIFICALLY,
    # not a generic deep-firing pattern. matched (diagonal) >> mismatched (off-diag).
    # =====================================================================
    dims = list(on_block_by_char.keys())
    matched_sp, mismatched_sp = [], []
    for d_on in dims:
        on_final = on_block_by_char[d_on][n_blocks - 1]  # final block output
        for d_off in dims:
            off_final = off_rates_by_char[d_off][n_blocks - 1]
            s = spearman(off_final, on_final)
            if math.isnan(s):
                continue
            (matched_sp if d_on == d_off else mismatched_sp).append(s)
    spec = {
        "matched_mean_spearman": float(np.mean(matched_sp)) if matched_sp else float("nan"),
        "mismatched_mean_spearman": float(np.mean(mismatched_sp)) if mismatched_sp else float("nan"),
    }
    spec["specificity_margin"] = spec["matched_mean_spearman"] - spec["mismatched_mean_spearman"]
    print(f"[ml-signed] ANTI-CHEAT specificity (final stage): matched="
          f"{spec['matched_mean_spearman']:.3f} mismatched={spec['mismatched_mean_spearman']:.3f} "
          f"margin={spec['specificity_margin']:.3f}")

    # =====================================================================
    # Aggregate per-layer + cumulative fidelity.
    # =====================================================================
    per_layer_agg = []
    for li in range(n_blocks):
        sps = [r["layers"][li]["spearman"] for r in per_input
               if not math.isnan(r["layers"][li]["spearman"])]
        pes = [r["layers"][li]["pearson"] for r in per_input
               if not math.isnan(r["layers"][li]["pearson"])]
        ovs = [r["layers"][li]["topk_overlap"] for r in per_input]
        per_layer_agg.append({
            "layer": li,
            "mean_spearman": float(np.mean(sps)) if sps else float("nan"),
            "mean_pearson": float(np.mean(pes)) if pes else float("nan"),
            "mean_topk_overlap": float(np.mean(ovs)) if ovs else float("nan"),
            "on_mean_rate": float(np.mean(sat_rate[li]["on_mean"])),
            "on_max_rate": float(np.mean(sat_rate[li]["on_max"])),
            "off_mean_rate": float(np.mean(sat_rate[li]["off_mean"])),
            "off_max_rate": float(np.mean(sat_rate[li]["off_max"])),
        })
    # Print the saturation diagnostic (the load-bearing Q3 evidence).
    print("[ml-signed] SATURATION diagnostic (on vs off firing rate per block):")
    for a in per_layer_agg:
        print("[ml-signed]   block %d: on_mean=%.3f on_max=%.3f | off_mean=%.3f off_max=%.3f%s"
              % (a["layer"], a["on_mean_rate"], a["on_max_rate"], a["off_mean_rate"],
                 a["off_max_rate"],
                 "  <- SATURATED to refractory ceiling" if a["on_max_rate"] <= 0.51
                 and a["on_mean_rate"] >= 0.49 else ""))
    cumul_sps = [r["cumulative_spearman"] for r in per_input
                 if not math.isnan(r["cumulative_spearman"])]
    cumulative_mean_spearman = float(np.mean(cumul_sps)) if cumul_sps else float("nan")
    on_active_final_any = any(r["layers"][-1]["on_active"] > 0 for r in per_input)

    # =====================================================================
    # Verdict (per scoping ladder step #1):
    #   GO = cumulative multi-layer fidelity >= ~0.8 WITH signed weights AND a large
    #        specificity margin AND every stage spikes -> multi-layer + signed are
    #        NO-sim/-edit-solved.
    #   PARTIAL = spikes + above-chance but cumulative < 0.8 (per-layer threshold-balance
    #             or surrogate-finetune the fallback).
    #   NEGATIVE / honest fidelity-wall = final stage silent OR cumulative collapses to
    #             noise (the key Q3 error-accumulation finding).
    # =====================================================================
    margin_ok = (not math.isnan(spec["specificity_margin"]) and
                 spec["specificity_margin"] > 0.1)
    GO_BAR = 0.8
    if not on_active_final_any:
        verdict = "NEGATIVE"
    elif (not math.isnan(cumulative_mean_spearman) and
          cumulative_mean_spearman >= GO_BAR and margin_ok):
        verdict = "GO"
    elif (not math.isnan(cumulative_mean_spearman) and
          cumulative_mean_spearman >= 0.4 and margin_ok):
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    verdict_line = (
        "ml-signed: blocks=%d signed_EI cumulative_spearman=%.3f "
        "per_layer_spearman=%s specificity_margin=%.3f final_spikes=%s -> %s" % (
            n_blocks, cumulative_mean_spearman,
            [None if math.isnan(a["mean_spearman"]) else round(a["mean_spearman"], 3)
             for a in per_layer_agg],
            spec["specificity_margin"], ("Y" if on_active_final_any else "N"), verdict,
        )
    )

    result = {
        "probe": "genseq_loopstep3_multilayer_signed",
        "artifact": str(NPZ_PATH.name),
        "n_blocks": n_blocks,
        "feature_sizes": feature_sizes,
        "n_total_neurons": layout.n_total,
        "neuron_model": "ADEX_as_LIF (signed E/I split-channel)",
        "signed_method": "E/I split-channel (each source feature duplicated into an "
                         "excitatory + inhibitory copy; W>0 from E-copy -> g_e, W<0 from "
                         "I-copy -> g_i; reuse-only, NO sim/ edit)",
        "drive_pA": drive_pA, "T_off": T, "n_steps_on": n_steps, "warmup": warmup,
        "global_gain_sweep": sweep_log,
        "locked_global_gain": best_gain,
        "per_layer_threshold_balance": per_layer_scale,
        "per_layer_balance_calibration_log": cal_log,
        "off_active_per_layer_cal": off_active,
        "per_layer_fidelity": per_layer_agg,
        "cumulative_mean_spearman": cumulative_mean_spearman,
        "anti_cheat_specificity": spec,
        "per_input": per_input,
        "final_stage_spikes": on_active_final_any,
        "go_bar": GO_BAR,
        "verdict_line": verdict_line,
        "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[ml-signed] wrote {OUT_PATH}")
    return result


if __name__ == "__main__":
    main()
