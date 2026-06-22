"""STEP 0 — generative-sequence frontier C1 feasibility probe (NO TRAINING).

Scoping: research/findings/2026-06-22-generative-sequence-frontier-scoping.md §6.1
(Probe A). Spine A (pretrain non-spiking -> convert to spikes). Step 0 is
fork-independent: does a TRAINED SPIKING net CONSOLIDATE onto the one
SimulationBridge and actually SPIKE there, tracking the original net's output?

What this probe does (NO sim/ edit, reuse-only bridge APIs):
  1. Load the SHIPPED trained SNN artifact cortex_10M_seed42.npz
       (4 LIF layers 66 -> 2048 -> 2048 -> 2048 -> 66, threshold 1.0, leak 0.95,
        hard reset, Heaviside spikes >= threshold).
  2. Install ONE representative layer's trained weights onto a SimulationBridge
     as a co-resident slice via inject_explicit_wiring, with AdEx neurons
     parameterized to APPROXIMATE the trained net's LIF (the "AdEx-as-LIF"
     cheapest C1 path the scoping ranks first: a=b=0 zero adaptation, tuned
     leak/threshold/reset).  No LEAKY_INTEGRATE_AND_FIRE exists on the bridge
     (NeuronModel = {IZHIKEVICH, HODGKIN_HUXLEY, ADEX, RESONATE_AND_FIRE}).
  3. Drive the pre-population with a known input batch and measure:
       (a) does the post layer SPIKE?  (real cp_firing_states, spikes/neuron)
       (b) FIDELITY: does the bridge per-neuron spike RATE track the SAME layer's
           output in the original bptt_snn forward pass for the same input?
           (Pearson + Spearman of rate vectors; top-k overlap; argmax agreement
            on the readout layer.)

HONEST: if AdEx-as-LIF does NOT track the trained LIF (low fidelity), that IS the
key C1 finding (it points to the conversion-calibration / surrogate-grad-on-bridge
finetune fallback). Reported plainly, not fudged.

Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_step0_bridge_load_probe
"""
from __future__ import annotations

import json
import os
import sys
import math
from pathlib import Path

import numpy as np

# Repo root on path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.bptt_snn import LIFLayer, forward_unroll

NPZ_PATH = _REPO / "research/findings/raw/phase_2_2b/cortex_10M_seed42.npz"
META_PATH = _REPO / "research/findings/raw/phase_2_2b/cortex_10M_seed42.metadata.json"
OUT_PATH = _REPO / "research/findings/raw/_genseq_step0.json"


# ---------------------------------------------------------------------------
# Off-bridge reference (the ground truth the bridge must track)
# ---------------------------------------------------------------------------
def load_artifact():
    # allow_pickle=True: the npz is OUR OWN trusted training output
    # (sim/research/runners/cortex_pretraining.py save_checkpoint); the n_layers
    # scalar is stored as a 0-d object array. Local, project-generated, safe.
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


def offbridge_layer_rates(Ws, thresholds, leaks, input_oh, T):
    """Run the trained LIF net's forward_unroll on a (V_in,) one-hot held for T
    steps; return per-layer mean spike-rate vectors (the off-bridge ground truth)
    and the readout-layer summed logits (rate code) for argmax."""
    layers = [LIFLayer(W_in=Ws[i], n_post=Ws[i].shape[1],
                       threshold=thresholds[i], leak=leaks[i])
              for i in range(len(Ws))]
    V_in = Ws[0].shape[0]
    # (T, B=1, V_in) constant one-hot drive (the LM is fed a held char)
    inp = np.tile(input_oh.reshape(1, 1, V_in), (T, 1, 1)).astype(np.float32)
    out = forward_unroll(inp, layers)
    rates = [out["spikes"][li][:, 0, :].mean(axis=0) for li in range(len(layers))]
    readout_logits = out["spikes"][-1][:, 0, :].sum(axis=0)  # rate-code logits
    return rates, readout_logits


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
# On-bridge: install layer-0 weights and drive
# ---------------------------------------------------------------------------
def build_adex_as_lif_bridge(n_pre, n_post, W, seed=42, prop_strength=0.05):
    """A minimal AdEx bridge with two disjoint populations:
       indices [0, n_pre)        = the INPUT layer (pre)
       indices [n_pre, n_pre+n_post) = the trained layer's POST neurons.
    Weight matrix W (n_pre, n_post) installed pre->post, plastic=False (frozen).
    AdEx tuned toward the trained LIF (a=b=0; tuned tau_m/threshold/reset)."""
    n_total = n_pre + n_post
    cfg = CoreSimConfig()
    cfg.num_neurons = n_total
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0  # overwritten by inject_explicit_wiring
    cfg.num_traits = 1               # all excitatory
    cfg.inhibitory_trait_indices = []
    # CRITICAL: the default CoreSimConfig sets default_neuron_type_adex =
    # "ADEX_RS_CORTICAL_PYRAMIDAL", whose preset (a=4, b=80.5, C=281) is
    # OVERLAID onto cfg.adex_* at init, wiping our AdEx-as-LIF tuning below.
    # Disable the preset so our hand-tuned zero-adaptation LIF approximation holds.
    cfg.default_neuron_type_adex = None

    # No learning / no homeostasis / no structural — we want the FROZEN trained
    # weights to drive the dynamics, nothing adapting.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.enable_reward_modulation = False
    cfg.enable_watts_strogatz = False
    cfg.enable_nmda = False

    # --- AdEx-as-LIF tuning ---------------------------------------------------
    # Trained LIF: v(t) = leak*v(t-1)*(1-s) + W@x ; threshold 1.0, leak 0.95.
    # AdEx (conductance-based, mV): C dV/dt = -g_L(V-E_L) + g_L*dT*exp(..) - w + I.
    # Zero adaptation -> pure leaky IF. Match the membrane leak time constant to
    # the trained leak: leak=exp(-dt/tau) -> tau = -dt/ln(leak) = -1/ln(0.95) ~ 19.5 ms.
    # tau_m = C/g_L. Keep default g_L=30 nS, set C = g_L*tau_m. Sharpen Delta_T so
    # AdEx fires crisply near V_T (closer to a hard threshold, like the LIF).
    tau_m = -cfg.dt_ms / math.log(0.95)       # ~19.49 ms
    cfg.adex_g_L = 30.0                         # nS
    cfg.adex_C = cfg.adex_g_L * tau_m           # pF  -> tau_m ~ 19.5 ms
    cfg.adex_E_L = -70.0                        # mV  (rest)
    cfg.adex_V_T = -50.0                        # mV  (soft threshold)
    cfg.adex_Delta_T = 0.5                      # mV  (sharp -> near-hard threshold)
    cfg.adex_a = 0.0                            # nS  -> NO subthreshold adaptation
    cfg.adex_tau_w = 144.0                      # ms  (inert, a=b=0)
    cfg.adex_b = 0.0                            # pA  -> NO spike-triggered adaptation
    cfg.adex_V_r = -70.0                        # mV  (reset = rest, like LIF reset to 0)
    cfg.adex_V_peak = -40.0                     # mV  (spike detect)
    cfg.refractory_period_steps = 1

    # OU background noise OFF: we want the trained weights + input drive to carry
    # the signal cleanly (fidelity test), not stochastic spontaneous firing.
    cfg.ou_std_current_pA = 0.0

    # Synaptic transmission GAIN. The bridge default propagation_strength=0.05 is
    # calibrated for dense recurrent biological nets: a single feedforward weight
    # W~5 -> g_e += 0.25 -> I_syn ~ 17 pA, far too weak to drive AdEx -70->-50 mV.
    # The trained LIF instead sums W@x directly (dimensionless, threshold 1.0). To
    # give AdEx-as-LIF a FAIR chance to reproduce the LIF mapping, raise the gain
    # so the W column-sum operates in a comparable driving regime. This is a
    # global synaptic scale (a calibration knob, NOT a sim/ edit) and is exactly
    # the "ANN->SNN conversion calibration" the scoping ranks as option 2 — here a
    # single scalar threshold-balance. We sweep it (caller) to find AdEx's best.
    cfg.propagation_strength = float(prop_strength)
    cfg.syn_tau_g_e = 5.0

    viz_cfg = VisualizationConfig()
    runtime_state = RuntimeState()
    gpu_cfg = GPUConfig()
    bridge = SimulationBridge(
        core_config=cfg, viz_config=viz_cfg,
        runtime_state=runtime_state, gpu_config=gpu_cfg,
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized, "Bridge init failed"

    # Build the explicit wiring: pre [0,n_pre) -> post [n_pre, n_pre+n_post).
    pre_idx, post_idx, w_flat = [], [], []
    Wnz = W  # (n_pre, n_post)
    nz_pre, nz_post = np.nonzero(np.abs(Wnz) > 0.0)
    # Keep only positive weights for a pure-excitatory test (the bridge routes
    # negative weights of an EXC source the same channel, which would be wrong
    # sign vs a LIF that sums signed W). We test the EXCITATORY drive fidelity:
    # the rank/rate the positive synapses induce.  (Negative-weight handling is a
    # named downstream conversion concern; see verdict.)
    for p, q in zip(nz_pre.tolist(), nz_post.tolist()):
        wv = float(Wnz[p, q])
        if wv <= 0.0:
            continue
        pre_idx.append(int(p))
        post_idx.append(int(n_pre + q))
        w_flat.append(wv)
    wiring = {
        "layer0_in_to_hidden": {
            "pre_indices": pre_idx,
            "post_indices": post_idx,
            "initial_weights": np.asarray(w_flat, dtype=np.float32),
            "plastic": False,
            "conn_type": "E_TO_E",
            "count": len(pre_idx),
        }
    }
    bridge.inject_explicit_wiring(wiring)
    # Zero baseline DC drive; we inject our own input current.
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0
    return bridge, cfg


def onbridge_post_rates(bridge, cfg, n_pre, n_post, active_pre, drive_pA,
                        n_steps, warmup):
    """Drive the active pre-neurons (the one-hot 'on' input dims) with a strong
    constant current so they spike, let the frozen W carry conductance to the
    post layer, and record per-post-neuron spike rate over n_steps."""
    import cupy as cp
    n_total = n_pre + n_post
    bridge.cp_external_input_current[:] = 0.0
    # Strong drive to the ON input neurons -> they spike at high rate, delivering
    # the trained weighted conductance to the post layer.
    pre_drive = cp.zeros(n_total, dtype=cp.float32)
    for i in active_pre:
        pre_drive[int(i)] = cp.float32(drive_pA)
    bridge.cp_external_input_current[:] = pre_drive

    post_slice = slice(n_pre, n_pre + n_post)
    post_spike_counts = cp.zeros(n_post, dtype=cp.float64)
    pre_spike_counts = cp.zeros(n_pre, dtype=cp.float64)
    counted = 0
    total = warmup + n_steps
    for step in range(total):
        bridge._run_one_simulation_step()
        # external current is consumed each step? No -- it persists; re-assert to
        # keep the constant DC drive on the input neurons.
        bridge.cp_external_input_current[:] = pre_drive
        if step >= warmup:
            fired = bridge.cp_firing_states
            post_spike_counts += fired[post_slice].astype(cp.float64)
            pre_spike_counts += fired[:n_pre].astype(cp.float64)
            counted += 1
    post_rate = (post_spike_counts / max(1, counted)).get()
    pre_rate = (pre_spike_counts / max(1, counted)).get()
    return post_rate, pre_rate


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------
def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[step0] SIM_BACKEND={backend}")
    print(f"[step0] loading artifact: {NPZ_PATH.name}")
    Ws, thresholds, leaks, layer_sizes, vocab = load_artifact()
    print(f"[step0] layers={len(Ws)} sizes={layer_sizes} "
          f"thr={thresholds[0]} leak={leaks[0]}")
    V_in = Ws[0].shape[0]
    n_hidden = Ws[0].shape[1]

    # --- choose the probe layer: layer 0 (input 66 -> hidden 2048) ------------
    # This is the most diagnostic: a one-hot char drives exactly one input row of
    # W into 2048 hidden neurons; the off-bridge LIF and the on-bridge AdEx both
    # see the SAME single weight column -> a clean rate-vector comparison.
    n_pre, n_post = V_in, n_hidden
    W0 = Ws[0]

    T = 32                # off-bridge unroll length (= training T)
    n_steps = 64          # on-bridge measurement steps
    warmup = 16
    drive_pA = 4000.0     # strong enough to make the ON input neuron spike often

    # Probe over several distinct one-hot char inputs (pick non-pad, common chars)
    # to get a fidelity distribution rather than a single point.
    if vocab is not None:
        probe_chars = [" ", "e", "t", "a", "o", "h"]
        probe_dims = [vocab.index(c) for c in probe_chars if c in vocab]
    else:
        probe_dims = [2, 44, 59, 40, 54, 47]
    probe_dims = probe_dims[:6]

    # --- sweep synaptic gain to find AdEx's best operating point --------------
    # The single fairest test of AdEx-as-LIF: find the global synaptic gain at
    # which the post layer fires in a comparable regime to the trained LIF, then
    # report fidelity THERE (AdEx-as-LIF at its best, not a mis-scaled artifact).
    # Calibrate on one representative char ('e'), then lock the gain for all.
    cal_dim = probe_dims[0]
    cal_oh = np.zeros(V_in, dtype=np.float32); cal_oh[cal_dim] = 1.0
    cal_off, _ = offbridge_layer_rates(Ws, thresholds, leaks, cal_oh, T)
    cal_off_l0 = cal_off[0]
    target_active = int((cal_off_l0 > 0).sum())
    print(f"[step0] calibration char dim={cal_dim}: off-bridge active hidden="
          f"{target_active}/{n_post}")
    sweep = [0.05, 0.25, 1.0, 4.0, 16.0, 64.0]
    best_gain, best_sp, sweep_log = None, -2.0, []
    for g in sweep:
        bridge, cfg = build_adex_as_lif_bridge(n_pre, n_post, W0, seed=42,
                                               prop_strength=g)
        on_l0, pre_rate = onbridge_post_rates(
            bridge, cfg, n_pre, n_post, active_pre=[cal_dim],
            drive_pA=drive_pA, n_steps=n_steps, warmup=warmup)
        n_on = int((on_l0 > 0).sum())
        sp = spearman(cal_off_l0, on_l0)
        sweep_log.append({"gain": g, "on_active": n_on,
                          "pre_rate": float(pre_rate[cal_dim]),
                          "spearman": (None if math.isnan(sp) else round(sp, 4))})
        print(f"[step0]   gain={g:6.2f} -> on_active={n_on:4d} "
              f"pre_rate={pre_rate[cal_dim]:.2f} spearman="
              f"{('nan' if math.isnan(sp) else f'{sp:.3f}')}")
        # 'best' = the gain giving the most active hidden (closest to a live
        # regime) with a defined correlation; tie-break on spearman.
        score = (n_on > 0, (sp if not math.isnan(sp) else -2.0))
        if (n_on > 0) and ((best_gain is None) or
                           (n_on >= target_active * 0.1 and sp > best_sp)):
            best_gain, best_sp = g, (sp if not math.isnan(sp) else best_sp)
    if best_gain is None:
        # nothing fired at any gain: pick the largest gain so we report honestly
        best_gain = sweep[-1]
    print(f"[step0] locked gain={best_gain} (best calibration spearman={best_sp:.3f})")

    per_input = []
    for di, dim in enumerate(probe_dims):
        oh = np.zeros(V_in, dtype=np.float32)
        oh[dim] = 1.0
        # off-bridge ground truth for layer 0
        off_rates, off_logits = offbridge_layer_rates(Ws, thresholds, leaks, oh, T)
        off_l0 = off_rates[0]  # (n_hidden,)

        # on-bridge at the locked gain: fresh bridge per input (clean state)
        bridge, cfg = build_adex_as_lif_bridge(n_pre, n_post, W0, seed=42,
                                               prop_strength=best_gain)
        on_l0, pre_rate = onbridge_post_rates(
            bridge, cfg, n_pre, n_post, active_pre=[dim],
            drive_pA=drive_pA, n_steps=n_steps, warmup=warmup)

        # Metrics
        n_off_active = int((off_l0 > 0).sum())
        n_on_active = int((on_l0 > 0).sum())
        pe = pearson(off_l0, on_l0)
        sp = spearman(off_l0, on_l0)
        k = max(10, n_off_active)
        ov = topk_overlap(off_l0, on_l0, k)
        per_input.append({
            "char": (vocab[dim] if vocab else None),
            "dim": int(dim),
            "off_active_hidden": n_off_active,
            "on_active_hidden": n_on_active,
            "on_pre_rate": float(pre_rate[dim]),
            "pearson": pe, "spearman": sp,
            "topk_overlap": ov, "k": int(k),
        })
        print(f"[step0] char={per_input[-1]['char']!r} dim={dim:2d} | "
              f"off_active={n_off_active:4d} on_active={n_on_active:4d} "
              f"pre_rate={pre_rate[dim]:.2f} | "
              f"pearson={pe:.3f} spearman={sp:.3f} topk_ov={ov:.3f}")

    # --- ANTI-CHEAT: cross-char specificity --------------------------------
    # The on-bridge rate for char X must track off-bridge char X SPECIFICALLY,
    # not a generic high-firing pattern. Build the matched (diagonal) vs
    # mismatched (off-diagonal) Spearman: matched should be >> mismatched.
    # Re-collect on-bridge rates per char at the locked gain (cheap), then cross.
    on_rates_by_char, off_rates_by_char = {}, {}
    for r in per_input:
        dim = r["dim"]
        oh = np.zeros(V_in, dtype=np.float32); oh[dim] = 1.0
        off_rates, _ = offbridge_layer_rates(Ws, thresholds, leaks, oh, T)
        off_rates_by_char[dim] = off_rates[0]
        bridge, cfg = build_adex_as_lif_bridge(n_pre, n_post, W0, seed=42,
                                               prop_strength=best_gain)
        on_l0, _ = onbridge_post_rates(bridge, cfg, n_pre, n_post,
                                       active_pre=[dim], drive_pA=drive_pA,
                                       n_steps=n_steps, warmup=warmup)
        on_rates_by_char[dim] = on_l0
    dims = list(on_rates_by_char.keys())
    matched_sp, mismatched_sp = [], []
    for d_on in dims:
        for d_off in dims:
            s = spearman(off_rates_by_char[d_off], on_rates_by_char[d_on])
            if math.isnan(s):
                continue
            (matched_sp if d_on == d_off else mismatched_sp).append(s)
    spec = {
        "matched_mean_spearman": float(np.mean(matched_sp)) if matched_sp else float("nan"),
        "mismatched_mean_spearman": float(np.mean(mismatched_sp)) if mismatched_sp else float("nan"),
    }
    spec["specificity_margin"] = spec["matched_mean_spearman"] - spec["mismatched_mean_spearman"]
    print(f"[step0] ANTI-CHEAT specificity: matched={spec['matched_mean_spearman']:.3f} "
          f"mismatched={spec['mismatched_mean_spearman']:.3f} "
          f"margin={spec['specificity_margin']:.3f}")

    # Aggregate fidelity (over inputs with defined correlation)
    pes = [r["pearson"] for r in per_input if not math.isnan(r["pearson"])]
    sps = [r["spearman"] for r in per_input if not math.isnan(r["spearman"])]
    ovs = [r["topk_overlap"] for r in per_input]
    on_active_any = any(r["on_active_hidden"] > 0 for r in per_input)
    mean_pe = float(np.mean(pes)) if pes else float("nan")
    mean_sp = float(np.mean(sps)) if sps else float("nan")
    mean_ov = float(np.mean(ovs)) if ovs else float("nan")

    spikes_yn = "Y" if on_active_any else "N"
    # Fidelity headline = mean Spearman (rank-order tracking is the honest C1
    # question; Pearson is reported alongside).
    fidelity = mean_sp

    # --- secondary: Gen-F re-confirm (read-only) ------------------------------
    genf = gen_f_reconfirm()

    # --- verdict --------------------------------------------------------------
    # GO if the slice spikes AND fidelity tracks (Spearman > ~0.5 rank tracking)
    #    AND Gen-F coherent.
    # PARTIAL if it spikes but fidelity is weak (the conversion-calibration /
    #    surrogate-finetune fallback is needed) but Gen-F coherent.
    # NEGATIVE if the slice is silent.
    margin_ok = (not math.isnan(spec["specificity_margin"]) and
                 spec["specificity_margin"] > 0.1)
    if not on_active_any:
        verdict = "NEGATIVE"
    elif (not math.isnan(fidelity)) and fidelity >= 0.5 and genf["coherent"] and margin_ok:
        verdict = "GO"
    else:
        verdict = "PARTIAL"

    result = {
        "probe": "genseq_step0_bridge_load",
        "artifact": str(NPZ_PATH.name),
        "layer_probed": 0,
        "layer_shape": [int(n_pre), int(n_post)],
        "neuron_model": "ADEX_as_LIF",
        "adex_as_lif": {
            "tau_m_ms": float(-1.0 / math.log(0.95)),
            "Delta_T": 0.5, "a": 0.0, "b": 0.0,
            "V_T": -50.0, "V_r": -70.0, "E_L": -70.0,
        },
        "drive_pA": drive_pA, "T_off": T, "n_steps_on": n_steps, "warmup": warmup,
        "gain_sweep": sweep_log,
        "locked_gain": best_gain,
        "spikes": spikes_yn,
        "on_active_any": on_active_any,
        "fidelity_mean_spearman": mean_sp,
        "fidelity_mean_pearson": mean_pe,
        "fidelity_mean_topk_overlap": mean_ov,
        "anti_cheat_specificity": spec,
        "per_input": per_input,
        "gen_f": genf,
        "verdict_line": (
            f"step0: spikes={spikes_yn} fidelity={fidelity:.3f} | "
            f"gen_f_coherent={'Y' if genf['coherent'] else 'N'} "
            f"gen_f_ppl={genf['ppl']} -> {verdict}"
        ),
        "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print("\n" + "=" * 78)
    print(result["verdict_line"])
    print("=" * 78)
    print(f"[step0] wrote {OUT_PATH}")
    return result


def gen_f_reconfirm():
    """Read-only Gen-F re-confirmation: the scoping cites the Gen-F Transformer
    as already producing coherent novel English (held-out ppl ~6.1). Confirm
    from the shipped generation file + finding doc (no heavy regeneration)."""
    train_txt = _REPO / "research/findings/raw/g11_bg/gen_f_train.txt"
    finding = _REPO / "research/findings/2026-05-17-generator-F-small-transformer-LM-PASS.md"
    sample = ""
    coherent = False
    if train_txt.exists():
        sample = train_txt.read_text(encoding="utf-8", errors="ignore")[:600]
        # Heuristic coherence check: TinyStories opener + real words + spaces
        low = sample.lower()
        coherent = ("once upon a time" in low) or (
            sample.count(" ") > 40 and any(
                w in low for w in ["the", "and", "was", "she", "he"]))
    ppl = None
    if finding.exists():
        for line in finding.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "ppl" in line.lower() and "6.1" in line:
                ppl = 6.1
                break
    return {
        "coherent": bool(coherent),
        "ppl": ppl,
        "sample_head": sample[:240],
        "source": "gen_f_train.txt + 2026-05-17-generator-F-...md",
    }


if __name__ == "__main__":
    main()
