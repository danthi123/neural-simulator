"""Phase-B — the RETINAL ESCAPE for the spiking learned cortex, BUILT ON THE BRIDGE.

The rate->spike wall (2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md): the real category structure
needs WHITENING (common-mode removal), but whitening yields a SIGNED low-magnitude differential that rate-
coded spiking (magnitude, non-negative) cannot carry. The RETINA solves exactly this:
  (1) analog center-surround WHITENING -- remove the common mode at the INPUT, pre-spike, full precision;
  (2) ON/OFF cells -- split the signed whitened drive into TWO non-negative spiking populations.
The numpy de-risk (`_phaseB_onoff_whitened_derisk.py`) reaches +0.327/gen 0.77 on the REAL corpus (host
+0.44), clearing the +0.30 structure bar. This runner realizes the mechanism ON the SimulationBridge with
NEURONS + SYNAPSES (no host whitening: the whitening is done by an inhibitory common-mode pool's synaptic
inhibition; the ON/OFF split is two cortex regions) -- per the BRAIN-BASED-ONLY standard. NO sim/ edits.

THE ARCHITECTURE (6 brain-region-framework regions; the genuine retinal ON/OFF dual pathway):
  hub_e  (EXC, input layer)   -- receives the encoded drive (PPMI/log); drives cortex_on's g_e + BOTH cm pools.
  hub_i  (INH, input layer)   -- identical drive; provides cortex_off's INHIBITION (the negated excitation).
  cm_i   (INH common-mode pool)-- excited densely+uniformly by hub_e => fires ~ the POPULATION MEAN (the
                                  common mode); INHIBITS cortex_on with weights matched to the hub_e->cortex_on
                                  row-sums => cortex_on's analog drive = W_on@drive - popmean*rowsum_on = the
                                  WHITENED (axis-1 / population-mean-subtracted) drive. (probe #2 proved the
                                  MATCHED weight is what makes the pool whiten where random/uniform failed.)
  cm_e   (EXC common-mode pool)-- excited densely+uniformly by hub_e => fires ~ popmean; EXCITES cortex_off
                                  with matched weights => cortex_off's analog drive = popmean*rowsum_off -
                                  W_off@drive = the NEGATED whitened drive.
  cortex_on  (EXC) -- fires on the POSITIVE whitened drive (ON cells).
  cortex_off (EXC) -- fires on the NEGATIVE whitened drive (OFF cells).
  concept code = concatenate(cortex_on spike counts, cortex_off spike counts) over the readout window.

WHY MATCHED weights (cm->cortex weight proportional to the hub->cortex row-sum): whitening at the INPUT is
W@(x - m*1) = W@x - m*(W@1), so each cortex neuron j must receive inhibition m*rowsum_j(W). The prior cm-pool
(CYCLE 61) used RANDOM/uniform cm->cortex weights -> subtracted the wrong direction -> failed. We install the
matched weights post-build via bridge.set_pathway_weights (the documented Gabor-preinit API; NOT a sim/ edit).

Incremental build / gate (each a numpy smoke first; see the sibling probes):
  Step 1  whitening front-end: read cortex_on's ANALOG drive (g_e - g_i) and confirm its Pearson(cos,S_true)
          matches the host-whitened drive (the NEURAL whitening ~= the host whitening).
  Step 2  ON/OFF cortex: the concat(ON,OFF) spike code carries the SIGNED whitened structure.
  Step 3  high spike budget + GATE on real: ON/OFF Pearson >= +0.30, generalizes (~0.77), beats a POINT
          control (single pop, no whitening/ON-OFF) by >= +0.10, permuted ~0. Match the numpy +0.33/gen 0.77.

Run (CPU smoke):  SIM_BACKEND=numpy python -u -m research.runners._phaseB_retinal_cortex --smoke
Run (real GATE):  SIM_BACKEND=cupy  python -u -m research.runners._phaseB_retinal_cortex --real --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners.spiking_sm_cortex import encode_drive, _step_with_time  # noqa: E402


# ===========================================================================
# Builder: the 6-region retinal ON/OFF bridge.
# ===========================================================================
def build_retinal_bridge(
    n_hub,
    n_cortex,
    seed,
    *,
    hub_to_cortex_density=0.5,
    w_on_mean=400.0,
    w_off_mean=400.0,
    w_jitter=0.0,
    cm_size=40,
    hub_to_cm_weight=1.0,
    cm_to_cortex_scale=1.0,
    sigma=0.05,
    alpha=0.05,
    enable_homeostasis=False,
    enable_divisive_gain=False,
    stdp_w_max=2000.0,
    enable_whitening=True,
):
    """Build the retinal ON/OFF cortex bridge. Returns (bridge, idx_dict, meta).

    idx_dict has keys: hub_e, hub_i, cm_i, cm_e, cortex_on, cortex_off (each a numpy int index array).
    meta has the per-pathway (pre, post, weight) info needed for the matched cm->cortex install + the
    host-side reference projection (W_on, W_off as DENSE [n_cortex x n_hub] matrices reconstructed from the
    actual CSR, so the host whitening reference uses the SAME random projection the bridge uses).

    enable_whitening=False builds the same regions but ZEROES the cm->cortex inhibition/excitation (an
    ablation: the POINT/no-whitening control with the SAME ON/OFF readout machinery).
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="hub_e", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="hub_i", n_neurons=n_hub, exc_fraction=0.0, internal_density=0.0,
                    plastic_internal=False),
        # the common-mode pools are INTERNEURONS: FS (non-adapting, linear f-I) so their firing tracks the
        # pooled common-mode drive LINEARLY (an RS neuron's spike-frequency adaptation made cm ANTI-track
        # popmean). cm_i is inhibitory (the center-surround); cm_e excites the OFF pathway.
        BrainRegion(name="cm_i", n_neurons=int(cm_size), exc_fraction=0.0, internal_density=0.0,
                    plastic_internal=False, izh_neuron_type="IZH2007_FS_CORTICAL_INTERNEURON"),
        BrainRegion(name="cm_e", n_neurons=int(cm_size), exc_fraction=1.0, internal_density=0.0,
                    plastic_internal=False, izh_neuron_type="IZH2007_FS_CORTICAL_INTERNEURON"),
        BrainRegion(name="cortex_on", n_neurons=n_cortex, exc_fraction=1.0, internal_density=0.0,
                    weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="cortex_off", n_neurons=n_cortex, exc_fraction=1.0, internal_density=0.0,
                    weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = [
        # the random projections W_on / W_off (the cortex receptive fields). Plastic-tagged so a later
        # step could LEARN them; default frozen-after-build for the random-projection gate.
        RegionPathway(from_region="hub_e", to_region="cortex_on", density=hub_to_cortex_density,
                      weight_mean=w_on_mean, weight_jitter=w_jitter, plastic=True,
                      plasticity_gate="hub_on"),
        RegionPathway(from_region="hub_i", to_region="cortex_off", density=hub_to_cortex_density,
                      weight_mean=w_off_mean, weight_jitter=w_jitter, plastic=True,
                      plasticity_gate="hub_off"),
        # the common-mode pools: hub_e drives BOTH (dense uniform) so each fires ~ pooled hub drive = popmean.
        RegionPathway(from_region="hub_e", to_region="cm_i", density=1.0,
                      weight_mean=float(hub_to_cm_weight), weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="hub_e", to_region="cm_e", density=1.0,
                      weight_mean=float(hub_to_cm_weight), weight_jitter=0.0, plastic=False),
        # the WHITENING subtractions: cm_i INHIBITS cortex_on, cm_e EXCITES cortex_off. Dense; weights are
        # OVERWRITTEN post-build with the matched (rowsum) pattern. plastic=False (fixed feedforward circuit).
        RegionPathway(from_region="cm_i", to_region="cortex_on", density=1.0,
                      weight_mean=1.0, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="cm_e", to_region="cortex_off", density=1.0,
                      weight_mean=1.0, weight_jitter=0.0, plastic=False),
    ]

    cfg.dt = 1.0
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    # The dendritic divisive (/marginal) gain SUPPRESSES high-frequency common drive -- which is exactly
    # the COMMON MODE the cm pool must track. With it ON, the cm pool ANTI-correlates with popmean (the
    # gain suppresses the cm pool's input most when the common mode is largest). So this retinal build
    # turns it OFF (default): the whitening is the cm-pool LATERAL INHIBITION (the retina's center-surround),
    # NOT the synaptic divisive gain. Matches the numpy reference (W@(Xn-mean), no divisive gain).
    cfg.enable_dendritic_divisive_gain = bool(enable_divisive_gain)
    cfg.dendritic_divisive_sigma = sigma
    cfg.dendritic_gain_ema_alpha = alpha
    cfg.enable_stdp = True
    cfg.enable_homeostasis = bool(enable_homeostasis)
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = float(stdp_w_max)

    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()

    idx = {r.name: np.asarray(bridge.region_manager.indices(r.name)) for r in cfg.brain_regions}

    # --- install the MATCHED cm->cortex weights (the load-bearing step from probe #2). ---
    # Reconstruct the actual hub->cortex CSR row-sums per cortex neuron, then set the (single) cm pool's
    # per-cortex-neuron weight = cm_to_cortex_scale * rowsum_j, so the pooled inhibition/excitation matches
    # m * rowsum_j(W) = the whitening subtraction. cm pool fires ~ popmean (pooled), so the per-neuron weight
    # carries the rowsum_j; the pool size just averages noise.
    from sim.backend import to_host
    csr = bridge.cp_connections

    def _rowsums(pre_idx, post_idx):
        sub = csr[np.asarray(pre_idx), :][:, np.asarray(post_idx)]
        d = np.asarray(to_host(sub.todense())) if hasattr(sub, "todense") else np.asarray(to_host(sub))
        return d.sum(0)  # per post neuron: sum over pre = rowsum_j(W)

    rs_on = _rowsums(idx["hub_e"], idx["cortex_on"])    # [n_cortex]
    rs_off = _rowsums(idx["hub_e"], idx["cortex_off"])  # (uses hub_e rowsums; W_off lives on hub_i->cortex_off)
    rs_off = _rowsums(idx["hub_i"], idx["cortex_off"])

    if enable_whitening:
        # cm_i -> cortex_on: weight[cm, j] = scale * rs_on[j] / cm_size (so SUM over cm pool = scale*rs_on[j]).
        cm_i, cm_e = idx["cm_i"], idx["cm_e"]
        n_cm = cm_i.size
        for (cm_pool, post_idx, rs) in [(cm_i, idx["cortex_on"], rs_on), (cm_e, idx["cortex_off"], rs_off)]:
            pre_list = np.repeat(cm_pool, post_idx.size)
            post_list = np.tile(post_idx, cm_pool.size)
            wvec = np.tile((cm_to_cortex_scale * rs / max(1, n_cm)).astype(np.float32), cm_pool.size)
            bridge.set_pathway_weights("cm_match", pre_list, post_list, wvec, add_missing=False)
    else:
        # ablation: zero the cm->cortex weights (no whitening; pure ON/OFF of the raw drive).
        for cm_pool, post_idx in [(idx["cm_i"], idx["cortex_on"]), (idx["cm_e"], idx["cortex_off"])]:
            pre_list = np.repeat(cm_pool, post_idx.size)
            post_list = np.tile(post_idx, cm_pool.size)
            bridge.set_pathway_weights("cm_zero", pre_list, post_list,
                                       np.zeros(pre_list.size, np.float32), add_missing=False)

    meta = {"rs_on": rs_on, "rs_off": rs_off}
    return bridge, idx, meta


# ===========================================================================
# Drive + readout helpers (mirror spiking_sm_cortex but for the dual hub_e/hub_i input).
# ===========================================================================
def _set_drive(bridge, idx, drive_row, drive_scale, cm_bias_pA=0.0):
    """Set the encoded drive on BOTH hub_e and hub_i (identical), zero elsewhere.

    When ``cm_bias_pA > 0``, ALSO add a UNIFORM tonic depolarizing current to BOTH cm pools (cm_i, cm_e)
    so they sit near threshold and fire SENSITIVELY in proportion to the pooled common-mode hub drive (a
    tonically-active interneuron). The same flat value for every cm neuron and every concept => carries NO
    per-concept info (concept-agnostic; legitimate -- the only per-concept signal is the hub drive)."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    drive = (np.asarray(drive_row, np.float64) * float(drive_scale)).astype(np.float32)
    bridge.cp_external_input_current[:] = 0.0
    is_cupy = type(bridge.cp_external_input_current).__module__.startswith("cupy")
    for region in ("hub_e", "hub_i"):
        hi = np.asarray(idx[region])
        if is_cupy:
            bridge.cp_external_input_current[xp.asarray(hi)] = xp.asarray(drive)
        else:
            bridge.cp_external_input_current[hi] = drive
    if cm_bias_pA:
        for region in ("cm_i", "cm_e"):
            ci = np.asarray(idx[region])
            if is_cupy:
                bridge.cp_external_input_current[xp.asarray(ci)] += float(cm_bias_pA)
            else:
                bridge.cp_external_input_current[ci] += float(cm_bias_pA)


def _freeze_all_gates(bridge, value):
    for g in list(getattr(bridge, "_plasticity_gate_values", {}).keys()):
        bridge.set_plasticity_gate(g, value)


def read_onoff_codes(bridge, idx, C_drive, *, drive_scale, window, settle, cm_bias_pA=0.0):
    """Per-concept concat(cortex_on, cortex_off) SPIKE-COUNT code, plasticity frozen.

    Accumulates the cortex spike counts ON-DEVICE (one D->H sync per concept, not per step) so window=1000
    on GPU isn't bottlenecked by a per-step host transfer."""
    from sim.backend import get_backend, to_host
    xp, _ = get_backend()
    on_idx = np.asarray(idx["cortex_on"]); off_idx = np.asarray(idx["cortex_off"])
    co_idx = np.concatenate([on_idx, off_idx])
    is_cupy = type(bridge.cp_external_input_current).__module__.startswith("cupy")
    co_dev = xp.asarray(co_idx) if is_cupy else co_idx
    Nc = int(np.asarray(C_drive).shape[0])
    codes = np.zeros((Nc, co_idx.size), dtype=np.float64)
    _freeze_all_gates(bridge, 0.0)
    try:
        for i in range(Nc):
            _set_drive(bridge, idx, C_drive[i], drive_scale, cm_bias_pA=cm_bias_pA)
            acc = xp.zeros(co_idx.size, dtype=xp.float32)
            for t in range(int(settle) + int(window)):
                _step_with_time(bridge)
                if t >= int(settle):
                    acc += bridge.cp_firing_states[co_dev].astype(xp.float32)
            codes[i] = np.asarray(to_host(acc)).astype(np.float64)
            bridge.cp_external_input_current[:] = 0.0
    finally:
        _freeze_all_gates(bridge, 1.0)
    return codes


def read_analog_whitened(bridge, idx, C_drive, *, drive_scale, window, settle, gi_scale=1.0, cm_bias_pA=0.0):
    """Per-concept cortex_on ANALOG signed drive (g_e - gi_scale*g_i), plasticity frozen.

    This is the WHITENING front-end readout (Step 1): cortex_on receives hub_e excitation (g_e) MINUS the
    cm_i common-mode inhibition (g_i). g_e - g_i is the analog signed whitened drive, BEFORE the spiking
    threshold splits it into ON/OFF. Confirm its structure matches the host-whitened drive."""
    from sim.backend import get_backend, to_host
    xp, _ = get_backend()
    on_idx = np.asarray(idx["cortex_on"])
    is_cupy = type(bridge.cp_external_input_current).__module__.startswith("cupy")
    on_dev = xp.asarray(on_idx) if is_cupy else on_idx
    Nc = int(np.asarray(C_drive).shape[0])
    codes = np.zeros((Nc, on_idx.size), dtype=np.float64)
    _freeze_all_gates(bridge, 0.0)
    try:
        for i in range(Nc):
            _set_drive(bridge, idx, C_drive[i], drive_scale, cm_bias_pA=cm_bias_pA)
            acc = xp.zeros(on_idx.size, dtype=xp.float32)
            for t in range(int(settle) + int(window)):
                _step_with_time(bridge)
                if t >= int(settle):
                    acc += (bridge.cp_conductance_g_e[on_dev]
                            - xp.float32(gi_scale) * bridge.cp_conductance_g_i[on_dev])
            codes[i] = np.asarray(to_host(acc)).astype(np.float64) / max(1, int(window))
            bridge.cp_external_input_current[:] = 0.0
    finally:
        _freeze_all_gates(bridge, 1.0)
    return codes


# ===========================================================================
# Host reference: the numpy whitening using the SAME random projection the bridge built.
# ===========================================================================
def host_whitened_drive(bridge, idx, X):
    """W_on @ (Xn - popmean) using the bridge's ACTUAL hub_e->cortex_on weights (dense from the CSR).

    The Step-1 target: the neural g_e-g_i should track THIS (the host doing the identical whitened
    projection with the same W). Returns [Nc x n_cortex]."""
    from sim.backend import to_host
    csr = bridge.cp_connections
    he, on = np.asarray(idx["hub_e"]), np.asarray(idx["cortex_on"])
    sub = csr[he, :][:, on]
    Won = (np.asarray(to_host(sub.todense())) if hasattr(sub, "todense") else np.asarray(to_host(sub))).T  # [n_cortex x n_hub]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xw = Xn - Xn.mean(1, keepdims=True)
    return (Won @ Xw.T).T


# ===========================================================================
# Gate runner.
# ===========================================================================
def _p(name, codes, S_true, labels, ref=None):
    p = _pearson_vs_Strue(_cos_sim(codes), S_true)
    g, ch = heldout_generalization(codes, labels)
    sil = float(np.mean(codes.sum(1) == 0))
    extra = ""
    if ref is not None:
        extra = f"  beats-ctrl={p - ref:+.3f}"
    print(f"  [{name:24s}] Pearson={p:+.3f}  gen={g:.3f} (chance {ch:.3f})  silent={sil:.2f}  "
          f"eff-rank={effective_rank(codes):.1f}{extra}", flush=True)
    return p, g, ch


def run_seed(seed, args, C, labels, S_true):
    C = np.asarray(C, np.float64); labels = np.asarray(labels)
    C_drive = encode_drive(C)  # log1p input (the Weber-Fechner compression)
    n_hub = C.shape[1]
    X = ppmi_matrix(C, 0.75) if args.ppmi_input else C_drive
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    print(f"\n[retinal cortex seed {seed}] {C.shape[0]}c x {n_hub}h; host PPMI+SVD={host_p:+.3f}; "
          f"DENSE (w{args.w_mean}, homeo off, ds{args.drive_scale}, win{args.window}, cm{args.cm_size})",
          flush=True)

    bp = dict(n_hub=n_hub, n_cortex=args.n_cortex, hub_to_cortex_density=args.density,
              w_on_mean=args.w_mean, w_off_mean=args.w_mean, cm_size=args.cm_size,
              hub_to_cm_weight=args.hub_to_cm_weight, cm_to_cortex_scale=args.cm_scale,
              enable_homeostasis=False)
    rp = dict(drive_scale=args.drive_scale, window=args.window, settle=8, cm_bias_pA=args.cm_bias_pA)

    t0 = time.time()
    # --- the WHITENED retinal bridge ---
    bW, idxW, _ = build_retinal_bridge(seed=seed, enable_whitening=True, **bp)
    if args.input_for_drive == "ppmi":
        drive_in = ppmi_matrix(C, 0.75)
    else:
        drive_in = C_drive

    # Step 1: the analog whitening front-end vs the host whitening reference.
    if args.step1:
        ga = read_analog_whitened(bW, idxW, drive_in, gi_scale=args.gi_scale, **rp)
        hostw = host_whitened_drive(bW, idxW, drive_in)
        p_neural = _pearson_vs_Strue(_cos_sim(ga), S_true)
        p_host = _pearson_vs_Strue(_cos_sim(hostw), S_true)
        # alignment: cosine between the neural and host whitened code similarity-matrices (off-diagonal)
        from numpy import triu_indices
        iu = triu_indices(ga.shape[0], 1)
        sa, sh = _cos_sim(ga)[iu], _cos_sim(hostw)[iu]
        align = float(np.corrcoef(sa, sh)[0, 1]) if (sa.std() > 1e-9 and sh.std() > 1e-9) else 0.0
        print(f"  [STEP1 whitening front-end] neural g_e-g_i Pearson(cos,S)={p_neural:+.3f}  "
              f"host-whitened Pearson={p_host:+.3f}  neural~host align(cos-sim)={align:+.3f}", flush=True)

    # the ON/OFF spike code (Step 2 + Step 3 gate).
    onoff = read_onoff_codes(bW, idxW, drive_in, **rp)
    # the POINT control: same bridge ON/OFF machinery but whitening DISABLED (cm->cortex zeroed) AND read
    # ONLY cortex_on (single population, no ON/OFF split, no whitening) = the genuine point/rate control.
    bP, idxP, _ = build_retinal_bridge(seed=seed, enable_whitening=False, **bp)
    pt = read_onoff_codes(bP, idxP, drive_in, **rp)[:, :args.n_cortex]  # cortex_on only

    p_on, g_on, ch = _p("ON/OFF whitened", onoff, S_true, labels)
    p_pt, g_pt, _ = _p("POINT (no-whiten 1pop)", pt, S_true, labels, ref=None)
    rng = np.random.RandomState(seed * 2718281 + 1); perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_p = _pearson_vs_Strue(_cos_sim(onoff), S_perm)
    print(f"  [anti-cheat] permuted={perm_p:+.3f} (~0); elapsed {time.time() - t0:.0f}s", flush=True)

    gates = {
        "structure": bool(p_on >= max(0.30, 0.70 * host_p)),
        "generalizes": bool(g_on > ch + 0.10),
        "beats_point": bool(p_on >= p_pt + 0.10),
        "permuted_collapses": bool(abs(perm_p) <= 0.15),
        "host_carries": bool(host_p >= 0.30),
    }
    print(f"  gates: {gates}  (whitened {p_on:+.3f}/gen {g_on:.3f} vs point {p_pt:+.3f}/gen {g_pt:.3f}, "
          f"host {host_p:+.3f})", flush=True)
    return {"seed": seed, "host": host_p, "onoff": p_on, "onoff_gen": g_on, "point": p_pt,
            "point_gen": g_pt, "permuted": perm_p, "chance": ch, "gates": gates}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="synthetic 64-concept calibration")
    p.add_argument("--real", action="store_true", help="real TinyStories corpus (the decisive gate)")
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=500)
    p.add_argument("--n-cortex", type=int, default=128)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--w-mean", type=float, default=400.0)
    p.add_argument("--cm-size", type=int, default=40)
    p.add_argument("--hub-to-cm-weight", type=float, default=1.0)
    p.add_argument("--cm-scale", type=float, default=1.0)
    p.add_argument("--drive-scale", type=float, default=40.0)
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--gi-scale", type=float, default=1.0)
    p.add_argument("--cm-bias-pA", type=float, default=0.0,
                   help="tonic depolarizing bias on the cm pools so they fire gradedly ~ popmean")
    p.add_argument("--ppmi-input", action="store_true", help="use PPMI rows for the host-ceiling print")
    p.add_argument("--input-for-drive", default="log", choices=["log", "ppmi"],
                   help="encode the bridge hub drive as log1p counts (default) or PPMI rows")
    p.add_argument("--step1", action="store_true", help="run+print the Step-1 whitening front-end check")
    p.add_argument("--out", default="research/findings/raw/_phaseB_retinal_cortex.json")
    args = p.parse_args()

    if args.smoke:
        Cs, ls, Ss, _ = build_concept_hub_counts(8, 8, 200, 12, 40.0, 4.0, 0.3, 42)
        seeds = [int(s) for s in args.seeds.split(",")]
        res = {str(s): run_seed(s, args, Cs, ls, Ss) for s in seeds}
    elif args.real:
        seeds = [int(s) for s in args.seeds.split(",")]
        res = {}
        for s in seeds:
            C, labels, S_true = build_real_corpus(s, args.n_hub)
            res[str(s)] = run_seed(s, args, C, labels, S_true)
    else:
        print("specify --smoke (synthetic) or --real (TinyStories). Use --step1 to print the front-end check.")
        return

    allgo = all(all(res[k]["gates"].values()) for k in res)
    struct = all(res[k]["gates"]["structure"] for k in res)
    beats = all(res[k]["gates"]["beats_point"] for k in res)
    if allgo:
        verdict = "GO"
    elif struct and not beats:
        verdict = "PARTIAL_structure_but_not_beating_point"
    elif beats and not struct:
        verdict = "PARTIAL_beats_point_below_bar"
    else:
        verdict = "NEGATIVE_or_PARTIAL"
    print(f"\n  RETINAL CORTEX GATE VERDICT: {verdict}", flush=True)
    out = {"verdict": verdict, "per_seed": res, "config": vars(args)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
