"""Phase-2 BRIDGE GATE: does the on-substrate SLOW PER-HUB INPUT-MEAN adaptation primitive recover the
axis-0 per-feature centering on the REAL corpus -- the brain-based DC/diagonal half of whitening?

The validated numpy mechanism (`_phaseB_perhub_adaptation_derisk.py`, 6-seed GO +0.311; spiking-mean D0
+0.298) is: each hub subtracts a SLOW running mean of its OWN input drive, m_h <- (1-a)m_h + a*x_h
(causal/lagged), output = onoff_code(W @ (x - m)). This runner realizes that ON the SimulationBridge with
the guarded default-off INPUT-MEAN primitive (sim/ commit on feat/input-mean-adapt): the hub neurons are
flagged input_mean_adapt=True, so the BRIDGE itself subtracts each hub's slow mean of its own input
CURRENT before the spike threshold (BRAIN-BASED-ONLY -- NOT a host x-mean). The ON/OFF split is the retina's
dual bipolar pathway (the sign-inverting mGluR6 OFF channel = a -x drive to an OFF hub), so the signed
centered drive x-m is carried by two non-negative spiking populations.

ARCHITECTURE (4 brain-region-framework regions; the centering is at the HUB INPUT, axis-0, by the primitive):
  hub_on  (EXC, input_mean_adapt=True) -- driven by +x; after subtracting its own slow mean m_on it fires
                                          ~ relu(x - m) = the ON half of the per-feature-centered drive.
  hub_off (EXC, input_mean_adapt=True) -- driven by -x (the OFF bipolar sign inversion); its own slow mean
                                          is -mean(x), so it fires ~ relu(-x - (-mean)) = relu(mean - x) =
                                          the OFF half.
  cortex_on  (EXC) -- a FIXED random projection W_on from hub_on (the plasticity gate is held CLOSED so
                      the projection does not learn -- this isolates the WHITENING, not the read-out learning).
  cortex_off (EXC) -- a FIXED random projection W_off from hub_off.
  concept code = concatenate(cortex_on spike counts, cortex_off spike counts).

STREAMING protocol (so the per-hub EMA converges to the CROSS-CONCEPT mean, axis-0): present the concepts
SHUFFLED over multiple epochs with adaptation ON (slow alpha). THEN freeze adaptation (alpha -> 0; the mean
stops moving) and read each concept's cortex spike-count code at the converged adaptation state.

GATE on the REAL corpus (build_real_corpus): Pearson(cos, S_true) of the per-hub-adapted ON/OFF code vs:
  - axis-0 SIGNED ceiling  : the numpy per-feature ceiling with a DENSE SIGNED Gaussian projection
                             (W = randn/sqrt(Nh)) -- the validated +0.31, but a per-SYNAPSE signed
                             projection (mixing + and - on the SAME afferent), which is NOT realizable on a
                             Dale's-law spiking substrate (a neuron is all-E or all-I). Context ceiling only.
  - axis-0 SUBSTRATE ceiling: the per-feature ceiling through the BRIDGE's ACTUAL (non-negative excitatory)
                             projection -- what an on-substrate code CAN reach. This is the honest target.
  - axis-1 cm-pool          : per-concept removal (the WRONG axis; the cm-pool's +0.25). The primitive must
                             at least beat this to justify itself over the shipped cm-pool.
  - POINT / no-centering    : adaptation OFF (no flagged region; m stays 0 -> raw drive).
  - permuted-clean          : shuffle labels -> ~0.
Plus: slow-alpha is LOAD-BEARING (a fast-alpha run must do WORSE); the mean is on-substrate (the primitive
computes it -- there is NO host x-mean in the neural path); 6 seeds for any GO claim.

DECISION (honest): the absolute target is marginal (the real category structure is moderate). A GO = the
bridge per-hub-adapted code recovers the SUBSTRATE axis-0 AND clears the bar AND beats cm-pool + point +
permuted. A PARTIAL/NEGATIVE (the spiking projection+readout loses it below bar) is a REAL finding -- report
it; do NOT tune to force green. Note (pre-measured by the host refs): a per-synapse signed projection is
Dale's-law-illegal, so the substrate axis-0 ceiling is far below the numpy signed +0.31; this gate measures
whether the on-substrate code reaches its OWN realizable ceiling and how the primitive compares to the
shipped cm-pool.

Run (CPU smoke):  SIM_BACKEND=numpy python -u -m research.runners._phaseB_input_mean_bridge --smoke --seeds 42
Run (real GATE):  SIM_BACKEND=cupy  python -u -m research.runners._phaseB_input_mean_bridge --real --seeds 42,43,44,45,46,47
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
from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners.spiking_sm_cortex import encode_drive, _step_with_time  # noqa: E402


# ===========================================================================
# Builder: the 4-region per-hub-adapting ON/OFF bridge.
# ===========================================================================
def build_input_mean_bridge(
    n_hub,
    n_cortex,
    seed,
    *,
    hub_to_cortex_density=0.5,
    w_mean=400.0,
    w_jitter=0.0,
    alpha=0.0,          # per-step EMA rate; set >0 to adapt, 0 to freeze. Runner toggles it.
    gain=1.0,
    adapt=True,         # if False, NO region is flagged -> the POINT / no-centering control (m never exists).
    stdp_w_max=2000.0,
    enable_ei=False,    # E/I signed projection: add an INHIBITORY hub->cortex pathway alongside the excitatory
                        # one so cortex drive = g_e(W_exc) - g_i(W_inh) = SIGNED (the de-risked fix for the
                        # excitatory-only projection-sign collapse, _phaseB_projection_sign_derisk.py).
    ei_inh_weight=None, # inhibitory hub->cortex weight (default = w_mean); tune to balance g_e vs g_i.
):
    """Build the per-hub-adapting ON/OFF cortex bridge. Returns (bridge, idx_dict).

    idx_dict keys: hub_on, hub_off, cortex_on, cortex_off.
    The hubs are flagged input_mean_adapt=True (when adapt=True) so the BRIDGE subtracts each hub's slow
    mean of its own input current before the threshold. The hub->cortex projections are FIXED random
    (plasticity gate closed at read time) -- the whitening is the only thing under test.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="hub_on", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.0,
                    input_mean_adapt=bool(adapt)),
        BrainRegion(name="hub_off", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.0,
                    input_mean_adapt=bool(adapt)),
        BrainRegion(name="cortex_on", n_neurons=n_cortex, exc_fraction=1.0, internal_density=0.0,
                    weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="cortex_off", n_neurons=n_cortex, exc_fraction=1.0, internal_density=0.0,
                    weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="hub_on", to_region="cortex_on", density=hub_to_cortex_density,
                      weight_mean=w_mean, weight_jitter=w_jitter, plastic=True, plasticity_gate="proj_on"),
        RegionPathway(from_region="hub_off", to_region="cortex_off", density=hub_to_cortex_density,
                      weight_mean=w_mean, weight_jitter=w_jitter, plastic=True, plasticity_gate="proj_off"),
    ]
    if enable_ei:
        # E/I signed projection: INHIBITORY hub copies (same drive + same input-mean adaptation as the
        # excitatory hubs -> they fire ~ relu(x-m) too) project an INDEPENDENT random inhibitory weight set
        # to the cortex, so each cortex neuron's effective receptive field = W_exc - W_inh = SIGNED. This is
        # the biologically-canonical E/I-balanced random projection that carries the axis-0-centered signal a
        # purely-excitatory (Dale's-law) projection collapses (_phaseB_projection_sign_derisk.py: exc-only
        # +0.04 -> E/I +0.26-0.30). exc_fraction=0.0 = an all-inhibitory population (projects g_i).
        wi = float(w_mean if ei_inh_weight is None else ei_inh_weight)
        cfg.brain_regions += [
            BrainRegion(name="hub_on_inh", n_neurons=n_hub, exc_fraction=0.0, internal_density=0.0,
                        plastic_internal=False, input_mean_adapt=bool(adapt)),
            BrainRegion(name="hub_off_inh", n_neurons=n_hub, exc_fraction=0.0, internal_density=0.0,
                        plastic_internal=False, input_mean_adapt=bool(adapt)),
        ]
        cfg.region_pathways += [
            RegionPathway(from_region="hub_on_inh", to_region="cortex_on", density=hub_to_cortex_density,
                          weight_mean=wi, weight_jitter=w_jitter, plastic=False),
            RegionPathway(from_region="hub_off_inh", to_region="cortex_off", density=hub_to_cortex_density,
                          weight_mean=wi, weight_jitter=w_jitter, plastic=False),
        ]

    cfg.dt = 1.0
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    # the per-hub input-mean primitive (the op under test). alpha is set/toggled by the runner: >0 to adapt
    # during streaming, 0 to FREEZE the converged mean before read-out. gain=1 = subtract the full mean.
    cfg.enable_input_mean_adapt = bool(adapt)
    cfg.input_mean_adapt_alpha = float(alpha)
    cfg.input_mean_adapt_gain = float(gain)
    # NO dendritic divisive gain, NO cm pool, NO homeostasis: the ONLY centering is the input-mean primitive
    # (so the gate measures exactly that op, isolated). Matches the numpy reference W@(x-m), no divisive gain.
    cfg.enable_dendritic_divisive_gain = False
    cfg.enable_stdp = True
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = float(stdp_w_max)

    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    idx = {r.name: np.asarray(bridge.region_manager.indices(r.name)) for r in cfg.brain_regions}
    return bridge, idx


# ===========================================================================
# Drive helpers: hub_on gets +x, hub_off gets -x (the OFF bipolar sign inversion).
# ===========================================================================
def _set_onoff_drive(bridge, idx, drive_row, drive_scale):
    """hub_on <- +drive_row*scale ; hub_off <- -drive_row*scale (the sign-inverting OFF bipolar channel).

    The MEAN SUBTRACTION is NOT done here -- the bridge's input-mean primitive does it on-substrate. Here we
    only render the (signed) sensory input to the ON and OFF hub populations (the retinal dual pathway)."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    drive = (np.asarray(drive_row, np.float64) * float(drive_scale)).astype(np.float32)
    bridge.cp_external_input_current[:] = 0.0
    is_cupy = type(bridge.cp_external_input_current).__module__.startswith("cupy")
    # Every ON hub (hub_on AND, with E/I, hub_on_inh) gets +drive; every OFF hub gets -drive. The inhibitory
    # copies receive the IDENTICAL drive + input-mean adaptation as their excitatory counterparts -> they
    # fire ~ relu(x-m) too, and project g_i (the E/I signed projection).
    for key in ("hub_on", "hub_on_inh"):
        if key in idx:
            ii = np.asarray(idx[key])
            bridge.cp_external_input_current[xp.asarray(ii) if is_cupy else ii] = (
                xp.asarray(drive) if is_cupy else drive)
    for key in ("hub_off", "hub_off_inh"):
        if key in idx:
            ii = np.asarray(idx[key])
            bridge.cp_external_input_current[xp.asarray(ii) if is_cupy else ii] = (
                xp.asarray(-drive) if is_cupy else -drive)


def _freeze_all_gates(bridge, value):
    for g in list(getattr(bridge, "_plasticity_gate_values", {}).keys()):
        bridge.set_plasticity_gate(g, value)


def stream_adapt(bridge, idx, C_drive, *, n_epochs, drive_scale, window, settle, seed,
                 learn_projection=False):
    """STREAM the concepts SHUFFLED over n_epochs with adaptation ON (cfg alpha as set at build).

    For each epoch, a fresh shuffled order; for each concept present its (signed) ON/OFF drive and run
    settle+window steps so the bridge's per-hub input-mean EMA integrates that concept's drive. The EMA is
    the bridge's own state (cp_input_mean_ema) -- it converges to each hub's CROSS-CONCEPT mean (axis-0).
    No read here.

    learn_projection=False (default): the read-out projection plasticity gate is held CLOSED so the random
    projection stays fixed and the ONLY thing learned is the per-hub input mean -- isolating the whitening op
    (the +0.155 random-projection gate). learn_projection=True (PHASE 3): the projection gates stay OPEN so
    STDP LEARNS the hub->cortex weights on the (input-mean-centered + E/I-projected) input -- the L1 similarity-
    matching realized as bounded-Hebbian STDP, which learns the LOW-RANK principal subspace (the off-diagonal
    de-risk: low-rank whitening reaches host +0.44, where the random projection caps at +0.31). The input-mean
    primitive (the centering) keeps adapting throughout; STDP learns the projection ON the centered drive."""
    rng = np.random.RandomState(seed * 100003 + 7)
    Nc = int(np.asarray(C_drive).shape[0])
    _freeze_all_gates(bridge, 0.0 if not learn_projection else 1.0)  # PHASE 3: leave the projection PLASTIC
    try:
        for _ep in range(int(n_epochs)):
            order = rng.permutation(Nc)
            for i in order:
                _set_onoff_drive(bridge, idx, C_drive[i], drive_scale)
                for _t in range(int(settle) + int(window)):
                    _step_with_time(bridge)
                bridge.cp_external_input_current[:] = 0.0
    finally:
        _freeze_all_gates(bridge, 1.0)


def read_onoff_codes(bridge, idx, C_drive, *, drive_scale, window, settle):
    """Per-concept concat(cortex_on, cortex_off) SPIKE-COUNT code, with adaptation + read-out FROZEN.

    The caller sets cfg.input_mean_adapt_alpha=0 BEFORE this (the converged mean stops moving), and the
    plasticity gate is closed here, so the read perturbs neither the mean nor the projection. Accumulates
    cortex spike counts ON-DEVICE (one D->H sync per concept)."""
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
            _set_onoff_drive(bridge, idx, C_drive[i], drive_scale)
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


# ===========================================================================
# Host numpy references (axis-0 substrate / axis-0 signed / axis-1 / point).
# These are REFERENCE CEILINGS for honest comparison, NOT part of the neural code path.
# ===========================================================================
def _host_projection(bridge, idx, pre_key="hub_on", post_key="cortex_on"):
    """Reconstruct the bridge's ACTUAL hub->cortex weights as a dense [n_cortex x n_hub] matrix W (so the
    numpy reference uses the IDENTICAL random projection the bridge uses). Returns W."""
    from sim.backend import to_host
    csr = bridge.cp_connections
    pre, post = np.asarray(idx[pre_key]), np.asarray(idx[post_key])
    sub = csr[pre, :][:, post]
    W = (np.asarray(to_host(sub.todense())) if hasattr(sub, "todense") else np.asarray(to_host(sub))).T
    return W  # [n_cortex x n_hub]


def _onoff_host(drive, gain, rng):
    """numpy ON/OFF Poisson code of a signed drive [Nc x k] -> [Nc x 2k] (matches the probe's onoff_code)."""
    on = rng.poisson(np.maximum(drive, 0.0) * gain).astype(np.float64)
    off = rng.poisson(np.maximum(-drive, 0.0) * gain).astype(np.float64)
    return np.concatenate([on, off], axis=1)


def host_axis_refs(bridge, idx, X, S_true, *, host_gain=500.0, seed=42, n_cortex=128):
    """The axis ceilings. Returns (p_axis0_substrate, p_axis0_signed, p_axis1, p_point).

      axis0_substrate = onoff(W_bridge @ (Xn - Xn.mean(0)))  -- the bridge's ACTUAL non-negative projection
                        (what an on-substrate, Dale's-law code CAN reach: the honest target).
      axis0_signed    = onoff(W_signed @ (Xn - Xn.mean(0)))   -- a DENSE per-SYNAPSE signed Gaussian
                        projection (the validated numpy +0.31), Dale's-law-ILLEGAL -- context ceiling only.
      axis1           = onoff(W_bridge @ (Xn - Xn.mean(1)))   -- the cm-pool WRONG-axis removal.
      point           = onoff(W_bridge @ Xn)                  -- no centering.
    """
    rng = np.random.RandomState(seed)
    W = _host_projection(bridge, idx, "hub_on", "cortex_on")  # [k x n_hub], the bridge's non-negative proj
    Nh = W.shape[1]
    Wsig = np.random.RandomState(seed * 13 + 1).randn(int(n_cortex), Nh) / np.sqrt(Nh)  # signed Gaussian
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cen0 = Xn - Xn.mean(0)
    def p_of(drive):
        return _pearson_vs_Strue(_cos_sim(_onoff_host(drive, host_gain, rng)), S_true)
    p_a0_sub = p_of((W @ cen0.T).T)
    p_a0_sig = p_of((Wsig @ cen0.T).T)
    p_a1 = p_of((W @ (Xn - Xn.mean(1, keepdims=True)).T).T)
    p_pt = p_of((W @ Xn.T).T)
    return p_a0_sub, p_a0_sig, p_a1, p_pt


# ===========================================================================
# Gate runner.
# ===========================================================================
def _report(name, codes, S_true, labels, ref=None):
    p = _pearson_vs_Strue(_cos_sim(codes), S_true)
    g, ch = heldout_generalization(codes, labels)
    sil = float(np.mean(codes.sum(1) == 0))
    extra = f"  beats={p - ref:+.3f}" if ref is not None else ""
    print(f"  [{name:28s}] Pearson={p:+.3f}  gen={g:.3f} (chance {ch:.3f})  silent={sil:.2f}  "
          f"eff-rank={effective_rank(codes):.1f}{extra}", flush=True)
    return p, g, ch


def run_seed(seed, args, C, labels, S_true):
    C = np.asarray(C, np.float64); labels = np.asarray(labels)
    C_drive = encode_drive(C)  # log1p input compression
    n_hub = C.shape[1]
    X = np.log1p(np.maximum(C, 0.0))
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    print(f"\n[input-mean bridge seed {seed}] {C.shape[0]}c x {n_hub}h; host PPMI+SVD={host_p:+.3f}; "
          f"(w{args.w_mean}, ds{args.drive_scale}, win{args.window}, epochs{args.epochs}, alpha{args.alpha:.2e})",
          flush=True)
    t0 = time.time()

    bp = dict(n_hub=n_hub, n_cortex=args.n_cortex, hub_to_cortex_density=args.density, w_mean=args.w_mean,
              gain=args.gain, enable_ei=bool(getattr(args, "enable_ei", False)),
              ei_inh_weight=getattr(args, "ei_inh_weight", None))
    rp = dict(drive_scale=args.drive_scale, window=args.window, settle=args.settle)
    # STREAMING uses a (possibly shorter) window -- the EMA just needs each presentation's mean drive, not a
    # long spike-count read -- with the slow per-step alpha matched to THAT window (alpha was derived from the
    # READ window in main; rescale it so the per-PRESENTATION rate is the same regardless of stream window).
    sw = int(getattr(args, "stream_window", 0)) or int(args.window)
    rp_stream = dict(drive_scale=args.drive_scale, window=sw, settle=args.settle)
    alpha_stream = float(args.alpha) * (args.settle + args.window) / (args.settle + sw)
    fast_stream = (float(args.fast_alpha) * (args.settle + args.window) / (args.settle + sw)
                   if args.fast_alpha else 0.0)

    # host axis references (axis-0 substrate ceiling, axis-0 signed ceiling, axis-1 cm-pool, point).
    bref, idxref = build_input_mean_bridge(seed=seed, alpha=0.0, adapt=True, **bp)
    p_a0_sub, p_a0_sig, p_a1, p_pt_host = host_axis_refs(
        bref, idxref, X, S_true, host_gain=args.host_gain, seed=seed, n_cortex=args.n_cortex)
    print(f"  [host refs (same W)] axis-0 SUBSTRATE={p_a0_sub:+.3f}  axis-0 SIGNED(numpy ceiling, Dale-illegal)"
          f"={p_a0_sig:+.3f}  axis-1 cm-pool={p_a1:+.3f}  point={p_pt_host:+.3f}", flush=True)

    # --- the NEURAL per-hub-adapted code: stream (adapt) -> freeze -> read. ---
    bA, idxA = build_input_mean_bridge(seed=seed, alpha=alpha_stream, adapt=True, **bp)
    stream_adapt(bA, idxA, C_drive, n_epochs=args.epochs, seed=seed,
                 learn_projection=bool(getattr(args, "learn_projection", False)), **rp_stream)
    # confirm the primitive actually converged (on-substrate, not host): record the per-hub EMA magnitude.
    from sim.backend import to_host as _toh
    ema = _toh(bA.cp_input_mean_ema)
    ema_on = float(np.abs(ema[np.asarray(idxA["hub_on"])]).mean())
    bA.core_config.input_mean_adapt_alpha = 0.0  # FREEZE the converged per-hub mean before read-out
    code_adapt = read_onoff_codes(bA, idxA, C_drive, **rp)

    # --- the POINT / no-centering control: adaptation OFF (no flagged region; m never exists). ---
    bP, idxP = build_input_mean_bridge(seed=seed, alpha=0.0, adapt=False, **bp)
    code_point = read_onoff_codes(bP, idxP, C_drive, **rp)

    # --- the FAST-alpha control (slow-alpha load-bearing): a too-fast EMA tracks the LAST concept, not the
    #     cross-concept mean -> must do WORSE than the slow-alpha adapt. ---
    code_fast = None
    if fast_stream and fast_stream > 0:
        bF, idxF = build_input_mean_bridge(seed=seed, alpha=fast_stream, adapt=True, **bp)
        stream_adapt(bF, idxF, C_drive, n_epochs=args.epochs, seed=seed, **rp_stream)
        bF.core_config.input_mean_adapt_alpha = 0.0
        code_fast = read_onoff_codes(bF, idxF, C_drive, **rp)

    p_adapt, g_adapt, ch = _report("per-hub ADAPT (slow)", code_adapt, S_true, labels)
    p_point, g_point, _ = _report("POINT (no-centering)", code_point, S_true, labels, ref=None)
    if code_fast is not None:
        p_fast, _, _ = _report("fast-alpha (must be worse)", code_fast, S_true, labels)
    else:
        p_fast = None
    print(f"  [primitive check] per-hub EMA |mean| on hub_on = {ema_on:.2f} (>0 => the on-substrate mean "
          f"converged; the centering is real)", flush=True)

    # permuted-label anti-cheat on the per-hub-adapted code.
    rng = np.random.RandomState(seed * 2718281 + 11); perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_p = _pearson_vs_Strue(_cos_sim(code_adapt), S_perm)
    print(f"  [anti-cheat] permuted={perm_p:+.3f} (~0)  "
          f"slow-vs-fast={'' if p_fast is None else f'{p_adapt - p_fast:+.3f} (slow must lead)'}  "
          f"elapsed {time.time() - t0:.0f}s", flush=True)

    gates = {
        "near_axis0_substrate": bool(p_adapt >= p_a0_sub - 0.04),  # reaches the bridge's OWN realizable axis-0
        "clears_bar": bool(p_adapt >= 0.30),                        # the +0.30 structure bar (numpy target)
        "beats_cmpool": bool(p_adapt >= p_a1 + 0.02),               # beats the axis-1 cm-pool (the WRONG axis)
        "beats_point": bool(p_adapt >= p_point + 0.05),             # beats the no-centering control
        "permuted_collapses": bool(abs(perm_p) <= 0.12),
        "slow_loadbearing": bool(p_fast is None or p_adapt - p_fast >= 0.03),
    }
    print(f"  gates: {gates}", flush=True)
    return {"seed": seed, "host": host_p, "axis0_substrate": p_a0_sub, "axis0_signed": p_a0_sig,
            "axis1_cmpool": p_a1, "point_host": p_pt_host, "adapt": p_adapt, "adapt_gen": g_adapt,
            "point": p_point, "fast": p_fast, "permuted": perm_p, "ema_on": ema_on, "chance": ch,
            "gates": gates, "elapsed_s": round(time.time() - t0, 1)}


def _verdict(res):
    """GO iff every seed clears the +0.30 bar AND beats cm-pool + point + permuted + slow-load-bearing.
    PARTIAL_substrate if it reaches its OWN substrate axis-0 + beats point + permuted-clean but the substrate
    ceiling itself is below the bar / cm-pool (the projection is the wall, the primitive works). NEGATIVE if
    it doesn't even reach the substrate axis-0 / beat point."""
    seeds = list(res.keys())
    allgo = all(all(res[k]["gates"].values()) for k in seeds)
    reaches_sub = all(res[k]["gates"]["near_axis0_substrate"] for k in seeds)
    beats_point = all(res[k]["gates"]["beats_point"] for k in seeds)
    perm_ok = all(res[k]["gates"]["permuted_collapses"] for k in seeds)
    slow_ok = all(res[k]["gates"]["slow_loadbearing"] for k in seeds)
    clears = all(res[k]["gates"]["clears_bar"] for k in seeds)
    if allgo:
        return "GO"
    if reaches_sub and beats_point and perm_ok and slow_ok and not clears:
        return "PARTIAL_reaches_substrate_axis0_but_projection_caps_below_bar"
    if beats_point and perm_ok:
        return "PARTIAL_beats_point_below_substrate_ceiling"
    return "NEGATIVE"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="synthetic 64-concept calibration (tiny)")
    p.add_argument("--real", action="store_true", help="real TinyStories corpus (the decisive gate)")
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=500)
    p.add_argument("--n-cortex", type=int, default=128)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--w-mean", type=float, default=400.0)
    p.add_argument("--drive-scale", type=float, default=40.0)
    p.add_argument("--window", type=int, default=1000, help="READ window (cortex spike-count quality)")
    p.add_argument("--stream-window", type=int, default=0,
                   help="STREAMING window per presentation (0 = same as --window). Shorter keeps the EMA "
                        "convergence cheap; the slow per-step alpha is auto-rescaled so the per-presentation "
                        "rate is unchanged.")
    p.add_argument("--settle", type=int, default=8)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--enable-ei", action="store_true",
                   help="E/I signed projection: add an inhibitory hub->cortex pathway alongside the excitatory "
                        "one (cortex drive = g_e - g_i = signed) -- the de-risked fix for the excitatory-only "
                        "projection-sign collapse.")
    p.add_argument("--ei-inh-weight", type=float, default=None,
                   help="inhibitory hub->cortex weight_mean (default = --w-mean); tune to balance g_e vs g_i.")
    p.add_argument("--host-gain", type=float, default=500.0)
    p.add_argument("--epochs", type=int, default=12, help="streaming epochs (the EMA convergence horizon)")
    p.add_argument("--learn-projection", action="store_true",
                   help="PHASE 3: leave the hub->cortex projection PLASTIC during streaming so STDP learns the "
                        "low-rank principal subspace (vs the default frozen random projection). The off-diagonal "
                        "de-risk shows low-rank learning reaches host +0.44 where the random projection caps "
                        "~+0.31.")
    p.add_argument("--alpha", type=float, default=None,
                   help="PER-STEP EMA rate during streaming. Default: derived from epochs+window so the "
                        "per-PRESENTATION rate ~0.03 (alpha_step = 0.03 / (settle+window)).")
    p.add_argument("--fast-alpha", type=float, default=None,
                   help="fast-alpha control per-step rate (load-bearing anti-cheat). Default ~25x the slow.")
    p.add_argument("--out", default="research/findings/raw/_phaseB_input_mean_bridge.json")
    args = p.parse_args()

    # derive the SLOW per-step alpha from the presentation length: the per-PRESENTATION EMA rate must be
    # ~0.02-0.05 (the validated slow band), and a presentation is (settle+window) steps, so per step it is
    # tiny. This is the load-bearing "the mean spans many presentations, not steps" conversion.
    def _derive(window):
        pres = args.settle + window
        a_slow = 0.03 / pres
        a_fast = 0.8 / pres  # per-presentation ~0.8 (EMA tracks ~the last presentation) -> must do worse
        return a_slow, a_fast

    if args.smoke:
        Cs, ls, Ss, _ = build_concept_hub_counts(8, 8, 200, 12, 40.0, 4.0, 0.3, 42)
        args.window = min(args.window, 60); args.epochs = min(args.epochs, 4)
        args.alpha, args.fast_alpha = _derive(args.window)
        seeds = [int(s) for s in args.seeds.split(",")]
        res = {str(s): run_seed(s, args, Cs, ls, Ss) for s in seeds}
    elif args.real:
        if args.alpha is None or args.fast_alpha is None:
            a_slow, a_fast = _derive(args.window)
            args.alpha = a_slow if args.alpha is None else args.alpha
            args.fast_alpha = a_fast if args.fast_alpha is None else args.fast_alpha
        seeds = [int(s) for s in args.seeds.split(",")]
        res = {}
        for s in seeds:
            C, labels, S_true = build_real_corpus(s, args.n_hub)
            res[str(s)] = run_seed(s, args, C, labels, S_true)
    else:
        print("specify --smoke (synthetic) or --real (TinyStories).")
        return

    verdict = _verdict(res)
    adapt_mean = float(np.mean([res[k]["adapt"] for k in res]))
    a0sub = float(np.mean([res[k]["axis0_substrate"] for k in res]))
    a0sig = float(np.mean([res[k]["axis0_signed"] for k in res]))
    a1_mean = float(np.mean([res[k]["axis1_cmpool"] for k in res]))
    pt_mean = float(np.mean([res[k]["point"] for k in res]))
    perm_mean = float(np.mean([res[k]["permuted"] for k in res]))
    fast_vals = [res[k]["fast"] for k in res if res[k]["fast"] is not None]
    fast_mean = float(np.mean(fast_vals)) if fast_vals else None
    print(f"\n  MEAN ({len(res)} seeds): per-hub ADAPT {adapt_mean:+.3f} | axis-0 substrate {a0sub:+.3f} | "
          f"axis-0 signed(numpy) {a0sig:+.3f} | axis-1 cm-pool {a1_mean:+.3f} | point {pt_mean:+.3f} | "
          f"fast {('n/a' if fast_mean is None else f'{fast_mean:+.3f}')} | permuted {perm_mean:+.3f}",
          flush=True)
    print(f"\n  INPUT-MEAN BRIDGE GATE VERDICT: {verdict}", flush=True)
    out = {"verdict": verdict,
           "mean": {"adapt": adapt_mean, "axis0_substrate": a0sub, "axis0_signed": a0sig,
                    "axis1_cmpool": a1_mean, "point": pt_mean, "fast": fast_mean, "permuted": perm_mean},
           "per_seed": res, "config": vars(args)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
