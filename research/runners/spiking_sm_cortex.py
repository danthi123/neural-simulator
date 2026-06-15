"""Phase B -- the spiking similarity-matching cortex BRIDGE BUILDER + PPMI-shaped input encoder (Task 1).

This is the substrate for the learned-graded similarity-matching cortex: a 2-region bridge with a
PLASTIC, gated hub -> cortex projection, the dendritic divisive (/marginal) gain, STDP + homeostasis,
and the OU background DISABLED (so a concept's code reflects only the drive-driven firing, not a
spontaneous baseline that would swamp the per-hub co-occurrence marginal the gain normalizes by).

Two public functions:
  build_sm_cortex_bridge(n_hub, n_cortex, seed, ...) -> (bridge, hub_idx, cortex_idx)
  encode_drive(C_row, log=True) -> the Weber-Fechner log1p(max(C,0)) input compression (the /marginal +
      threshold are applied LATER by the bridge's dendritic gain + rheobase, NOT here).

Template: research/runners/dendritic_cortex_forward_codes_derisk._build_cortex_bridge -- same brain-region
framework + dendritic-gain + OU-off pattern; the DIFFERENCE is the plastic gated hub->cortex pathway +
STDP/homeostasis on + the soft-bound stdp_w_max raised above the design weight.

NO new sim/ edits (uses the existing brain-region framework + the Phase-1 dendritic gain).

Run (CPU smoke):
  SIM_BACKEND=numpy python -m research.runners.spiking_sm_cortex
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def build_sm_cortex_bridge(
    n_hub,
    n_cortex,
    seed,
    *,
    density=0.1,
    weight_mean=0.05,
    weight_jitter=0.02,
    sigma=0.05,
    alpha=0.05,
    enable_hebbian_learning=False,
    # --- Task 3 (C1a) competitive-STDP knobs. ALL default to the Task-1/2 behavior so existing
    #     callers / tests are byte-preserved (cortex_exc_fraction=1.0 + internal_density=0.0 = no WTA;
    #     homeostasis overrides None = the slow navigation defaults from CoreSimConfig). ---
    cortex_exc_fraction=1.0,
    cortex_internal_density=0.0,
    cortex_inh_weight_mean=6.0,
    cortex_exc_weight_mean=0.3,
    homeostasis_ema_alpha=None,
    homeostasis_threshold_adapt_rate=None,
    homeostasis_target_rate=None,
    stdp_w_max=30.0,
):
    """Build a 2-region similarity-matching cortex bridge.

    hub region   : n_hub excitatory neurons, internal_density 0 (a pure input layer).
    cortex region: n_cortex neurons, the learned read-out layer.
    pathway      : a PLASTIC hub -> cortex projection tagged plasticity_gate="hub_to_cortex" so it can be
                   frozen/thawed at runtime via bridge.set_plasticity_gate("hub_to_cortex", v).

    Enabled: the brain-region framework, the dendritic divisive (/marginal) gain, STDP, and homeostasis.
    The STDP soft-bound stdp_w_max is raised ABOVE the design weight so STDP does not collapse the weights
    (CLAUDE.md soft-bound gotcha). The OU background process is DISABLED.

    Task 3 (C1a) competitive-STDP recipe (opt-in, default OFF so Task-1/2 stays byte-identical):
      * WTA: set ``cortex_exc_fraction < 1.0`` (e.g. 0.8) so the RegionManager auto-creates an inhibitory
        cortex subset, AND ``cortex_internal_density > 0`` (e.g. 0.5) so the cortex has within-region
        connectivity. The inhibitory cortex neurons route their output through the inhibitory conductance
        channel (the bridge sets their trait inhibitory from region_manager.inhibitory_indices). With a
        STRONG ``cortex_inh_weight_mean`` this is E->I->E feedback inhibition = winner-take-all (only a few
        cortex neurons win per presentation -> they get the pre-before-post LTP timing -> they differentiate
        into per-category receptive fields). The within-region inhibitory weight field is BrainRegion's
        ``inh_weight_mean`` (sim/regions.py:584; the I->E weight in the MIXED internal wiring).
      * Fast adaptive-threshold homeostasis (Diehl-Cook theta): the CoreSimConfig defaults are DELIBERATELY
        slowed for navigation (homeostasis_ema_alpha=0.0002 ~5s tau, threshold_adapt_rate=0.0005). Pass
        ``homeostasis_ema_alpha``/``homeostasis_threshold_adapt_rate`` (e.g. 0.05/0.03) to override them on
        THIS bridge's cfg only (NOT the global defaults) so theta actually equalizes per-concept firing
        rates within the run (no dead/dominant cortex units).

    Returns (bridge, hub_idx, cortex_idx) where the index arrays are the per-region neuron index slices.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="hub", n_neurons=n_hub, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(
            name="cortex",
            n_neurons=n_cortex,
            exc_fraction=cortex_exc_fraction,
            internal_density=cortex_internal_density,
            exc_weight_mean=cortex_exc_weight_mean,
            inh_weight_mean=cortex_inh_weight_mean,
            weight_jitter=0.0,            # deterministic WTA strength (no extra seed-dependent jitter)
            plastic_internal=False,       # the lateral-inhibition WTA is FIXED; only hub->cortex learns
        ),
    ]
    cfg.region_pathways = [
        RegionPathway(
            from_region="hub",
            to_region="cortex",
            density=density,
            weight_mean=weight_mean,
            weight_jitter=weight_jitter,
            plastic=True,
            plasticity_gate="hub_to_cortex",
        ),
    ]

    cfg.dt = 1.0
    cfg.dt_ms = 1.0  # keep dt_ms == dt: the per-step time advance (the STDP delta_t clock) uses dt_ms.
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed

    # the concept code must reflect the DRIVE-driven firing only -- spontaneous OU noise would give every
    # hub a baseline firing rate that swamps the per-hub co-occurrence marginal the gain normalizes by.
    cfg.enable_ou_process = False

    # the dendritic per-presynaptic-source divisive (/marginal) gain (D2 Phase 1): g = sigma/(sigma + a_src).
    cfg.enable_dendritic_divisive_gain = True
    cfg.dendritic_divisive_sigma = sigma
    cfg.dendritic_gain_ema_alpha = alpha

    # the learning rules: STDP (Hebbian timing rule) + homeostasis (intrinsic excitability regulation).
    cfg.enable_stdp = True
    cfg.enable_homeostasis = True
    # Task 3 (C1a) fast adaptive-threshold homeostasis: override the SLOW navigation defaults
    # (ema_alpha=0.0002 ~5s, adapt_rate=0.0005) on THIS bridge's cfg ONLY when given, so the Diehl-Cook
    # theta equalizes per-concept firing rates within the run. None = leave the (slow) defaults (Task-1/2).
    if homeostasis_ema_alpha is not None:
        cfg.homeostasis_ema_alpha = float(homeostasis_ema_alpha)
    if homeostasis_threshold_adapt_rate is not None:
        cfg.homeostasis_threshold_adapt_rate = float(homeostasis_threshold_adapt_rate)
    if homeostasis_target_rate is not None:
        cfg.homeostasis_target_rate = float(homeostasis_target_rate)
    # structural plasticity (synaptogenesis) is NOT part of the similarity-matching learn (the hub->cortex
    # topology is fixed; only the WEIGHTS learn) AND it triggers a pre-existing capacity bug on a plasticity-
    # gated pathway (cp_plasticity_rate_gain isn't resized when the synapse nnz grows -> IndexError at
    # bridge.py:6378, which silently corrupts the STDP update). Disable it for this build.
    cfg.enable_structural_plasticity = False
    # Hebbian learning carries a per-sub-step weight DECAY (~1e-5) that, over the 100K+ steps of Phase-B
    # training, collapses STDP-grown weights to the ~0.05 floor (CLAUDE.md text-IO fix #1; every g* / Tier-1
    # learning recipe sets this False and relies on STDP). Default OFF so the learned hub->cortex weights
    # survive extended training; opt-in only.
    cfg.enable_hebbian_learning = enable_hebbian_learning
    # soft-bound STDP collapses weights when weight_mean > stdp_w_max (CLAUDE.md gotcha). The cap MUST sit
    # well above the design weight so STDP can grow/adjust without clipping. For C1a the hub->cortex
    # pathway needs a STRONG weight_mean (so the cortex fires from hub drive, not just the co-fire), so the
    # cap is a kwarg (default 30.0 preserves Task-1/2); set it above the C1a design weight (e.g. 200).
    cfg.stdp_w_max = float(stdp_w_max)

    rt = RuntimeState()
    rt.actual_seed_used = seed

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=rt,
        gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data()

    hub_idx = np.asarray(bridge.region_manager.indices("hub"))
    cortex_idx = np.asarray(bridge.region_manager.indices("cortex"))
    return bridge, hub_idx, cortex_idx


def _set_hub_drive(bridge, hub_idx, drive_row, drive_scale, cortex_idx=None, cofire_pA=0.0):
    """Set bridge.cp_external_input_current on hub_idx to drive_row * drive_scale (zero elsewhere).

    When ``cofire_pA > 0`` AND ``cortex_idx`` is given, ALSO add a UNIFORM ``cofire_pA`` depolarizing
    current to EVERY cortex neuron (the C1a non-specific co-fire teaching drive). The same flat value is
    added for every cortex neuron and every concept, so it carries NO per-concept information (the only
    concept-specific signal is the hub drive, set by the environment) -- legitimately unsupervised.

    Handles the cupy/numpy split: on cupy the per-index assignment needs an xp array (mirrors
    dendritic_cortex_forward_codes_derisk._present)."""
    from sim.backend import get_backend

    xp, _ = get_backend()
    hub_idx = np.asarray(hub_idx)
    drive = (np.asarray(drive_row, dtype=np.float64) * float(drive_scale)).astype(np.float32)
    bridge.cp_external_input_current[:] = 0.0
    is_cupy = type(bridge.cp_external_input_current).__module__.startswith("cupy")
    if is_cupy:
        bridge.cp_external_input_current[xp.asarray(hub_idx)] = xp.asarray(drive)
    else:
        bridge.cp_external_input_current[hub_idx] = drive
    if cofire_pA and cortex_idx is not None:
        cortex_idx = np.asarray(cortex_idx)
        if is_cupy:
            bridge.cp_external_input_current[xp.asarray(cortex_idx)] += float(cofire_pA)
        else:
            bridge.cp_external_input_current[cortex_idx] += float(cofire_pA)


def _step_with_time(bridge):
    """Run ONE simulation step AND advance the simulation clock (current_time_ms += dt_ms).

    CRITICAL (the Task-2 scaffold bug, root-caused 2026-06-15): bridge._run_one_simulation_step() does NOT
    advance runtime_state.current_time_ms on its own -- the canonical stepper (bridge.py:3641) does. The
    STDP rule stamps every spike with current_time_ms and scores delta_t = t_post - t_pre. If the clock is
    frozen at 0, EVERY (pre, post) pair has delta_t == 0 -> neither the LTP (delta_t>0) nor the LTD
    (delta_t<0) branch fires -> STDP is a complete NO-OP (the weights never move, the cortex never
    differentiates). Advancing the clock per step (exactly as bridge.step_simulation does) is what makes
    the spike-timing learning rule actually run. Pure runner-level fix; NO sim/ edit."""
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    bridge.runtime_state.current_time_step += 1


def _hub_to_cortex_mean_weight(bridge, hub_idx, cortex_idx):
    """Mean of the hub->cortex synaptic weights (the plastic projection's overall magnitude).

    Used to record the weight trajectory during training -- a collapse-to-floor (the silent-target trap)
    shows as a geometric decay toward ~0. Slices cp_connections rows=hub_idx, cols=cortex_idx."""
    from sim.backend import to_host

    csr = bridge.cp_connections
    sub = csr[np.asarray(hub_idx), :][:, np.asarray(cortex_idx)]
    data = np.asarray(to_host(sub.data))
    if data.size == 0:
        return 0.0
    return float(data.mean())


def train_sm_cortex(
    bridge,
    C_drive,
    hub_idx,
    cortex_idx,
    *,
    n_epochs=2,
    drive_scale=12.0,
    window=20,
    settle=6,
    cofire_pA=0.0,
    record_weight_trajectory=False,
):
    """Train the plastic hub->cortex projection by presenting each concept's drive (plasticity ON).

    For each epoch, for each concept row i: drive the hub neurons with C_drive[i] * drive_scale, run
    settle + window steps so the cortex fires and the plastic hub->cortex STDP potentiates co-active
    (hub, cortex) pairs. The external current is zeroed at the end of each presentation.

    Task 3 (C1a) co-fire teaching drive: when ``cofire_pA > 0``, a UNIFORM ``cofire_pA`` depolarizing
    current is ADDED to ALL cortex neurons during each presentation window (the same flat value for every
    cortex neuron and every concept -> carries NO per-concept info). It guarantees the WTA winners reliably
    cross threshold so the hub->cortex STDP gets the pre-before-post (LTP) timing instead of the silent-
    target net-LTD collapse. The HUB drive (concept-specific) determines WHICH winners potentiate.

    The hub_to_cortex plasticity gate is left at its default (1.0 = full plasticity) -- STDP runs.

    Returns the bridge (mutated in place), OR -- when ``record_weight_trajectory=True`` -- a list of the
    hub->cortex mean weight sampled once per epoch (the collapse-guard instrument; a geometric decay to ~0
    would diagnose the silent-target trap).
    """
    from sim.backend import to_host  # noqa: F401  (re-export availability check; used by callers)

    Nc = int(np.asarray(C_drive).shape[0])
    traj = []
    if record_weight_trajectory:
        traj.append(round(_hub_to_cortex_mean_weight(bridge, hub_idx, cortex_idx), 4))
    for _ in range(int(n_epochs)):
        for i in range(Nc):
            _set_hub_drive(bridge, hub_idx, C_drive[i], drive_scale,
                           cortex_idx=cortex_idx, cofire_pA=cofire_pA)
            for _t in range(int(settle) + int(window)):
                _step_with_time(bridge)
            bridge.cp_external_input_current[:] = 0.0
        if record_weight_trajectory:
            traj.append(round(_hub_to_cortex_mean_weight(bridge, hub_idx, cortex_idx), 4))
    if record_weight_trajectory:
        return traj
    return bridge


def read_codes(
    bridge,
    C_drive,
    hub_idx,
    cortex_idx,
    *,
    drive_scale=12.0,
    window=20,
    settle=6,
):
    """Read the per-concept cortex spike-count codes with plasticity FROZEN.

    Freezes the hub->cortex pathway (set_plasticity_gate("hub_to_cortex", 0.0)) so the read does not
    perturb the learned weights, then for each concept presents its drive (same protocol as training)
    and accumulates the cortex region's SPIKE COUNTS over the `window` steps (summing
    bridge.cp_firing_states[cortex_idx] each step after `settle`). Restores the gate to 1.0 afterward.

    Returns an [Nc x n_cortex] numpy array of per-concept cortex spike-count codes.
    """
    from sim.backend import to_host

    cortex_idx = np.asarray(cortex_idx)
    Nc = int(np.asarray(C_drive).shape[0])
    codes = np.zeros((Nc, cortex_idx.size), dtype=np.float64)

    bridge.set_plasticity_gate("hub_to_cortex", 0.0)
    try:
        for i in range(Nc):
            _set_hub_drive(bridge, hub_idx, C_drive[i], drive_scale)
            acc = np.zeros(cortex_idx.size, dtype=np.float64)
            for t in range(int(settle) + int(window)):
                _step_with_time(bridge)
                if t >= int(settle):
                    fired = np.asarray(to_host(bridge.cp_firing_states))[cortex_idx]
                    acc += fired.astype(np.float64)
            codes[i] = acc
            bridge.cp_external_input_current[:] = 0.0
    finally:
        bridge.set_plasticity_gate("hub_to_cortex", 1.0)
    return codes


def encode_drive(C_row, log=True):
    """PPMI-shaped input encoder: the Weber-Fechner input compression of a concept's count row.

    Returns log1p(max(C_row, 0)) when log=True (the perceptual compression of raw co-occurrence counts),
    else max(C_row, 0) (the clipped raw counts). Same length / dtype-float as the input.

    The /marginal normalization and the threshold are applied LATER by the bridge's dendritic divisive
    gain + the neuron rheobase -- NOT here.
    """
    arr = np.asarray(C_row, dtype=np.float64)
    clipped = np.maximum(arr, 0.0)
    if log:
        return np.log1p(clipped)
    return clipped


if __name__ == "__main__":
    # CPU smoke: build a tiny bridge + show the encoder.
    b, hub, cortex = build_sm_cortex_bridge(n_hub=200, n_cortex=64, seed=42)
    print(f"built sm-cortex bridge: n_hub={len(hub)} n_cortex={len(cortex)} "
          f"total_neurons={int(b.cp_membrane_potential_v.shape[0])}")
    print("encode_drive([0,1,3,7], log=True) =", encode_drive(np.array([0.0, 1.0, 3.0, 7.0])))
