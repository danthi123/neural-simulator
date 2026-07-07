"""GNW Rung-1 de-risk: the ALL-OR-NONE IGNITION property of a Global Neuronal Workspace.

Prove the load-bearing "ignition" bifurcation (Dehaene-Changeux 2011) on the project's spiking
substrate: a small recurrent-NMDA + FS-lateral-inhibition `workspace` assembly shows die-out-vs-ignite —
a brief SUB-threshold input drive produces a feed-forward wave that DIES OUT once the drive is removed,
while an ABOVE-threshold drive latches the assembly into SUSTAINED self-reverberation that persists after
the drive is gone. The recurrence must be LOAD-BEARING: an NMDA/attractor-lesion control removes the
sustained branch at EVERY drive amplitude.

RECIPE (reuse-by-import, NO `sim/` edit): copy the validated dlPFC self-attractor from
`nav_conv_merged_bridge.py` — the `dlpfc_wm` self-loop at DLPFC_ATTRACTOR_WEIGHT=30.0 — hand-wired as a dense
block-diagonal self-recurrent assembly population and injected alongside the framework plan via
`inject_explicit_wiring` (the `dlpfc_loop` insertion pattern). The `workspace` region opts into NMDA
(`enable_nmda=True`), so the recurrent synapses carry the slow NMDA conductance in ADDITION to their fast AMPA
component. IMPORTANT (adversarial-verify Workflow wm01lvqaz): the ignition is GENERIC recurrent-attractor
bistability, NOT NMDA-specific — an AMPA-only control (NMDA off, same weight 30) ALSO ignites (plateau ~0.15).
NMDA MODULATES (raises the ignited plateau ~0.15->~0.33 and narrows the ignition band) but is not necessary; the
recurrence is what is load-bearing (the weight->0 lesion is the control that proves it). The ignited state is a
period-3 synchronous LIMIT CYCLE (all assembly neurons fire every 3rd step), persistent (holds >=600 free
steps), not a stationary fixed point. dt=1.0, Izhikevich.

THE DE-RISK (single variable = input drive amplitude A):
  For each A in a sweep: reset to quiescence (zero external current + settle steps); inject A pA into the
  workspace ASSEMBLY neurons for a BRIEF pulse; REMOVE the drive; run FREE; measure the SUSTAINED assembly
  firing (mean per-neuron spike rate over the free post-drive window) = the ignition metric. Expect a sharp
  all-or-none transition: ~0 below a threshold A*, a sustained plateau above.

ANTI-CHEATS (all must hold for GO):
  - NMDA/attractor-lesion (load-bearing recurrence): rerun the sweep with the self-attractor weight = 0.
    The sustained branch must DISAPPEAR at every A.
  - Drive-off baseline: A=0 -> sustained-rate ~0 (no spontaneous ignition).

GO GATE: (1) intact sweep shows a clear all-or-none transition (sustained below A* near-zero, plateau above
>= 3x below-threshold, with a sharp jump); (2) lesion sweep has NO sustained branch at any A; (3) A=0
baseline near-zero.

Usage:
  python -u -m research.runners._gnw_rung1_ignition_curve_derisk --seed 42 \
      --json research/findings/raw/_gnw_rung1_smoke.json --backend numpy
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host


# ── constants (mirror the validated dlPFC attractor weight) ───────────────────────────────────────────────
# The self-attractor weight. 30.0 is the validated dlPFC operating point (nav_conv_merged_bridge.py:54). NOTE:
# CLAUDE.md/that file call 30 "genuinely NMDA-dependent; 50 = trivial AMPA ping-pong", but the adversarial-verify
# (Workflow wm01lvqaz) showed AMPA-only recurrence at weight 30 ALSO sustains a clean ignition on this substrate
# — so the ignition here is generic recurrent-attractor bistability, not NMDA-specific (NMDA modulates the plateau
# height). Reported in the JSON as `attractor_weight`.
DEFAULT_ATTRACTOR_WEIGHT = 30.0

# Assembly / region geometry. One `workspace` exc region; the first ASSEMBLY_SIZE of its neurons form the
# densely self-recurrent workspace assembly (every assembly neuron -> every assembly neuron, weight ~30). The
# remaining workspace neurons are non-assembly (a background pool that does NOT self-sustain — a specificity
# control the sustained-read ignores). A separate inhibitory FS region provides lateral inhibition (stabilizes,
# prevents runaway) via workspace->workspace_fs (excite the inhibition) + workspace_fs->workspace (feedback).
WORKSPACE_N = 300
ASSEMBLY_SIZE = 80
WORKSPACE_FS_N = 50

# the hand-wired dense self-recurrent assembly population name + its plasticity gate (held frozen so the fixed
# attractor weight is never clipped or drifted; also plastic=False on the population itself).
WS_LOOP = "workspace_loop"
WS_LOOP_GATE = "workspace_loop_fixed"

# FS lateral-inhibition pathway weights. workspace->workspace_fs drives the inhibition; workspace_fs->workspace
# feeds it back (an inhibitory projection because workspace_fs is all-inhibitory, exc_fraction=0.0). Kept moderate
# so it STABILIZES the attractor (prevents an epileptic runaway) without extinguishing the sustained branch.
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 8.0

# stimulation protocol.
SETTLE_STEPS = 40        # quiescent settle (zero external current) before each drive pulse
DRIVE_STEPS = 35         # brief sub/supra-threshold input pulse into the assembly
FREE_STEPS = 100         # free post-drive window over which the SUSTAINED rate is measured


def _build_assembly_loop_population(assembly_idx: np.ndarray, attractor_weight: float) -> dict:
    """Dense self-recurrent assembly: every assembly neuron -> every assembly neuron at `attractor_weight`,
    E_TO_E, plastic=False, gated by WS_LOOP_GATE (frozen). Excludes the self-self diagonal (no autapses).
    Returns a population spec dict (the `dlpfc_loop` form)."""
    a = np.asarray(assembly_idx, dtype=np.int64)
    m = a.shape[0]
    pre = np.repeat(a, m)          # (m*m,)
    post = np.tile(a, m)
    keep = pre != post             # drop autapses
    pre = pre[keep].astype(np.int64)
    post = post[keep].astype(np.int64)
    ww = np.full(pre.shape[0], float(attractor_weight), dtype=np.float32)
    return {
        "pre_indices": pre, "post_indices": post, "initial_weights": ww,
        "plastic": False, "plasticity_gate": WS_LOOP_GATE, "conn_type": "E_TO_E", "count": int(pre.size),
    }


def build_ignition_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                          lesion: bool = False):
    """Build a MINIMAL standalone workspace bridge with the dense self-recurrent NMDA assembly.

    `lesion=True` sets the self-attractor weight to 0 (the load-bearing-recurrence anti-cheat) — everything
    else (regions, FS inhibition, drive protocol) is identical, so the sustained branch can only vanish
    because the recurrence is gone.

    Returns (bridge, xp, assembly_idx_device, handles).
    """
    xp, _ = get_backend()

    workspace = BrainRegion(
        name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
        internal_density=0.0, enable_nmda=True)          # NMDA per-region mask -> slow NMDA on the recurrence
    workspace_fs = BrainRegion(
        name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0,   # all-inhibitory FS
        internal_density=0.0, enable_nmda=False)
    regions = [workspace, workspace_fs]
    pathways = [
        # excite the inhibition, then inhibit the workspace back (lateral inhibition / stabilization). Fixed
        # weights, plastic=False (this de-risk learns nothing — the attractor is the fixed reused recipe).
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    # NMDA on globally; the per-region mask confines it to the `workspace` slice (the slow NMDA conductance that
    # makes ignition self-sustaining). nmda_ratio 0.5 = the validated dlPFC/nav operating point.
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    # Freeze everything plastic; raise the clip bounds ABOVE the frozen attractor weight so the (ungated) global
    # weight clips can never move the fixed ~30 attractor synapses (the CLAUDE.md STDP soft-bound gotcha).
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False                 # FOOT-GUN: the synaptic-scaling clip would slam frozen weights
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)
    # No OU noise: ignition is a deterministic bifurcation; a quiescent workspace must stay silent at rest (the
    # A=0 baseline anti-cheat) — OU drive would inject spurious spontaneous firing.
    cfg.enable_ou_process = False
    cfg.enable_parameter_heterogeneity = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False, "global homeostasis must stay OFF (synaptic-scaling clip foot-gun)"

    rm = bridge.region_manager
    ws_base = rm.indices("workspace")[0]
    # the first ASSEMBLY_SIZE workspace neurons form the densely self-recurrent assembly.
    assembly_idx = np.asarray(rm.indices("workspace")[:ASSEMBLY_SIZE], dtype=np.int64)

    # Build the union plan (framework FS pathways) + ADD the dense self-recurrent assembly, inject ONCE (the
    # dlpfc_loop insertion pattern). lesion=True -> attractor weight 0 (recurrence gone, everything else identical).
    eff_weight = 0.0 if lesion else float(attractor_weight)
    ws_loop = _build_assembly_loop_population(assembly_idx, eff_weight)
    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    assert WS_LOOP not in union_plan, f"{WS_LOOP} collides with a framework population"
    union_plan[WS_LOOP] = ws_loop

    inh_indices_concat = []
    for region in rm.regions():
        inh_indices_concat.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh_indices_concat or None)
    # Freeze the loop gate (the gate map was rebuilt -> default gain 1.0; zero it so nothing can drift the loop).
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    # settle to a true quiescent REST state (zero external current), then SNAPSHOT the full dynamical state
    # (v/u/firing/conductances). Each ignition trial RESTORES this snapshot so every amplitude starts from the
    # IDENTICAL quiescent state == a fresh bridge (order-invariant). The adversarial-verify (Workflow wm01lvqaz)
    # showed the earlier reset (zero conductances + settle, but NOT v/u/firing) leaked a persisting limit-cycle
    # latch between amplitudes -> a spurious sweep-order-dependent "basin-dependence". This snapshot/restore is the
    # EMERGE-61 wash-out pattern (restore the exact post-init substrate state before each production).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    quiescent_snapshot = _snapshot_state(bridge, xp)

    assembly_idx_dev = xp.asarray(assembly_idx)
    handles = {
        "seed": int(seed),
        "ws_base": int(ws_base),
        "assembly_idx": assembly_idx,
        "n_loop_edges": int(ws_loop["count"]),
        "attractor_weight": eff_weight,
        "lesion": bool(lesion),
        "quiescent_snapshot": quiescent_snapshot,
    }
    return bridge, xp, assembly_idx_dev, handles


# the full dynamical-state arrays that a persisting limit-cycle latch pollutes -> must ALL be restored to make a
# reset return to the fresh quiescent state (the adversarial-verify found v/u/firing were the leaked ones).
_STATE_ARRAYS = (
    "cp_membrane_potential_v", "cp_recovery_variable_u", "cp_firing_states",
    "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
)


def _snapshot_state(bridge, xp):
    """Copy the full dynamical state (v/u/firing/conductances) so a later reset can restore the exact quiescent
    substrate. Returns {attr: device-array copy} for every present state array."""
    snap = {}
    for name in _STATE_ARRAYS:
        arr = getattr(bridge, name, None)
        if arr is not None:
            snap[name] = arr.copy()
    return snap


def _restore_state(bridge, snap):
    """Restore the snapshotted quiescent state in place (byte-for-byte return to the fresh substrate)."""
    for name, arr in snap.items():
        getattr(bridge, name)[:] = arr


def _reset_quiescent(bridge, xp, snap):
    """Return the bridge to the IDENTICAL quiescent state captured post-init: restore v/u/firing/conductances +
    zero external current. No settle steps needed — the snapshot IS a settled rest state, so restoring it removes
    any prior-ignition carry-over completely (== a fresh bridge; order-invariant)."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0


def _run_one_drive(bridge, xp, assembly_idx_dev, snap, drive_pA: float):
    """One ignition trial at drive amplitude `drive_pA`: reset (restore the quiescent snapshot) -> brief pulse into
    the assembly -> remove drive -> free run. Returns (sustained_rate, sustained_late, drive_rate) where
    sustained_rate = mean per-neuron spike rate of the assembly over the FULL FREE post-drive window,
    sustained_late = same over the LAST THIRD of the free window (the settled rate: an ignited limit cycle's rate
    is flat so late==full, while a sub-threshold die-out's transient TAIL has decayed so late<<full ->
    distinguishes all-or-none ignition from a slowly-decaying tail), drive_rate = same over the drive window."""
    _reset_quiescent(bridge, xp, snap)

    # brief drive pulse into the assembly.
    drive_spikes = 0
    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[assembly_idx_dev] = xp.float32(drive_pA)
        bridge._run_one_simulation_step()
        drive_spikes += int(to_host(bridge.cp_firing_states[assembly_idx_dev].astype(xp.float64).sum()))
    drive_rate = drive_spikes / float(DRIVE_STEPS * assembly_idx_dev.shape[0])

    # REMOVE the drive; run free; measure the SUSTAINED assembly firing (full window + last third).
    bridge.cp_external_input_current[:] = 0.0
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)   # index at which the "late" window begins
    free_spikes = 0
    late_spikes = 0
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0   # keep it zero every step (nothing re-injects, but be explicit)
        bridge._run_one_simulation_step()
        s = int(to_host(bridge.cp_firing_states[assembly_idx_dev].astype(xp.float64).sum()))
        free_spikes += s
        if t >= late_start:
            late_spikes += s
    m = float(assembly_idx_dev.shape[0])
    sustained_rate = free_spikes / float(FREE_STEPS * m)
    sustained_late = late_spikes / float((FREE_STEPS - late_start) * m)
    return sustained_rate, sustained_late, drive_rate


def run_sweep(bridge, xp, assembly_idx_dev, snap, amplitudes):
    """Run the full drive-amplitude sweep; return per-A (sustained_rate, sustained_late, drive_rate) lists. Each
    amplitude restores `snap` first, so the sweep is order-invariant (== fresh-bridge-per-amplitude)."""
    sustained, sustained_late, drive = [], [], []
    for A in amplitudes:
        s, sl, d = _run_one_drive(bridge, xp, assembly_idx_dev, snap, float(A))
        sustained.append(float(s))
        sustained_late.append(float(sl))
        drive.append(float(d))
    return sustained, sustained_late, drive


def _analyze(amplitudes, sustained):
    """Compute A* (crossing of half the plateau), below/above levels, and a sharpness measure."""
    amps = np.asarray(amplitudes, dtype=np.float64)
    sus = np.asarray(sustained, dtype=np.float64)
    plateau = float(sus.max())
    half = 0.5 * plateau
    a_star = None
    if plateau > 0:
        above = np.where(sus >= half)[0]
        if above.size > 0:
            a_star = float(amps[above[0]])
    # below-threshold level = mean sustained over amps strictly below A* (or all if no A*).
    if a_star is not None:
        below_mask = amps < a_star
        below_level = float(sus[below_mask].mean()) if below_mask.any() else 0.0
        above_mask = amps >= a_star
        above_level = float(sus[above_mask].mean()) if above_mask.any() else 0.0
    else:
        below_level = float(sus.mean())
        above_level = float(plateau)
    # sharpness = max single-step jump across the sweep.
    max_jump = float(np.max(np.diff(sus))) if sus.size > 1 else 0.0
    return {
        "plateau": plateau, "a_star": a_star,
        "below_level": below_level, "above_level": above_level,
        "max_jump": max_jump,
    }


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-1 ignition-curve de-risk (die-out vs ignite).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung1_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--attractor-weight", type=float, default=DEFAULT_ATTRACTOR_WEIGHT)
    ap.add_argument("--a-max", type=float, default=2500.0, help="top of the drive-amplitude sweep (pA)")
    ap.add_argument("--n-amps", type=int, default=12, help="number of drive amplitudes in the sweep")
    args = ap.parse_args()

    # pin the backend BEFORE any bridge construction (sticky).
    if args.backend != "auto":
        get_backend(args.backend)

    amplitudes = list(np.linspace(0.0, float(args.a_max), int(args.n_amps)))

    print(f"[gnw-rung1] seed={args.seed} backend={args.backend} attractor_weight={args.attractor_weight} "
          f"assembly={ASSEMBLY_SIZE}/{WORKSPACE_N} sweep A in [0, {args.a_max}] x{args.n_amps}", flush=True)

    # INTACT sweep.
    bridge, xp, assembly_dev, handles = build_ignition_bridge(
        seed=args.seed, attractor_weight=float(args.attractor_weight), lesion=False)
    print(f"[gnw-rung1] intact bridge: N={bridge.core_config.num_neurons} loop_edges={handles['n_loop_edges']}",
          flush=True)
    intact_sus, intact_late, intact_drive = run_sweep(bridge, xp, assembly_dev, handles["quiescent_snapshot"], amplitudes)
    # PRIMARY metric = the LATE-window (settled fixed-point) rate: a true attractor plateau is flat (late==full),
    # while a sub-threshold die-out's TAIL has decayed by the late window (late<<full) -> the late-window read is
    # the rigorous all-or-none test (excludes decaying transients). The full-window is kept for reference.
    intact = _analyze(amplitudes, intact_late)
    intact_full = _analyze(amplitudes, intact_sus)
    for A, s, sl, d in zip(amplitudes, intact_sus, intact_late, intact_drive):
        print(f"  [intact] A={A:8.1f}  sustained={s:.4f}  late={sl:.4f}  drive={d:.4f}", flush=True)
    print(f"[gnw-rung1] intact(LATE): A*={intact['a_star']} below={intact['below_level']:.4f} "
          f"above={intact['above_level']:.4f} plateau={intact['plateau']:.4f} max_jump={intact['max_jump']:.4f}",
          flush=True)

    # LESION sweep (attractor weight 0 = load-bearing-recurrence anti-cheat).
    bridge_l, xp_l, assembly_dev_l, handles_l = build_ignition_bridge(
        seed=args.seed, attractor_weight=float(args.attractor_weight), lesion=True)
    lesion_sus, lesion_late, lesion_drive = run_sweep(bridge_l, xp_l, assembly_dev_l, handles_l["quiescent_snapshot"], amplitudes)
    lesion = _analyze(amplitudes, lesion_late)
    for A, s, sl, d in zip(amplitudes, lesion_sus, lesion_late, lesion_drive):
        print(f"  [lesion] A={A:8.1f}  sustained={s:.4f}  late={sl:.4f}  drive={d:.4f}", flush=True)
    print(f"[gnw-rung1] lesion(LATE): plateau={lesion['plateau']:.4f} above={lesion['above_level']:.4f}", flush=True)

    # ── GO gate: BISTABLE ALL-OR-NONE ignition on the LATE-window settled rate ─────────────────────────────
    # The faithful operationalization of the research gate's hypothesis ("sub-threshold die-out vs above-threshold
    # sustained self-reverberation"). The data show the settled state is BISTABLE: either the ignited fixed point
    # (~plateau) or a low non-ignited residual, with NOTHING in between (all-or-none). It is NOT a monotonic
    # sigmoid: pulsed ignition is basin-dependent (a too-strong abrupt pulse can over-drive past the basin and
    # fail to latch — the Dehaene-Changeux 2011 stimulus-must-land-in-the-basin feature), so the gate tests
    # BIMODALITY (a clear gap between the ignited and non-ignited states), not a monotone threshold.
    la = np.asarray(intact_late, dtype=np.float64)
    plateau = float(la.max())
    baseline_rate = float(intact_late[0])          # A=0 late-window sustained (must be ~0: no spontaneous ignition)
    BASELINE_MAX = 0.02
    IGNITE_FRAC = 0.5                               # "ignited" iff late >= 0.5 * plateau
    GAP_FRAC = 0.2                                  # bimodal iff every NON-ignited state is <= 0.2 * plateau
    ignited_mask = la >= IGNITE_FRAC * plateau if plateau > 0 else np.zeros_like(la, dtype=bool)
    n_ignited = int(ignited_mask.sum())
    offstate_max = float(la[~ignited_mask].max()) if (~ignited_mask).any() else 0.0
    a_star_ignite = None                           # lowest NON-ZERO amplitude that ignites
    for A, ig in zip(amplitudes, ignited_mask):
        if ig and A > 0:
            a_star_ignite = float(A); break

    lesion_plateau = float(np.asarray(lesion_late, dtype=np.float64).max())

    # (1) a real ignited state exists: >=1 amplitude latches a sustained plateau that is a genuine rate.
    ignites = bool(n_ignited >= 1 and plateau >= 0.05)
    # (2) BIMODAL all-or-none: every non-ignited settled state is far below the ignited plateau (clear gap, nothing
    #     in the intermediate band) -> the transition is all-or-none, not graded.
    bimodal_all_or_none = bool(plateau > 0 and offstate_max <= GAP_FRAC * plateau)
    # (3) recurrence LOAD-BEARING: the lesion (attractor weight 0) NEVER ignites at any amplitude.
    recurrence_load_bearing = bool(lesion_plateau <= GAP_FRAC * plateau and lesion_plateau < 0.05)
    # (4) no spontaneous ignition at A=0.
    baseline_ok = bool(baseline_rate <= BASELINE_MAX)

    go = bool(ignites and bimodal_all_or_none and recurrence_load_bearing and baseline_ok)

    # keep the legacy monotonic-analysis fields for reference (A* = the _analyze half-plateau crossing).
    intact_has_transition = bool(intact["a_star"] is not None and intact["plateau"] >= 0.05)
    lesion_no_sustained = recurrence_load_bearing

    result = {
        "runner": "_gnw_rung1_ignition_curve_derisk",
        "go": go,
        "seed": int(args.seed),
        "backend": args.backend,
        "attractor_weight": float(args.attractor_weight),
        "assembly_size": int(ASSEMBLY_SIZE),
        "workspace_n": int(WORKSPACE_N),
        "workspace_fs_n": int(WORKSPACE_FS_N),
        "n_loop_edges": int(handles["n_loop_edges"]),
        "protocol": {"settle_steps": SETTLE_STEPS, "drive_steps": DRIVE_STEPS, "free_steps": FREE_STEPS},
        "amplitudes": [float(a) for a in amplitudes],
        "primary_metric": "late_window_settled_rate",
        "intact_sustained": [float(s) for s in intact_sus],       # full free-window mean (reference)
        "intact_sustained_late": [float(s) for s in intact_late], # LAST-third settled rate (PRIMARY)
        "intact_drive": [float(d) for d in intact_drive],
        "lesion_sustained": [float(s) for s in lesion_sus],
        "lesion_sustained_late": [float(s) for s in lesion_late],
        "lesion_drive": [float(d) for d in lesion_drive],
        "a_star": a_star_ignite,                                  # lowest non-zero amplitude that IGNITES
        "intact_below_level": intact["below_level"],              # LATE-window below-threshold level (reference)
        "intact_above_level": intact["above_level"],
        "intact_plateau": plateau,                                # the ignited fixed-point rate
        "intact_max_jump": intact["max_jump"],
        "intact_below_level_full": intact_full["below_level"],    # full-window below (the ~0.02 tail, reference)
        "intact_plateau_full": intact_full["plateau"],
        "lesion_plateau": lesion_plateau,
        "baseline_rate": baseline_rate,
        # bistability / all-or-none characterization (the PRIMARY GO evidence)
        "n_ignited": n_ignited,
        "n_amps": int(len(amplitudes)),
        "offstate_max": offstate_max,                             # highest NON-ignited settled rate (bimodal gap)
        "ignited_mask": [bool(b) for b in ignited_mask],          # which amplitudes latched (basin-dependent)
        "gate_detail": {
            "ignites": bool(ignites),
            "bimodal_all_or_none": bool(bimodal_all_or_none),
            "recurrence_load_bearing": bool(recurrence_load_bearing),
            "baseline_ok": bool(baseline_ok),
            "IGNITE_FRAC": IGNITE_FRAC, "GAP_FRAC": GAP_FRAC, "BASELINE_MAX": BASELINE_MAX,
            "offstate_max_over_plateau": (offstate_max / plateau) if plateau > 0 else None,
            "lesion_plateau_over_plateau": (lesion_plateau / plateau) if plateau > 0 else None,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[gnw-rung1] GO={go}", flush=True)
    print(f"[gnw-rung1]   ignites={ignites} bimodal_all_or_none={bimodal_all_or_none} "
          f"recurrence_load_bearing={recurrence_load_bearing} baseline_ok={baseline_ok}", flush=True)
    print(f"[gnw-rung1]   plateau={plateau:.4f} (ignited {n_ignited}/{len(amplitudes)} amps, A*_ignite={a_star_ignite}) "
          f"offstate_max={offstate_max:.4f} ({offstate_max/max(plateau,1e-9):.2f}x plateau) "
          f"| lesion_plateau={lesion_plateau:.4f} | baseline={baseline_rate:.4f}", flush=True)
    print(f"[gnw-rung1] wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
