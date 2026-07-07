"""GNW Rung-3 de-risk: the REPORT == REASONING identity on the spiking workspace (the GWT paper's key result).

Builds on Rung-1 (ignition GO) + Rung-2 (mutual exclusion GO). Rung-3 demonstrates the load-bearing GWT finding
(Gurnee et al. 2026, §3.5.2/§7; Dehaene-Changeux 2011 global broadcast): the SAME ignited workspace assembly is
what the brain would SAY (a downstream "report" readout) AND what it REASONS with (a downstream "reason" consumer)
-- so a causal swap of the ignited content flips BOTH together, and a workspace lesion breaks the (workspace-only)
reasoning while a peripheral/direct path keeps the report alive (the paper's dissociation: ablate the workspace ->
the model still parses/recalls/speaks but fails internal reasoning -> report-reps == reasoning-reps).

This is the cheap-first de-risk with ABSTRACT downstream regions (report/reason). The project value: it is the
substrate for unifying the project's existing SPEAK (A->W read-out) and REASON (EMERGE inference) faculties into
ONE workspace -- the real-faculty wire-in is Rung-3b.

MECHANISM (reuse-by-import, NO `sim/` edit): a `workspace` region with TWO content assemblies (A, B), each a dense
self-recurrent loop at weight 30 (Rung-1) sharing an inhibitory `workspace_fs` pool (Rung-2 mutual exclusion) ->
exactly one content ignites at a time. Two downstream regions read the workspace via fixed content-specific
projections: `report` (report_A/report_B: what the brain would SAY -- which content is in the workspace) and
`reason` (reason_A/reason_B: a downstream reasoning consumer). `report` ALSO has a peripheral DIRECT input path
(so it survives a workspace lesion); `reason` reads ONLY the workspace.

THE DE-RISK:
  (1) IDENTITY: ignite content A -> report says A AND reason concludes A; ignite B -> report B AND reason B ->
      the report and the reasoning are driven by the SAME ignited assembly.
  (2) CAUSAL SWAP: swapping which content ignites flips BOTH report and reason together (the membership test).
  (3) WORKSPACE LESION (the assembly cannot ignite): reason collapses (reasoning needs the workspace).
  (4) DISSOCIATION: under the SAME lesion, the peripheral DIRECT path keeps report alive (a peripheral faculty
      survives) while reason stays dead -> report-reps == reasoning-reps (the workspace reps ARE the reasoning reps).

GO GATE: identity (both follow the ignited content, 2/2) AND causal-swap flips both AND workspace-lesion collapses
reason AND the dissociation holds (direct path keeps report alive under lesion, reason dead).

Usage:
  python -u -m research.runners._gnw_rung3_report_reasoning_identity_derisk --seed 42 \
      --json research/findings/raw/_gnw_rung3_smoke.json --backend numpy
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

from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS, DRIVE_STEPS, FREE_STEPS,
)

WORKSPACE_N = 300
ASSEMBLY_SIZE = 80
WORKSPACE_FS_N = 50
DOWNSTREAM_N = 100          # report / reason regions; first half = _A pop, second half = _B pop
DOWN_HALF = DOWNSTREAM_N // 2
WS_LOOP_GATE = "workspace_loop_fixed"
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 16.0
BCAST_WEIGHT = 12.0        # workspace assembly -> downstream content pop (drives the readout when the content ignites)
IGNITE_FRAC = 0.5
SOLO_PLATEAU = 1.0 / 3.0
DIRECT_PERIPHERAL_PA = 2500.0   # the peripheral direct drive into a report pop (survives a workspace lesion)


def _dense_projection(pre_idx: np.ndarray, post_idx: np.ndarray, weight: float, gate: str) -> dict:
    """Every pre neuron -> every post neuron at `weight`, E_TO_E, plastic=False, frozen gate."""
    pre = np.repeat(np.asarray(pre_idx, dtype=np.int64), len(post_idx))
    post = np.tile(np.asarray(post_idx, dtype=np.int64), len(pre_idx))
    ww = np.full(pre.shape[0], float(weight), dtype=np.float32)
    return {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
            "initial_weights": ww, "plastic": False, "plasticity_gate": gate,
            "conn_type": "E_TO_E", "count": int(pre.size)}


def build_broadcast_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                           lesion_workspace: bool = False, reason_reads_workspace: bool = True):
    """Workspace (2 assemblies + shared inhibition) + downstream `report` and `reason` regions read via fixed
    content-specific projections. `lesion_workspace=True` zeroes the assembly self-recurrence (the workspace cannot
    ignite). Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()

    regions = [
        BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="report", n_neurons=DOWNSTREAM_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="reason", n_neurons=DOWNSTREAM_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    pathways = [
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
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process",
              "enable_parameter_heterogeneity"):
        setattr(cfg, f, False)
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    A = ws[:ASSEMBLY_SIZE]
    B = ws[ASSEMBLY_SIZE:2 * ASSEMBLY_SIZE]
    rep = np.asarray(rm.indices("report"), dtype=np.int64)
    rea = np.asarray(rm.indices("reason"), dtype=np.int64)
    rep_A, rep_B = rep[:DOWN_HALF], rep[DOWN_HALF:]
    rea_A, rea_B = rea[:DOWN_HALF], rea[DOWN_HALF:]

    w_loop = 0.0 if lesion_workspace else float(attractor_weight)
    union = dict(rm.build_wiring_plan(seed=int(seed)))
    union["ws_loop_A"] = _build_assembly_loop_population(A, w_loop)
    union["ws_loop_B"] = _build_assembly_loop_population(B, w_loop)
    # content-specific broadcast: workspace assembly -> its report + reason content pops.
    union["A_to_report"] = _dense_projection(A, rep_A, BCAST_WEIGHT, WS_LOOP_GATE)
    union["B_to_report"] = _dense_projection(B, rep_B, BCAST_WEIGHT, WS_LOOP_GATE)
    if reason_reads_workspace:
        union["A_to_reason"] = _dense_projection(A, rea_A, BCAST_WEIGHT, WS_LOOP_GATE)
        union["B_to_reason"] = _dense_projection(B, rea_B, BCAST_WEIGHT, WS_LOOP_GATE)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {"A": A, "B": B, "rep_A": rep_A, "rep_B": rep_B, "rea_A": rea_A, "rea_B": rea_B,
           "A_dev": xp.asarray(A), "B_dev": xp.asarray(B),
           "rep_A_dev": xp.asarray(rep_A), "rep_B_dev": xp.asarray(rep_B),
           "rea_A_dev": xp.asarray(rea_A), "rea_B_dev": xp.asarray(rea_B)}
    return bridge, xp, idx, snap


def _rate(bridge, xp, idx_dev, n_late):
    return int(to_host(bridge.cp_firing_states[idx_dev].astype(xp.float64).sum())), n_late


def _run_trial(bridge, xp, idx, snap, ignite: str, direct_report=None):
    """Restore quiescence -> drive the ignite-content assembly -> free -> read report_A/B + reason_A/B late rates.
    `direct_report` in {'A','B',None}: also inject a peripheral DIRECT drive into that report pop every step (the
    dissociation path that survives a workspace lesion)."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    drive_dev = idx["A_dev"] if ignite == "A" else idx["B_dev"]
    direct_dev = None
    if direct_report == "A":
        direct_dev = idx["rep_A_dev"]
    elif direct_report == "B":
        direct_dev = idx["rep_B_dev"]

    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[drive_dev] = xp.float32(2500.0)
        if direct_dev is not None:
            bridge.cp_external_input_current[direct_dev] = xp.float32(DIRECT_PERIPHERAL_PA)
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    acc = {k: 0 for k in ("rep_A", "rep_B", "rea_A", "rea_B")}
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        if direct_dev is not None:                     # the peripheral path is sustained (an external, non-workspace input)
            bridge.cp_external_input_current[direct_dev] = xp.float32(DIRECT_PERIPHERAL_PA)
        bridge._run_one_simulation_step()
        if t >= late_start:
            acc["rep_A"] += int(to_host(bridge.cp_firing_states[idx["rep_A_dev"]].astype(xp.float64).sum()))
            acc["rep_B"] += int(to_host(bridge.cp_firing_states[idx["rep_B_dev"]].astype(xp.float64).sum()))
            acc["rea_A"] += int(to_host(bridge.cp_firing_states[idx["rea_A_dev"]].astype(xp.float64).sum()))
            acc["rea_B"] += int(to_host(bridge.cp_firing_states[idx["rea_B_dev"]].astype(xp.float64).sum()))
    denom = float((FREE_STEPS - late_start) * DOWN_HALF)
    return {k: v / denom for k, v in acc.items()}


def _decode(rates, kind):
    """'A' / 'B' / 'none' from the two content-pop rates of `kind` ('rep' or 'rea')."""
    a, b = rates[f"{kind}_A"], rates[f"{kind}_B"]
    thr = IGNITE_FRAC * SOLO_PLATEAU * 0.3   # a downstream pop is "reporting" a content at ~a fraction of its drive
    if a < thr and b < thr:
        return "none"
    return "A" if a >= b else "B"


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-3 report==reasoning identity de-risk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung3_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    args = ap.parse_args()
    if args.backend != "auto":
        get_backend(args.backend)

    print(f"[gnw-rung3] seed={args.seed} backend={args.backend}", flush=True)

    # ── INTACT: identity + causal swap ─────────────────────────────────────────────────────────────────────
    bridge, xp, idx, snap = build_broadcast_bridge(seed=args.seed, lesion_workspace=False)
    igA = _run_trial(bridge, xp, idx, snap, "A")
    igB = _run_trial(bridge, xp, idx, snap, "B")
    rep_A_dec, rea_A_dec = _decode(igA, "rep"), _decode(igA, "rea")
    rep_B_dec, rea_B_dec = _decode(igB, "rep"), _decode(igB, "rea")
    print(f"  [ignite A] report={rep_A_dec} reason={rea_A_dec}  ({igA})", flush=True)
    print(f"  [ignite B] report={rep_B_dec} reason={rea_B_dec}  ({igB})", flush=True)
    # identity: on ignite-A both say A; on ignite-B both say B.
    identity = (rep_A_dec == "A" and rea_A_dec == "A" and rep_B_dec == "B" and rea_B_dec == "B")
    # causal swap: report AND reason both flip A->B together (same source).
    causal_swap = (rep_A_dec == "A" and rep_B_dec == "B" and rea_A_dec == "A" and rea_B_dec == "B")

    # ── WORKSPACE LESION: reason collapses; DISSOCIATION: the direct path keeps report alive ────────────────
    bridge_l, xp_l, idx_l, snap_l = build_broadcast_bridge(seed=args.seed, lesion_workspace=True)
    les_ig = _run_trial(bridge_l, xp_l, idx_l, snap_l, "A")                       # try to ignite A, no recurrence
    les_reason_dead = _decode(les_ig, "rea") == "none"                            # reasoning collapses under lesion
    # dissociation: same lesion, but the peripheral DIRECT path drives report_A -> report survives, reason dead.
    les_direct = _run_trial(bridge_l, xp_l, idx_l, snap_l, "A", direct_report="A")
    dissoc_report_alive = _decode(les_direct, "rep") == "A"
    dissoc_reason_dead = _decode(les_direct, "rea") == "none"
    dissociation = bool(les_reason_dead and dissoc_report_alive and dissoc_reason_dead)
    print(f"  [lesion] reason={_decode(les_ig,'rea')} (dead={les_reason_dead}) | "
          f"[lesion+direct] report={_decode(les_direct,'rep')} reason={_decode(les_direct,'rea')} "
          f"(report_alive={dissoc_report_alive} reason_dead={dissoc_reason_dead})", flush=True)

    go = bool(identity and causal_swap and dissociation)

    result = {
        "runner": "_gnw_rung3_report_reasoning_identity_derisk", "go": go, "seed": int(args.seed),
        "backend": args.backend,
        "ignite_A": {"report": rep_A_dec, "reason": rea_A_dec, "rates": igA},
        "ignite_B": {"report": rep_B_dec, "reason": rea_B_dec, "rates": igB},
        "lesion": {"reason": _decode(les_ig, "rea"), "rates": les_ig},
        "lesion_direct": {"report": _decode(les_direct, "rep"), "reason": _decode(les_direct, "rea"), "rates": les_direct},
        "gate_detail": {"identity": bool(identity), "causal_swap": bool(causal_swap),
                        "workspace_lesion_collapses_reason": bool(les_reason_dead),
                        "dissociation": dissociation},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[gnw-rung3] GO={go}  (identity={identity} causal_swap={causal_swap} dissociation={dissociation})", flush=True)
    print(f"[gnw-rung3] wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
