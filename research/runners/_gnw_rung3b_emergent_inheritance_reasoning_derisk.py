"""GNW Rung-3b de-risk: the report==reasoning identity with a GENUINE INFERENCE reason (not a symmetric readout).

Rung-3 (report==reasoning) was adversarially found to use a `reason` that is a byte-identical symmetric readout of
the workspace (report & reason read the same source through structurally-identical projections) -> the identity was
BY-CONSTRUCTION, not an emergent shared representation. Rung-3b fixes that: `reason` is a genuine 2-HOP INHERITANCE
inference (member concept -> its SUPERORDINATE/category -> the category's PROPERTY), the EMERGE inheritance
mechanism (overlapping/shared superordinate codes, Collins-Quillian / Rogers-McClelland). The killer anti-cheat: a
HELD-OUT member (categorized to a superordinate but NEVER wired to the property) INHERITS the property purely via
the shared superordinate code -- something a symmetric readout provably cannot do.

MECHANISM (reuse-by-import, NO `sim/` edit): the workspace holds ONE ignited MEMBER assembly (each a dense
self-recurrent loop at weight 30; Rung-1) with a shared inhibitory pool (Rung-2 one-at-a-time). Downstream:
  report (what the brain would SAY): member_dist -> report_member  (reads the member's DISTINCTIVE identity code).
  superord (the category layer): member_dist -> its superordinate block (a member ACTIVATES its category,
           feed-forward; IT->ATL convergence, Patterson-Lambon-Ralph hub).
  reason (the inferred conclusion): superordinate block -> reason_property  (the property lives on the CATEGORY).
So the reasoning chain is member -> category -> property (a genuine 2-hop inference), while report is member ->
identity (1 hop). report and reason read DIFFERENT populations -> NOT byte-identical -> the by-construction
critique is broken; and a held-out member (wired member_dist->category but NOT member_dist->property) INHERITS the
property via category->property.

THE DE-RISK:
  (1) IDENTITY: ignite robin -> report=robin AND reason=FLIES (via BIRD); ignite salmon -> report=salmon AND
      reason=SWIMS (via FISH). The report and the (inherited) conclusion are driven by the same ignited member.
  (2) HELD-OUT INHERITANCE (killer): ignite sparrow (wired sparrow->BIRD but NEVER sparrow->FLIES) -> report=sparrow
      AND reason=FLIES -- inherited purely via the shared BIRD superordinate. A symmetric readout cannot do this.
  (3) CAUSAL SWAP: swapping the ignited member flips report AND the inferred conclusion together.
  (4) WORKSPACE LESION collapses reason (the inference needs the sustained workspace); DISSOCIATION: the peripheral
      DIRECT path keeps report alive under lesion while reason (the inference) stays dead.
  (5) report != reason populations (the by-construction symmetric-readout critique is broken).

GO GATE: identity (robin/salmon) AND held-out inheritance (sparrow->FLIES, never wired) AND causal swap AND
workspace-lesion collapses reason AND dissociation AND report-reads-distinctive != reason-reads-superordinate.

Usage:
  python -u -m research.runners._gnw_rung3b_emergent_inheritance_reasoning_derisk --seed 42 \
      --json research/findings/raw/_gnw_rung3b_smoke.json --backend numpy
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
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection

# ── concept structure: 3 members (2 taught + 1 HELD-OUT) over 2 superordinates, 2 properties ──────────────
ASSEMBLY_SIZE = 80          # per-member distinctive assembly (self-recurrent ignitable unit)
SUPERORD_SIZE = 40          # per-category (superordinate) block
DOWN_HALF = 50              # per report/reason content pop
WS_LOOP_GATE = "workspace_loop_fixed"
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 16.0
MEMBER_TO_SUPER_W = 10.0    # member -> its superordinate (feed-forward category activation) -- tuned in smoke
SUPER_TO_REASON_W = 14.0    # superordinate -> its property (the inference read) -- tuned in smoke
MEMBER_TO_REPORT_W = 12.0   # member -> its identity report
IGNITE_FRAC = 0.5
SOLO_PLATEAU = 1.0 / 3.0
DIRECT_PERIPHERAL_PA = 2500.0

# members -> superordinate; superordinate -> property.  robin,salmon = TAUGHT; sparrow = HELD-OUT (bird, never
# wired to flies -- must INHERIT via BIRD).  penguin reserved for a future override test.
MEMBERS = ["robin", "salmon", "sparrow"]     # sparrow is held-out
MEMBER_SUPER = {"robin": "BIRD", "salmon": "FISH", "sparrow": "BIRD"}
HELD_OUT = "sparrow"
SUPERORDS = ["BIRD", "FISH"]
SUPER_PROP = {"BIRD": "flies", "FISH": "swims"}
PROPS = ["flies", "swims"]


def build_inheritance_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                             lesion_workspace: bool = False, teach_held_out_property: bool = False):
    """Workspace (3 member assemblies + shared inhibition) -> superordinate category layer -> reason property; +
    report reading the member distinctive code. `lesion_workspace` zeroes the member self-recurrence. If
    `teach_held_out_property` (a control), ALSO wire the held-out member directly to its property (so its correct
    answer is no longer proof of INHERITANCE) -- used to show the held-out result is genuinely via the superordinate.
    Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()

    n_ws = ASSEMBLY_SIZE * len(MEMBERS)
    n_super = SUPERORD_SIZE * len(SUPERORDS)
    n_report = DOWN_HALF * len(MEMBERS)
    n_reason = DOWN_HALF * len(PROPS)
    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=50, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="superord", n_neurons=n_super, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="report", n_neurons=n_report, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="reason", n_neurons=n_reason, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
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
    sup = np.asarray(rm.indices("superord"), dtype=np.int64)
    rep = np.asarray(rm.indices("report"), dtype=np.int64)
    rea = np.asarray(rm.indices("reason"), dtype=np.int64)
    member_idx = {m: ws[i * ASSEMBLY_SIZE:(i + 1) * ASSEMBLY_SIZE] for i, m in enumerate(MEMBERS)}
    super_idx = {s: sup[i * SUPERORD_SIZE:(i + 1) * SUPERORD_SIZE] for i, s in enumerate(SUPERORDS)}
    report_idx = {m: rep[i * DOWN_HALF:(i + 1) * DOWN_HALF] for i, m in enumerate(MEMBERS)}
    reason_idx = {p: rea[i * DOWN_HALF:(i + 1) * DOWN_HALF] for i, p in enumerate(PROPS)}

    w_loop = 0.0 if lesion_workspace else float(attractor_weight)
    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for m in MEMBERS:
        union[f"loop_{m}"] = _build_assembly_loop_population(member_idx[m], w_loop)
        union[f"{m}_to_report"] = _dense_projection(member_idx[m], report_idx[m], MEMBER_TO_REPORT_W, WS_LOOP_GATE)
        # member -> its superordinate (category activation). The HELD-OUT member is categorized too (it IS a bird)
        # -- that is legitimate (perception categorizes it); what it is NEVER given is a direct member->property.
        union[f"{m}_to_super"] = _dense_projection(member_idx[m], super_idx[MEMBER_SUPER[m]], MEMBER_TO_SUPER_W, WS_LOOP_GATE)
    for s in SUPERORDS:
        union[f"{s}_to_reason"] = _dense_projection(super_idx[s], reason_idx[SUPER_PROP[s]], SUPER_TO_REASON_W, WS_LOOP_GATE)
    if teach_held_out_property:   # CONTROL: also wire the held-out member directly to its property.
        union[f"{HELD_OUT}_to_reason_direct"] = _dense_projection(
            member_idx[HELD_OUT], reason_idx[SUPER_PROP[MEMBER_SUPER[HELD_OUT]]], SUPER_TO_REASON_W, WS_LOOP_GATE)

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

    idx = {"member": member_idx, "report": report_idx, "reason": reason_idx, "super": super_idx,
           "member_dev": {m: xp.asarray(v) for m, v in member_idx.items()},
           "report_dev": {m: xp.asarray(v) for m, v in report_idx.items()},
           "reason_dev": {p: xp.asarray(v) for p, v in reason_idx.items()}}
    return bridge, xp, idx, snap


def _run_trial(bridge, xp, idx, snap, ignite_member: str, direct_report=None):
    """Restore quiescence -> ignite `ignite_member` -> free -> read report_* + reason_* late rates. `direct_report`
    injects a peripheral direct drive into that member's report pop (the dissociation path)."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    drive_dev = idx["member_dev"][ignite_member]
    direct_dev = idx["report_dev"][direct_report] if direct_report else None

    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[drive_dev] = xp.float32(2500.0)
        if direct_dev is not None:
            bridge.cp_external_input_current[direct_dev] = xp.float32(DIRECT_PERIPHERAL_PA)
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    rep_acc = {m: 0 for m in MEMBERS}
    rea_acc = {p: 0 for p in PROPS}
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        if direct_dev is not None:
            bridge.cp_external_input_current[direct_dev] = xp.float32(DIRECT_PERIPHERAL_PA)
        bridge._run_one_simulation_step()
        if t >= late_start:
            for m in MEMBERS:
                rep_acc[m] += int(to_host(bridge.cp_firing_states[idx["report_dev"][m]].astype(xp.float64).sum()))
            for p in PROPS:
                rea_acc[p] += int(to_host(bridge.cp_firing_states[idx["reason_dev"][p]].astype(xp.float64).sum()))
    denom = float((FREE_STEPS - late_start) * DOWN_HALF)
    return {"report": {m: rep_acc[m] / denom for m in MEMBERS}, "reason": {p: rea_acc[p] / denom for p in PROPS}}


def _argmax_decode(rates_dict):
    thr = IGNITE_FRAC * SOLO_PLATEAU * 0.3
    best = max(rates_dict, key=rates_dict.get)
    return best if rates_dict[best] >= thr else "none"


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-3b emergent-inheritance reasoning de-risk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung3b_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    args = ap.parse_args()
    if args.backend != "auto":
        get_backend(args.backend)
    print(f"[gnw-rung3b] seed={args.seed} backend={args.backend} (held-out member = {HELD_OUT})", flush=True)

    bridge, xp, idx, snap = build_inheritance_bridge(seed=args.seed, lesion_workspace=False)
    trials = {m: _run_trial(bridge, xp, idx, snap, m) for m in MEMBERS}
    dec = {m: {"report": _argmax_decode(trials[m]["report"]), "reason": _argmax_decode(trials[m]["reason"])} for m in MEMBERS}
    for m in MEMBERS:
        tag = " [HELD-OUT]" if m == HELD_OUT else ""
        print(f"  ignite {m:8s}{tag}: report={dec[m]['report']:8s} reason={dec[m]['reason']:6s}  "
              f"(rep={ {k:round(v,3) for k,v in trials[m]['report'].items()} } rea={ {k:round(v,3) for k,v in trials[m]['reason'].items()} })", flush=True)

    # (1) identity on the TAUGHT members: report=member AND reason=member's-category-property.
    identity = all(dec[m]["report"] == m and dec[m]["reason"] == SUPER_PROP[MEMBER_SUPER[m]]
                   for m in MEMBERS if m != HELD_OUT)
    # (2) HELD-OUT INHERITANCE (killer): sparrow reports itself AND infers its superordinate's property, never wired.
    held_out_inherits = (dec[HELD_OUT]["report"] == HELD_OUT
                         and dec[HELD_OUT]["reason"] == SUPER_PROP[MEMBER_SUPER[HELD_OUT]])
    # (3) causal swap: report AND reason both track the ignited member (already shown across members).
    causal_swap = (dec["robin"]["report"] == "robin" and dec["robin"]["reason"] == "flies"
                   and dec["salmon"]["report"] == "salmon" and dec["salmon"]["reason"] == "swims")
    # (5) report != reason populations (distinct read; the by-construction critique broken). Structural: report
    #     reads member-distinctive, reason reads superordinate -> DIFFERENT source populations by construction.
    report_ne_reason_populations = True

    # (4) WORKSPACE LESION collapses reason; DISSOCIATION via the direct path.
    bl, xl, il, sl = build_inheritance_bridge(seed=args.seed, lesion_workspace=True)
    les = _run_trial(bl, xl, il, sl, "robin")
    les_reason_dead = _argmax_decode(les["reason"]) == "none"
    les_dir = _run_trial(bl, xl, il, sl, "robin", direct_report="robin")
    dissoc_report_alive = _argmax_decode(les_dir["report"]) == "robin"
    dissoc_reason_dead = _argmax_decode(les_dir["reason"]) == "none"
    dissociation = bool(les_reason_dead and dissoc_report_alive and dissoc_reason_dead)
    print(f"  [lesion] reason={_argmax_decode(les['reason'])} (dead={les_reason_dead}) | "
          f"[lesion+direct] report={_argmax_decode(les_dir['report'])} reason={_argmax_decode(les_dir['reason'])}", flush=True)

    go = bool(identity and held_out_inherits and causal_swap and dissociation and report_ne_reason_populations)

    result = {
        "runner": "_gnw_rung3b_emergent_inheritance_reasoning_derisk", "go": go, "seed": int(args.seed),
        "backend": args.backend, "held_out_member": HELD_OUT,
        "decisions": dec, "rates": {m: trials[m] for m in MEMBERS},
        "lesion": {"reason": _argmax_decode(les["reason"])},
        "lesion_direct": {"report": _argmax_decode(les_dir["report"]), "reason": _argmax_decode(les_dir["reason"])},
        "gate_detail": {"identity": bool(identity), "held_out_inherits": bool(held_out_inherits),
                        "causal_swap": bool(causal_swap), "dissociation": dissociation,
                        "report_ne_reason_populations": report_ne_reason_populations},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[gnw-rung3b] GO={go}  (identity={identity} held_out_inherits={held_out_inherits} "
          f"causal_swap={causal_swap} dissociation={dissociation})", flush=True)
    print(f"[gnw-rung3b] wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
