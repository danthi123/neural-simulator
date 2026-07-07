"""GNW Rung-3c.1 de-risk: the workspace reasons by inheritance through a LEARNED (not hand-wired) category.

Rung-3b's adversarial-verify confirmed a GENUINE 2-hop held-out inheritance BUT scoped it: the category STRUCTURE
(member->superordinate) was HAND-WIRED. Rung-3c.1 closes that: the member->superordinate categorization is LEARNED
from CO-OCCURRENCE experience (EMERGE-30 / competitive-Hebbian category formation, Cui-Ahmad-Hawkins 2017,
Rogers-McClelland). Members that co-occur with the same CONTEXT tokens (the environment) develop a shared learned
representation; a class property placed on that emergent context is inherited -- and a HELD-OUT member (streamed
with the context, never taught the property) inherits via the LEARNED grouping. The killer anti-cheat: PERMUTED-
CONTEXT (members co-occur with a SCRAMBLED context -> no coherent category emerges -> held-out inheritance collapses),
isolating the LEARNED co-occurrence (not any hand-placement) as the cause.

MECHANISM (reuse-by-import, NO `sim/` edit): the on-bridge rate-Hebbian co-occurrence pattern (the validated
`train_*_convergence` in nav_conv_merged_bridge.py: a PLASTIC near-floor pathway, everything else frozen, co-drive
pre+post, Hebbian strengthens co-active). Regions: workspace (member self-recurrent assemblies, Rung-1) + `context`
(the environment's context tokens = the substrate on which the superordinate EMERGES) + report + reason. The
member->context pathway is PLASTIC (Hebbian, near-floor init); context->reason (property) + member->report
(identity) are fixed. STREAM: co-drive each member's assembly + its category's context tokens -> the Hebbian LEARNS
member->context (all bird-members converge on the bird-context cells = the emergent BIRD superordinate). Freeze.
QUERY: ignite a member -> its LEARNED member->context fires the emergent superordinate -> context->reason (property).

THE DE-RISK (6-seed):
  (1) LEARNED IDENTITY+INHERITANCE: after streaming, ignite robin -> report=robin AND reason=flies (via the LEARNED
      robin->bird-context->flies); salmon -> salmon/swims.
  (2) HELD-OUT (killer): the held-out member streamed with bird-context but NEVER taught the property -> ignite it ->
      reason=flies, inherited via the LEARNED shared context.
  (3) PERMUTED-CONTEXT anti-cheat: re-stream with each member's context SCRAMBLED -> no coherent superordinate ->
      held-out inheritance collapses (proves the LEARNED co-occurrence carries the structure, not a hand-placement).
  (4) NO-LEARNING: skip the stream -> member->context stays at floor -> reason abstains (none).
GO GATE: learned identity + held-out inheritance (learned) + permuted-context collapses it + no-learning abstains.

Usage:
  python -u -m research.runners._gnw_rung3c_learned_inheritance_derisk --seed 42 \
      --json research/findings/raw/_gnw_rung3c_smoke.json --backend numpy
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

ASSEMBLY_SIZE = 80
CONTEXT_PER_CAT = 40        # context cells the environment provides per category (bird-context, fish-context)
DOWN_HALF = 50
WS_LOOP_GATE = "workspace_loop_fixed"
LEARN_CAT_GATE = "learn_cat"          # the ONLY plastic gate (member->context); frozen after streaming
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 16.0
MEMBER_TO_CONTEXT_INIT = 0.01         # FLOOR init: the all-to-all must NOT itself drive the context (else every
                                      # member co-fires every context -> non-selective runaway of the pure-potentiation
                                      # rule). The context fires ONLY from the external (environment) drive during
                                      # streaming, so the pre&post coincidence is SELECTIVE (robin fires + the DRIVEN
                                      # bird-context fires -> robin->bird grows; fish silent -> robin->fish stays floor).
CONTEXT_TO_REASON_W = 14.0
MEMBER_TO_REPORT_W = 12.0
IGNITE_FRAC = 0.5
SOLO_PLATEAU = 1.0 / 3.0
HEBB_RATE = 0.05
HEBB_MAX = 20.0
STREAM_EPOCHS = 45         # more co-occurrence -> the DRIVEN-category coincidence dominates any noise-induced
                           # spurious cross-category coincidence (firms the per-seed selectivity to 6/6)
STREAM_STEPS = 20          # co-fire steps per (member, context) presentation
MEMBER_DRIVE_PA = 2500.0
CONTEXT_DRIVE_PA = 2200.0   # drive the DRIVEN category strongly so its coincidence dominates any noise-induced
                           # spurious firing of the OTHER category (which the pure-potentiation Hebbian would lock in)
QUERY_DRIVE_PA = 4200.0     # heterogeneity raises the ignition threshold (Rank-2b); the QUERY needs a stronger drive
                           # than the stream co-fire to reliably ignite the member assembly under het+noise.

MEMBERS = ["robin", "salmon", "canary"]     # canary is HELD-OUT (streamed w/ bird-context, never taught property)
MEMBER_CAT = {"robin": "BIRD", "salmon": "FISH", "canary": "BIRD"}
HELD_OUT = "canary"
CATS = ["BIRD", "FISH"]
CAT_PROP = {"BIRD": "flies", "FISH": "swims"}
PROPS = ["flies", "swims"]


def build_learned_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                         heterogeneity: bool = False, ou_noise_pA: float = 0.0):
    """Workspace (member assemblies + inhibition) + context (emergent superordinate) + report + reason. The
    member->context pathway is PLASTIC (Hebbian, near-floor init); context->reason + member->report are fixed.
    NOTE (adversarial-verify wvfbcc9sn, corrected): heterogeneity + OU noise are DEFAULT OFF. An earlier hypothesis
    that they were REQUIRED (to desynchronise a period-3 limit cycle so the spike-timing Hebbian coincidence fires)
    was REFUTED: during streaming BOTH the member and its context are force-driven every step (MEMBER_DRIVE_PA /
    CONTEXT_DRIVE_PA), so the internal firing phase is irrelevant and the coincidence fires densely regardless. The
    real fixes were `hebbian_weight_decay=0` (so the sparse potentiation accumulates) + a strong context drive. With
    het+noise OFF the de-risk is 6/6; het threshold-variance was in fact the SOLE cause of the seed-44 failure."""
    xp, _ = get_backend()
    n_ws = ASSEMBLY_SIZE * len(MEMBERS)
    n_ctx = CONTEXT_PER_CAT * len(CATS)
    n_rep = DOWN_HALF * len(MEMBERS)
    n_rea = DOWN_HALF * len(PROPS)
    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=50, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="context", n_neurons=n_ctx, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="report", n_neurons=n_rep, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="reason", n_neurons=n_rea, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    # member -> context: ALL-TO-ALL plastic near-floor (the Hebbian carves which context each member drives).
    # the FS mutual-inhibition pathways are tagged with WS_LOOP_GATE (held at 0) so the global Hebbian cannot drift
    # them during streaming (plastic=False alone does not stop the potentiation of co-active synapses; the frozen
    # GATE zeroes their plasticity RATE while leaving their inhibitory TRANSMISSION intact). member->context is the
    # ONLY learning pathway (LEARN_CAT_GATE).
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False, plasticity_gate=WS_LOOP_GATE),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False, plasticity_gate=WS_LOOP_GATE),
        RegionPathway(from_region="workspace", to_region="context", density=1.0,
                      weight_mean=MEMBER_TO_CONTEXT_INIT, weight_jitter=0.0, plastic=True,
                      plasticity_gate=LEARN_CAT_GATE),
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
    # Hebbian ON (gated to member->context only; every other pathway is plastic=False -> frozen). STDP/reward/
    # homeostasis/STP/structural OFF, OU/heterogeneity OFF (deterministic).
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = HEBB_RATE
    cfg.hebbian_max_weight = HEBB_MAX
    # ZERO the every-step Hebbian weight decay (bridge.py:6948). It collapses the SPARSE spike-timing potentiation
    # (only ~4% of member->context synapses coincide per presentation, but the decay hits ALL gain>0 synapses every
    # step) to a low equilibrium (~0.05, far below the ~10 needed to drive the context). With decay 0 the selective
    # co-occurrence potentiation ACCUMULATES. (The self-loops/report/reason are gain 0 -> unaffected by decay anyway.)
    cfg.hebbian_weight_decay = 0.0
    for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    ctx = np.asarray(rm.indices("context"), dtype=np.int64)
    rep = np.asarray(rm.indices("report"), dtype=np.int64)
    rea = np.asarray(rm.indices("reason"), dtype=np.int64)
    member_idx = {m: ws[i * ASSEMBLY_SIZE:(i + 1) * ASSEMBLY_SIZE] for i, m in enumerate(MEMBERS)}
    cat_ctx_idx = {c: ctx[i * CONTEXT_PER_CAT:(i + 1) * CONTEXT_PER_CAT] for i, c in enumerate(CATS)}
    report_idx = {m: rep[i * DOWN_HALF:(i + 1) * DOWN_HALF] for i, m in enumerate(MEMBERS)}
    reason_idx = {p: rea[i * DOWN_HALF:(i + 1) * DOWN_HALF] for i, p in enumerate(PROPS)}

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for m in MEMBERS:
        union[f"loop_{m}"] = _build_assembly_loop_population(member_idx[m], float(attractor_weight))
        union[f"{m}_to_report"] = _dense_projection(member_idx[m], report_idx[m], MEMBER_TO_REPORT_W, WS_LOOP_GATE)
    # the PROPERTY lives on the emergent context (category) cells: context-cat -> its property (fixed, "taught").
    for c in CATS:
        union[f"{c}ctx_to_reason"] = _dense_projection(cat_ctx_idx[c], reason_idx[CAT_PROP[c]], CONTEXT_TO_REASON_W, WS_LOOP_GATE)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)
    bridge.set_plasticity_gate(LEARN_CAT_GATE, 0.0)   # closed until streaming

    idx = {"member": member_idx, "cat_ctx": cat_ctx_idx, "report": report_idx, "reason": reason_idx,
           "member_dev": {m: xp.asarray(v) for m, v in member_idx.items()},
           "cat_ctx_dev": {c: xp.asarray(v) for c, v in cat_ctx_idx.items()},
           "report_dev": {m: xp.asarray(v) for m, v in report_idx.items()},
           "reason_dev": {p: xp.asarray(v) for p, v in reason_idx.items()}}
    return bridge, xp, idx


def stream_categorization(bridge, xp, idx, permute_seed=None):
    """STREAM: co-drive each member + its category's context tokens -> Hebbian LEARNS member->context. If
    `permute_seed` is not None, each member is co-driven with a SCRAMBLED (random) context (the permuted-context
    anti-cheat) -> no coherent category emerges. Each presentation starts from a CLEAN quiescent state (restore
    the pre-stream snapshot's v/u/firing/conductances) so the self-recurrent member ignition does not carry over
    between presentations (the Rung-1 limit-cycle latch would otherwise cross-contaminate the co-occurrence)."""
    # a clean quiescent dynamical state (weights change during streaming, but the rest state does not).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    q0 = _snapshot_state(bridge, xp)

    bridge.set_plasticity_gate(LEARN_CAT_GATE, 1.0)   # open the member->context Hebbian
    if permute_seed is not None:
        # PERMUTED-CONTEXT anti-cheat: SHUFFLE which category-context each member co-occurs with (a derangement so
        # the held-out member co-occurs with a DIFFERENT category than its true one). The category STRUCTURE is thus
        # broken: the held-out member no longer shares the BIRD-context with the taught bird -> it learns a different
        # member->context -> it infers a DIFFERENT (or no) property. This isolates the LEARNED co-occurrence (not any
        # hand-placement) as what carries the inheritance.
        rng = np.random.default_rng(int(permute_seed))
        assign = None
        for _ in range(50):
            perm = list(rng.permutation(len(MEMBERS)))
            cand = {MEMBERS[i]: MEMBER_CAT[MEMBERS[perm[i]]] for i in range(len(MEMBERS))}
            if cand[HELD_OUT] != MEMBER_CAT[HELD_OUT]:   # the held-out MUST be displaced to a different category
                assign = cand
                break
        assign = assign or {m: ("FISH" if MEMBER_CAT[m] == "BIRD" else "BIRD") for m in MEMBERS}  # fallback: flip
        member_ctx = {m: idx["cat_ctx_dev"][assign[m]] for m in MEMBERS}
    else:
        member_ctx = {m: idx["cat_ctx_dev"][MEMBER_CAT[m]] for m in MEMBERS}

    for _ in range(STREAM_EPOCHS):
        for m in MEMBERS:
            _restore_state(bridge, q0)                # CLEAN slate per presentation (no ignition carry-over)
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(STREAM_STEPS):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[idx["member_dev"][m]] = xp.float32(MEMBER_DRIVE_PA)
                bridge.cp_external_input_current[member_ctx[m]] = xp.float32(CONTEXT_DRIVE_PA)
                bridge._run_one_simulation_step()
    bridge.set_plasticity_gate(LEARN_CAT_GATE, 0.0)   # FREEZE the learned categorization
    _restore_state(bridge, q0)                        # the query snapshot is the CLEAN quiescent state + learned weights
    bridge.cp_external_input_current[:] = 0.0
    return _snapshot_state(bridge, xp)


def _run_query(bridge, xp, idx, snap, ignite_member: str):
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx["member_dev"][ignite_member]] = xp.float32(QUERY_DRIVE_PA)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    rep_acc = {m: 0 for m in MEMBERS}
    rea_acc = {p: 0 for p in PROPS}
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            for m in MEMBERS:
                rep_acc[m] += int(to_host(bridge.cp_firing_states[idx["report_dev"][m]].astype(xp.float64).sum()))
            for p in PROPS:
                rea_acc[p] += int(to_host(bridge.cp_firing_states[idx["reason_dev"][p]].astype(xp.float64).sum()))
    denom = float((FREE_STEPS - late_start) * DOWN_HALF)
    return {"report": {m: rep_acc[m] / denom for m in MEMBERS}, "reason": {p: rea_acc[p] / denom for p in PROPS}}


def _decode(rates):
    thr = IGNITE_FRAC * SOLO_PLATEAU * 0.2
    best = max(rates, key=rates.get)
    return best if rates[best] >= thr else "none"


def _held_out_inherits(bridge, xp, idx, snap):
    q = _run_query(bridge, xp, idx, snap, HELD_OUT)
    return _decode(q["report"]) == HELD_OUT and _decode(q["reason"]) == CAT_PROP[MEMBER_CAT[HELD_OUT]], q


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-3c.1 LEARNED-inheritance de-risk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung3c_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    args = ap.parse_args()
    if args.backend != "auto":
        get_backend(args.backend)
    print(f"[gnw-rung3c] seed={args.seed} (held-out={HELD_OUT}; category LEARNED from co-occurrence)", flush=True)

    # MAIN: learn the categorization from real co-occurrence.
    bridge, xp, idx = build_learned_bridge(seed=args.seed)
    snap = stream_categorization(bridge, xp, idx, permute_seed=None)
    trials = {m: _run_query(bridge, xp, idx, snap, m) for m in MEMBERS}
    dec = {m: {"report": _decode(trials[m]["report"]), "reason": _decode(trials[m]["reason"])} for m in MEMBERS}
    for m in MEMBERS:
        tag = " [HELD-OUT]" if m == HELD_OUT else ""
        print(f"  ignite {m:8s}{tag}: report={dec[m]['report']:8s} reason={dec[m]['reason']:6s}  "
              f"(rea={ {k: round(v,3) for k,v in trials[m]['reason'].items()} })", flush=True)
    identity = all(dec[m]["report"] == m and dec[m]["reason"] == CAT_PROP[MEMBER_CAT[m]]
                   for m in MEMBERS if m != HELD_OUT)
    held_out_inherits = dec[HELD_OUT]["report"] == HELD_OUT and dec[HELD_OUT]["reason"] == CAT_PROP[MEMBER_CAT[HELD_OUT]]

    # PERMUTED-CONTEXT anti-cheat: re-learn with scrambled context -> held-out inheritance should COLLAPSE.
    bp, xpp, ip = build_learned_bridge(seed=args.seed)
    snp = stream_categorization(bp, xpp, ip, permute_seed=args.seed + 777)
    perm_ok, perm_q = _held_out_inherits(bp, xpp, ip, snp)
    permuted_collapses = not perm_ok

    # NO-LEARNING anti-cheat: skip the stream -> member->context at floor -> reason abstains.
    bn, xpn, iin = build_learned_bridge(seed=args.seed)
    bn.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bn._run_one_simulation_step()
    snap_n = _snapshot_state(bn, xpn)
    nl_q = _run_query(bn, xpn, iin, snap_n, HELD_OUT)
    no_learning_abstains = _decode(nl_q["reason"]) == "none"

    go = bool(identity and held_out_inherits and permuted_collapses and no_learning_abstains)
    print(f"  [permuted-context] held-out reason={_decode(perm_q['reason'])} -> collapses={permuted_collapses}", flush=True)
    print(f"  [no-learning] held-out reason={_decode(nl_q['reason'])} -> abstains={no_learning_abstains}", flush=True)

    result = {
        "runner": "_gnw_rung3c_learned_inheritance_derisk", "go": go, "seed": int(args.seed),
        "backend": args.backend, "held_out_member": HELD_OUT,
        "decisions": dec, "rates": {m: trials[m] for m in MEMBERS},
        "permuted_context": {"held_out_reason": _decode(perm_q["reason"]), "collapses": permuted_collapses},
        "no_learning": {"held_out_reason": _decode(nl_q["reason"]), "abstains": no_learning_abstains},
        "gate_detail": {"identity": bool(identity), "held_out_inherits": bool(held_out_inherits),
                        "permuted_collapses": bool(permuted_collapses), "no_learning_abstains": bool(no_learning_abstains)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[gnw-rung3c] GO={go}  (identity={identity} held_out_inherits={held_out_inherits} "
          f"permuted_collapses={permuted_collapses} no_learning_abstains={no_learning_abstains})", flush=True)
    print(f"[gnw-rung3c] wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
