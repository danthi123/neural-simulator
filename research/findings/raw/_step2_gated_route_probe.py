"""Step 2 — Task 1 DE-RISK probe: parser-role-gated synaptic routing on the unified bridge.

THE ONE FALSIFIABLE QUESTION (design doc + step-2 plan Task 1):
On the unified bridge (the conversational PARSER at neuron offset 0 and the COMPOSER at offset 126 on ONE
`SimulationBridge`), can the parser's role-ensemble FIRING — when the parser assigns a word to a role — OPEN a
transmission gate that routes a drive SELECTIVELY to that role's target and NOT the others? If yes, step 2
(replacing the Python `{role: word}` hand-off with a synaptic route) is de-risked. If no, step 2 needs a
different design — reported honestly (no hacked pass).

WHAT THIS PROBE TESTS (the routing MECHANISM only — NOT the composer's bind; that is step-2 Task 2):
On the SAME bridge that holds the trained parser, add one MARKER input pool and three TARGET pools
(agent_target / action_target / patient_target). Wire strong excitatory routes  marker → <role>_target  each
tagged with a `transmission_gate` named `route_<role>`, and couple each gate to that role's parser ensemble so
the ensemble's firing OPENS the gate. Then:

  * Drive the parser conjunction for (position 0, active)  → the AGENT ensemble fires → `route_agent` opens →
    the marker reaches agent_target ONLY (action/patient gates stay closed → their targets stay silent).
  * Drive (position 2, active)  → the PATIENT ensemble fires → the SAME marker reaches patient_target ONLY.
    (A re-bind of the same source to a different role purely by changing which parser conjunction is driven —
     ZERO weight change. A grown-weight model could not do this.)
  * Control: with NO parser drive (no ensemble firing) the marker reaches NO target (all gates closed).

Terms (defined once):
  * transmission gate  = a per-pathway multiplier on a route's effective synaptic CURRENT in [0,1], opened/closed
    at runtime WITHOUT changing weights (`bridge.set_transmission_gate(name, value)` /
    `cp_transmission_gain`; shipped + validated in `tests/test_transmission_gate.py`).
  * gate↔pool coupling = a per-step hook (`bridge._apply_gate_couplings`, run inside
    `_run_one_simulation_step`) that opens a gate when a control pool's EMA-smoothed firing rate ≥ threshold.
  * role ensemble      = the parser's output neurons for one role (agent/action/patient), at
    `BridgeParser.role_idx[role]` (offset onto the shared bridge).
  * marker / target pools = small fresh neuron blocks added past the composer slice for THIS de-risk (the
    composer's own fill banks are NOT used here — bind integration is Task 2).

>>> WHY A LOCAL COUPLING HELPER (and NOT `bridge.couple_gate_to_pool`):
The shipped public `bridge.couple_gate_to_pool(gate_name, control_region_name)` resolves the control pool by
REGION NAME and REQUIRES the brain-region framework (`region_manager`); it raises RuntimeError when
`region_manager is None`. The unified bridge is built WITHOUT the brain-region framework — its parser/composer
are `inject_explicit_wiring` populations addressed by RAW neuron indices, and `region_manager` is None. So the
public name-based entry point does not fit this bridge.

The per-step gating LOGIC, however, is identical and index-based: `couple_gate_to_pool` only uses the region
name to look up `control_idx` (an int array of the control pool's neuron indices), then appends a coupling dict
to `bridge._gate_couplings`; `_apply_gate_couplings` thereafter reads `control_idx` (RAW indices) every step.
So `_couple_gate_to_indices` below appends the SAME coupling-dict shape with the parser's raw `role_idx[role]`
indices, reusing the shipped, validated `_apply_gate_couplings` hook + `set_transmission_gate` UNCHANGED. This
is a runner-side wiring helper, NOT a `sim/` edit (the gating primitives are public; only the wire-time
name→indices resolution — the part that needs `region_manager` — is replaced by indices we already hold).

This IS a de-risk finding to surface: step-2 Task 2's `hear_synaptic` will need either this same index-coupling
helper or a small additive `sim/` overload of `couple_gate_to_pool` that accepts indices (the unified bridge has
no region_manager). That is a Task-2 design decision; it does not change the answer to the Task-1 question.

Runs on the validated production (CuPy/GPU) backend (the parser's Hebbian convergence + the gate operating
point are GPU-validated, not NumPy). proj_dim=64 is small + fast.
"""
from __future__ import annotations

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host

from research.runners.unified_brain_bridge import (
    PARSER_SLICE_SIZE, merge_population_into_shared_bridge,
)
from research.runners.brain_conversational_agent import BridgeParser
from research.runners.core_sim_composition import CoreSimComposer

# De-risk pool sizes + the route operating point (mirrors the validated transmission_gate test: 40→40
# all-to-all at weight 300, drive 1500 pA → an OPEN gate makes the target fire, a CLOSED gate keeps it silent).
N_MARKER = 40
N_TARGET = 40
ROUTE_WEIGHT = 300.0
ROUTE_GATE_PLASTICITY = "step2_route_fixed"   # one plasticity gate over ALL marker→target routes (held 0.0)
MARKER_DRIVE_PA = 1500.0
PARSER_DRIVE_PA = 2500.0                       # the parser's own conjunction/ensemble drive (BridgeParser.drive)
GATE_THRESHOLD = 0.05                          # couple_gate_to_pool default; parser ensembles fire well above
READOUT_STEPS = 80
SETTLE_STEPS = 40
ROLES = ("agent", "action", "patient")


def _couple_gate_to_indices(bridge, gate_name, control_idx, threshold=GATE_THRESHOLD, alpha=0.3,
                            open_value=1.0):
    """Append an activity-driven gate↔pool coupling using RAW neuron indices (the parser's role ensemble),
    reusing the shipped per-step `_apply_gate_couplings` hook UNCHANGED. This is exactly the coupling dict that
    `bridge.couple_gate_to_pool` builds internally — only the name→indices resolution (which needs the
    brain-region framework the unified bridge lacks) is replaced by indices we already hold. No `sim/` edit.

    See this module's docstring (">>> WHY A LOCAL COUPLING HELPER") for the rationale + the Task-2 implication.
    """
    if gate_name not in bridge._transmission_gate_to_synapses:
        raise KeyError(f"No transmission gate named '{gate_name}'. "
                       f"Known: {list(bridge._transmission_gate_to_synapses.keys())}")
    xp, _ = get_backend()
    bridge._gate_couplings.append({
        "gate_name": gate_name,
        "control_idx": xp.asarray(np.asarray(control_idx, dtype=np.int64)),
        "threshold": float(threshold), "alpha": float(alpha), "open_value": float(open_value),
        "ema": 0.0, "last_value": None,
    })


def build_probe_bridge(seed=42, proj_dim=64):
    """Build ONE SimulationBridge holding the trained parser (offset 0) + composer (offset 126) — the faithful
    UNIFIED arrangement — PLUS, past the composer slice, a marker pool and three role-target pools with gated
    marker→target routes, each gate coupled to its parser role ensemble.

    Returns a dict of everything the readout needs: bridge, parser, composer, the marker/target index arrays,
    and the gate names. Wiring/training ORDER mirrors `UnifiedBrainBridge.__init__` (composer first, parser
    next deferred, marker/routes, parser trained LAST so no later re-injection resets the trained weights).
    """
    proj_dim = int(proj_dim)
    composer_slice = 8 * proj_dim
    composer_offset = PARSER_SLICE_SIZE                       # 126
    marker_offset = composer_offset + composer_slice         # past parser + composer
    target_offsets = {r: marker_offset + N_MARKER + i * N_TARGET for i, r in enumerate(ROLES)}
    total = marker_offset + N_MARKER + len(ROLES) * N_TARGET

    # Config matches the parser's / unified bridge's (Izhikevich, dt=1ms, global Hebbian ON for the parser;
    # the FIXED composer + route populations are protected by plasticity gates, not by the global flag).
    cfg = CoreSimConfig()
    cfg.num_neurons = total
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_max_weight = 400.0
    cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # 1) Composer first (FIXED bind population, gated to 0.0) — faithful to the unified build order.
    composer = CoreSimComposer(seed=seed, proj_dim=proj_dim, shared_bridge=bridge,
                               index_offset=composer_offset, concepts=_synthetic_concepts(proj_dim))
    # 2) Parser next: plastic "parse" population at offset 0; DEFER training until all populations are wired.
    parser = BridgeParser(seed=seed, shared_bridge=bridge, index_offset=0, defer_train=True)

    # 3) Marker + three gated role routes (marker → <role>_target), each tagged transmission_gate=route_<role>
    #    AND plasticity_gate=ROUTE_GATE_PLASTICITY (held 0.0: FIXED weights on a Hebbian-ON bridge — Task-1
    #    finding — so Hebbian decay cannot drift them; this keeps the zero-weight-change assertion clean).
    marker = np.arange(marker_offset, marker_offset + N_MARKER, dtype=np.int64)
    targets = {r: np.arange(target_offsets[r], target_offsets[r] + N_TARGET, dtype=np.int64) for r in ROLES}
    gate_of = {r: f"route_{r}" for r in ROLES}
    route_plan = {}
    for r in ROLES:
        pre, post = [], []
        for s in marker:                                     # all-to-all marker → this role's target
            for t in targets[r]:
                pre.append(int(s)); post.append(int(t))
        route_plan[f"route_{r}_pop"] = {
            "pre_indices": pre, "post_indices": post,
            "initial_weights": np.full(len(pre), ROUTE_WEIGHT, dtype=np.float32),
            "plastic": False, "plasticity_gate": ROUTE_GATE_PLASTICITY,
            "transmission_gate": gate_of[r], "conn_type": "E_TO_E", "count": len(pre),
        }
    merge_population_into_shared_bridge(bridge, route_plan, gates_to_zero=(ROUTE_GATE_PLASTICITY,))

    # Couple each route gate to its parser role ensemble (raw indices — see helper docstring).
    for r in ROLES:
        _couple_gate_to_indices(bridge, gate_of[r], parser.role_idx[r])

    # 4) Train the parser LAST (no further wiring/re-injection follows → trained weights persist; the gated
    #    composer + route weights stay frozen under this global-Hebbian training).
    parser.train()

    xp, _ = get_backend()
    return {
        "bridge": bridge, "parser": parser, "composer": composer,
        "marker": xp.asarray(marker), "targets": {r: xp.asarray(targets[r]) for r in ROLES},
        "gate_of": gate_of, "proj_dim": proj_dim,
    }


def _synthetic_concepts(proj_dim=64, seed=0):
    """8 orthonormal concept codes (no `denoise64` cache dependency) — same helper the step-1 shared-bridge
    tests use. The composer is incidental to this de-risk; it is wired only to reproduce the unified bridge's
    exact neuron arrangement so the parser ensembles sit at the production offsets."""
    rng = np.random.default_rng(seed)
    words = ["dog", "cat", "go", "come", "north", "south", "river", "look"]
    q, _ = np.linalg.qr(rng.standard_normal((proj_dim, proj_dim)))
    return {w: q[i] for i, w in enumerate(words)}


def _close_all_gates_and_reset_couplings(probe):
    """Close every route gate and reset each gate-coupling EMA so a previous condition's open gate does not
    leak into the next. (`merge_population_into_shared_bridge` leaves transmission gates at the inject default
    1.0=OPEN; the couplings then drive them — but between conditions we re-arm from a known CLOSED state.)"""
    bridge = probe["bridge"]
    for r in ROLES:
        bridge.set_transmission_gate(probe["gate_of"][r], 0.0)
    for c in bridge._gate_couplings:
        c["ema"] = 0.0
        c["last_value"] = None


def _quiet_settle(bridge, n=SETTLE_STEPS):
    """Run the bridge with NO external drive so membrane state + gate-coupling EMAs decay between conditions."""
    cfg = bridge.core_config
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(n):
        bridge.runtime_state.current_time_ms += cfg.dt_ms   # CLAUDE.md gotcha: step does NOT advance the clock
        bridge._run_one_simulation_step()


def _run_condition(probe, position=None, drive_marker=True, readout_steps=READOUT_STEPS):
    """One condition. If `position` is not None, drive the parser conjunction k = position*2 + 0 (active voice)
    — which should fire the role ensemble the parser learned for that position (pos 0→agent, 2→patient). If
    `drive_marker`, also drive the marker pool. Run a readout window with the gate-couplings live, accumulating
    (a) each role TARGET's firing rate and (b) the driven role ensemble's own firing rate (non-vacuity check).

    Returns {agent_target, action_target, patient_target, ensemble_rate}. Rates are per-step firing fractions.
    """
    bridge = probe["bridge"]; parser = probe["parser"]; cfg = bridge.core_config
    xp, _ = get_backend()

    _close_all_gates_and_reset_couplings(probe)
    _quiet_settle(bridge)

    cur = xp.zeros(cfg.num_neurons, dtype=xp.float32)
    driven_role = None
    if position is not None:
        k = position * 2 + 0                                 # active voice
        cur[parser.conj_arr[k]] = PARSER_DRIVE_PA            # drive the conjunction; the LEARNED route fires
        driven_role = {0: "agent", 2: "patient"}[position]   # parser ground truth (active)
    if drive_marker:
        cur[probe["marker"]] = MARKER_DRIVE_PA
    bridge.cp_external_input_current[:] = cur

    acc_t = {r: 0.0 for r in ROLES}
    acc_ens = 0.0
    for _ in range(readout_steps):
        bridge.runtime_state.current_time_ms += cfg.dt_ms
        bridge._run_one_simulation_step()
        fired = bridge.cp_firing_states
        for r in ROLES:
            acc_t[r] += float(to_host(fired[probe["targets"][r]].astype(xp.float64).mean()))
        if driven_role is not None:
            acc_ens += float(to_host(fired[parser.role_arr[driven_role]].astype(xp.float64).mean()))
    bridge.cp_external_input_current[:] = 0.0

    out = {f"{r}_target": acc_t[r] / readout_steps for r in ROLES}
    out["ensemble_rate"] = (acc_ens / readout_steps) if driven_role is not None else 0.0
    return out


def _route_weights(probe):
    """Read every marker→target ROUTE synapse weight from the bridge CSR (the load-bearing quantity for the
    thalamocortical claim: the ROUTE must re-bind with ZERO weight change). This is measured over the route
    population ONLY — NOT the whole bridge: the parser's `"parse"` population is plastic under global Hebbian,
    so it LEGITIMATELY drifts during the readout drives (it is the trained comprehension region). The
    re-binding claim is about the route, which is plasticity-gated to 0.0 and must not change."""
    bridge = probe["bridge"]
    marker = to_host(probe["marker"])
    csr = bridge.cp_connections
    vals = []
    for r in ROLES:
        tgt = to_host(probe["targets"][r])
        for s in marker:
            for t in tgt:
                vals.append(float(to_host(csr[int(s), int(t)])))
    return np.asarray(vals, dtype=np.float64)


def run_gated_route_probe(seed=42, proj_dim=64, verbose=False):
    """Run the full de-risk: build the unified probe bridge, then the three conditions
    (agent-drive / patient-drive / no-drive control), measuring selective routing and confirming the ROUTE
    re-binds across them with ZERO weight change. Returns a results dict consumed by
    `test_step2_parser_role_gated_route_is_selective`.
    """
    probe = build_probe_bridge(seed=seed, proj_dim=proj_dim)

    rw_before = _route_weights(probe)
    agent_drive = _run_condition(probe, position=0, drive_marker=True)     # → agent ensemble → agent_target
    patient_drive = _run_condition(probe, position=2, drive_marker=True)   # → patient ensemble → patient_target
    no_drive = _run_condition(probe, position=None, drive_marker=True)     # control: no ensemble → no target
    rw_after = _route_weights(probe)

    res = {
        "agent_drive": agent_drive,
        "patient_drive": patient_drive,
        "no_drive": no_drive,
        # ROUTE weight change across the conditions (NOT the whole bridge — the plastic parser drifts by design).
        "weight_delta": float(np.abs(rw_after - rw_before).max()),
        "route_weights_at_design": bool(np.all(rw_before == ROUTE_WEIGHT)),
        "seed": int(seed), "proj_dim": int(proj_dim),
    }
    if verbose:
        import json
        print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(run_gated_route_probe(seed=42, proj_dim=64), indent=2))
