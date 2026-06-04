"""BG-driven thalamocortical gate selection: the basal ganglia bind a verb->motor mapping by disinhibiting
thalamic pools, whose activity opens the cortical route gates. Closes the loop on binding-by-gating.
"""
from research.runners.gated_compose_bg_demo import (
    build_bg_gated_bridge, bind_via_bg, couple_all_route_gates, decode_with_bg,
)
from research.runners.gated_compose_demo import VERBS, TRUE_MAP, decode


def test_bg_selects_binding_via_thalamic_disinhibition():
    for seed in (42, 43, 44):
        sb = build_bg_gated_bridge(seed=seed)
        opened = bind_via_bg(sb, TRUE_MAP)                              # BG disinhibits thal -> thal opens gates
        assert opened == TRUE_MAP, f"seed {seed}: gates opened {opened} != BG selection {TRUE_MAP}"
        ok = sum(decode(sb, v)[0] == TRUE_MAP[v] for v in VERBS)
        assert ok == 4, f"seed {seed}: BG-gated routing {ok}/4"


def test_bg_reselection_rebinds():
    sb = build_bg_gated_bridge(seed=42)
    bind_via_bg(sb, TRUE_MAP)
    assert all(decode(sb, v)[0] == TRUE_MAP[v] for v in VERBS)
    permuted = {"GO": "S", "COME": "W", "STOP": "E", "LOOK": "N"}
    opened = bind_via_bg(sb, permuted)                                  # BG re-selects -> thal re-opens new gates
    assert opened == permuted
    assert all(decode(sb, v)[0] == permuted[v] for v in VERBS)         # re-bound by BG re-selection


def test_bridge_internal_gate_coupling():
    # the loop fully IN-SUBSTRATE: gates are driven by thalamic firing inside _run_one_simulation_step
    # (couple_gate_to_pool), not a runner read. Disinhibiting thal pools -> bridge opens the cortical gates.
    for seed in (42, 43):
        sb = build_bg_gated_bridge(seed=seed)
        couple_all_route_gates(sb)                                     # gate g_X_Y <- thal_X_Y firing, in-step
        ok = sum(decode_with_bg(sb, v, TRUE_MAP) == TRUE_MAP[v] for v in VERBS)
        assert ok == 4, f"seed {seed}: bridge-internal coupled routing {ok}/4"
