"""Sequencing bound primitives by gating: the BG steps through an ordered plan of (verb,motor) bindings,
producing the ordered motor sequence -- including TEMPORAL VARIABLE BINDING (the same verb bound to different
motors at different positions), which gated re-binding allows and grown static weights cannot.
"""
from research.runners.gated_sequence_demo import produce_sequence
from research.runners.gated_compose_bg_demo import build_bg_gated_bridge, couple_all_route_gates


def test_gated_sequence_produces_ordered_output():
    for seed in (42, 43):
        sb = build_bg_gated_bridge(seed=seed)
        couple_all_route_gates(sb)
        plan = [("GO", "N"), ("STOP", "W"), ("COME", "S")]
        assert produce_sequence(sb, plan) == [m for _, m in plan]


def test_gated_sequence_temporal_variable_binding():
    # GO is bound to N at position 0 and S at position 2 (with LOOK->E between) -- the same role, different
    # fillers, in one sequence. A grown-weight model cannot represent this (GO's weight is a constant).
    sb = build_bg_gated_bridge(seed=42)
    couple_all_route_gates(sb)
    assert produce_sequence(sb, [("GO", "N"), ("LOOK", "E"), ("GO", "S")]) == ["N", "E", "S"]
