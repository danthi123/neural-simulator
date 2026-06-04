"""Compositional binding by gating on the verb->motor vocabulary (the v16 compose problem, solved by the
transmission gate). Where STDP-grown verb->motor weights "went silent" (5/20 seed-fragile, could not re-bind),
gated routing binds the mapping deterministically AND re-binds to any other mapping with zero weight change.
"""
import numpy as np

from research.runners.gated_compose_demo import (
    build_gated_compose_bridge, bind_mapping, decode, TRUE_MAP, VERBS,
)


def test_gated_compose_binds_true_mapping_multi_seed():
    for seed in (42, 43, 44):
        sb = build_gated_compose_bridge(seed=seed)
        bind_mapping(sb, TRUE_MAP)
        ok = sum(decode(sb, v)[0] == TRUE_MAP[v] for v in VERBS)
        assert ok == 4, f"seed {seed}: gated binding {ok}/4 (expected 4/4 deterministic)"


def test_gated_compose_rebinds_to_permuted_mapping_zero_weight_change():
    from sim.backend import to_host
    sb = build_gated_compose_bridge(seed=42)
    bind_mapping(sb, TRUE_MAP)
    assert all(decode(sb, v)[0] == TRUE_MAP[v] for v in VERBS)        # bound to the true mapping
    w_before = float(np.abs(to_host(sb.cp_connections.data)).sum())

    permuted = {"GO": "S", "COME": "W", "STOP": "E", "LOOK": "N"}     # a totally different mapping
    bind_mapping(sb, permuted)
    assert all(decode(sb, v)[0] == permuted[v] for v in VERBS)        # re-bound on command -- binding follows the gates
    w_after = float(np.abs(to_host(sb.cp_connections.data)).sum())
    assert abs(w_after - w_before) < 1e-3                             # the synaptic WEIGHTS never changed
