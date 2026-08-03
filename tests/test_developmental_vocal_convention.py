import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from research.runners._developmental_vocal_convention_derisk import (
    InteractiveListenerWorld,
    RawVocalAction,
    VocalConvention,
)
from research.runners._homeostatic_spiking_reward_plasticity_derisk import build_bridge
from sim.backend import get_backend, to_host
from research.runners.nav_conv_merged_bridge import (
    GEN_PERCEPTION,
    VOCAL_INTENT_PREFIX,
    VOCAL_LEARNING_GATE,
    VOCAL_REFERENT_PREFIX,
    VOCAL_SILENCE,
    VOCAL_SOCIAL_CUE,
    VOCAL_SPEAK,
    _developmental_vocal_regions_pathways,
)


def test_vocal_slice_has_factorized_learnable_routes_and_speak_gate():
    regions, pathways = _developmental_vocal_regions_pathways()
    names = {region.name for region in regions}
    assert {
        VOCAL_SOCIAL_CUE,
        VOCAL_SPEAK,
        VOCAL_SILENCE,
        f"{VOCAL_INTENT_PREFIX}0",
        f"{VOCAL_INTENT_PREFIX}1",
        f"{VOCAL_REFERENT_PREFIX}0",
        f"{VOCAL_REFERENT_PREFIX}1",
    } <= names
    routes = {(route.from_region, route.to_region): route for route in pathways}
    for i in range(2):
        intent = f"{VOCAL_INTENT_PREFIX}{i}"
        referent = f"{VOCAL_REFERENT_PREFIX}{i}"
        assert routes[("drive_agrp", intent)].plasticity_gate == VOCAL_LEARNING_GATE
        assert routes[(VOCAL_SOCIAL_CUE, intent)].plasticity_gate == VOCAL_LEARNING_GATE
        assert routes[(GEN_PERCEPTION, referent)].plasticity_gate == VOCAL_LEARNING_GATE
    assert routes[("drive_agrp", VOCAL_SPEAK)].plasticity_gate == VOCAL_LEARNING_GATE
    assert routes[(VOCAL_SOCIAL_CUE, VOCAL_SILENCE)].plasticity_gate == VOCAL_LEARNING_GATE


def test_listener_meaning_follows_external_permutable_convention():
    identity = VocalConvention.identity()
    swapped = VocalConvention.swapped()
    action = RawVocalAction(intent_channel=0, referent_channel=0)
    identity_result = InteractiveListenerWorld(identity, "apple", "need").apply(action)
    swapped_result = InteractiveListenerWorld(swapped, "apple", "need").apply(action)
    assert identity_result["success"]
    assert identity_result["consequence"] == "resource_delivered"
    assert not swapped_result["success"]
    assert swapped_result["decoded"] == ["report", "river"]


@pytest.mark.parametrize("branchless", [False, True])
def test_deferred_stdp_tags_eligibility_but_waits_for_reward(branchless):
    bridge, cfg = build_bridge(seed=42, n=20)
    xp, _ = get_backend()
    cue = np.asarray(bridge.region_manager.indices("cue"), dtype=np.int64)
    motor = np.asarray(bridge.region_manager.indices("motor"), dtype=np.int64)
    cue_x = xp.asarray(cue)
    motor_x = xp.asarray(motor)
    cfg.reward_defer_stdp_weight_update = True
    cfg.enable_branchless_plasticity = branchless
    cfg.reward_learning_rate = 1.0
    cfg.current_reward_signal = 0.0
    cfg.enable_ou_process = False
    before = np.asarray(to_host(bridge.cp_connections.data)).copy()

    def step():
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += cfg.dt_ms

    for _ in range(16):
        for _ in range(3):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[cue_x] = 400.0
            step()
        for _ in range(3):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[cue_x] = 400.0
            bridge.cp_external_input_current[motor_x] = 350.0
            step()

    tagged = np.asarray(to_host(
        bridge.cp_eligibility_trace[: bridge.cp_connections.nnz]
    ))
    after_tag = np.asarray(to_host(bridge.cp_connections.data))
    assert np.any(np.abs(tagged) > 0.0)
    np.testing.assert_array_equal(after_tag, before)

    cfg.current_reward_signal = 1.0
    bridge.cp_external_input_current[:] = 0.0
    step()
    after_reward = np.asarray(to_host(bridge.cp_connections.data))
    assert np.any(np.abs(after_reward - before) > 0.0)


def test_coactivity_eligibility_requires_reward_for_weight_change():
    bridge, cfg = build_bridge(seed=43, n=20)
    bridge.strict_step_errors = True
    xp, _ = get_backend()
    cue = np.asarray(bridge.region_manager.indices("cue"), dtype=np.int64)
    motor = np.asarray(bridge.region_manager.indices("motor"), dtype=np.int64)
    coo = bridge.cp_connections.tocoo(copy=False)
    rows = np.asarray(to_host(coo.row))
    cols = np.asarray(to_host(coo.col))
    route = np.flatnonzero(np.isin(rows, cue) & np.isin(cols, motor))
    route_x = xp.asarray(route, dtype=xp.int64)
    cue_x = xp.asarray(cue)
    motor_x = xp.asarray(motor)

    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_reward_modulation = True
    cfg.reward_eligibility_from_coactivity = True
    cfg.reward_coactivity_trace_tau_ms = 5.0
    cfg.reward_coactivity_threshold = 0.001
    cfg.reward_coactivity_scale = 1.0
    cfg.reward_learning_rate = 1.0
    cfg.current_reward_signal = 0.0
    bridge.cp_reward_coactivity_trace = xp.zeros(
        int(cfg.num_neurons), dtype=xp.float32
    )
    bridge.cp_reward_eligibility_synapse_indices = route_x
    bridge.cp_connections.data[route_x] = xp.float32(0.0)
    before = np.asarray(to_host(bridge.cp_connections.data)).copy()

    def step():
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += cfg.dt_ms

    for _ in range(20):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[cue_x] = 400.0
        bridge.cp_external_input_current[motor_x] = 350.0
        step()

    tagged = np.asarray(to_host(bridge.cp_eligibility_trace[route_x]))
    assert np.any(tagged > 0.0)
    np.testing.assert_array_equal(
        np.asarray(to_host(bridge.cp_connections.data)), before
    )

    cfg.current_reward_signal = 1.0
    bridge.cp_external_input_current[:] = 0.0
    step()
    after_reward = np.asarray(to_host(bridge.cp_connections.data))
    assert np.any(after_reward[route] > before[route])
    outside = np.ones(after_reward.size, dtype=bool)
    outside[route] = False
    np.testing.assert_array_equal(after_reward[outside], before[outside])

    # A presynaptic trace without a postsynaptic output event is not enough to
    # create credit. Reset transient state, remove the learned weight, and
    # drive only the cue population.
    bridge.cp_connections.data[route_x] = xp.float32(0.0)
    bridge.cp_eligibility_trace[:] = 0.0
    bridge.cp_reward_coactivity_trace[:] = 0.0
    cfg.current_reward_signal = 0.0
    for _ in range(100):
        bridge.cp_external_input_current[:] = 0.0
        step()
    for _ in range(20):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[cue_x] = 400.0
        step()
    assert not np.any(
        np.asarray(to_host(bridge.cp_eligibility_trace[route_x])) > 0.0
    )
