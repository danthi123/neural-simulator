import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._vocal_action_credit_gate import (
    ACTION_COLLATERAL_GATE,
    CHANNELS,
    CREDIT_CUE,
    CREDIT_PLASTICITY_GATE,
    DOPAMINE_PATH_GATE,
    SNC,
    _actor,
    _yoked_schedule,
    build_credit_bridge,
)
from sim.backend import to_host


def test_credit_gate_has_only_two_declared_plastic_routes():
    bridge, routes = build_credit_bridge(seed=7)
    plastic = np.asarray(to_host(bridge.cp_synapse_plastic_mask), dtype=bool)
    expected = np.zeros(plastic.size, dtype=bool)
    for channel in CHANNELS:
        expected[routes[channel]] = True

    np.testing.assert_array_equal(plastic, expected)
    assert bridge.list_plasticity_gates() == [CREDIT_PLASTICITY_GATE]
    assert set(bridge._transmission_gate_to_synapses) >= {
        ACTION_COLLATERAL_GATE,
        DOPAMINE_PATH_GATE,
    }


def test_credit_gate_uses_shared_cue_and_spiking_dopamine_source():
    bridge, routes = build_credit_bridge(seed=11)
    pathways = bridge.region_manager.pathways()

    for channel in CHANNELS:
        cue_routes = [
            pathway for pathway in pathways
            if pathway.from_region == CREDIT_CUE
            and pathway.to_region == _actor(channel)
        ]
        assert len(cue_routes) == 1
        assert cue_routes[0].plasticity_gate == CREDIT_PLASTICITY_GATE
        assert len(routes[channel]) > 0
    dopamine = bridge.neuromodulator_manager._config_by_name("dopamine")
    assert dopamine.production_rules[0].source_regions == [SNC]
    assert dopamine.production_rules[0].rule_type == "from_region_firing_signed"


def test_credit_route_scope_matches_only_cue_actor_synapses():
    bridge, routes = build_credit_bridge(seed=13)
    scoped = set(np.asarray(to_host(
        bridge.cp_reward_eligibility_synapse_indices
    ), dtype=np.int64).tolist())
    expected = set(np.concatenate([routes[0], routes[1]]).tolist())

    assert scoped == expected


def test_yoked_schedule_preserves_rewards_but_breaks_trial_alignment():
    contingent = [True, False, False, True, True, False]
    yoked = _yoked_schedule(contingent)

    assert yoked != contingent
    assert sum(yoked) == sum(contingent)
