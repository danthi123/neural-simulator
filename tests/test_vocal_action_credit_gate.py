import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._vocal_action_credit_gate import (
    ACTION_COLLATERAL_GATE,
    CHANNELS,
    CREDIT_CUE,
    CREDIT_PLASTICITY_GATE,
    CALIBRATION_SEEDS,
    DEVELOPMENT_SEEDS,
    DOPAMINE_PATH_GATE,
    SNC,
    HELD_OUT_SEEDS,
    VALUE_PLASTICITY_GATE,
    VALUE_TO_SNC_GATE,
    _actor,
    _value,
    _yoked_schedule,
    build_credit_bridge,
    credit_config,
    validate_calibration_seeds,
)
from sim.backend import to_host


def test_credit_gate_v1_has_only_two_declared_plastic_routes():
    bridge, routes = build_credit_bridge(seed=7)
    plastic = np.asarray(to_host(bridge.cp_synapse_plastic_mask), dtype=bool)
    expected = np.zeros(plastic.size, dtype=bool)
    for channel in CHANNELS:
        expected[routes.actor[channel]] = True

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
        assert len(routes.actor[channel]) > 0
    dopamine = bridge.neuromodulator_manager._config_by_name("dopamine")
    assert dopamine.production_rules[0].source_regions == [SNC]
    assert dopamine.production_rules[0].rule_type == "from_region_firing_signed"


def test_credit_route_scope_matches_only_cue_actor_synapses():
    bridge, routes = build_credit_bridge(seed=13)
    scoped = set(np.asarray(to_host(
        bridge.cp_reward_eligibility_synapse_indices
    ), dtype=np.int64).tolist())
    expected = set(routes.all_indices().tolist())

    assert scoped == expected


def test_credit_gate_v2_adds_action_value_critic_routes():
    config = credit_config("v2")
    bridge, routes = build_credit_bridge(seed=17, config=config)
    pathways = bridge.region_manager.pathways()
    plastic = np.asarray(to_host(bridge.cp_synapse_plastic_mask), dtype=bool)
    expected = np.zeros(plastic.size, dtype=bool)
    expected[routes.all_indices()] = True

    np.testing.assert_array_equal(plastic, expected)
    assert set(bridge.list_plasticity_gates()) == {
        CREDIT_PLASTICITY_GATE,
        VALUE_PLASTICITY_GATE,
    }
    assert bridge.get_plasticity_gate_value(
        CREDIT_PLASTICITY_GATE
    ) == config.actor_plasticity_gain
    assert bridge.get_plasticity_gate_value(VALUE_PLASTICITY_GATE) == 1.0
    assert bridge.core_config.enable_gabab is True
    assert config.cue_pA == 6000.0
    assert config.gabab_propagation_strength == 0.00004
    assert set(routes.value) == set(CHANNELS)
    for channel in CHANNELS:
        value_paths = [
            pathway for pathway in pathways
            if pathway.from_region == _value(channel)
            and pathway.to_region == SNC
        ]
        assert len(value_paths) == 1
        assert value_paths[0].receptor == "gaba_b"
        assert value_paths[0].transmission_gate == VALUE_TO_SNC_GATE


def test_yoked_schedule_preserves_rewards_but_breaks_trial_alignment():
    contingent = [True, False, False, True, True, False]
    yoked = _yoked_schedule(contingent)

    assert yoked != contingent
    assert sum(yoked) == sum(contingent)


def test_gate_b_seed_policy_locks_development_and_held_out_sets():
    assert validate_calibration_seeds(CALIBRATION_SEEDS) == CALIBRATION_SEEDS
    for seed in DEVELOPMENT_SEEDS + HELD_OUT_SEEDS:
        try:
            validate_calibration_seeds([seed])
        except ValueError as error:
            assert "calibration seeds" in str(error)
        else:
            raise AssertionError(f"reserved seed {seed} was accepted")
