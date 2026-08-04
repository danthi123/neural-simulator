import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _vocal_action_credit_gate_v10_policy as v10
from sim.backend import to_host


def test_v10_formal_execution_and_nonreserved_seeds_are_sealed():
    with pytest.raises(ValueError, match="reserved seed 0"):
        v10.build_v10_bridge(seed=1)
    with pytest.raises(ValueError, match="formal phases are sealed"):
        v10.validate_phase("calibration")
    with pytest.raises(ValueError, match="formal execution is sealed"):
        v10.run_formal_seed(42)


def test_v10_construction_owns_only_selector_policy_synapses():
    result = v10.run_construction_smoke()
    structure = result["structure"]

    assert structure["all_checks_pass"] is True
    assert structure["policy_synapses"] == 4 * 60 * 36
    assert structure["plastic_synapses"] == structure["policy_synapses"]
    assert structure["route_signs"] == {"d1": [1.0], "d2": [-1.0]}
    assert all("actor" not in name for name in structure["region_names"])


def test_v10_condition_builds_start_identically_without_host_trace_edits():
    intact, intact_handles = v10.build_v10_bridge(coactivity=True)
    lesion, lesion_handles = v10.build_v10_bridge(coactivity=False)

    np.testing.assert_array_equal(v10._weights(intact), v10._weights(lesion))
    np.testing.assert_array_equal(
        intact_handles["all_policy"], lesion_handles["all_policy"]
    )
    assert intact.core_config.reward_coactivity_trace_input_gain == 0.0
    assert lesion.core_config.reward_coactivity_trace_input_gain == 0.0
    assert intact.cp_reward_coactivity_trace is not None
    assert lesion.cp_reward_coactivity_trace is None
    assert intact.strict_step_errors is True
    assert lesion.strict_step_errors is True
    assert np.max(np.abs(np.asarray(
        to_host(intact.cp_reward_coactivity_trace)
    ))) == 0.0

    for structure in (
        v10.structural_audit(intact, intact_handles),
        v10.structural_audit(lesion, lesion_handles),
    ):
        assert structure["all_checks_pass"] is True
        assert structure["route_sizes"] == {
            "d1_0": 2160,
            "d1_1": 2160,
            "d2_0": 2160,
            "d2_1": 2160,
        }


def test_v10_clip_path_control_enters_branch_without_moving_weights():
    result = v10.run_clip_path_control()

    assert result["diagnostic_reward_signal"] == 1.0
    assert result["reward_learning_rate"] == 0.0
    assert result["coactivity_enabled"] is False
    assert result["initial_policy_weights_inside_hebbian_bounds"] is True
    assert result["weights_byte_identical"] is True
    assert result["initial_weight_hash"] == result["final_weight_hash"]


def test_v10_net_eligibility_removes_only_expected_exponential_carryover():
    row = {
        "decision_step": 9,
        "pretrial": {
            "route_means": {
                "d1": {"0": 2.0, "1": 4.0},
                "d2": {"0": 1.0, "1": 3.0},
            }
        },
        "snapshots": {
            "decision": {
                "route_means": {
                    "d1": {"0": 5.0, "1": 6.0},
                    "d2": {"0": 4.0, "1": 5.0},
                }
            },
            "action_end": None,
            "pre_outcome": None,
        },
    }

    updated = v10._with_net_values(row)
    decay = np.exp(-10.0 / v10.ELIGIBILITY_TAU_MS)
    assert updated["snapshots"]["decision"]["net_route_means"]["d1"]["0"] == pytest.approx(
        5.0 - 2.0 * decay
    )
    assert updated["snapshots"]["decision"]["net_route_means"]["d2"]["1"] == pytest.approx(
        5.0 - 3.0 * decay
    )
