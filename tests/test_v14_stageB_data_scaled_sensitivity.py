from __future__ import annotations

import numpy as np
import pytest

from tools import v14_stageB_data_scaled_sensitivity as sensitivity


def _linear_predictions(probes: np.ndarray, matrix: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return baseline[None, :] + probes @ matrix


def test_controller_recovers_one_uncertainty_unit_scales_without_numeric_bounds():
    coordinate_ids = ["sodium.alpha", "kv3.beta"]
    uncertainty = np.array([0.1, 0.2, 0.4])
    response_matrix = np.array([[2.0, 0.5, -1.0], [0.25, -3.0, 1.5]])
    baseline = np.array([0.4, 0.5, 0.6])
    controller = sensitivity.DataScaledProbeController(coordinate_ids, uncertainty)

    for _ in range(100):
        probes = controller.probe_matrix()
        controller.observe(_linear_predictions(probes, response_matrix, baseline))
        if controller.complete:
            break
    assert controller.complete
    receipt = controller.receipt()
    assert receipt["status"] == "complete"
    assert all(row["status"] == "data_scaled" for row in receipt["coordinates"])
    assert all(row["selected_step_is_biological_bound"] is False for row in receipt["coordinates"])

    expected_sensitivity = np.sqrt(
        np.mean((response_matrix / uncertainty[None, :]) ** 2, axis=1)
    )
    selected = np.array([row["selected_step"] for row in receipt["coordinates"]])
    np.testing.assert_allclose(selected, 1.0 / expected_sensitivity, rtol=2e-8, atol=0.0)

    final_predictions = _linear_predictions(
        controller.probe_matrix(), response_matrix, baseline
    )
    jacobian = sensitivity.standardized_jacobian(
        final_predictions, selected, uncertainty
    )
    np.testing.assert_allclose(
        jacobian, response_matrix / uncertainty[None, :], rtol=1e-9, atol=1e-10
    )
    diagnostic = sensitivity.singular_diagnostics(jacobian)
    assert len(diagnostic["singular_values"]) == 2
    assert np.asarray(diagnostic["parameter_space_directions"]).shape == (2, 2)
    assert np.asarray(diagnostic["target_space_directions"]).shape == (2, 3)
    assert diagnostic["scientific_verdict"] is None


def test_invalid_probe_pair_marks_direction_unresolved_without_stopping_other_coordinates():
    controller = sensitivity.DataScaledProbeController(["valid", "invalid"], [0.1])
    predictions = np.array([[0.0], [1.0], [-1.0], [np.nan], [np.nan]])
    controller.observe(predictions)
    receipt = controller.receipt()
    by_id = {row["coordinate_id"]: row for row in receipt["coordinates"]}
    assert by_id["invalid"]["status"] == "unresolved_invalid_before_target_response"
    assert by_id["invalid"]["selected_step"] is None
    assert by_id["valid"]["status"] in {"bisecting", "data_scaled"}


def test_probe_order_and_input_validation_are_fail_closed():
    controller = sensitivity.DataScaledProbeController(["a", "b"], [0.1, 0.2])
    probes = controller.probe_matrix()
    assert probes.shape == (5, 2)
    np.testing.assert_array_equal(probes[0], [0.0, 0.0])
    assert probes[1, 0] > 0.0 and probes[2, 0] < 0.0
    assert probes[3, 1] > 0.0 and probes[4, 1] < 0.0

    with pytest.raises(sensitivity.DataScaledSensitivityError, match="unique"):
        sensitivity.DataScaledProbeController(["same", "same"], [0.1])
    with pytest.raises(sensitivity.DataScaledSensitivityError, match="positive"):
        sensitivity.DataScaledProbeController(["a"], [0.0])
    with pytest.raises(sensitivity.DataScaledSensitivityError, match="shape"):
        controller.observe(np.zeros((4, 2)))
    with pytest.raises(sensitivity.DataScaledSensitivityError, match="invalid"):
        sensitivity.standardized_jacobian(np.zeros((3, 1)), [0.0], [0.1])
