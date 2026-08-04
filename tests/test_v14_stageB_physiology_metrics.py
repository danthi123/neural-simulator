import numpy as np
import pytest

from tools.v14_stageB_physiology_metrics import (
    PhysiologyMetricError,
    action_potential_shape,
    ahp_depth,
    detect_depolarization_block,
    inhibitory_release_metrics,
    input_resistance,
    spike_train_metrics,
)


TRACE_CONTRACT = {
    "time_unit": "s",
    "voltage_unit": "mV",
    "sample_interval_s": 0.001,
    "recording_start_s": 0.0,
    "burn_in_start_s": 0.0,
    "burn_in_end_s": 0.002,
}


def test_spike_train_rate_cv_and_cv2_use_only_post_burn_in_window():
    result = spike_train_metrics(
        [0.001, 0.010, 0.020, 0.040, 0.070],
        time_unit="s",
        sample_interval_s=0.001,
        recording_start_s=0.0,
        recording_end_s=0.100,
        burn_in_start_s=0.0,
        burn_in_end_s=0.010,
        window_start_s=0.010,
        window_end_s=0.090,
    )
    intervals = np.array([0.010, 0.020, 0.030])
    assert result["spike_count"] == 4
    assert result["firing_rate_hz"] == pytest.approx(50.0)
    assert result["isi_cv"] == pytest.approx(np.std(intervals) / np.mean(intervals))
    expected_cv2 = np.mean(2 * np.abs(np.diff(intervals)) / (intervals[1:] + intervals[:-1]))
    assert result["isi_cv2"] == pytest.approx(expected_cv2)


def test_spike_train_reports_insufficient_isi_metrics_without_inventing_values():
    result = spike_train_metrics(
        [0.02, 0.04], time_unit="s", sample_interval_s=0.001,
        recording_start_s=0.0, recording_end_s=0.1,
        burn_in_start_s=0.0, burn_in_end_s=0.01,
        window_start_s=0.01, window_end_s=0.09,
    )
    assert result["isi_cv"] is None
    assert result["isi_cv2"] is None


def test_action_potential_threshold_rule_and_interpolated_half_width():
    time = np.arange(0.0, 0.011, 0.001)
    voltage = np.array([-65, -65, -65, -60, -40, 20, -20, -50, -70, -68, -66], dtype=float)
    result = action_potential_shape(
        time, voltage, **TRACE_CONTRACT, search_start_s=0.002,
        peak_time_s=0.005, search_end_s=0.010, dvdt_threshold_mV_per_ms=10.0,
    )
    assert result["threshold_time_s"] == pytest.approx(0.003)
    assert result["threshold_voltage_mV"] == -60.0
    assert result["half_amplitude_voltage_mV"] == -20.0
    assert result["half_width_ms"] == pytest.approx(5.0 / 3.0)
    assert "forward-difference" in result["threshold_rule"]
    assert "linear-interpolated" in result["width_rule"]


def test_action_potential_rejects_peak_that_is_not_a_sample_or_maximum():
    time = np.arange(0.0, 0.011, 0.001)
    voltage = np.array([-65, -65, -65, -60, -40, 20, -20, -50, -70, -68, -66], dtype=float)
    with pytest.raises(PhysiologyMetricError, match="recorded sample"):
        action_potential_shape(
            time, voltage, **TRACE_CONTRACT, search_start_s=0.002,
            peak_time_s=0.0055, search_end_s=0.010, dvdt_threshold_mV_per_ms=10.0,
        )


def test_ahp_depth_uses_explicit_reference_and_discrete_minimum():
    time = np.arange(0.0, 0.011, 0.001)
    voltage = np.array([-65, -65, -65, 20, -20, -68, -75, -72, -69, -67, -66], dtype=float)
    result = ahp_depth(
        time, voltage, **TRACE_CONTRACT, reference_voltage_mV=-60.0,
        window_start_s=0.004, window_end_s=0.009,
    )
    assert result == {
        "reference_voltage_mV": -60.0,
        "minimum_voltage_mV": -75.0,
        "minimum_time_s": pytest.approx(0.006),
        "ahp_depth_mV": 15.0,
    }


def test_depolarization_block_requires_spike_free_plateau_of_requested_duration():
    time = np.arange(0.0, 0.021, 0.001)
    voltage = np.full(time.shape, -65.0)
    voltage[8:16] = -30.0
    common = dict(
        **TRACE_CONTRACT, recording_end_s=0.021, window_start_s=0.005,
        window_end_s=0.020, plateau_voltage_mV=-40.0,
        minimum_plateau_duration_s=0.007,
    )
    detected = detect_depolarization_block(time, voltage, [], **common)
    assert detected["detected"] is True
    assert detected["longest_spike_free_plateau_s"] == pytest.approx(0.008)

    interrupted = detect_depolarization_block(time, voltage, [0.010], **common)
    assert interrupted["detected"] is False
    assert interrupted["longest_spike_free_plateau_s"] == 0.0


def test_input_resistance_uses_signed_voltage_and_current_deflection():
    time = np.arange(0.0, 0.011, 0.001)
    voltage = np.array([-65, -65, -65, -65, -65, -70, -70, -70, -70, -70, -70], dtype=float)
    result = input_resistance(
        time, voltage, **TRACE_CONTRACT, current_unit="nA",
        baseline_start_s=0.002, baseline_end_s=0.005,
        steady_state_start_s=0.006, steady_state_end_s=0.010,
        current_step_nA=-0.1,
    )
    assert result["voltage_deflection_mV"] == -5.0
    assert result["input_resistance_MOhm"] == pytest.approx(50.0)


def test_release_metrics_report_latency_and_rate_overshoot_without_pass_band():
    result = inhibitory_release_metrics(
        [0.02, 0.04, 0.06, 0.105, 0.115, 0.125],
        time_unit="s", sample_interval_s=0.001,
        recording_start_s=0.0, recording_end_s=0.2,
        burn_in_start_s=0.0, burn_in_end_s=0.01,
        baseline_start_s=0.01, baseline_end_s=0.09,
        release_time_s=0.1, overshoot_end_s=0.14, recovery_end_s=0.18,
    )
    assert result["baseline_rate_hz"] == pytest.approx(37.5)
    assert result["recovery_latency_s"] == pytest.approx(0.005)
    assert result["post_release_rate_hz"] == pytest.approx(75.0)
    assert result["rate_overshoot_hz"] == pytest.approx(37.5)
    assert result["rate_overshoot_ratio"] == pytest.approx(2.0)


def test_release_metrics_handle_no_recovery_and_zero_baseline_rate():
    result = inhibitory_release_metrics(
        [], time_unit="s", sample_interval_s=0.001,
        recording_start_s=0.0, recording_end_s=0.2,
        burn_in_start_s=0.0, burn_in_end_s=0.01,
        baseline_start_s=0.01, baseline_end_s=0.09,
        release_time_s=0.1, overshoot_end_s=0.14, recovery_end_s=0.18,
    )
    assert result["recovery_latency_s"] is None
    assert result["rate_overshoot_ratio"] is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"time_unit": "ms"}, "time_unit"),
        ({"voltage_unit": "V"}, "voltage_unit"),
        ({"sample_interval_s": 0.002}, "sample_interval_s"),
        ({"recording_start_s": -0.001}, "cropped"),
        ({"burn_in_start_s": -0.001}, "burn_in_start_s"),
    ],
)
def test_trace_contract_rejects_wrong_units_sampling_and_hidden_burn_in(change, message):
    time = np.arange(0.0, 0.011, 0.001)
    voltage = np.full(time.shape, -65.0)
    kwargs = {**TRACE_CONTRACT, **change}
    with pytest.raises(PhysiologyMetricError, match=message):
        ahp_depth(
            time, voltage, **kwargs, reference_voltage_mV=-60.0,
            window_start_s=0.004, window_end_s=0.009,
        )


def test_windows_before_burn_in_or_outside_recording_are_rejected():
    with pytest.raises(PhysiologyMetricError, match="before burn-in"):
        spike_train_metrics(
            [0.02], time_unit="s", sample_interval_s=0.001,
            recording_start_s=0.0, recording_end_s=0.1,
            burn_in_start_s=0.0, burn_in_end_s=0.03,
            window_start_s=0.02, window_end_s=0.09,
        )


def test_malformed_spikes_and_zero_current_are_rejected():
    with pytest.raises(PhysiologyMetricError, match="strictly increasing"):
        spike_train_metrics(
            [0.04, 0.02], time_unit="s", sample_interval_s=0.001,
            recording_start_s=0.0, recording_end_s=0.1,
            burn_in_start_s=0.0, burn_in_end_s=0.01,
            window_start_s=0.01, window_end_s=0.09,
        )

    time = np.arange(0.0, 0.011, 0.001)
    voltage = np.full(time.shape, -65.0)
    with pytest.raises(PhysiologyMetricError, match="nonzero"):
        input_resistance(
            time, voltage, **TRACE_CONTRACT, current_unit="nA",
            baseline_start_s=0.002, baseline_end_s=0.005,
            steady_state_start_s=0.006, steady_state_end_s=0.010,
            current_step_nA=0.0,
        )
