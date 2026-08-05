"""Deterministic, acceptance-band-free physiology metrics for V14 Stage B.

All windows are half-open ``[start, end)`` intervals. Trace functions require the
uncropped recording, its declared start, an explicit burn-in interval, and the
sampling interval. This prevents a caller from presenting a pre-trimmed trace as
though no hidden burn-in occurred.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


class PhysiologyMetricError(ValueError):
    """Raised when a physiology metric input or protocol is malformed."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysiologyMetricError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PhysiologyMetricError(f"{name} must be a finite number")
    return result


def _require_unit(actual: str, expected: str, name: str) -> None:
    if actual != expected:
        raise PhysiologyMetricError(f"{name} must be {expected!r}")


def _as_vector(values: Sequence[float], name: str, *, minimum_size: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < minimum_size or not np.all(np.isfinite(array)):
        raise PhysiologyMetricError(
            f"{name} must be a one-dimensional finite array with at least {minimum_size} values"
        )
    return array


def _validate_window(start_s: float, end_s: float, name: str) -> tuple[float, float]:
    start = _finite_number(start_s, f"{name}_start_s")
    end = _finite_number(end_s, f"{name}_end_s")
    if end <= start:
        raise PhysiologyMetricError(f"{name} must have positive duration")
    return start, end


def _validate_trace(
    time_s: Sequence[float],
    voltage_mV: Sequence[float],
    *,
    time_unit: str,
    voltage_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    _require_unit(time_unit, "s", "time_unit")
    _require_unit(voltage_unit, "mV", "voltage_unit")
    dt = _finite_number(sample_interval_s, "sample_interval_s")
    if dt <= 0:
        raise PhysiologyMetricError("sample_interval_s must be positive")
    time = _as_vector(time_s, "time_s", minimum_size=2)
    voltage = _as_vector(voltage_mV, "voltage_mV", minimum_size=2)
    if time.size != voltage.size:
        raise PhysiologyMetricError("time_s and voltage_mV must have equal length")
    if np.any(np.diff(time) <= 0) or not np.allclose(
        np.diff(time), dt, rtol=1e-9, atol=max(1e-12, dt * 1e-9)
    ):
        raise PhysiologyMetricError("time_s must be strictly increasing at sample_interval_s")

    recording_start = _finite_number(recording_start_s, "recording_start_s")
    burn_start, burn_end = _validate_window(burn_in_start_s, burn_in_end_s, "burn_in")
    tolerance = max(1e-12, dt * 1e-9)
    if not math.isclose(time[0], recording_start, abs_tol=tolerance):
        raise PhysiologyMetricError("trace is cropped or does not begin at recording_start_s")
    if not math.isclose(burn_start, recording_start, abs_tol=tolerance):
        raise PhysiologyMetricError("burn_in_start_s must equal recording_start_s")
    if burn_end > time[-1] + tolerance:
        raise PhysiologyMetricError("burn-in extends beyond the trace")
    return time, voltage, dt, burn_end


def _trace_window(
    time: np.ndarray,
    values: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    burn_in_end_s: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    start, end = _validate_window(start_s, end_s, name)
    if start < burn_in_end_s:
        raise PhysiologyMetricError(f"{name} starts before burn-in ends")
    if start < time[0] or end > time[-1]:
        raise PhysiologyMetricError(f"{name} lies outside the recorded trace")
    mask = (time >= start) & (time < end)
    if not np.any(mask):
        raise PhysiologyMetricError(f"{name} contains no samples")
    return time[mask], values[mask]


def _spikes(
    spike_times_s: Sequence[float],
    *,
    time_unit: str,
    recording_start_s: float,
    recording_end_s: float,
) -> np.ndarray:
    _require_unit(time_unit, "s", "time_unit")
    spikes = np.asarray(spike_times_s, dtype=float)
    if spikes.ndim != 1 or not np.all(np.isfinite(spikes)):
        raise PhysiologyMetricError("spike_times_s must be a one-dimensional finite array")
    if spikes.size and np.any(np.diff(spikes) <= 0):
        raise PhysiologyMetricError("spike_times_s must be strictly increasing and unique")
    start, end = _validate_window(recording_start_s, recording_end_s, "recording")
    if spikes.size and (spikes[0] < start or spikes[-1] >= end):
        raise PhysiologyMetricError("spike_times_s lie outside the declared recording")
    return spikes


def spike_train_metrics(
    spike_times_s: Sequence[float],
    *,
    time_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    recording_end_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
    window_start_s: float,
    window_end_s: float,
) -> dict[str, float | int | None | str]:
    """Return post-burn-in rate, ISI CV, and local CV2.

    CV uses the population standard deviation of complete within-window ISIs.
    CV2 is the mean of ``2*abs(ISI[n+1]-ISI[n])/(ISI[n+1]+ISI[n])``.
    Insufficient spike counts produce ``None`` rather than a fabricated value.
    """
    dt = _finite_number(sample_interval_s, "sample_interval_s")
    if dt <= 0:
        raise PhysiologyMetricError("sample_interval_s must be positive")
    recording_start, recording_end = _validate_window(
        recording_start_s, recording_end_s, "recording"
    )
    burn_start, burn_end = _validate_window(burn_in_start_s, burn_in_end_s, "burn_in")
    if not math.isclose(burn_start, recording_start, abs_tol=max(1e-12, dt * 1e-9)):
        raise PhysiologyMetricError("burn_in_start_s must equal recording_start_s")
    if burn_end > recording_end:
        raise PhysiologyMetricError("burn-in extends beyond the recording")
    window_start, window_end = _validate_window(window_start_s, window_end_s, "window")
    if window_start < burn_end:
        raise PhysiologyMetricError("window starts before burn-in ends")
    if window_start < recording_start or window_end > recording_end:
        raise PhysiologyMetricError("window lies outside the declared recording")

    spikes = _spikes(
        spike_times_s,
        time_unit=time_unit,
        recording_start_s=recording_start,
        recording_end_s=recording_end,
    )
    selected = spikes[(spikes >= window_start) & (spikes < window_end)]
    intervals = np.diff(selected)
    cv = None
    cv2 = None
    if intervals.size >= 2:
        mean_isi = float(np.mean(intervals))
        cv = float(np.std(intervals, ddof=0) / mean_isi)
        adjacent_sum = intervals[1:] + intervals[:-1]
        cv2 = float(np.mean(2.0 * np.abs(np.diff(intervals)) / adjacent_sum))
    return {
        "window_convention": "half-open",
        "spike_count": int(selected.size),
        "firing_rate_hz": float(selected.size / (window_end - window_start)),
        "isi_cv": cv,
        "isi_cv2": cv2,
    }


def peak_conductance(
    time_s: Sequence[float],
    conductance_nS: Sequence[float],
    *,
    time_unit: str,
    conductance_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
    window_start_s: float,
    window_end_s: float,
) -> dict[str, float]:
    """Return the discrete peak from an uncropped conductance trace."""
    _require_unit(time_unit, "s", "time_unit")
    _require_unit(conductance_unit, "nS", "conductance_unit")
    dt = _finite_number(sample_interval_s, "sample_interval_s")
    if dt <= 0:
        raise PhysiologyMetricError("sample_interval_s must be positive")
    time = _as_vector(time_s, "time_s", minimum_size=2)
    conductance = _as_vector(conductance_nS, "conductance_nS", minimum_size=2)
    if time.size != conductance.size:
        raise PhysiologyMetricError("time_s and conductance_nS must have equal length")
    if np.any(np.diff(time) <= 0) or not np.allclose(
        np.diff(time), dt, rtol=1e-9, atol=max(1e-12, dt * 1e-9)
    ):
        raise PhysiologyMetricError("time_s must be strictly increasing at sample_interval_s")
    if np.any(conductance < 0):
        raise PhysiologyMetricError("conductance_nS must be nonnegative")

    recording_start = _finite_number(recording_start_s, "recording_start_s")
    burn_start, burn_end = _validate_window(burn_in_start_s, burn_in_end_s, "burn_in")
    tolerance = max(1e-12, dt * 1e-9)
    if not math.isclose(time[0], recording_start, abs_tol=tolerance):
        raise PhysiologyMetricError("trace is cropped or does not begin at recording_start_s")
    if not math.isclose(burn_start, recording_start, abs_tol=tolerance):
        raise PhysiologyMetricError("burn_in_start_s must equal recording_start_s")
    window_time, window_conductance = _trace_window(
        time,
        conductance,
        start_s=window_start_s,
        end_s=window_end_s,
        burn_in_end_s=burn_end,
        name="conductance window",
    )
    index = int(np.argmax(window_conductance))
    return {
        "peak_conductance_nS": float(window_conductance[index]),
        "peak_time_s": float(window_time[index]),
    }


def action_potential_shape(
    time_s: Sequence[float],
    voltage_mV: Sequence[float],
    *,
    time_unit: str,
    voltage_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
    search_start_s: float,
    peak_time_s: float,
    search_end_s: float,
    dvdt_threshold_mV_per_ms: float,
) -> dict[str, float | str]:
    """Measure one AP using a documented discrete threshold and interpolated width.

    Threshold is the first pre-peak sample whose forward difference reaches the
    caller-supplied dV/dt criterion. Width is measured at half amplitude between
    that threshold voltage and the discrete peak, with linear interpolation on
    the rising and falling crossings.
    """
    time, voltage, dt, burn_end = _validate_trace(
        time_s,
        voltage_mV,
        time_unit=time_unit,
        voltage_unit=voltage_unit,
        sample_interval_s=sample_interval_s,
        recording_start_s=recording_start_s,
        burn_in_start_s=burn_in_start_s,
        burn_in_end_s=burn_in_end_s,
    )
    search_start, search_end = _validate_window(search_start_s, search_end_s, "search")
    if search_start < burn_end or search_end > time[-1]:
        raise PhysiologyMetricError("search window is outside the post-burn-in trace")
    peak_time = _finite_number(peak_time_s, "peak_time_s")
    if not search_start < peak_time < search_end:
        raise PhysiologyMetricError("peak_time_s must be inside the search window")
    threshold = _finite_number(dvdt_threshold_mV_per_ms, "dvdt_threshold_mV_per_ms")
    if threshold <= 0:
        raise PhysiologyMetricError("dvdt_threshold_mV_per_ms must be positive")

    indices = np.flatnonzero((time >= search_start) & (time < search_end))
    peak_index = int(indices[np.argmin(np.abs(time[indices] - peak_time))])
    if not math.isclose(time[peak_index], peak_time, abs_tol=max(1e-12, dt * 1e-9)):
        raise PhysiologyMetricError("peak_time_s must identify a recorded sample")
    if peak_index <= indices[0] or peak_index >= indices[-1]:
        raise PhysiologyMetricError("peak requires pre-peak and post-peak samples")
    if voltage[peak_index] != np.max(voltage[indices]):
        raise PhysiologyMetricError("peak_time_s does not identify the search-window maximum")

    dvdt = np.diff(voltage) / (dt * 1000.0)
    candidates = indices[(indices < peak_index) & (dvdt[indices] >= threshold)]
    if not candidates.size:
        raise PhysiologyMetricError("no pre-peak dV/dt threshold crossing was found")
    threshold_index = int(candidates[0])
    threshold_voltage = float(voltage[threshold_index])
    peak_voltage = float(voltage[peak_index])
    if peak_voltage <= threshold_voltage:
        raise PhysiologyMetricError("AP peak must exceed threshold voltage")
    half_voltage = threshold_voltage + 0.5 * (peak_voltage - threshold_voltage)

    rising = _crossing_time(time, voltage, threshold_index, peak_index, half_voltage, rising=True)
    falling = _crossing_time(time, voltage, peak_index, indices[-1], half_voltage, rising=False)
    return {
        "threshold_rule": "first pre-peak forward-difference sample at criterion",
        "width_rule": "linear-interpolated half amplitude",
        "threshold_time_s": float(time[threshold_index]),
        "threshold_voltage_mV": threshold_voltage,
        "peak_time_s": float(time[peak_index]),
        "peak_voltage_mV": peak_voltage,
        "half_amplitude_voltage_mV": half_voltage,
        "half_width_ms": float((falling - rising) * 1000.0),
    }


def _crossing_time(
    time: np.ndarray,
    voltage: np.ndarray,
    start_index: int,
    end_index: int,
    level: float,
    *,
    rising: bool,
) -> float:
    for index in range(start_index, end_index):
        left = voltage[index] - level
        right = voltage[index + 1] - level
        crossed = left <= 0 <= right if rising else left >= 0 >= right
        if crossed:
            delta = voltage[index + 1] - voltage[index]
            if delta == 0:
                return float(time[index])
            fraction = (level - voltage[index]) / delta
            return float(time[index] + fraction * (time[index + 1] - time[index]))
    direction = "rising" if rising else "falling"
    raise PhysiologyMetricError(f"no {direction} half-amplitude crossing was found")


def ahp_depth(
    time_s: Sequence[float],
    voltage_mV: Sequence[float],
    *,
    time_unit: str,
    voltage_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
    reference_voltage_mV: float,
    window_start_s: float,
    window_end_s: float,
) -> dict[str, float]:
    """Return reference voltage minus the discrete minimum in an explicit AHP window."""
    time, voltage, _, burn_end = _validate_trace(
        time_s, voltage_mV, time_unit=time_unit, voltage_unit=voltage_unit,
        sample_interval_s=sample_interval_s, recording_start_s=recording_start_s,
        burn_in_start_s=burn_in_start_s, burn_in_end_s=burn_in_end_s,
    )
    window_time, window_voltage = _trace_window(
        time, voltage, start_s=window_start_s, end_s=window_end_s,
        burn_in_end_s=burn_end, name="AHP window",
    )
    reference = _finite_number(reference_voltage_mV, "reference_voltage_mV")
    index = int(np.argmin(window_voltage))
    minimum = float(window_voltage[index])
    return {
        "reference_voltage_mV": reference,
        "minimum_voltage_mV": minimum,
        "minimum_time_s": float(window_time[index]),
        "ahp_depth_mV": reference - minimum,
    }


def interspike_voltage_nadirs(
    time_s: Sequence[float],
    voltage_mV: Sequence[float],
    spike_times_s: Sequence[float],
    *,
    time_unit: str,
    voltage_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    recording_end_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
) -> dict[str, float | int | str | list[float]]:
    """Measure the voltage nadir in every complete interspike interval.

    This is a project analysis convention for a directional total-AHP assay. It
    is not a reconstruction of a source-reported medium-AHP amplitude.
    """

    time, voltage, dt, burn_end = _validate_trace(
        time_s,
        voltage_mV,
        time_unit=time_unit,
        voltage_unit=voltage_unit,
        sample_interval_s=sample_interval_s,
        recording_start_s=recording_start_s,
        burn_in_start_s=burn_in_start_s,
        burn_in_end_s=burn_in_end_s,
    )
    recording_end = _finite_number(recording_end_s, "recording_end_s")
    if not math.isclose(
        recording_end, time[-1] + dt, abs_tol=max(1e-12, dt * 1e-9)
    ):
        raise PhysiologyMetricError(
            "recording_end_s must equal one sample interval past the final sample"
        )
    spikes = _spikes(
        spike_times_s,
        time_unit=time_unit,
        recording_start_s=recording_start_s,
        recording_end_s=recording_end,
    )
    if spikes.size < 2:
        raise PhysiologyMetricError("at least two spikes are required for an interspike nadir")
    if spikes[0] < burn_end:
        raise PhysiologyMetricError("the first complete interspike interval begins before burn-in ends")

    nadirs: list[float] = []
    for start, end in zip(spikes[:-1], spikes[1:], strict=True):
        mask = (time >= start) & (time < end)
        if not np.any(mask):
            raise PhysiologyMetricError("an interspike interval contains no voltage samples")
        nadirs.append(float(np.min(voltage[mask])))
    return {
        "event_selection": "all-complete-half-open-interspike-intervals",
        "complete_interspike_interval_count": len(nadirs),
        "median_interspike_voltage_nadir_mV": float(np.median(nadirs)),
        "interspike_voltage_nadirs_mV": nadirs,
    }


def detect_depolarization_block(
    time_s: Sequence[float],
    voltage_mV: Sequence[float],
    spike_times_s: Sequence[float],
    *,
    time_unit: str,
    voltage_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    recording_end_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
    window_start_s: float,
    window_end_s: float,
    plateau_voltage_mV: float,
    minimum_plateau_duration_s: float,
) -> dict[str, float | bool | None]:
    """Detect a caller-defined, spike-free depolarized plateau.

    This reports protocol evidence only. The plateau voltage and duration are
    explicit inputs and are not biological acceptance bands defined here.
    """
    time, voltage, dt, burn_end = _validate_trace(
        time_s, voltage_mV, time_unit=time_unit, voltage_unit=voltage_unit,
        sample_interval_s=sample_interval_s, recording_start_s=recording_start_s,
        burn_in_start_s=burn_in_start_s, burn_in_end_s=burn_in_end_s,
    )
    if not math.isclose(recording_end_s, time[-1] + dt, abs_tol=max(1e-12, dt * 1e-9)):
        raise PhysiologyMetricError("recording_end_s must equal one sample interval past the final sample")
    window_time, window_voltage = _trace_window(
        time, voltage, start_s=window_start_s, end_s=window_end_s,
        burn_in_end_s=burn_end, name="block window",
    )
    spikes = _spikes(
        spike_times_s, time_unit=time_unit, recording_start_s=recording_start_s,
        recording_end_s=recording_end_s,
    )
    plateau = _finite_number(plateau_voltage_mV, "plateau_voltage_mV")
    minimum_duration = _finite_number(minimum_plateau_duration_s, "minimum_plateau_duration_s")
    if minimum_duration <= 0:
        raise PhysiologyMetricError("minimum_plateau_duration_s must be positive")

    mask = window_voltage >= plateau
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True])
    longest = 0.0
    detected_start = None
    detected_end = None
    for first, last in zip(starts, stops, strict=True):
        start = float(window_time[first])
        end = float(window_time[last] + dt)
        duration = end - start
        has_spike = np.any((spikes >= start) & (spikes < end))
        if not has_spike and duration > longest:
            longest = duration
            detected_start = start
            detected_end = end
    return {
        "detected": bool(longest + 1e-15 >= minimum_duration),
        "longest_spike_free_plateau_s": float(longest),
        "plateau_start_s": detected_start,
        "plateau_end_s": detected_end,
    }


def input_resistance(
    time_s: Sequence[float],
    voltage_mV: Sequence[float],
    *,
    time_unit: str,
    voltage_unit: str,
    current_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
    baseline_start_s: float,
    baseline_end_s: float,
    steady_state_start_s: float,
    steady_state_end_s: float,
    current_step_nA: float,
) -> dict[str, float]:
    """Estimate input resistance as mean voltage deflection / current step."""
    _require_unit(current_unit, "nA", "current_unit")
    time, voltage, _, burn_end = _validate_trace(
        time_s, voltage_mV, time_unit=time_unit, voltage_unit=voltage_unit,
        sample_interval_s=sample_interval_s, recording_start_s=recording_start_s,
        burn_in_start_s=burn_in_start_s, burn_in_end_s=burn_in_end_s,
    )
    _, baseline = _trace_window(
        time, voltage, start_s=baseline_start_s, end_s=baseline_end_s,
        burn_in_end_s=burn_end, name="baseline window",
    )
    _, steady = _trace_window(
        time, voltage, start_s=steady_state_start_s, end_s=steady_state_end_s,
        burn_in_end_s=burn_end, name="steady-state window",
    )
    if baseline_end_s > steady_state_start_s:
        raise PhysiologyMetricError("baseline and steady-state windows must not overlap")
    current = _finite_number(current_step_nA, "current_step_nA")
    if current == 0:
        raise PhysiologyMetricError("current_step_nA must be nonzero")
    baseline_mean = float(np.mean(baseline))
    steady_mean = float(np.mean(steady))
    delta = steady_mean - baseline_mean
    return {
        "baseline_voltage_mV": baseline_mean,
        "steady_state_voltage_mV": steady_mean,
        "voltage_deflection_mV": delta,
        "current_step_nA": current,
        "input_resistance_MOhm": delta / current,
    }


def inhibitory_release_metrics(
    spike_times_s: Sequence[float],
    *,
    time_unit: str,
    sample_interval_s: float,
    recording_start_s: float,
    recording_end_s: float,
    burn_in_start_s: float,
    burn_in_end_s: float,
    baseline_start_s: float,
    baseline_end_s: float,
    release_time_s: float,
    recovery_end_s: float,
    overshoot_end_s: float,
) -> dict[str, float | int | None | str]:
    """Measure first-spike recovery latency and explicit-window rate overshoot."""
    dt = _finite_number(sample_interval_s, "sample_interval_s")
    if dt <= 0:
        raise PhysiologyMetricError("sample_interval_s must be positive")
    recording_start, recording_end = _validate_window(
        recording_start_s, recording_end_s, "recording"
    )
    burn_start, burn_end = _validate_window(burn_in_start_s, burn_in_end_s, "burn_in")
    if not math.isclose(burn_start, recording_start, abs_tol=max(1e-12, dt * 1e-9)):
        raise PhysiologyMetricError("burn_in_start_s must equal recording_start_s")
    baseline_start, baseline_end = _validate_window(
        baseline_start_s, baseline_end_s, "baseline"
    )
    release = _finite_number(release_time_s, "release_time_s")
    recovery_end = _finite_number(recovery_end_s, "recovery_end_s")
    overshoot_end = _finite_number(overshoot_end_s, "overshoot_end_s")
    if baseline_start < burn_end:
        raise PhysiologyMetricError("baseline starts before burn-in ends")
    if not baseline_end <= release < overshoot_end <= recovery_end <= recording_end:
        raise PhysiologyMetricError("release and recovery windows are not ordered within the recording")
    spikes = _spikes(
        spike_times_s, time_unit=time_unit, recording_start_s=recording_start,
        recording_end_s=recording_end,
    )
    baseline_count = int(np.sum((spikes >= baseline_start) & (spikes < baseline_end)))
    baseline_rate = baseline_count / (baseline_end - baseline_start)
    overshoot_count = int(np.sum((spikes >= release) & (spikes < overshoot_end)))
    post_rate = overshoot_count / (overshoot_end - release)
    recovery_spikes = spikes[(spikes >= release) & (spikes < recovery_end)]
    latency = float(recovery_spikes[0] - release) if recovery_spikes.size else None
    return {
        "window_convention": "half-open",
        "recovery_latency_s": latency,
        "baseline_spike_count": baseline_count,
        "baseline_rate_hz": float(baseline_rate),
        "overshoot_spike_count": overshoot_count,
        "post_release_rate_hz": float(post_rate),
        "rate_overshoot_hz": float(post_rate - baseline_rate),
        "rate_overshoot_ratio": float(post_rate / baseline_rate) if baseline_rate > 0 else None,
    }
